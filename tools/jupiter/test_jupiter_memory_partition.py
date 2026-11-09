#!/usr/bin/env python3
"""Offline tests for GGUF weight sizing and GPU/RPC memory-based splitting."""
from __future__ import annotations

import socket
import struct
import threading
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from jupiter_memory_partition import (
    GGUFWeights, DeviceMemory, GIB, MIB, plan_memory_split,
    read_gguf_weights, pc_gpu_memory, phone_rpc_memory,
)


def fake_gguf(path: Path) -> None:
    # Sparse 4-block GGUF: 64MiB each + 32MiB output tensor.
    infos = [(f'blk.{j}.attn.weight', j*64*MIB) for j in range(4)]
    infos.append(('output.weight', 4*64*MIB))
    def gguf_string(text: str) -> bytes:
        data = text.encode('utf-8')
        return struct.pack('<Q', len(data))+data
    with path.open('wb') as f:
        scalar_meta = [
            ('qwen3.block_count', 4),
            ('general.alignment', 32),
            ('qwen3.embedding_length', 5120),
            ('qwen3.attention.head_count', 64),
            ('qwen3.attention.head_count_kv', 8),
            ('qwen3.attention.key_length', 128),
            ('qwen3.attention.value_length', 128),
        ]
        f.write(b'GGUF'+struct.pack('<IQQ', 3, len(infos), len(scalar_meta)+1))
        f.write(gguf_string('general.architecture') +
                struct.pack('<I', 8)+gguf_string('qwen3'))
        for key, value in scalar_meta:
            f.write(gguf_string(key)+struct.pack('<II', 4, value))
        for name, offset in infos:
            f.write(gguf_string(name)+struct.pack('<I', 1) +
                    struct.pack('<Q', 1)+struct.pack('<IQ', 0, offset))
        start = (f.tell()+31)//32*32
        f.seek(start+4*64*MIB+32*MIB-1)
        f.write(b'\0')


class JupiterMemoryPlannerTest(unittest.TestCase):
    def test_gguf_block_sizes_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mini.gguf'
            fake_gguf(path)
            model = read_gguf_weights(path)
            self.assertEqual(model.architecture, 'qwen3')
            self.assertEqual(model.layers, (64*MIB,)*4)
            self.assertEqual(model.other_bytes, 32*MIB)
            self.assertEqual(model.total_weights, 288*MIB)
            self.assertEqual(model.embedding_length, 5120)
            self.assertEqual(model.attention_head_count_kv, 8)
            self.assertEqual(model.kv_bytes_per_layer(2048), 8*MIB)
            self.assertEqual(model.kv_bytes_per_layer(4096), 16*MIB)

    def test_memory_balance_override_and_insufficient_total(self):
        model = GGUFWeights('qwen3', (64*MIB,)*4, 32*MIB, 0)
        plan = plan_memory_split(model, DeviceMemory(200*MIB, 300*MIB),
                                 DeviceMemory(200*MIB, 300*MIB), 0, 0)
        self.assertEqual((plan.pc_layers, plan.phone_layers), (2, 2))
        with self.assertRaisesRegex(ValueError, 'INFEASIBLE'):
            plan_memory_split(model, DeviceMemory(100*MIB, 300*MIB),
                              DeviceMemory(100*MIB, 300*MIB), 0, 0)
        with self.assertRaisesRegex(ValueError, 'INFEASIBLE'):
            plan_memory_split(model, DeviceMemory(150*MIB, 300*MIB),
                              DeviceMemory(250*MIB, 300*MIB), 0, 0,
                              pc_layers_override=3)

    def test_cpu_fallback_auto_split_for_large_model(self):
        # 64 layers, 256 MiB each = 16 GiB. Local GPU and phone
        # each have a 4 GiB usable weight budget; CPU must hold the rest.
        weights = GGUFWeights('qwen2', (256*MIB,)*64, 0, 0)
        result = plan_memory_split(
            weights,
            DeviceMemory(4*GIB, 6*GIB),
            DeviceMemory(4*GIB, 8*GIB),
            pc_reserve_mib=0, phone_reserve_mib=0,
            cpu=DeviceMemory(16*GIB, 24*GIB),
            cpu_reserve_mib=2048,
        )
        self.assertGreater(result.cpu_layers, 0)
        self.assertEqual(result.cpu_layers+result.pc_gpu_layers +
                         result.phone_layers, 64)
        self.assertEqual(result.cpu_weight_bytes +
                         result.pc_weight_bytes +
                         result.phone_weight_bytes, 16*GIB)
        self.assertLessEqual(result.cpu_weight_bytes, result.cpu_budget_bytes)
        self.assertLessEqual(result.pc_weight_bytes, result.pc_budget_bytes)
        self.assertLessEqual(result.phone_weight_bytes, result.phone_budget_bytes)

    def test_fixed_resume_split_is_kept_even_if_auto_would_change(self):
        model = GGUFWeights('qwen3', (100*MIB,)*10, 0, 0)
        cpu = DeviceMemory(1024*MIB, 1024*MIB)
        pc = DeviceMemory(1024*MIB, 1024*MIB)
        phone = DeviceMemory(1024*MIB, 1024*MIB)
        automatic = plan_memory_split(
            model, pc, phone, 0, 0, cpu=cpu, cpu_reserve_mib=0
        )
        self.assertEqual(automatic.cpu_layers, 0)
        pinned = plan_memory_split(
            model, pc, phone, 0, 0, cpu=cpu, cpu_reserve_mib=0,
            cpu_layers_override=3, pc_layers_override=6,
        )
        self.assertEqual((pinned.cpu_layers, pinned.pc_layers), (3, 6))
        self.assertEqual((pinned.pc_gpu_layers, pinned.phone_layers), (3, 4))
        with self.assertRaisesRegex(ValueError, 'INFEASIBLE'):
            plan_memory_split(
                model, pc, phone, 0, 0,
                cpu=DeviceMemory(200*MIB, 1024*MIB), cpu_reserve_mib=0,
                cpu_layers_override=3, pc_layers_override=6,
            )
        with self.assertRaisesRegex(ValueError, 'saved CPU layer boundary'):
            plan_memory_split(
                model, pc, phone, 0, 0, cpu=cpu,
                cpu_layers_override=6, pc_layers_override=5,
            )

    def test_cpu_is_not_used_if_two_accelerators_suffice(self):
        weights = GGUFWeights('qwen3', (64*MIB,)*4, 32*MIB, 0)
        result = plan_memory_split(
            weights, DeviceMemory(200*MIB, 300*MIB),
            DeviceMemory(200*MIB, 300*MIB), 0, 0,
            cpu=DeviceMemory(10*GIB, 12*GIB), cpu_reserve_mib=1024,
        )
        self.assertEqual(result.cpu_layers, 0)

    def test_headroom_forces_cpu_layers_and_accounts_for_kv(self):
        weights = GGUFWeights(
            'qwen3', (288*MIB,)*64, 0, 0,
            embedding_length=5120, attention_head_count=64,
            attention_head_count_kv=8,
            attention_key_length=128, attention_value_length=128,
        )
        gpu = DeviceMemory(int(5.36*GIB), 6*GIB)
        phone = DeviceMemory(int(14.76*GIB), 16*GIB)
        cpu = DeviceMemory(int(17.90*GIB), 24*GIB)
        plan = plan_memory_split(
            weights, gpu, phone, 512, 512,
            cpu=cpu, cpu_reserve_mib=2048,
            context=2048, ubatch=128, window=2, graph_prebuild=True,
            cpu_workspace_mib=512, pc_workspace_mib=1024,
            phone_workspace_mib=1536, safety_percent=10,
        )
        self.assertGreater(plan.cpu_layers, 0)
        self.assertEqual(plan.kv_per_layer_bytes, 8*MIB)
        self.assertEqual(plan.cpu_kv_bytes+plan.pc_kv_bytes+plan.phone_kv_bytes,
                         64*8*MIB)
        for weight, kv, budget in (
            (plan.cpu_weight_bytes, plan.cpu_kv_bytes, plan.cpu_budget_bytes),
            (plan.pc_weight_bytes, plan.pc_kv_bytes, plan.pc_budget_bytes),
            (plan.phone_weight_bytes, plan.phone_kv_bytes, plan.phone_budget_bytes),
        ):
            self.assertLessEqual(weight+kv, budget)
        self.assertGreater(plan.phone_workspace_bytes, 1536*MIB)
        self.assertGreater(plan.pc_safety_bytes, 0)

    def test_qwen3_rope_head_dimension_fallback(self):
        model = GGUFWeights(
            'qwen3', (288*MIB,)*64, 0, 0,
            embedding_length=5120, attention_head_count=64,
            attention_head_count_kv=8, rope_dimension_count=128,
        )
        # The naive embedding/head_count value is 80, but K/V
        # use head dimension 128 on this Qwen3 layout.
        self.assertEqual(model.kv_bytes_per_layer(2048), 8*MIB)

    def test_missing_metadata_requires_explicit_kv_override(self):
        weights = GGUFWeights('other', (64*MIB,)*4, 0, 0)
        with self.assertRaisesRegex(ValueError, 'head_count_kv'):
            plan_memory_split(
                weights, DeviceMemory(4*GIB, 4*GIB),
                DeviceMemory(4*GIB, 4*GIB),
                context=2048,
            )
        plan = plan_memory_split(
            weights, DeviceMemory(4*GIB, 4*GIB),
            DeviceMemory(4*GIB, 4*GIB),
            context=2048, kv_per_layer_mib=8,
        )
        self.assertEqual(plan.kv_per_layer_bytes, 8*MIB)

    def test_pc_free_memory_from_nvidia_smi(self):
        class Result:
            stdout = ' 5000, 6144\n'
        with patch('jupiter_memory_partition.subprocess.run', return_value=Result()):
            memory = pc_gpu_memory()
        self.assertEqual(memory.free_bytes, 5000*MIB)
        self.assertEqual(memory.total_bytes, 6144*MIB)

    def test_rpc_hello_and_get_device_memory(self):
        def read_exact(sock, n):
            buf = b''
            while len(buf) < n:
                data = sock.recv(n-len(buf))
                if not data:
                    raise ConnectionError('socket closed')
                buf += data
            return buf
        server = socket.socket()
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        port = server.getsockname()[1]
        errors = []

        def serve():
            try:
                conn, _ = server.accept()
                with conn:
                    for cmd_expected, request_expected, response in [
                        (14, bytes(24), bytes([4, 0, 4, 0])+bytes(24)),
                        (15, b'', struct.pack('<I', 1)),
                        (11, struct.pack('<I', 0), struct.pack('<QQ', 6*GIB, 8*GIB)),
                    ]:
                        cmd, size = struct.unpack('<BQ', read_exact(conn, 9))
                        self.assertEqual((cmd, read_exact(conn, size)),
                                         (cmd_expected, request_expected))
                        conn.sendall(struct.pack('<Q', len(response))+response)
            except Exception as exc:
                errors.append(exc)
            finally:
                server.close()
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            memory = phone_rpc_memory(f'127.0.0.1:{port}')
            self.assertEqual((memory.free_bytes, memory.total_bytes),
                             (6*GIB, 8*GIB))
        finally:
            worker.join(timeout=5)
        self.assertFalse(errors, errors)


if __name__ == '__main__':
    unittest.main()
