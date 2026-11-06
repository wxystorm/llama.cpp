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
        f.write(b'GGUF'+struct.pack('<IQQ', 3, len(infos), 3))
        f.write(gguf_string('general.architecture') +
                struct.pack('<I', 8)+gguf_string('qwen3'))
        f.write(gguf_string('qwen3.block_count')+struct.pack('<II', 4, 4))
        f.write(gguf_string('general.alignment')+struct.pack('<II', 4, 32))
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
