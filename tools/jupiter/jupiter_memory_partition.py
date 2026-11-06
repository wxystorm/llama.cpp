#!/usr/bin/env python3
"""Memory-based two-device Jupiter split preflight (GGUF + CUDA + phone RPC).

Estimate weight placement from GGUF tensor offsets, read free memory from
nvidia-smi and llama.cpp RPC GET_DEVICE_MEMORY, then choose the memory-feasible
PC/Phone contiguous layer boundary. Never infer CPU spill: Jupiter's GPU+RPC
layer path requires all repeating layers offloaded (-ngl all).

This is a conservative weight-budget check, not proof that runtime allocations
or per-buffer max allocation limits will fit on either device.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import socket
import struct
import subprocess

GIB = 1024 ** 3
MIB = 1024 ** 2
_GGUF_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                     7: 1, 10: 8, 11: 8, 12: 8}
_RPC_HELLO = 14
_RPC_DEVICE_COUNT = 15
_RPC_GET_DEVICE_MEMORY = 11
_RPC_CAPS_BYTES = 24
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass(frozen=True)
class GGUFWeights:
    architecture: str
    layers: tuple[int, ...]
    other_bytes: int
    file_bytes: int

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def total_weights(self) -> int:
        return sum(self.layers) + self.other_bytes


@dataclass(frozen=True)
class DeviceMemory:
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class MemoryPlan:
    pc_layers: int
    phone_layers: int
    pc_weight_bytes: int
    phone_weight_bytes: int
    pc_available_bytes: int
    phone_available_bytes: int
    pc_budget_bytes: int
    phone_budget_bytes: int
    pc_reserve_bytes: int
    phone_reserve_bytes: int
    model: GGUFWeights


def _read_exact(f, n: int) -> bytes:
    if n < 0 or n > (1 << 29):
        raise ValueError(f"invalid GGUF metadata field length: {n}")
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF header")
    return data


def _read_num(f, code: str) -> int:
    return int(struct.unpack('<' + code,
                             _read_exact(f, struct.calcsize('<' + code)))[0])


def _gguf_string(f) -> str:
    return _read_exact(f, _read_num(f, 'Q')).decode('utf-8', 'replace')


def _metadata_value(f, value_type: int, keep: bool = False):
    if value_type == 8:  # GGUF_TYPE_STRING
        if keep:
            return _gguf_string(f)
        f.seek(_read_num(f, 'Q'), 1)
        return None
    if value_type == 9:  # GGUF_TYPE_ARRAY
        element_type = _read_num(f, 'I')
        count = _read_num(f, 'Q')
        if count > 10_000_000:
            raise ValueError("GGUF metadata array too large")
        if element_type in _GGUF_VALUE_SIZES:
            f.seek(_GGUF_VALUE_SIZES[element_type] * count, 1)
        elif element_type == 8:
            for _ in range(count):
                f.seek(_read_num(f, 'Q'), 1)
        else:
            raise ValueError(f"unsupported GGUF array element type {element_type}")
        return None
    size = _GGUF_VALUE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    if keep:
        return int.from_bytes(_read_exact(f, size), 'little')
    f.seek(size, 1)
    return None


def read_gguf_weights(path: Path) -> GGUFWeights:
    """Estimate per-layer tensor bytes from GGUF offsets, without loading weights."""
    file_bytes = path.stat().st_size
    with path.open('rb') as f:
        if _read_exact(f, 4) != b'GGUF':
            raise ValueError(f"not a GGUF model: {path}")
        version = _read_num(f, 'I')
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _read_num(f, 'Q')
        metadata_count = _read_num(f, 'Q')
        if tensor_count > 1_000_000 or metadata_count > 5_000_000:
            raise ValueError("unreasonable GGUF header counts")
        metadata = {}
        for _ in range(metadata_count):
            key = _gguf_string(f)
            value_type = _read_num(f, 'I')
            keep = (key == 'general.architecture' or key == 'general.alignment'
                    or key.endswith('.block_count') or key == 'split.count'
                    or key == 'general.split.count')
            value = _metadata_value(f, value_type, keep)
            if keep:
                metadata[key] = value
        arch = metadata.get('general.architecture')
        if not isinstance(arch, str) or not arch:
            raise ValueError('GGUF general.architecture not found')
        n_layers = metadata.get(arch + '.block_count')
        if not isinstance(n_layers, int) or not 1 < n_layers < 1024:
            raise ValueError(f'GGUF {arch}.block_count missing or invalid')
        if any(metadata.get(key, 1) > 1 for key in
               ('split.count', 'general.split.count')):
            raise ValueError('sharded GGUF is not supported by the memory estimator')
        alignment = metadata.get('general.alignment') or 32
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096:
            raise ValueError('invalid GGUF alignment')
        offsets = []
        for _ in range(tensor_count):
            name = _gguf_string(f)
            n_dims = _read_num(f, 'I')
            if n_dims > 8:
                raise ValueError('invalid GGUF tensor dimensionality')
            f.seek(8 * n_dims, 1)
            _read_num(f, 'I')  # ggml tensor type
            offset = _read_num(f, 'Q')
            offsets.append((offset, name))
        data_start = ((f.tell() + alignment - 1)//alignment)*alignment
    data_size = file_bytes-data_start
    if data_size <= 0 or not offsets:
        raise ValueError('GGUF has no tensor payload')
    offsets.sort()
    layer_sizes = [0]*n_layers
    other_size = 0
    for j, (offset, name) in enumerate(offsets):
        next_offset = offsets[j+1][0] if j+1 < len(offsets) else data_size
        if not 0 <= offset < next_offset <= data_size:
            raise ValueError('GGUF tensor offsets invalid or sharded')
        size = next_offset-offset
        match = _BLOCK_RE.match(name)
        if match:
            index = int(match.group(1))
            if index >= n_layers:
                raise ValueError(f'GGUF tensor index outside layer count: {name}')
            layer_sizes[index] += size
        else:
            other_size += size
    if any(size <= 0 for size in layer_sizes):
        raise ValueError('GGUF is missing at least one complete repeating layer')
    return GGUFWeights(arch, tuple(layer_sizes), other_size, file_bytes)


def pc_gpu_memory() -> DeviceMemory:
    """Query NVIDIA GPU index 0, usually the local CUDA0 device."""
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=memory.free,memory.total',
           '--format=csv,noheader,nounits']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10, check=True)
        line = next((x for x in proc.stdout.splitlines() if x.strip()), '')
        free, total = [int(v.strip()) for v in line.split(',')[:2]]
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as exc:
        raise ValueError('cannot query PC GPU memory with nvidia-smi; '
                         'use --pc-free-mib explicitly') from exc
    if not 0 < free <= total:
        raise ValueError('nvidia-smi returned invalid memory values')
    return DeviceMemory(free*MIB, total*MIB)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        data = sock.recv(length-len(chunks))
        if not data:
            raise ConnectionError('phone RPC closed connection')
        chunks.extend(data)
    return bytes(chunks)


def _rpc(sock: socket.socket, command: int, request: bytes,
         expected_size: int) -> bytes:
    sock.sendall(struct.pack('<BQ', command, len(request))+request)
    count = struct.unpack('<Q', _recv_exact(sock, 8))[0]
    if count != expected_size:
        raise ConnectionError(
            f'phone RPC cmd={command} response size {count}, '
            f'expected {expected_size}')
    return _recv_exact(sock, count)


def phone_rpc_memory(endpoint: str, device: int = 0) -> DeviceMemory:
    """Query llama.cpp RPC device free/total bytes with transport upgrade off."""
    try:
        host, port_text = endpoint.rsplit(':', 1)
        host = host.strip('[]')
        port = int(port_text)
        if not 0 < port < 65536 or device < 0:
            raise ValueError('invalid RPC port or device')
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.settimeout(10)
            hello = _rpc(sock, _RPC_HELLO, bytes(_RPC_CAPS_BYTES),
                         4+_RPC_CAPS_BYTES)
            if hello[0] < 1:
                raise ConnectionError('invalid RPC HELLO version')
            count = struct.unpack('<I',
                                  _rpc(sock, _RPC_DEVICE_COUNT, b'', 4))[0]
            if device >= count:
                raise ValueError(f'RPC device {device} not found (count={count})')
            free, total = struct.unpack('<QQ',
                                         _rpc(sock, _RPC_GET_DEVICE_MEMORY,
                                              struct.pack('<I', device), 16))
    except (OSError, ConnectionError) as exc:
        raise ValueError(
            f'cannot query phone RPC {endpoint} device {device} memory: {exc}; '
            'use --phone-free-mib if unavailable') from exc
    if not 0 < free <= total:
        raise ValueError(f'phone RPC returned invalid memory: free={free} total={total}')
    return DeviceMemory(free, total)


def plan_memory_split(weights: GGUFWeights, pc: DeviceMemory,
                      phone: DeviceMemory, pc_reserve_mib: int = 512,
                      phone_reserve_mib: int = 512,
                      pc_layers_override: int | None = None) -> MemoryPlan:
    if pc_reserve_mib < 0 or phone_reserve_mib < 0:
        raise ValueError('memory reserves must be nonnegative')
    pc_reserve, phone_reserve = pc_reserve_mib*MIB, phone_reserve_mib*MIB
    pc_budget = pc.free_bytes-pc_reserve
    phone_budget = phone.free_bytes-phone_reserve
    if pc_budget <= 0 or phone_budget <= 0:
        raise ValueError('device free memory smaller than the safety reserve')
    # Non-block tensor bytes conservatively belong to Phone (output is there).
    required = weights.total_weights
    if required > pc_budget+phone_budget:
        raise ValueError(
            f'INFEASIBLE: GGUF tensor payload ~{required/GIB:.2f} GiB exceeds '
            f'combined usable GPU+RPC memory {(pc_budget+phone_budget)/GIB:.2f} GiB '
            f'(PC {pc_budget/GIB:.2f}, Phone {phone_budget/GIB:.2f}); '
            'Jupiter does not automatically spill repeating layers to PC CPU')
    pc_prefix = 0
    choices = []
    for index in range(1, weights.n_layers):
        pc_prefix += weights.layers[index-1]
        phone_weight = required-pc_prefix
        if pc_prefix <= pc_budget and phone_weight <= phone_budget:
            if pc_layers_override is None or index == pc_layers_override:
                score = abs(pc_prefix/pc_budget-phone_weight/phone_budget)
                choices.append((score, index, pc_prefix, phone_weight))
    if not choices:
        note = (f' for --pc-layers={pc_layers_override}'
                if pc_layers_override is not None else '')
        raise ValueError(
            f'INFEASIBLE: no contiguous memory split{note} fits '
            f'PC usable={pc_budget/GIB:.2f} GiB and '
            f'Phone usable={phone_budget/GIB:.2f} GiB; '
            f'nonlayer weights={weights.other_bytes/GIB:.2f} GiB')
    _, pc_layers, pc_weight, phone_weight = min(choices)
    return MemoryPlan(
        pc_layers=pc_layers, phone_layers=weights.n_layers-pc_layers,
        pc_weight_bytes=pc_weight, phone_weight_bytes=phone_weight,
        pc_available_bytes=pc.free_bytes, phone_available_bytes=phone.free_bytes,
        pc_budget_bytes=pc_budget, phone_budget_bytes=phone_budget,
        pc_reserve_bytes=pc_reserve, phone_reserve_bytes=phone_reserve,
        model=weights,
    )
