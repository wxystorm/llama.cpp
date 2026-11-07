#!/usr/bin/env python3
"""Memory-based Jupiter split preflight (GGUF + PC RAM/CUDA + Phone RPC).

Read GGUF per-layer tensor offsets and *current* available device memory.
Choose three contiguous regions: PC CPU prefix, PC GPU middle, Phone RPC suffix.
The 2-device mode remains valid when no CPU layers are required. These are
weight capacity estimates, not guarantees of runtime KV, graph or buffer fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import socket
import struct
import subprocess
import sys

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
    embedding_length: int = 0
    attention_head_count: int = 0
    attention_head_count_kv: int = 0
    attention_key_length: int = 0
    attention_value_length: int = 0

    def kv_bytes_per_layer(self, context: int, dtype_bytes: float = 2.0) -> int:
        """Estimate per-layer K+V allocation for a full context in this model."""
        if context < 1 or not 0 < dtype_bytes <= 8:
            raise ValueError("invalid KV context length or bytes per element")
        if self.attention_head_count_kv <= 0:
            raise ValueError(
                "GGUF attention.head_count_kv is absent; unable to size "
                "the KV cache safely. Supply --kv-per-layer-mib override."
            )
        head_dim = (self.embedding_length // self.attention_head_count
                    if self.attention_head_count > 0
                    and self.embedding_length % self.attention_head_count == 0
                    else 0)
        key_dim = self.attention_key_length or head_dim
        value_dim = self.attention_value_length or head_dim
        if key_dim < 1 or value_dim < 1:
            raise ValueError(
                "GGUF key/value head dimensions missing; supply "
                "--kv-per-layer-mib override."
            )
        return math.ceil(
            context * self.attention_head_count_kv *
            (key_dim + value_dim) * dtype_bytes
        )

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
    cpu_layers: int = 0
    cpu_weight_bytes: int = 0
    cpu_available_bytes: int = 0
    cpu_budget_bytes: int = 0
    cpu_reserve_bytes: int = 0
    kv_per_layer_bytes: int = 0
    cpu_kv_bytes: int = 0
    pc_kv_bytes: int = 0
    phone_kv_bytes: int = 0
    cpu_workspace_bytes: int = 0
    pc_workspace_bytes: int = 0
    phone_workspace_bytes: int = 0
    cpu_safety_bytes: int = 0
    pc_safety_bytes: int = 0
    phone_safety_bytes: int = 0
    context_tokens: int = 0
    kv_dtype_bytes: float = 2.0

    @property
    def pc_gpu_layers(self) -> int:
        return self.pc_layers - self.cpu_layers


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
                    or key == 'general.split.count'
                    or key.endswith((
                        '.embedding_length',
                        '.attention.head_count',
                        '.attention.head_count_kv',
                        '.attention.key_length',
                        '.attention.value_length',
                    )))
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
    return GGUFWeights(
        arch, tuple(layer_sizes), other_size, file_bytes,
        embedding_length=int(metadata.get(arch + '.embedding_length') or 0),
        attention_head_count=int(metadata.get(arch + '.attention.head_count') or 0),
        attention_head_count_kv=int(metadata.get(arch + '.attention.head_count_kv') or 0),
        attention_key_length=int(metadata.get(arch + '.attention.key_length') or 0),
        attention_value_length=int(metadata.get(arch + '.attention.value_length') or 0),
    )


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


def pc_cpu_memory() -> DeviceMemory:
    """Get available host RAM; never mistake total RAM for allocatable RAM."""
    if sys.platform == 'win32':
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise ValueError('Windows GlobalMemoryStatusEx failed')
        memory = DeviceMemory(int(stat.ullAvailPhys), int(stat.ullTotalPhys))
    elif sys.platform.startswith('linux'):
        fields = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key] = int(value.strip().split()[0]) * 1024
        memory = DeviceMemory(fields['MemAvailable'], fields['MemTotal'])
    else:
        raise ValueError('host RAM query unavailable; pass --cpu-free-mib')
    if not 0 < memory.free_bytes <= memory.total_bytes:
        raise ValueError('invalid available PC RAM')
    return memory


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


def estimate_workspaces(weights: GGUFWeights, ubatch: int, window: int,
                        graph_prebuild: bool,
                        cpu_workspace_mib: int, pc_workspace_mib: int,
                        phone_workspace_mib: int) -> tuple[int, int, int]:
    """Workspace budget: configured floor + activation slots + graph prebuild.

    These are adjustable allowances, NOT a claim to predict ggml allocator
    high-water marks. Actual buffer requirements are architecture-dependent.
    """
    if ubatch < 1 or window < 1 or min(
        cpu_workspace_mib, pc_workspace_mib, phone_workspace_mib
    ) < 0:
        raise ValueError("invalid graph workspace or ubatch/window settings")
    embd = weights.embedding_length or 4096
    # Number of live activation copies in the pipeline. Conservatively assume
    # FP32 intermediates and 12 reusable residual/attention working tensors.
    activation_bytes = ubatch * embd * 4 * 12 * window
    graph_bytes = 128*MIB if graph_prebuild else 0
    return (
        cpu_workspace_mib*MIB + activation_bytes//2,
        pc_workspace_mib*MIB + activation_bytes + graph_bytes,
        phone_workspace_mib*MIB + activation_bytes + graph_bytes,
    )


def plan_memory_split(weights: GGUFWeights, pc: DeviceMemory,
                      phone: DeviceMemory, pc_reserve_mib: int = 512,
                      phone_reserve_mib: int = 512,
                      pc_layers_override: int | None = None,
                      cpu: DeviceMemory | None = None,
                      cpu_reserve_mib: int = 2048,
                      *,
                      context: int = 0,
                      ubatch: int = 128,
                      window: int = 2,
                      graph_prebuild: bool = False,
                      kv_dtype_bytes: float = 2.0,
                      kv_per_layer_mib: float | None = None,
                      cpu_workspace_mib: int = 0,
                      pc_workspace_mib: int = 0,
                      phone_workspace_mib: int = 0,
                      safety_percent: float = 0.0) -> MemoryPlan:
    """Allocate contiguous CPU->CUDA->Phone layers with KV/graph headroom.

    1. Budget fixed system/device reserves and safety margins.
    2. Budget per-device graph workspace (ubatch/window/prebuild-dependent).
    3. Account for per-layer KV cache at the configured context length.
    4. Fit weight + KV per layer; first minimize CPU layers, then balance
       accelerator utilization, never silently overcommitting memory.

    Exact backend allocation is architecture/runtime-specific; this remains
    a preflight estimate and the actual model load must still be validated.
    """
    if min(pc_reserve_mib, phone_reserve_mib, cpu_reserve_mib) < 0:
        raise ValueError("fixed memory reserves must be nonnegative")
    if (not math.isfinite(safety_percent) or not 0 <= safety_percent <= 50
            or not 0 < kv_dtype_bytes <= 8 or context < 0):
        raise ValueError("invalid KV dtype, context length, or safety percentage")
    if kv_per_layer_mib is not None and (
        not math.isfinite(kv_per_layer_mib) or kv_per_layer_mib < 0
    ):
        raise ValueError("invalid --kv-per-layer-mib override")
    if context:
        kv_per_layer = (
            math.ceil(kv_per_layer_mib*MIB)
            if kv_per_layer_mib is not None
            else weights.kv_bytes_per_layer(context, kv_dtype_bytes)
        )
    else:
        kv_per_layer = 0
    cpu_workspace, pc_workspace, phone_workspace = estimate_workspaces(
        weights, ubatch, window, graph_prebuild,
        cpu_workspace_mib, pc_workspace_mib, phone_workspace_mib,
    )
    pc_reserve, phone_reserve = pc_reserve_mib*MIB, phone_reserve_mib*MIB
    cpu_reserve = cpu_reserve_mib*MIB if cpu is not None else 0
    pc_safety = math.ceil(pc.free_bytes*safety_percent/100)
    phone_safety = math.ceil(phone.free_bytes*safety_percent/100)
    cpu_safety = (
        math.ceil(cpu.free_bytes*safety_percent/100)
        if cpu is not None else 0
    )
    pc_budget = pc.free_bytes-pc_reserve-pc_workspace-pc_safety
    phone_budget = phone.free_bytes-phone_reserve-phone_workspace-phone_safety
    cpu_budget = (max(0, cpu.free_bytes-cpu_reserve-cpu_workspace-cpu_safety)
                  if cpu is not None else 0)
    if pc_budget <= 0 or phone_budget <= 0:
        raise ValueError(
            "INFEASIBLE: GPU or Phone memory smaller than reserve + "
            "KV/graph workspace + safety allowance; "
            f"PC weight+KV budget={pc_budget/GIB:.2f} GiB, "
            f"Phone weight+KV budget={phone_budget/GIB:.2f} GiB"
        )

    required = weights.total_weights + weights.n_layers*kv_per_layer
    if required > pc_budget+phone_budget+cpu_budget:
        raise ValueError(
            f"INFEASIBLE: GGUF weights {weights.total_weights/GIB:.2f} GiB "
            f"+ total KV {weights.n_layers*kv_per_layer/GIB:.2f} GiB "
            f"exceed combined memory after reserves/graph/safety "
            f"{(pc_budget+phone_budget+cpu_budget)/GIB:.2f} GiB "
            f"(CPU {cpu_budget/GIB:.2f}, PC GPU {pc_budget/GIB:.2f}, "
            f"Phone {phone_budget/GIB:.2f})"
        )
    # Repeating layers own KV on their assigned execution backend.
    prefix = [0]
    for size in weights.layers:
        prefix.append(prefix[-1]+size+kv_per_layer)

    choices = []
    for cpu_layers in range(weights.n_layers-1):
        cpu_used = prefix[cpu_layers]
        if cpu_used > cpu_budget:
            break
        for pc_layers in range(cpu_layers+1, weights.n_layers):
            if pc_layers_override is not None and pc_layers != pc_layers_override:
                continue
            gpu_used = prefix[pc_layers]-cpu_used
            # Output head and all other non-block tensors are placed on Phone
            # in the Jupiter model loader. Conservatively charge all there.
            phone_used = required-prefix[pc_layers]
            if gpu_used > pc_budget or phone_used > phone_budget:
                continue
            score = (
                cpu_layers,
                abs(gpu_used/pc_budget-phone_used/phone_budget),
                pc_layers,
            )
            choices.append((score, cpu_layers, pc_layers))
        if choices:
            break  # Keep as many layers as possible off the PC CPU.

    if not choices:
        boundary = (f" for --pc-layers={pc_layers_override}"
                    if pc_layers_override is not None else "")
        raise ValueError(
            f"INFEASIBLE: no contiguous CPU/CUDA/Phone split{boundary}; "
            f"usable weight+KV budgets: CPU {cpu_budget/GIB:.2f} GiB, "
            f"GPU {pc_budget/GIB:.2f} GiB, Phone {phone_budget/GIB:.2f} GiB "
            f"(KV={kv_per_layer/MIB:.1f} MiB per layer)."
        )
    _, cpu_layers, pc_layers = min(choices)
    pc_weights = sum(weights.layers[cpu_layers:pc_layers])
    cpu_weights = sum(weights.layers[:cpu_layers])
    phone_weights = weights.total_weights-sum(weights.layers[:pc_layers])
    return MemoryPlan(
        pc_layers=pc_layers, phone_layers=weights.n_layers-pc_layers,
        pc_weight_bytes=pc_weights, phone_weight_bytes=phone_weights,
        pc_available_bytes=pc.free_bytes, phone_available_bytes=phone.free_bytes,
        pc_budget_bytes=pc_budget, phone_budget_bytes=phone_budget,
        pc_reserve_bytes=pc_reserve, phone_reserve_bytes=phone_reserve,
        model=weights,
        cpu_layers=cpu_layers, cpu_weight_bytes=cpu_weights,
        cpu_available_bytes=cpu.free_bytes if cpu is not None else 0,
        cpu_budget_bytes=cpu_budget, cpu_reserve_bytes=cpu_reserve,
        kv_per_layer_bytes=kv_per_layer,
        cpu_kv_bytes=cpu_layers*kv_per_layer,
        pc_kv_bytes=(pc_layers-cpu_layers)*kv_per_layer,
        phone_kv_bytes=(weights.n_layers-pc_layers)*kv_per_layer,
        cpu_workspace_bytes=cpu_workspace if cpu is not None else 0,
        pc_workspace_bytes=pc_workspace,
        phone_workspace_bytes=phone_workspace,
        cpu_safety_bytes=cpu_safety,
        pc_safety_bytes=pc_safety,
        phone_safety_bytes=phone_safety,
        context_tokens=context,
        kv_dtype_bytes=kv_dtype_bytes,
    )
