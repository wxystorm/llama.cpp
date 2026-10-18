#!/usr/bin/env python3
"""
Offline FIFO simulator for the PC Chunk Ring experiment.

Layer modes:
  S = split     : CPU ATTN -> GPU FFN
  G = GPU full  : GPU ATTN + GPU FFN as one GPU FIFO task

The simulator deliberately keeps the scheduler simple:
  complete -> append next task -> FIFO execute

Weight complexity lives in the separate prefetch/residency lane.  Each layer's
GPU weights stay live until the last chunk finishes that layer's GPU task, then
that layer is evicted immediately and the freed slot can be reused.

This is a first-order planning model.  Stage times are scaled linearly from the
measured 128-token anchors.  It intentionally does not model KV-length-dependent
Attention growth yet.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


# Measured on RTX 4050 Laptop GPU / host CPU, Qwen3-30B-A3B-Q4_K_M,
# 4 real layers, 128-token probe.
DEFAULT_CPU_ATTN_MS = 12.226070
DEFAULT_CPU_FFN_MS  = 26.633137
DEFAULT_GPU_ATTN_MS = 0.545657
DEFAULT_GPU_FFN_MS  = 8.411080
DEFAULT_BASE_TOKENS = 128

DEFAULT_ATTN_WEIGHT_MIB = 10.392
DEFAULT_FFN_WEIGHT_MIB  = 374.508


@dataclass
class Task:
    kind: str          # A=CPU ATTN, F=GPU FFN, G=GPU FULL
    layer: int
    chunk: int
    tokens: int
    enqueue_ms: float


@dataclass
class TimelineRow:
    start_ms: float
    end_ms: float
    lane: str
    kind: str
    layer: int
    chunk: int
    tokens: int
    mode: str
    queue_wait_ms: float = 0.0
    weight_mib: float = 0.0


@dataclass
class SimResult:
    makespan_ms: float
    cpu_busy_ms: float
    gpu_busy_ms: float
    xfer_busy_ms: float
    gpu_weight_stall_ms: float
    max_cpu_queue: int
    max_gpu_queue: int
    peak_live_layers: int
    peak_live_weight_mib: float
    timeline: List[TimelineRow]


class Simulator:
    def __init__(
        self,
        pattern: str,
        chunk_sizes: List[int],
        vram_window: int,
        preload_layers: int,
        base_tokens: int,
        cpu_attn_ms: float,
        cpu_ffn_ms: float,
        gpu_attn_ms: float,
        gpu_ffn_ms: float,
        attn_weight_mib: float,
        ffn_weight_mib: float,
        link_gbs: float,
    ) -> None:
        self.pattern = pattern
        self.n_layers = len(pattern)
        self.chunk_sizes = chunk_sizes
        self.n_chunks = len(chunk_sizes)
        self.vram_window = vram_window
        self.preload_layers = preload_layers

        self.base_tokens = base_tokens
        self.cpu_attn_ms = cpu_attn_ms
        self.cpu_ffn_ms = cpu_ffn_ms  # reference only in S/G simulation
        self.gpu_attn_ms = gpu_attn_ms
        self.gpu_ffn_ms = gpu_ffn_ms

        self.attn_weight_mib = attn_weight_mib
        self.ffn_weight_mib = ffn_weight_mib
        self.link_gbs = link_gbs

        self.cpu_q: Deque[Task] = deque()
        self.gpu_q: Deque[Task] = deque()

        # A GPU task is not scheduler-ready until both its activation and its
        # layer weights are ready.  Weight-blocked tasks live here, outside the
        # GPU FIFO, so a missing weight can never head-of-line block unrelated
        # ready GPU work.
        self.gpu_pending: List[Deque[Task]] = [
            deque() for _ in range(self.n_layers)
        ]

        # Weight states: ABSENT, LOADING, RESIDENT, EVICTED.
        self.weight_state = ["ABSENT"] * self.n_layers
        self.live_weight_mib = 0.0
        self.live_layers = 0
        self.peak_live_weight_mib = 0.0
        self.peak_live_layers = 0
        self.next_prefetch = 0

        self.gpu_done_count = [0] * self.n_layers
        self.finished_chunks = 0

        self.cpu_busy = False
        self.gpu_busy = False
        self.xfer_busy = False

        self.cpu_busy_ms = 0.0
        self.gpu_busy_ms = 0.0
        self.xfer_busy_ms = 0.0

        self.gpu_blocked_since: Optional[float] = None
        self.gpu_weight_stall_ms = 0.0

        self.max_cpu_queue = 0
        self.max_gpu_queue = 0

        self.timeline: List[TimelineRow] = []

        self.events: List[Tuple[float, int, str, object]] = []
        self.event_seq = 0

    def layer_weight_mib(self, layer: int) -> float:
        # Split layers need only the GPU FFN weights. GPU-full layers need
        # both ATTN and FFN weights.
        if self.pattern[layer] == "G":
            return self.ffn_weight_mib + self.attn_weight_mib
        return self.ffn_weight_mib

    def transfer_ms(self, layer: int) -> float:
        # MiB -> bytes; GB/s is decimal, matching common PCIe bandwidth usage.
        nbytes = self.layer_weight_mib(layer) * 1024.0 * 1024.0
        return nbytes / (self.link_gbs * 1.0e9) * 1000.0

    def scaled_ms(self, anchor_ms: float, tokens: int) -> float:
        return anchor_ms * (float(tokens) / float(self.base_tokens))

    def cpu_task_ms(self, task: Task) -> float:
        assert task.kind == "A"
        return self.scaled_ms(self.cpu_attn_ms, task.tokens)

    def gpu_task_ms(self, task: Task) -> float:
        if task.kind == "F":
            return self.scaled_ms(self.gpu_ffn_ms, task.tokens)
        if task.kind == "G":
            return self.scaled_ms(
                self.gpu_attn_ms + self.gpu_ffn_ms, task.tokens
            )
        raise ValueError("invalid GPU task kind: " + task.kind)

    def push_event(self, end_ms: float, kind: str, payload: object) -> None:
        self.event_seq += 1
        heapq.heappush(
            self.events, (end_ms, self.event_seq, kind, payload)
        )

    def enqueue_cpu(self, task: Task) -> None:
        self.cpu_q.append(task)
        self.max_cpu_queue = max(self.max_cpu_queue, len(self.cpu_q))

    def enqueue_gpu(self, task: Task) -> None:
        # "ready -> push_back" is the only scheduler rule.  A task whose
        # activation is ready but whose weights are not resident is a weight
        # dependency, not a FIFO entry.
        if self.weight_state[task.layer] == "RESIDENT":
            self.gpu_q.append(task)
            self.max_gpu_queue = max(
                self.max_gpu_queue, len(self.gpu_q)
            )
        else:
            self.gpu_pending[task.layer].append(task)

    def release_weight_ready_tasks(self, layer: int) -> None:
        pending = self.gpu_pending[layer]
        while pending:
            self.gpu_q.append(pending.popleft())
        self.max_gpu_queue = max(
            self.max_gpu_queue, len(self.gpu_q)
        )

    def has_weight_blocked_gpu_work(self) -> bool:
        return any(self.gpu_pending[layer]
                   for layer in range(self.n_layers))

    def enqueue_next_layer(self, layer: int, chunk: int, tokens: int, now: float) -> None:
        next_layer = layer + 1
        if next_layer >= self.n_layers:
            self.finished_chunks += 1
            return

        mode = self.pattern[next_layer]
        if mode == "S":
            self.enqueue_cpu(Task("A", next_layer, chunk, tokens, now))
        else:
            self.enqueue_gpu(Task("G", next_layer, chunk, tokens, now))

    def reserve_weight_slot(self, layer: int) -> None:
        assert self.weight_state[layer] == "ABSENT"
        self.weight_state[layer] = "LOADING"
        self.live_layers += 1
        self.live_weight_mib += self.layer_weight_mib(layer)
        self.peak_live_layers = max(self.peak_live_layers, self.live_layers)
        self.peak_live_weight_mib = max(
            self.peak_live_weight_mib, self.live_weight_mib
        )

    def evict_layer(self, layer: int, now: float) -> None:
        assert self.weight_state[layer] == "RESIDENT"
        assert not self.gpu_pending[layer]
        self.weight_state[layer] = "EVICTED"
        self.live_layers -= 1
        self.live_weight_mib -= self.layer_weight_mib(layer)
        # Avoid tiny negative floating point residue.
        if abs(self.live_weight_mib) < 1e-9:
            self.live_weight_mib = 0.0

        self.timeline.append(
            TimelineRow(
                start_ms=now,
                end_ms=now,
                lane="WEIGHT",
                kind="EVICT",
                layer=layer,
                chunk=-1,
                tokens=0,
                mode=self.pattern[layer],
                weight_mib=self.layer_weight_mib(layer),
            )
        )

    def try_start_transfer(self, now: float) -> None:
        if self.xfer_busy:
            return
        if self.next_prefetch >= self.n_layers:
            return
        if self.live_layers >= self.vram_window:
            return

        layer = self.next_prefetch
        self.next_prefetch += 1
        self.reserve_weight_slot(layer)

        dur = self.transfer_ms(layer)
        self.xfer_busy = True
        self.xfer_busy_ms += dur

        row = TimelineRow(
            start_ms=now,
            end_ms=now + dur,
            lane="XFER",
            kind="LOAD",
            layer=layer,
            chunk=-1,
            tokens=0,
            mode=self.pattern[layer],
            weight_mib=self.layer_weight_mib(layer),
        )
        self.timeline.append(row)
        self.push_event(now + dur, "XFER_DONE", layer)

    def try_start_cpu(self, now: float) -> None:
        if self.cpu_busy or not self.cpu_q:
            return

        task = self.cpu_q.popleft()
        dur = self.cpu_task_ms(task)
        self.cpu_busy = True
        self.cpu_busy_ms += dur

        row = TimelineRow(
            start_ms=now,
            end_ms=now + dur,
            lane="CPU",
            kind="ATTN",
            layer=task.layer,
            chunk=task.chunk,
            tokens=task.tokens,
            mode=self.pattern[task.layer],
            queue_wait_ms=now - task.enqueue_ms,
        )
        self.timeline.append(row)
        self.push_event(now + dur, "CPU_DONE", task)

    def try_start_gpu(self, now: float) -> None:
        if self.gpu_busy:
            return

        if not self.gpu_q:
            # Weight stall means the GPU has activation-ready work, but none
            # of it is scheduler-ready because its weights are still absent.
            if self.has_weight_blocked_gpu_work():
                if self.gpu_blocked_since is None:
                    self.gpu_blocked_since = now
            else:
                self.gpu_blocked_since = None
            return

        if self.gpu_blocked_since is not None:
            self.gpu_weight_stall_ms += (
                now - self.gpu_blocked_since
            )
            self.gpu_blocked_since = None

        task = self.gpu_q.popleft()
        assert self.weight_state[task.layer] == "RESIDENT"
        dur = self.gpu_task_ms(task)
        self.gpu_busy = True
        self.gpu_busy_ms += dur

        row = TimelineRow(
            start_ms=now,
            end_ms=now + dur,
            lane="GPU",
            kind="FFN" if task.kind == "F" else "FULL",
            layer=task.layer,
            chunk=task.chunk,
            tokens=task.tokens,
            mode=self.pattern[task.layer],
            queue_wait_ms=now - task.enqueue_ms,
        )
        self.timeline.append(row)
        self.push_event(now + dur, "GPU_DONE", task)

    def dispatch(self, now: float) -> None:
        # Weight management is independent from the compute FIFO scheduler.
        self.try_start_transfer(now)
        self.try_start_cpu(now)
        self.try_start_gpu(now)

    def initialize(self) -> None:
        if self.preload_layers > self.vram_window:
            raise ValueError("preload_layers cannot exceed vram_window")
        if self.preload_layers > self.n_layers:
            raise ValueError("preload_layers cannot exceed n_layers")

        # Preloaded layers are treated as resident before t=0, so their copy
        # time is intentionally outside the simulated makespan.
        for layer in range(self.preload_layers):
            self.weight_state[layer] = "RESIDENT"
            self.live_layers += 1
            self.live_weight_mib += self.layer_weight_mib(layer)

        self.next_prefetch = self.preload_layers
        self.peak_live_layers = self.live_layers
        self.peak_live_weight_mib = self.live_weight_mib

        # Core Chunk Ring invariant: all initial activation-ready chunks are
        # admitted in chunk order.  GPU work is appended to the scheduler FIFO
        # only after both activation and weight dependencies are satisfied.
        for chunk, tokens in enumerate(self.chunk_sizes):
            if self.pattern[0] == "S":
                self.enqueue_cpu(Task("A", 0, chunk, tokens, 0.0))
            else:
                self.enqueue_gpu(Task("G", 0, chunk, tokens, 0.0))

    def run(self) -> SimResult:
        self.initialize()
        now = 0.0
        self.dispatch(now)

        while self.finished_chunks < self.n_chunks:
            if not self.events:
                raise RuntimeError(
                    "simulation deadlocked: "
                    f"finished={self.finished_chunks}/{self.n_chunks} "
                    f"cpu_q={len(self.cpu_q)} gpu_q={len(self.gpu_q)} "
                    f"gpu_pending={sum(len(q) for q in self.gpu_pending)} "
                    f"next_prefetch={self.next_prefetch} "
                    f"live_layers={self.live_layers}"
                )

            now = self.events[0][0]
            batch: List[Tuple[float, int, str, object]] = []
            while self.events and abs(self.events[0][0] - now) < 1e-12:
                batch.append(heapq.heappop(self.events))

            for _, _, event_kind, payload in batch:
                if event_kind == "XFER_DONE":
                    layer = int(payload)
                    self.xfer_busy = False
                    assert self.weight_state[layer] == "LOADING"
                    self.weight_state[layer] = "RESIDENT"
                    self.release_weight_ready_tasks(layer)

                elif event_kind == "CPU_DONE":
                    task = payload
                    assert isinstance(task, Task)
                    self.cpu_busy = False
                    # Split layer: ATTN completion appends FFN to the GPU FIFO.
                    self.enqueue_gpu(
                        Task("F", task.layer, task.chunk, task.tokens, now)
                    )

                elif event_kind == "GPU_DONE":
                    task = payload
                    assert isinstance(task, Task)
                    self.gpu_busy = False

                    self.gpu_done_count[task.layer] += 1
                    if self.gpu_done_count[task.layer] == self.n_chunks:
                        # Last chunk has left this layer's GPU stage.  No future
                        # task can touch these GPU weights, so eviction is exact.
                        self.evict_layer(task.layer, now)

                    self.enqueue_next_layer(
                        task.layer, task.chunk, task.tokens, now
                    )
                else:
                    raise RuntimeError("unknown event: " + event_kind)

            self.dispatch(now)

        makespan = now
        if self.gpu_blocked_since is not None:
            self.gpu_weight_stall_ms += makespan - self.gpu_blocked_since
            self.gpu_blocked_since = None

        return SimResult(
            makespan_ms=makespan,
            cpu_busy_ms=self.cpu_busy_ms,
            gpu_busy_ms=self.gpu_busy_ms,
            xfer_busy_ms=self.xfer_busy_ms,
            gpu_weight_stall_ms=self.gpu_weight_stall_ms,
            max_cpu_queue=self.max_cpu_queue,
            max_gpu_queue=self.max_gpu_queue,
            peak_live_layers=self.peak_live_layers,
            peak_live_weight_mib=self.peak_live_weight_mib,
            timeline=self.timeline,
        )


def split_by_chunk_size(tokens: int, chunk_size: int) -> List[int]:
    if tokens <= 0 or chunk_size <= 0:
        raise ValueError("tokens and chunk_size must be positive")
    result = []
    left = tokens
    while left > 0:
        n = min(left, chunk_size)
        result.append(n)
        left -= n
    return result


def even_pattern(n_layers: int, gpu_full_layers: int) -> str:
    if gpu_full_layers < 0 or gpu_full_layers > n_layers:
        raise ValueError("gpu_full_layers out of range")

    # Bresenham-style even distribution.  The result contains exactly the
    # requested number of G layers and avoids a giant relief block unless the
    # requested density itself requires it.
    out: List[str] = []
    acc = 0
    for _ in range(n_layers):
        acc += gpu_full_layers
        if acc >= n_layers:
            out.append("G")
            acc -= n_layers
        else:
            out.append("S")
    return "".join(out)


def parse_pattern(raw: Optional[str], n_layers: int, gpu_full_layers: int) -> str:
    if raw is None:
        return even_pattern(n_layers, gpu_full_layers)

    pattern = "".join(re.findall(r"[SGsg]", raw)).upper()
    if len(pattern) != n_layers:
        raise ValueError(
            f"pattern has {len(pattern)} layers, expected {n_layers}"
        )
    return pattern


def max_run(pattern: str, kind: str) -> int:
    best = 0
    cur = 0
    for ch in pattern:
        if ch == kind:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def write_csv(path: str, rows: List[TimelineRow]) -> None:
    rows = sorted(rows, key=lambda r: (r.start_ms, r.end_ms, r.lane, r.layer, r.chunk))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "start_ms", "end_ms", "duration_ms",
            "lane", "kind", "layer", "chunk", "tokens", "mode",
            "queue_wait_ms", "weight_mib",
        ])
        for r in rows:
            w.writerow([
                f"{r.start_ms:.6f}",
                f"{r.end_ms:.6f}",
                f"{r.end_ms - r.start_ms:.6f}",
                r.lane,
                r.kind,
                r.layer,
                r.chunk,
                r.tokens,
                r.mode,
                f"{r.queue_wait_ms:.6f}",
                f"{r.weight_mib:.6f}",
            ])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline FIFO simulator for CPU-ATTN / GPU-FFN Chunk Ring"
    )
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--tokens", type=int, default=358)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="128 -> 128,128,102 for 358 tokens; 64 -> 6 chunks",
    )

    p.add_argument(
        "--gpu-full",
        type=int,
        default=14,
        help="number of G layers when --pattern is omitted",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="explicit S/G string; separators are ignored",
    )

    p.add_argument(
        "--vram-window",
        type=int,
        default=4,
        help="maximum simultaneously loading/resident GPU layer weights",
    )
    p.add_argument(
        "--preload-layers",
        type=int,
        default=0,
        help="first N layer weights resident before t=0; copy time excluded",
    )

    p.add_argument("--base-tokens", type=int, default=DEFAULT_BASE_TOKENS)
    p.add_argument("--cpu-attn-ms", type=float, default=DEFAULT_CPU_ATTN_MS)
    p.add_argument("--cpu-ffn-ms", type=float, default=DEFAULT_CPU_FFN_MS)
    p.add_argument("--gpu-attn-ms", type=float, default=DEFAULT_GPU_ATTN_MS)
    p.add_argument("--gpu-ffn-ms", type=float, default=DEFAULT_GPU_FFN_MS)

    p.add_argument(
        "--attn-weight-mib", type=float, default=DEFAULT_ATTN_WEIGHT_MIB
    )
    p.add_argument(
        "--ffn-weight-mib", type=float, default=DEFAULT_FFN_WEIGHT_MIB
    )
    p.add_argument("--link-gbs", type=float, default=11.0)

    p.add_argument(
        "--trace",
        action="store_true",
        help="print the complete CPU/GPU/XFER/eviction timeline",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="optional timeline CSV output path",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.layers <= 0:
        raise SystemExit("--layers must be positive")
    if args.tokens <= 0 or args.chunk_size <= 0:
        raise SystemExit("--tokens/--chunk-size must be positive")
    if args.vram_window <= 0:
        raise SystemExit("--vram-window must be positive")
    if args.base_tokens <= 0:
        raise SystemExit("--base-tokens must be positive")
    if args.link_gbs <= 0.0:
        raise SystemExit("--link-gbs must be positive")

    pattern = parse_pattern(args.pattern, args.layers, args.gpu_full)
    chunks = split_by_chunk_size(args.tokens, args.chunk_size)

    n_g = pattern.count("G")
    n_s = pattern.count("S")

    # Aggregate service demand before FIFO fill/drain and weight stalls.  This
    # is useful for verifying the 34S/14G load-balancing calculation.
    token_scale_sum = sum(chunks) / float(args.base_tokens)
    ideal_cpu_work_ms = (
        n_s * args.cpu_attn_ms * token_scale_sum
    )
    ideal_gpu_work_ms = (
        (
            n_s * args.gpu_ffn_ms
            + n_g * (args.gpu_attn_ms + args.gpu_ffn_ms)
        )
        * token_scale_sum
    )

    sim = Simulator(
        pattern=pattern,
        chunk_sizes=chunks,
        vram_window=args.vram_window,
        preload_layers=args.preload_layers,
        base_tokens=args.base_tokens,
        cpu_attn_ms=args.cpu_attn_ms,
        cpu_ffn_ms=args.cpu_ffn_ms,
        gpu_attn_ms=args.gpu_attn_ms,
        gpu_ffn_ms=args.gpu_ffn_ms,
        attn_weight_mib=args.attn_weight_mib,
        ffn_weight_mib=args.ffn_weight_mib,
        link_gbs=args.link_gbs,
    )
    result = sim.run()

    print(
        "[FIFO_SIM_CONFIG]"
        f" layers={args.layers}"
        f" tokens={args.tokens}"
        f" chunk_size={args.chunk_size}"
        f" chunks={len(chunks)}"
        f" chunk_tokens={','.join(str(x) for x in chunks)}"
        f" split_layers={n_s}"
        f" gpu_full_layers={n_g}"
        f" vram_window={args.vram_window}"
        f" preload_layers={args.preload_layers}"
        f" link_gbs={args.link_gbs:.3f}"
    )
    print(
        "[FIFO_SIM_STAGE]"
        f" base_tokens={args.base_tokens}"
        f" cpu_attn_ms={args.cpu_attn_ms:.6f}"
        f" cpu_ffn_ref_ms={args.cpu_ffn_ms:.6f}"
        f" gpu_attn_ms={args.gpu_attn_ms:.6f}"
        f" gpu_ffn_ms={args.gpu_ffn_ms:.6f}"
        f" attn_weight_mib={args.attn_weight_mib:.3f}"
        f" ffn_weight_mib={args.ffn_weight_mib:.3f}"
    )
    print(
        "[FIFO_SIM_PATTERN]"
        f" pattern={pattern}"
        f" max_S_run={max_run(pattern, 'S')}"
        f" max_G_run={max_run(pattern, 'G')}"
    )
    print(
        "[FIFO_SIM_IDEAL_WORK]"
        f" cpu_ms={ideal_cpu_work_ms:.3f}"
        f" gpu_ms={ideal_gpu_work_ms:.3f}"
        f" gpu_over_cpu={ideal_gpu_work_ms / ideal_cpu_work_ms if ideal_cpu_work_ms > 0.0 else 0.0:.4f}x"
        " note=ignores_FIFO_fill_drain_and_weight_stalls"
    )

    makespan = result.makespan_ms
    cpu_util = result.cpu_busy_ms / makespan if makespan > 0.0 else 0.0
    gpu_util = result.gpu_busy_ms / makespan if makespan > 0.0 else 0.0
    xfer_util = result.xfer_busy_ms / makespan if makespan > 0.0 else 0.0

    print(
        "[FIFO_SIM_RESULT]"
        f" makespan_ms={makespan:.3f}"
        f" ms_per_layer={makespan / args.layers:.3f}"
        f" cpu_busy_ms={result.cpu_busy_ms:.3f}"
        f" gpu_busy_ms={result.gpu_busy_ms:.3f}"
        f" xfer_busy_ms={result.xfer_busy_ms:.3f}"
        f" gpu_weight_stall_ms={result.gpu_weight_stall_ms:.3f}"
    )
    print(
        "[FIFO_SIM_LANES]"
        f" cpu_util={cpu_util:.4f}"
        f" gpu_util={gpu_util:.4f}"
        f" xfer_util={xfer_util:.4f}"
        f" max_cpu_queue={result.max_cpu_queue}"
        f" max_gpu_queue={result.max_gpu_queue}"
    )
    print(
        "[FIFO_SIM_RESIDENCY]"
        f" peak_live_layers={result.peak_live_layers}"
        f" peak_live_weight_mib={result.peak_live_weight_mib:.3f}"
        f" window={args.vram_window}"
    )

    if args.trace:
        for r in sorted(
            result.timeline,
            key=lambda x: (
                x.start_ms, x.end_ms, x.lane, x.layer, x.chunk
            ),
        ):
            print(
                "[FIFO_SIM_TRACE]"
                f" lane={r.lane}"
                f" kind={r.kind}"
                f" layer={r.layer}"
                f" chunk={r.chunk}"
                f" tokens={r.tokens}"
                f" mode={r.mode}"
                f" start_ms={r.start_ms:.3f}"
                f" end_ms={r.end_ms:.3f}"
                f" dur_ms={r.end_ms - r.start_ms:.3f}"
                f" queue_wait_ms={r.queue_wait_ms:.3f}"
                f" weight_mib={r.weight_mib:.3f}"
            )

    if args.csv:
        write_csv(args.csv, result.timeline)
        print(f"[FIFO_SIM_CSV] path={args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
