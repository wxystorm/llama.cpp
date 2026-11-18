#!/usr/bin/env python3
"""Measure the cost of Prima's one-open/write/close-per-event trace pattern.

No CUDA, model or Phone required. Compare native Linux /tmp to WSL DrvFs
(/mnt/d). This probes filesystem overhead only, not total profiler overhead.
"""
from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path


def measure(directory: Path, iterations: int):
    durations = []
    line = b"fig3_probe,0,123,12345678901234,12345678901235,GPU Compute,CUDA_GRAPH,0,probe\n"
    fd, name = tempfile.mkstemp(prefix="prima_io_probe_", suffix=".csv", dir=directory)
    os.close(fd)
    try:
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            f = os.open(name, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(f, line)
            finally:
                os.close(f)
            durations.append(time.perf_counter_ns() - t0)
        return {
            "directory": str(directory),
            "n": iterations,
            "total_ms": round(sum(durations) / 1e6, 3),
            "avg_ms": round(sum(durations) / (len(durations) * 1e6), 5),
            "p50_ms": round(statistics.median(durations) / 1e6, 5),
            "p95_ms": round(sorted(durations)[int(0.95*(len(durations)-1))] / 1e6, 5),
        }
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--windows-dir", type=Path,
                        default=Path("/mnt/d/fig3-prima14b-fixed39-1"))
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    if not 10 <= args.iterations <= 5000:
        parser.error("--iterations must be between 10 and 5000")
    for directory in (args.linux_dir, args.windows_dir):
        if not directory.is_dir():
            print("[PRIMA_FIG3_IO] NOT_FOUND", directory)
            continue
        try:
            print("[PRIMA_FIG3_IO]", measure(directory, args.iterations),
                  flush=True)
        except OSError as error:
            print("[PRIMA_FIG3_IO] FAILED", directory, error, flush=True)


if __name__ == "__main__":
    main()
