#!/usr/bin/env python3
"""Compare native Phone-owned synchronous vs bounded PC/Phone overlap.

Uses real llama-cli executions, not synthetic RPC requests. The Phone server
must stay running. Full F32 logits are compared, not just Top-1 token IDs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from run_phone_owned_full_prefill import logits_comparison, read_logits


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--pc-layers", type=int, default=32)
    p.add_argument("--cpu-layers", type=int, default=16)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--target-tokens", type=int, default=175)
    p.add_argument("--chunks", default="88,87")
    p.add_argument("--out-dir", default="phone-owned-async-ab")
    return p


def main() -> int:
    args = parser().parse_args()
    parts = [int(x) for x in args.chunks.split(",")]
    if len(parts) < 2:
        raise ValueError("async overlap A/B requires at least two Chunks")
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_phone_owned_full_prefill.py")
    results = {}
    logs = {}
    for variant in ("sync", "async"):
        case_dir = output / variant
        command = [
            sys.executable, str(runner),
            "--exe", args.exe,
            "--model", args.model,
            "--rpc", args.rpc,
            "--prompt-file", args.prompt_file,
            "--sessions", str(args.sessions),
            "--pc-layers", str(args.pc_layers),
            "--cpu-layers", str(args.cpu_layers),
            "--context", str(args.context),
            "--target-tokens", str(args.target_tokens),
            "--chunks", args.chunks,
            "--mode", "native",
            "--out-dir", str(case_dir),
        ]
        if variant == "async":
            command.append("--async-phone")
        print(f"[PHONE_ASYNC_AB] mode={variant} status=RUN", flush=True)
        subprocess.run(command, check=True)
        report = json.loads((case_dir / "results.json").read_text(
            encoding="utf-8"))
        results[variant] = report["native"]
        log = (case_dir / "native-full-prefill.log").read_text(
            encoding="utf-8", errors="replace")
        logs[variant] = log
        if "phase=COMPLETE" not in log or "output=REAL_LOGITS" not in log:
            raise RuntimeError(f"{variant}: full Prefill not completed")
        if variant == "async":
            if (log.count("phase=ASYNC_ENQUEUE") != len(parts) or
                    log.count("phase=CHUNK_DONE mode=ASYNC") != len(parts)):
                raise RuntimeError(
                    "async path did not enqueue and finish every Chunk")
            if "phase=ASYNC_WORKER" not in log:
                raise RuntimeError("native worker not enabled")
        else:
            if "phase=ASYNC_ENQUEUE" in log:
                raise RuntimeError("sync control accidentally enabled overlap")
    # A queued RPC on its own thread is not automatically evidence of
    # useful overlap. Record whether the previous Phone RPC spans the
    # enqueue of the next PC-produced Chunk; this is a timing hint, not
    # an exclusive PC-compute measurement.
    async_log = logs["async"]
    enqueued = {
        int(m.group(1)): int(m.group(2))
        for m in re.finditer(
            r"phase=ASYNC_ENQUEUE chunk=(\\d+) tokens=\\d+ t_us=(\\d+)",
            async_log)
    }
    completed = {
        int(m.group(1)): (int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"phase=CHUNK_DONE mode=ASYNC[^\\n]*?chunk=(\\d+)"
            r"[^\\n]*?t_begin_us=(\\d+) t_done_us=(\\d+)",
            async_log)
    }
    overlap_windows = sum(
        1 for chunk in range(1, len(parts))
        if chunk in enqueued and chunk - 1 in completed and
        completed[chunk - 1][0] < enqueued[chunk] < completed[chunk - 1][1]
    )
    sync_logits = read_logits(output / "sync" / "native-logits.f32")
    async_logits = read_logits(output / "async" / "native-logits.f32")
    parity = logits_comparison(sync_logits, async_logits)
    report = {
        "parity": parity,
        "phone_rpc_spanned_next_enqueue": overlap_windows,
        "candidate_overlap_boundaries": len(parts) - 1,
        "sync_prefill_ms": results["sync"]["prompt_eval_ms"],
        "async_prefill_ms": results["async"]["prompt_eval_ms"],
        "sync_log": str(output / "sync" / "native-full-prefill.log"),
        "async_log": str(output / "async" / "native-full-prefill.log"),
    }
    passed = (
        parity["top1_same"] and parity["cosine"] is not None and
        parity["cosine"] >= 0.999 and
        parity["relative_rmse"] is not None and
        parity["relative_rmse"] <= 0.01
    )
    report["status"] = "PASS" if passed else "FAIL"
    dest = output / "async-ab-results.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print("[PHONE_ASYNC_AB] " + json.dumps(report, ensure_ascii=False),
          flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PHONE_ASYNC_AB] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
