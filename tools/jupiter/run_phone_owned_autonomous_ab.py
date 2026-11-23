#!/usr/bin/env python3
"""Real Qwen3 Prefill A/B: synchronous Phone RUN vs autonomous Phone FIFO/P2C.

Each variant uses the same fresh Prompt, contiguous CPU/CUDA/Phone placement,
Chunk plan, Session count, private HTP KV and PC output head. All F32 logits
must match. Output records Prepare cache class to avoid cold/warm confusion.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from inspect_phone_owned_overlap import prepare_cost
from run_phone_owned_full_prefill import logits_comparison, read_logits
from phone_owned_models import inspect_model, resolve_splits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-size", choices=("auto", "14b", "32b"), default="auto")
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--pc-layers", type=int, default=None)
    p.add_argument("--cpu-layers", type=int, default=None)
    p.add_argument("--cuda-layers", type=int, default=None)
    p.add_argument("--phone-layers", type=int, default=None)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--target-tokens", type=int, default=175)
    p.add_argument("--chunks", default="88,87")
    p.add_argument("--out-dir", default="jupiter-autonomous-ab")
    p.add_argument("--reverse-order", action="store_true")
    args = p.parse_args()
    spec = inspect_model(Path(args.model), args.model_size)
    layout, cpu = resolve_splits(
        spec, layouts=str(args.pc_layers) if args.pc_layers is not None else None,
        cpu_layers=args.cpu_layers, cuda_layers=args.cuda_layers,
        phone_layers=args.phone_layers)
    args.pc_layers, args.cpu_layers = layout[0], cpu
    args.model_size = spec.family
    print(f"[PHONE_AUTONOMOUS_AB] model=Qwen3-{spec.family.upper()}"
          f" CPU={cpu} CUDA={args.pc_layers-cpu}"
          f" HTP={spec.layers-args.pc_layers}", flush=True)

    chunks = [int(x) for x in args.chunks.split(",")]
    if len(chunks) < 1 or sum(chunks) != args.target_tokens:
        raise ValueError("Jupiter plan must have one or more Chunks "
                         "summing to --target-tokens")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_phone_owned_full_prefill.py")
    order = ("autonomous", "sync") if args.reverse_order else ("sync", "autonomous")
    results = {}
    logs = {}
    for mode in order:
        case_dir = out / mode
        cmd = [
            sys.executable, str(runner),
            "--exe", args.exe,
            "--model", args.model,
            "--model-size", args.model_size,
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
        if mode == "autonomous":
            cmd.append("--autonomous-phone")
        print(f"[PHONE_AUTONOMOUS_AB] mode={mode} status=RUN", flush=True)
        subprocess.run(cmd, check=True)
        results[mode] = json.loads(
            (case_dir / "results.json").read_text(encoding="utf-8"))["native"]
        log = (case_dir / "native-full-prefill.log").read_text(
            encoding="utf-8", errors="replace")
        logs[mode] = log
        if "phase=COMPLETE" not in log or "output=REAL_LOGITS" not in log:
            raise RuntimeError(f"{mode}: no complete final-PC-logits marker")
        if mode == "autonomous":
            if log.count("phase=PHONE_AUTO_PUSH") != len(chunks):
                raise RuntimeError("some autonomous Chunk PUSHs are missing")
            if log.count("phase=PHONE_AUTO_P2C_DONE") != 1:
                raise RuntimeError("autonomous final P2C completion is missing")
            if "phase=PHONE_AUTO_START" not in log:
                raise RuntimeError("Phone autonomous stream was not armed")
        elif "phase=PHONE_AUTO_PUSH" in log:
            raise RuntimeError("synchronous control used autonomous RPC")

    parity = logits_comparison(
        read_logits(out / "sync" / "native-logits.f32"),
        read_logits(out / "autonomous" / "native-logits.f32"),
    )
    sync_prepare = prepare_cost(logs["sync"])
    auto_prepare = prepare_cost(logs["autonomous"])
    same_cache_class = (
        sync_prepare is not None and auto_prepare is not None and
        sync_prepare["zero_explicit_prepare_phases"] ==
        auto_prepare["zero_explicit_prepare_phases"]
    )
    correct = (
        parity["top1_same"] and parity["cosine"] is not None and
        parity["cosine"] >= 0.999 and parity["relative_rmse"] is not None and
        parity["relative_rmse"] <= 0.01
    )
    sync_ms = results["sync"]["prompt_eval_ms"]
    auto_ms = results["autonomous"]["prompt_eval_ms"]
    record = {
        "order": list(order),
        "model": "Qwen3-" + spec.family.upper(),
        "model_layers": spec.layers,
        "cpu_layers": cpu,
        "cuda_layers": args.pc_layers-cpu,
        "phone_layers": spec.layers-args.pc_layers,
        "status": "PASS" if correct else "FAIL",
        "parity_input_a": "sync_native",
        "parity_input_b": "phone_autonomous",
        "parity": parity,
        "sync_prefill_ms": sync_ms,
        "autonomous_prefill_ms": auto_ms,
        "observed_ratio": sync_ms / auto_ms if sync_ms and auto_ms else None,
        "sync_prepare": sync_prepare,
        "autonomous_prepare": auto_prepare,
        "same_prepare_cost_class": same_cache_class,
        "performance_interpretation": (
            "SAME_PREPARE_CLASS_SINGLE_PAIR_REPEAT_AND_CHECK_SERVER_LOG"
            if same_cache_class else
            "NOT_COMPARABLE_COLD_WARM_PREPARE"
        ),
        "phone_log_required": [
            "PHONE_AUTO START",
            "PHONE_AUTO INPUT_PUSH",
            "PHONE_AUTO EXEC_DONE",
            "PHONE_AUTO P2C_RETURN",
            "PHONE_AUTO STOP",
            "PREPARE_CACHE",
            "RESET graph_cache",
        ],
    }
    target = out / "autonomous-ab-results.json"
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[PHONE_AUTONOMOUS_AB] " +
          json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if correct else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PHONE_AUTONOMOUS_AB] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
