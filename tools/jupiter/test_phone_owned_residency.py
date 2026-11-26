#!/usr/bin/env python3
"""Exercise cross-process Phone-owned HTP Graph residency (S1 or S4).

Keep the same Phone RPC server running. This launches independent llama-cli
processes with an identical execution template, checks that runs 2..N skipped
remote model/Graph/Buffer preparation, and compares their real final logits.
This is an integration check requiring real PC/Phone hardware, not CI.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from run_phone_owned_full_prefill import read_logits, logits_comparison
from phone_owned_models import inspect_model, resolve_splits


PREPARE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\] phase=PREPARE_DONE"
    r"[^\n]*?graphs=(\d+) load_ms=([\d.]+)"
    r" build_ms=([\d.]+) buffer_ms=([\d.]+)"
    r" precompile_ms=([\d.]+) phone_wall_ms=([\d.]+)"
    r" pc_wall_ms=([\d.]+) status=PASS"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-size", choices=("auto", "14b", "24b", "32b"), default="auto")
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
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out-dir", default="phone-owned-residency-check")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    spec = inspect_model(Path(args.model), args.model_size)
    positions, cpu = resolve_splits(
        spec, layouts=str(args.pc_layers) if args.pc_layers is not None else None,
        cpu_layers=args.cpu_layers, cuda_layers=args.cuda_layers,
        phone_layers=args.phone_layers)
    args.pc_layers, args.cpu_layers = positions[0], cpu
    args.model_size = spec.family
    if args.runs < 2:
        raise ValueError("--runs must be >= 2")
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_phone_owned_full_prefill.py")
    rows = []
    reference = None
    for run in range(1, args.runs + 1):
        work = output / f"run-{run:02d}"
        command = [
            sys.executable, str(runner),
            "--exe", args.exe, "--model", args.model,
            "--model-size", args.model_size,
            "--rpc", args.rpc, "--prompt-file", args.prompt_file,
            "--sessions", str(args.sessions),
            "--pc-layers", str(args.pc_layers),
            "--cpu-layers", str(args.cpu_layers),
            "--context", str(args.context),
            "--target-tokens", str(args.target_tokens),
            "--chunks", args.chunks,
            "--mode", "native", "--out-dir", str(work),
        ]
        print(f"[PHONE_RESIDENCY_TEST] RUN {run}/{args.runs}", flush=True)
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"run={run} failed (exit={result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        log_path = work / "native-full-prefill.log"
        log = log_path.read_text(encoding="utf-8", errors="replace")
        preps = list(PREPARE.finditer(log))
        if len(preps) != 1:
            raise RuntimeError(
                f"run={run}: expected exactly one successful PREPARE_DONE; "
                f"found {len(preps)}; inspect {log_path}"
            )
        prep = preps[0]
        stats = dict(zip(
            ("graphs", "load_ms", "build_ms", "buffer_ms",
             "precompile_ms", "phone_wall_ms", "pc_wall_ms"),
            (float(v) for v in prep.groups()),
        ))
        logits = read_logits(work / "native-logits.f32")
        top1 = max(range(len(logits)), key=lambda j: logits[j])
        report = json.loads((work / "results.json").read_text(encoding="utf-8"))
        row = {
            "run": run,
            "top1": top1,
            "prompt_eval_ms": report["native"]["prompt_eval_ms"],
            "prepare": stats,
            "log": str(log_path),
            "warm_hit": run > 1,
        }
        if reference is None:
            reference = logits
        else:
            parity = logits_comparison(reference, logits)
            row["same_prompt_parity"] = parity
            if (not parity["top1_same"] or
                    parity["cosine"] is None or parity["cosine"] < 0.999 or
                    parity["relative_rmse"] is None or
                    parity["relative_rmse"] > 0.01):
                raise RuntimeError(
                    f"run={run}: same-prompt logits changed after KV reset: "
                    f"{json.dumps(parity)}; inspect {log_path}"
                )
            # A true resident hit skips ALL four preparation phases.
            # Merely observing a fast model load is not proof of Graph reuse.
            for key in ("load_ms", "build_ms", "buffer_ms", "precompile_ms"):
                if stats[key] != 0:
                    raise RuntimeError(
                        f"run={run}: Graph cache MISS (nonzero {key}="
                        f"{stats[key]}). Check the Phone RPC log for "
                        f"'PREPARE_CACHE status=MISS' or a missing RESET. "
                        f"PC log: {log_path}"
                    )
        rows.append(row)
        print(
            f"[PHONE_RESIDENCY_TEST] run={run} top1={top1} "
            f"prefill_ms={row['prompt_eval_ms']} "
            f"prepare_pc_ms={stats['pc_wall_ms']} "
            f"graph_buffer_ms={stats['buffer_ms']} "
            f"cache={'HIT' if run > 1 else 'BASELINE'} status=PASS",
            flush=True,
        )
    path = output / "residency-results.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[PHONE_RESIDENCY_TEST] report={path} status=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PHONE_RESIDENCY_TEST] status=FAIL reason={exc}", file=sys.stderr)
        raise SystemExit(1)
