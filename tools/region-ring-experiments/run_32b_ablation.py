#!/usr/bin/env python3
"""Qwen3-32B, 64-layer GPU/HTP weight-Bank and Cycle ablation.

Modes:
    A0 fixed G8/P6 x4 + G8, dual Bank + rank+2 prefetch
    A1 same fixed layout, dual Bank, demand-loaded weights
    A2 single Bank, coarse profiled multi-Cycle planning, GPU-tail head
    A3 single Bank, memory-greedy G -> [C] -> P (one Cycle)

Default: two ShareGPT40 prompts, one trial per mode. Does NOT run all 40.
The existing resumable 32B runner supplies independent G/C/N profiling,
memory probes, prefill metrics and HTP-only Phone setup.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_32b_all40.py")
FIXED_64 = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
DEFAULT_PROMPTS = ("prompt_128_01.txt", "prompt_512_01.txt")
MODES = ("A0", "A1", "A2", "A3")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   default=Path(r"E:\llama\models\Qwen3-32B-Q4_K_M.gguf"))
    p.add_argument("--rpc", default="10.39.52.114:50052")
    p.add_argument("--prompt-dir", type=Path,
                   default=Path(r"E:\llama\datasets\sharegpt\sharegpt40"))
    p.add_argument("--exe", type=Path, default=ROOT /
                   "build-tensor-v2/bin/Release/llama-completion.exe")
    p.add_argument("--out", type=Path, default=ROOT / "32b-bank-cycle-ablation")
    p.add_argument("--profile-csv", type=Path,
                   help="reuse an existing independent G/C/N profile.csv")
    p.add_argument("--mode", choices=MODES, action="append",
                   help="repeat to select modes; default all four")
    p.add_argument("--prompt", action="append",
                   help="repeat to select filenames; default 128_01 and 512_01")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--context-padding", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.repeat < 1 or args.repeat > 20:
        p.error("--repeat must be between 1 and 20")
    if args.ubatch < 1 or args.batch < args.ubatch:
        p.error("require batch >= ubatch >= 1")
    args.modes = tuple(dict.fromkeys(args.mode or MODES))
    args.prompts = tuple(dict.fromkeys(args.prompt or DEFAULT_PROMPTS))
    if not args.prompts:
        p.error("select at least one prompt")
    if not args.dry_run:
        for label, path in (("model", args.model), ("exe", args.exe),
                            ("runner", RUNNER)):
            if not path.is_file():
                p.error(f"{label} file is missing: {path}")
        if not args.prompt_dir.is_dir():
            p.error(f"prompt directory is missing: {args.prompt_dir}")
        for name in args.prompts:
            if Path(name).name != name or not (
                    args.prompt_dir / name).is_file():
                p.error(f"prompt file is missing: {name}")
        if args.profile_csv and not args.profile_csv.is_file():
            p.error(f"profile CSV not found: {args.profile_csv}")
    return args


def build_command(args, mode, out, shared_profile):
    cmd = [sys.executable, str(RUNNER),
           "--exe", str(args.exe),
           "--model", str(args.model),
           "--model-layers", "64",
           "--rpc", args.rpc,
           "--prompt-dir", str(args.prompt_dir),
           "--out", str(out),
           "--auto-plan", "--htp-only", "--profile-every", "0",
           "--profile-grid", "quick",
           "--streaming", "on",
           "--batch", str(args.batch), "--ubatch", str(args.ubatch),
           "--n-predict", "1", "--threads", str(args.threads),
           "--context-padding", str(args.context_padding),
           "--no-mmap", "--trace-tasks", "--stop-on-fail",
           "--max-retries", "0", "--log-verbosity", "4",
           "--env", f"LLAMA_REGION_RING_ABLATION={mode}"]
    for name in args.prompts:
        cmd += ["--only-prompt", name]
    if mode in ("A0", "A1"):
        cmd += ["--force-plan-layout", FIXED_64,
                "--force-plan-xc", "128",
                "--force-plan-xg", "128",
                "--force-plan-xp", "128"]
    if shared_profile:
        cmd += ["--real-stage-profile-csv", str(shared_profile)]
    return cmd


def read_summary(folder, selected):
    path = folder / "summary.csv"
    if not path.is_file():
        raise RuntimeError(f"runner did not write summary: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # The existing runner contains every manifest prompt in summary.csv,
    # even when only two are selected.
    result = {}
    for r in rows:
        name = r.get("prompt") or r.get("file") or r.get("name") or ""
        if name in selected:
            result[name] = r
    if set(result) != set(selected):
        raise RuntimeError(f"missing selected rows in {path}: "
                           f"{set(selected) - set(result)}")
    bad = [name for name, r in result.items()
           if r.get("status") != "PASS"]
    if bad:
        raise RuntimeError(f"non-PASS result(s) in {folder}: "
                           + ", ".join(f"{n}={result[n].get('status')}"
                                       for n in bad))
    return result


def check_runtime(folder, mode, selected):
    for name in selected:
        metrics = folder / "runs" / Path(name).stem / "attempt-0001" / "metrics.jsonl"
        if not metrics.is_file():
            raise RuntimeError(f"missing metrics: {metrics}")
        with metrics.open(encoding="utf-8") as f:
            rings = [json.loads(line) for line in f if line.strip()
                     and '"type":"region_ring"' in line]
        rings = [r for r in rings if r.get("type") == "region_ring"
                 and int(r.get("tokens", 0)) > 1]
        if not rings:
            raise RuntimeError(f"no prefill ring metrics in {metrics}")
        for r in rings:
            expected_banks = 1 if mode in ("A2", "A3") else 2
            if r.get("ablation_mode") != mode or                     r.get("gpu_bank_count") != expected_banks:
                raise RuntimeError(f"runtime ignored requested {mode}: "
                                   f"{metrics} ({r.get('ablation_mode')}, "
                                   f"{r.get('gpu_bank_count')} Banks)")
            if mode != "A0" and r.get("gpu_bank_streaming") is not False:
                raise RuntimeError(f"{mode} unexpectedly enabled GPU Bank "
                                   f"Mover: {metrics}")
        print(f"[ABLATION_CHECK] mode={mode} prompt={name} "
              f"banks={expected_banks} mover={'ON' if mode == 'A0' else 'OFF'} "
              "status=PASS", flush=True)


def run(args):
    # Run A2 first to produce exactly ONE native independent C/G/N profile,
    # then reuse it unchanged for A0/A1/A3. If A2 was not requested, the
    # first requested mode owns that calibration. All results retain their
    # own immutable run-config.json and resumable attempts.
    modes = (("A2",) + tuple(m for m in args.modes if m != "A2")
             if "A2" in args.modes else args.modes)
    profile = args.profile_csv.resolve() if args.profile_csv else None
    if profile is None:
        first = modes[0]
        candidate = args.out / f"{first}-r01" / "independent-profile/profile.csv"
        if candidate.is_file():
            profile = candidate.resolve()
    table = []
    for mode in modes:
        for repeat in range(1, args.repeat + 1):
            out = args.out / f"{mode}-r{repeat:02d}"
            cmd = build_command(args, mode, out, profile)
            print(f"\n[ABLATION] mode={mode} repeat={repeat}/{args.repeat} "
                  f"prompts={','.join(args.prompts)} out={out}", flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(cmd), flush=True)
                continue
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                raise RuntimeError(f"{mode} runner failed with code {rc}; "
                                   f"see {out / 'runs'}")
            check_runtime(out, mode, args.prompts)
            rows = read_summary(out, args.prompts)
            for name in args.prompts:
                table.append({"mode": mode, "repeat": repeat,
                              "prompt": name, **rows[name]})
            if profile is None:
                new_profile = out / "independent-profile/profile.csv"
                if not new_profile.is_file() or not (
                        new_profile.with_suffix(".json")).is_file():
                    raise RuntimeError("independent G/C/N profile missing; "
                                       f"expected {new_profile}")
                profile = new_profile.resolve()
                print(f"[ABLATION_PROFILE] event=SHARED csv={profile}",
                      flush=True)
        if not args.dry_run:
            args.out.mkdir(parents=True, exist_ok=True)
            out_csv = args.out / "ablation_summary.csv"
            keys = list(dict.fromkeys(k for row in table for k in row))
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(table)
            print(f"[ABLATION_SUMMARY] {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"[ABLATION_ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
