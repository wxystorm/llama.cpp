#!/usr/bin/env python3
"""Resumable ShareGPT40: bootstrap native GCNA profile, refresh each 5 PASS,
run the existing apply-wide/E-Beam/DAG Planner independently per prompt.

This wrapper delegates actual inference and model-owned Phone NPU configure
to run_prompt_sweep.py. It does not invent a second Planner or copy Phone logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from v2_profile_coordinator import prepare_native_profile

GROUPS = (128, 256, 512, 1024)
CHUNKS = (32, 48, 56, 64, 80, 96, 112, 128, 160)
COLUMNS = ("prompt", "group", "tokens_hint", "profile_version",
           "status", "attempt", "prefill_tps", "prefill_ms",
           "plan_layout", "runtime_layout", "xc", "xg", "xp",
           "npu_policy", "plan_applied", "predicted_ms", "result_path")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def expected_prompts(folder: Path) -> list[tuple[int, Path]]:
    items = [(g, folder / f"prompt_{g}_{i:02}.txt")
             for g in GROUPS for i in range(1, 11)]
    missing = [p.name for _, p in items if not p.is_file()]
    if missing:
        raise ValueError("Missing ShareGPT40 prompts: " + ", ".join(missing))
    return items


def token_hints(folder: Path) -> dict[str, int]:
    path = folder / "manifest.csv"
    result: dict[str, int] = {}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fields = {re.sub("[^a-z0-9]", "", (k or "").lower()): v
                      for k, v in row.items()}
            name = Path(fields.get("file", "")).name
            for key in ("modeltokens", "tokenstokenizer", "tokenizerntokens",
                        "prompttokens", "ntokens", "tokens"):
                value = fields.get(key, "")
                if value and value.isdigit() and int(value) > 0:
                    result[name] = int(value)
                    break
    return result


def get_hint(name: str, group: int, manifest: dict[str, int]) -> int:
    # Group 1024 includes prompts up to 2000 tokens. Never use the group
    # name as a falsely exact token count when allocating context.
    return manifest.get(name, {128: 255, 256: 511, 512: 1023,
                               1024: 2000}[group])


def latest(path: Path) -> dict:
    files = sorted(path.glob("attempt-*/result.json"))
    for file in reversed(files):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (ValueError, OSError):
            pass
    return {}


def results(out: Path, items: list[tuple[int, Path]],
            hints: dict[str, int]) -> list[dict]:
    rows = []
    for group, path in items:
        r = latest(out / "runs" / path.stem)
        rows.append({"prompt": path.name, "group": group,
                     "tokens_hint": get_hint(path.name, group, hints),
                     **{k: r.get(k, "") for k in COLUMNS
                        if k not in ("prompt", "group", "tokens_hint")}})
    return rows


def save_summary(out: Path, items: list[tuple[int, Path]],
                 hints: dict[str, int]) -> None:
    dst = out / "summary.csv"
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in results(out, items, hints):
            w.writerow(row)
    os.replace(tmp, dst)


def parse_npu_policy(stderr: Path) -> str:
    if not stderr.is_file():
        return "UNKNOWN"
    policy = "NONE"
    with stderr.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[REGION_RING_NPU_MEMORY]" in line:
                m = re.search(r"\bpolicy=(RESIDENT|STREAMING_2BANK|DISABLED)", line)
                if m:
                    policy = m.group(1)
            if "[PHONE_NPU_PLAN]" in line and "event=INSTALL" in line:
                policy = policy if policy != "NONE" else "PLAN_HTP"
    return policy


def collect_run(worker_root: Path) -> tuple[dict, Path]:
    files = sorted(worker_root.glob("*/prompt_*/result.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one worker result, found {len(files)}")
    return json.loads(files[0].read_text(encoding="utf-8")), files[0]


def profile_version_path(out: Path) -> Path:
    return out / "profile-state.json"


def ensure_profile(args: argparse.Namespace, root: Path,
                   out: Path, completed: int) -> int:
    state_path = profile_version_path(out)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    refreshed_at = int(state.get("refreshed_at", -1))
    version = int(state.get("version", -1))
    # Refresh only when another prompt remains; a completed batch of five
    # does not refresh needlessly at the very end of the campaign.
    refresh = completed > 0 and completed % args.profile_every == 0 and refreshed_at < completed
    if version < 0 or refresh:
        version = max(version + 1, 0)
        print(f"[PROFILE] event={'REFRESH' if refresh else 'BOOTSTRAP'} "
              f"version={version} completed={completed} "
              f"chunks={','.join(map(str, CHUNKS))} "
              f"refresh_chunks={args.refresh_chunks if refresh else 'ALL'}", flush=True)
        csv_file = prepare_native_profile(
            root, args.model, args.rpc, args.threads,
            force=refresh, dry_run=False, directory=args.profile_dir,
            profiler_exe=args.native_profile_exe,
            required_kvs=(512, 1024, 2048, 4096),
            refresh_chunks=tuple(int(v) for v in args.refresh_chunks.split(",")) if refresh else ())
        state = {"version": version, "refreshed_at": completed,
                 "profile_csv": str(csv_file),
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(state_path, state)
        print(f"[PROFILE] status=PASS version={version} csv={csv_file}", flush=True)
    else:
        print(f"[PROFILE] event=REUSE version={version} completed={completed}", flush=True)
    return version


def run_one(args: argparse.Namespace, root: Path, out: Path,
            group: int, prompt: Path, hint: int,
            version: int, attempt: int) -> dict:
    parent = out / "runs" / prompt.stem / f"attempt-{attempt:04d}"
    parent.mkdir(parents=True, exist_ok=False)
    staged = parent / "input"
    staged.mkdir()
    # Existing runtime runner expects prompt_<integer>.txt. Use the model
    # token hint from manifest; fallback to the safe upper group bound.
    shutil.copy2(prompt, staged / f"prompt_{hint}.txt")
    output = parent / "worker"
    command = [
        sys.executable,
        str(root / "tools/region-ring-experiments/run_prompt_sweep.py"),
        "--exe", str(args.exe),
        "--model", str(args.model),
        "--rpc", args.rpc,
        "--prompt-dir", str(staged),
        "--out", str(output),
        "--planner", "apply-wide",
        "--streaming", "on",
        "--v2-profile-dir", str(args.profile_dir),
        "--native-profile-exe", str(args.native_profile_exe),
        "--batch", str(args.batch), "--ubatch", str(args.ubatch),
        "--threads", str(args.threads), "--n-predict", str(args.n_predict),
        "--context-padding", str(args.context_padding),
        "--log-verbosity", "4",
    ]
    if args.no_mmap:
        command.append("--no-mmap")
    print(f"[PROMPT] event=PLAN_AND_RUN prompt={prompt.name} "
          f"tokens_hint={hint} profile=V{version} attempt={attempt}", flush=True)
    with (parent / "wrapper.stdout.log").open("wb") as stdout, \
         (parent / "wrapper.stderr.log").open("wb") as stderr:
        process = subprocess.run(command, cwd=root, stdout=stdout,
                                 stderr=stderr, check=False)
    try:
        raw, result_file = collect_run(output)
    except (RuntimeError, ValueError, OSError) as exc:
        return {"status": "FAIL", "attempt": attempt,
                "profile_version": version, "result_path": "",
                "error": str(exc), "exit_code": process.returncode}
    worker_log = result_file.parent / "engine.stderr.log"
    row = {
        "status": raw.get("status", "FAIL") if process.returncode == 0 else "FAIL",
        "attempt": attempt,
        "profile_version": version,
        "prefill_tps": raw.get("prefill_tps"),
        "prefill_ms": raw.get("prefill_ms"),
        "plan_layout": raw.get("plan_layout"),
        "runtime_layout": raw.get("runtime_layout"),
        "xc": raw.get("xc"), "xg": raw.get("xg"), "xp": raw.get("xp"),
        "npu_policy": parse_npu_policy(worker_log),
        "plan_applied": raw.get("plan_applied"),
        "predicted_ms": raw.get("predicted_ms"),
        "result_path": str(result_file),
        "exit_code": process.returncode,
    }
    if row["status"] == "PASS" and not row["plan_layout"]:
        row["status"] = "NO_PLAN"
    print(f"[PROMPT_PLAN] prompt={prompt.name} status={row['status']} "
          f"applied={row['plan_applied']} layout={row['plan_layout']} "
          f"runtime={row['runtime_layout']} "
          f"XC={row['xc']} XG={row['xg']} XN={row['xp']} "
          f"npu_weights={row['npu_policy']} predicted_ms={row['predicted_ms']}",
          flush=True)
    print(f"[PROMPT_RESULT] prompt={prompt.name} prefill={row['prefill_tps']} "
          f"tok/s prefill_ms={row['prefill_ms']} attempt={attempt}",
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--native-profile-exe", type=Path, default=None)
    ap.add_argument("--profile-dir", type=Path, default=None)
    ap.add_argument("--profile-every", type=int, default=5)
    ap.add_argument("--refresh-chunks", default="96,128")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=1)
    ap.add_argument("--context-padding", type=int, default=256)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="run at most this many previously-unfinished prompts")
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    for label in ("model", "prompt_dir"):
        setattr(args, label, getattr(args, label).expanduser().resolve())
    args.out = args.out.expanduser().resolve()
    args.exe = (args.exe or root / "build-tensor-v2/bin/Release/llama-completion.exe").resolve()
    args.native_profile_exe = (args.native_profile_exe or
                               root / "build-tensor-v2/bin/Release/region-ring-v2-stage-profile.exe").resolve()
    args.profile_dir = (args.profile_dir or
                        root / "model-profiles" / args.model.stem / "native-gcna-v2").resolve()
    for path in (args.model, args.exe, args.native_profile_exe):
        if not path.is_file():
            ap.error(f"Missing required file: {path}")
    if not args.prompt_dir.is_dir():
        ap.error(f"Missing prompt directory: {args.prompt_dir}")
    if args.profile_every <= 0 or args.max_retries < 0 or args.limit < 0 or \
            args.batch < args.ubatch or args.ubatch < 1 or args.n_predict < 1:
        ap.error("Invalid profile cadence, retry count, or compute settings")
    refresh = tuple(int(s) for s in args.refresh_chunks.split(",") if s)
    if not refresh or any(x not in CHUNKS for x in refresh):
        ap.error("--refresh-chunks must be a nonempty subset of nine Plan widths")
    items = expected_prompts(args.prompt_dir)
    hints = token_hints(args.prompt_dir)
    if args.dry_run:
        print(f"[DRY_RUN] model={args.model} rpc={args.rpc} "
              f"prompts={len(items)} profile_every={args.profile_every} "
              f"chunks={CHUNKS}", flush=True)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out / "run-config.json"
    settings = {"schema": 1, "model": str(args.model),
                "model_size": args.model.stat().st_size,
                "exe": str(args.exe), "native_profile_exe": str(args.native_profile_exe),
                "rpc": args.rpc, "prompt_dir": str(args.prompt_dir),
                "profile_every": args.profile_every,
                "refresh_chunks": list(refresh),
                "chunks": list(CHUNKS),
                "batch": args.batch, "ubatch": args.ubatch,
                "n_predict": args.n_predict, "threads": args.threads,
                "no_mmap": args.no_mmap, "context_padding": args.context_padding}
    if config_file.is_file():
        if json.loads(config_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("Existing output configuration differs; use a new --out")
    else:
        atomic_json(config_file, settings)
    executed = 0
    attempted_failed = False
    for group, prompt in items:
        past = latest(args.out / "runs" / prompt.stem)
        if past.get("status") == "PASS":
            print(f"[SKIP_PASS] {prompt.name}", flush=True)
            continue
        if args.limit and executed >= args.limit:
            break
        completed = sum(latest(args.out / "runs" / p.stem).get("status") == "PASS"
                        for _, p in items)
        version = ensure_profile(args, root, args.out, completed)
        attempts = list((args.out / "runs" / prompt.stem).glob("attempt-*/result.json"))
        first_attempt = len(attempts) + 1
        row = {}
        for attempt in range(first_attempt, first_attempt + args.max_retries + 1):
            row = run_one(args, root, args.out, group, prompt,
                          get_hint(prompt.name, group, hints), version, attempt)
            row["prompt"] = prompt.name
            row["group"] = group
            row["tokens_hint"] = get_hint(prompt.name, group, hints)
            atomic_json(args.out / "runs" / prompt.stem /
                        f"attempt-{attempt:04d}" / "result.json", row)
            save_summary(args.out, items, hints)
            if row["status"] == "PASS":
                break
        executed += 1
        attempted_failed = attempted_failed or row.get("status") != "PASS"
    save_summary(args.out, items, hints)
    done = sum(latest(args.out / "runs" / p.stem).get("status") == "PASS"
               for _, p in items)
    print(f"[PROFILE5_SUMMARY] PASS={done}/40 remaining={40-done} "
          f"csv={args.out / 'summary.csv'}", flush=True)
    return 0 if done == 40 or (args.limit and not attempted_failed) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[PROFILE5_INTERRUPTED] Rerun same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[PROFILE5_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
