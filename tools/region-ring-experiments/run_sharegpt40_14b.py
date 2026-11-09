#!/usr/bin/env python3
"""Fixed Qwen3-14B Region Ring ShareGPT40 sweep with safe resume/reruns.

Each prompt has its own current attempt directory. Before a retry, the old
attempt moves into history/<prompt>/attempt-NNN; successful runs are skipped
on restart unless explicitly selected for rerun.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from run_prompt_sweep import (git_head, result_row, write_phone_copy_timeline,
                              write_phone_input_push_timeline)
import chunk_dag

PATTERN = re.compile(r"^prompt_(128|256|512|1024)_(\d{2})\.txt$", re.I)
LAYOUT = "G9,P1,G9,P1,G9,P1,G10"
CHUNK = 128
BATCH = 2048
UBATCH = 2048
CONTEXT = 4096
PREDICT = 32
THREADS = 8
SEED = 1234
FIELDS = (
    "prompt_file", "group", "attempt", "status", "actual_prefill_tokens",
    "prefill_ms", "prefill_tps", "actual_decode_tokens", "decode_ms",
    "decode_tps", "xg", "xp", "predicted_dag_ms",
    "exit_code", "commit", "run_dir",
)

# Explicit PC configuration to avoid inheriting an older run's layout/mode.
RING_ENV = {
    "LLAMA_HYBRID_REGION_RING_V1": "1",
    "LLAMA_HYBRID_REGION_RING_EXEC": "fifo-pc-phone",
    "LLAMA_HYBRID_REGION_RING_PLANNER": "apply-wide",
    "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE": "1",
    "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS": "64",
    "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM": "1",
    "LLAMA_HYBRID_REGION_RING_DUPLEX_PROFILE": "0",
    "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT": LAYOUT,
    "LLAMA_HYBRID_REGION_RING_FORCE_XC": str(CHUNK),
    "LLAMA_HYBRID_REGION_RING_FORCE_XG": str(CHUNK),
    "LLAMA_HYBRID_REGION_RING_FORCE_XP": str(CHUNK),
    "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM": "1",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS": "1",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL": "1",
    "LLAMA_REGION_RING_PHONE_INPUT_PUSH": "1",
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM": "1",
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES": "1",
    "LLAMA_REGION_RING_PHONE_GRAPHLESS": "1",
    "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW": "1",
    "LLAMA_REGION_RING_PHONE_PLAN_ASYNC": "1",
    "LLAMA_REGION_RING_PRIORITY_BANK_MOVER": "1",
    "LLAMA_REGION_RING_PRIORITY_PHONE_PUSH": "1",
    "LLAMA_REGION_RING_TASK_TRACE": "1",
    "LLAMA_REGION_RING_COALESCE_TINY_TAIL": "1",
}


def atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def prompt_id(value: str) -> str:
    name = Path(value).name
    if not name.lower().endswith(".txt"):
        name += ".txt"
    if not PATTERN.fullmatch(name):
        raise ValueError(f"invalid prompt name: {value} (use prompt_512_05)")
    return name.lower()


def names(values: list[str]) -> set[str]:
    return {prompt_id(item.strip()) for value in values
            for item in value.split(",") if item.strip()}


def summary(out: Path, prompts: list[tuple[int, Path]]) -> dict[str, int]:
    values = []
    counts: dict[str, int] = {}
    for group, prompt in prompts:
        row = read_json(out / prompt.stem / "result.json")
        status = row.get("status", "PENDING")
        if row and row.get("prompt_sha256") != hashlib.sha256(prompt.read_bytes()).hexdigest():
            status = "STALE"
        counts[status] = counts.get(status, 0) + 1
        values.append({name: (group if name == "group" else
                              row.get(name, "") if row else "")
                       for name in FIELDS} | {"prompt_file": prompt.name, "status": status})
    temp = out / "summary.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(values)
    os.replace(temp, out / "summary.csv")
    return counts


def archive(out: Path, folder: Path, previous: dict) -> int:
    history = out / "history" / folder.name
    history.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for item in history.iterdir()
                if (m := re.fullmatch(r"attempt-(\d+)", item.name))]
    number = max(int(previous.get("attempt") or 1), max(existing, default=0) + 1)
    destination = history / f"attempt-{number:03d}"
    # Never destroy a prior attempt even if a folder was manually renamed.
    while destination.exists():
        number += 1
        destination = history / f"attempt-{number:03d}"
    shutil.move(str(folder), str(destination))
    return number + 1


def run_prompt(args: argparse.Namespace, out: Path, group: int,
               prompt: Path, attempt: int, commit: str,
               plan: dict | None = None) -> tuple[dict, bool]:
    folder = out / prompt.stem
    folder.mkdir()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("LLAMA_HYBRID_REGION_RING_", "LLAMA_REGION_RING_")):
            env.pop(key)
    env.update(RING_ENV)
    if plan is not None:
        env["LLAMA_HYBRID_REGION_RING_FORCE_XG"] = str(plan["xg"])
        env["LLAMA_HYBRID_REGION_RING_FORCE_XP"] = str(plan["xp"])
    env["LLAMA_HYBRID_REGION_RING_PLAN_TOKENS"] = str(group)
    env["LLAMA_REGION_RING_METRICS_FILE"] = str(folder / "metrics.jsonl")
    command = [
        str(args.exe), "-m", str(args.model), "--rpc", args.rpc,
        "--split-mode", "tensor", "--flash-attn", "on",
        "-c", str(args.context), "-b", str(BATCH), "-ub", str(UBATCH),
        "-n", str(PREDICT), "-t", str(THREADS), "--seed", str(SEED),
        "--no-conversation", "-f", str(prompt), "-lv", "4", "--no-mmap",
    ]
    if plan is not None:
        atomic_json(folder / "plan.json", plan)
    atomic_json(folder / "config.json", {
        "command": command, "cwd": str(args.root), "commit": commit,
        "attempt": attempt, "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "ring_environment": {k: v for k, v in env.items()
                             if k.startswith(("LLAMA_HYBRID_REGION_RING_", "LLAMA_REGION_RING_"))},
    })
    interrupted = False
    code = -1
    start = time.perf_counter()
    with (folder / "output.stdout.txt").open("wb") as stdout, \
         (folder / "engine.stderr.log").open("wb") as stderr:
        process = None
        try:
            process = subprocess.Popen(command, cwd=args.root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr)
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            code = 130
        except OSError as exc:
            stderr.write(f"LAUNCH_ERROR: {exc}\n".encode("utf-8"))
    elapsed = 1000 * (time.perf_counter() - start)
    base_args = SimpleNamespace(model=args.model, planner="apply-wide",
                                skip_profile=True, batch=BATCH, ubatch=UBATCH,
                                threads=THREADS, seed=SEED, n_predict=PREDICT,
                                streaming="on", no_mmap=True)
    row = result_row(base_args, folder, prompt, group, args.context,
                     code, elapsed, commit)
    row.update(attempt=attempt, group=group,
               xg=(plan or {}).get("xg", CHUNK),
               xp=(plan or {}).get("xp", CHUNK),
               predicted_dag_ms=(plan or {}).get("predicted_dag_ms"),
               finished_utc=datetime.now(timezone.utc).isoformat())
    if interrupted:
        row["status"] = "INTERRUPTED"
    write_phone_input_push_timeline(folder)
    write_phone_copy_timeline(folder)
    atomic_json(folder / "result.json", row)
    return row, interrupted


def ensure_chunk_profile(args: argparse.Namespace, prompts: list[tuple[int, Path]],
                         commit: str) -> dict:
    """Run the full fixed-layout profile *once*, cache verified real traces."""
    candidates = tuple(x for x in chunk_dag.GRID if x <= 128)
    representative = next(((g, p) for g, p in prompts if g == 512), None)
    if representative is None:
        representative = min(prompts, key=lambda pair: abs(pair[0] - 512))
    group, prompt = representative
    profile_dir = args.out / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "chunk-profile.json"
    signature = dict(schema=1, model=str(args.model),
                     model_size=args.model.stat().st_size,
                     executable=str(args.exe), rpc=args.rpc,
                     prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
                     prompt=prompt.name, context=args.context,
                     grid=list(candidates), fixed_layout=chunk_dag.LAYOUT)
    existing = read_json(profile_path)
    if not args.update_profile and existing.get("signature") == signature:
        if len(existing.get("probes", [])) == len(candidates):
            print(f"[PROFILE] REUSE {profile_path}", flush=True)
            return existing
    probes = []
    for x in candidates:
        probe_dir = profile_dir / "probes" / f"x{x}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        folder = probe_dir / prompt.stem
        previous = read_json(folder / "result.json")
        sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
        if (folder.exists() and
            (args.update_profile or previous.get("status") != "PASS" or
             previous.get("prompt_sha256") != sha)):
            attempt = archive(probe_dir, folder, previous)
        else:
            attempt = 1
        if not folder.exists():
            print(f"[PROFILE] measure G/P X={x} using {prompt.name}", flush=True)
            # Probe is fixed-layout. Only device Chunk widths change.
            probe_plan = {"xg": x, "xp": x, "profile_probe": True}
            row, interrupted = run_prompt(args, probe_dir, group, prompt,
                                          attempt, commit, probe_plan)
            if interrupted or row["status"] != "PASS":
                raise RuntimeError(f"Chunk profile failed at X={x}: {folder}")
        probes.append(chunk_dag.extract_probe(folder, x))
    profile = {"signature": signature, "probes": probes,
               "saved_utc": datetime.now(timezone.utc).isoformat()}
    if args.update_profile and profile_path.exists():
        history = profile_dir / "history"
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        old = history / f"chunk-profile-{stamp}.json"
        if old.exists():
            old = history / f"chunk-profile-{stamp}-{os.getpid()}.json"
        shutil.copy2(profile_path, old)
    atomic_json(profile_path, profile)
    print(f"[PROFILE] READY points={sum(len(x['stages']) for x in probes)} "
          f"stored={profile_path}", flush=True)
    return profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--fixed-chunks", action="store_true",
                    help="disable cached-profile DAG planning; use XG=XP=128")
    ap.add_argument("--update-profile", action="store_true",
                    help="rerun the one-time Chunk calibration, keep old probe history")
    ap.add_argument("--only", action="append", default=[], metavar="PROMPT",
                    help="run only named prompts (unless already PASS)")
    ap.add_argument("--rerun", action="append", default=[], metavar="PROMPT",
                    help="force rerun named prompts, even if PASS")
    ap.add_argument("--rerun-below", type=float, default=0,
                    help="rerun PASS prompts with prefill tok/s below threshold")
    ap.add_argument("--limit", type=int, default=0, help="maximum processes this invocation")
    ap.add_argument("--delay", type=float, default=2.0, help="pause between processes in seconds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.root = Path(__file__).resolve().parents[2]
    for key in ("exe", "model", "prompt_dir", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.exe.is_file() or not args.model.is_file() or not args.prompt_dir.is_dir():
        ap.error("--exe, --model and --prompt-dir must exist")
    if args.context < 2560 or args.limit < 0 or args.delay < 0 or args.rerun_below < 0:
        ap.error("context must be >=2560; limit, delay and rerun-below must be >=0")
    prompts = [(int(m.group(1)), p) for p in args.prompt_dir.iterdir()
               if p.is_file() and (m := PATTERN.fullmatch(p.name))]
    prompts.sort(key=lambda item: (item[0], item[1].name))
    if not prompts:
        ap.error("no prompt_128_01.txt-style inputs found")
    try:
        only = names(args.only)
        rerun = names(args.rerun)
    except ValueError as exc:
        ap.error(str(exc))
    available = {p.name.lower() for _, p in prompts}
    missing = (only | rerun) - available
    if missing:
        ap.error("prompt(s) not found: " + ", ".join(sorted(missing)))
    selected = only | rerun if (only or rerun) else available
    configuration = {
        "schema": 1, "exe": str(args.exe), "model": str(args.model),
        "model_bytes": args.model.stat().st_size, "rpc": args.rpc,
        "prompt_dir": str(args.prompt_dir), "context": args.context,
        "batch": BATCH, "ubatch": UBATCH, "n_predict": PREDICT,
        "threads": THREADS, "seed": SEED, "no_mmap": True,
        "ring_environment": RING_ENV,
        "chunk_planning": "FIXED_128" if args.fixed_chunks else "PROFILE_DAG_V1",
    }
    config_path = args.out / "run-config.json"
    if config_path.exists() and read_json(config_path) != configuration:
        ap.error(f"run configuration differs: {config_path}; use a different --out")
    if args.out.exists() and not config_path.exists() and any(args.out.iterdir()):
        ap.error(f"output folder is not an initialized sweep: {args.out}")
    commit = git_head(args.root)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            atomic_json(config_path, configuration)
    profile = None
    if not args.fixed_chunks and not args.dry_run:
        profile = ensure_chunk_profile(args, prompts, commit)
    launched = 0
    stopped = False
    for index, (group, prompt) in enumerate(prompts):
        if prompt.name.lower() not in selected:
            continue
        folder = args.out / prompt.stem
        previous = read_json(folder / "result.json")
        sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
        is_pass = previous.get("status") == "PASS" and previous.get("prompt_sha256") == sha
        try:
            slow = args.rerun_below > 0 and float(previous.get("prefill_tps") or 0) < args.rerun_below
        except (ValueError, TypeError):
            slow = True
        force = prompt.name.lower() in rerun
        if is_pass and not force and not slow:
            print(f"[SKIP] {prompt.name}: PASS {previous.get('prefill_tps')} tok/s", flush=True)
            continue
        if args.limit and launched >= args.limit:
            break
        reason = "force" if force else "slow" if is_pass and slow else "retry" if folder.exists() else "new"
        if args.dry_run:
            print(f"[DRY] {prompt.name}: {reason}", flush=True)
            launched += 1
            continue
        attempt = archive(args.out, folder, previous) if folder.exists() else 1
        print(f"[{launched + 1}] RUN {prompt.name} group={group} attempt={attempt} reason={reason}", flush=True)
        plan = None
        if profile is not None:
            measured = chunk_dag.manifest_tokens(args.prompt_dir / "manifest.csv", prompt.name)
            plan_tokens = measured or {128: 192, 256: 384, 512: 768, 1024: 1500}[group]
            plan = chunk_dag.choose(profile, plan_tokens)
            plan["token_source"] = "manifest" if measured else "group_midpoint_estimate"
            print(f"  [DAG] tokens~{plan_tokens} XG={plan['xg']} "
                  f"XP={plan['xp']} predicted={plan['predicted_dag_ms']}ms", flush=True)
        row, interrupted = run_prompt(args, args.out, group, prompt, attempt, commit, plan)
        launched += 1
        summary(args.out, prompts)
        print(f"  {row['status']}: prefill={row['prefill_tps']} tok/s ",
              f"({row['prefill_ms']} ms), decode={row['decode_tps']} tok/s", flush=True)
        if interrupted:
            stopped = True
            break
        if args.delay and index < len(prompts) - 1:
            time.sleep(args.delay)
    if args.dry_run:
        return 0
    counts = summary(args.out, prompts)
    print(f"Summary: {args.out / 'summary.csv'} | " +
          " ".join(f"{key}={value}" for key, value in sorted(counts.items())), flush=True)
    if stopped:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 1 if counts.get("FAIL", 0) or counts.get("INTERRUPTED", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
