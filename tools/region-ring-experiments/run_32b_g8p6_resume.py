#!/usr/bin/env python3
"""Qwen3-32B G8/P6 resumable sweep. Keep all attempts and the fastest PASS."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime

import run_prompt_sweep as old

LAYOUT = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
CHUNK = 128
PROMPT_RE = re.compile(r"^prompt_(\d+)(?:_(\d+))?\.txt$", re.I)
COUNT_RE = re.compile(r"Total number of tokens:\s*(\d+)", re.I)
SUMMARY_COLUMNS = (
    "prompt_file", "group", "model_tokens", "context", "status",
    "attempts", "best_prefill_tps", "best_prefill_ms", "best_run_dir",
    "latest_status", "latest_prefill_tps", "latest_run_dir",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def write_csv(path, columns, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prompt_list(folder):
    prompts = []
    for path in folder.iterdir():
        match = PROMPT_RE.fullmatch(path.name)
        if match and path.is_file():
            prompts.append((int(match.group(1)),
                            int(match.group(2) or 0), path))
    return sorted(prompts, key=lambda row: (row[0], row[1], row[2].name))


def context_size(tokens, predict=1, padding=256, minimum=1024):
    if tokens <= 0 or predict <= 0:
        raise ValueError("invalid token count or predict")
    return max(minimum, ((tokens + predict + padding + 255) // 256) * 256)


def actual_model_tokens(tokenizer, model, prompt):
    proc = subprocess.run([
        str(tokenizer), "-m", str(model), "-f", str(prompt),
        "--ids", "--show-count", "--log-disable",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, timeout=180,
        text=True, encoding="utf-8", errors="replace")
    match = COUNT_RE.search(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0 or not match:
        raise RuntimeError("llama-tokenize failed for " + prompt.name
                           + ": " + (proc.stdout + proc.stderr)[-800:])
    result = int(match.group(1))
    if result <= 0:
        raise RuntimeError("empty model token count: " + prompt.name)
    return result


def tokens_for_prompts(args, prompts):
    filename = args.out / "token_counts.json"
    stat = args.model.stat()
    key = [str(args.model), stat.st_size, stat.st_mtime_ns,
           str(args.tokenizer), args.tokenizer.stat().st_size]
    cache = read_json(filename, {})
    if cache.get("model_key") != key:
        cache = {"model_key": key, "prompts": {}}
    result = {}
    for _, _, prompt in prompts:
        digest = sha256(prompt)
        value = cache["prompts"].get(prompt.name, {})
        if value.get("sha256") != digest or not value.get("tokens"):
            value = {
                "sha256": digest,
                "tokens": actual_model_tokens(args.tokenizer, args.model, prompt),
            }
            cache["prompts"][prompt.name] = value
            write_json(filename, cache)
        result[prompt.name] = value
        print(f"[TOKENS] {prompt.name}: model={value['tokens']}", flush=True)
    print(f"[TOKENS] source=model count={len(result)}", flush=True)
    return result


def folders_for(out, prompt):
    parent = out / "attempts" / prompt.stem
    if not parent.is_dir():
        return []
    dirs = [p for p in parent.iterdir()
            if p.is_dir() and re.fullmatch(r"attempt-\d+", p.name)]
    return sorted(dirs, key=lambda p: int(p.name.split("-")[1]))


def prior_runs(out, prompt, digest, min_tps):
    dirs = folders_for(out, prompt)
    records = []
    best = None
    for folder in dirs:
        row = read_json(folder / "result.json", {})
        config = read_json(folder / "config.json", {})
        row = dict(row)
        row.setdefault("status", "INTERRUPTED")
        row["run_dir"] = str(folder)
        row["attempt"] = int(folder.name.split("-")[1])
        row["group"] = int(PROMPT_RE.fullmatch(prompt.name).group(1))
        row["model_tokens"] = config.get("model_tokens")
        row["recorded_at"] = config.get("started_at")
        records.append(row)
        if row.get("status") != "PASS" or row.get("prompt_sha256") != digest:
            continue
        try:
            score = float(row.get("prefill_tps") or 0)
        except (ValueError, TypeError):
            score = 0.0
        if score > 0 and (best is None or
                          score > float(best["prefill_tps"])):
            best = row
    status = (
        ("PASS" if float(best["prefill_tps"]) >= min_tps else "LOW_TPS")
        if best else ("FAIL" if dirs else "PENDING")
    )
    return {
        "status": status, "best": best,
        "latest": records[-1] if records else None,
        "attempts": len(dirs), "records": records,
    }


def overview(args, prompts, token_map):
    summaries, attempts = [], []
    passed = 0
    for group, _, prompt in prompts:
        entry = token_map[prompt.name]
        tokens, digest = entry["tokens"], entry["sha256"]
        previous = prior_runs(args.out, prompt, digest, args.min_tps)
        best, latest = previous["best"], previous["latest"]
        passed += int(previous["status"] == "PASS")
        summaries.append({
            "prompt_file": prompt.name, "group": group,
            "model_tokens": tokens,
            "context": context_size(tokens, args.predict, args.padding,
                                    args.min_context),
            "status": previous["status"], "attempts": previous["attempts"],
            "best_prefill_tps": best.get("prefill_tps") if best else None,
            "best_prefill_ms": best.get("prefill_ms") if best else None,
            "best_run_dir": best.get("run_dir") if best else None,
            "latest_status": latest.get("status") if latest else None,
            "latest_prefill_tps": latest.get("prefill_tps") if latest else None,
            "latest_run_dir": latest.get("run_dir") if latest else None,
        })
        attempts.extend(previous["records"])
    write_csv(args.out / "summary.csv", SUMMARY_COLUMNS, summaries)
    write_csv(args.out / "attempts.csv",
              ["attempt", "group", "model_tokens", "recorded_at"] + old.FIELDS,
              attempts)
    return passed, len(prompts) - passed


def stable_config(args, commit):
    stat = args.model.stat()
    return {
        "version": 1, "model": str(args.model),
        "model_size": stat.st_size, "model_mtime_ns": stat.st_mtime_ns,
        "exe": str(args.exe), "tokenizer": str(args.tokenizer),
        "prompt_dir": str(args.prompt_dir), "rpc": args.rpc,
        "git_commit": commit, "layout": LAYOUT,
        "xc": CHUNK, "xg": CHUNK, "xp": CHUNK,
        "batch": args.batch, "ubatch": args.ubatch,
        "predict": args.predict, "threads": args.threads,
        "seed": args.seed, "min_context": args.min_context,
        "padding": args.padding, "log_verbosity": args.log_verbosity,
        "ram_preload": not args.no_ram_preload,
        "trace": not args.no_trace, "mover_priority": args.mover_priority,
    }


def ensure_config(args, commit):
    path = args.out / "sweep-config.json"
    current = stable_config(args, commit)
    previous = read_json(path)
    if previous is None:
        # Do not overwrite an unrelated sweep's results directory.
        if any(args.out.iterdir()):
            raise RuntimeError(
                "Existing output has no sweep-config.json: " + str(args.out)
                + ". Use a fresh -Out to preserve those results."
            )
        write_json(path, current)
    elif previous != current:
        changed = [key for key in sorted(set(current) | set(previous))
                   if current.get(key) != previous.get(key)]
        raise RuntimeError(
            "Existing output configuration differs: "
            + ", ".join(changed)
            + ". Use a different -Out to avoid mixing measurements."
        )


def inference_env(args, folder, tokens):
    env = os.environ.copy()
    for key in (
        "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT",
        "LLAMA_HYBRID_REGION_RING_FORCE_XC",
        "LLAMA_HYBRID_REGION_RING_FORCE_XG",
        "LLAMA_HYBRID_REGION_RING_FORCE_XP",
        "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE",
        "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS",
        "LLAMA_REGION_RING_PRIORITY_BANK_MOVER",
        "LLAMA_REGION_RING_GPU_BANK_READY_FIRST",
    ):
        env.pop(key, None)
    env.update({
        "LLAMA_HYBRID_REGION_RING_V1": "1",
        "LLAMA_HYBRID_REGION_RING_EXEC": "fifo-pc-phone",
        "LLAMA_HYBRID_REGION_RING_PLANNER": "apply-wide",
        "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE": "1",
        "LLAMA_HYBRID_REGION_RING_PLAN_TOKENS": str(tokens),
        "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS": str(CHUNK),
        "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM": "1",
        "LLAMA_HYBRID_REGION_RING_DUPLEX_PROFILE": "0",
        "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT": LAYOUT,
        "LLAMA_HYBRID_REGION_RING_FORCE_XC": str(CHUNK),
        "LLAMA_HYBRID_REGION_RING_FORCE_XG": str(CHUNK),
        "LLAMA_HYBRID_REGION_RING_FORCE_XP": str(CHUNK),
        "LLAMA_REGION_RING_METRICS_FILE": str(folder / "metrics.jsonl"),
        "LLAMA_REGION_RING_PC_GPU_RAM_PRELOAD":
            "0" if args.no_ram_preload else "1",
        "LLAMA_REGION_RING_TASK_TRACE": "0" if args.no_trace else "1",
        "LLAMA_REGION_RING_GPU_BANK_READY_FIRST": "0",
        "GGML_SCHED_WEIGHT_STAGE_PINNED": "0",
    })
    if args.mover_priority:
        env["LLAMA_REGION_RING_PRIORITY_BANK_MOVER"] = "1"
    return env


def run_prompt(args, root, commit, group, prompt, tokens, digest):
    previous = folders_for(args.out, prompt)
    number = max((int(p.name.split("-")[1]) for p in previous), default=0) + 1
    folder = (args.out / "attempts" / prompt.stem /
              f"attempt-{number:04d}")
    folder.mkdir(parents=True, exist_ok=False)
    ctx = context_size(tokens, args.predict, args.padding, args.min_context)
    env = inference_env(args, folder, tokens)
    command = [
        str(args.exe), "-m", str(args.model), "--rpc", args.rpc,
        "--split-mode", "tensor", "--flash-attn", "on",
        "-c", str(ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
        "-n", str(args.predict), "-t", str(args.threads),
        "--seed", str(args.seed), "--no-conversation",
        "-f", str(prompt), "-lv", str(args.log_verbosity),
    ]
    config = {
        "command": command, "cwd": str(root), "repeat": 1,
        "prompt_file": str(prompt), "prompt_hint": group,
        "prompt_sha256": digest, "model_tokens": tokens,
        "context": ctx, "attempt": number,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "no_mmap": False, "skip_profile": True,
        "layout": LAYOUT, "xc": CHUNK, "xg": CHUNK, "xp": CHUNK,
        "ring_environment": {
            key: value for key, value in env.items()
            if key.startswith("LLAMA_HYBRID_REGION_RING_")
            or key.startswith("LLAMA_REGION_RING_")
            or key == "GGML_SCHED_WEIGHT_STAGE_PINNED"
        },
    }
    write_json(folder / "config.json", config)
    print(f"[PLAN] {prompt.name}: prompt_tokens={tokens} context={ctx} "
          f"layout={LAYOUT} XC={CHUNK} XG={CHUNK} XP={CHUNK} "
          f"profile=SKIPPED predict={args.predict}", flush=True)
    start = time.perf_counter()
    code, interrupted = 127, False
    with (folder / "output.stdout.txt").open("wb") as stdout, \
         (folder / "engine.stderr.log").open("wb") as stderr:
        process = None
        try:
            process = subprocess.Popen(
                command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
            )
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            code = 130
        except OSError as error:
            stderr.write(f"LAUNCH_ERROR: {error}\n".encode("utf-8"))
    elapsed = 1000.0 * (time.perf_counter() - start)
    row = old.result_row(
        args, folder, prompt, group, ctx, code, elapsed, commit
    )
    row["prompt_sha256"] = digest
    if interrupted:
        row["status"] = "INTERRUPTED"
    old.write_phone_input_push_timeline(folder)
    old.write_phone_copy_timeline(folder)
    write_json(folder / "result.json", row)
    print(f"  {row['status']}: prefill={row['prefill_tps']} tok/s "
          f"({row['prefill_ms']} ms), decode={row['decode_tps']} tok/s",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    exe_dir = repo / "build-tensor-v2" / "bin" / "Release"
    ap.add_argument("--model", type=Path,
                    default=Path(r"E:\llama\models\Qwen3-32B-Q4_K_M.gguf"))
    ap.add_argument("--rpc", default="10.39.52.114:50052")
    ap.add_argument("--exe", type=Path, default=exe_dir / "llama-completion.exe")
    ap.add_argument("--tokenizer", type=Path,
                    default=exe_dir / "llama-tokenize.exe")
    ap.add_argument("--prompt-dir", type=Path,
                    default=Path(r"D:\prima.cpp\prompts"))
    ap.add_argument("--out", type=Path,
                    default=repo / "32b-g8p6-sharegpt40")
    ap.add_argument("--limit", type=int, default=0,
                    help="max prompts this invocation; 0 = all pending")
    ap.add_argument("--rerun", default="",
                    help="rerun this exact prompt, e.g. prompt_512_01.txt")
    ap.add_argument("--rerun-all", action="store_true")
    ap.add_argument("--min-tps", type=float, default=0.0)
    ap.add_argument("--group", type=int, action="append")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--predict", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-context", type=int, default=1024)
    ap.add_argument("--padding", type=int, default=256)
    ap.add_argument("--log-verbosity", type=int, default=4)
    ap.add_argument("--mover-priority", action="store_true")
    ap.add_argument("--no-ram-preload", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()
    # Reuse the existing sweep's telemetry-to-CSV parser unchanged.
    args.planner = "apply-wide"
    args.skip_profile = True
    args.streaming = "on"
    args.no_mmap = False
    if (args.limit < 0 or args.min_tps < 0 or args.ubatch <= 0 or
            args.batch < args.ubatch or args.predict <= 0 or
            args.min_context <= 0 or args.padding < 0):
        ap.error("invalid limit/tps/batch/context/predict arguments")
    if args.rerun and args.rerun_all:
        ap.error("--rerun and --rerun-all are mutually exclusive")
    for name in ("model", "exe", "tokenizer", "prompt_dir", "out"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("model", "exe", "tokenizer"):
        if not getattr(args, name).is_file():
            ap.error(f"--{name} not found: {getattr(args, name)}")
    if not args.prompt_dir.is_dir():
        ap.error(f"Prompt directory does not exist: {args.prompt_dir}")
    all_prompts = prompt_list(args.prompt_dir)
    if not all_prompts:
        ap.error("No prompt_128_01.txt / prompt_128.txt files found")
    args.out.mkdir(parents=True, exist_ok=True)
    commit = old.git_head(repo)
    ensure_config(args, commit)
    tokens = tokens_for_prompts(args, all_prompts)
    target = args.rerun.lower().removesuffix(".txt")
    if target and not target.startswith("prompt_"):
        target = "prompt_" + target
    if target and not any(p.stem.lower() == target
                          for _, _, p in all_prompts):
        ap.error("--rerun did not match any prompt")
    candidates = []
    for group, _, prompt in all_prompts:
        prior = prior_runs(args.out, prompt, tokens[prompt.name]["sha256"],
                           args.min_tps)
        if target:
            eligible, reason = prompt.stem.lower() == target, "rerun"
        elif args.group and group not in args.group:
            eligible, reason = False, "other-group"
        elif args.rerun_all:
            eligible, reason = True, "rerun-all"
        else:
            eligible = prior["status"] != "PASS"
            reason = ("new" if prior["attempts"] == 0 else
                      "low-tps" if prior["status"] == "LOW_TPS" else "retry")
        if eligible:
            candidates.append((group, prompt, reason))
    if args.limit:
        candidates = candidates[:args.limit]
    passed, pending = overview(args, all_prompts, tokens)
    if args.status_only or not candidates:
        print(f"Summary: {args.out / 'summary.csv'} | "
              f"PASS={passed} PENDING={pending}", flush=True)
        return 0
    for index, (group, prompt, reason) in enumerate(candidates, 1):
        attempts = len(folders_for(args.out, prompt)) + 1
        print(f"[{index}] RUN {prompt.name} group={group} "
              f"attempt={attempts} reason={reason}", flush=True)
        row = run_prompt(
            args, repo, commit, group, prompt,
            tokens[prompt.name]["tokens"], tokens[prompt.name]["sha256"]
        )
        passed, pending = overview(args, all_prompts, tokens)
        print(f"Summary: {args.out / 'summary.csv'} | "
              f"PASS={passed} PENDING={pending}", flush=True)
        if row["status"] == "INTERRUPTED":
            return 130
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as error:
        print(f"[SWEEP_ERROR] {error}", file=sys.stderr)
        sys.exit(1)
