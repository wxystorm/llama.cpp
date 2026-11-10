#!/usr/bin/env python3
"""Independent, resumable ShareGPT40 runner for the 32B Region Ring engine.

Passes are cached only for the same configuration, model/executable identity and
prompt contents. Every attempt has its own logs, telemetry and result.json.
Python 3.9+; Windows PowerShell users can call run_32b_all40.ps1.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

GROUPS = (128, 256, 512, 1024)
FIXED_LAYOUT = "G8,P6,G8,P6,G8,P6,G8,P6,G8"  # 5*G8 + 4*P6 = 64 layers
CHUNK_TOKENS = 128
FORCED_RING_ENV = {
    "LLAMA_HYBRID_REGION_RING_PLANNER": "apply-wide",
    "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE": "1",
    "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS": str(CHUNK_TOKENS),
    "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT": FIXED_LAYOUT,
    "LLAMA_HYBRID_REGION_RING_FORCE_XC": str(CHUNK_TOKENS),
    "LLAMA_HYBRID_REGION_RING_FORCE_XG": str(CHUNK_TOKENS),
    "LLAMA_HYBRID_REGION_RING_FORCE_XP": str(CHUNK_TOKENS),
    "LLAMA_HYBRID_REGION_RING_DUPLEX_PROFILE": "0",
    "LLAMA_REGION_RING_COALESCE_TINY_TAIL": "1",
}
PROTECTED_RING_ENV = set(FORCED_RING_ENV) | {
    "LLAMA_REGION_RING_METRICS_FILE",
    "LLAMA_HYBRID_REGION_RING_PLAN_TOKENS",
    "LLAMA_HYBRID_REGION_RING_V1",
    "LLAMA_HYBRID_REGION_RING_EXEC",
}
UPPER_BOUNDS = {128: 255, 256: 511, 512: 1023, 1024: 2000}
METRIC_RX = r"(\d+(?:\.\d+)?)"
RESULT_COLUMNS = (
    "prompt", "group", "prompt_tokens_hint", "context", "status", "attempt",
    "exit_code", "prefill_tokens", "prefill_ms", "prefill_tps",
    "decode_tokens", "decode_ms", "decode_tps", "wall_ms", "layout",
    "ring_complete", "error", "result_file",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def digest(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def file_id(path: Path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_manifest(folder: Path):
    path = folder / "manifest.csv"
    if not path.is_file():
        return {}
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            normalized = {re.sub(r"[^a-z0-9]", "", (key or "").lower()): val
                          for key, val in record.items()}
            name = Path(normalized.get("file", "")).name
            tokens = None
            for field in ("modeltokens", "tokenstokenizer", "tokenizerlength",
                          "tokenizerntokens", "tokenizerjstokens", "prompttokens", "ntokens", "tokens"):
                val = normalized.get(field, "")
                if val and val.strip().isdigit() and int(val) > 0:
                    tokens = int(val)
                    break
            result[name] = tokens
    return result


def prompts_in(folder: Path):
    expected = [(group, folder / f"prompt_{group}_{i:02}.txt")
                for group in GROUPS for i in range(1, 11)]
    missing = [path.name for _, path in expected if not path.is_file()]
    if missing:
        raise ValueError("ShareGPT40 requires all 40 prompts. Missing: " + ", ".join(missing))
    return expected


def parse_metrics(folder: Path, exit_code, wall_ms):
    perf = {}
    ring = []
    path = folder / "metrics.jsonl"
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                kind = entry.get("type") if isinstance(entry, dict) else None
                if kind == "perf":
                    perf = entry
                elif kind == "measurement_start":
                    ring.clear()  # discard warmup telemetry
                elif kind == "region_ring":
                    ring.append(entry)

    def tail_text(path: Path, limit=4 * 1024 * 1024):
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")

    logs = "\n".join(tail_text(folder / name)
                     for name in ("engine.stderr.log", "output.stdout.txt"))

    def last(pattern):
        found = re.findall(pattern, logs, re.IGNORECASE)
        return found[-1] if found else None

    pre = last(r"prompt eval time\s*=\s*" + METRIC_RX + r"\s*ms\s*/\s*(\d+)\s*tokens")
    dec = last(r"(?<!prompt )eval time\s*=\s*" + METRIC_RX + r"\s*ms\s*/\s*(\d+)\s*runs")
    pre_ms = perf.get("prefill_ms", float(pre[0]) if pre else None)
    pre_tokens = perf.get("prefill_tokens", int(pre[1]) if pre else None)
    dec_ms = perf.get("decode_ms", float(dec[0]) if dec else None)
    dec_tokens = perf.get("decode_tokens", int(dec[1]) if dec else None)

    def tps(native, tokens, milliseconds):
        if native is not None:
            return native
        return round(tokens * 1000 / milliseconds, 3) if tokens and milliseconds else None

    ring_complete = bool(ring) and all(
        bool(item.get("complete")) and item.get("status") == 0 for item in ring
    )
    if exit_code is None:
        status = "INTERRUPTED"
    elif exit_code == -9:
        status = "TIMEOUT"
    elif exit_code != 0:
        status = "FAIL"
    elif not perf and not ring:
        status = "NO_METRICS"
    elif not perf:
        status = "NO_PERF"
    elif not ring:
        status = "NO_RING"
    elif not pre_ms or not pre_tokens:
        status = "NO_PREFILL"
    elif not ring_complete:
        status = "INCOMPLETE_RING"
    elif any(item.get("layout") != FIXED_LAYOUT for item in ring):
        status = "WRONG_LAYOUT"
    else:
        status = "PASS"
    return {
        "status": status, "exit_code": exit_code, "wall_ms": round(wall_ms, 3),
        "prefill_tokens": pre_tokens, "prefill_ms": pre_ms,
        "prefill_tps": tps(perf.get("prefill_tps"), pre_tokens, pre_ms),
        "decode_tokens": dec_tokens, "decode_ms": dec_ms,
        "decode_tps": tps(perf.get("decode_tps"), dec_tokens, dec_ms),
        "ring_complete": ring_complete,
        "layout": ring[0].get("layout", "") if ring else "",
    }


def all_results(out: Path, prompts):
    output = {}
    for group, prompt in prompts:
        folder = out / "runs" / prompt.stem
        attempts = sorted(folder.glob("attempt-*/result.json"))
        if attempts:
            result_file = attempts[-1]
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
                result["result_file"] = str(result_file)
                if result.get("status") == "RUNNING":
                    result["status"] = "INTERRUPTED"  # interrupted process or power loss
            except (ValueError, OSError):
                result = {"status": "INTERRUPTED", "result_file": str(result_file)}
        else:
            result = {"status": "PENDING", "attempt": 0}
        result.setdefault("prompt", prompt.name)
        result.setdefault("group", group)
        output[prompt.name] = result
    return output


def write_summary(out: Path, prompts):
    latest = all_results(out, prompts)
    rows = [{column: latest[prompt.name].get(column, "") for column in RESULT_COLUMNS}
            for _, prompt in prompts]
    path = out / "summary.csv"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    counts = {name: sum(v["status"] == name for v in latest.values())
              for name in ("PASS", "PENDING", "FAIL", "NO_METRICS", "NO_PERF",
                           "NO_RING", "NO_PREFILL", "INCOMPLETE_RING", "WRONG_LAYOUT", "TIMEOUT", "INTERRUPTED")}
    save_json(out / "summary.json", {"updated_at": utc_now(), "total": len(prompts),
                                      "counts": counts, "summary_csv": str(path)})
    return counts


@contextlib.contextmanager
def single_runner(out: Path):
    lock_path = out / ".runner.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"x")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Output directory is already in use: {out}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def choose_context(group, hint, args):
    estimate = hint if hint is not None else UPPER_BOUNDS[group]
    context = max(args.min_context, ((estimate + args.n_predict + args.context_padding + 255) // 256) * 256)
    if context > args.max_context:
        raise ValueError(f"group {group}: context {context} exceeds --max-context {args.max_context}")
    return context, estimate


def build_process(args, prompt, ctx, estimate, metrics_path):
    env = os.environ.copy()
    # Never inherit a stale layout/chunk/profile policy from another experiment.
    for key in PROTECTED_RING_ENV:
        env.pop(key, None)
    env.pop("LLAMA_HYBRID_REGION_RING_LAYOUT", None)
    env.update({
        "LLAMA_HYBRID_REGION_RING_V1": "1",
        "LLAMA_HYBRID_REGION_RING_EXEC": "fifo-pc-phone",
        "LLAMA_HYBRID_REGION_RING_PLAN_TOKENS": str(estimate),
        "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM": "1" if args.streaming == "on" else "0",
        "LLAMA_REGION_RING_METRICS_FILE": str(metrics_path),
        **FORCED_RING_ENV,
    })
    # Additional diagnostics are permitted, but cannot undo the fixed topology.
    for entry in args.env:
        key, value = entry.split("=", 1)
        env[key] = value
    command = [str(args.exe), "-m", str(args.model), "--rpc", args.rpc,
               "--split-mode", "tensor", "--flash-attn", "on",
               "-c", str(ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
               "-n", str(args.n_predict), "-t", str(args.threads),
               "--seed", str(args.seed), "--no-conversation",
               "-f", str(prompt), "-lv", str(args.log_verbosity)]
    if args.no_mmap:
        command.append("--no-mmap")
    return command, env


def execute(args, prompt, group, context, estimate, out: Path):
    folder = out / "runs" / prompt.stem
    folder.mkdir(parents=True, exist_ok=True)
    attempt = max((int(p.name.split("-")[-1]) for p in folder.glob("attempt-[0-9]*")
                   if p.is_dir() and p.name.split("-")[-1].isdigit()), default=0) + 1
    trial = folder / f"attempt-{attempt:04}"
    trial.mkdir(exist_ok=False)
    command, env = build_process(args, prompt, context, estimate, trial / "metrics.jsonl")
    row = {"prompt": prompt.name, "group": group, "attempt": attempt,
           "context": context, "prompt_tokens_hint": estimate,
           "status": "RUNNING", "start_time": utc_now(), "command": command,
           "env_overrides": {key: val for key, val in env.items()
                             if key.startswith("LLAMA_HYBRID_REGION_RING_")
                             or key in ("LLAMA_REGION_RING_METRICS_FILE", "LLAMA_REGION_RING_COALESCE_TINY_TAIL")
                             or key in [e.split("=", 1)[0] for e in args.env]},
           "error": ""}
    result_file = trial / "result.json"
    save_json(result_file, row)
    start = time.monotonic()
    code = None
    try:
        with (trial / "output.stdout.txt").open("wb") as stdout, (trial / "engine.stderr.log").open("wb") as stderr:
            try:
                with subprocess.Popen(command, cwd=args.repo_root, env=env,
                                      stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr) as proc:
                    try:
                        proc.wait(timeout=args.timeout if args.timeout > 0 else None)
                        code = proc.returncode
                    except (KeyboardInterrupt, subprocess.TimeoutExpired):
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        if sys.exc_info()[0] is subprocess.TimeoutExpired:
                            code = -9
                        else:
                            raise
            except OSError as exc:
                row["error"] = f"LAUNCH_ERROR: {exc}"
                stderr.write((row["error"] + "\n").encode("utf-8"))
                code = 127
    finally:
        elapsed = (time.monotonic() - start) * 1000
        row.update(parse_metrics(trial, code, elapsed))
        row["end_time"] = utc_now()
        row["result_file"] = str(result_file)
        save_json(result_file, row)
    return row


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rpc", required=True)
    parser.add_argument("--prompt-dir", type=Path, default=Path(r"E:\llama\datasets\sharegpt\sharegpt40"))
    parser.add_argument("--out", type=Path, default=Path("32b-g8p6-x128-n1-all40"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--streaming", choices=("on", "off"), default="on")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=1, help="generated tokens per prompt (default 1 for prefill benchmark)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-verbosity", type=int, default=4)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--min-context", type=int, default=256, help="minimum Context; auto-sized per prompt and 256-aligned")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--context-padding", type=int, default=64, help="tokens reserved beyond prompt and generated token")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="extra retries per failed prompt in the current invocation")
    parser.add_argument("--retry-delay", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=0, help="seconds; 0 disables")
    parser.add_argument("--only-group", type=int, choices=GROUPS, action="append")
    parser.add_argument("--only-prompt", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="max selected prompts this invocation; 0=all")
    parser.add_argument("--rerun-all", action="store_true", help="rerun PASS too; preserve old attempts")
    parser.add_argument("--retry-failed-only", action="store_true", help="skip PASS and never-run prompts")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without touching results")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    for key in ("exe", "model", "prompt_dir", "out", "repo_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for label, path, kind in (("--exe", args.exe, "file"), ("--model", args.model, "file"),
                              ("--prompt-dir", args.prompt_dir, "dir"), ("--repo-root", args.repo_root, "dir")):
        if not (path.is_file() if kind == "file" else path.is_dir()):
            parser.error(f"{label} does not exist: {path}")
    if args.batch <= 0 or args.ubatch <= 0 or args.ubatch > args.batch:
        parser.error("require --batch >= --ubatch > 0")
    if args.n_predict <= 0 or args.min_context <= 0 or args.context_padding < 0 or args.max_context < args.min_context:
        parser.error("invalid prediction/context sizes")
    if args.max_retries < 0 or args.retry_delay < 0 or args.timeout < 0 or args.limit < 0:
        parser.error("invalid retry/timeout/limit")
    if not 0 <= args.log_verbosity <= 5:
        parser.error("--log-verbosity must be 0..5")
    if args.retry_failed_only and args.rerun_all:
        parser.error("--retry-failed-only conflicts with --rerun-all")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", e, re.DOTALL) for e in args.env):
        parser.error("--env must use NAME=VALUE")
    forbidden = sorted({entry.split("=", 1)[0] for entry in args.env} & PROTECTED_RING_ENV)
    if forbidden:
        parser.error("--env cannot override fixed Ring settings: " + ", ".join(forbidden))
    return args


def main(argv=None):
    args = get_args(argv)
    prompts = prompts_in(args.prompt_dir)
    manifest = read_manifest(args.prompt_dir)
    decisions = {}
    for group, prompt in prompts:
        context, estimate = choose_context(group, manifest.get(prompt.name), args)
        decisions[prompt.name] = {"context": context, "prompt_tokens_hint": estimate,
                                  "sha256": digest(prompt)}
    selections = [(g, p) for g, p in prompts
                  if (not args.only_group or g in args.only_group)
                  and (not args.only_prompt or p.name in args.only_prompt)]
    if args.only_prompt and set(args.only_prompt) - {p.name for _, p in prompts}:
        raise ValueError("Unknown --only-prompt file")
    if args.limit:
        selections = selections[:args.limit]
    if not selections:
        raise ValueError("No prompts selected")
    config = {"version": 2, "model": file_id(args.model), "exe": file_id(args.exe),
              "rpc": args.rpc, "prompt_dir": str(args.prompt_dir),
              "prompts": decisions, "planner_search": "SKIPPED_FORCED_LAYOUT",
              "streaming": args.streaming, "fixed_layout": FIXED_LAYOUT,
              "fixed_xc": CHUNK_TOKENS, "fixed_xg": CHUNK_TOKENS,
              "fixed_xp": CHUNK_TOKENS, "tiny_tail_coalesce": True,
              "batch": args.batch, "ubatch": args.ubatch, "n_predict": args.n_predict,
              "threads": args.threads, "seed": args.seed, "log_verbosity": args.log_verbosity,
              "no_mmap": args.no_mmap, "min_context": args.min_context,
              "max_context": args.max_context, "context_padding": args.context_padding,
              "env": sorted(args.env)}
    first_group, first_prompt = prompts[0]
    _, effective_env = build_process(args, first_prompt,
                                     decisions[first_prompt.name]["context"],
                                     decisions[first_prompt.name]["prompt_tokens_hint"],
                                     Path("telemetry-placeholder.jsonl"))
    config["effective_runtime_env"] = {
        key: value for key, value in sorted(effective_env.items())
        if key.startswith(("LLAMA_", "GGML_", "CUDA_"))
        and key != "LLAMA_REGION_RING_METRICS_FILE"
    }
    print(f"[FIXED_RING] layout={FIXED_LAYOUT} XC={CHUNK_TOKENS} "
          f"XG={CHUNK_TOKENS} XP={CHUNK_TOKENS} profile=SKIPPED "
          "tail_coalesce=1", flush=True)
    if args.dry_run:
        for group, prompt in selections:
            ctx = decisions[prompt.name]["context"]
            estimate = decisions[prompt.name]["prompt_tokens_hint"]
            command, _ = build_process(args, prompt, ctx, estimate, Path("metrics.jsonl"))
            print(f"[DRY_RUN] {prompt.name} group={group} tokens_hint={estimate} ctx={ctx}\n  " + subprocess.list2cmdline(command))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    with single_runner(args.out):
        config_file = args.out / "run-config.json"
        if config_file.is_file():
            old = json.loads(config_file.read_text(encoding="utf-8"))
            if old != config:
                differences = [key for key in sorted(set(old) | set(config)) if old.get(key) != config.get(key)]
                raise ValueError("Existing results use a different configuration (" + ", ".join(differences[:10]) + "). Use another --out directory.")
        else:
            if any((args.out / "runs").glob("*/attempt-*/result.json")):
                raise ValueError("Existing runs without run-config.json. Use another --out directory.")
            save_json(config_file, config)
        write_summary(args.out, prompts)
        total = len(selections)
        for i, (group, prompt) in enumerate(selections, 1):
            previous = all_results(args.out, prompts)[prompt.name]
            if previous["status"] == "PASS" and not args.rerun_all:
                print(f"[{i}/{total}] SKIP PASS {prompt.name}", flush=True)
                continue
            if args.retry_failed_only and previous["status"] == "PENDING":
                print(f"[{i}/{total}] SKIP PENDING {prompt.name}", flush=True)
                continue
            for retry in range(args.max_retries + 1):
                ctx = decisions[prompt.name]["context"]
                estimate = decisions[prompt.name]["prompt_tokens_hint"]
                print(f"[{i}/{total}] RUN {prompt.name} group={group} ctx={ctx} retry={retry}/{args.max_retries}", flush=True)
                try:
                    row = execute(args, prompt, group, ctx, estimate, args.out)
                except KeyboardInterrupt:
                    write_summary(args.out, prompts)
                    print("\nInterrupted; rerun the same command and --out to resume.", file=sys.stderr)
                    return 130
                counts = write_summary(args.out, prompts)
                print(f"  {row['status']}: prefill={row.get('prefill_tps')} tok/s "
                      f"({row.get('prefill_ms')} ms), decode={row.get('decode_tps')} tok/s "
                      f"attempt={row['attempt']}", flush=True)
                if row["status"] == "PASS":
                    break
                if retry < args.max_retries and args.retry_delay:
                    time.sleep(args.retry_delay)
            if row["status"] != "PASS" and args.stop_on_fail:
                break
        counts = write_summary(args.out, prompts)
        print(f"Summary: {args.out / 'summary.csv'} | "
              f"PASS={counts['PASS']} PENDING={counts['PENDING']} "
              f"FAILED={len(prompts)-counts['PASS']-counts['PENDING']}")
        return 0 if all(all_results(args.out, prompts)[p.name]["status"] == "PASS"
                        for _, p in selections) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ALL40_INTERRUPTED] Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[ALL40_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
