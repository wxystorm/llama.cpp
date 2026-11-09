#!/usr/bin/env python3
"""Qwen3-14B ShareGPT40 prefill-only Region Ring sweep with safe resume.

Defaults: G9,P1 x3 + G10, one predicted token, 128-token G/P Chunks, no
profiling. Each GGUF-tokenized prompt gets the smallest 256-aligned Context
that safely fits the prompt, one predicted token and a 64-token margin.
Independent XG/XP and per-group/per-prompt overrides remain available for
imbalanced pipelines. Old attempts are archived on retry.
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
MIN_CONTEXT = 1024
CONTEXT_PADDING = 64
CONTEXT_ALIGN = 256
PREDICT = 1
THREADS = 8
SEED = 1234
FIELDS = (
    "prompt_file", "group", "attempt", "status", "actual_prefill_tokens",
    "prefill_ms", "prefill_tps", "actual_decode_tokens", "decode_ms",
    "decode_tps", "context", "plan_tokens", "xc", "xg", "xp", "layout_fixed",
    "cuda_kv_mib", "gpu_bank_a_mib", "gpu_bank_b_mib",
    "gpu_banks_plus_kv_mib", "predicted_dag_ms",
    "exit_code", "commit", "run_dir",
)

# Explicit PC configuration to avoid inheriting an older run's layout/mode.
RING_ENV = {
    "LLAMA_HYBRID_REGION_RING_V1": "1",
    "LLAMA_HYBRID_REGION_RING_EXEC": "fifo-pc-phone",
    "LLAMA_HYBRID_REGION_RING_PLANNER": "apply-wide",
    "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE": "1",
    "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS": str(CHUNK),
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




def context_for_tokens(token_count: int, predict: int = PREDICT,
                       minimum: int = MIN_CONTEXT,
                       padding: int = CONTEXT_PADDING,
                       fixed: int | None = None) -> int:
    """Minimum safe Context, aligned to 256 tokens for the GGML KV allocator.

    Reserve for the one prediction plus a small tokenizer/runtime margin.
    Never silently truncate prompts to fit an explicitly undersized context.
    """
    if token_count <= 0 or predict < 1 or padding < 0 or minimum <= 0:
        raise ValueError("invalid token count or context parameters")
    required = token_count + predict + padding
    if fixed is not None:
        if fixed < required:
            raise ValueError(
                f"fixed context {fixed} cannot fit {token_count} prompt + "
                f"{predict} predicted + {padding} safety tokens "
                f"(requires >= {required})")
        return fixed
    return max(minimum, ((required + CONTEXT_ALIGN - 1) // CONTEXT_ALIGN)
               * CONTEXT_ALIGN)


def validate_layout(layout: str, expected_layers: int = 40) -> str:
    """A PC GPU-first, GPU-last Region Ring with the expected model depth."""
    parts = layout.split(",")
    matched = [re.fullmatch(r"([GP])([1-9][0-9]*)", p) for p in parts]
    if (len(parts) < 3 or len(parts) % 2 == 0 or
            any(m is None for m in matched) or
            sum(int(m.group(2)) for m in matched) != expected_layers or
            any(m.group(1) != ("G" if i % 2 == 0 else "P")
                for i, m in enumerate(matched))):
        raise ValueError(
            f"invalid layout {layout!r}: expected alternating G/P, "
            f"starting and ending in G, with exactly {expected_layers} layers")
    return layout


def load_chunk_overrides(path: Path | None) -> dict:
    """Optional JSON: {"groups":{"1024":{"xp":96}},
                           "prompts":{"prompt_1024_10":{"xg":112}}}."""
    if path is None:
        return {"groups": {}, "prompts": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - {"groups", "prompts"}:
        raise ValueError("chunk override JSON needs groups/prompts objects only")
    out = {"groups": {}, "prompts": {}}
    for kind in out:
        entries = value.get(kind, {})
        if not isinstance(entries, dict):
            raise ValueError(f"chunk overrides {kind} must be an object")
        for key, widths in entries.items():
            if kind == "groups":
                if str(key) not in ("128", "256", "512", "1024"):
                    raise ValueError(f"unknown ShareGPT group: {key}")
                name = str(key)
            else:
                name = prompt_id(str(key))
            if (not isinstance(widths, dict) or not widths or
                    set(widths) - {"xc", "xg", "xp"}):
                raise ValueError(f"invalid chunk settings for {key}")
            for field, width in widths.items():
                if type(width) is not int or not 1 <= width <= BATCH:
                    raise ValueError(f"{key}: {field} must be an integer in [1,{BATCH}]")
            if name in out[kind]:
                raise ValueError(f"duplicate chunk override: {name}")
            out[kind][name] = dict(widths)
    return out


def fixed_plan(args: argparse.Namespace, group: int, prompt: Path,
               token_count: int, overrides: dict) -> dict:
    widths = {"xc": args.xc, "xg": args.xg, "xp": args.xp}
    widths.update(overrides["groups"].get(str(group), {}))
    widths.update(overrides["prompts"].get(prompt.name.lower(), {}))
    return {
        **widths, "context": context_for_tokens(
            token_count, minimum=args.min_context,
            padding=args.context_padding, fixed=args.context),
        "plan_tokens": token_count,
        "layout_fixed": args.layout,
        "token_source": ("model_tokenizer" if args.token_source == "model"
                         else "manifest"),
        "model": "fixed_manual_no_profile",
        "predicted_dag_ms": None,
    }

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



def gpu_memory_from_log(folder: Path) -> dict:
    """Record physical CUDA KV and reserved weight Bank peak from engine logs.

    This is a lower bound on VRAM demand: graph buffers, CUDA workspace,
    runtime allocations and WDDM's real residency budget are not included.
    """
    path = folder / "engine.stderr.log"
    if not path.is_file():
        return {}
    log = path.read_text(encoding="utf-8", errors="replace")
    kv_values = re.findall(
        r"CUDA0 KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", log)
    peaks = re.findall(
        r"\[REGION_RING_GPU_BANK_PEAK_SUM\][^\r\n]*", log)
    peak = next((line for line in reversed(peaks) if "status=READY" in line),
                peaks[-1] if peaks else "")
    def value(key: str) -> float | None:
        found = re.search(r"(?:^|\s)" + re.escape(key) +
                          r"=([0-9]+(?:\.[0-9]+)?)", peak)
        return float(found.group(1)) if found else None
    a = value("capacity_a_mib")
    b = value("capacity_b_mib")
    kv = float(kv_values[-1]) if kv_values else None
    total = round(a + b + kv, 3) if None not in (a, b, kv) else None
    return {
        "cuda_kv_mib": kv, "gpu_bank_a_mib": a,
        "gpu_bank_b_mib": b, "gpu_banks_plus_kv_mib": total,
    }


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
    token_count = (plan or {}).get("plan_tokens",
                    getattr(args, "token_counts", {}).get(prompt.name.lower(), group))
    context = (plan or {}).get("context") or context_for_tokens(
        token_count, minimum=getattr(args, "min_context", MIN_CONTEXT),
        padding=getattr(args, "context_padding", CONTEXT_PADDING),
        fixed=getattr(args, "context", None))
    env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"] = getattr(args, "layout", LAYOUT)
    for key in ("xc", "xg", "xp"):
        env["LLAMA_HYBRID_REGION_RING_FORCE_" + key.upper()] = str(
            (plan or {}).get(key, getattr(args, key, CHUNK)))
    env["LLAMA_HYBRID_REGION_RING_PLAN_TOKENS"] = str(token_count)
    env["LLAMA_REGION_RING_METRICS_FILE"] = str(folder / "metrics.jsonl")
    command = [
        str(args.exe), "-m", str(args.model), "--rpc", args.rpc,
        "--split-mode", "tensor", "--flash-attn", "on",
        "-c", str(context), "-b", str(BATCH), "-ub", str(UBATCH),
        "-n", str(PREDICT), "-t", str(THREADS), "--seed", str(SEED),
        "--no-conversation", "-f", str(prompt), "-lv", "4",
    ]
    if not getattr(args, "mmap", False):
        command.append("--no-mmap")
    if plan is not None:
        atomic_json(folder / "plan.json", plan)
    atomic_json(folder / "config.json", {
        "command": command, "cwd": str(args.root), "commit": commit,
        "prompt_tokens": token_count, "context": context,
        "layout": env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"],
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
                                streaming="on",
                                no_mmap=not getattr(args, "mmap", False))
    row = result_row(base_args, folder, prompt, token_count, context,
                     code, elapsed, commit)
    row.update(gpu_memory_from_log(folder))
    row.update(attempt=attempt, group=group, context=context,
               plan_tokens=token_count,
               layout_fixed=env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"],
               xc=(plan or {}).get("xc", getattr(args, "xc", CHUNK)),
               xg=(plan or {}).get("xg", getattr(args, "xg", CHUNK)),
               xp=(plan or {}).get("xp", getattr(args, "xp", CHUNK)),
               predicted_dag_ms=(plan or {}).get("predicted_dag_ms"),
               finished_utc=datetime.now(timezone.utc).isoformat())
    if interrupted:
        row["status"] = "INTERRUPTED"
    write_phone_input_push_timeline(folder)
    write_phone_copy_timeline(folder)
    atomic_json(folder / "result.json", row)
    return row, interrupted


def model_token_counts(args: argparse.Namespace, prompts: list[tuple[int, Path]],
                       persist: bool = True) -> dict[str, int]:
    """Count each prompt with the *selected GGUF* tokenizer; cache by model and prompt hash."""
    tokenizer = args.tokenizer_exe
    if not tokenizer.is_file():
        raise FileNotFoundError(
            f"GGUF tokenizer binary not found: {tokenizer}. "
            "Build llama-tokenize, pass --tokenizer-exe, or use --token-source manifest.")
    model_info = args.model.stat()
    tokenizer_info = tokenizer.stat()
    signature = {
        "model": str(args.model), "model_bytes": model_info.st_size,
        "model_mtime_ns": model_info.st_mtime_ns,
        "tokenizer": str(tokenizer), "tokenizer_bytes": tokenizer_info.st_size,
        "tokenizer_mtime_ns": tokenizer_info.st_mtime_ns,
        "flags": ["--ids", "--show-count", "--no-escape", "--log-disable"],
    }
    cache_file = args.out / "prompt-tokens.json"
    cache = read_json(cache_file)
    items = cache.get("prompts", {}) if cache.get("signature") == signature else {}
    if args.update_tokens:
        items = {}
    counts = {}
    for _, prompt in prompts:
        key = prompt.name.lower()
        digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
        previous = items.get(key, {})
        count = previous.get("tokens") if previous.get("prompt_sha256") == digest else None
        if not isinstance(count, int) or count <= 0:
            command = [
                str(tokenizer), "-m", str(args.model), "-f", str(prompt),
                "--ids", "--show-count", "--no-escape", "--log-disable",
            ]
            proc = subprocess.run(command, cwd=args.root,
                                  stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=120, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"llama-tokenize failed for {prompt.name} (code {proc.returncode}): "
                    + proc.stderr[-700:])
            matches = re.findall(r"Total number of tokens:\s*(\d+)", proc.stdout)
            if len(matches) != 1:
                raise RuntimeError(
                    f"llama-tokenize missing unique count for {prompt.name}: "
                    + proc.stdout[-250:])
            count = int(matches[0])
            if count <= 0:
                raise RuntimeError(f"llama-tokenize returned {count} tokens for {prompt.name}")
            print(f"[TOKENS] {prompt.name}: model={count}", flush=True)
        counts[key] = count
        items[key] = {"prompt_sha256": digest, "tokens": count}
        # Persist after each prompt, but keep --dry-run strictly read-only.
        if persist:
            atomic_json(cache_file, {"signature": signature, "prompts": items})
    return counts


def ensure_chunk_profile(args: argparse.Namespace, prompts: list[tuple[int, Path]],
                         commit: str) -> dict:
    """Run the full fixed-layout profile *once*, cache verified real traces."""
    # Each GRID width gets one real, fixed-layout measurement. With a cached
    # profile the widened search adds only missing widths; old PASS probes
    # remain untouched unless --update-profile is explicitly requested.
    candidates = tuple(chunk_dag.GRID)
    representative = next(((g, p) for g, p in prompts if g == 512), None)
    if representative is None:
        representative = min(prompts, key=lambda pair: abs(pair[0] - 512))
    group, prompt = representative
    profile_dir = args.out / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "chunk-profile.json"
    token_count = getattr(args, "token_counts", {}).get(prompt.name.lower(), group)
    profile_context = context_for_tokens(
        token_count, minimum=getattr(args, "min_context", MIN_CONTEXT),
        padding=getattr(args, "context_padding", CONTEXT_PADDING),
        fixed=getattr(args, "context", None))
    signature = dict(schema=2, model=str(args.model),
                     model_size=args.model.stat().st_size,
                     executable=str(args.exe), rpc=args.rpc,
                     prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
                     prompt=prompt.name, context=profile_context,
                     predict=PREDICT, layout=chunk_dag.LAYOUT,
                     grid=list(candidates))
    existing = read_json(profile_path)
    print(f"[PROFILE] chunk_grid={list(candidates)} "
          f"measurements={len(candidates)} dag_combinations={len(candidates)**2}",
          flush=True)
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
            probe_plan = {"xc": getattr(args, "xc", CHUNK), "xg": x, "xp": x,
                          "plan_tokens": token_count, "context": profile_context,
                          "profile_probe": True}
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


def main(argv: list[str] | None = None, *,
         description: str | None = None,
         defaults: dict | None = None) -> int:
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description=description or __doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", default=defaults.get("layout", LAYOUT))
    ap.add_argument("--model-layers", type=int,
                    default=defaults.get("model_layers", 40),
                    help="required Transformer depth for forced G/P layout (default 40)")
    mmap_flags = ap.add_mutually_exclusive_group()
    mmap_flags.add_argument("--mmap", dest="mmap", action="store_true",
                            help="enable memory-mapped model weights (recommended for 32B)")
    mmap_flags.add_argument("--no-mmap", dest="mmap", action="store_false",
                            help="load model weights into memory (14B default)")
    ap.set_defaults(mmap=bool(defaults.get("mmap", False)))
    ap.add_argument("--context", type=int, default=None,
                    help="optional fixed Context; default is minimum safe 256-aligned per Prompt")
    ap.add_argument("--min-context", type=int, default=MIN_CONTEXT,
                    help="minimum Context for auto-sizing (default: 1024)")
    ap.add_argument("--context-padding", type=int, default=CONTEXT_PADDING,
                    help="reserved tokens beyond Prompt + one predicted token (default: 64)")
    ap.add_argument("--xc", type=int, default=CHUNK)
    ap.add_argument("--xg", type=int, default=CHUNK)
    ap.add_argument("--xp", type=int, default=CHUNK)
    ap.add_argument("--chunk-overrides", type=Path, default=None,
                    help="optional JSON overrides per group and/or per Prompt")
    ap.add_argument("--fixed-chunks", action="store_true",
                    help="legacy alias; fixed XG/XP=128 is already the default")
    ap.add_argument("--profile-chunks", action="store_true",
                    help="opt-in legacy G9/P1 Chunk DAG calibration, not used by default")
    ap.add_argument("--token-source", choices=("model", "manifest"), default="model",
                    help="GGUF tokenizer (preferred), or manifest.csv counts")
    ap.add_argument("--tokenizer-exe", type=Path, default=None,
                    help="llama-tokenize binary; defaults to build-tensor-v2/bin/Release")
    ap.add_argument("--update-tokens", action="store_true",
                    help="force refresh of the model Token count cache")
    ap.add_argument("--update-profile", action="store_true",
                    help="rerun opt-in legacy Chunk calibration")
    ap.add_argument("--only", action="append", default=[], metavar="PROMPT",
                    help="run only named prompts (unless already PASS)")
    ap.add_argument("--rerun", action="append", default=[], metavar="PROMPT",
                    help="force rerun named prompts, even if PASS")
    ap.add_argument("--rerun-below", type=float, default=0,
                    help="rerun PASS prompts with prefill tok/s below threshold")
    ap.add_argument("--limit", type=int, default=0, help="maximum processes this invocation")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="pause between processes in seconds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.root = Path(__file__).resolve().parents[2]
    for key in ("exe", "model", "prompt_dir", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.chunk_overrides is not None:
        args.chunk_overrides = args.chunk_overrides.expanduser().resolve()
    if args.tokenizer_exe is None:
        args.tokenizer_exe = args.exe.with_name(
            "llama-tokenize.exe" if os.name == "nt" else "llama-tokenize")
    args.tokenizer_exe = args.tokenizer_exe.expanduser().resolve()
    if not args.exe.is_file() or not args.model.is_file() or not args.prompt_dir.is_dir():
        ap.error("--exe, --model and --prompt-dir must exist")
    if args.limit < 0 or args.delay < 0 or args.rerun_below < 0:
        ap.error("limit, delay and rerun-below must be >=0")
    if not 1 <= args.model_layers <= 256:
        ap.error("model-layers must be in [1,256]")
    if args.min_context <= 0 or args.context_padding < 0 or (
            args.context is not None and args.context <= 0):
        ap.error("context must be positive and context-padding nonnegative")
    if any(type(x) is not int or x < 1 or x > BATCH for x in
           (args.xc, args.xg, args.xp)):
        ap.error("XC/XG/XP must be in [1, 2048]")
    try:
        args.layout = validate_layout(args.layout, args.model_layers)
        overrides = load_chunk_overrides(args.chunk_overrides)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    if args.profile_chunks and (
            args.fixed_chunks or args.layout != chunk_dag.LAYOUT or
            overrides != {"groups": {}, "prompts": {}}):
        ap.error("--profile-chunks supports only legacy G9/P1 and no manual overrides")
    if args.update_profile and not args.profile_chunks:
        ap.error("--update-profile requires --profile-chunks")

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
        "schema": 2, "exe": str(args.exe), "model": str(args.model),
        "model_bytes": args.model.stat().st_size, "rpc": args.rpc,
        "prompt_dir": str(args.prompt_dir), "layout": args.layout,
        "context_policy": {
            "fixed": args.context, "minimum": args.min_context,
            "padding": args.context_padding, "alignment": CONTEXT_ALIGN,
        },
        "batch": BATCH, "ubatch": UBATCH, "n_predict": PREDICT,
        "threads": THREADS, "seed": SEED, "no_mmap": not args.mmap,
        "ring_environment": {
            **RING_ENV, "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT": args.layout,
        },
        "chunk_planning": "PROFILE_DAG_LEGACY" if args.profile_chunks
                          else "FIXED_MANUAL_NO_PROFILE",
        "xc": args.xc, "xg": args.xg, "xp": args.xp,
        "chunk_overrides": overrides,
        "token_source": args.token_source,
        "tokenizer_exe": str(args.tokenizer_exe) if args.token_source == "model" else None,
    }
    # Preserve exact 14B run configuration for restart compatibility.
    if args.model_layers != 40:
        configuration["model_layers"] = args.model_layers
    config_path = args.out / "run-config.json"
    existing_config = read_json(config_path) if config_path.exists() else {}
    if config_path.exists() and existing_config != configuration:
        ap.error(f"run configuration differs: {config_path}; use a different --out")
    if args.out.exists() and not config_path.exists() and any(args.out.iterdir()):
        ap.error(f"output folder is not an initialized sweep: {args.out}")

    commit = git_head(args.root)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            atomic_json(config_path, configuration)

    try:
        if args.token_source == "model":
            token_counts = model_token_counts(args, prompts, persist=not args.dry_run)
        else:
            token_counts = {
                p.name.lower(): chunk_dag.manifest_tokens(
                    args.prompt_dir / "manifest.csv", p.name)
                for _, p in prompts
            }
        args.token_counts = token_counts
        for group, prompt in prompts:
            if prompt.name.lower() not in selected:
                continue
            context_for_tokens(
                token_counts[prompt.name.lower()],
                minimum=args.min_context, padding=args.context_padding,
                fixed=args.context)
        print(f"[TOKENS] source={args.token_source} count={len(token_counts)}",
              flush=True)
    except (ValueError, OSError, RuntimeError) as exc:
        ap.error(str(exc))

    profile = None
    if args.profile_chunks and not args.dry_run:
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
            slow = args.rerun_below > 0 and (
                float(previous.get("prefill_tps") or 0) < args.rerun_below)
        except (ValueError, TypeError):
            slow = True
        force = prompt.name.lower() in rerun
        plan_tokens = token_counts[prompt.name.lower()]
        plan = fixed_plan(args, group, prompt, plan_tokens, overrides)
        if profile is not None:
            selected_plan = chunk_dag.choose(profile, plan_tokens)
            plan.update(selected_plan)
            plan["xc"] = args.xc
            plan["layout_fixed"] = args.layout
            plan["token_source"] = ("model_tokenizer"
                                    if args.token_source == "model" else "manifest")
        old_plan = read_json(folder / "plan.json") if is_pass else {}
        needs_replan = is_pass and any(
            old_plan.get(key) != plan.get(key)
            for key in ("context", "plan_tokens", "xc", "xg", "xp",
                        "layout_fixed", "token_source"))
        if is_pass and not force and not slow and not needs_replan:
            if old_plan != plan and not args.dry_run:
                atomic_json(folder / "plan.json", plan)
                print(f"[PLAN_REFRESH] {prompt.name}: same Context/Chunks",
                      flush=True)
            print(f"[SKIP] {prompt.name}: PASS {previous.get('prefill_tps')} tok/s",
                  flush=True)
            continue
        if args.limit and launched >= args.limit:
            break
        reason = ("force" if force else
                  "context_or_chunks_changed" if needs_replan else
                  "slow" if is_pass and slow else
                  "retry" if folder.exists() else "new")
        print(f"[PLAN] {prompt.name}: prompt_tokens={plan_tokens} "
              f"context={plan['context']} layout={args.layout} "
              f"XC={plan['xc']} XG={plan['xg']} XP={plan['xp']} "
              f"profile={'legacy' if profile is not None else 'SKIPPED'} "
              f"predict={PREDICT}", flush=True)
        if args.dry_run:
            print(f"[DRY] {prompt.name}: {reason}", flush=True)
            launched += 1
            continue
        attempt = archive(args.out, folder, previous) if folder.exists() else 1
        print(f"[{launched + 1}] RUN {prompt.name} group={group} "
              f"attempt={attempt} reason={reason}", flush=True)
        row, interrupted = run_prompt(args, args.out, group, prompt,
                                      attempt, commit, plan)
        if row.get("actual_prefill_tokens") is not None:
            if int(row["actual_prefill_tokens"]) != plan_tokens:
                print(f"  [TOKEN_MISMATCH] {prompt.name}: "
                      f"model_tokenizer={plan_tokens} "
                      f"actual={row['actual_prefill_tokens']}",
                      file=sys.stderr, flush=True)
        launched += 1
        summary(args.out, prompts)
        print(f"  {row['status']}: prefill={row['prefill_tps']} tok/s "
              f"({row['prefill_ms']} ms), decode={row['decode_tps']} tok/s",
              flush=True)
        if row.get("gpu_banks_plus_kv_mib") is not None:
            print(f"  [GPU_MEMORY] BankA={row['gpu_bank_a_mib']:.1f} MiB "
                  f"BankB={row['gpu_bank_b_mib']:.1f} MiB "
                  f"CUDA_KV={row['cuda_kv_mib']:.1f} MiB "
                  f"subtotal={row['gpu_banks_plus_kv_mib']:.1f} MiB "
                  "(excludes workspace/other CUDA/WDDM)", flush=True)
        if interrupted:
            stopped = True
            break
        if args.delay and index < len(prompts) - 1:
            time.sleep(args.delay)
    if args.dry_run:
        return 0
    counts = summary(args.out, prompts)
    print(f"Summary: {args.out / 'summary.csv'} | " +
          " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
          flush=True)
    if stopped:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 1 if counts.get("FAIL", 0) or counts.get("INTERRUPTED", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
