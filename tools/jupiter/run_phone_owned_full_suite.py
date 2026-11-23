#!/usr/bin/env python3
"""One-command Qwen3-14B Phone-owned Prefill suite over all ShareGPT prompts.

Per Prompt: exact chat-templated Token count -> valid Chunk/KV plan ->
optional same-shape Graph warmup -> Sync/Phone-autonomous native A/B ->
complete F32 logits parity -> atomic checkpoint and grouped CSV summaries.

All runs are strictly sequential: the Phone RPC server owns one native HTP
execution at a time. Decode is deliberately not exercised.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from inspect_phone_owned_overlap import prepare_cost
from run_phone_owned_full_prefill import logits_comparison, read_logits
from run_phone_owned_multi_prompt import count_model_tokens

PROMPT_NAME = re.compile(r"^prompt_(.+)_(\d+)$", re.IGNORECASE)
CHUNK_BUCKETS = (32, 64, 88, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
FIELDNAMES = (
    "layout", "group", "prompt", "kind", "repeat", "status", "tokens",
    "context", "chunks", "chunk_size", "n_chunks", "first_mode",
    "sync_ms", "autonomous_ms", "sync_tok_s", "autonomous_tok_s",
    "speedup", "sync_cache_hit", "autonomous_cache_hit",
    "warm_comparable", "top1_same", "max_abs", "relative_rmse",
    "cosine", "revisit_parity", "error", "case_dir",
)


def group_key(name: str) -> tuple:
    m = PROMPT_NAME.fullmatch(Path(name).stem)
    if not m:
        return (10**9, name, 10**9, name.lower())
    group, index = m.groups()
    number = re.match(r"\d+", group)
    return (int(number.group()) if number else 10**9,
            group, int(index), name.lower())


def group_of(path: Path) -> str:
    m = PROMPT_NAME.fullmatch(path.stem)
    if not m:
        raise ValueError(f"Unexpected Prompt filename: {path.name}")
    return m.group(1)


def choose_chunks(tokens: int, preferred: int = 88,
                  max_chunks: int = 16) -> list[int]:
    if not 2 <= tokens <= 4096 or preferred <= 0 or not 2 <= max_chunks <= 16:
        raise ValueError("Native supports 2..4096 tokens and 2..16 Chunks")
    minimum = math.ceil(tokens / max_chunks)
    target = max(preferred, minimum)
    if target > preferred:
        target = next((x for x in CHUNK_BUCKETS if x >= target), target)
    # Single-Chunk autonomous Prefill would not exercise the task pipeline.
    if target >= tokens:
        target = (tokens + 1) // 2
    plan = [target] * (tokens // target)
    if tokens % target:
        plan.append(tokens % target)
    if not 2 <= len(plan) <= max_chunks or sum(plan) != tokens:
        raise ValueError(f"Cannot fit {tokens} tokens in {max_chunks} Chunks")
    return plan


def choose_context(tokens: int, floor: int = 512, cap: int = 4096) -> int:
    if tokens < 2 or cap < 256 or cap % 256 or floor < 256 or floor % 256:
        raise ValueError("Context must be >=256 and a multiple of 256")
    ctx = max(floor, math.ceil(tokens / 256) * 256)
    if ctx > cap:
        raise ValueError(
            f"Prompt requires KV context {ctx}, exceeding configured cap {cap}")
    return ctx


def file_signature(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path.resolve()), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def digest_prompt(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def next_attempt(case_dir: Path) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(x.name.removeprefix("attempt-"))
               for x in case_dir.glob("attempt-*")
               if x.is_dir() and x.name.removeprefix("attempt-").isdigit()]
    path = case_dir / f"attempt-{max(numbers, default=0)+1:03d}"
    path.mkdir()
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, help="rebuilt llama-cli.exe")
    p.add_argument("--tokenizer", required=True, help="matching llama-tokenize.exe")
    p.add_argument("--model", required=True, help="Qwen3-14B GGUF")
    p.add_argument("--rpc", required=True, help="running Phone host:port")
    p.add_argument("--prompts-dir", required=True,
                   help="directory containing prompt_0_01.txt, etc.")
    p.add_argument("--pattern", default="prompt_*_*.txt")
    p.add_argument("--out-dir", default="jupiter-autonomous-full-suite")
    p.add_argument("--layouts", default="32",
                   help="comma-separated first Phone layers, e.g. 32,34,36,39")
    p.add_argument("--cpu-layers", type=int, default=16)
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--chunk-size", type=int, default=88,
                   help="preferred Chunk; increases to keep <=16 Chunks")
    p.add_argument("--context-floor", type=int, default=512)
    p.add_argument("--context-cap", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=1,
                   help="complete Sync/Auto A/B pairs per Prompt")
    p.add_argument("--expected-count", type=int, default=0,
                   help="fail preflight if not exactly this many unique Prompts")
    p.add_argument("--mode-order", choices=("alternate", "sync-first",
                                            "autonomous-first"),
                   default="alternate")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip per-pair Sync Graph warmup; may bias measurements")
    p.add_argument("--revisit-first", action="store_true",
                   help="insert A-B-A with two distinct shapes in each group")
    p.add_argument("--resume", action="store_true",
                   help="skip only matching completed PASS cases")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop at first failed request instead of collecting failures")
    p.add_argument("--timeout-seconds", type=int, default=1800,
                   help="per child run; kill the whole local subprocess tree")
    p.add_argument("--dry-run", action="store_true",
                   help="plan/tokenize only; does not contact Phone RPC")
    return p.parse_args()


def discover(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        (p.resolve() for p in root.glob(pattern) if p.is_file()),
        key=lambda p: group_key(p.name))
    if not files:
        raise ValueError(f"No Prompt files matching {pattern!r} in {root}")
    if len({p.name.lower() for p in files}) != len(files):
        raise ValueError("Duplicate Prompt filenames")
    for path in files:
        group_of(path)
    return files


def preflight(args: argparse.Namespace, files: list[Path],
              layouts: list[int], output: Path) -> dict:
    cache_file = output / "token-cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) \
        if cache_file.exists() else {}
    model_id = file_signature(Path(args.model))
    tokenizer_id = file_signature(Path(args.tokenizer))
    tokens = {}
    groups = defaultdict(list)
    for index, prompt_file in enumerate(files, 1):
        content = prompt_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
        if not content:
            raise ValueError(f"Empty Prompt: {prompt_file}")
        sha = digest_prompt(prompt_file)
        cache_key = hashlib.sha256(json.dumps(
            [str(model_id), str(tokenizer_id), sha],
            sort_keys=True).encode()).hexdigest()
        if cache_key not in cache:
            cache[cache_key] = count_model_tokens(
                Path(args.tokenizer), Path(args.model), content)
            write_json(cache_file, cache)
        n_tokens = int(cache[cache_key])
        ctx = choose_context(
            n_tokens, args.context_floor, args.context_cap)
        chunks = choose_chunks(n_tokens, args.chunk_size)
        row = {"name": prompt_file.name, "path": str(prompt_file),
               "sha256": sha, "group": group_of(prompt_file),
               "tokens": n_tokens, "context": ctx, "chunks": chunks}
        tokens[prompt_file] = row
        groups[row["group"]].append(row)
        print(f"[PHONE_SUITE_PLAN] {index}/{len(files)} "
              f"group={row['group']} prompt={prompt_file.name}"
              f" tokens={n_tokens} ctx={ctx} chunks={chunks}", flush=True)
    config = {
        "format": 1,
        "exe": file_signature(Path(args.exe)),
        "tokenizer": tokenizer_id,
        "model": model_id,
        "rpc": args.rpc, "prompt_root": str(Path(args.prompts_dir).resolve()),
        "pattern": args.pattern, "layouts": layouts,
        "cpu_layers": args.cpu_layers, "sessions": args.sessions,
        "chunk_size": args.chunk_size,
        "context_floor": args.context_floor, "context_cap": args.context_cap,
        "repeats": args.repeats,
        "mode_order": args.mode_order, "warmup": not args.no_warmup,
        "revisit_first": args.revisit_first,
        "prompts": list(tokens.values()),
    }
    signature = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = {"sha256": signature, "config": config}
    manifest_path = output / "suite-manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("sha256") != signature:
            raise ValueError("Existing output uses different files/options. "
                             "Choose a new --out-dir, not --resume.")
        if not args.resume and any(output.glob("cases/**/case-result.json")):
            raise ValueError("This output already has results; pass --resume "
                             "or choose a new --out-dir.")
    write_json(manifest_path, manifest)
    return {"groups": groups, "tokens": tokens, "signature": signature}


def sequence_for_group(items: list[dict], revisit: bool) -> list[dict]:
    sequence = [dict(x, kind="normal") for x in items]
    if not revisit or len(items) < 2:
        return sequence
    a = items[0]
    b = next((x for x in items[1:]
              if (x["tokens"], x["context"], x["chunks"]) !=
                 (a["tokens"], a["context"], a["chunks"])), None)
    if b is None:
        print(f"[PHONE_SUITE] group={a['group']} no distinct B shape;"
              " A-B-A skipped", flush=True)
        return sequence
    others = [x for x in items[1:] if x["path"] != b["path"]]
    return [dict(a, kind="normal"), dict(b, kind="normal"),
            dict(a, kind="revisit")] + [
                dict(x, kind="normal") for x in others]


def build_cases(plan: dict, layouts: list[int], repeats: int,
                revisit: bool) -> list[dict]:
    cases = []
    for layout in layouts:
        for group, prompts in plan["groups"].items():
            for entry in sequence_for_group(prompts, revisit):
                for repeat in range(1, repeats + 1):
                    case = dict(entry)
                    case.update({"layout": layout, "repeat": repeat})
                    cases.append(case)
    return cases


def case_directory(output: Path, case: dict) -> Path:
    kind = "revisit-" if case["kind"] == "revisit" else ""
    return (output / "cases" / f"l{case['layout']:02d}" /
            f"g{case['group']}" / f"{kind}{Path(case['path']).stem}" /
            f"r{case['repeat']:02d}")


def cli_for_case(args: argparse.Namespace, case: dict,
                 mode: str, work: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_phone_owned_full_prefill.py")),
        "--exe", str(Path(args.exe).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--rpc", args.rpc, "--prompt-file", case["path"],
        "--sessions", str(args.sessions),
        "--pc-layers", str(case["layout"]),
        "--cpu-layers", str(args.cpu_layers),
        "--context", str(case["context"]),
        "--target-tokens", str(case["tokens"]),
        "--chunks", ",".join(map(str, case["chunks"])),
        "--mode", "native", "--out-dir", str(work),
    ]
    if mode == "autonomous":
        command.append("--autonomous-phone")
    return command


def child_run(command: list[str], stdout_path: Path, timeout: int) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stream:
        options = {"stdin": subprocess.DEVNULL, "stdout": stream,
                   "stderr": subprocess.STDOUT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        child = subprocess.Popen(command, **options)
        try:
            code = child.wait(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired as exc:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(child.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False)
            else:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise RuntimeError(
                f"Timeout after {timeout}s; local child tree killed. "
                "Check Phone active-run recovery before continuing.") from exc
    if code:
        tail = "\n".join(
            stdout_path.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-28:])
        raise RuntimeError(f"Runner failed exit={code}; {stdout_path}\n{tail}")


def run_mode(args: argparse.Namespace, case: dict,
             mode: str, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    child_run(cli_for_case(args, case, mode, work),
              work / "runner.stdout.log", args.timeout_seconds)
    report = json.loads(
        (work / "results.json").read_text(encoding="utf-8"))["native"]
    log_file = work / "native-full-prefill.log"
    log = log_file.read_text(encoding="utf-8", errors="replace")
    prep = prepare_cost(log)
    if not prep or "phase=COMPLETE" not in log or \
            "output=REAL_LOGITS" not in log:
        raise RuntimeError(f"Missing native completion/Prepare: {log_file}")
    if report.get("plan") != case["chunks"]:
        raise RuntimeError(f"Native ran wrong Chunk plan: {log_file}")
    if mode == "autonomous":
        if log.count("phase=PHONE_AUTO_PUSH") != len(case["chunks"]) or \
                log.count("phase=PHONE_AUTO_P2C_DONE") != 1 or \
                "phase=PHONE_AUTO_START" not in log:
            raise RuntimeError(f"Missing Phone PUSH/P2C event: {log_file}")
    else:
        if "phase=PHONE_AUTO_PUSH" in log:
            raise RuntimeError(f"Sync unexpectedly used Phone autonomous: {log_file}")
    latency = report.get("prompt_eval_ms")
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or \
            latency <= 0:
        raise RuntimeError(f"Invalid Prefill timing in {work}/results.json")
    logits = work / "native-logits.f32"
    if not logits.is_file():
        raise RuntimeError(f"Missing full F32 Logits: {logits}")
    return {"ms": float(latency), "prepare": prep,
            "cache_hit": prep["zero_explicit_prepare_phases"],
            "top1": report["top1"],
            "logits": str(logits), "log": str(log_file),
            "timing_tokens": report.get("prompt_eval_timing_tokens")}


def parity_ok(parity: dict) -> bool:
    return (parity["top1_same"] and parity["cosine"] is not None and
            parity["cosine"] >= 0.999 and
            parity["relative_rmse"] is not None and
            parity["relative_rmse"] <= 0.01)


def perform_case(args: argparse.Namespace, case: dict,
                 output: Path, position: int) -> dict:
    root = case_directory(output, case)
    attempt = next_attempt(root)
    order = ["sync", "autonomous"]
    if args.mode_order == "autonomous-first" or \
            (args.mode_order == "alternate" and position % 2):
        order.reverse()
    if not args.no_warmup:
        run_mode(args, case, "sync", attempt / "warmup")
    runs = {}
    for mode in order:
        runs[mode] = run_mode(args, case, mode, attempt / mode)
    parity = logits_comparison(
        read_logits(Path(runs["sync"]["logits"])),
        read_logits(Path(runs["autonomous"]["logits"])))
    if not parity_ok(parity):
        raise RuntimeError(
            f"Complete Logits parity FAIL for {case['name']}: {parity}")
    comparable = runs["sync"]["cache_hit"] and runs["autonomous"]["cache_hit"]
    row = dict(case, status="PASS",
               first_mode=order[0], runs=runs, parity=parity,
               warm_comparable=comparable,
               sync_ms=runs["sync"]["ms"],
               autonomous_ms=runs["autonomous"]["ms"],
               speedup=runs["sync"]["ms"] / runs["autonomous"]["ms"],
               case_dir=str(root), attempt_dir=str(attempt))
    return row


def csv_row(row: dict) -> dict:
    runs = row.get("runs", {})
    sync, auto = runs.get("sync", {}), runs.get("autonomous", {})
    parity = row.get("parity") or {}
    tokens = row.get("tokens")
    return {
        "layout": row["layout"], "group": row["group"],
        "prompt": row["name"], "kind": row["kind"],
        "repeat": row["repeat"], "status": row["status"],
        "tokens": tokens, "context": row["context"],
        "chunks": ",".join(map(str, row["chunks"])),
        "chunk_size": row["chunks"][0], "n_chunks": len(row["chunks"]),
        "first_mode": row.get("first_mode", ""),
        "sync_ms": sync.get("ms"), "autonomous_ms": auto.get("ms"),
        "sync_tok_s": round(tokens*1000/sync["ms"], 3) if sync.get("ms") else "",
        "autonomous_tok_s": round(tokens*1000/auto["ms"], 3) if auto.get("ms") else "",
        "speedup": row.get("speedup"),
        "sync_cache_hit": sync.get("cache_hit"),
        "autonomous_cache_hit": auto.get("cache_hit"),
        "warm_comparable": row.get("warm_comparable"),
        "top1_same": parity.get("top1_same"),
        "max_abs": parity.get("max_abs"),
        "relative_rmse": parity.get("relative_rmse"),
        "cosine": parity.get("cosine"),
        "revisit_parity": row.get("revisit_parity"),
        "error": row.get("error", ""),
        "case_dir": row.get("case_dir", ""),
    }


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    temp.replace(path)


def group_summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "PASS" and row["kind"] == "normal":
            groups[(row["layout"], row["group"])].append(row)
    result = []
    for (layout, group), items in groups.items():
        warm = [x for x in items if x.get("warm_comparable")]
        ratios = [x["speedup"] for x in warm]
        sync_ms = [x["sync_ms"] for x in items]
        auto_ms = [x["autonomous_ms"] for x in items]
        result.append({
            "layout": layout, "group": group, "passed_pairs": len(items),
            "distinct_prompts": len({x["name"] for x in items}),
            "warm_pairs": len(warm),
            "sync_mean_ms": round(statistics.mean(sync_ms), 3),
            "autonomous_mean_ms": round(statistics.mean(auto_ms), 3),
            "sync_median_ms": round(statistics.median(sync_ms), 3),
            "autonomous_median_ms": round(statistics.median(auto_ms), 3),
            "warm_geomean_speedup": (
                round(statistics.geometric_mean(ratios), 4) if ratios else ""),
            "warm_mean_speedup": (
                round(statistics.mean(ratios), 4) if ratios else ""),
            "warm_mean_sync_ms": (
                round(statistics.mean(x["sync_ms"] for x in warm), 3)
                if warm else ""),
            "warm_mean_autonomous_ms": (
                round(statistics.mean(x["autonomous_ms"] for x in warm), 3)
                if warm else ""),
            "sync_weighted_tok_s": round(
                1000*sum(x["tokens"] for x in items)/sum(sync_ms), 3),
            "autonomous_weighted_tok_s": round(
                1000*sum(x["tokens"] for x in items)/sum(auto_ms), 3),
        })
    return sorted(result, key=lambda x: (x["layout"], group_key(
        f"prompt_{x['group']}_01.txt")))


def publish(output: Path, rows: list[dict],
            expected_cases: int, manifest_sha: str) -> None:
    by_status = {k: sum(x["status"] == k for x in rows)
                 for k in ("PASS", "FAIL", "INTERRUPTED")}
    summaries = group_summaries(rows)
    all_warm = [r["speedup"] for r in rows if r["status"] == "PASS"
                and r["kind"] == "normal" and r.get("warm_comparable")]
    write_json(output / "all-results.json", rows)
    write_csv(output / "all-results.csv", [csv_row(r) for r in rows],
              FIELDNAMES)
    write_csv(output / "group-summary.csv", summaries,
              tuple(summaries[0]) if summaries else
              ("layout", "group", "passed_pairs", "warm_pairs"))
    summary = {
        "manifest_sha256": manifest_sha,
        "expected_cases": expected_cases,
        "finished_cases": len(rows),
        "remaining_cases": expected_cases-len(rows),
        "passed": by_status["PASS"],
        "failed": by_status["FAIL"],
        "interrupted": by_status["INTERRUPTED"],
        "warm_comparable_normal_pairs": len(all_warm),
        "warm_geomean_speedup": (
            round(statistics.geometric_mean(all_warm), 4)
            if all_warm else None),
        "group_summary": summaries,
        "interpretation": (
            "Compare speedups only for warm-comparable A/B pairs; "
            "one full suite is a first measurement, not a confidence interval."
        ),
    }
    write_json(output / "summary.json", summary)


def run() -> int:
    args = parse_args()
    layouts = list(dict.fromkeys(int(s.strip()) for s in args.layouts.split(",")))
    if not layouts or any(x < 32 or x > 39 or
                          args.cpu_layers < 0 or args.cpu_layers >= x
                          for x in layouts):
        raise ValueError("Layouts must be first Phone layers 32..39 "
                         "with CPU prefix < PC layers")
    if args.repeats < 1 or args.expected_count < 0 or args.chunk_size < 1 or \
            args.timeout_seconds < 0:
        raise ValueError("Invalid repeat, expected-count, Chunk or timeout")
    for name in ("exe", "tokenizer", "model"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = discover(Path(args.prompts_dir), args.pattern)
    if args.expected_count and len(files) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} unique Prompts, "
                         f"found {len(files)}. Refusing partial experiment.")
    plan = preflight(args, files, layouts, output)
    cases = build_cases(plan, layouts, args.repeats, args.revisit_first)
    print(f"[PHONE_SUITE] prompts={len(files)} layouts={layouts}"
          f" repeats={args.repeats} total_pairs={len(cases)}"
          f" warmup={not args.no_warmup} status=PLANNED", flush=True)
    if args.dry_run:
        print(f"[PHONE_SUITE] manifest={output/'suite-manifest.json'}"
              " dry_run=PASS")
        return 0

    rows = []
    reference = {}
    stopped = False
    for position, case in enumerate(cases):
        root = case_directory(output, case)
        stored = root / "case-result.json"
        if args.resume and stored.exists():
            previous = json.loads(stored.read_text(encoding="utf-8"))
            logs = [Path(x.get("logits", "")) for x in
                    previous.get("runs", {}).values()]
            if previous.get("status") == "PASS" and len(logs) == 2 and \
                    all(p.is_file() for p in logs):
                row = previous
                print(f"[PHONE_SUITE] {position+1}/{len(cases)}"
                      f" prompt={case['name']} layout={case['layout']}"
                      " status=SKIP_PASS", flush=True)
            else:
                row = None
        else:
            row = None
        if row is None:
            print(f"[PHONE_SUITE] {position+1}/{len(cases)}"
                  f" layout=L{case['layout']} group={case['group']}"
                  f" prompt={case['name']} kind={case['kind']}"
                  f" repeat={case['repeat']} status=RUN", flush=True)
            try:
                row = perform_case(args, case, output, position)
                if case["kind"] == "revisit":
                    key = (case["layout"], case["group"],
                           case["path"], case["repeat"])
                    initial = reference.get(key)
                    if initial is None:
                        raise RuntimeError("Cannot compare A-B-A: initial A missing")
                    parity = {}
                    for mode in ("sync", "autonomous"):
                        parity[mode] = logits_comparison(
                            read_logits(Path(initial["runs"][mode]["logits"])),
                            read_logits(Path(row["runs"][mode]["logits"])))
                        if not parity_ok(parity[mode]):
                            raise RuntimeError(
                                f"A-B-A KV isolation changed {mode} logits: "
                                f"{parity[mode]}")
                    row["revisit_parity"] = {
                        m: {"max_abs": p["max_abs"], "top1_same": p["top1_same"]}
                        for m, p in parity.items()}
                write_json(stored, row)
            except KeyboardInterrupt:
                row = dict(case, status="INTERRUPTED",
                           error="User interrupted", case_dir=str(root))
                stopped = True
            except Exception as exc:
                row = dict(case, status="FAIL", error=str(exc),
                           case_dir=str(root))
                write_json(stored, row)
        if row["status"] == "PASS" and case["kind"] == "normal":
            reference[(case["layout"], case["group"],
                       case["path"], case["repeat"])] = row
        rows.append(row)
        publish(output, rows, len(cases), plan["signature"])
        if row["status"] == "PASS":
            print(f"[PHONE_SUITE] result=PASS"
                  f" sync_ms={row['sync_ms']:.2f}"
                  f" autonomous_ms={row['autonomous_ms']:.2f}"
                  f" ratio={row['speedup']:.3f}"
                  f" warm_comparable={row['warm_comparable']}", flush=True)
        else:
            print(f"[PHONE_SUITE] result={row['status']}"
                  f" prompt={case['name']} error={row['error'][:300]}",
                  file=sys.stderr, flush=True)
        if stopped or (args.fail_fast and row["status"] != "PASS"):
            break

    failures = sum(row["status"] != "PASS" for row in rows)
    complete = len(rows) == len(cases)
    success = complete and failures == 0
    print(f"[PHONE_SUITE_DONE] status={'PASS' if success else 'PARTIAL_OR_FAIL'}"
          f" done={len(rows)}/{len(cases)} failures={failures}"
          f" summary={output/'summary.json'}"
          f" csv={output/'group-summary.csv'}", flush=True)
    return 0 if success else 2


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"[PHONE_SUITE] FATAL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
