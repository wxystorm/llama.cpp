#!/usr/bin/env python3
"""One-command Qwen3-14B/32B Phone-owned Prefill suite over ShareGPT prompts.

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
from phone_owned_models import inspect_model, resolve_splits
from run_prefill_bench import read_plan

PROMPT_NAME = re.compile(r"^prompt_(.+)_(\d+)$", re.IGNORECASE)
CHUNK_BUCKETS = (32, 64, 88, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
# ShareGPT40: exactly five Prompt-length groups, ten prompts each.
# Do not accidentally include the separate 2048 / 2048-4000 group.
DEFAULT_GROUPS = "0,128,256,512,1024"
FIELDNAMES = (
    "layout", "group", "prompt", "kind", "repeat", "status", "tokens",
    "context", "chunks", "chunk_size", "n_chunks",
    "min_chunk", "plan_source", "plan_file", "first_mode",
    "sync_ms", "autonomous_ms", "sync_tok_s", "autonomous_tok_s",
    "speedup", "sync_cache_hit", "autonomous_cache_hit",
    "warm_comparable", "top1_same", "max_abs", "relative_rmse",
    "cosine", "revisit_parity", "revisit_first_cache_hit",
    "warmup_cache_hit", "error", "case_dir",
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


def choose_context(tokens: int) -> int:
    # Jupiter baseline: fixed Context and Batch, not per-Prompt dynamic KV.
    if tokens < 2 or tokens > 2048:
        raise ValueError(
            f"Actual chat-formatted Prompt has {tokens} tokens; "
            "fixed Jupiter Context=2048 cannot fit it")
    return 2048


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
    p.add_argument("--model", required=True, help="dense Qwen3-14B/32B GGUF")
    p.add_argument("--model-size", choices=("auto", "14b", "32b"),
                   default="auto", help="verify GGUF architecture; default auto")
    p.add_argument("--rpc", required=True, help="running Phone host:port")
    p.add_argument("--prompts-dir", required=True,
                   help="directory containing prompt_0_01.txt, etc.")
    p.add_argument("--pattern", default="prompt_*_*.txt")
    p.add_argument("--groups", default=DEFAULT_GROUPS,
                   help="comma-separated filename groups to test; default "
                        "0,128,256,512,1024 (5x10=50); 'all' opts in to "
                        "other groups including 2048")
    p.add_argument("--out-dir", default=None,
                   help="default: jupiter-autonomous-{14b|32b}-dp-2048")
    p.add_argument("--layouts", default=None,
                   help="first Phone layers: 14B 24..39; 32B 48..63")
    p.add_argument("--cpu-layers", type=int, default=None,
                   help="PC CPU layers (default 16 for 14B, 44 for 32B)")
    p.add_argument("--cuda-layers", type=int, default=None,
                   help="PC CUDA layers; derives CPU when not explicitly set")
    p.add_argument("--phone-layers", type=int, default=None,
                   help="Phone HTP suffix layer count 1..16; conflicts with --layouts")
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--sequence-policy", choices=("jupiter-dp", "fixed-control"),
                   default="jupiter-dp",
                   help="default: original Jupiter Stage2 b + measured q(x,y) Sequence DP")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="used only for explicitly labeled fixed-control ablation")
    p.add_argument("--min-chunk-json", default=None,
                   help="optional original Stage2 min-chunk.json; without it, "
                        "measure Stage2 on the same PC/Phone model split")
    p.add_argument("--stage2-repeats", type=int, default=2)
    p.add_argument("--q-repeats", type=int, default=2)
    p.add_argument("--refine-repeats", type=int, default=2)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-chunk", type=int, default=128,
                   help="original Jupiter q(x,y) max UBatch/Chunk size")
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
    p.add_argument("--continue-on-error", action="store_true",
                   help="continue after failed Phone RPC; off by default to avoid cascading active-run failures")
    p.add_argument("--timeout-seconds", type=int, default=1800,
                   help="per child run; kill the whole local subprocess tree")
    p.add_argument("--dry-run", action="store_true",
                   help="plan/tokenize only; does not contact Phone RPC")
    return p.parse_args()


def discover(root: Path, pattern: str,
             groups: set[str] | None = None) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates = sorted(
        (p.resolve() for p in root.glob(pattern) if p.is_file()),
        key=lambda p: group_key(p.name))
    if not candidates:
        raise ValueError(f"No Prompt files matching {pattern!r} in {root}")
    if len({p.name.lower() for p in candidates}) != len(candidates):
        raise ValueError("Duplicate Prompt filenames")
    files = [p for p in candidates if groups is None or group_of(p) in groups]
    if not files:
        raise ValueError(f"No Prompt files in requested groups {groups}; "
                         f"pattern={pattern!r}, directory={root}")
    if groups is not None:
        actual_groups = sorted({group_of(p) for p in files},
                               key=lambda s: group_key(f"prompt_{s}_01.txt"))
        print(f"[PHONE_SUITE_DISCOVER] selected={len(files)} "
              f"excluded={len(candidates)-len(files)} "
              f"groups={','.join(actual_groups)} "
              f"status=FILTERED_BEFORE_TOKENIZATION", flush=True)
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
        ctx = choose_context(n_tokens)
        chunks = (choose_chunks(n_tokens, args.chunk_size)
                  if args.sequence_policy == "fixed-control" else [])
        row = {"name": prompt_file.name, "path": str(prompt_file),
               "sha256": sha, "group": group_of(prompt_file),
               "tokens": n_tokens, "context": ctx, "chunks": chunks}
        tokens[prompt_file] = row
        groups[row["group"]].append(row)
        print(f"[PHONE_SUITE_PLAN] {index}/{len(files)} "
              f"group={row['group']} prompt={prompt_file.name}"
              f" tokens={n_tokens} ctx={ctx} chunks={chunks or 'JUPITER_DP_PENDING'}", flush=True)
    config = {
        "format": 1,
        "exe": file_signature(Path(args.exe)),
        "tokenizer": tokenizer_id,
        "model": model_id, "model_size": args.model_size,
        "model_layers": args.model_layers,
        "phone_layer_counts": [args.model_layers - x for x in layouts],
        "rpc": args.rpc, "prompt_root": str(Path(args.prompts_dir).resolve()),
        "pattern": args.pattern, "groups": args.groups,
        "layouts": layouts,
        "cpu_layers": args.cpu_layers, "sessions": args.sessions,
        "sequence_policy": args.sequence_policy,
        "chunk_size": args.chunk_size, "context": 2048, "batch": 2048,
        "min_chunk_json": (file_signature(Path(args.min_chunk_json))
                           if args.min_chunk_json else None),
        "stage2_repeats": args.stage2_repeats,
        "q_repeats": args.q_repeats,
        "refine_repeats": args.refine_repeats,
        "max_rounds": args.max_rounds, "max_chunk": args.max_chunk,
        "repeats": args.repeats,
        "mode_order": args.mode_order, "warmup": not args.no_warmup,
        "revisit_first": args.revisit_first,
        "prompts": list(tokens.values()),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(
            Path(__file__).with_name("run_phone_owned_full_prefill.py").read_bytes()
        ).hexdigest(),
        "jupiter_planner_code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "jupiter_planner.py", "run_min_chunk_profile.py",
                "min_chunk_profile.py", "run_q_sweep.py",
                "run_sequence_pipeline.py", "refine_sequence_q.py",
            )
        },
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
        "--model-size", args.model_size,
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


def child_run(command: list[str], stdout_path: Path, timeout: int,
              env: dict | None = None) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stream:
        options = {"stdin": subprocess.DEVNULL, "stdout": stream,
                   "stderr": subprocess.STDOUT, "env": env}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        child = subprocess.Popen(command, **options)
        try:
            code = child.wait(timeout=timeout if timeout > 0 else None)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(child.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False)
            else:
                os.killpg(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise RuntimeError(
                f"Timeout after {timeout}s; local child tree killed. "
                "Check Phone active-run recovery before continuing.") from exc
    if code:
        tail = "\n".join(
            stdout_path.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-28:])
        raise RuntimeError(f"Runner failed exit={code}; {stdout_path}\n{tail}")


def calibration_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    # Jupiter's original q(x,y) calibrator and model partition MUST run
    # without the Phone-native graph replacement being benchmarked.
    env.update({
        "LLAMA_JUPITER_PHONE_OWNED_NATIVE": "0",
        "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "0",
        "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "0",
        "LLAMA_JUPITER_HTP_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": "2048",
    })
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    return env


def stage2_for_layout(args: argparse.Namespace, cases: list[dict],
                      layout: int, output: Path) -> Path:
    if args.min_chunk_json:
        source = Path(args.min_chunk_json).resolve()
    else:
        stage2 = output / "calibration" / f"l{layout:02d}" / "stage2"
        source = stage2 / "min-chunk.json"
        if not source.is_file():
            # Jupiter Stage2 is a *hardware* measurement, not a per-Prompt
            # heuristic. Measure one representative prompt with the same
            # model, CPU/GPU boundary, Phone HTP sessions and Context.
            available = [x for x in cases if x["layout"] == layout and
                         x["kind"] == "normal"]
            if not available:
                raise ValueError(f"No Prompt available for Stage2 L{layout}")
            sample = min(available,
                         key=lambda x: (abs(x["tokens"] - 357), x["name"]))
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_min_chunk_profile.py")),
                "--exe", str(Path(args.exe).resolve()),
                "--model", str(Path(args.model).resolve()),
                "--rpc", args.rpc,
                "--prompt-file", sample["path"],
                "--pc-layers", str(layout),
                "--cpu-layers", str(args.cpu_layers),
                "--target-tokens", str(sample["tokens"]),
                "--max-chunk", str(args.max_chunk),
                "--batch", "2048", "--context", "2048",
                "--repeats", str(args.stage2_repeats),
                "--out-dir", str(stage2), "--resume",
            ]
            print(f"[PHONE_SUITE_STAGE2] layout={layout}"
                  f" sample={sample['name']} tokens={sample['tokens']}"
                  " policy=MEASURED_MINIMUM_B status=RUN", flush=True)
            child_run(command, stage2 / "stage2-runner.stdout.log", 0,
                      env=calibration_env(args))
    if not source.is_file():
        raise RuntimeError(f"Jupiter Stage2 min-chunk.json missing: {source}")
    info = json.loads(source.read_text(encoding="utf-8-sig"))
    if (int(info["pc_layers"]) != layout or
            int(info.get("cpu_layers", 0)) != args.cpu_layers or
            int(info["max_chunk"]) != args.max_chunk or
            not info.get("candidates")):
        raise ValueError(
            f"Jupiter Stage2 calibration does not match CPU/PC boundary, "
            f"maximum Chunk or measured candidates: {source}")
    return source


def exact_jupiter_dp_plan(args: argparse.Namespace, case: dict,
                          output: Path, stage2_json: Path) -> dict:
    target = output / "calibration" / f"l{case['layout']:02d}" / \
        "plans" / case["name"].removesuffix(".txt")
    stage3 = target / "stage3"
    plan_file = target / "final.plan"
    summary_path = stage3 / "sequence-summary.json"
    existing = (json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file() else None)
    if existing and (existing.get("pc_layers") != case["layout"] or
                     existing.get("cpu_layers") != args.cpu_layers or
                     existing.get("prompt_tokens") != case["tokens"]):
        raise ValueError("Old Jupiter Stage3 results use a different "
                         f"CPU/PC boundary or Prompt shape: {target}; "
                         "choose a new output directory")
    status = "FULLY_MEASURED_AND_VERIFIED"
    ready = (plan_file.is_file() and existing is not None and
             existing.get("status") == status and
             existing.get("pc_layers") == case["layout"] and
             existing.get("cpu_layers") == args.cpu_layers and
             existing.get("prompt_tokens") == case["tokens"])
    if not ready:
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_sequence_pipeline.py")),
            "--exe", str(Path(args.exe).resolve()),
            "--model", str(Path(args.model).resolve()),
            "--rpc", args.rpc, "--prompt-file", case["path"],
            "--min-chunk-json", str(stage2_json),
            "--target-tokens", str(case["tokens"]),
            "--reuse-b-profile-for-new-length",
            "--pc-layers", str(case["layout"]),
            "--cpu-layers", str(args.cpu_layers),
            "--q-repeats", str(args.q_repeats),
            "--refine-repeats", str(args.refine_repeats),
            "--max-rounds", str(args.max_rounds),
            "--max-subsequences", "16",
            "--batch", "2048", "--context", "2048",
            "--ubatch", str(args.max_chunk), "--threads", "8",
            "--out-dir", str(stage3),
            "--plan-output", str(plan_file), "--resume",
        ]
        print(f"[PHONE_SUITE_STAGE3] layout={case['layout']}"
              f" prompt={case['name']} tokens={case['tokens']}"
              " policy=EXACT_MEASURED_Q_DP status=RUN", flush=True)
        child_run(command, target / "stage3-runner.stdout.log", 0,
                  env=calibration_env(args))
    if not plan_file.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Jupiter Stage3 plan incomplete: {target}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != status:
        raise RuntimeError(f"Jupiter DP not fully measured and verified: "
                           f"{summary.get('status')}")
    lines = {}
    for raw in plan_file.read_text(encoding="utf-8-sig").splitlines():
        cleaned = raw.split("#", 1)[0]
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        lines[key.strip()] = value.strip()
    if (int(lines.get("model.pc_layers", -1)) != case["layout"] or
            int(lines.get("model.cpu_layers", 0)) != args.cpu_layers):
        raise ValueError(f"Jupiter model/CPU placement mismatch: {plan_file}")
    chunks = read_plan(plan_file).get(case["tokens"])
    if (not chunks or len(chunks) > 16 or
            sum(chunks) != case["tokens"] or
            chunks != summary.get("chunks") or
            max(chunks) > args.max_chunk):
        raise ValueError(f"Native Phone cannot execute exact Jupiter DP "
                         f"plan; no fixed-Chunk fallback: {plan_file}")
    planned = dict(case)
    planned["chunks"] = chunks
    planned["plan_file"] = str(plan_file)
    planned["plan_source"] = "JUPITER_STAGE2_B_QXY_EXACT_DP"
    planned["min_chunk"] = summary["min_chunk"]
    print(f"[PHONE_SUITE_DP_READY] layout={case['layout']} "
          f"prompt={case['name']} b={planned['min_chunk']}"
          f" chunks={chunks} status=READY", flush=True)
    return planned


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
    timing_tokens = report.get("prompt_eval_timing_tokens")
    if not isinstance(timing_tokens, int) or \
            abs(timing_tokens - case["tokens"]) > 1:
        raise RuntimeError(
            f"Jupiter timing token count {timing_tokens} differs from "
            f"planned {case['tokens']} by >1; invalid Prefill sample: {work}")
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
    warmup_info = None
    if not args.no_warmup:
        warm = run_mode(args, case, "sync", attempt / "warmup")
        # For A-B-A, this warmup is the FIRST revisit of shape A after B:
        # preserve its cache status before the paired measured runs refill it.
        warmup_info = {"cache_hit": warm["cache_hit"],
                       "prepare": warm["prepare"], "log": warm["log"]}
        # Warmup is not a measurement and need not keep another F32 dump.
        Path(warm["logits"]).unlink(missing_ok=True)
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
               warmup=warmup_info,
               revisit_first_cache_hit=(
                   warmup_info["cache_hit"] if warmup_info is not None
                   else runs[order[0]]["cache_hit"]
               ) if case["kind"] == "revisit" else None,
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
        "chunk_size": max(row["chunks"], default=0),
        "n_chunks": len(row["chunks"]),
        "min_chunk": row.get("min_chunk"),
        "plan_source": row.get("plan_source", ""),
        "plan_file": row.get("plan_file", ""),
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
        "revisit_first_cache_hit": row.get("revisit_first_cache_hit"),
        "warmup_cache_hit": (
            row["warmup"]["cache_hit"] if row.get("warmup") else None),
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
    # Validate against the GGUF before any expensive Stage2/Stage3 calibration.
    model = inspect_model(Path(args.model), args.model_size)
    layouts, args.cpu_layers = resolve_splits(
        model, layouts=args.layouts, cpu_layers=args.cpu_layers,
        cuda_layers=args.cuda_layers, phone_layers=args.phone_layers)
    args.model_size = model.family
    args.model_layers = model.layers
    if args.out_dir is None:
        args.out_dir = f"jupiter-autonomous-{model.family}-dp-2048"
    print(f"[PHONE_SUITE_MODEL] model=Qwen3-{model.family.upper()}"
          f" layers={model.layers} CPU={args.cpu_layers}"
          f" first_phone={layouts} GPU={[x-args.cpu_layers for x in layouts]}"
          f" HTP={[model.layers-x for x in layouts]}"
          " context=2048 status=VALIDATED", flush=True)
    if (args.repeats < 1 or args.expected_count < 0 or args.chunk_size < 1 or
            args.timeout_seconds < 0 or args.stage2_repeats < 1 or
            args.q_repeats < 1 or args.refine_repeats < 1 or
            args.max_rounds < 1 or args.max_chunk < 1 or
            args.max_chunk > 2048):
        raise ValueError("Invalid repeat, expected-count, Chunk or timeout")
    if args.min_chunk_json and not Path(args.min_chunk_json).is_file():
        raise FileNotFoundError(args.min_chunk_json)
    if args.min_chunk_json and len(layouts) != 1:
        raise ValueError("--min-chunk-json is layout-specific; use one layout "
                         "or let Stage2 measure each layout separately")
    for name in ("exe", "tokenizer", "model"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    group_arg = args.groups.strip()
    if not group_arg:
        raise ValueError("--groups must be a comma-separated list or 'all'")
    groups = None if group_arg.lower() == "all" else {
        x.strip() for x in group_arg.split(",") if x.strip()
    }
    if groups is not None and (not groups or len(groups) != len(group_arg.split(","))):
        raise ValueError("--groups contains an empty or repeated group")
    files = discover(Path(args.prompts_dir), args.pattern, groups=groups)
    if (args.expected_count == 50 and groups == set(DEFAULT_GROUPS.split(","))
            and args.pattern == "prompt_*_*.txt"):
        counts = {g: sum(group_of(p) == g for p in files) for g in groups}
        if any(n != 10 for n in counts.values()):
            raise ValueError(f"ShareGPT40 requires exactly 10 prompts per "
                             f"group 0/128/256/512/1024; found {counts}")
    if args.expected_count and len(files) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} unique Prompts, "
                         f"found {len(files)}. Refusing partial experiment.")
    plan = preflight(args, files, layouts, output)
    cases = build_cases(plan, layouts, args.repeats, args.revisit_first)
    print(f"[PHONE_SUITE] model=Qwen3-{model.family.upper()}"
          f" prompts={len(files)} layouts={layouts}"
          f" repeats={args.repeats} total_pairs={len(cases)}"
          f" context=2048 batch=2048 sequence_policy={args.sequence_policy}"
          f" warmup={not args.no_warmup} status=PLANNED", flush=True)
    if args.dry_run:
        print(f"[PHONE_SUITE] manifest={output/'suite-manifest.json'}"
              " dry_run=PASS")
        return 0

    rows = []
    reference = {}
    stage2_by_layout = {}
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
                if args.sequence_policy == "jupiter-dp":
                    layout = case["layout"]
                    if layout not in stage2_by_layout:
                        stage2_by_layout[layout] = stage2_for_layout(
                            args, cases, layout, output)
                    case = exact_jupiter_dp_plan(
                        args, case, output, stage2_by_layout[layout])
                else:
                    case["plan_source"] = "JUPITER_FIXED_CHUNK_ABLATION"
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
        if stopped or (not args.continue_on_error and row["status"] != "PASS"):
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
