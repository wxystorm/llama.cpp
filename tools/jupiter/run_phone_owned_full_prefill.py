#!/usr/bin/env python3
"""Real Qwen3-14B full-Prefill A/B (PC CPU16/CUDA16 + Phone HTP8).

Legacy and native runs use the SAME raw prompt and token schedule.
Native mode changes the actual llama.cpp model graph, not just a smoke RPC.
This driver measures real final logits, not generated text equality.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

PLAN_RE = re.compile(r"\[JUPITER_SEQUENCE_PLAN\].*?total=(\d+).*?chunks=([0-9,]+)")
TIMING_RE = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
NATIVE_DONE = "[JUPITER_PHONE_OWNED_PREFILL] phase=COMPLETE"
NATIVE_CHUNK = "[JUPITER_PHONE_OWNED_PREFILL] phase=CHUNK_DONE"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, help="rebuilt native-branch llama-cli.exe")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True, help="same endpoint used for --rpc")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--pc-layers", type=int, default=32, help="first Phone HTP layer, 32..39")
    p.add_argument("--cpu-layers", type=int, default=16, help="PC CPU prefix layer count")
    p.add_argument("--context", type=int, default=2048,
                   help="KV capacity, 256..4096; formal full suite locks 2048")
    p.add_argument("--target-tokens", type=int, default=175)
    p.add_argument("--chunks", default="88,87")
    p.add_argument("--mode", choices=("both", "legacy", "native"), default="both")
    p.add_argument("--async-phone", action="store_true",
                   help="legacy PC worker overlap; retains synchronous Phone RUN RPC")
    p.add_argument("--autonomous-phone", action="store_true",
                   help="Phone server owns FIFO HTP worker; PC PUSH triggers execution and final P2C returns asynchronously")
    p.add_argument("--stage-dump", action="store_true",
                   help="debug only: export PC intermediate F32 stage tensors "
                        "(inserts scheduler synchronization; not for timing)")
    p.add_argument("--out-dir", default="phone-owned-full-prefill-14b")
    p.add_argument("--verbosity", type=int, default=4)
    return p.parse_args()


def read_logits(path: Path) -> array.array:
    raw = path.read_bytes()
    if len(raw) < 4 or len(raw) % 4:
        raise ValueError(f"invalid binary F32 logits: {path}")
    result = array.array("f")
    result.frombytes(raw)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"nonfinite logit in {path}")
    return result


def logits_comparison(a: array.array, b: array.array) -> dict:
    if len(a) != len(b):
        raise ValueError(f"logit dimensions differ: {len(a)} vs {len(b)}")
    n = len(a)
    max_index = max(range(n), key=lambda i: abs(float(a[i])-float(b[i])))
    dot = math.fsum(float(x)*float(y) for x, y in zip(a, b))
    sq_a = math.fsum(float(x)*float(x) for x in a)
    sq_b = math.fsum(float(y)*float(y) for y in b)
    sq_diff = math.fsum((float(x)-float(y))**2 for x, y in zip(a, b))
    top_a = max(range(n), key=lambda i: a[i])
    top_b = max(range(n), key=lambda i: b[i])
    return {
        "count": n,
        "legacy_top1": top_a,
        "native_top1": top_b,
        "top1_same": top_a == top_b,
        "legacy_top1_logit": float(a[top_a]),
        "native_top1_logit": float(b[top_b]),
        "max_abs": abs(float(a[max_index])-float(b[max_index])),
        "max_abs_index": max_index,
        "mae": math.fsum(abs(float(x)-float(y)) for x, y in zip(a, b))/n,
        "rmse": math.sqrt(sq_diff/n),
        "relative_rmse": math.sqrt(sq_diff/sq_a) if sq_a else None,
        "cosine": dot/math.sqrt(sq_a*sq_b) if sq_a and sq_b else None,
        "numeric_parity": "MEASURED_REVIEW_REQUIRED",
    }


def main() -> int:
    args = arguments()
    exe, model, prompt_path = (
        Path(args.exe).resolve(),
        Path(args.model).resolve(),
        Path(args.prompt_file).resolve(),
    )
    for p in (exe, model, prompt_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    chunks = [int(x.strip()) for x in args.chunks.split(",")]
    if (not chunks or any(x <= 0 or x > 4096 for x in chunks)
            or sum(chunks) != args.target_tokens or len(chunks) > 16):
        raise ValueError("native requires exact Jupiter Sequence DP cuts summing "
                         "to --target-tokens with at most 16 Chunks")
    if args.async_phone and args.autonomous_phone:
        raise ValueError("--async-phone and --autonomous-phone are exclusive")
    if (args.async_phone or args.autonomous_phone) and args.stage_dump:
        raise ValueError("stream overlap and stage-dump cannot be combined")
    if args.async_phone and len(chunks) < 2:
        raise ValueError("PC-worker overlap requires at least two Prompt Chunks; "
                         "Phone-autonomous Jupiter DP accepts a single Chunk")
    if not 32 <= args.pc_layers <= 39 or not 0 <= args.cpu_layers < args.pc_layers:
        raise ValueError("native requires 32..39 PC layers and 0 <= CPU layers < PC layers")
    if not 256 <= args.context <= 4096 or args.context % 256 != 0:
        raise ValueError("native requires 256..4096 Context divisible by 256")
    if args.target_tokens > args.context or args.target_tokens < 2 or len(chunks) > 16:
        raise ValueError("native requires 2..context tokens and at most 16 chunks")
    prompt = prompt_path.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not prompt:
        raise ValueError("empty Prompt")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_modes = ("legacy", "native") if args.mode == "both" else (args.mode,)
    results = {}
    for mode in run_modes:
        native = mode == "native"
        env = os.environ.copy()
        env.pop("LLAMA_JUPITER_PLAN_FILE", None)
        env.update({
            "LLAMA_JUPITER_PREFILL": "1",
            "LLAMA_JUPITER_PC_LAYERS": str(args.pc_layers),
            "LLAMA_JUPITER_CPU_LAYERS": str(args.cpu_layers),
            "LLAMA_JUPITER_HTP_SESSIONS": str(args.sessions),
            "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(args.sessions),
            "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": str(args.context),
            "LLAMA_JUPITER_SEQUENCE_PLAN": ",".join(map(str, chunks)),
            "LLAMA_JUPITER_WINDOW": "2",
            "LLAMA_JUPITER_GRAPH_PREBUILD": "0",
            "LLAMA_JUPITER_LOCAL_GRAPH_REUSE": "0",
            "LLAMA_JUPITER_ASYNC_C2P": "0",
            "LLAMA_JUPITER_PHONE_OWNED_NATIVE": "1" if native else "0",
            "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "1" if native and args.async_phone else "0",
            "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "1" if native and args.autonomous_phone else "0",
            "LLAMA_JUPITER_PHONE_OWNED_ENDPOINT": args.rpc,
            "LLAMA_JUPITER_TIMELINE": "1",
            "LLAMA_JUPITER_TRACE": "1",
            "LLAMA_JUPITER_SCHED_ALLOC_TIMING": "1",
            "LLAMA_JUPITER_LOGITS_HASH": "1",
            "LLAMA_JUPITER_LOGITS_DUMP_FILE": str(out / f"{mode}-logits.f32"),
        })
        if args.stage_dump:
            env["LLAMA_JUPITER_STAGE_DUMP_DIR"] = str(out / f"{mode}-pc-stages")
            env["LLAMA_JUPITER_STAGE_DUMP_TOKENS"] = str(args.target_tokens)
        else:
            env.pop("LLAMA_JUPITER_STAGE_DUMP_DIR", None)
            env.pop("LLAMA_JUPITER_STAGE_DUMP_TOKENS", None)
        command = [
            str(exe), "-m", str(model),
            "--rpc", args.rpc,
            "--split-mode", "layer", "-ngl", "all",
            "-c", str(args.context), "-b", str(args.context), "-ub", str(max(chunks)),
            "-n", "1", "-t", "8",
            "--flash-attn", "on",
            "--no-warmup",
            "--no-conversation", "--single-turn", "--simple-io",
            "--no-show-timings", "-lv", str(args.verbosity),
            "-p", prompt,
        ]
        log_path = out / f"{mode}-full-prefill.log"
        print(f"[FULL_PREFILL_AB] mode={mode} sessions={args.sessions} log={log_path}", flush=True)
        with log_path.open("wb") as f:
            proc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                  stdout=f, stderr=subprocess.STDOUT, check=False)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode:
            tail = "\n".join(log.splitlines()[-55:])
            raise RuntimeError(f"{mode} exited {proc.returncode}:\n{tail}")
        # The llama-server initialization normally probes seq_rm by doing a
        # synthetic two-token decode. In one-shot Phone-owned mode that would
        # consume the private HTP KV/run before the real 175-token Prompt.
        # Require the new memory-only probe bypass and exact request boundary.
        if native:
            probe_marker = (
                "[JUPITER_PHONE_OWNED_PREFILL] phase=SEQ_RM_PROBE"
            )
            if probe_marker not in log or "synthetic_decode=SKIPPED" not in log:
                raise RuntimeError(
                    "native: mandatory memory-only seq_rm probe bypass "
                    "missing; rebuild llama-cli from the updated branch. "
                    f"See {log_path}"
                )
            prepared_tokens = [
                int(x) for x in re.findall(
                    r"\[JUPITER_PHONE_OWNED_PREFILL\] phase=PREPARE_BEGIN"
                    r"[^\n]*?\btokens=(\d+)", log
                )
            ]
            if prepared_tokens != [args.target_tokens]:
                if len(prepared_tokens) == 1:
                    raise RuntimeError(
                        f"native: actual chat-formatted Prompt has "
                        f"{prepared_tokens[0]} tokens, but the requested "
                        f"Chunk Plan totals {args.target_tokens}. The CLI "
                        "uses /v1/chat/completions, which adds model-specific "
                        "Chat Template tokens. For multi-Prompt runs, rebuild "
                        "llama-tokenize with --chat-user and use "
                        "run_phone_owned_multi_prompt.py; do not hardcode "
                        f"a token offset. See {log_path}"
                    )
                raise RuntimeError(
                    f"native: expected exactly one real Prompt PREPARE_BEGIN "
                    f"with {args.target_tokens} tokens, got {prepared_tokens}; "
                    "an extra compatibility request or incomplete request "
                    f"may have consumed the Phone-owned run. See {log_path}"
                )
        plan_matches = list(PLAN_RE.finditer(log))
        if not plan_matches:
            # The CLI server wrapper may exit with status 0 after an internal
            # request failure. Report the *earlier* failure rather than
            # misdiagnosing a missing Plan as the root cause.
            lines = log.splitlines()
            clues = [
                line for line in lines
                if any(needle in line.lower() for needle in (
                    "jupiter_phone_owned", "jupiter_sequence_decision",
                    "jupiter_stage_plan", "reason=", "error", "fail",
                    "could not", "unable to", "unsupported", "exception",
                    "not found", "model loaded", "launch_slot",
                    "stop processing", "prompt eval time",
                ))
            ]
            clue_text = "\n".join(clues[-35:]) or "(no matching milestones)"
            tail_text = "\n".join(lines[-45:]) or "(empty log)"
            raise RuntimeError(
                f"{mode}: no JUPITER_SEQUENCE_PLAN was emitted. "
                f"Possibilities: the request failed before init_batch, "
                f"or the explicit chunk plan was rejected. "
                f"Inspect the original cause below; no model speedup/parity "
                f"can be inferred.\n"
                f"--- key milestones/errors ---\n{clue_text}\n"
                f"--- last 45 log lines ---\n{tail_text}\n"
                f"--- full log: {log_path}"
            )
        observed_total = int(plan_matches[-1].group(1))
        observed = [int(x) for x in plan_matches[-1].group(2).split(",")]
        if observed_total != args.target_tokens or observed != chunks:
            raise RuntimeError(
                f"{mode}: observed {observed_total} / {observed}, expected "
                f"{args.target_tokens} / {chunks}")
        if native and NATIVE_DONE not in log:
            raise RuntimeError(f"native suffix incomplete; see {log_path}")
        if native and args.autonomous_phone:
            if (log.count("phase=PHONE_AUTO_PUSH") != len(chunks) or
                    log.count("phase=PHONE_AUTO_P2C_DONE") != 1):
                raise RuntimeError(
                    f"Phone autonomous input/P2C contract incomplete; see {log_path}")
        elif native and log.count(NATIVE_CHUNK) != len(chunks):
            raise RuntimeError(
                f"native suffix did not complete all {len(chunks)} chunks; see {log_path}")
        if native and "output=REAL_LOGITS" not in log:
            raise RuntimeError("native mode did not produce PC head logits")
        if not native and NATIVE_DONE in log:
            raise RuntimeError("legacy run unexpectedly used native suffix")
        logits_path = out / f"{mode}-logits.f32"
        logits = read_logits(logits_path)
        timing = list(TIMING_RE.finditer(log))
        results[mode] = {
            "log": str(log_path),
            "logits": str(logits_path),
            "pc_stage_dump_dir": str(out / f"{mode}-pc-stages") if args.stage_dump else None,
            "timing_valid_for_performance": not args.stage_dump,
            "logits_count": len(logits),
            "top1": max(range(len(logits)), key=lambda i: logits[i]),
            "plan": observed,
            "prompt_eval_ms": float(timing[-1].group(1)) if timing else None,
            "prompt_eval_timing_tokens": int(timing[-1].group(2)) if timing else None,
        }
        print(f"[FULL_PREFILL_AB] mode={mode} top1={results[mode]['top1']} "
              f"prompt_eval_ms={results[mode]['prompt_eval_ms']} status=PASS", flush=True)
    if all(x in results for x in ("legacy", "native")):
        result = logits_comparison(
            read_logits(Path(results["legacy"]["logits"])),
            read_logits(Path(results["native"]["logits"])),
        )
        results["parity"] = result
        print("[FULL_PREFILL_AB] logits_numeric=" + json.dumps(result, ensure_ascii=False),
              flush=True)
    report = out / "results.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"[FULL_PREFILL_AB] report={report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FULL_PREFILL_AB] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
