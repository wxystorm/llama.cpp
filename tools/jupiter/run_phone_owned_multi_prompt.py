#!/usr/bin/env python3
"""Run different Qwen3-14B prompts against persistent Phone-owned HTP Graphs.

A -> B -> A checks whether the first Graph survives creation of a second
execution shape. The phone RPC server must remain alive between invocations.
Only the existing fresh-prefill, 512-token native pilot is supported.
"""
from __future__ import annotations

import argparse
import array
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from run_phone_owned_full_prefill import read_logits, logits_comparison
from test_phone_owned_residency import PREPARE

TOKEN_COUNT = re.compile(r"Total number of tokens:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, help="Windows llama-cli.exe")
    p.add_argument("--tokenizer", required=True, help="matching llama-tokenize.exe")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompts", nargs="+", required=True,
                   help="two or more UTF-8 .txt files; order is preserved")
    p.add_argument("--revisit-first", action="store_true",
                   help="replay first prompt after all supplied prompts (A-B-A)")
    p.add_argument("--require-revisit-hit", action="store_true",
                   help="fail if the replayed first shape was evicted")
    p.add_argument("--sessions", type=int, choices=(1, 4), default=4)
    p.add_argument("--pc-layers", type=int, default=32)
    p.add_argument("--cpu-layers", type=int, default=16)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk-size", type=int, default=88)
    p.add_argument("--out-dir", default="phone-owned-multi-prompt-s4")
    return p.parse_args()


def count_model_tokens(
        tokenizer: Path, model: Path, prompt: str) -> int:
    # The CLI sends one user message to /v1/chat/completions; its Server
    # expands the model's Jinja Chat Template before tokenization. Count
    # THAT exact formatted Prompt, not raw .txt tokens (Qwen3-14B may add
    # eight special/message-boundary tokens).
    # Read stdin to avoid the Windows command-line length limit.
    env = os.environ.copy()
    env["LLAMA_JUPITER_PREFILL"] = "0"
    env["LLAMA_JUPITER_PHONE_OWNED_NATIVE"] = "0"
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    result = subprocess.run(
        [str(tokenizer), "-m", str(model), "--stdin",
         "--chat-user", "--show-count", "--log-disable"],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, check=False,
    )
    match = TOKEN_COUNT.search(result.stdout)
    if result.returncode or not match:
        raise RuntimeError(
            f"tokenizer failed: exit={result.returncode}; "
            f"stdout_tail={result.stdout[-1000:]!r}; "
            f"stderr_tail={result.stderr[-1000:]!r}"
        )
    return int(match.group(1))


def chunks_for(n_tokens: int, size: int) -> list[int]:
    if not 2 <= n_tokens <= 4096 or not 1 <= size <= 4096:
        raise ValueError(
            f"native requires 2..4096 tokens and chunk size 1..4096; "
            f"got {n_tokens} and {size}"
        )
    chunks = [size] * (n_tokens // size)
    if n_tokens % size:
        chunks.append(n_tokens % size)
    if len(chunks) > 16:
        raise ValueError(f"Phone owned graph limit: >16 chunks ({chunks})")
    return chunks


def run() -> int:
    args = parse_args()
    if args.require_revisit_hit and not args.revisit_first:
        raise ValueError("--require-revisit-hit needs --revisit-first")
    if not 1 <= args.chunk_size <= 4096:
        raise ValueError("--chunk-size must be 1..4096")
    if not 256 <= args.context <= 4096 or args.context % 256 != 0:
        raise ValueError("--context must be 256..4096 and divisible by 256")
    if not 32 <= args.pc_layers <= 39 or not 0 <= args.cpu_layers < args.pc_layers:
        raise ValueError("--pc-layers must be 32..39 and CPU prefix must be smaller")
    exe = Path(args.exe).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    model = Path(args.model).resolve()
    prompts = [Path(s).resolve() for s in args.prompts]
    if len(prompts) < 2 or len(set(prompts)) < 2:
        raise ValueError("at least two different Prompt files are needed")
    for path in [exe, tokenizer, model, *prompts]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Cache tokenization results: expensive GGUF vocab initialization does
    # not need to be repeated when revisiting the first Prompt.
    token_counts: dict[Path, int] = {}
    for path in dict.fromkeys(prompts):
        prompt = path.read_text(encoding="utf-8-sig").rstrip("\r\n")
        if not prompt:
            raise ValueError(f"empty Prompt: {path}")
        token_counts[path] = count_model_tokens(tokenizer, model, prompt)
        if token_counts[path] > args.context:
            raise ValueError(f"{path.name}: tokens exceed context {args.context}")
        chunks = chunks_for(token_counts[path], args.chunk_size)
        print(f"[PHONE_MULTI_PROMPT] prompt={path.name} "
              f"tokens={token_counts[path]} chunks={chunks}", flush=True)

    shapes = {(token_counts[path], tuple(chunks_for(
        token_counts[path], args.chunk_size))) for path in prompts}
    if args.require_revisit_hit and len(shapes) < 2:
        raise ValueError(
            "all Prompt files have the same Token/Chunk shape; choose "
            "different Token counts to test multi-shape caching"
        )
    sequence = [*prompts, prompts[0]] if args.revisit_first else prompts
    reference: dict[Path, array.array] = {}
    records: list[dict] = []
    runner = Path(__file__).with_name("run_phone_owned_full_prefill.py")
    for index, path in enumerate(sequence, 1):
        n_tokens = token_counts[path]
        chunks = chunks_for(n_tokens, args.chunk_size)
        work = output / f"case-{index:02d}-{path.stem}"
        command = [
            sys.executable, str(runner),
            "--exe", str(exe), "--model", str(model),
            "--rpc", args.rpc, "--prompt-file", str(path),
            "--sessions", str(args.sessions),
            "--pc-layers", str(args.pc_layers),
            "--cpu-layers", str(args.cpu_layers),
            "--context", str(args.context),
            "--target-tokens", str(n_tokens),
            "--chunks", ",".join(map(str, chunks)),
            "--mode", "native", "--out-dir", str(work),
        ]
        print(f"[PHONE_MULTI_PROMPT] RUN {index}/{len(sequence)} "
              f"prompt={path.name} tokens={n_tokens}", flush=True)
        process = subprocess.run(
            command, capture_output=True, text=True,
            errors="replace", check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"prompt={path.name} failed: "
                f"{process.stdout[-2500:]} {process.stderr[-2500:]}"
            )

        log_path = work / "native-full-prefill.log"
        log = log_path.read_text(encoding="utf-8", errors="replace")
        prep_matches = list(PREPARE.finditer(log))
        if len(prep_matches) != 1:
            raise RuntimeError(
                f"prompt={path.name} expected one PREPARE_DONE; "
                f"found {len(prep_matches)} in {log_path}"
            )
        prep = dict(zip(
            ("graphs", "load_ms", "build_ms", "buffer_ms",
             "precompile_ms", "phone_wall_ms", "pc_wall_ms"),
            (float(v) for v in prep_matches[0].groups()),
        ))
        cache_hit = (prep["graphs"] > 0 and
                     all(prep[key] == 0.0 for key in (
                         "load_ms", "build_ms", "buffer_ms", "precompile_ms")))
        result = json.loads((work / "results.json").read_text(encoding="utf-8"))
        logits = read_logits(work / "native-logits.f32")
        row = {
            "index": index,
            "prompt": str(path),
            "tokens": n_tokens,
            "chunks": chunks,
            "cache_hit": cache_hit,
            "prepare": prep,
            "top1": result["native"]["top1"],
            "prefill_ms": result["native"]["prompt_eval_ms"],
            "log": str(log_path),
        }
        if path in reference:
            parity = logits_comparison(reference[path], logits)
            row["same_prompt_parity"] = parity
            if not parity["top1_same"] or (
                    parity["cosine"] is None or
                    parity["cosine"] < 0.999 or
                    parity["relative_rmse"] is None or
                    parity["relative_rmse"] > 0.01):
                raise RuntimeError(
                    f"KV isolation/replay parity failure for {path.name}: "
                    f"{parity}"
                )
        else:
            reference[path] = logits

        records.append(row)
        (output / "multi-prompt-results.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[PHONE_MULTI_PROMPT] run={index} prompt={path.name} "
            f"tokens={n_tokens} cache={'HIT' if cache_hit else 'MISS'} "
            f"prepare_pc_ms={prep['pc_wall_ms']:.3f} "
            f"prefill_ms={row['prefill_ms']} top1={row['top1']} "
            f"status=PASS", flush=True,
        )
        if (args.require_revisit_hit and
                index == len(sequence) and not cache_hit):
            raise RuntimeError(
                "revisited first Graph was evicted: increase the Phone's "
                "GGML_RPC_PHONE_OWNED_CACHE_BUDGET_MIB, use smaller prompts, "
                "or check CACHE_EVICT/RESET in the Phone log"
            )
    print(f"[PHONE_MULTI_PROMPT] report={output/'multi-prompt-results.json'} "
          "status=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"[PHONE_MULTI_PROMPT] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
