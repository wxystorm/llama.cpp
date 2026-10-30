#!/usr/bin/env python3
"""Stage 5: run Jupiter's measured planner and formal Prefill A/B per prompt.

One model/PC-phone partition, many prompt files. Prompt lengths come from
the actual llama.cpp prefill token probe, never filenames or character counts.
Keep every prompt's q profile, runtime plan, benchmark and configuration
separate so different lengths cannot contaminate one another.

No changes to the llama.cpp executor or transport.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

from run_q_sweep import (
    build_cli_command,
    infer_runtime_prompt_tokens,
    stream_process,
)
from run_prefill_bench import read_plan
from run_sequence_pipeline import resolve_stage2


SUMMARY_FIELDS = [
    "prompt", "actual_tokens", "status", "pc_layers", "min_chunk",
    "min_chunk_status", "chunk_count", "dp_chunks",
    "default_median_ms", "fixed64_median_ms", "dp_median_ms",
    "default_tok_s", "dp_tok_s", "speedup_vs_default",
    "sample_count", "case_dir", "final_plan", "error",
]


def discover_prompts(
    prompt_dir: Path | None, explicit: list[str],
    pattern: str, recursive: bool,
) -> list[Path]:
    found = []
    if prompt_dir is not None:
        if not prompt_dir.is_dir():
            raise ValueError(f"prompt directory not found: {prompt_dir}")
        found += list(prompt_dir.rglob(pattern) if recursive else prompt_dir.glob(pattern))
    found += [Path(value) for value in explicit]
    unique: dict[str, Path] = {}
    for item in found:
        path = item.resolve()
        if not path.is_file():
            raise ValueError(f"prompt is not a file: {path}")
        unique[str(path)] = path
    return sorted(unique.values(), key=lambda p: str(p).lower())


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def case_identity(
    prompt: Path, args: argparse.Namespace, stage2: Path, pc_layers: int
) -> dict[str, Any]:
    # Do not hash bench-repeats or max-rounds: extending completed runs must
    # be resumable without invalidating their q measurements.
    return {
        "prompt": str(prompt.resolve()),
        "prompt_sha256": sha256_bytes(prompt.read_bytes()),
        "exe": input_identity(Path(args.exe)),
        "model": input_identity(Path(args.model)),
        "stage2_json": str(stage2.resolve()),
        "stage2_sha256": sha256_bytes(stage2.read_bytes()),
        "pc_layers": pc_layers,
        "rpc": args.rpc,
        "min_chunk_override": args.min_chunk,
        "batch": args.batch, "context": args.context,
        "ubatch": args.ubatch, "threads": args.threads,
        "window": args.window, "verbosity": args.verbosity,
        "q_repeats": args.q_repeats,
        "refine_repeats": args.refine_repeats,
        "modes": args.modes,
        "async_c2p": False,
        "profile_stage_for_benchmark": False,
        "timing_token_tolerance": 1,
    }


def case_folder(out: Path, prompt: Path, config: dict) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", prompt.stem).strip("._-")
    slug = (slug or "prompt")[:48]
    fingerprint = sha256_bytes(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )[:12]
    return out / f"{slug}-{fingerprint}"


def probe_tokens(
    args: argparse.Namespace, prompt: Path, pc_layers: int,
    probe_log: Path, resume: bool,
) -> tuple[int, str]:
    if resume and probe_log.is_file():
        try:
            n, source = infer_runtime_prompt_tokens(probe_log)
            if n > 2:
                print(f"[JUPITER_SUITE_PROBE] action=REUSE tokens={n} "
                      f"source={source} log={probe_log}", flush=True)
                return n, source
        except (RuntimeError, ValueError):
            pass

    text = prompt.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not text:
        raise ValueError(f"prompt is empty: {prompt}")
    env = os.environ.copy()
    env.update({
        "LLAMA_JUPITER_PREFILL": "1",
        "LLAMA_JUPITER_PC_LAYERS": str(pc_layers),
        "LLAMA_JUPITER_WINDOW": "1",
        "LLAMA_JUPITER_PROFILE_STAGE": "0",
        "LLAMA_JUPITER_TRACE": "1",
        "LLAMA_JUPITER_Q_SWEEP": "1",
        "LLAMA_JUPITER_ASYNC_C2P": "0",
        "LLAMA_JUPITER_LOGITS_TOP10": "0",
    })
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    command_args = argparse.Namespace(
        rpc=args.rpc, max_chunk=args.ubatch, batch=args.batch,
        context=args.context, threads=args.threads, extra_arg=[],
    )
    command = build_cli_command(
        command_args, Path(args.exe).resolve(), Path(args.model).resolve(),
        text
    )
    print(f"[JUPITER_SUITE_PROBE] action=RUN prompt={prompt.name}", flush=True)
    code = stream_process(command, env, probe_log)
    if code:
        raise RuntimeError(
            f"prompt token probe failed with exit={code}; log={probe_log}"
        )
    n, source = infer_runtime_prompt_tokens(probe_log)
    if n <= 2:
        raise ValueError(f"prompt must contain more than 2 tokens: {prompt}")
    if n > args.batch or n > args.context:
        raise ValueError(
            f"actual prompt has {n} tokens, exceeds --batch={args.batch} "
            f"or --context={args.context}; increase limits"
        )
    return n, source


def launch_logged(command: list[str], log_path: Path, phase: str) -> None:
    print(f"[JUPITER_SUITE_PHASE] phase={phase} action=START "
          f"log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-18:]
        )
        raise RuntimeError(
            f"{phase} exited {result.returncode}; log={log_path}\n{tail}"
        )
    print(f"[JUPITER_SUITE_PHASE] phase={phase} action=DONE", flush=True)


def read_benchmark_results(
    csv_path: Path, total: int, modes: list[str], repeats: int
) -> dict[str, float | int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if not samples:
        raise ValueError(f"empty Prefill benchmark: {csv_path}")
    medians: dict[str, float] = {}
    for mode in modes:
        rows = [row for row in samples if row["mode"] == mode]
        runs = {int(row["repeat"]) for row in rows}
        if len(rows) != repeats or runs != set(range(1, repeats + 1)):
            raise ValueError(
                f"mode={mode} has {len(rows)} runs; expected {repeats}"
            )
        if any(int(row["tokens"]) != total for row in rows):
            raise ValueError(f"actual token count mismatch for mode={mode}")
        medians[mode] = statistics.median(float(row["prefill_ms"]) for row in rows)
    result: dict[str, float | int] = {"sample_count": repeats}
    for mode, ms in medians.items():
        result[f"{mode}_median_ms"] = round(ms, 3)
        result[f"{mode}_tok_s"] = round(total * 1000.0 / ms, 3)
    if "default" in medians and "dp" in medians:
        result["speedup_vs_default"] = round(
            medians["default"] / medians["dp"], 4
        )
    return result


def save_suite(out: Path, rows: list[dict], identity: dict) -> None:
    with (out / "suite-results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    (out / "suite-summary.json").write_text(
        json.dumps(
            {"configuration": identity, "prompts": rows},
            indent=2, ensure_ascii=False
        ) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jupiter multi-prompt measured Sequence DP + real Prefill A/B"
    )
    parser.add_argument("--exe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rpc", required=True)
    parser.add_argument("--prompts-dir", type=Path, default=None)
    parser.add_argument("--prompt-file", action="append", default=[])
    parser.add_argument("--glob", default="*.txt")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-chunk-json", required=True,
                        help="Stage-2 hardware chunk calibration; reuse for all prompts")
    parser.add_argument("--min-chunk", type=int, default=None,
                        help="optional fixed b for length-isolated experiments")
    parser.add_argument("--pc-layers", type=int, default=None)
    parser.add_argument("--q-repeats", type=int, default=2)
    parser.add_argument("--refine-repeats", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--bench-repeats", type=int, default=7)
    parser.add_argument("--modes", default="default,fixed64,dp")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--verbosity", type=int, default=3)
    parser.add_argument("--out-dir", default="jupiter-prompt-suite")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from run_prefill_bench import modes_arg
    modes = modes_arg(args.modes)
    args.modes = modes
    source = Path(args.min_chunk_json).resolve()
    for key in ("exe", "model"):
        path = Path(getattr(args, key)).resolve()
        if not path.is_file():
            raise ValueError(f"missing {key}: {path}")
    if not source.is_file():
        raise ValueError(f"missing Stage-2 JSON: {source}")
    if not args.prompts_dir and not args.prompt_file:
        raise ValueError("provide --prompts-dir or at least one --prompt-file")
    if (args.q_repeats < 1 or args.refine_repeats < 1 or
            args.bench_repeats < 1 or args.max_rounds < 1 or
            args.ubatch < 1 or args.batch < args.ubatch or
            args.context < args.batch or args.window < 1):
        raise ValueError("invalid repeats or batch/context/ubatch/window settings")
    if "default" not in modes or "dp" not in modes:
        raise ValueError("formal suite must include default and dp")
    source_data = json.loads(source.read_text(encoding="utf-8-sig"))
    pc = int(source_data["pc_layers"])
    if args.pc_layers is not None and args.pc_layers != pc:
        raise ValueError("PC layer override does not match Stage-2 calibration")
    if int(source_data["max_chunk"]) > args.ubatch:
        raise ValueError("--ubatch smaller than Stage-2 maximum chunk")
    if args.min_chunk is not None and args.min_chunk <= 0:
        raise ValueError("--min-chunk must be positive")

    prompts = discover_prompts(args.prompts_dir, args.prompt_file,
                               args.glob, args.recursive)
    if not prompts:
        raise ValueError("no prompt files found")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    suite_config = {
        "model": input_identity(Path(args.model)),
        "exe": input_identity(Path(args.exe)),
        "rpc": args.rpc, "pc_layers": pc,
        "stage2_json": str(source), "stage2_sha256": sha256_bytes(source.read_bytes()),
        "fixed_min_chunk": args.min_chunk,
        "q_repeats": args.q_repeats,
        "refine_repeats": args.refine_repeats,
        "bench_repeats": args.bench_repeats,
        "batch": args.batch, "context": args.context,
        "ubatch": args.ubatch, "window": args.window,
        "threads": args.threads, "modes": modes,
        "profile_stage_for_benchmark": 0,
        "async_c2p_for_benchmark": 0,
        "timing_token_tolerance": 1,
    }
    if args.dry_run:
        for prompt in prompts:
            identity = case_identity(prompt, args, source, pc)
            print(f"[JUPITER_SUITE_DRY_RUN] prompt={prompt} "
                  f"dir={case_folder(out,prompt,identity)}")
        return 0

    rows: list[dict] = []
    failures = 0
    for index, prompt in enumerate(prompts, 1):
        identity = case_identity(prompt, args, source, pc)
        case = case_folder(out, prompt, identity)
        case.mkdir(parents=True, exist_ok=True)
        config_file = case / "configuration.json"
        if config_file.is_file():
            old = json.loads(config_file.read_text(encoding="utf-8"))
            if old != identity:
                raise ValueError(f"case configuration mismatch: {case}")
            if not args.resume:
                raise ValueError(f"existing case: {case}; pass --resume")
        else:
            config_file.write_text(
                json.dumps(identity, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8"
            )

        row: dict[str, Any] = {
            "prompt": str(prompt), "status": "STARTED", "pc_layers": pc,
            "case_dir": str(case), "final_plan": str(case / "final.plan"),
        }
        print(f"\n[JUPITER_SUITE_CASE] index={index}/{len(prompts)} "
              f"prompt={prompt.name} dir={case}", flush=True)
        try:
            actual, source_name = probe_tokens(
                args, prompt, pc, case / "token-probe.log", args.resume
            )
            row["actual_tokens"] = actual
            if actual > args.batch or actual > args.context:
                raise ValueError(f"tokens {actual} exceed batch/context")
            _, _, b, _, b_status = resolve_stage2(
                source_data, target_tokens=actual,
                fixed_min_chunk=args.min_chunk
            )
            row["min_chunk"] = b
            row["min_chunk_status"] = b_status
            print(f"[JUPITER_SUITE_TOKENIZED] prompt={prompt.name} "
                  f"actual_tokens={actual} source={source_name} "
                  f"min_chunk={b} status={b_status}", flush=True)

            plan = case / "final.plan"
            stage3_dir = case / "stage3"
            stage3_summary = stage3_dir / "sequence-summary.json"
            ready = False
            if args.resume and plan.is_file() and stage3_summary.is_file():
                summary = json.loads(stage3_summary.read_text(encoding="utf-8"))
                planned = read_plan(plan).get(actual)
                ready = bool(
                    summary.get("status") == "FULLY_MEASURED_AND_VERIFIED"
                    and summary.get("pc_layers") == pc
                    and summary.get("prompt_tokens") == actual
                    and summary.get("min_chunk") == b
                    and planned == summary.get("chunks")
                )
            if ready:
                print(f"[JUPITER_SUITE_PHASE] phase=STAGE3 "
                      f"action=REUSE plan={plan}", flush=True)
            else:
                pipeline = [
                    sys.executable, str(Path(__file__).resolve().parent /
                                      "run_sequence_pipeline.py"),
                    "--exe", str(Path(args.exe).resolve()),
                    "--model", str(Path(args.model).resolve()),
                    "--rpc", args.rpc, "--prompt-file", str(prompt),
                    "--min-chunk-json", str(source),
                    "--target-tokens", str(actual),
                    "--reuse-b-profile-for-new-length",
                    "--pc-layers", str(pc),
                    "--q-repeats", str(args.q_repeats),
                    "--refine-repeats", str(args.refine_repeats),
                    "--max-rounds", str(args.max_rounds),
                    "--batch", str(args.batch), "--context", str(args.context),
                    "--ubatch", str(args.ubatch), "--threads", str(args.threads),
                    "--out-dir", str(stage3_dir),
                    "--plan-output", str(plan), "--resume",
                ]
                if args.min_chunk is not None:
                    pipeline += ["--min-chunk", str(args.min_chunk)]
                launch_logged(pipeline, case / "stage3-run.log", "STAGE3")

            summary = json.loads(stage3_summary.read_text(encoding="utf-8"))
            if summary.get("status") != "FULLY_MEASURED_AND_VERIFIED":
                raise ValueError("Stage-3 result is not fully measured and verified")
            chunks = read_plan(plan).get(actual)
            if chunks != summary.get("chunks"):
                raise ValueError("final runtime plan differs from Stage-3 summary")
            row["dp_chunks"] = ",".join(map(str, chunks))
            row["chunk_count"] = len(chunks)

            bench = [
                sys.executable, str(Path(__file__).resolve().parent /
                                  "run_prefill_bench.py"),
                "--exe", str(Path(args.exe).resolve()),
                "--model", str(Path(args.model).resolve()),
                "--rpc", args.rpc, "--prompt-file", str(prompt),
                "--plan", str(plan), "--target-tokens", str(actual),
                "--pc-layers", str(pc), "--repeats", str(args.bench_repeats),
                "--modes", ",".join(modes), "--ubatch", str(args.ubatch),
                "--batch", str(args.batch), "--context", str(args.context),
                "--threads", str(args.threads), "--window", str(args.window),
                "--verbosity", str(args.verbosity),
                "--timing-token-tolerance", "1",
                "--out-dir", str(case / "bench"), "--resume",
            ]
            # Deliberately NO --async-c2p or --comm-timing here: all
            # policies run the same original Jupiter data plane.
            launch_logged(bench, case / "bench-run.log", "BENCH")
            data = read_benchmark_results(
                case / "bench" / "results.csv", actual, modes, args.bench_repeats
            )
            row.update(data)
            row["status"] = "PASS"
            print(f"[JUPITER_SUITE_RESULT] prompt={prompt.name} "
                  f"tokens={actual} dp_chunks={row['dp_chunks']} "
                  f"default_ms={row['default_median_ms']:.3f} "
                  f"dp_ms={row['dp_median_ms']:.3f} "
                  f"speedup={row['speedup_vs_default']:.4f}x",
                  flush=True)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            failures += 1
            row["status"] = "FAILED"
            row["error"] = str(exc)
            print(f"[JUPITER_SUITE_ERROR] prompt={prompt.name} "
                  f"reason={str(exc)[:500]}", file=sys.stderr, flush=True)
        rows.append(row)
        save_suite(out, rows, suite_config)
        if row["status"] != "PASS" and not args.continue_on_error:
            break

    print(f"[JUPITER_SUITE_DONE] passed={sum(r['status']=='PASS' for r in rows)} "
          f"failed={failures} total={len(prompts)} "
          f"results={out / 'suite-results.csv'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[JUPITER_SUITE_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[JUPITER_SUITE_FATAL] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
