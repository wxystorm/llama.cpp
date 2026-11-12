#!/usr/bin/env python3
"""Batch-run Jupiter's measured Sequence DP on the ShareGPT40 prompt files.

For every selected prompt:
  1. Probe the *actual* llama.cpp prefill token count (not the filename).
  2. Run Stage 3 synchronized q(x,y) calibration, DP and exact-q refinement.
  3. Verify the selected runtime chunk plan.
  4. Benchmark ONLY the DP mode without calibration instrumentation.

The probe and calibration/verification runs are NOT scored as benchmarks.
This script never invokes the Default or Fixed64 formal benchmark policies.
The minimum chunk is an explicit experiment constraint, not a new Stage-2
hardware measurement. If 96 is infeasible for a short prompt, the highest
feasible minimum <=96 is used and recorded.
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
import struct
import sys

from jupiter_context import (context_for_group, context_for_tokens,
                             model_token_counts, tokenizer_path)
from min_chunk_profile import partition_feasible
from run_prefill_bench import read_plan
from run_prompt_suite import (
    input_identity,
    launch_logged,
    probe_tokens,
    read_benchmark_results,
    sha256_bytes,
)


FIELDS = [
    "group", "prompt", "actual_tokens", "context", "min_chunk", "max_chunk",
    "dp_chunks", "chunk_count", "prefill_ms", "prefill_tok_s",
    "bench_repeats", "graph_prebuild", "refinement_policy", "status", "case_dir",
    "final_plan", "error",
]
NAME_RE = re.compile(r"^prompt_(\d+)_(\d+)\.txt$", re.IGNORECASE)

# GGUF v2/v3 metadata is stored before the tensor data. Read only the
# architecture.block_count field rather than deriving layer count from the
# model filename or assuming the 14B model's 40 layers.
_GGUF_SCALAR_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
    10: 8, 11: 8, 12: 8,
}


def gguf_layer_count(model: Path) -> int:
    """Read *.block_count from GGUF metadata without loading weights."""
    def read_exact(stream, length: int) -> bytes:
        data = stream.read(length)
        if len(data) != length:
            raise ValueError(f"truncated GGUF metadata: {model}")
        return data

    def read_u64(stream) -> int:
        return struct.unpack("<Q", read_exact(stream, 8))[0]

    def skip_value(stream, field_type: int) -> None:
        if field_type in _GGUF_SCALAR_SIZES:
            read_exact(stream, _GGUF_SCALAR_SIZES[field_type])
        elif field_type == 8:  # STRING
            length = read_u64(stream)
            if length > model.stat().st_size - stream.tell():
                raise ValueError(f"invalid GGUF string length: {model}")
            stream.seek(length, 1)
        elif field_type == 9:  # ARRAY
            element_type = struct.unpack("<I", read_exact(stream, 4))[0]
            length = read_u64(stream)
            if element_type in _GGUF_SCALAR_SIZES:
                bytes_to_skip = length * _GGUF_SCALAR_SIZES[element_type]
                if bytes_to_skip > model.stat().st_size - stream.tell():
                    raise ValueError(f"invalid GGUF array length: {model}")
                stream.seek(bytes_to_skip, 1)
            elif element_type == 8:
                for _ in range(length):
                    skip_value(stream, element_type)
            else:
                raise ValueError(f"unsupported GGUF array metadata type {element_type}")
        else:
            raise ValueError(f"unsupported GGUF metadata type {field_type}")

    with model.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError(f"not a GGUF model: {model}")
        version = struct.unpack("<I", read_exact(stream, 4))[0]
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}: {model}")
        _tensor_count = read_u64(stream)
        metadata_count = read_u64(stream)
        for _ in range(metadata_count):
            key_length = read_u64(stream)
            if key_length > 4096:
                raise ValueError(f"invalid GGUF metadata key length: {model}")
            key = read_exact(stream, key_length).decode("utf-8")
            field_type = struct.unpack("<I", read_exact(stream, 4))[0]
            if key.endswith(".block_count"):
                if field_type not in (2, 4, 10):
                    raise ValueError(f"non-integer GGUF block_count ({field_type}): {model}")
                size = _GGUF_SCALAR_SIZES[field_type]
                count = int.from_bytes(read_exact(stream, size), "little")
                if not 1 <= count <= 4096:
                    raise ValueError(f"invalid GGUF block_count={count}: {model}")
                return count
            skip_value(stream, field_type)
    raise ValueError(f"missing architecture.block_count GGUF metadata: {model}")


def discover(folder: Path, groups: list[int]) -> list[tuple[int, Path]]:
    if not folder.is_dir():
        raise ValueError(f"prompt folder does not exist: {folder}")
    found = []
    for item in folder.iterdir():
        match = NAME_RE.fullmatch(item.name)
        if item.is_file() and match and int(match.group(1)) in groups:
            found.append((int(match.group(1)), int(match.group(2)), item.resolve()))
    found.sort(key=lambda item: (item[0], item[1], item[2].name.lower()))
    return [(group, path) for group, _, path in found]


def feasible_min_chunk(total: int, preferred: int, maximum: int) -> int:
    """Keep b as close to the calibrated 96-token choice as feasibility allows."""
    if total <= 0 or preferred <= 0 or maximum <= 0:
        raise ValueError("token/chunk parameters must be positive")
    for minimum in range(min(preferred, maximum, total), 0, -1):
        if partition_feasible(total, minimum, maximum):
            return minimum
    raise ValueError(f"no legal partition for N={total} max_chunk={maximum}")


def case_args_for_prompt(args: argparse.Namespace, group: int, prompt: Path,
                         counts: dict[str, int]) -> argparse.Namespace:
    """Select the actual context and valid effective batch for a single case."""
    case_args = argparse.Namespace(**vars(args))
    if getattr(args, "group_context", False):
        case_args.context = context_for_group(group)
    elif getattr(args, "auto_context", False):
        token_count = counts[prompt.name.lower()]
        case_args.context = context_for_tokens(
            token_count, minimum=args.min_context, padding=args.context_padding)
    if case_args.context is None or case_args.context < args.ubatch:
        raise ValueError(f"invalid context for {prompt.name}: {case_args.context}")
    # Jupiter requires ubatch <= batch <= context; keep 2048 for long prompts.
    case_args.batch = min(args.batch, case_args.context)
    return case_args


def case_key(group: int, prompt: Path, args: argparse.Namespace) -> dict:
    identity = {
        "group": group, "prompt": str(prompt),
        "prompt_sha256": sha256_bytes(prompt.read_bytes()),
        "exe": input_identity(Path(args.exe)),
        "model": input_identity(Path(args.model)),
        "rpc": args.rpc, "pc_layers": args.pc_layers,
        "preferred_min_chunk": args.min_chunk,
        "max_chunk": args.max_chunk, "ubatch": args.ubatch,
        "batch": args.batch, "context": args.context,
        "threads": args.threads, "window": args.window,
        "q_repeats": args.q_repeats,
        "refine_repeats": args.refine_repeats,
        "graph_prebuild": args.graph_prebuild,
    }
    # HTP session topology and budgets change model placement; a PASS
    # from an earlier 1-session benchmark must never be reused for 4.
    if args.htp_sessions > 1:
        identity.update({
            "htp_sessions": args.htp_sessions,
            "htp_global_budget_mib": args.htp_global_budget_mib,
            "htp_session_vmem_mib": args.htp_session_vmem_mib,
            "htp_reserve_mib": args.htp_reserve_mib,
            "htp_kv_element_bytes": args.htp_kv_element_bytes,
        })
    # Preserve legacy two-device case hashes for CPU=0 resumes.
    if args.cpu_layers:
        identity["cpu_layers"] = args.cpu_layers
    return identity


def existing_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read saved results before rewriting the cumulative CSV."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["prompt"]: row for row in csv.DictReader(handle)
                if row.get("prompt")}


def save_results(out: Path, rows: list[dict], args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dp-results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    by_group: dict[int, list[dict]] = {}
    for row in rows:
        by_group.setdefault(int(row["group"]), []).append(row)
    group_rows: list[dict] = []
    for group, values in sorted(by_group.items()):
        passed = [x for x in values if x["status"] == "PASS"]
        group_rows.append({
            "group": group,
            "completed": len(values),
            "passed": len(passed),
            "failed": len(values) - len(passed),
            "median_actual_tokens": (
                statistics.median(int(x["actual_tokens"]) for x in passed)
                if passed else ""
            ),
            "median_prefill_ms": (
                round(statistics.median(float(x["prefill_ms"]) for x in passed), 3)
                if passed else ""
            ),
            "median_tok_s": (
                round(statistics.median(float(x["prefill_tok_s"]) for x in passed), 3)
                if passed else ""
            ),
        })
    with (out / "dp-groups.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "group", "completed", "passed", "failed",
            "median_actual_tokens", "median_prefill_ms", "median_tok_s",
        ])
        writer.writeheader()
        writer.writerows(group_rows)

    (out / "dp-summary.json").write_text(
        json.dumps({
            "experiment": "Jupiter Sequence DP only",
            "prompts_dir": str(Path(args.prompts_dir).resolve()),
            "benchmark_repeats": args.bench_repeats,
            "graph_prebuild": args.graph_prebuild,
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stage3_ready(stage3: Path, plan: Path, actual: int, b: int,
                 args: argparse.Namespace) -> bool:
    summary_path = stage3 / "sequence-summary.json"
    if not plan.is_file() or not summary_path.is_file():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        chunks = read_plan(plan).get(actual)
        return bool(
            data.get("status") in (
                "FULLY_MEASURED_AND_VERIFIED",
                "BUDGETED_MEASURED_AND_VERIFIED",
            )
            and data.get("prompt_tokens") == actual
            and data.get("pc_layers") == args.pc_layers
            and data.get("cpu_layers", 0) == getattr(args, "cpu_layers", 0)
            and data.get("min_chunk") == b
            and data.get("max_chunk") == args.max_chunk
            and data.get("setup_source") == "FIXED_EXPLICIT"
            and chunks and chunks == data.get("chunks")
        )
    except (ValueError, OSError, KeyError, TypeError):
        return False


def benchmark_ready(bench: Path, prompt: Path, plan: Path, actual: int,
                    chunks: list[int], args: argparse.Namespace) -> dict | None:
    manifest = bench / "configuration.json"
    result_csv = bench / "results.csv"
    if not manifest.is_file() or not result_csv.is_file():
        return None
    try:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        text = prompt.read_text(encoding="utf-8-sig").rstrip("\r\n")
        required = {
            "prompt_sha256": sha256_bytes(text.encode("utf-8")),
            "exe": str(Path(args.exe).resolve()),
            "model": str(Path(args.model).resolve()),
            "target_tokens": actual, "dp_chunks": chunks,
            "pc_layers": args.pc_layers,
            "modes": ["dp"],
            "repeats": args.bench_repeats, "ubatch": args.ubatch,
            "batch": args.batch, "context": args.context,
            "window": args.window, "graph_prebuild": args.graph_prebuild,
            "rpc": args.rpc, "plan": str(plan.resolve()),
        }
        if getattr(args, "cpu_layers", 0):
            required["cpu_layers"] = args.cpu_layers
        if any(stored.get(key) != value for key, value in required.items()):
            return None
        return read_benchmark_results(result_csv, actual, ["dp"], args.bench_repeats)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run_case(group: int, prompt: Path, root: Path, args: argparse.Namespace) -> dict:
    identity = case_key(group, prompt, args)
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    case = root / "cases" / f"{prompt.stem}-{fingerprint}"
    case.mkdir(parents=True, exist_ok=True)
    config = case / "configuration.json"
    if config.is_file():
        saved = json.loads(config.read_text(encoding="utf-8"))
        if saved != identity:
            raise ValueError(f"case identity conflict: {case}")
        if not args.resume:
            raise ValueError(f"case exists: {case}; rerun with --resume")
    else:
        config.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    row = {
        "group": group, "prompt": prompt.name, "status": "STARTED",
        "graph_prebuild": int(args.graph_prebuild),
        "bench_repeats": args.bench_repeats,
        "refinement_policy": "PENDING",
        "case_dir": str(case), "final_plan": str(case / "final.plan"),
    }
    try:
        row["context"] = args.context
        tokens, source = probe_tokens(
            args, prompt, args.pc_layers, case / "token-probe.log", args.resume
        )
        row["actual_tokens"] = tokens
        if tokens > args.batch or tokens > args.context:
            raise ValueError(f"N={tokens} exceeds batch/context ({args.batch}/{args.context})")
        if (getattr(args, "auto_context", False)
                or getattr(args, "group_context", False)):
            required = tokens + 1 + args.context_padding
            if required > args.context:
                policy = ("GGUF auto" if getattr(args, "auto_context", False)
                          else f"group={group} reference")
                raise ValueError(
                    f"runtime probe N={tokens} needs context >= {required} "
                    f"with {args.context_padding} safety tokens, but "
                    f"{policy} context is {args.context}. "
                    "Use --context N or --context-policy auto; "
                    "do not silently truncate the prompt.")
        minimum = feasible_min_chunk(tokens, args.min_chunk, args.max_chunk)
        row["min_chunk"] = minimum
        row["max_chunk"] = args.max_chunk
        print(f"[JUPITER_SHAREGPT40_DP_TOKENS] prompt={prompt.name} "
              f"actual={tokens} source={source} b={minimum} max={args.max_chunk}",
              flush=True)
        stage3 = case / "stage3"
        plan = case / "final.plan"
        if args.resume and stage3_ready(stage3, plan, tokens, minimum, args):
            print(f"[JUPITER_SHAREGPT40_DP_STAGE3] action=REUSE plan={plan}",
                  flush=True)
        else:
            cmd = [
                sys.executable, str(Path(__file__).with_name("run_sequence_pipeline.py")),
                "--exe", str(Path(args.exe).resolve()),
                "--model", str(Path(args.model).resolve()),
                "--rpc", args.rpc, "--prompt-file", str(prompt),
                "--fixed-stage3", "--pc-layers", str(args.pc_layers),
                "--cpu-layers", str(args.cpu_layers),
                "--target-tokens", str(tokens),
                "--min-chunk", str(minimum), "--max-chunk", str(args.max_chunk),
                "--q-repeats", str(args.q_repeats),
                "--refine-repeats", str(args.refine_repeats),
                "--max-rounds", str(args.max_rounds),
                "--batch", str(args.batch), "--context", str(args.context),
                "--ubatch", str(args.ubatch), "--threads", str(args.threads),
                "--out-dir", str(stage3), "--plan-output", str(plan),
                "--resume",
            ]
            if args.budgeted_refine:
                cmd.append("--accept-measured-candidate")
            launch_logged(cmd, case / "stage3-run.log", "STAGE3")
        if not stage3_ready(stage3, plan, tokens, minimum, args):
            raise RuntimeError(f"Stage3 did not verify a complete DP plan: {stage3}")
        summary = json.loads(
            (stage3 / "sequence-summary.json").read_text(encoding="utf-8")
        )
        row["refinement_policy"] = summary.get(
            "refinement_policy", "REOPTIMIZE_UNTIL_EXACT_Q_COVERAGE"
        )
        chunks = read_plan(plan)[tokens]
        row["dp_chunks"] = ",".join(map(str, chunks))
        row["chunk_count"] = len(chunks)

        bench_dir = case / "bench"
        measurements = (
            benchmark_ready(bench_dir, prompt, plan, tokens, chunks, args)
            if args.resume else None
        )
        if measurements is None:
            cmd = [
                sys.executable, str(Path(__file__).with_name("run_prefill_bench.py")),
                "--exe", str(Path(args.exe).resolve()),
                "--model", str(Path(args.model).resolve()),
                "--rpc", args.rpc, "--prompt-file", str(prompt),
                "--plan", str(plan), "--target-tokens", str(tokens),
                "--pc-layers", str(args.pc_layers),
                "--cpu-layers", str(args.cpu_layers), "--modes", "dp",
                "--repeats", str(args.bench_repeats),
                "--batch", str(args.batch), "--context", str(args.context),
                "--ubatch", str(args.ubatch), "--threads", str(args.threads),
                "--window", str(args.window),
                "--timing-token-tolerance", "1", "--out-dir", str(bench_dir),
            ]
            if args.graph_prebuild:
                cmd.append("--graph-prebuild")
            # Resume partial bench measurements; the inner runner validates
            # the configuration before accepting any existing timing log.
            if args.resume:
                cmd.append("--resume")
            launch_logged(cmd, case / "bench-run.log", "DP_BENCH")
            measurements = read_benchmark_results(
                bench_dir / "results.csv", tokens, ["dp"], args.bench_repeats
            )
        else:
            print(f"[JUPITER_SHAREGPT40_DP_BENCH] action=REUSE "
                  f"prompt={prompt.name}", flush=True)
        row["prefill_ms"] = measurements["dp_median_ms"]
        row["prefill_tok_s"] = measurements["dp_tok_s"]
        row["status"] = "PASS"
        print(f"[JUPITER_SHAREGPT40_DP_RESULT] prompt={prompt.name} "
              f"N={tokens} chunks={row['dp_chunks']} "
              f"prefill_ms={row['prefill_ms']:.3f} "
              f"tok_s={row['prefill_tok_s']:.3f} status=PASS", flush=True)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        row["status"] = "FAIL"
        row["error"] = str(exc)
        print(f"[JUPITER_SHAREGPT40_DP_ERROR] prompt={prompt.name} "
              f"error={str(exc)[:700]}", file=sys.stderr, flush=True)
    return row


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts-dir", default=r"E:\llama\datasets\sharegpt\sharegpt40")
    p.add_argument("--groups", default="128,256,512,1024")
    p.add_argument("--exe", default=r".\build-jupiter\bin\Release\llama-cli.exe")
    p.add_argument("--model", default=r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf")
    p.add_argument("--rpc", default="10.39.52.114:10139")
    p.add_argument("--pc-layers", type=int, default=32,
                   help="total CPU + CUDA layers on PC; remaining layers go to Phone")
    p.add_argument("--model-layers", type=int, default=None,
                   help="override GGUF architecture.block_count; normally read automatically")
    p.add_argument("--cpu-layers", type=int, default=0)
    p.add_argument("--htp-sessions", type=int, choices=(1, 2, 3, 4), default=1,
                   help="HTP FastRPC sessions for contiguous Phone layer placement; "
                        "default 1 preserves the original Jupiter baseline")
    p.add_argument("--htp-global-budget-mib", type=int, default=8192,
                   help="global Phone budget across active HTP sessions (MiB)")
    p.add_argument("--htp-session-vmem-mib", type=int, default=3072,
                   help="per-session VMEM allocation budget (MiB)")
    p.add_argument("--htp-reserve-mib", type=int, default=512,
                   help="reserve per session for KV/graphs/activation overhead (MiB)")
    p.add_argument("--htp-kv-element-bytes", type=int, choices=(1, 2, 4), default=2,
                   help="bytes per K or V element for conservative KV planning")
    p.add_argument("--min-chunk", type=int, default=96,
                   help="preferred lower bound b; reduce only if no legal partition")
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--context", type=int, default=None,
                   help="fixed context override (takes precedence over context-policy)")
    p.add_argument("--context-policy", choices=("group", "auto"), default="group",
                   help="group (default): 128=512, 256=768, 512=1280, "
                        "1024=2304, matching reference logs; "
                        "auto: tokenize each prompt and calculate context")
    p.add_argument("--min-context", type=int, default=1024)
    p.add_argument("--context-padding", type=int, default=64)
    p.add_argument("--tokenizer-exe", default=None,
                   help="llama-tokenize path; default: next to --exe")
    p.add_argument("--ubatch", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--q-repeats", type=int, default=1)
    p.add_argument("--refine-repeats", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--budgeted-refine", action="store_true",
                   help="use the last measured DP candidate after the "
                        "refinement round budget; do not claim fixed-point "
                        "convergence")
    p.add_argument("--only-incomplete", action="store_true",
                   help="skip existing PASS records and preserve them "
                        "in cumulative CSV")
    p.add_argument("--bench-repeats", type=int, default=1)
    p.add_argument("--graph-prebuild", action="store_true",
                   help="opt in to the existing PC/Phone graph-prebuild path")
    p.add_argument("--out-dir", default="jupiter-sharegpt40-dp")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N prompts (smoke test)")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if (args.pc_layers <= 0 or args.cpu_layers < 0
            or args.cpu_layers >= args.pc_layers or args.min_chunk <= 0 or args.max_chunk <= 0
            or args.min_chunk > args.max_chunk
            or args.max_chunk > args.ubatch or args.ubatch > args.batch
            or (args.context is not None and args.context < args.ubatch)
            or args.min_context < args.ubatch or args.context_padding < 0
            or args.q_repeats < 1
            or args.refine_repeats < 1 or args.bench_repeats < 1
            or args.max_rounds < 1 or args.window < 1
            or args.htp_global_budget_mib < 1
            or args.htp_session_vmem_mib < 1
            or args.htp_reserve_mib < 1
            or args.htp_session_vmem_mib <= args.htp_reserve_mib):
        raise ValueError("invalid devices/chunk/batch/repeats configuration")
    try:
        groups = [int(x.strip()) for x in args.groups.split(",")]
    except ValueError as exc:
        raise ValueError("--groups must be comma-separated integers") from exc
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("--groups must contain distinct integers")
    if args.context is None and args.context_policy == "group":
        for group in groups:
            context_for_group(group)  # fail early if an unsupported group is requested
    prompts = discover(Path(args.prompts_dir), groups)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        prompts = prompts[:args.limit]
    if not prompts:
        raise ValueError("no matching prompt_NN_XX.txt files found")

    # Child stages (token probe, q-profile, DP verification and Prefill)
    # inherit one placement policy with the actual model's repeating-layer
    # count, not the Qwen3-14B-only constant of 40.
    if args.model_layers is None and Path(args.model).is_file():
        args.model_layers = gguf_layer_count(Path(args.model))
    if args.model_layers is not None and (
            args.model_layers < 2 or not 0 < args.pc_layers < args.model_layers):
        raise ValueError(
            f"invalid --pc-layers {args.pc_layers} for "
            f"--model-layers {args.model_layers}; leave Phone at least one layer")
    os.environ["LLAMA_JUPITER_HTP_SESSIONS"] = str(args.htp_sessions)
    active_sessions = (min(args.htp_sessions, args.model_layers - args.pc_layers)
                       if args.model_layers is not None else 0)
    if not args.dry_run and active_sessions < 1:
        raise ValueError("--model must exist or --model-layers must be supplied")
    if active_sessions > 0:
        os.environ["LLAMA_JUPITER_HTP_ACTIVE_SESSIONS"] = str(active_sessions)
    else:
        os.environ.pop("LLAMA_JUPITER_HTP_ACTIVE_SESSIONS", None)
    os.environ["LLAMA_JUPITER_HTP_GLOBAL_BUDGET_MIB"] = str(args.htp_global_budget_mib)
    os.environ["LLAMA_JUPITER_HTP_SESSION_VMEM_MIB"] = str(args.htp_session_vmem_mib)
    os.environ["LLAMA_JUPITER_HTP_RESERVE_MIB"] = str(args.htp_reserve_mib)
    os.environ["LLAMA_JUPITER_HTP_KV_ELEMENT_BYTES"] = str(args.htp_kv_element_bytes)
    for group in groups:
        count = sum(1 for g, _ in prompts if g == group)
        print(f"[JUPITER_SHAREGPT40_DP_GROUP] group={group} prompts={count}",
              flush=True)
    print(f"[JUPITER_SHAREGPT40_DP_CONFIG] files={len(prompts)} "
          f"modes=dp q_repeats={args.q_repeats} "
          f"refine_repeats={args.refine_repeats} "
          f"bench_repeats={args.bench_repeats} "
          f"graph_prebuild={int(args.graph_prebuild)} "
          f"context_policy={('FIXED' if args.context is not None else args.context_policy.upper())} "
          f"n_predict=1 model_layers={args.model_layers or 'UNKNOWN'} "
          f"pc_layers={args.pc_layers} cpu_layers={args.cpu_layers} "
          f"htp_sessions={args.htp_sessions} htp_global_budget_mib={args.htp_global_budget_mib} "
          f"htp_session_vmem_mib={args.htp_session_vmem_mib}",
          flush=True)

    if args.dry_run:
        for group, prompt in prompts:
            print(f"[JUPITER_SHAREGPT40_DP_FILE] group={group} path={prompt}")
        return 0
    for key in ("exe", "model"):
        if not Path(getattr(args, key)).is_file():
            raise ValueError(f"{key} not found: {getattr(args, key)}")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Count with the same GGUF tokenizer used by Region Ring. The actual
    # Jupiter prefill probe below remains the authoritative DP token count.
    args.auto_context = args.context is None and args.context_policy == "auto"
    args.group_context = args.context is None and args.context_policy == "group"
    counts = (model_token_counts(
        args.model, tokenizer_path(args.exe, args.tokenizer_exe),
        [prompt for _, prompt in prompts], out / "prompt-tokens.json")
        if args.auto_context else {})
    old_rows = existing_rows(out / "dp-results.csv") if args.only_incomplete else {}
    rows: list[dict] = []
    for index, (group, prompt) in enumerate(prompts, 1):
        case_args = case_args_for_prompt(args, group, prompt, counts)
        os.environ["LLAMA_JUPITER_HTP_CONTEXT_TOKENS"] = str(case_args.context)
        detail = (f" gguf_tokens={counts[prompt.name.lower()]}"
                  if args.auto_context else "")
        print(f"[JUPITER_CONTEXT] prompt={prompt.name} group={group} "
              f"policy={('FIXED' if args.context is not None else args.context_policy.upper())}"
              f"{detail} context={case_args.context} batch={case_args.batch} "
              f"ubatch={case_args.ubatch} predict=1",
              flush=True)
        previous = old_rows.get(prompt.name)
        if previous and previous.get("status") == "PASS":
            # CSV alone is insufficient: do not skip a PASS measured with
            # another context/batch/prompt hash or executable.
            saved_case = Path(previous.get("case_dir") or "")
            config_path = saved_case / "configuration.json"
            try:
                stored_identity = json.loads(config_path.read_text(encoding="utf-8"))
                compatible = stored_identity == case_key(group, prompt, case_args)
            except (OSError, ValueError, TypeError):
                compatible = False
            if compatible:
                rows.append(previous)
                print(f"[JUPITER_SHAREGPT40_DP_SKIP] {index}/{len(prompts)} "
                      f"prompt={prompt.name} reason=VERIFIED_EXISTING_PASS",
                      flush=True)
                continue
        print(f"\n[JUPITER_SHAREGPT40_DP_CASE] {index}/{len(prompts)} "
              f"group={group} prompt={prompt.name}", flush=True)
        row = run_case(group, prompt, out, case_args)
        rows.append(row)
        save_results(out, rows, args)
        if row["status"] != "PASS" and args.fail_fast:
            break
    save_results(out, rows, args)
    passed = sum(row["status"] == "PASS" for row in rows)
    failed = len(rows) - passed
    print(f"[JUPITER_SHAREGPT40_DP_DONE] passed={passed} failed={failed} "
          f"processed={len(rows)}/{len(prompts)} "
          f"results={out / 'dp-results.csv'} "
          f"groups={out / 'dp-groups.csv'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[JUPITER_SHAREGPT40_DP_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[JUPITER_SHAREGPT40_DP_FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
