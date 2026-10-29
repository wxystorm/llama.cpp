#!/usr/bin/env python3
"""Refine a Jupiter sequence DP plan by calibrating its missing q(x,prefix) points.

The initial sweep observes fixed-x chains and leaves many arbitrary DP cuts
unmeasured. A plan inferred from those holes is only a proposal. This command
calibrates the exact requested chunk at its exact prefix, merges only those
samples into the original CSV, and re-solves DP until all final chunks have
direct observations (or reports an incomplete candidate).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from jupiter_planner import (
    QProfile,
    _write_plan,
    sequence_missing_points,
    sequence_points,
    solve_sequence,
)
from run_q_sweep import (
    build_cli_command,
    reconstruct_plan_from_log,
    run_checked,
    split_feasible_tail,
    stream_process,
)


FIELDNAMES = ["x", "y", "latency_ms", "device", "input_ms", "compute_ms", "wall_ms"]


def target_partition(
    total: int,
    x: int,
    y: int,
    minimum: int,
    maximum: int,
) -> list[int]:
    """Create a legal full-prompt plan containing an EXACT (x,prefix=y).

    Prefix/suffix may be rebalanced to stay in [minimum,maximum]; the
    targeted chunk is never resized or moved. This is important because
    q(x,y) includes attention over exactly y previous prompt tokens.
    """
    if total <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid total or chunk bounds")
    if x < minimum or x > maximum or y < 0 or y + x > total:
        raise ValueError(f"unreachable q({x},{y}) for total={total}")

    def fill(span: int) -> list[int]:
        if span == 0:
            return []
        if span < minimum:
            raise ValueError(
                f"cannot position target q({x},{y}): adjacent span "
                f"{span} is smaller than minimum chunk {minimum}"
            )
        return split_feasible_tail(span, minimum, maximum)

    chunks = fill(y) + [x] + fill(total - y - x)
    points = sequence_points(chunks)
    if sum(chunks) != total or (x, y) not in points:
        raise RuntimeError("internal targeted plan reconstruction failed")
    return chunks


def read_profile(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in FIELDNAMES):
            raise ValueError(
                f"profile must contain {FIELDNAMES}; file={path}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"q profile contains no samples: {path}")
    return rows


def write_profile(rows: Sequence[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in FIELDNAMES})


def targeted_rows(
    sample_csv: Path,
    x: int,
    y: int,
) -> list[dict[str, str]]:
    raw_rows = read_profile(sample_csv)
    rows = [
        {name: row[name] for name in FIELDNAMES}
        for row in raw_rows
        if int(row["x"]) == x and int(row["y"]) == y
    ]
    devices = {row["device"] for row in rows}
    if len(devices) < 2:
        raise RuntimeError(
            f"q({x},{y}) expected at least two backend stage samples, "
            f"found {sorted(devices)} in {sample_csv}"
        )
    return rows


def measured_cost(
    profile: QProfile,
    chunks: Sequence[int],
    devices: int,
) -> float:
    stage_times = [profile.q(x, y) for x, y in sequence_points(chunks)]
    return sum(stage_times) + (devices - 1) * max(stage_times)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Target-calibrate unmeasured q(x,y) positions in Jupiter DP"
    )
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--profile", required=True, help="original q-profile-all.csv")
    p.add_argument("--plan-output", default="jupiter-refined.plan")
    p.add_argument("--out-dir", default="jupiter-q-refinement")
    p.add_argument("--target-tokens", type=int, default=None)
    p.add_argument("--pc-layers", type=int, default=25)
    p.add_argument("--min-chunk", type=int, default=32)
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--max-subsequences", type=int, default=None)
    p.add_argument("--metric", choices=["compute_ms", "wall_ms"], default="compute_ms")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--extra-arg", action="append", default=[],
        help="extra llama-cli arg; repeat once per argument",
    )
    return p


def main() -> int:
    args = get_parser().parse_args()
    if (
        args.min_chunk <= 0 or args.max_chunk < args.min_chunk
        or args.repeats < 1 or args.max_rounds < 1
        or args.batch > args.context or args.max_chunk > args.batch
    ):
        raise ValueError("invalid chunk bounds, batch/context or repeat/round count")

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    source_profile = Path(args.profile).resolve()
    out = Path(args.out_dir).resolve()
    plan_output = Path(args.plan_output).resolve()
    for name, path in [
        ("exe", exe), ("model", model), ("prompt", prompt_file),
        ("profile", source_profile),
    ]:
        if not path.is_file():
            raise ValueError(f"{name} file not found: {path}")

    prompt_text = prompt_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not prompt_text:
        raise ValueError("prompt file is empty")

    source_rows = read_profile(source_profile)
    inferred_total = max(int(row["x"]) + int(row["y"]) for row in source_rows)
    total = args.target_tokens or inferred_total
    if total <= 0 or total > args.batch or total > args.context:
        raise ValueError(f"target={total} exceeds batch/context or is invalid")
    if total != inferred_total:
        print(
            f"[JUPITER_REFINE_TOKEN_WARNING] profile_max={inferred_total} "
            f"requested={total}; runtime plan application will be checked"
        )

    out.mkdir(parents=True, exist_ok=True)
    merged_profile = out / "q-profile-refined.csv"
    state_file = out / "configuration.json"
    config = {
        "exe": str(exe), "exe_size": exe.stat().st_size,
        "exe_mtime_ns": exe.stat().st_mtime_ns,
        "model": str(model), "rpc": args.rpc,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "profile": str(source_profile),
        "profile_sha256": hashlib.sha256(source_profile.read_bytes()).hexdigest(),
        "total": total, "metric": args.metric,
        "pc_layers": args.pc_layers,
        "min_chunk": args.min_chunk, "max_chunk": args.max_chunk,
        "max_subsequences": args.max_subsequences,
        "repeats": args.repeats,
        "batch": args.batch, "context": args.context,
        "threads": args.threads, "extra_arg": args.extra_arg,
    }
    if state_file.exists():
        if not args.resume:
            raise RuntimeError(
                f"refinement output already exists: {out}; "
                "use a new --out-dir or pass --resume"
            )
        old = json.loads(state_file.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError(
                f"--resume configuration mismatch in {state_file}; "
                "use a new --out-dir for changed inputs"
            )
        if not merged_profile.exists():
            raise RuntimeError(
                f"missing saved refinement profile: {merged_profile}"
            )
        all_rows = read_profile(merged_profile)
    else:
        if args.resume and merged_profile.exists():
            raise RuntimeError(
                f"found {merged_profile} without a matching configuration file"
            )
        all_rows = source_rows
        write_profile(all_rows, merged_profile)
        state_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    env = os.environ.copy()
    env.update({
        "LLAMA_JUPITER_PREFILL": "1",
        "LLAMA_JUPITER_PC_LAYERS": str(args.pc_layers),
        "LLAMA_JUPITER_WINDOW": "1",
        "LLAMA_JUPITER_TRACE": "1",
        "LLAMA_JUPITER_PROFILE_STAGE": "1",
        "LLAMA_JUPITER_Q_SWEEP": "1",
        "LLAMA_JUPITER_LOGITS_TOP10": "0",
    })
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    cli = build_cli_command(args, exe, model, prompt_text)
    planner = Path(__file__).resolve().parent / "jupiter_planner.py"

    print(
        f"[JUPITER_REFINE_CONFIG] tokens={total} metric={args.metric} "
        f"repeats={args.repeats} max_rounds={args.max_rounds} "
        f"q_csv={merged_profile}",
        flush=True,
    )

    for iteration in range(1, args.max_rounds + 1):
        profile = QProfile(all_rows, metric=args.metric)
        solution = solve_sequence(
            profile, total, 2, args.min_chunk, args.max_chunk,
            args.max_subsequences,
        )
        missing = sequence_missing_points(profile, solution.chunks)
        print(
            f"[JUPITER_REFINE_CANDIDATE] round={iteration} "
            f"chunks={','.join(map(str, solution.chunks))} "
            f"predicted_ms={solution.predicted_ms:.3f} "
            f"measured_positions={len(solution.chunks)-len(missing)}/{len(solution.chunks)} "
            f"missing={';'.join(f'{x}:{y}' for x,y in missing) or 'NONE'}",
            flush=True,
        )
        if not missing:
            break

        for x, y in missing:
            chunks = target_partition(
                total, x, y, args.min_chunk, args.max_chunk
            )
            plan_str = ",".join(map(str, chunks))
            for repeat in range(1, args.repeats + 1):
                stem = f"q-x{x}-y{y}-r{repeat}"
                log_path = out / f"{stem}.log"
                sample_path = out / f"{stem}.csv"

                reused = False
                if args.resume and log_path.exists() and sample_path.exists():
                    actual, _ = reconstruct_plan_from_log(
                        log_path, total, args.min_chunk
                    )
                    if actual == chunks:
                        try:
                            samples = targeted_rows(sample_path, x, y)
                            reused = True
                        except (RuntimeError, ValueError):
                            reused = False

                if not reused:
                    print(
                        f"[JUPITER_REFINE_SAMPLE] target={x}:{y} "
                        f"repeat={repeat}/{args.repeats} plan={plan_str}",
                        flush=True,
                    )
                    run_env = env.copy()
                    run_env["LLAMA_JUPITER_SEQUENCE_PLAN"] = plan_str
                    rc = stream_process(cli, run_env, log_path)
                    if rc:
                        raise RuntimeError(
                            f"llama-cli failed (exit {rc}) for q({x},{y}): "
                            f"{log_path}"
                        )
                    actual, _ = reconstruct_plan_from_log(
                        log_path, total, args.min_chunk
                    )
                    if actual != chunks:
                        raise RuntimeError(
                            f"targeted sequence plan was not applied for "
                            f"q({x},{y}); expected={plan_str} "
                            f"actual={actual}; log={log_path}"
                        )
                    run_checked([
                        sys.executable, str(planner), "extract-q",
                        "--log", str(log_path),
                        "--output", str(sample_path),
                        "--metric", "compute_ms",
                        "--min-tokens", str(args.min_chunk),
                    ])
                    samples = targeted_rows(sample_path, x, y)

                # Save each repeat immediately, preserving useful partial
                # work even if the next full model load fails.
                all_rows.extend(samples)
                write_profile(all_rows, merged_profile)
                print(
                    f"[JUPITER_REFINE_MEASURED] x={x} y={y} "
                    f"repeat={repeat} stages={len(samples)} "
                    f"csv={merged_profile}",
                    flush=True,
                )

    profile = QProfile(all_rows, metric=args.metric)
    final = solve_sequence(
        profile, total, 2, args.min_chunk, args.max_chunk,
        args.max_subsequences,
    )
    missing = sequence_missing_points(profile, final.chunks)
    candidate_path = out / "candidate.plan"
    _write_plan(
        str(candidate_path),
        {
            "model.pc_layers": str(args.pc_layers),
            f"sequence.{total}": ",".join(map(str, final.chunks)),
        },
    )

    print(
        f"[JUPITER_REFINE_RESULT] tokens={total} metric={args.metric} "
        f"chunks={','.join(map(str, final.chunks))} "
        f"predicted_ms={final.predicted_ms:.3f} "
        f"missing={';'.join(f'{x}:{y}' for x,y in missing) or 'NONE'} "
        f"candidate={candidate_path}",
        flush=True,
    )
    if missing:
        print(
            f"[JUPITER_REFINE_INCOMPLETE] unmeasured={len(missing)} "
            f"profile={merged_profile} "
            "continue with --resume and a larger --max-rounds; "
            "the requested final plan was NOT overwritten",
            file=sys.stderr, flush=True,
        )
        return 2

    _write_plan(
        str(plan_output),
        {
            "model.pc_layers": str(args.pc_layers),
            f"sequence.{total}": ",".join(map(str, final.chunks)),
        },
    )
    print(
        f"[JUPITER_REFINE_DONE] status=FULLY_MEASURED "
        f"total={total} k={len(final.chunks)} "
        f"predicted_ms={final.predicted_ms:.3f} "
        f"plan={plan_output} profile={merged_profile}",
        flush=True,
    )
    print(
        "[JUPITER_REFINE_NOTICE] This is a measured-q candidate, not a "
        "verified wall-time speedup; compare against default using "
        "run_prefill_bench.py with PROFILE_STAGE=0.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[JUPITER_REFINE_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[JUPITER_REFINE_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
