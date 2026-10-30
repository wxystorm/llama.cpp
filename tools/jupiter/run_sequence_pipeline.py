#!/usr/bin/env python3
"""Stage 3 of Jupiter: q(x,y) sweep -> Sequence DP -> exact-q refinement.

Reuses the existing calibrated profiler and DP without changing llama.cpp.
Produces a complete runtime plan only when all final q(x,y) coordinates are
directly measured on both PC accelerator and Phone RPC; verifies the plan in
a separate normal (non-profiling) inference run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from jupiter_planner import QProfile, _read_csv, sequence_missing_points
from min_chunk_profile import partition_feasible, select_min_chunk
from refine_sequence_q import build_resume_config
from run_prefill_bench import PLAN_RE, build_cli, parse_timing, read_plan, run_one


def resolve_stage2(summary: dict) -> tuple[int, int, int, int, str]:
    """Recheck b against the full prompt, including old stage-2 reports.

    The first stage-2 script could report b=128 for a 357-token prompt even
    though no legal [128,128] sequence existed. Reuse its measured candidate
    medians instead of asking the user to repeat expensive device profiling.
    """
    pc_layers = int(summary["pc_layers"])
    total = int(summary["prompt_tokens"])
    max_chunk = int(summary["max_chunk"])
    threshold = float(summary["threshold"])
    if pc_layers <= 0 or total <= 0 or max_chunk <= 0:
        raise ValueError("invalid stage-2 model/prompt/chunk parameters")
    candidates = summary.get("candidates", [])
    if not candidates:
        raise ValueError("min-chunk.json has no measured candidates")
    rows = [
        {
            "status": "OK",
            "chunk_tokens": str(item["chunk_tokens"]),
            "pc_compute_ms": str(item["pc_compute_ms"]),
            "phone_compute_ms": str(item["phone_compute_ms"]),
        }
        for item in candidates if int(item.get("samples", 0)) > 0
    ]
    chosen, _all, status = select_min_chunk(
        rows, threshold=threshold, total_tokens=total, max_chunk=max_chunk
    )
    if not partition_feasible(total, chosen.tokens, max_chunk):
        raise ValueError("selected b is not feasible for the full prompt")
    return pc_layers, total, chosen.tokens, max_chunk, status


def first_chunk_candidates(total: int, minimum: int, maximum: int,
                           step: int = 16) -> list[int]:
    """Cover the actual legal first-chunk interval, not impossible x values.

    The end of a full-prompt partition must also satisfy the same b and
    ubatch bound. For total=357, b=96, max=128, returns 101,112,128.
    """
    if step <= 0 or not partition_feasible(total, minimum, maximum):
        raise ValueError("invalid search step or no legal full-prompt partition")
    feasible = [
        x for x in range(minimum, min(total, maximum) + 1)
        if x == total or partition_feasible(total - x, minimum, maximum)
    ]
    if not feasible:
        raise ValueError("no feasible first chunk for this prompt")
    aligned = range(((minimum + step - 1) // step) * step, maximum + 1, step)
    return sorted(set([feasible[0], feasible[-1]] +
                      [x for x in aligned if x in feasible]))


def read_complete_plan(path: Path, total: int, pc_layers: int,
                       minimum: int, maximum: int,
                       measured_profile: Path) -> list[int]:
    if not path.is_file() or not measured_profile.is_file():
        raise ValueError("final plan or refined q-profile is missing")
    model_value = None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("model.pc_layers="):
            model_value = int(line.split("=", 1)[1])
    if model_value != pc_layers:
        raise ValueError("final plan model boundary does not match profiling")
    chunks = read_plan(path).get(total)
    if (not chunks or sum(chunks) != total or
            any(not minimum <= x <= maximum for x in chunks)):
        raise ValueError("final plan has an invalid or missing sequence partition")
    profile = QProfile(_read_csv(str(measured_profile)), metric="compute_ms")
    missing = sequence_missing_points(profile, chunks)
    if missing:
        raise ValueError(f"final plan contains unmeasured q(x,y) points: {missing}")
    return chunks


def choose_refinement_workspace(
    out: Path, expected_config: dict
) -> tuple[Path, bool, str]:
    """Preserve stale refinements; resume only matching calibration inputs.

    The q sweep can regenerate a CSV at the same path with different timing
    samples, which changes profile_sha256. A plain --resume must reject
    that, but the orchestration layer can safely use a separate workspace
    for the new calibration without deleting the old targeted samples.
    """
    base = out / "refinement"

    def workspace_state(path: Path) -> str:
        if not path.exists():
            return "NEW"
        if not path.is_dir():
            raise ValueError(f"refinement workspace is not a directory: {path}")
        state_file = path / "configuration.json"
        if state_file.is_file():
            try:
                current = json.loads(state_file.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError(
                    f"cannot read refinement configuration {state_file}: {exc}"
                ) from exc
            if current == expected_config:
                return "RESUME"
        if not any(path.iterdir()):
            return "NEW"
        return "CONFLICT"

    state = workspace_state(base)
    if state == "RESUME":
        return base, True, "RESUME_MATCH"
    if state == "NEW":
        return base, False, "NEW"

    fingerprint = hashlib.sha256(
        json.dumps(expected_config, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()[:12]
    chosen = out / f"refinement-{fingerprint}"
    state = workspace_state(chosen)
    if state == "CONFLICT":
        raise ValueError(
            f"hash-specific refinement workspace also has incompatible data: "
            f"{chosen}; use a different --out-dir"
        )
    return chosen, state == "RESUME", (
        "ISOLATE_OLD_CONFIG_REUSE" if state == "RESUME"
        else "ISOLATE_OLD_CONFIG_NEW"
    )


def checked(command: list[str]) -> int:
    print("[JUPITER_STAGE3_TOOL] " +
          (subprocess.list2cmdline(command) if os.name == "nt"
           else " ".join(command)), flush=True)
    return subprocess.run(command, check=False).returncode


def verify_runtime(args: argparse.Namespace, plan: Path, chunks: list[int],
                   pc_layers: int, total: int, output: Path) -> float | None:
    """Verify exact plan application with profiling disabled (not formal A/B)."""
    log_path = output / "verify-runtime.log"
    prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig").rstrip("\r\n")
    env = os.environ.copy()
    env.update({
        "LLAMA_JUPITER_PREFILL": "1",
        "LLAMA_JUPITER_PC_LAYERS": str(pc_layers),
        "LLAMA_JUPITER_PLAN_FILE": str(plan),
        "LLAMA_JUPITER_TRACE": "1",
        "LLAMA_JUPITER_PROFILE_STAGE": "0",
        "LLAMA_JUPITER_ASYNC_C2P": "0",
        "LLAMA_JUPITER_TIMELINE": "0",
        "LLAMA_JUPITER_Q_SWEEP": "0",
        "LLAMA_JUPITER_WINDOW": "2",
        "LLAMA_JUPITER_LOGITS_TOP10": "0",
    })
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    command = build_cli(args, Path(args.exe).resolve(),
                        Path(args.model).resolve(), prompt, 4)
    run_one(command, env, log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    observed = [
        (int(m.group(1)), [int(x) for x in m.group(2).split(",")])
        for m in PLAN_RE.finditer(text)
    ]
    if (total, chunks) not in observed:
        raise ValueError(f"runtime did not apply final Jupiter plan: {log_path}")
    try:
        latency_ms, _tokens = parse_timing(log_path)
    except RuntimeError:
        latency_ms = None
    print(f"[JUPITER_STAGE3_VERIFY] status=PASS "
          f"chunks={','.join(map(str,chunks))} log={log_path}", flush=True)
    return latency_ms


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jupiter stage 3: measured Sequence DP")
    for key in ("exe", "model", "rpc", "prompt-file"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--min-chunk-json", default="jupiter-min-chunk/min-chunk.json")
    p.add_argument("--pc-layers", type=int, default=None,
                   help="override only when it equals stage-2 PC layer count")
    p.add_argument("--target-tokens", type=int, default=None)
    p.add_argument("--x-values", default=None,
                   help="comma-separated first-chunk seeds; auto if omitted")
    p.add_argument("--q-repeats", type=int, default=2)
    p.add_argument("--refine-repeats", type=int, default=2)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--ubatch", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out-dir", default="jupiter-sequence-stage3")
    p.add_argument("--plan-output", default="jupiter-stage3.plan")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-runtime-verify", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    source = Path(args.min_chunk_json).resolve()
    if not source.is_file():
        raise ValueError(f"stage-2 min-chunk JSON not found: {source}")
    summary = json.loads(source.read_text(encoding="utf-8-sig"))
    pc, total, b, max_chunk, status = resolve_stage2(summary)
    if args.pc_layers is not None and args.pc_layers != pc:
        raise ValueError("PC layer override mismatches calibrated stage-2 result")
    if args.target_tokens is not None and args.target_tokens != total:
        raise ValueError("target tokens mismatch stage-2 calibration")
    if (args.q_repeats < 1 or args.refine_repeats < 1 or args.max_rounds < 1 or
            args.ubatch < max_chunk or args.batch < args.ubatch or
            args.context < args.batch):
        raise ValueError("invalid repeat count, max-rounds or context/batch bounds")

    for name in ("exe", "model", "prompt_file"):
        if not Path(getattr(args, name)).is_file():
            raise ValueError(f"input file missing: {name}={getattr(args,name)}")

    candidates = first_chunk_candidates(total, b, max_chunk)
    if args.x_values:
        chosen = sorted(set(int(s) for s in args.x_values.split(",")))
        candidates = [x for x in chosen
                      if b <= x <= max_chunk and
                      (x == total or partition_feasible(total-x, b, max_chunk))]
        if not candidates:
            raise ValueError("no requested x-values admit a legal full-prompt partition")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    final_plan = Path(args.plan_output).resolve()
    script_dir = Path(__file__).resolve().parent
    q_out = out / "q-sweep"
    initial = out / "initial-dp.plan"

    old_b = int(summary["min_chunk"])
    if old_b != b:
        print(f"[JUPITER_STAGE3_B_CORRECTION] measured_b={old_b} "
              f"feasible_b={b} reason=FULL_PROMPT_PARTITION", flush=True)
    print(f"[JUPITER_STAGE3_CONFIG] pc_layers={pc} phone_layers=remaining "
          f"tokens={total} min_chunk={b} max_chunk={max_chunk} "
          f"status={status} x_values={','.join(map(str,candidates))}", flush=True)

    common = [
        "--exe", str(Path(args.exe).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--rpc", args.rpc,
        "--prompt-file", str(Path(args.prompt_file).resolve()),
        "--pc-layers", str(pc),
        "--batch", str(args.batch),
        "--context", str(args.context),
        "--threads", str(args.threads),
    ]
    q_command = [
        sys.executable, str(script_dir / "run_q_sweep.py"), *common,
        "--target-tokens", str(total),
        "--x-values", ",".join(map(str,candidates)),
        "--min-chunk", str(b), "--max-chunk", str(max_chunk),
        "--repeats", str(args.q_repeats), "--run-dp",
        "--plan-output", str(initial), "--out-dir", str(q_out),
    ]
    # Existing q-sweep runs already validate the exact sequence plan before
    # reuse. Keep measured q-profile bytes stable across Stage-3 restarts.
    q_command.append("--resume")
    if checked(q_command):
        raise RuntimeError("initial q-sweep / Sequence DP failed")

    profiles = list(q_out.glob(f"tokens-{total}-*/q-profile-all.csv"))
    if len(profiles) != 1:
        raise ValueError(f"expected one q profile, found {len(profiles)} in {q_out}")
    q_profile = profiles[0]
    prompt_text = Path(args.prompt_file).read_text(
        encoding="utf-8-sig"
    ).rstrip("\r\n")
    refine_args = argparse.Namespace(
        rpc=args.rpc,
        metric="compute_ms", pc_layers=pc,
        min_chunk=b, max_chunk=max_chunk,
        max_subsequences=None, repeats=args.refine_repeats,
        batch=args.batch, context=args.context,
        threads=args.threads, extra_arg=[],
    )
    expected_config = build_resume_config(
        refine_args,
        Path(args.exe).resolve(), Path(args.model).resolve(),
        prompt_text, q_profile.resolve(), total
    )
    refine_out, can_resume, workspace_action = choose_refinement_workspace(
        out, expected_config
    )
    print(
        f"[JUPITER_STAGE3_REFINEMENT_WORKSPACE] action={workspace_action} "
        f"path={refine_out} reuse={int(can_resume)}",
        flush=True,
    )
    refine_command = [
        sys.executable, str(script_dir / "refine_sequence_q.py"), *common,
        "--profile", str(q_profile),
        "--target-tokens", str(total),
        "--min-chunk", str(b), "--max-chunk", str(max_chunk),
        "--repeats", str(args.refine_repeats),
        "--max-rounds", str(args.max_rounds),
        "--metric", "compute_ms",
        "--out-dir", str(refine_out), "--plan-output", str(final_plan),
    ]
    if can_resume:
        refine_command.append("--resume")
    rc = checked(refine_command)
    if rc == 2:
        print("[JUPITER_STAGE3_INCOMPLETE] targeted q refinement needs more "
              "rounds; rerun with --resume --max-rounds N (same parameters)",
              file=sys.stderr, flush=True)
        return 2
    if rc:
        raise RuntimeError(f"targeted q refinement failed, exit={rc}")

    chunks = read_complete_plan(
        final_plan, total, pc, b, max_chunk,
        refine_out / "q-profile-refined.csv"
    )
    runtime_latency_ms = None
    if not args.skip_runtime_verify:
        runtime_latency_ms = verify_runtime(
            args, final_plan, chunks, pc, total, out
        )

    report = {
        "status": "FULLY_MEASURED_AND_VERIFIED" if not args.skip_runtime_verify
                  else "FULLY_MEASURED_NOT_RUNTIME_VERIFIED",
        "pc_layers": pc, "prompt_tokens": total, "min_chunk": b,
        "max_chunk": max_chunk, "min_chunk_status": status,
        "initial_b_from_json": old_b,
        "chunks": chunks, "q_points": len(chunks),
        "q_profile": str(refine_out / "q-profile-refined.csv"),
        "refinement_dir": str(refine_out),
        "plan": str(final_plan),
        "runtime_verification_ms": runtime_latency_ms,
        "note": "Calibration times are synchronized; verify-runtime is not a formal benchmark.",
    }
    report_file = out / "sequence-summary.json"
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("[JUPITER_STAGE3_DONE] status=" + report["status"] +
          f" plan={final_plan} chunks={','.join(map(str,chunks))} "
          f"report={report_file}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"[JUPITER_STAGE3_ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
