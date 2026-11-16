#!/usr/bin/env python3
"""Resumable Qwen3-32B Runtime ablations with a frozen layer placement.

Place next to run_32b_all40.py in tools/region-ring-experiments/.

Four cases (each uses the same five ShareGPT groups by default):
  full          G8,P6,G8,P6,G8,P6,G8,P6,G8; two banks, prefetch, drain-first
  no_drain      same plan; FIFO instead of drain-first
  no_prefetch   same plan; two banks, synchronous refill
  no_multicycle G8,G8,G8,G8,P24,G8; two banks, prefetch, drain-first

Unlike --force-plan-layout, this wrapper supplies the C++ native LOCKED_LAYOUT
chunk-only path, so each request can choose XC/XG/XP without repartitioning
model weights. The no_multicycle case requires the accompanying opt-in C++
header patch and a rebuilt PC executable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_32b_all40 as base  # noqa: E402

FULL_LAYOUT = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
SINGLE_PASS_LAYOUT = "G8,G8,G8,G8,P24,G8"
GROUPS = (128, 256, 512, 1024, 2048)
CASE_ORDER = ("full", "no_drain", "no_prefetch", "no_multicycle")

CASE_SETTINGS = {
    "full":          (FULL_LAYOUT, "1", "1", "0"),
    "no_drain":      (FULL_LAYOUT, "1", "0", "0"),
    "no_prefetch":   (FULL_LAYOUT, "0", "1", "0"),
    "no_multicycle": (SINGLE_PASS_LAYOUT, "1", "1", "1"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_layout(layout: str) -> None:
    entries = [(p[0], int(p[1:])) for p in layout.split(",")]
    if sum(depth for _, depth in entries) != 64:
        raise ValueError("Layout must cover all 64 layers")
    if sum(depth for kind, depth in entries if kind == "G") != 40:
        raise ValueError("Layout must have 40 GPU layers")
    if sum(depth for kind, depth in entries if kind == "P") != 24:
        raise ValueError("Layout must have 24 HTP layers")
    if any(depth <= 0 for _, depth in entries):
        raise ValueError("Layout must not contain zero/negative regions")
    if any(depth != 8 for kind, depth in entries if kind == "G"):
        raise ValueError("Every GPU region must have exactly eight layers")


def require_native_patch(exe: Path, skip_build_check: bool) -> None:
    header = ROOT / "src" / "llama-region-ring.h"
    if not header.is_file() or "LLAMA_REGION_RING_ALLOW_ADJACENT_GPU" not in header.read_text(
            encoding="utf-8"):
        raise RuntimeError(
            "Adjacent GPU layout support is missing from this checkout. "
            "Switch to feat/32b-runtime-ablation-singlepass-20261116, "
            "rebuild llama-completion.exe, and retry.")
    if not skip_build_check and exe.is_file() and exe.stat().st_mtime < header.stat().st_mtime:
        raise RuntimeError(
            "The executable predates the patched header. Rebuild the PC "
            "llama-completion.exe or use --skip-build-check if timestamps "
            "are unreliable (this cannot guarantee the patch is compiled).")


def copy_profile(source: Path, case_out: Path, current_groups: set[int]) -> None:
    source = source.resolve()
    target = case_out / "independent-profile" / "profile.csv"
    metadata = source.with_suffix(".json")
    if not source.is_file() or not metadata.is_file():
        raise ValueError(f"Profile requires matching .csv and .json: {source}")
    # No guessed HTP data: verify that these are native G/C/N samples.
    required_kvs = base.LONG_HTP_KV_ANCHORS if 2048 in current_groups else ()
    base.verify_independent_profile(source, required_kvs)
    target.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in ((source, target), (metadata, target.with_suffix(".json"))):
        if dst.exists():
            if sha256(src) != sha256(dst):
                raise ValueError(
                    f"Existing profile differs: {dst}. Use a new --out or "
                    "an identical profile, never silently overwrite an epoch.")
        else:
            shutil.copy2(src, dst)


def runner_argv(args: argparse.Namespace, case: str, limit: int) -> list[str]:
    _, prefetch, drain, adjacent = CASE_SETTINGS[case]
    result = [
        "--exe", str(args.exe),
        "--model", str(args.model),
        "--model-layers", "64",
        "--rpc", args.rpc,
        "--repo-root", str(ROOT),
        "--prompt-dir", str(args.prompt_dir),
        "--out", str(args.out / case),
        "--auto-plan", "--htp-only", "--streaming", "on",
        "--profile-grid", args.profile_grid,
        "--batch", str(args.batch), "--ubatch", str(args.ubatch),
        "--n-predict", "1", "--threads", str(args.threads),
        "--min-context", str(args.min_context),
        "--max-context", str(args.max_context),
        "--context-padding", str(args.context_padding),
        "--max-retries", str(args.max_retries),
        "--retry-delay", str(args.retry_delay),
        "--inter-prompt-delay-seconds", str(args.inter_prompt_delay_seconds),
        "--timeout", str(args.timeout),
        "--log-verbosity", "4", "--no-mmap", "--limit", str(limit),
        "--env", "LLAMA_REGION_RING_GPU_BANK_COUNT=2",
        "--env", f"LLAMA_REGION_RING_GPU_BANK_PREFETCH={prefetch}",
        "--env", f"LLAMA_REGION_RING_DRAIN_FIRST_CYCLE={drain}",
        "--env", "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=1",
        "--env", f"LLAMA_REGION_RING_ALLOW_ADJACENT_GPU={adjacent}",
    ]
    for group in args.groups:
        result += ["--only-group", str(group)]
    for prompt in args.only_prompt:
        result += ["--only-prompt", prompt]
    if args.trace_tasks:
        result.append("--trace-tasks")
    if args.boost_bank_mover:
        result.append("--boost-bank-mover")
    if args.rerun_all:
        result.append("--rerun-all")
    if args.retry_failed_only:
        result.append("--retry-failed-only")
    if args.dry_run:
        result.append("--dry-run")
    return result


def run_case(args: argparse.Namespace, case: str) -> int:
    layout, prefetch, drain, adjacent = CASE_SETTINGS[case]
    check_layout(layout)
    if case == "no_multicycle" and not args.dry_run:
        require_native_patch(args.exe, args.skip_build_check)
    case_out = args.out / case
    print(f"\n[ABLATION] case={case} layout={layout}", flush=True)
    print("[ABLATION_POLICY] gpu_banks=2 bank_sequence=A,B,A,B,A "
          f"prefetch={prefetch} drain_first={drain} "
          "planner=LOCKED_LAYOUT_CHUNK_ONLY groups=" +
          ",".join(map(str, args.groups)), flush=True)
    if args.profile_from and not args.dry_run:
        copy_profile(args.profile_from, case_out, set(args.groups))

    # The existing 50-prompt runner has a protected LOCKED_LAYOUT key and
    # intentionally no public override, because ordinary auto runs lock only
    # after the first PASS. Inject the preselected placement *only* into its
    # parsed namespace. Its own build_process() then installs the native
    # LOCKED_LAYOUT environment variable, performs real profiling, and selects
    # independent per-request Chunk sizes with full memory/DAG guards.
    # No subprocess environment escapes or user-global env are modified.
    previous = base.get_args

    def with_locked_layout(argv: Optional[list[str]] = None):
        parsed = previous(argv)
        if not (parsed.auto_plan and parsed.htp_only):
            raise ValueError("prelocked runtime experiments require auto-plan HTP")
        parsed.locked_layout = layout
        return parsed

    base.get_args = with_locked_layout
    try:
        # The existing runner is responsible for atomic state, immutable
        # run-config, Profile V0, retries, metrics and normal resume.
        code = base.main(runner_argv(args, case, 1 if args.smoke else 0))
    finally:
        base.get_args = previous
    if not args.dry_run:
        invalid = audit_case(case_out, case, set(args.groups))
        for warning in invalid[:12]:
            print(f"[ABLATION_AUDIT_FAIL] {case}: {warning}", file=sys.stderr)
        if len(invalid) > 12:
            print(f"[ABLATION_AUDIT_FAIL] and {len(invalid) - 12} more", file=sys.stderr)
        if invalid:
            return 3
    return code


def audit_case(case_out: Path, case: str, groups: set[int]) -> list[str]:
    expected, prefetch, drain, adjacent = CASE_SETTINGS[case]
    path = case_out / "summary.csv"
    if not path.is_file():
        return ["Missing summary.csv"]
    invalid = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row.get("group") or 0) not in groups or row.get("status") != "PASS":
                continue
            prompt = row.get("prompt", "unknown")
            if row.get("plan_layout") != expected:
                invalid.append(f"{prompt}: wrong actual Plan {row.get('plan_layout')}")
            if row.get("plan_source") != "locked_chunk_only":
                invalid.append(f"{prompt}: dynamic Chunk path not acknowledged: "
                               f"{row.get('plan_source')}")
            if row.get("plan_applied", "").lower() != "true":
                invalid.append(f"{prompt}: Plan was not applied")
            if prefetch == "1" and row.get("gpu_bank_streaming", "").lower() != "true":
                invalid.append(f"{prompt}: two-bank background mover inactive")
            if prefetch == "0" and row.get("gpu_bank_streaming", "").lower() == "true":
                invalid.append(f"{prompt}: prefetch was not disabled")
            src = row.get("result_file", "")
            if not src or not Path(src).is_file():
                invalid.append(f"{prompt}: missing individual result.json")
                continue
            result = json.loads(Path(src).read_text(encoding="utf-8"))
            env = result.get("env_overrides", {})
            for name, value in (
                ("LLAMA_REGION_RING_GPU_BANK_COUNT", "2"),
                ("LLAMA_REGION_RING_GPU_BANK_PREFETCH", prefetch),
                ("LLAMA_REGION_RING_DRAIN_FIRST_CYCLE", drain),
                ("LLAMA_REGION_RING_ALLOW_ADJACENT_GPU", adjacent),
                ("LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT", expected),
            ):
                if str(env.get(name)) != value:
                    invalid.append(f"{prompt}: expected {name}={value}")
    return invalid


def write_comparison(out: Path, cases: list[str]) -> None:
    columns = ["case", "prompt", "group", "status", "plan_layout",
               "plan_source", "plan_xc", "plan_xg", "plan_xp",
               "prefill_tokens", "prefill_ms", "prefill_tps", "wall_ms",
               "gpu_bank_streaming", "bank_refills", "bank_h2d_ms",
               "bank_gpu_wait_ms", "bank_mover_wall_ms",
               "gpu_weight_stage_ms", "plan_predicted_ms", "result_file"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runtime-ablation-comparison.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            src = out / case / "summary.csv"
            if not src.is_file():
                continue
            with src.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    writer.writerow({key: case if key == "case" else row.get(key, "")
                                     for key in columns})
    print(f"[ABLATION_COMPARISON] {path}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rpc", required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASE_ORDER + ("all",),
                        default=["no_multicycle"])
    parser.add_argument("--groups", nargs="+", type=int, choices=GROUPS,
                        default=list(GROUPS))
    parser.add_argument("--only-prompt", action="append", default=[])
    parser.add_argument("--profile-from", type=Path,
                        help="Reusable native profile.csv with matching profile.json")
    parser.add_argument("--profile-grid", choices=("quick", "full"), default="quick")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-context", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--context-padding", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--inter-prompt-delay-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--trace-tasks", action="store_true")
    parser.add_argument("--boost-bank-mover", action="store_true")
    parser.add_argument("--rerun-all", action="store_true")
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="run only the first selected prompt of each case")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build-check", action="store_true")
    args = parser.parse_args(argv)
    args.groups = sorted(set(args.groups))
    if args.batch < args.ubatch or args.ubatch < 1:
        parser.error("require batch >= ubatch > 0")
    if args.retry_failed_only and args.rerun_all:
        parser.error("--retry-failed-only and --rerun-all conflict")
    args.exe = args.exe.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.prompt_dir = args.prompt_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.profile_from is not None:
        args.profile_from = args.profile_from.expanduser().resolve()
    cases = list(CASE_ORDER) if "all" in args.cases else list(dict.fromkeys(args.cases))
    for case in cases:
        check_layout(CASE_SETTINGS[case][0])
    codes: dict[str, int] = {}
    current_profile = args.profile_from
    for case in cases:
        if current_profile is not None:
            args.profile_from = current_profile
        try:
            codes[case] = run_case(args, case)
        except KeyboardInterrupt:
            print("[ABLATION_INTERRUPTED] Rerun same command to resume", file=sys.stderr)
            return 130
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"[ABLATION_ERROR] {case}: {exc}", file=sys.stderr)
            codes[case] = 2
        profile = args.out / case / "independent-profile" / "profile.csv"
        if current_profile is None and profile.is_file() and profile.with_suffix(".json").is_file():
            current_profile = profile
        if codes[case] != 0:
            print(f"[ABLATION_STATUS] case={case} status=FAIL code={codes[case]}",
                  file=sys.stderr)
    if not args.dry_run:
        write_comparison(args.out, cases)
        summary = {"cases": codes, "groups": args.groups,
                   "layout": {case: CASE_SETTINGS[case][0] for case in cases}}
        (args.out / "runtime-ablation-status.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all(code == 0 for code in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())