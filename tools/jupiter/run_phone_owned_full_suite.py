#!/usr/bin/env python3
"""One-command Qwen3-14B/32B Phone-owned Prefill suite over ShareGPT prompts.

Per Prompt: exact chat-templated Token count -> valid Chunk/KV plan ->
one selected native mode OR optional same-shape warmup + Sync/Autonomous A/B
with complete F32 logits parity -> atomic checkpoint and grouped CSV summaries.
Single-mode runs execute once per Prompt/repeat, with no extra warmup or revisit.

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
from resident_runtime import ResidentServer, timeout_marker
from native_calibration import BACKEND as NATIVE_CALIBRATION
import offline_profile as offline

PROMPT_NAME = re.compile(r"^prompt_(.+)_(\d+)$", re.IGNORECASE)
CHUNK_BUCKETS = (32, 64, 88, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
# ShareGPT40: exactly five Prompt-length groups, ten prompts each.
# Do not accidentally include the separate 2048 / 2048-4000 group.
DEFAULT_GROUPS = "0,128,256,512,1024"
# Only this exact previously shipped operational implementation may migrate.
# Git for Windows can check out these identical files with either LF or CRLF.
# All binaries, model, inputs, planner hashes and measurement options must
# still match byte-for-byte the old manifest configuration.
PRE_RECOVERY_CLIENT_SHA256 = ["cd94abdd509e680da0e98a40bd0e09fa8dc34dbf6314a8a4b8185ea83f3c881f","a5c3347df8fb4d947fa6d675e10a76512d8b3f143b9ca29b619479eae57c43de"]
PRE_RECOVERY_DRIVER_SHA256 = ["ac6cb9b98f22a4ce317e6842e0c0cccdefafcb2a31db2d6d4957f46df374fba7","f2d45deb7cfa65191f0dce4e4a954066316aac251881c33a45f114e26d9348e1"]
# Exact LF/CRLF fingerprints of the previously shipped native probe code.
# Only the redundant unmeasured token probe is being removed. No changed
# model, binary, device layout, measurement parameters or planner is allowed.
PRE_NATIVE_PROBE_DRIVER_SHA256 = ["cf9818f07731da1d9e56cb0573475546c644f44eeda7cb2f970152667ce5eebe","9d3c37cfec79ea776b06664e2e2dedba5e71a57635ccd33be3f60b004a683432"]
PRE_NATIVE_PROBE_PLANNER_SHA256 = {
    "run_q_sweep.py": ["f9653266e02460e2270fcecc6dc92fa8f20fce69cee0d27968293dd3f4a2f592","07f507cf5aace2cb6650fc6478d52ed7efdf5f92289969851f3efa029666cf9a"],
    "run_min_chunk_profile.py": ["ad14a87a84167f5beb29f5684eb342f6a1aae59e8be776a197c7b6706a842c4a","3aec35a77d3c2705418bbf75fec13f927786c3e8af987fff42eed9fe31d53b45"],
}
FIELDNAMES = (
    "layout", "group", "prompt", "kind", "repeat", "status", "tokens",
    "context", "chunks", "chunk_size", "n_chunks",
    "min_chunk", "plan_source", "plan_file", "run_mode", "first_mode",
    "execution_backend", "calibration_backend", "legacy_calibration_validation", "prefill_metric", "sync_reset_ms", "autonomous_reset_ms",
    "sync_request_wall_ms", "autonomous_request_wall_ms",
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
    p.add_argument("--resident-server", default=None,
                   help="opt-in rebuilt llama-server: resident calibration then native Prefill")
    p.add_argument("--resident-start-timeout", type=int, default=1800)
    p.add_argument("--legacy-request-timeout", type=int, default=120,
                   help="seconds per Legacy calibration HTTP request; native unchanged")
    p.add_argument("--legacy-restart-prompts", type=int, default=10,
                   help="restart Legacy after this many NEW completed Stage3 plans; 0 disables")
    p.add_argument("--legacy-timeout-retries", type=int, default=2,
                   help="additional worker restarts per timed-out Stage2/3 job")
    p.add_argument("--legacy-calibration", action="store_true",
                   help="explicit historical Legacy RPC baseline, not the migrated executor")
    p.add_argument("--skip-legacy-logits-check", action="store_true",
                   help="resident Stage2/3 timing only; tolerate nonfinite legacy logits, "
                        "never weaken native output checks")
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
                   help="first Phone layer: 14B 1..39; 32B 1..63; "
                        "CPU prefix and HTP memory constraints still apply")
    p.add_argument("--cpu-layers", type=int, default=None,
                   help="PC CPU layers (default 16 for 14B, 44 for 32B)")
    p.add_argument("--cuda-layers", type=int, default=None,
                   help="PC CUDA layers; derives CPU when not explicitly set")
    p.add_argument("--phone-layers", type=int, default=None,
                   help="Phone suffix layers up to min(model_layers-1, 64 RPC slots); "
                        "real HTP feasibility checked at model load/Prepare")
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--sequence-policy", choices=(offline.POLICY, "jupiter-dp", "fixed-control"),
                   default=offline.POLICY,
                   help="default: reusable offline Jupiter DP; jupiter-dp is the historical per-Prompt calibration")
    p.add_argument("--offline-profile-dir", default=None,
                   help="share one hardware/layout profile across compatible experiment output directories")
    p.add_argument("--profile-tag", default="",
                   help="identify Phone binary/driver/device configuration; change when Phone changes")
    p.add_argument("--profile-only", action="store_true",
                   help="build/reuse the hardware profile without tokenizing or running benchmark Prompts")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="used only for explicitly labeled fixed-control ablation")
    p.add_argument("--min-chunk-json", default=None,
                   help="optional original Stage2 min-chunk.json; without it, "
                        "measure Stage2 on the same PC/Phone model split")
    p.add_argument("--stage2-repeats", type=int, default=2)
    p.add_argument("--q-repeats", type=int, default=2)
    p.add_argument("--refine-repeats", type=int, default=2)
    p.add_argument("--max-rounds", type=int, default=0,
                   help="historical jupiter-dp only: 0 means refine until measured; offline mode never refines per Prompt")
    p.add_argument("--max-chunk", type=int, default=None,
                   help="offline: 256 by default; historical per-Prompt/fixed-control: 128")
    p.add_argument("--run-mode", choices=("both", "sync", "autonomous"),
                   default="autonomous",
                   help="both: original A/B with parity; single mode: one "
                        "native Prefill per Prompt/repeat, no warmup or revisit")
    p.add_argument("--repeats", type=int, default=1,
                   help="measured executions (single mode) or A/B pairs per Prompt")
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


def manifest_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def migrate_pre_recovery_manifest(old, config, resume):
    """Allow the known control-layer-only upgrade, never arbitrary mismatch."""
    previous = old.get("config")
    if not resume or not isinstance(previous, dict):
        return False
    if old.get("sha256") != manifest_digest(previous):
        return False
    if (config.get("execution_backend") != "resident-server" or
            previous.get("resident_client_sha256") not in PRE_RECOVERY_CLIENT_SHA256 or
            previous.get("driver_sha256") not in PRE_RECOVERY_DRIVER_SHA256):
        return False
    expected = dict(config)
    expected.pop("legacy_recovery_policy", None)
    expected["resident_client_sha256"] = previous["resident_client_sha256"]
    expected["driver_sha256"] = previous["driver_sha256"]
    return expected == previous


def migrate_native_probe_manifest(old, config, resume):
    """Resume existing native measured data only across this exact probe fix."""
    previous = old.get("config")
    if (not resume or not isinstance(previous, dict) or
            old.get("sha256") != manifest_digest(previous) or
            previous.get("calibration_backend") != NATIVE_CALIBRATION or
            config.get("calibration_backend") != NATIVE_CALIBRATION or
            config.get("execution_backend") != "resident-server" or
            previous.get("driver_sha256") not in PRE_NATIVE_PROBE_DRIVER_SHA256):
        return False
    previous_codes = previous.get("jupiter_planner_code_sha256")
    if not isinstance(previous_codes, dict):
        return False
    expected = dict(config)
    expected["driver_sha256"] = previous["driver_sha256"]
    expected_codes = dict(config["jupiter_planner_code_sha256"])
    for name, allowed in PRE_NATIVE_PROBE_PLANNER_SHA256.items():
        if previous_codes.get(name) not in allowed:
            return False
        expected_codes[name] = previous_codes[name]
    expected["jupiter_planner_code_sha256"] = expected_codes
    return expected == previous


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
        "execution_backend": "resident-server" if args.resident_server else "cli",
        "calibration_backend": getattr(args, "calibration_backend", "legacy"),
        "legacy_calibration_validation": ("NOT_USED" if getattr(args, "calibration_backend", "legacy") != "legacy"
                                          else "TIMING_ONLY" if getattr(args, "skip_legacy_logits_check", False)
                                          else "FINITE_ONLY" if args.resident_server else "CLI_UNVERIFIED"),
        "resident_server": (file_signature(Path(args.resident_server))
                            if args.resident_server else None),
        "resident_client_sha256": hashlib.sha256(
            Path(__file__).with_name("resident_runtime.py").read_bytes()).hexdigest(),
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
        "run_mode": args.run_mode,
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
                "run_sequence_pipeline.py", "refine_sequence_q.py", "native_calibration.py",
            )
        },
    }
    if args.sequence_policy == offline.POLICY:
        config["offline_profile_dir"] = str(offline.profile_directory(args, output, layouts[0]))
        config["offline_profile_signature"] = offline.digest_config(offline.hardware_config(args, layouts[0]))
        config["jupiter_planner_code_sha256"]["offline_profile.py"] = offline.digest_file(
            Path(__file__).with_name("offline_profile.py"))
    if args.resident_server and getattr(args, "calibration_backend", "legacy") == "legacy":
        config["legacy_recovery_policy"] = {
            "request_timeout_seconds": getattr(args, "legacy_request_timeout", 120),
            "restart_new_plans": getattr(args, "legacy_restart_prompts", 10),
            "timeout_retries": getattr(args, "legacy_timeout_retries", 2),
        }
    signature = manifest_digest(config)
    manifest = {"sha256": signature, "config": config}
    manifest_path = output / "suite-manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("sha256") != signature:
            if migrate_native_probe_manifest(old, config, args.resume):
                reason = "NATIVE_UNMEASURED_PROBE_ONLY"
                backup_prefix = "suite-manifest.pre-native-probe-"
            elif migrate_pre_recovery_manifest(old, config, args.resume):
                reason = "RECOVERY_CONTROL_ONLY"
                backup_prefix = "suite-manifest.pre-recovery-"
            else:
                raise ValueError("Existing output uses different files/options. "
                                 "Choose a new --out-dir, not --resume.")
            backup = output / (backup_prefix + old["sha256"][:16] + ".json")
            if backup.exists():
                if json.loads(backup.read_text(encoding="utf-8")) != old:
                    raise ValueError("Existing manifest migration backup differs")
            else:
                write_json(backup, old)
            manifest["previous_manifest_sha256"] = old["sha256"]
            print(f"[PHONE_SUITE_MANIFEST_MIGRATION] old={old['sha256']} "
                  f"new={signature} backup={backup} "
                  f"reason={reason} status=PASS", flush=True)
        elif "previous_manifest_sha256" in old:
            manifest["previous_manifest_sha256"] = old["previous_manifest_sha256"]
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
    # The original DP objective is independent of the Graph execution
    # mechanism. Main calibration uses the SAME Phone-owned executor.
    native = getattr(args, "calibration_backend", "legacy") == NATIVE_CALIBRATION
    env.update({
        "LLAMA_JUPITER_PHONE_OWNED_NATIVE": str(int(native)),
        "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": str(int(native)),
        "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "0",
        "LLAMA_JUPITER_HTP_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": "2048",
    })
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    env.update(getattr(args, "_resident_env", {}))
    return env


class LegacyCalibration:
    """Own one Legacy model at a time; retry only explicitly recorded timeouts.

    Rotation is a scheduling policy, not an alteration of the DP objective.
    Never retry a numerical/configuration/plan error as a transport timeout.
    """
    def __init__(self, args, layout, output, ubatch):
        self.args, self.layout, self.output, self.ubatch = args, layout, output, ubatch
        self.runtime = None
        self.new_plans = 0

    def __enter__(self):
        return self

    def start(self):
        if self.runtime is None:
            runtime = ResidentServer(self.args, self.layout, False, self.output, self.ubatch)
            runtime.__enter__()
            self.runtime = runtime
            self.args._resident_env = runtime.child_env()
            self.new_plans = 0

    def close(self):
        try:
            if self.runtime is not None:
                self.runtime.close()
        finally:
            self.runtime = None
            self.args._resident_env = {}
            self.new_plans = 0

    def __exit__(self, *_exc):
        self.close()

    def run(self, command, log, *, count_plan=False):
        retries = self.args.legacy_timeout_retries
        for attempt in range(retries + 1):
            self.start()
            marker = timeout_marker(self.runtime.log)
            try:
                child_run(command, log, 0, env=calibration_env(self.args))
            except RuntimeError:
                if not marker.is_file():
                    raise
                failure = json.loads(marker.read_text(encoding="utf-8"))
                if failure.get("reason") != "HTTP_REQUEST_TIMEOUT":
                    raise
                # Preserve the failed child output before the next child
                # truncates its regular stdout log. Exact q CSVs stay intact.
                archived = log.with_name(log.stem + ".timeout-" + failure["id"] + log.suffix)
                if log.is_file():
                    log.replace(archived)
                print(f"[PHONE_SUITE_LEGACY_TIMEOUT] attempt={attempt+1}/{retries+1} "
                      f"sample={failure['sample_log']} saved={archived}", flush=True)
                self.close()
                if attempt == retries:
                    raise RuntimeError("Legacy HTTP timeout retry budget exhausted. "
                                       "Restart Phone RPC/HTP and resume the same output directory.")
                print(f"[PHONE_SUITE_LEGACY_RESTART] reason=HTTP_TIMEOUT "
                      f"retry={attempt+1}/{retries} resume=1", flush=True)
                continue
            if marker.is_file():
                raise RuntimeError("Legacy child unexpectedly succeeded after a request timeout")
            if count_plan:
                self.new_plans += 1
                limit = self.args.legacy_restart_prompts
                if limit and self.new_plans >= limit:
                    print(f"[PHONE_SUITE_LEGACY_ROTATE] completed_new_plans={self.new_plans} "
                          "reason=PERIODIC_RESTART", flush=True)
                    self.close()
            return


def stage2_for_layout(args: argparse.Namespace, cases: list[dict],
                      layout: int, output: Path, legacy=None) -> Path:
    if args.min_chunk_json:
        source = Path(args.min_chunk_json).resolve()
    else:
        calibration_dir = ("native-calibration" if
            getattr(args, "calibration_backend", "legacy") == NATIVE_CALIBRATION else "calibration")
        stage2 = output / calibration_dir / f"l{layout:02d}" / "stage2"
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
            if legacy is None:
                child_run(command, stage2 / "stage2-runner.stdout.log", 0,
                          env=calibration_env(args))
            else:
                legacy.run(command, stage2 / "stage2-runner.stdout.log")
    if not source.is_file():
        raise RuntimeError(f"Jupiter Stage2 min-chunk.json missing: {source}")
    info = json.loads(source.read_text(encoding="utf-8-sig"))
    if (getattr(args, "calibration_backend", "legacy") == NATIVE_CALIBRATION and
            info.get("calibration_backend") != NATIVE_CALIBRATION):
        raise ValueError("Native Stage2 profile required; do not reuse Legacy data")
    if (int(info["pc_layers"]) != layout or
            int(info.get("cpu_layers", 0)) != args.cpu_layers or
            int(info["max_chunk"]) != args.max_chunk or
            not info.get("candidates")):
        raise ValueError(
            f"Jupiter Stage2 calibration does not match CPU/PC boundary, "
            f"maximum Chunk or measured candidates: {source}")
    return source


def exact_jupiter_dp_plan(args: argparse.Namespace, case: dict,
                          output: Path, stage2_json: Path, legacy=None) -> dict:
    native = getattr(args, "calibration_backend", "legacy") == NATIVE_CALIBRATION
    target = output / ("native-calibration" if native else "calibration") / f"l{case['layout']:02d}" / \
        "plans" / case["name"].removesuffix(".txt")
    stage3 = target / "stage3"
    plan_file = target / "final.plan"
    summary_path = stage3 / "sequence-summary.json"
    existing = (json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file() else None)
    if existing and ((native and existing.get("calibration_backend") != NATIVE_CALIBRATION) or
                     existing.get("pc_layers") != case["layout"] or
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
        if legacy is None:
            child_run(command, target / "stage3-runner.stdout.log", 0,
                      env=calibration_env(args))
        else:
            legacy.run(command, target / "stage3-runner.stdout.log", count_plan=True)
    elif legacy is not None:
        print(f"[PHONE_SUITE_DP_RESUME] layout={case['layout']} "
              f"prompt={case['name']} status=REUSE_PLAN", flush=True)
    if not plan_file.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Jupiter Stage3 plan incomplete: {target}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (native and summary.get("calibration_backend") != NATIVE_CALIBRATION):
        raise RuntimeError("Native Stage3 profile provenance missing")
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
    planned["plan_source"] = ("JUPITER_NATIVE_STAGE2_B_QXY_EXACT_DP" if native
                              else "JUPITER_STAGE2_B_QXY_EXACT_DP")
    planned["min_chunk"] = summary["min_chunk"]
    print(f"[PHONE_SUITE_DP_READY] layout={case['layout']} "
          f"prompt={case['name']} b={planned['min_chunk']}"
          f" chunks={chunks} status=READY", flush=True)
    return planned


def run_mode(args: argparse.Namespace, case: dict,
             mode: str, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(getattr(args, "_resident_env", {}))
    child_env["LLAMA_JUPITER_PROFILE_STAGE"] = "0"
    child_env["LLAMA_JUPITER_Q_SWEEP"] = "0"
    child_run(cli_for_case(args, case, mode, work),
              work / "runner.stdout.log", args.timeout_seconds, env=child_env)
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
            "timing_tokens": report.get("prompt_eval_timing_tokens"),
            "execution_backend": report.get("execution_backend", "cli"),
            "prefill_metric": report.get("prefill_metric", "CLI_PROMPT_EVAL"),
            "resident_reset_ms": report.get("resident_reset_ms"),
            "resident_request_wall_ms": report.get("resident_request_wall_ms")}


def parity_ok(parity: dict) -> bool:
    return (parity["top1_same"] and parity["cosine"] is not None and
            parity["cosine"] >= 0.999 and
            parity["relative_rmse"] is not None and
            parity["relative_rmse"] <= 0.01)


def selected_modes(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return ("sync", "autonomous")
    if mode in ("sync", "autonomous"):
        return (mode,)
    raise ValueError(f"Unknown run mode: {mode}")


def completed_case_matches(previous: dict, mode: str) -> bool:
    runs = previous.get("runs", {})
    return (previous.get("status") == "PASS" and
            set(runs) == set(selected_modes(mode)) and
            all(Path(x.get("logits", "")).is_file() for x in runs.values()))


def perform_case(args: argparse.Namespace, case: dict,
                 output: Path, position: int) -> dict:
    root = case_directory(output, case)
    attempt = next_attempt(root)
    mode_choice = args.run_mode
    order = list(selected_modes(mode_choice))
    paired = mode_choice == "both"
    if paired and (args.mode_order == "autonomous-first" or
                   (args.mode_order == "alternate" and position % 2)):
        order.reverse()
    warmup_info = None
    if paired and not args.no_warmup:
        warm = run_mode(args, case, "sync", attempt / "warmup")
        # Preserve the first A-B-A revisit cache state.
        warmup_info = {"cache_hit": warm["cache_hit"],
                       "prepare": warm["prepare"], "log": warm["log"]}
        Path(warm["logits"]).unlink(missing_ok=True)
    runs = {mode: run_mode(args, case, mode, attempt / mode)
            for mode in order}
    parity = None
    comparable = None
    speedup = None
    sync_ms = runs["sync"]["ms"] if "sync" in runs else None
    autonomous_ms = runs["autonomous"]["ms"] if "autonomous" in runs else None
    if paired:
        parity = logits_comparison(
            read_logits(Path(runs["sync"]["logits"])),
            read_logits(Path(runs["autonomous"]["logits"])))
        if not parity_ok(parity):
            raise RuntimeError(
                f"Complete Logits parity FAIL for {case['name']}: {parity}")
        comparable = runs["sync"]["cache_hit"] and runs["autonomous"]["cache_hit"]
        speedup = sync_ms / autonomous_ms
    return dict(case, status="PASS", run_mode=mode_choice,
                first_mode=order[0], runs=runs, parity=parity,
                warmup=warmup_info,
                revisit_first_cache_hit=(
                    warmup_info["cache_hit"] if warmup_info is not None
                    else runs[order[0]]["cache_hit"]
                ) if case["kind"] == "revisit" else None,
                warm_comparable=comparable, sync_ms=sync_ms,
                autonomous_ms=autonomous_ms, speedup=speedup,
                case_dir=str(root), attempt_dir=str(attempt))


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
        "run_mode": row.get("run_mode", "both"),
        "first_mode": row.get("first_mode", ""),
        "execution_backend": row.get("execution_backend", "cli"),
        "calibration_backend": row.get("calibration_backend", "legacy"),
        "legacy_calibration_validation": row.get("legacy_calibration_validation", "CLI_UNVERIFIED"),
        "prefill_metric": (auto or sync).get("prefill_metric", ""),
        "sync_reset_ms": sync.get("resident_reset_ms"),
        "autonomous_reset_ms": auto.get("resident_reset_ms"),
        "sync_request_wall_ms": sync.get("resident_request_wall_ms"),
        "autonomous_request_wall_ms": auto.get("resident_request_wall_ms"),
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
        sync = [x for x in items if x.get("sync_ms") is not None]
        auto = [x for x in items if x.get("autonomous_ms") is not None]
        def mean_ms(selected, key):
            return round(statistics.mean(x[key] for x in selected), 3) if selected else ""
        def median_ms(selected, key):
            return round(statistics.median(x[key] for x in selected), 3) if selected else ""
        def weighted_tok_s(selected, key):
            return (round(1000*sum(x["tokens"] for x in selected)/
                          sum(x[key] for x in selected), 3) if selected else "")
        result.append({
            "layout": layout, "group": group,
            "run_mode": items[0].get("run_mode", "both"),
            "execution_backend": items[0].get("execution_backend", "cli"),
            "calibration_backend": items[0].get("calibration_backend", "legacy"),
            "legacy_calibration_validation": items[0].get("legacy_calibration_validation", "CLI_UNVERIFIED"),
            "passed_cases": len(items),
            "passed_pairs": sum(x.get("speedup") is not None for x in items),
            "distinct_prompts": len({x["name"] for x in items}),
            "sync_runs": len(sync), "autonomous_runs": len(auto),
            "sync_cache_hits": sum(bool(x.get("runs", {}).get("sync", {}).get("cache_hit")) for x in sync),
            "autonomous_cache_hits": sum(bool(x.get("runs", {}).get("autonomous", {}).get("cache_hit")) for x in auto),
            "warm_pairs": len(warm),
            "sync_mean_ms": mean_ms(sync, "sync_ms"),
            "autonomous_mean_ms": mean_ms(auto, "autonomous_ms"),
            "sync_median_ms": median_ms(sync, "sync_ms"),
            "autonomous_median_ms": median_ms(auto, "autonomous_ms"),
            "warm_geomean_speedup": (
                round(statistics.geometric_mean(ratios), 4) if ratios else ""),
            "warm_mean_speedup": (
                round(statistics.mean(ratios), 4) if ratios else ""),
            "warm_mean_sync_ms": mean_ms(warm, "sync_ms"),
            "warm_mean_autonomous_ms": mean_ms(warm, "autonomous_ms"),
            "sync_weighted_tok_s": weighted_tok_s(sync, "sync_ms"),
            "autonomous_weighted_tok_s": weighted_tok_s(auto, "autonomous_ms"),
        })
    return sorted(result, key=lambda x: (x["layout"], group_key(
        f"prompt_{x['group']}_01.txt")))


def publish(output: Path, rows: list[dict],
            expected_cases: int, manifest_sha: str, mode: str = "both") -> None:
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
              ("layout", "group", "run_mode", "passed_cases", "passed_pairs", "warm_pairs"))
    summary = {
        "manifest_sha256": manifest_sha,
        "run_mode": mode,
        "execution_backend": (rows[0].get("execution_backend", "cli") if rows else ""),
        "calibration_backend": (rows[0].get("calibration_backend", "legacy") if rows else ""),
        "legacy_calibration_validation": (rows[0].get("legacy_calibration_validation", "CLI_UNVERIFIED") if rows else ""),
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
            if mode == "both" else
            "Single-mode native Prefill: no per-Prompt warmup, revisit or A/B "
            "parity; cache hits/misses are recorded. Preparation may be included "
            "in a cache-miss measurement."
        ),
    }
    write_json(output / "summary.json", summary)


def run() -> int:
    args = parse_args()
    # Retain backward-compatible programmatic callers with no run_mode.
    args.run_mode = getattr(args, "run_mode", "both")
    args.resident_server = getattr(args, "resident_server", None)
    args.resident_start_timeout = getattr(args, "resident_start_timeout", 1800)
    args.legacy_request_timeout = getattr(args, "legacy_request_timeout", 120)
    args.legacy_restart_prompts = getattr(args, "legacy_restart_prompts", 10)
    args.legacy_timeout_retries = getattr(args, "legacy_timeout_retries", 2)
    if (args.legacy_request_timeout <= 0 or args.legacy_restart_prompts < 0 or
            args.legacy_timeout_retries < 0):
        raise ValueError("invalid Legacy recovery policy")
    args.legacy_calibration = getattr(args, "legacy_calibration", False)
    args.profile_only = getattr(args, "profile_only", False)
    args.offline_profile_dir = getattr(args, "offline_profile_dir", None)
    args.profile_tag = getattr(args, "profile_tag", "")
    is_offline = args.sequence_policy == offline.POLICY
    if args.max_chunk is None:
        args.max_chunk = 256 if is_offline else 128
    if is_offline and args.legacy_calibration:
        raise ValueError("offline Native profile does not use Legacy; select jupiter-dp explicitly for the historical baseline")
    if not is_offline and (args.profile_only or args.offline_profile_dir or args.profile_tag):
        raise ValueError("offline profile options require " + offline.POLICY)
    if is_offline:
        offline.check_options(args)
        if not args.resident_server:
            raise ValueError("offline Native profile requires --resident-server (-UseServer)")
        if args.min_chunk_json:
            raise ValueError("offline mode measures its own layout-specific b; no historical Stage2 import")
    args.calibration_backend = ("none" if args.sequence_policy == "fixed-control" else
                                "legacy" if args.legacy_calibration else NATIVE_CALIBRATION)
    if args.calibration_backend == NATIVE_CALIBRATION and not args.resident_server and not args.dry_run:
        raise ValueError("Native DP calibration requires --resident-server (-UseServer); "
                         "use --legacy-calibration only for an explicit historical baseline")
    print(f"[PHONE_SUITE_CALIBRATION_BACKEND] mode={args.calibration_backend} "
          f"profile={'OFFLINE_SHARED' if is_offline else 'HISTORICAL_PER_PROMPT' if args.sequence_policy == 'jupiter-dp' else 'NONE'}", flush=True)
    if args.resident_server and args.calibration_backend == "legacy":
        print(f"[PHONE_SUITE_LEGACY_POLICY] request_timeout_seconds={args.legacy_request_timeout} "
              f"restart_new_plans={args.legacy_restart_prompts} "
              f"timeout_retries={args.legacy_timeout_retries}", flush=True)
    args.skip_legacy_logits_check = getattr(args, "skip_legacy_logits_check", False)
    if args.skip_legacy_logits_check and (not args.resident_server or args.calibration_backend != "legacy"):
        raise ValueError("--skip-legacy-logits-check requires explicit --legacy-calibration; "
                         "native calibration never skips finite output validation")
    if args.skip_legacy_logits_check:
        print("[PHONE_SUITE_WARNING] legacy_calibration_validation=TIMING_ONLY "
              "accuracy=UNVERIFIED native_logits_check=STRICT", flush=True)
    selected_modes(args.run_mode)
    if args.run_mode != "both":
        args.no_warmup = True
        args.revisit_first = False
    # Validate against the GGUF before any expensive Stage2/Stage3 calibration.
    model = inspect_model(Path(args.model), args.model_size)
    layouts, args.cpu_layers = resolve_splits(
        model, layouts=args.layouts, cpu_layers=args.cpu_layers,
        cuda_layers=args.cuda_layers, phone_layers=args.phone_layers)
    args.model_size = model.family
    args.model_layers = model.layers
    if args.out_dir is None:
        args.out_dir = f"jupiter-autonomous-{model.family}-{'offline-dp' if is_offline else 'dp'}-2048"
        if args.run_mode != "both":
            args.out_dir += f"-{args.run_mode}-only"
        if args.resident_server:
            args.out_dir += "-resident"
        if args.calibration_backend == NATIVE_CALIBRATION:
            args.out_dir += "-native-profile"
        if args.skip_legacy_logits_check:
            args.out_dir += "-legacy-timing-only"
    if args.resident_server:
        if len(layouts) != 1:
            raise ValueError("resident experiment requires one immutable layout per suite")
        if args.continue_on_error:
            raise ValueError("resident fails closed after a device error; resume after recovery")
        if args.min_chunk_json:
            raise ValueError("resident requires its own Stage2 calibration; do not mix CLI profiles")
        if not Path(args.resident_server).is_file():
            raise FileNotFoundError(args.resident_server)
        if args.resident_start_timeout <= 0:
            raise ValueError("resident startup timeout must be positive")
    print(f"[PHONE_SUITE_MODEL] model=Qwen3-{model.family.upper()}"
          f" layers={model.layers} CPU={args.cpu_layers}"
          f" first_phone={layouts} GPU={[x-args.cpu_layers for x in layouts]}"
          f" HTP={[model.layers-x for x in layouts]}"
          " context=2048 status=VALIDATED", flush=True)
    if (args.repeats < 1 or args.expected_count < 0 or args.chunk_size < 1 or
            args.timeout_seconds < 0 or args.stage2_repeats < 1 or
            args.q_repeats < 1 or args.refine_repeats < 1 or
            args.max_rounds < 0 or args.max_chunk < 1 or
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
    if args.profile_only:
        if args.dry_run:
            raise ValueError("--profile-only and --dry-run are exclusive")
        result = ensure_offline_profile(args, layouts[0], output)
        print(f"[PHONE_SUITE_PROFILE_ONLY] root={result['root']} b={result['b']} "
              f"lengths={len(result['plans'])} status=PASS", flush=True)
        return 0
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
    for case in cases:
        case["run_mode"] = args.run_mode
        case["execution_backend"] = "resident-server" if args.resident_server else "cli"
        case["calibration_backend"] = args.calibration_backend
        case["legacy_calibration_validation"] = (
            "NOT_USED" if args.calibration_backend != "legacy" else
            "TIMING_ONLY" if args.skip_legacy_logits_check else
            "FINITE_ONLY" if args.resident_server else "CLI_UNVERIFIED")
    print(f"[PHONE_SUITE] model=Qwen3-{model.family.upper()}"
          f" prompts={len(files)} layouts={layouts} run_mode={args.run_mode}"
          f" repeats={args.repeats} total_cases={len(cases)}"
          f" backend={'resident-server' if args.resident_server else 'cli'}"
          f" measured_runs={len(cases)*len(selected_modes(args.run_mode))}"
          f" context=2048 batch=2048 sequence_policy={args.sequence_policy}"
          f" warmup={not args.no_warmup} status=PLANNED", flush=True)
    if args.dry_run:
        print(f"[PHONE_SUITE] manifest={output/'suite-manifest.json'}"
              " dry_run=PASS")
        return 0

    if not args.resident_server:
        return execute_cases(args, cases, output, plan)
    return execute_resident_suite(args, cases, output, plan, layouts[0])


def ensure_offline_profile(args, layout, output):
    root = offline.profile_directory(args, output, layout)
    manifest = offline.open_profile(args, layout, root)
    cached = offline.load_ready(root, manifest)
    if cached is not None:
        print(f"[JUPITER_OFFLINE_PROFILE] root={root} b={cached['b']} "
              f"lengths={len(cached['plans'])} status=REUSE", flush=True)
        return cached
    print(f"[JUPITER_OFFLINE_PROFILE] root={root} status=BUILD", flush=True)
    # Calibration graph/KV namespace is isolated from the unprofiled worker.
    with ResidentServer(args, layout, True, root, args.max_chunk, calibration=True) as runtime:
        result = offline.build_profile(args, layout, root, manifest, runtime.child_env())
    print(f"[JUPITER_OFFLINE_PROFILE] root={root} b={result['b']} "
          f"lengths={len(result['plans'])} status=READY", flush=True)
    return result


def bind_offline_profile(args, layout, output):
    # Never combine formal rows from two different measured profiles, even
    # when the profile directory/configuration has the same name.
    root = offline.profile_directory(args, output, layout)
    path = Path(output) / "offline-profile-binding.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if old is not None:
        ready_path = root / "profile-ready.json"
        if (not ready_path.is_file() or
                offline.digest_file(ready_path) != old.get("profile_ready_sha256")):
            raise ValueError("bound offline profile missing or changed; restore it or choose a new experiment directory")
    result = ensure_offline_profile(args, layout, output)
    binding = {"profile_dir": str(result["root"]),
               "profile_signature": result["ready"]["signature"],
               "profile_ready_sha256": result["ready_sha256"],
               "validation_status": result["ready"]["validation_status"]}
    if old is not None and old != binding:
        raise ValueError("existing experiment uses a different offline profile")
    if old is None and any(Path(output).glob("cases/**/case-result.json")):
        raise ValueError("existing experiment rows have no offline profile binding; use a new directory")
    write_json(path, binding)
    return result


def offline_case_plan(case, profile):
    total = case["tokens"]
    cuts = profile["plans"].get(total)
    if cuts is None:
        raise ValueError(f"offline profile has no sequence for {total} tokens")
    return dict(case, chunks=list(cuts), min_chunk=profile["b"],
                plan_file=str(profile["root"] / "sequence.plan"),
                plan_source=("JUPITER_OFFLINE_SHORT_SINGLE_CHUNK" if total < profile["b"] else
                             "JUPITER_NATIVE_OFFLINE_INTERPOLATED_DP"))


def execute_resident_suite(args, cases, output, plan, layout):
    # Never load two copies of a 14B/32B model at once. Profile the same
    # native autonomous execution path, then use a separate unprofiled
    # worker for the formal measurements. Legacy is explicit opt-in only.
    def pending_case(case):
        path = case_directory(output, case) / "case-result.json"
        if not args.resume or not path.is_file():
            return True
        return not completed_case_matches(
            json.loads(path.read_text(encoding="utf-8")), args.run_mode)

    pending = [case for case in cases if pending_case(case)]
    resident_plans = {}
    ubatch = args.max_chunk
    if args.sequence_policy == offline.POLICY:
        profile = bind_offline_profile(args, layout, output)
        for case in pending:
            key = (case["layout"], case["path"])
            if key in resident_plans:
                continue
            chosen = offline_case_plan(case, profile)
            resident_plans[key] = chosen
            print(f"[PHONE_SUITE_OFFLINE_LOOKUP] layout={layout} prompt={case['name']} "
                  f"tokens={case['tokens']} b={chosen['min_chunk']} "
                  f"chunks={chosen['chunks']} status=READY", flush=True)
        if not pending:
            return execute_cases(args, cases, output, plan, resident_plans={})
    elif not pending:
        return execute_cases(args, cases, output, plan, resident_plans={})
    elif args.sequence_policy == "jupiter-dp" and args.calibration_backend == NATIVE_CALIBRATION:
        with ResidentServer(args, layout, True, output, ubatch, calibration=True) as runtime:
            args._resident_env = runtime.child_env()
            try:
                stage2 = stage2_for_layout(args, cases, layout, output)
                for index, case in enumerate(pending, 1):
                    key = (case["layout"], case["path"])
                    if key in resident_plans:
                        continue
                    print(f"[PHONE_SUITE_RESIDENT_CALIBRATION] {index}/{len(pending)} "
                          f"prompt={case['name']} mode=PHONE_AUTONOMOUS", flush=True)
                    resident_plans[key] = exact_jupiter_dp_plan(args, case, output, stage2)
            finally:
                args._resident_env = {}
    elif args.sequence_policy == "jupiter-dp":
        with LegacyCalibration(args, layout, output, ubatch) as legacy:
            stage2 = stage2_for_layout(args, cases, layout, output, legacy=legacy)
            for index, case in enumerate(pending, 1):
                key = (case["layout"], case["path"])
                if key in resident_plans:
                    continue
                print(f"[PHONE_SUITE_RESIDENT_CALIBRATION] {index}/{len(pending)} "
                      f"prompt={case['name']}", flush=True)
                resident_plans[key] = exact_jupiter_dp_plan(
                    args, case, output, stage2, legacy=legacy)
    else:
        ubatch = max(ubatch, max(max(case["chunks"]) for case in cases))
    with ResidentServer(args, layout, True, output, ubatch) as runtime:
        args._resident_env = runtime.child_env()
        try:
            return execute_cases(args, cases, output, plan, resident_plans=resident_plans)
        finally:
            args._resident_env = {}


def execute_cases(args, cases, output, plan, resident_plans=None):
    rows = []
    reference = {}
    stage2_by_layout = {}
    stopped = False
    for position, case in enumerate(cases):
        root = case_directory(output, case)
        stored = root / "case-result.json"
        if args.resume and stored.exists():
            previous = json.loads(stored.read_text(encoding="utf-8"))
            if completed_case_matches(previous, args.run_mode):
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
                if args.sequence_policy in ("jupiter-dp", offline.POLICY):
                    layout = case["layout"]
                    if resident_plans is not None:
                        planned = resident_plans[(layout, case["path"])]
                        case = dict(case, **{k: planned[k] for k in
                                    ("chunks", "plan_file", "plan_source", "min_chunk")})
                    else:
                        if args.sequence_policy == offline.POLICY:
                            raise RuntimeError("offline DP requires a bound shared sequence table")
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
        publish(output, rows, len(cases), plan["signature"], args.run_mode)
        if row["status"] == "PASS":
            if args.run_mode == "both":
                print(f"[PHONE_SUITE] result=PASS"
                      f" sync_ms={row['sync_ms']:.2f}"
                      f" autonomous_ms={row['autonomous_ms']:.2f}"
                      f" ratio={row['speedup']:.3f}"
                      f" warm_comparable={row['warm_comparable']}", flush=True)
            else:
                selected = row["runs"][args.run_mode]
                print(f"[PHONE_SUITE] result=PASS mode={args.run_mode}"
                      f" prefill_ms={selected['ms']:.2f}"
                      f" tok_s={row['tokens']*1000/selected['ms']:.3f}"
                      f" cache_hit={selected['cache_hit']}", flush=True)
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
