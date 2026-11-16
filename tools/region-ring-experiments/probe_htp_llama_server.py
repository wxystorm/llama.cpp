#!/usr/bin/env python3
"""Experimental resident llama-server smoke test for HTP-only Ring Prefill.

By default the server keeps ONE startup Chunk Plan across requests.
--per-request-chunks opts into the native locked-placement Planner in
llama_context::decode(), with actual per-request token counts and runtime
Stage widths. This remains distinct from reloading a model per Prompt.
No actual capacity checks are bypassed; only the Planner's estimates are.

Example:
  python tools/region-ring-experiments/probe_htp_llama_server.py \
    --server-exe E:/llama/llama-tensor-v2/build-tensor-v2/bin/Release/llama-server.exe \
    --model E:/llama/models/Qwen3-32B-Q4_K_M.gguf \
    --rpc 10.238.229.239:50052 \
    --prompt-dir E:/llama/datasets/sharegpt/sharegpt50_short \
    --profile-source E:/llama/llama-tensor-v2/32b-sharegpt50-short-mmap-test/independent-profile \
    --out E:/llama/llama-tensor-v2/32b-server-smoke \
    --port 18080
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import run_32b_all40 as runner

# Qwen3-32B: the final G7 is essential; G6 would total only 63 layers.
LAYOUT = "G7,N4,G7,N4,G8,N5,G7,N4,G7,N4,G7"


def server_command(server_exe, model, rpc, port, threads=8):
    """Use one slot and a whole-prompt UBatch to avoid unsupported splits."""
    return [
        str(server_exe), "-m", str(model), "--rpc", rpc,
        "--split-mode", "tensor", "--flash-attn", "on",
        "-c", "2048", "-b", "2048", "-ub", "2048",
        "-t", str(threads), "--parallel", "1",
        "--no-cont-batching", "--no-cache-prompt",
        "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--no-context-shift", "--no-warmup",
        "--host", "127.0.0.1", "--port", str(port),
        "-lv", "4",
    ]


def completion_payload(text):
    return {
        "prompt": text,
        "n_predict": 1,
        "cache_prompt": False,
        "stream": False,
        "temperature": 0,
        "seed": 1234,
        "id_slot": 0,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-exe", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-dir", type=Path, required=True)
    p.add_argument("--profile-source", type=Path, required=True,
                   help="directory containing existing profile.csv and profile.json")
    p.add_argument("--out", type=Path, required=True,
                   help="new independent output directory, never the main-study out")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--all50", action="store_true",
                   help="run all 50 unique prompts, starting with prompt_1024_10.txt")
    p.add_argument("--per-request-chunks", action="store_true",
                   help="call native locked XC/XG/XP Planner on each real HTTP prefill")
    p.add_argument("--max-plan-xg", type=int, default=256,
                   choices=(32, 48, 64, 80, 96, 112, 128, 160, 192, 256),
                   help="native GPU Chunk search ceiling (default: 256)")
    p.add_argument("--max-plan-xp", type=int, default=256,
                   choices=(32, 48, 64, 80, 96, 112, 128, 160, 192, 256),
                   help="native Phone Chunk search ceiling (default: 256)")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted --all50 run; skip verified completed requests")
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N unique prompts (0=all 50), for staged testing")
    p.add_argument("--inter-prompt-delay-seconds", type=float, default=None,
                   help="delay between HTTP requests (default: 5 for --all50, 0 for smoke)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        p.error("--port must be 1024..65535")
    for label in ("server_exe", "model"):
        if not getattr(args, label).is_file():
            p.error("--" + label.replace("_", "-") + " not found")
    if not args.prompt_dir.is_dir():
        p.error("--prompt-dir not found")
    if args.threads <= 0 or args.startup_timeout <= 0:
        p.error("--threads and --startup-timeout must be positive")
    if args.resume and not args.all50:
        p.error("--resume requires --all50 (the smoke test repeats the same prompt)")
    if args.limit < 0 or args.limit > 50 or (args.limit and not args.all50):
        p.error("--limit requires --all50 and must be between 1 and 50")
    if args.inter_prompt_delay_seconds is None:
        args.inter_prompt_delay_seconds = 5.0 if args.all50 else 0.0
    if not 0 <= args.inter_prompt_delay_seconds <= 3600:
        p.error("--inter-prompt-delay-seconds must be between 0 and 3600")
    return args


def prompt_files(folder, all50):
    if not all50:
        names = ["prompt_1024_10.txt", "prompt_0_01.txt",
                 "prompt_1024_10.txt"]
    else:
        anchor = "prompt_1024_10.txt"
        # Exactly 50 distinct cases. Long anchor FIRST checks that the
        # startup plan and fresh HTP KV work before shorter requests.
        names = [anchor] + [
            f"prompt_{group}_{i:02d}.txt"
            for group in runner.MAIN50_SHORT_GROUPS for i in range(1, 11)
            if f"prompt_{group}_{i:02d}.txt" != anchor
        ]
        assert len(names) == 50 and len(set(names)) == 50
    paths = [folder / name for name in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError("Missing ShareGPT Prompt: " + ", ".join(missing[:4]))
    return paths


def fetch_health(port):
    req = Request(f"http://127.0.0.1:{port}/health")
    with urlopen(req, timeout=3) as response:
        data = json.load(response)
        return response.status == 200 and data.get("status") == "ok"


def wait_for_ready(proc, port, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup: {proc.returncode}")
        try:
            if fetch_health(port):
                return
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.75)
    raise RuntimeError("llama-server startup health check timed out")


def post_completion(port, prompt):
    payload = completion_payload(prompt)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(f"http://127.0.0.1:{port}/completion", data=data,
                  headers={"Content-Type": "application/json"},
                  method="POST")
    try:
        with urlopen(req, timeout=360) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(result, dict) or not isinstance(result.get("timings"), dict):
        raise RuntimeError("Response lacks timing fields: " +
                           str(result)[:400])
    t = result["timings"]
    cache_n = int(t.get("cache_n") or 0)
    prompt_n = int(t.get("prompt_n") or 0)
    prompt_ms = float(t.get("prompt_ms") or 0)
    if cache_n != 0:
        raise RuntimeError(
            f"Invalid Prefill comparison: reused {cache_n} cached tokens")
    if prompt_n < 32 or prompt_ms <= 0:
        raise RuntimeError(
            f"HTP-only Prefill not confirmed: prompt_n={prompt_n}, ms={prompt_ms}")
    return {
        "prompt_n": prompt_n,
        "cache_n": cache_n,
        "prompt_ms": prompt_ms,
        "prefill_tps": float(t.get("prompt_per_second") or
                             (1000.0 * prompt_n / prompt_ms)),
        "predicted_n": int(t.get("predicted_n") or 0),
    }


def log_tail(path, max_lines=45):
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=max_lines))


def first_error_markers(path, limit=14):
    """Read the ENTIRE server log; last-45-lines often hide first failure.

    Prioritize worker/Phone RPC/PUSH root-cause messages over secondary
    P2C drain failures and the final HTTP 500. Keep memory bounded.
    """
    import re
    primary = re.compile(
        r"\[(?:REGION_RING_WORKER|PHONE_INPUT_PUSH|PHONE_OWNED_PLAN|"
        r"PHONE_HTP_ONLY|PHONE_AUTO|REGION_RING_GPU_BANK_MOVER|"
        r"REGION_RING_P2C_JOIN|REGION_RING_REQUEST_CHUNK_PLAN)\].*"
        r"(?:event=EXCEPTION|event=FAIL|status=FAIL|status=REJECT|reason=)|"
        r"\[REGION_RING\].*(?:event=DEADLOCK|event=FAIL)|"
        r"ggml_.*(?:error -?[0-9]+|failed)",
        re.IGNORECASE,
    )
    secondary = re.compile(
        r"\[(?:P2C_STREAM|REGION_RING_SUM|PHONE_INPUT_STREAM)\]"
        r".*(?:status=FAIL|status=-1)|"
        r"llama_decode: failed to decode|srv.*Compute error",
        re.IGNORECASE,
    )
    root, followon = [], []
    if not path.is_file():
        return "server log not found: " + str(path)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            target = root if primary.search(line) else (
                followon if secondary.search(line) else None)
            if target is not None and len(target) < limit:
                target.append(f"{number}: {line.rstrip()}")
            if len(root) >= limit:
                break
    if not root:
        return "No first-cause marker found; inspect the full log.\n" + (
            "\n".join(followon) or "(no secondary errors found)")
    return "\n".join(root + followon[:4])


# A native per-request Plan is a C++ decision in llama_context::decode;
# do not treat a Python-side length heuristic as evidence that the server
# executed a new Stage shape. Require the source runtime log for every PASS.
REQUEST_PLAN_RE = re.compile(
    r"\[REGION_RING_REQUEST_CHUNK_PLAN\] event=SELECT "
    r"tokens=(\d+) kv_start=0 layout=([^\s]+) "
    r"XC=(\d+) XG=(\d+) XP=(\d+).*"
    r"planner_ms=([0-9.]+).*status=PASS")
# Runtime may widen a tiny tail uniformly (e.g. selected XG32 becomes
# actually executed XG33 for a 66-token request). Preserve both values.
REQUEST_EXEC_RE = re.compile(
    r"\[REGION_RING_SUM\] tokens=(\d+) XC=(\d+) XG=(\d+) XP=(\d+)"
    r".*complete=1 .*status=0")


def read_request_chunk_plan(log_path, offset, expected_tokens,
                            max_plan_xg=256, max_plan_xp=256):
    """Read only logs appended since the previous HTTP request finished."""
    with Path(log_path).open("rb") as stream:
        stream.seek(offset)
        data = stream.read().decode("utf-8", errors="replace")
        next_offset = stream.tell()
    matches = list(REQUEST_PLAN_RE.finditer(data))
    if len(matches) != 1:
        raise RuntimeError(
            "Native per-request Chunk decision missing or ambiguous "
            f"(matches={len(matches)}, expected_tokens={expected_tokens}); "
            f"check {log_path} for REGION_RING_REQUEST_CHUNK_PLAN")
    tokens, layout, xc, xg, xp, planner_ms = matches[0].groups()
    if int(tokens) != expected_tokens:
        raise RuntimeError(
            f"Native Chunk Plan token mismatch: planner={tokens}, "
            f"server_timing={expected_tokens}")
    chunks = (int(xc), int(xg), int(xp))
    # This layout contains only GPU and Phone Regions. XC is a modeled
    # but unused CPU-stage width and is intentionally not capped by
    # --max-plan-xg/--max-plan-xp. Only the physical GPU/Phone widths
    # must stay within the configured ceiling (up to native 256).
    if not (16 <= int(xc) <= min(256, expected_tokens) and
            16 <= int(xg) <= max_plan_xg and
            16 <= int(xp) <= min(max_plan_xp, expected_tokens)):
        raise RuntimeError(
            "Native Chunk Plan outside controlled width bounds: "
            + str(chunks))
    if 2 * int(xg) > expected_tokens:
        raise RuntimeError(
            f"Two-Bank request has insufficient tokens: "
            f"XG={xg}, tokens={expected_tokens}")
    if layout != "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7":
        raise RuntimeError("Native Chunk Planner changed fixed layout: " + layout)
    executed = list(REQUEST_EXEC_RE.finditer(data))
    if len(executed) != 1:
        raise RuntimeError(
            "Completed native Region Ring execution missing or ambiguous"
            f" (matches={len(executed)}, expected_tokens={expected_tokens}); "
            f"check {log_path} for REGION_RING_SUM")
    exec_tokens, xc_actual, xg_actual, xp_actual = executed[0].groups()
    if int(exec_tokens) != expected_tokens:
        raise RuntimeError(
            f"Native Region Ring execution token mismatch: "
            f"actual={exec_tokens}, timing={expected_tokens}")
    xca, xga, xpa = int(xc_actual), int(xg_actual), int(xp_actual)
    # A selected 32 may execute as 33 after LLAMA_REGION_RING_COALESCE_TINY_TAIL.
    # Do not confuse that uniform-width adjustment with a different Plan.
    for name, selected, actual in (
            ("XG", chunks[1], xga), ("XP", chunks[2], xpa)):
        if not selected <= actual <= selected + selected // 4:
            raise RuntimeError(
                f"Runtime {name}={actual} differs unexpectedly "
                f"from planned {name}={selected}")
    return {
        "plan_xc": chunks[0],
        "plan_xg": chunks[1],
        "plan_xp": chunks[2],
        "planner_ms": float(planner_ms),
        "runtime_xc": xca,
        "runtime_xg": xga,
        "runtime_xp": xpa,
    }, next_offset


def group_for_prompt(name):
    """The 0 group is actually 64..128 tokenizer tokens."""
    return int(name.split("_", 2)[1])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sweep_config(args, command, env, all_prompts, anchor_tokens):
    """Fingerprint every factor that must remain constant across a resume.

    --limit is deliberately excluded: a staged first 10 and resumed full
    50 must share exactly the same experiment identity.
    """
    runtime_env = {
        key: value for key, value in sorted(env.items())
        if key.startswith(("LLAMA_", "GGML_"))
    }
    base = {
        "schema": 1,
        "server_command": command,
        "server_binary": file_identity(args.server_exe),
        "model": file_identity(args.model),
        "rpc": args.rpc,
        "prompt_dir": str(args.prompt_dir.resolve()),
        "prompts": [p.name for p in all_prompts],
        "prompt_sha256": {p.name: sha256_file(p) for p in all_prompts},
        "manifest_sha256": sha256_file(args.prompt_dir / "manifest.csv"),
        "profile_sha256": {
            name: sha256_file(args.profile_source / name)
            for name in ("profile.csv", "profile.json")
        },
        "anchor_tokens": anchor_tokens,
        "runtime_env": runtime_env,
        "request": {"n_predict": 1, "cache_prompt": False,
                    "parallel": 1, "context": 2048},
        "inter_prompt_delay_seconds": args.inter_prompt_delay_seconds,
    }
    encoded = json.dumps(
        base, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "configuration": base,
    }


def load_completed_requests(path, prompts, fingerprint):
    """Fail closed on edited, duplicated, mixed, or non-prefix result rows."""
    if not path.is_file():
        return []
    completed = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL row {number} in {path}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row {number} in {path}: {exc}") from exc
            if number > len(prompts) or not isinstance(record, dict):
                raise ValueError("Unexpected extra or invalid result row "
                                 + str(number))
            expected = prompts[number - 1].name
            if (record.get("index") != number or
                    record.get("prompt") != expected or
                    record.get("run_fingerprint") != fingerprint or
                    record.get("cache_n") != 0 or
                    record.get("status") != "PASS"):
                raise ValueError(
                    f"Resume result mismatch at row {number}: expected "
                    f"{expected}, PASS, zero cache, matching fingerprint")
            completed.append(record)
    return completed


def report_rows(records, group_order):
    """Summarize per-prompt rates and token/time-weighted aggregate rate."""
    rows = []
    for group in [*group_order, "ALL"]:
        subset = [r for r in records if group == "ALL" or
                  r["group"] == group]
        expected = 50 if group == "ALL" else 10
        count = len(subset)
        tokens = sum(r["prompt_n"] for r in subset)
        prefill_ms = sum(r["prompt_ms"] for r in subset)
        rates = [r["prefill_tps"] for r in subset]
        decisions = [
            (r["plan_xc"], r["plan_xg"], r["plan_xp"])
            for r in subset if all(
                isinstance(r.get(k), int) and r.get(k) > 0
                for k in ("plan_xc", "plan_xg", "plan_xp"))
        ]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        plan_distribution = ";".join(
            f"XC{p[0]}-XG{p[1]}-XP{p[2]}:{n}"
            for p, n in sorted(decision_counts.items()))
        planner_times = [
            r["planner_ms"] for r in subset
            if isinstance(r.get("planner_ms"), (float, int))
        ]
        row = {
            "group": group,
            "completed": count,
            "expected": expected,
            "status": "PASS" if count == expected else "INCOMPLETE",
            "total_tokens": tokens,
            "total_prefill_ms": round(prefill_ms, 3),
            "weighted_tps": round(1000 * tokens / prefill_ms, 3)
                if prefill_ms else "",
            "median_tps": round(statistics.median(rates), 3)
                if rates else "",
            "mean_tps": round(statistics.mean(rates), 3)
                if rates else "",
            "min_tps": round(min(rates), 3) if rates else "",
            "max_tps": round(max(rates), 3) if rates else "",
            "median_prefill_ms": round(
                statistics.median([r["prompt_ms"] for r in subset]), 3)
                if subset else "",
            "chunk_plan_variants": len(decision_counts)
                if decisions else "",
            "chunk_plan_distribution": plan_distribution,
            "mean_planner_ms": round(statistics.mean(planner_times), 3)
                if planner_times else "",
            "median_planner_ms": round(statistics.median(planner_times), 3)
                if planner_times else "",
        }
        rows.append(row)
    return rows


def write_reports(out, records):
    """Rebuild readable tables after every PASS, including after resume."""
    out.mkdir(parents=True, exist_ok=True)
    detail_fields = [
        "index", "group", "prompt", "manifest_n", "prompt_n",
        "token_delta", "cache_n", "prompt_ms", "prefill_tps",
        "request_wall_ms", "predicted_n", "planner_mode",
        "plan_xc", "plan_xg", "plan_xp", "planner_ms",
        "runtime_xc", "runtime_xg", "runtime_xp",
        "runtime_layout", "status",
    ]
    with (out / "requests.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    rows = report_rows(records, runner.MAIN50_SHORT_GROUPS)
    with (out / "summary.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps({"status": "PASS" if len(records) == 50
                    else "INCOMPLETE",
                    "completed": len(records), "expected": 50,
                    "groups": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def prepare_sweep_output(args, config, prompts):
    """Never silently overwrite a previous 50-Prompt benchmark."""
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "server-run-config.json"
    metrics_path = args.out / "requests.jsonl"
    if not args.resume and (config_path.exists() or metrics_path.exists()):
        raise ValueError(
            "Existing --all50 results would be overwritten. "
            "Use --resume for the exact same experiment or a fresh --out.")
    if args.resume and (config_path.exists() != metrics_path.exists()):
        # Before the first request, a run may have created the config but
        # not the JSONL. That is safe; a JSONL without a config is not.
        if metrics_path.exists():
            raise ValueError(
                "Cannot resume requests.jsonl without server-run-config.json")
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != config["fingerprint"]:
            raise ValueError(
                "Resume configuration changed (binary, model, Profile, "
                "prompts, Chunk flags or delay). Use a new --out.")
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    completed = load_completed_requests(
        metrics_path, prompts, config["fingerprint"])
    write_reports(args.out, completed)
    return completed


def sweep_log_path(out, resume):
    """A resumed server keeps earlier logs for postmortem comparisons."""
    first = out / "llama-server.log"
    if not resume or not first.exists():
        return first
    index = 1
    while True:
        candidate = out / f"llama-server-resume-{index:03d}.log"
        if not candidate.exists():
            return candidate
        index += 1


def main(argv=None):
    args = parse_args(argv)
    # Avoid accidentally connecting to another unrelated server on this port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port already in use: " + str(args.port))

    source = args.profile_source / "profile.csv"
    runner.verify_independent_profile(source, short_chunks=True)
    anchor = args.prompt_dir / "prompt_1024_10.txt"
    manifest = runner.read_manifest(args.prompt_dir)
    if anchor.name not in manifest:
        raise ValueError("Anchor token count absent from manifest.csv")
    anchor_tokens = int(manifest[anchor.name])
    if not 32 <= anchor_tokens < 2048:
        raise ValueError("Invalid 2048-context anchor token count")
    prompts = prompt_files(args.prompt_dir, args.all50)
    if args.all50:
        # Refuse a mismatched or truncated dataset before any expensive
        # model load. Group 0 must remain the real 64..128-token group.
        for path in prompts:
            tokens = manifest.get(path.name)
            if tokens is None or not 64 <= int(tokens) < 2048:
                raise ValueError(
                    f"Missing/out-of-context manifest token count: {path.name}={tokens}")
            if group_for_prompt(path.name) == 0 and not 64 <= int(tokens) <= 128:
                raise ValueError(
                    f"Group 0 must be 64..128 actual tokens: {path.name}={tokens}")

    # Reuse the tested runner's full Phone HTP / CUDA / RPC environment.
    # The independent Profile may contain old X=8 entries but the startup
    # locked Chunk search is capped at XG=XP=32, and never selects X=8.
    fake_args = runner.get_args([
        "--exe", str(args.server_exe), "--model", str(args.model),
        "--model-layers", "64", "--rpc", args.rpc,
        "--prompt-dir", str(args.prompt_dir), "--out", str(args.out),
        "--repo-root", str(Path(__file__).resolve().parents[2]),
        "--auto-plan", "--htp-only", "--streaming", "on",
        "--sharegpt50-short-dataset",
        "--locked-layout", LAYOUT, "--locked-short-chunks",
        "--skip-plan-memory-checks",
        "--max-plan-xg", str(args.max_plan_xg),
        "--max-plan-xp", str(args.max_plan_xp),
        "--batch", "2048", "--ubatch", "2048", "--n-predict", "1",
        "--threads", str(args.threads), "--min-context", "256",
        "--max-context", "2048", "--context-padding", "0",
    ])
    command = server_command(args.server_exe, args.model, args.rpc,
                             args.port, args.threads)
    # Model initialization uses the long-Prompt anchor, while the opt-in
    # request hook selects new Chunk widths for each fresh HTTP Prefill.
    target_profile = args.out / "independent-profile" / "profile.csv"
    if not args.dry_run:
        target_profile.parent.mkdir(parents=True, exist_ok=True)
        for name in ("profile.csv", "profile.json"):
            src = args.profile_source / name
            dst = target_profile.parent / name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
    else:
        target_profile = source
    fake_args.real_stage_profile_csv = target_profile
    _, env = runner.build_process(fake_args, anchor, 2048,
                                  anchor_tokens, args.out / "metrics.jsonl")
    # Explicitly guard against inheriting a different user's runtime flags.
    assert env["LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT"] == (
        "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7")
    assert env["LLAMA_REGION_RING_LOCKED_SKIP_MEMORY_CHECKS"] == "1"
    assert env["LLAMA_REGION_RING_LOCKED_MAX_XG"] == str(args.max_plan_xg)
    assert env["LLAMA_REGION_RING_LOCKED_MAX_XP"] == str(args.max_plan_xp)
    assert env["LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS"] == "1"
    if args.per_request_chunks:
        env["LLAMA_REGION_RING_PER_REQUEST_CHUNKS"] = "1"
    else:
        # Do not inherit a previous experimental run's per-request flag.
        env.pop("LLAMA_REGION_RING_PER_REQUEST_CHUNKS", None)
    print("[SERVER_PROBE] policy=" +
          ("PER_REQUEST_NATIVE_LOCKED_CHUNK_SEARCH" if args.per_request_chunks
           else "ONE_STARTUP_PLAN NOT_PER_REQUEST_CHUNK_SEARCH") +
          " layout=" + env["LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT"], flush=True)
    print(f"[SERVER_PROBE] chunk=XG_MAX{args.max_plan_xg}_XP_MAX{args.max_plan_xp} min16_short_min32_long"
          " kv_cache=DISABLED parallel=1 no_cont_batch=1"
          " memory_estimates=SKIPPED allocators=UNCHANGED", flush=True)
    print("[SERVER_PROBE] command=" + subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "requests.jsonl"
    completed = []
    if args.all50:
        config = sweep_config(args, command, env, prompts, anchor_tokens)
        completed = prepare_sweep_output(args, config, prompts)
        selected_count = args.limit or len(prompts)
        print(f"[SERVER_SWEEP] experiment=ALL50_UNIQUE"
              f" long_anchor=FIRST selected={selected_count}/50"
              f" existing_pass={len(completed)}"
              f" resume={int(args.resume)}"
              f" delay_seconds={args.inter_prompt_delay_seconds:.1f}"
              f" fingerprint={config['fingerprint'][:12]}", flush=True)
        if len(completed) >= selected_count:
            print(f"[SERVER_SWEEP] status="
                  f"{'PASS' if len(completed) == 50 else 'PARTIAL'}"
                  f" completed={len(completed)}/50"
                  f" no_new_requests=1 results={metrics_path}", flush=True)
            return 0

    log_path = sweep_log_path(args.out, args.resume)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(command, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_ready(proc, args.port, args.startup_timeout)
            print("[SERVER_PROBE] health=READY model=RESIDENT", flush=True)
            open_mode = "a" if args.all50 and args.resume else "w"
            with metrics_path.open(open_mode, encoding="utf-8") as output:
                requested_this_process = 0
                log_offset = 0
                for i, path in enumerate(prompts, 1):
                    if args.all50 and args.limit and i > args.limit:
                        break
                    if args.all50 and i <= len(completed):
                        continue
                    if (requested_this_process > 0 and
                            args.inter_prompt_delay_seconds > 0):
                        time.sleep(args.inter_prompt_delay_seconds)
                    prompt = path.read_text(encoding="utf-8-sig")
                    began = time.monotonic()
                    record = post_completion(args.port, prompt)
                    # Do not charge CSV/log parsing to the HTTP request wall.
                    http_wall_ms = round((time.monotonic() - began) * 1000, 3)
                    if args.per_request_chunks:
                        decision, log_offset = read_request_chunk_plan(
                            log_path, log_offset, record["prompt_n"],
                            args.max_plan_xg, args.max_plan_xp)
                        record.update(decision)
                    record.update({
                        "prompt": path.name, "index": i,
                        "group": group_for_prompt(path.name),
                        "manifest_n": int(manifest[path.name]),
                        "token_delta": record["prompt_n"] -
                                       int(manifest[path.name]),
                        "request_wall_ms": http_wall_ms,
                        "planner_mode": (
                            "PER_REQUEST_LOCKED" if args.per_request_chunks
                            else "STARTUP_ONLY"),
                        "runtime_layout": fake_args.locked_layout,
                        "status": "PASS",
                    })
                    if args.all50:
                        record["run_fingerprint"] = config["fingerprint"]
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    requested_this_process += 1
                    if args.all50:
                        completed.append(record)
                        write_reports(args.out, completed)
                    print(f"[SERVER_PROBE] {i}/{len(prompts)}"
                          f" prompt={path.name}"
                          f" group={record['group']}"
                          f" prompt_n={record['prompt_n']}"
                          f" cache_n={record['cache_n']}"
                          f" prefill_ms={record['prompt_ms']:.3f}"
                          f" tps={record['prefill_tps']:.3f}"
                          + (f" XC={record['plan_xc']}"
                             f" XG={record['plan_xg']}"
                             f" XP={record['plan_xp']}"
                             f" actual_XG={record['runtime_xg']}"
                             f" actual_XP={record['runtime_xp']}"
                             f" planner_ms={record['planner_ms']:.3f}"
                             if args.per_request_chunks else ""),
                          flush=True)
            if args.all50:
                finished = len(completed) == 50
                print(f"[SERVER_SWEEP] status={'PASS' if finished else 'PARTIAL'}"
                      f" completed={len(completed)}/50"
                      f" requests={metrics_path}"
                      f" summary={args.out / 'summary.csv'}"
                      f" details={args.out / 'requests.csv'}"
                      f" log={log_path}", flush=True)
            else:
                print(f"[SERVER_PROBE] status=PASS results={metrics_path}",
                      flush=True)
            return 0
        except Exception:
            print("[SERVER_PROBE] status=FAIL first_error_markers:\n"
                  + first_error_markers(log_path),
                  file=sys.stderr, flush=True)
            print("[SERVER_PROBE] server_log_tail:\n"
                  + log_tail(log_path), file=sys.stderr, flush=True)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[SERVER_PROBE] error={error}", file=sys.stderr)
        sys.exit(1)
