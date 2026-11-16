#!/usr/bin/env python3
"""Experimental resident llama-server smoke test for HTP-only Ring Prefill.

Unlike run_32b_all40.py, the server keeps ONE startup Chunk Plan for every
request. This is NOT a replacement for the per-Prompt Chunk-only main study.
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
import json
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
                   help="experimental 50-Prompt server sweep with ONE fixed Chunk Plan")
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
    return args


def prompt_files(folder, all50):
    if not all50:
        names = ["prompt_1024_10.txt", "prompt_0_01.txt",
                 "prompt_1024_10.txt"]
    else:
        names = [
            f"prompt_{group}_{i:02d}.txt"
            for group in runner.MAIN50_SHORT_GROUPS for i in range(1, 11)
        ] + ["prompt_1024_10.txt"]
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
        r"\\[(?:REGION_RING_WORKER|PHONE_INPUT_PUSH|PHONE_OWNED_PLAN|"
        r"PHONE_HTP_ONLY|PHONE_AUTO|REGION_RING_GPU_BANK_MOVER|"
        r"REGION_RING_P2C_JOIN)\\].*"
        r"(?:event=EXCEPTION|event=FAIL|status=FAIL|status=REJECT|reason=)|"
        r"\\[REGION_RING\\].*(?:event=DEADLOCK|event=FAIL)|"
        r"ggml_.*(?:error -?[0-9]+|failed)",
        re.IGNORECASE,
    )
    secondary = re.compile(
        r"\\[(?:P2C_STREAM|REGION_RING_SUM|PHONE_INPUT_STREAM)\\]"
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
        "--max-plan-xg", "32", "--max-plan-xp", "32",
        "--batch", "2048", "--ubatch", "2048", "--n-predict", "1",
        "--threads", str(args.threads), "--min-context", "256",
        "--max-context", "2048", "--context-padding", "0",
    ])
    command = server_command(args.server_exe, args.model, args.rpc,
                             args.port, args.threads)
    # A persistent server cannot re-plan XC/XG/XP on each HTTP request:
    # model initialization sees this fixed long-Prompt planning size.
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
    assert env["LLAMA_REGION_RING_LOCKED_MAX_XG"] == "32"
    assert env["LLAMA_REGION_RING_LOCKED_MAX_XP"] == "32"
    assert env["LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS"] == "1"
    print("[SERVER_PROBE] policy=ONE_STARTUP_PLAN NOT_PER_REQUEST_CHUNK_SEARCH"
          " layout=" + env["LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT"], flush=True)
    print("[SERVER_PROBE] chunk=XG_XP_MAX32 min16"
          " kv_cache=DISABLED parallel=1 no_cont_batch=1"
          " memory_estimates=SKIPPED allocators=UNCHANGED", flush=True)
    print("[SERVER_PROBE] command=" + subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "llama-server.log"
    metrics_path = args.out / "requests.jsonl"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(command, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_ready(proc, args.port, args.startup_timeout)
            print("[SERVER_PROBE] health=READY model=RESIDENT", flush=True)
            with metrics_path.open("w", encoding="utf-8") as output:
                for i, path in enumerate(prompts, 1):
                    prompt = path.read_text(encoding="utf-8-sig")
                    began = time.monotonic()
                    record = post_completion(args.port, prompt)
                    record.update({
                        "prompt": path.name, "index": i,
                        "request_wall_ms": round(
                            (time.monotonic() - began) * 1000, 3),
                        "planner_mode": "STARTUP_ONLY",
                        "runtime_layout": fake_args.locked_layout,
                    })
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"[SERVER_PROBE] {i}/{len(prompts)}"
                          f" prompt={path.name}"
                          f" prompt_n={record['prompt_n']}"
                          f" cache_n={record['cache_n']}"
                          f" prefill_ms={record['prompt_ms']:.3f}"
                          f" tps={record['prefill_tps']:.3f}", flush=True)
            print(f"[SERVER_PROBE] status=PASS results={metrics_path}", flush=True)
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
