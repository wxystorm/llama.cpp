#!/usr/bin/env python3
"""32B final planner ablation: memory-proportional G40/P24, native DAG chunks.

The harness skips *estimated* memory guards but never changes the real
CUDA, host, or Phone HTP allocators. It records actual OOM evidence if the
resident server exits during startup or any of the 50 requests.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

LAYOUT = "G7,P5,G7,P5,G7,P5,G7,P5,G7,P4,G5"

# A crash without one of these first-cause markers is NOT proof of OOM.
OOM_RE = re.compile(
    r"out of memory|\bENOMEM\b|std::bad_alloc|cudaErrorMemoryAllocation|"
    r"CUDA_ERROR_OUT_OF_MEMORY|CL_OUT_OF_HOST_MEMORY|"
    r"CL_MEM_OBJECT_ALLOCATION_FAILURE|GGML_STATUS_ALLOC_FAILED|"
    r"HTP_SESSION_MEMORY_INFEASIBLE|memory allocation failed|"
    r"not enough memory|failed to allocate|cannot allocate memory",
    re.IGNORECASE,
)
MEMORY_EXIT_CODES = {0xC0000017, 0xC000009A}


def classify_failure(out: Path, exit_code: int):
    """Read the most recent real server log and preserve the failure evidence."""
    logs = sorted(out.glob("llama-server*.log"),
                  key=lambda path: path.stat().st_mtime_ns)
    log = logs[-1] if logs else None
    evidence = []
    if log is not None:
        with log.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if OOM_RE.search(line):
                    evidence.append(line.rstrip()[:700])
                    if len(evidence) == 15:
                        break
    if (exit_code & 0xFFFFFFFF) in MEMORY_EXIT_CODES:
        evidence.append("Windows memory exit code: " +
                        hex(exit_code & 0xFFFFFFFF))

    records = out / "requests.jsonl"
    passed = 0
    if records.exists():
        with records.open("r", encoding="utf-8") as stream:
            passed = sum(1 for line in stream if line.strip())
    failed_prompt = None
    cfg = out / "server-run-config.json"
    if cfg.exists():
        try:
            names = json.loads(cfg.read_text(encoding="utf-8")).get(
                "configuration", {}).get("prompts", [])
            if passed < len(names):
                failed_prompt = names[passed]
        except (ValueError, KeyError, TypeError):
            pass

    result = {
        "status": "OOM" if evidence else "CRASH_UNCONFIRMED",
        "experiment": "WO_PROFILE_GUIDED_PLACEMENT_G40_P24",
        "layout": LAYOUT,
        "completed_pass": passed,
        "failed_prompt": failed_prompt,
        "wrapper_exit_code": exit_code,
        "server_log": str(log) if log else None,
        "oom_evidence": evidence,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (out / "failures.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main(argv=None):
    root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rpc", required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--profile-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    script = args.root / "tools" / "region-ring-experiments" / "probe_htp_llama_server.py"
    server = args.root / "build-tensor-v2" / "bin" / "Release" / "llama-server.exe"
    for path in (script, server, args.model, args.prompt_dir,
                 args.profile_source / "profile.csv",
                 args.profile_source / "profile.json"):
        if not path.exists():
            parser.error("required file/directory not found: " + str(path))
    if not 0 <= args.limit <= 50:
        parser.error("--limit must be 0..50")

    command = [
        sys.executable, str(script),
        "--server-exe", str(server),
        "--model", str(args.model),
        "--model-layers", "64",
        "--locked-layout", LAYOUT,
        "--rpc", args.rpc,
        "--prompt-dir", str(args.prompt_dir),
        "--profile-source", str(args.profile_source),
        "--out", str(args.out),
        "--port", str(args.port),
        "--threads", str(args.threads),
        "--all50", "--per-request-chunks",
        "--memory-proportional-placement",
    ]
    if args.resume:
        command.append("--resume")
    if args.limit:
        command += ["--limit", str(args.limit)]
    print("[MEMORY_PROPORTIONAL] layout=" + LAYOUT +
          " GPU=40 PHONE_HTP=24 CPU_COMPUTE=0"
          " scoring=DAG chunks=DYNAMIC"
          " bank=2 prefetch=ASYNC drain_first=1"
          " planner_memory_checks=SKIPPED"
          " runtime_allocators=UNCHANGED", flush=True)
    status = subprocess.run(command, cwd=str(args.root),
                            check=False).returncode
    if status:
        failure = classify_failure(args.out, status)
        print("[MEMORY_PROPORTIONAL] status=" + failure["status"] +
              " passed=" + str(failure["completed_pass"]) +
              " failed_prompt=" + str(failure["failed_prompt"]) +
              " details=" + str(args.out / "failure.json"),
              file=sys.stderr, flush=True)
        return status if status > 0 else 1
    print("[MEMORY_PROPORTIONAL] status=PASS"
          " summary=" + str(args.out / "summary.csv"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
