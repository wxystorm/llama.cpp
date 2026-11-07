#!/usr/bin/env python3
"""Resumable 40-prompt ShareGPT benchmark for PRIMA rank 0.

Each prompt is an independent llama-cli process. Only a successful exit with
parsed prompt-evaluation timing counts as PASS. Interrupted/failed cases retry.
On phone, run phone_worker_loop.sh to restart rank 1 between cases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


LENGTHS = (128, 256, 512, 1024)
FIELDNAMES = (
    "case_id", "group", "prompt_file", "status",
    "actual_prefill_tokens", "prefill_ms", "prefill_tps",
    "decode_tokens", "decode_ms", "decode_tps",
    "planner_k", "planner_W",
    "pc_layers", "pc_gpu_layers", "phone_layers", "phone_gpu_layers",
    "assignment_json", "exit_code", "wall_s", "started_at", "finished_at",
    "log_path", "config_signature", "error",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_perf(log: str) -> dict:
    out: dict = {}
    for line in log.splitlines():
        is_prompt = "prompt eval time" in line
        is_decode = re.search(r"\beval time\s*=", line) and not is_prompt
        if not is_prompt and not is_decode:
            continue
        m = re.search(r"eval time\s*=\s*([\d,.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens?|runs?)", line)
        t = re.search(r"([\d,.]+)\s*tokens?\s+per\s+second", line)
        if not m:
            continue
        prefix = "prefill" if is_prompt else "decode"
        out[prefix + "_ms"] = parse_number(m.group(1))
        out[("actual_prefill_tokens" if is_prompt else "decode_tokens")] = int(m.group(2))
        if t:
            out[prefix + "_tps"] = parse_number(t.group(1))
        elif parse_number(m.group(1)) > 0:
            out[prefix + "_tps"] = int(m.group(2)) * 1000.0 / parse_number(m.group(1))

    p = re.search(r"(?m)^\s*-\s*k\s*=\s*(\d+)\s*$", log)
    w = re.search(r"(?m)^\s*-\s*W\s*=\s*(\d+)\s*$", log)
    if p:
        out["planner_k"] = int(p.group(1))
    if w:
        out["planner_W"] = int(w.group(1))

    assignments: dict = {}
    matcher = re.compile(
        r"(?m)^([^\n:]+):\s*\n"
        r"\s*-\s*Device Index\s*:\s*(\d+)\s*\n"
        r"\s*-\s*Assignment Set\s*:\s*(M\d+)\s*\n"
        r"\s*-\s*N Layer Window\s*:\s*(\d+)\s*\n"
        r"\s*-\s*N GPU Layers\s*:\s*(\d+)"
    )
    for m in matcher.finditer(log):
        rank = int(m.group(2))
        layers = int(m.group(4))
        gpu_layers = int(m.group(5))
        assignments[str(rank)] = {
            "name": m.group(1).strip(),
            "set": m.group(3),
            "layers": layers,
            "gpu_layers": gpu_layers,
        }
        if rank == 0:
            out["pc_layers"], out["pc_gpu_layers"] = layers, gpu_layers
        if rank == 1:
            out["phone_layers"], out["phone_gpu_layers"] = layers, gpu_layers
    if assignments:
        out["assignment_json"] = json.dumps(assignments, ensure_ascii=False, sort_keys=True)
    return out


def write_summary(output: Path) -> None:
    records = []
    for length in LENGTHS:
        for index in range(1, 11):
            case_id = f"prompt_{length}_{index:02d}"
            path = output / "cases" / (case_id + ".json")
            if path.exists():
                try:
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    records.append({"case_id": case_id, "group": length, "status": "CORRUPT"})
    tmp = output / "summary.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, output / "summary.csv")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4 groups x 10 ShareGPT files with resumable results.")
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("./llama-cli"))
    parser.add_argument("--out", type=Path, default=Path("./sharegpt40-prima-results"))
    parser.add_argument("--master", required=True, help="PC address reachable from phone")
    parser.add_argument("--next", required=True, help="Phone address reachable from PC")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per case; 0 means unlimited")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds before next case, allowing phone restart")
    parser.add_argument("--force", action="store_true", help="rerun even PASS with matching config")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any(x in extra for x in ("-m", "--model", "-f", "--file", "--world", "--rank", "--master", "--next")):
        parser.error("Do not override model/prompt/world/rank/network in extra args")
    args.extra = extra
    if args.ctx < 8 or args.predict < 1 or args.timeout < 0 or args.delay < 0:
        parser.error("Invalid ctx/predict/timeout/delay")
    return args


def main() -> int:
    args = build_args()
    prompt_dir = args.prompt_dir.expanduser().resolve()
    model = args.model.expanduser().resolve()
    binary = args.binary.expanduser().resolve()
    output = args.out.expanduser().resolve()
    expected = [
        (length, f"prompt_{length}_{index:02d}", prompt_dir / f"prompt_{length}_{index:02d}.txt")
        for length in LENGTHS
        for index in range(1, 11)
    ]
    missing = [str(path) for _, _, path in expected if not path.is_file()]
    if missing:
        print("[ERROR] Prompt files missing:\n" + "\n".join(missing), file=sys.stderr)
        return 2
    if not model.is_file() or not binary.is_file():
        print(f"[ERROR] Missing binary or model: {binary}, {model}", file=sys.stderr)
        return 2

    model_stat = model.stat()
    signature_data = {
        "model": str(model),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "binary": str(binary),
        "binary_mtime_ns": binary.stat().st_mtime_ns,
        "master": args.master, "next": args.next,
        "ctx": args.ctx, "predict": args.predict, "extra": args.extra,
    }
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()[:16]
    if not args.dry_run:
        (output / "cases").mkdir(parents=True, exist_ok=True)
        (output / "logs").mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(signature_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_summary(output)
    print(f"[PRIMA_SWEEP] prompts=40 signature={signature} output={output}", flush=True)

    completed, failed = 0, 0
    for index, (length, case_id, prompt_path) in enumerate(expected, start=1):
        result_path = output / "cases" / f"{case_id}.json"
        old = None
        if result_path.is_file():
            try:
                old = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if not args.force and old and old.get("status") == "PASS" and old.get("config_signature") == signature:
            completed += 1
            print(f"[{index:02d}/40] SKIP {case_id} (PASS)", flush=True)
            continue

        log_path = output / "logs" / f"{case_id}.log"
        command = [
            str(binary), "-m", str(model), "-f", str(prompt_path),
            "-c", str(args.ctx), "-n", str(args.predict),
            "--world", "2", "--rank", "0",
            "--master", args.master, "--next", args.next,
            "--prefetch",
        ] + args.extra
        print(f"[{index:02d}/40] RUN {case_id} (group={length})", flush=True)
        if args.dry_run:
            print(" ".join(repr(x) for x in command), flush=True)
            continue

        result = {
            "case_id": case_id, "group": length, "prompt_file": str(prompt_path),
            "status": "RUNNING", "started_at": now_iso(), "config_signature": signature,
            "log_path": str(log_path), "error": "",
        }
        atomic_json(result_path, result)
        write_summary(output)
        started = time.monotonic()
        proc = None
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
                log_handle.write("[PRIMA_SWEEP_COMMAND] " + json.dumps(command, ensure_ascii=False) + "\n")
                log_handle.flush()
                proc = subprocess.Popen(command, cwd=str(binary.parent), stdout=log_handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=args.timeout if args.timeout else None)
                except subprocess.TimeoutExpired:
                    terminate(proc)
                    code = 124
                    result["error"] = f"TIMEOUT_{args.timeout}s"
            log = log_path.read_text(encoding="utf-8", errors="replace")
            result.update(parse_perf(log))
            result["exit_code"] = code
            if code == 0 and result.get("actual_prefill_tokens", 0) > 0 and result.get("prefill_ms", 0) > 0:
                result["status"] = "PASS"
                completed += 1
            else:
                result["status"] = "FAIL"
                result["error"] = result["error"] or ("MISSING_PREFILL_METRICS" if code == 0 else f"EXIT_{code}")
                failed += 1
        except KeyboardInterrupt:
            if proc is not None:
                terminate(proc)
            result["status"] = "INTERRUPTED"
            result["exit_code"] = 130
            result["error"] = "CTRL_C"
            result["wall_s"] = round(time.monotonic() - started, 3)
            result["finished_at"] = now_iso()
            atomic_json(result_path, result)
            write_summary(output)
            print("\n[PRIMA_SWEEP] interrupted; rerun the same command to resume.", flush=True)
            return 130
        except (OSError, RuntimeError) as exc:
            if proc is not None:
                terminate(proc)
            result["status"] = "FAIL"
            result["error"] = repr(exc)
            failed += 1
        result["wall_s"] = round(time.monotonic() - started, 3)
        result["finished_at"] = now_iso()
        atomic_json(result_path, result)
        write_summary(output)
        print(f"[{index:02d}/40] {result['status']} {case_id} "
              f"tokens={result.get('actual_prefill_tokens', '?')} "
              f"prefill={result.get('prefill_tps', '?')} tok/s "
              f"wall={result['wall_s']}s log={log_path}", flush=True)
        if result["status"] != "PASS" and not args.continue_on_error:
            print("[PRIMA_SWEEP] stopped after failure; inspect log and rerun to retry.", flush=True)
            return 1
        if index < len(expected) and args.delay:
            time.sleep(args.delay)

    if not args.dry_run:
        print(f"[PRIMA_SWEEP] completed={completed}/40 failed={failed}; CSV={output / 'summary.csv'}")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
