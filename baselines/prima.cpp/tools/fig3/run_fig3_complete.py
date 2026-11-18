#!/usr/bin/env python3
"""One-profile + clean normal Prefill, 2-rank Prima Fig.3 collector.

Phone: start phone_worker_fig3.sh using the EXACT same run_id BEFORE PC.
PC: run this on WSL Rank 0. The first session is profiled, followed by
one clean warmup and three clean timings. --analyze-only never runs inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from analyze_fig3 import inspect
from layout_guard import (parse_fixed, local_layer_counts, summarize_log,
                          validate_layout, compare_runs)

PERF = re.compile(r"prompt eval time\s*=\s*([\d,.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens?|runs?)",
                  flags=re.I)

def timing(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = PERF.findall(text)
    if not matches:
        raise RuntimeError(f"No normal prompt-eval time in {path}")
    ms, tokens = matches[-1]
    return {"prefill_ms": float(ms.replace(",", "")), "tokens": int(tokens)}


def atomic_json(path, doc):
    tmp = Path(str(path)+".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def clock_sample(host, port, phase):
    with socket.create_connection((host, port), timeout=4) as conn:
        conn.settimeout(4)
        t0 = time.monotonic_ns()
        conn.sendall(b"PING\n")
        buf = bytearray()
        while not buf.endswith(b"\n") and len(buf) < 80:
            part = conn.recv(80 - len(buf))
            if not part:
                raise RuntimeError("Phone clock responder disconnected")
            buf.extend(part)
        t3 = time.monotonic_ns()
    t1, t2 = (int(v) for v in buf.decode("ascii").strip().split(","))
    if t2 < t1:
        raise RuntimeError("Phone clock reversed")
    delta = max(0, t3 - t0 - (t2 - t1))
    return {
        "phase": phase, "pc_t0": t0, "phone_t1": t1,
        "phone_t2": t2, "pc_t3": t3,
        "offset_phone_minus_pc_ns": ((t1 + t2) - (t0 + t3)) // 2,
        "uncertainty_ns": delta // 2,
    }


def clock_batch(host, port, phase):
    result = []
    for i in range(16):
        result.append(clock_sample(host, port, phase))
        if i < 15:
            time.sleep(0.025)
    quality = min(x["uncertainty_ns"] for x in result) / 1e6
    if quality > 20:
        raise RuntimeError(f"Phone clock uncertainty {quality:.2f} ms > 20 ms")
    print(f"[PRIMA_FIG3] clock {phase}: 16 samples; best +/- {quality:.3f} ms", flush=True)
    return result


def clock_during_profile(args, stop, result, failures):
    """Keep sparse four-stamp observations DURING GPU/Phone activity."""
    while not stop.is_set():
        try:
            result.append(clock_sample(args.clock_host, args.clock_port, "during"))
        except Exception as error:
            failures.append(str(error)[:180])
        stop.wait(args.clock_interval)


def start_clock_sampling(args):
    stop = threading.Event()
    result = []
    failures = []
    thread = threading.Thread(
        target=clock_during_profile, args=(args, stop, result, failures),
        daemon=True, name="prima_fig3_clock")
    thread.start()
    return stop, thread, result, failures


def build_cmd(args):
    return [
        str(args.binary.resolve()), "-m", str(args.model.resolve()),
        "-f", str(args.prompt.resolve()), "-c", str(args.context), "-n", "1",
        "--world", "2", "--rank", "0",
        "--master", args.master, "--next", args.next,
        "--prefetch", "-b", str(args.batch), "-ub", str(args.ubatch),
        "-t", str(args.threads),
    ] + (["-lw", args.fixed_layers, "-ngl", str(args.pc_ngl)]
         if args.fixed_layers else []) + args.extra


def launch(args, cmd, log, env):
    print("[PRIMA_FIG3] START", log.name, flush=True)
    with log.open("wb") as handle:
        proc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL,
                              stdout=handle, stderr=subprocess.STDOUT,
                              timeout=args.timeout if args.timeout else None)
    if proc.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"llama-cli status={proc.returncode}: {log}\n{tail}")
    value = timing(log)
    value["layout"] = summarize_log(log)
    expected = (parse_fixed(args.fixed_layers) + (args.pc_ngl,)
                if args.fixed_layers else None)
    validate_layout(value["layout"], expected)
    print("[PRIMA_FIG3] EFFECTIVE PLACEMENT:", value["layout"]["effective"], flush=True)
    print("[PRIMA_FIG3] PASS %s %d tokens %.3f ms" %
          (log.name, value["tokens"], value["prefill_ms"]), flush=True)
    return value


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("./llama-cli"))
    p.add_argument("--model", type=Path, default=Path("/mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf"))
    p.add_argument("--prompt", type=Path, default=Path("/mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt"))
    p.add_argument("--run-id", default="prima-fig3-14b")
    p.add_argument("--out", type=Path, default=Path("./fig3-prima14b-complete"))
    p.add_argument("--master", default="10.238.229.214")
    p.add_argument("--next", default="10.238.229.239")
    p.add_argument("--clock-host", default="10.238.229.239")
    p.add_argument("--clock-port", type=int, default=19139)
    p.add_argument("--clock-interval", type=float, default=0.5,
                   help="during-Profile clock sampling interval in seconds")
    p.add_argument("--clock-mode",
                   choices=["strict", "affine-diagnostic", "piecewise-diagnostic"],
                   default="strict",
                   help="diagnostic modes are NEVER paper-ready; "
                        "piecewise-diagnostic preserves dense anchor evidence "
                        "but flags all clock steps")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--fixed-layers", default="",
                   help="manual deterministic replay, e.g. 39,1; keeps Phone active")
    p.add_argument("--pc-ngl", type=int, default=None,
                   help="PC CUDA layers for --fixed-layers, e.g. 21")
    p.add_argument("--model-layers", type=int, default=0,
                   help="full model layer count, e.g. 64 for 32B; manual -lw "
                        "windows may repeat across multiple ring cycles")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-only", action="store_true",
                   help="run one instrumented Profile only, retaining raw evidence; "
                        "resume later for normal timing")
    p.add_argument("--trace-tmp-dir", type=Path, default=Path("/tmp"),
                   help="Linux native filesystem for temporary Rank0 CSV; "
                        "do NOT use /mnt/c or /mnt/d on WSL")
    p.add_argument("--between", type=float, default=30.0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--phone-user", default="u0_a283")
    p.add_argument("--ssh-port", type=int, default=8022)
    p.add_argument("--phone-file", default="")
    p.add_argument("--skip-scp", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any(x in extra for x in ("-m", "--model", "-f", "--file", "--world",
                                "--rank", "--master", "--next", "-n", "--predict",
                                "-lw", "--layer-window", "--n-layer-window",
                                "-ngl", "--n-gpu-layers")):
        p.error("Do not override the comparison model, prompt, topology, or predict count")
    args.extra = extra
    if bool(args.fixed_layers) != (args.pc_ngl is not None):
        p.error("--fixed-layers and --pc-ngl must be supplied together")
    if args.fixed_layers:
        try:
            pc, phone = parse_fixed(args.fixed_layers, args.model_layers)
        except ValueError as err:
            p.error(str(err))
        if args.pc_ngl < 0:
            p.error("--pc-ngl must be nonnegative")
        pc_actual = (local_layer_counts(pc, phone, args.model_layers)[0]
                     if args.model_layers else pc)
        if args.pc_ngl > pc_actual:
            p.error("--pc-ngl exceeds PC's estimated layers across all ring "
                    "cycles; pass --model-layers for multi-cycle layouts")
    if not (0 < args.ubatch <= args.batch <= args.context):
        p.error("need 0 < ubatch <= batch <= context")
    if args.clock_interval < 0.5 or args.clock_interval > 5.0:
        p.error("clock-interval must be between 0.5 and 5 seconds")
    if args.warmups < 0 or args.repeats < 3 or args.between < 0:
        p.error("need >=3 normal repetitions, >=0 warmups")
    if not args.trace_tmp_dir.is_dir():
        p.error("--trace-tmp-dir must be an existing directory")
    if str(args.trace_tmp_dir.resolve()).startswith(("/mnt/c/", "/mnt/d/", "/mnt/e/")):
        p.error("--trace-tmp-dir must be on a native Linux filesystem, not Windows DrvFs")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.run_id):
        p.error("unsafe run-id")
    return args


def finalize(args, root, manifest):
    phone = root/"phone_events.csv"
    if args.phone_file:
        src = Path(args.phone_file).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        if src != phone.resolve():
            shutil.copyfile(src, phone)
    elif not phone.exists() and not args.skip_scp:
        remote_file = (f"/data/data/com.termux/files/home/prima-fig3-phone/"
                       f"{manifest['run_id']}/phone.csv")
        remote = f"{args.phone_user}@{args.next}:{remote_file}"
        cmd = ["scp", "-P", str(args.ssh_port), remote, str(phone)]
        print("[PRIMA_FIG3] SCP", remote, flush=True)
        try:
            process = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
            if process.returncode:
                print("[PRIMA_FIG3] SCP unavailable:", process.stderr.strip(), flush=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print("[PRIMA_FIG3] SCP unavailable:", error, flush=True)
    if not phone.exists():
        manifest["status"] = "WAITING_FOR_PHONE_CSV"
        atomic_json(root/"manifest.json", manifest)
        print("[PRIMA_FIG3] NEED_PHONE_ONLY. Do not rerun inference. Copy Phone CSV and run --analyze-only.")
        return
    if not (root/"pc_events.csv").exists():
        raise RuntimeError("Missing PC raw trace; do not produce invented Fig.3 proportions")
    if all("layout" in r for r in [manifest.get("profile") or {}] +
           list(manifest.get("warmup", [])) + list(manifest.get("normal", []))):
        compare_runs(manifest["profile"], manifest.get("warmup", []) +
                     manifest.get("normal", []))
    elif args.clock_mode != "affine-diagnostic":
        raise ValueError("No audited layout in legacy manifest; cannot claim "
                         "same Prima placement. Use a NEW run-id and fixed replay.")
    else:
        print("[PRIMA_FIG3] LEGACY LAYOUT UNVERIFIED - DIAGNOSTIC ONLY",
              flush=True)
    ns = argparse.Namespace(manifest=str(root/"manifest.json"),
                            pc=str(root/"pc_events.csv"), phone=str(phone),
                            out=str(root/"profile"), clock_mode=args.clock_mode)
    try:
        summary = inspect(ns)
    except Exception as error:
        manifest["status"] = "ANALYSIS_FAILED"
        manifest["analysis_error"] = str(error)
        atomic_json(root/"manifest.json", manifest)
        raise
    manifest["status"] = summary["paper_fidelity"]
    atomic_json(root/"manifest.json", manifest)
    print("[PRIMA_FIG3] ANALYSIS:", summary["paper_fidelity"], root/"profile", flush=True)


def main():
    args = args_parser()
    root = args.out.resolve()
    cmd = build_cmd(args)
    print("[PRIMA_FIG3] runner:", " ".join(cmd), flush=True)
    print("[PRIMA_FIG3] Phone run id:", args.run_id, flush=True)
    # In --analyze-only mode no inference is launched. The CLI defaults
    # are irrelevant: the placement provenance belongs to the saved run.
    if args.analyze_only:
        saved = root/"manifest.json"
        if saved.exists():
            saved_manifest = json.loads(saved.read_text(encoding="utf-8"))
            print("[PRIMA_FIG3] ANALYZE_ONLY (no inference); ORIGINAL PLACEMENT:",
                  saved_manifest.get("placement_mode", "UNRECORDED"),
                  "layout=", saved_manifest.get("requested_layout"),
                  "pc_ngl=", saved_manifest.get("requested_pc_gpu_layers"),
                  flush=True)
        else:
            print("[PRIMA_FIG3] ANALYZE_ONLY: manifest missing", flush=True)
    else:
        print("[PRIMA_FIG3] PLACEMENT MODE:",
              ("FIXED_LAYOUT_REPLAY " + args.fixed_layers +
               " GPU=" + str(args.pc_ngl))
              if args.fixed_layers else "AUTO_PLAN_STRICT_COMPARISON",
              flush=True)
    if args.dry_run:
        print("[PRIMA_FIG3] DRY_RUN_PASS (no inference, no file writes)")
        return

    root.mkdir(exist_ok=True, parents=True)
    path = root/"manifest.json"
    if args.analyze_only:
        if not path.exists():
            raise RuntimeError("No manifest.json to analyze without inference")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["run_id"] != args.run_id:
            raise RuntimeError("Run ID mismatch; supply original --run-id")
        finalize(args, root, manifest)
        return

    for p in (args.model, args.prompt, args.binary):
        if not p.is_file():
            raise FileNotFoundError(p)
    if path.exists():
        if not args.resume:
            raise RuntimeError("Manifest exists. Pass --resume or choose a new output folder/run-id")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest["run_id"] != args.run_id or
            manifest["model"] != str(args.model.resolve()) or
            manifest["prompt"] != str(args.prompt.resolve()) or
            manifest.get("full_command") != cmd):
            raise RuntimeError("Resume model/prompt/run-id conflict")
    else:
        manifest = dict(
            run_id=args.run_id, model=str(args.model.resolve()),
            prompt=str(args.prompt.resolve()),
            prompt_sha256=hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
            binary=str(args.binary.resolve()),
            branch="exp/prima-fig3-full-trace-20261118",
            world=2, rank0="WSL CPU+CUDA", rank1="Termux native CPU",
            master=args.master, next=args.next, context=args.context,
            batch=args.batch, ubatch=args.ubatch,
            full_command=cmd,
            placement_mode=("FIXED_LAYOUT_REPLAY" if args.fixed_layers else
                            "AUTO_PLAN_STRICT_COMPARISON"),
            requested_layout=args.fixed_layers or None,
            requested_pc_gpu_layers=args.pc_ngl,
            normal=[], warmup=[], profile=None, clock_samples=[],
            status="RUNNING", created_at=datetime.now(timezone.utc).isoformat(),
            note="Prima native Piped-ring/HiGHS unchanged; Profile and normal measured separately")
        atomic_json(path, manifest)
    env = os.environ.copy()
    for key in ("PRIMA_FIG3_FILE", "PRIMA_FIG3_RANK", "PRIMA_FIG3_RUN_ID"):
        env.pop(key, None)
    # A single post-model-load log line is outside measured Prefill.
    env["PRIMA_FIG3_LAYOUT_AUDIT"] = "1"

    if not manifest["profile"]:
        # Validate the actual Phone clock service before heavy model loads.
        samples = clock_batch(args.clock_host, args.clock_port, "before")
        manifest["clock_samples"].extend(samples)
        atomic_json(path, manifest)
        trace = root/"pc_events.csv"
        if trace.exists():
            raise RuntimeError("PC trace already exists but Profile manifest missing; inspect before overwrite")
        # /mnt/d on WSL is DrvFs. The trace emitter historically performs
        # open/write/close for EVERY event (~1200/Prefill). Directly writing
        # to NTFS can dominate the measured compute time. Stage the *raw*
        # trace on a native Linux filesystem, then copy once after exit.
        with tempfile.TemporaryDirectory(prefix="prima_fig3_", dir=args.trace_tmp_dir) as td:
            local_trace = Path(td)/"pc_events.csv"
            profile_env = dict(env, PRIMA_FIG3_FILE=str(local_trace), PRIMA_FIG3_RANK="0",
                               PRIMA_FIG3_RUN_ID=args.run_id)
            print("[PRIMA_FIG3] TRACE FAST PATH:", str(local_trace),
                  "(one final copy to", str(trace)+")", flush=True)
            stop, thread, during_samples, clock_errors = start_clock_sampling(args)
            try:
                result = launch(args, cmd, root/"profile_run.log", profile_env)
            finally:
                stop.set()
                thread.join(timeout=12)
                manifest["clock_samples"].extend(during_samples)
                manifest["clock_poll_failures"] = clock_errors[:20]
                if local_trace.exists():
                    shutil.copyfile(local_trace, trace)
                    manifest["trace_staging"] = "NATIVE_LINUX_TEMP_COPY_ONCE"
                    manifest["trace_bytes"] = trace.stat().st_size
                atomic_json(path, manifest)
        manifest["profile"] = result
        atomic_json(path, manifest)
        print("[PRIMA_FIG3] during-Profile clock samples:",
              len(during_samples), "errors:", len(clock_errors), flush=True)
        if not trace.exists() or trace.stat().st_size < 100:
            raise RuntimeError("PC instrumentation absent: no profile source events")
        samples = clock_batch(args.clock_host, args.clock_port, "after")
        manifest["clock_samples"].extend(samples)
        atomic_json(path, manifest)
    if args.profile_only:
        print("[PRIMA_FIG3] PROFILE_ONLY complete, Prefill=%.3f ms; "
              "existing trace and clock samples saved; later use --resume "
              "without --profile-only for normal runs."
              % manifest["profile"]["prefill_ms"], flush=True)
        return
    for i in range(len(manifest["warmup"]), args.warmups):
        if args.between:
            time.sleep(args.between)
        value = launch(args, cmd, root/f"warmup_{i+1}.log", env)
        manifest["warmup"].append(value)
        atomic_json(path, manifest)
        compare_runs(manifest["profile"], manifest["warmup"])
    for i in range(len(manifest["normal"]), args.repeats):
        if args.between:
            time.sleep(args.between)
        value = launch(args, cmd, root/f"normal_{i+1}.log", env)
        manifest["normal"].append(value)
        atomic_json(path, manifest)
        compare_runs(manifest["profile"], manifest["warmup"] + manifest["normal"])
    if any(abs(r["tokens"]-manifest["profile"]["tokens"])>2
           for r in manifest["normal"]):
        raise RuntimeError("Normal vs Profile token count mismatch")
    finalize(args, root, manifest)

if __name__ == "__main__":
    main()
