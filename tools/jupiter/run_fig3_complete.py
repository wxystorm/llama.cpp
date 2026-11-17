#!/usr/bin/env python3
"""One-time complete Jupiter Fig.3 collection, with resumable PHONE-ONLY finish.

Uses the INFOCOM Jupiter layer pipeline unmodified: PC prefix + Phone suffix,
default asynchronous chunk/window policy, original RPC. One instrumented run,
plus an independent clean Prefill baseline (optional warmup/repeats). No q(x,y)
sweep or stage-forced synchronization. Absent Phone SSH never reruns inference.
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
import sys
from datetime import datetime, timezone

try:
    from build_fig3_complete import analyze
except ImportError:
    from tools.jupiter.build_fig3_complete import analyze

TIMING = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",re.IGNORECASE)
BRANCH = "exp/jupiter-fig3-full-trace-20261118"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe",default=r".\build-jupiter\bin\Release\llama-cli.exe")
    p.add_argument("--model",default=r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf")
    p.add_argument("--prompt-file",
                   default=r"E:\llama\datasets\sharegpt\sharegpt50_short\prompt_1024_01.txt")
    p.add_argument("--rpc",default="10.238.229.239:10139")
    p.add_argument("--pc-layers",type=int,default=24)
    p.add_argument("--context",type=int,default=2048)
    p.add_argument("--batch",type=int,default=2048)
    p.add_argument("--ubatch",type=int,default=128)
    p.add_argument("--window",type=int,default=2)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--plan-file",default="")
    p.add_argument("--sequence-plan",default="")
    p.add_argument("--async-c2p",action="store_true",
                   help="preserve an existing async-C2P baseline; default OFF")
    p.add_argument("--local-graph-reuse",action="store_true")
    p.add_argument("--normal-repeats",type=int,default=3)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--out-dir",default="fig3-jupiter14b-complete")
    p.add_argument("--resume",action="store_true",
                   help="reuse all completed Profile and normal runs after interruption")
    p.add_argument("--analyze-only",action="store_true",
                   help="never invoke inference; use archived PC data and Phone CSV")
    p.add_argument("--dry-run",action="store_true",
                   help="test configuration & print command; no model inference")
    p.add_argument("--phone-events",default="",
                   help="local copied Phone CSV to process (used by --analyze-only too)")
    p.add_argument("--phone-user",default="u0_a283")
    p.add_argument("--phone-host",default="")
    p.add_argument("--phone-ssh-port",type=int,default=8022)
    p.add_argument("--phone-remote-file",
                   default="/data/data/com.termux/files/home/jupiter-fig3-phone.csv")
    p.add_argument("--skip-scp",action="store_true")
    p.add_argument("--clock-limit-ms",type=float,default=20.0)
    return p


def identity(path: Path):
    return {"path":str(path.resolve()),"size":path.stat().st_size,
            "mtime_ns":path.stat().st_mtime_ns}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def check_paths(args, run_modes: bool):
    if args.pc_layers<=0 or args.pc_layers>=40:
        raise ValueError("14B requires 1..39 PC layers; remaining 40-PC are Phone layers")
    if not (1<=args.ubatch<=args.batch<=args.context):
        raise ValueError("require 1 <= ubatch <= batch <= context")
    if args.window<1 or args.threads<1 or args.warmups<0 or args.normal_repeats<1:
        raise ValueError("invalid window/threads/warmups/normal-repeats")
    if args.sequence_plan and args.plan_file:
        raise ValueError("specify one of --plan-file or --sequence-plan, not both")
    if run_modes:
        for name in ("exe","model","prompt_file"):
            p=Path(getattr(args,name))
            if not p.is_file(): raise FileNotFoundError(f"{name}: {p}")
    host,sep,port=args.rpc.rpartition(":")
    if not sep or not host or not port.isdecimal():
        raise ValueError("--rpc must be host:port")
    return host,int(port)


def cli(args, prompt: str):
    return [str(Path(args.exe).resolve()),"-m",str(Path(args.model).resolve()),
            "--rpc",args.rpc,"--split-mode","layer","-ngl","all",
            "-ub",str(args.ubatch),"-b",str(args.batch),
            "-c",str(args.context),"-n","1","-t",str(args.threads),
            "--flash-attn","on","-lv","3","--no-conversation",
            "--single-turn","--simple-io","-p",prompt]


def env_for(args, profile: bool, run_id: str, profile_dir: Path):
    env=os.environ.copy()
    for key in ("LLAMA_JUPITER_SEQUENCE_PLAN","LLAMA_JUPITER_PLAN_FILE",
                "LLAMA_JUPITER_FIG3_DIR","LLAMA_JUPITER_FIG3_RUN_ID"):
        env.pop(key,None)
    env.update({
        "LLAMA_JUPITER_PREFILL":"1",
        "LLAMA_JUPITER_PC_LAYERS":str(args.pc_layers),
        "LLAMA_JUPITER_WINDOW":str(args.window),
        "LLAMA_JUPITER_PROFILE_STAGE":"0",
        "LLAMA_JUPITER_TRACE":"0",
        "LLAMA_JUPITER_Q_SWEEP":"0",
        "LLAMA_JUPITER_LOGITS_TOP10":"0",
        "LLAMA_JUPITER_ASYNC_C2P":"1" if args.async_c2p else "0",
        "LLAMA_JUPITER_LOCAL_GRAPH_REUSE":"1" if args.local_graph_reuse else "0",
        "LLAMA_GRAPH_REUSE_DISABLE":"0",
        "LLAMA_JUPITER_TIMELINE":"1" if profile else "0",
    })
    if args.plan_file:
        env["LLAMA_JUPITER_PLAN_FILE"]=str(Path(args.plan_file).resolve())
    if args.sequence_plan:
        env["LLAMA_JUPITER_SEQUENCE_PLAN"]=args.sequence_plan
    if profile:
        env["LLAMA_JUPITER_FIG3_DIR"]=str(profile_dir.resolve())
        env["LLAMA_JUPITER_FIG3_RUN_ID"]=run_id
    return env


def parse_timing(log: Path):
    text=log.read_text(encoding="utf-8",errors="replace")
    hits=TIMING.findall(text)
    if not hits: raise RuntimeError(f"No genuine Prefill timing in {log}")
    ms,tokens=hits[-1]
    if float(ms)<=0 or int(tokens)<2: raise RuntimeError(f"Invalid Prefill timing {ms} {tokens}")
    return {"prefill_ms":float(ms),"tokens":int(tokens)}


def launch(cmd:list[str],env:dict,log:Path):
    log.parent.mkdir(parents=True,exist_ok=True)
    print(f"[JUPITER_FIG3] START {log.name}",flush=True)
    with log.open("wb") as stream:
        result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=stream,
                              stderr=subprocess.STDOUT,env=env,check=False)
    if result.returncode:
        last="\n".join(log.read_text(encoding="utf-8",errors="replace").splitlines()[-15:])
        raise RuntimeError(f"llama-cli failed rc={result.returncode} log={log}\n{last}")
    info=parse_timing(log)
    print(f"[JUPITER_FIG3] PASS {log.name}: {info['prefill_ms']:.3f} ms, "
          f"{info['tokens']} tokens",flush=True)
    return info


def retrieve_phone(args,root):
    dest=root/"phone_events.csv"
    if args.phone_events:
        src=Path(args.phone_events).resolve()
        if not src.is_file(): raise FileNotFoundError(f"Phone CSV: {src}")
        if src != dest.resolve(): shutil.copyfile(src,dest)
        return dest
    if dest.is_file():
        return dest
    if args.skip_scp:
        return None
    host=args.phone_host or args.rpc.rpartition(":")[0]
    remote=f"{args.phone_user}@{host}:{args.phone_remote_file}"
    print(f"[JUPITER_FIG3] SCP Phone telemetry {remote}",flush=True)
    try:
        result=subprocess.run(["scp","-P",str(args.phone_ssh_port),
                               remote,str(dest)],capture_output=True,text=True,
                              timeout=30,check=False)
    except (FileNotFoundError,subprocess.TimeoutExpired) as ex:
        print(f"[JUPITER_FIG3] SCP not available: {ex}",flush=True)
        return None
    if result.returncode:
        print(f"[JUPITER_FIG3] SCP failed: {result.stderr.strip()}",flush=True)
        return None
    return dest if dest.is_file() else None


def write_paper_row(root: Path,breakdown:dict):
    row={
        "framework":"Jupiter",
        "normal_prefill_ms":breakdown["normal_prefill_ms"],
        "profile_envelope_ms":breakdown["profile_envelope_ms"],
        "tokens":breakdown["tokens"],
        "pc_layers":breakdown["pc_layers"],
        "phone_layers":breakdown["phone_layers"],
        "phone_compute_pct":breakdown["phone_compute_pct"],
        "gpu_compute_pct":breakdown["gpu_compute_pct"],
        "cpu_compute_pct":breakdown["cpu_compute_pct"],
        "pcie_copy_pct":breakdown["pcie_copy_pct"],
        "rpc_collective_sync_pct":breakdown["wi_fi_sync_pct"],
        "waiting_other_pct":breakdown["waiting_other_pct"],
        "clock_uncertainty_ms":breakdown["clock_uncertainty_ms"],
        "overlap_ms":breakdown["raw_event_overlap_ms"],
        "semantics":"exclusive_priority_projection_not_causal_critical_path"
    }
    with (root/"fig3_jupiter_paper_row.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        writer.writeheader();writer.writerow(row)


def finalize(args,root,manifest):
    profile_dir=root/"profile"
    required=["pc_prefill_windows.csv","pc_chunks.csv","pc_cuda_events.csv",
              "pc_cpu_events.csv","pc_rpc_events.csv","pc_clock_sync.csv"]
    missing=[f for f in required if not (profile_dir/f).is_file()]
    if missing:
        raise RuntimeError(f"PC source missing {missing}; logs retained under {root}. "
                           "This is a backend instrumentation/build problem, not a Phone SCP problem.")
    phone=retrieve_phone(args,root)
    if phone is None:
        print("[JUPITER_FIG3] NEED_PHONE_ONLY. PC and native GPU/RPC traces have been SAVED.",flush=True)
        print(f"[JUPITER_FIG3] Phone file: {args.phone_remote_file}",flush=True)
        print(f"[JUPITER_FIG3] Copy phone CSV to {root/'phone_events.csv'} and then run:",flush=True)
        print(f'  python tools/jupiter/run_fig3_complete.py --out-dir "{root}" --analyze-only',flush=True)
        manifest["analysis_status"]="WAITING_FOR_PHONE_CSV"
        write_json(root/"manifest.json",manifest)
        return
    if not manifest.get("normal"):
        print("[JUPITER_FIG3] No unprofiled reference timing in manifest",flush=True)
        return
    normal_ms=statistics.median(r["prefill_ms"] for r in manifest["normal"])
    formal=argparse.Namespace(
        profile_dir=str(profile_dir),phone_events=str(phone),
        run_id=manifest["run_id"],normal_prefill_ms=normal_ms,
        pc_layers=args.pc_layers,expected_tokens=manifest["profile"]["tokens"],
        endpoint=args.rpc,max_clock_uncertainty_ms=args.clock_limit_ms,
        out_dir=str(profile_dir),allow_multiple_windows=False,
    )
    val=analyze(formal)
    write_paper_row(root,val)
    manifest["analysis_status"]="PASS"
    manifest["normal_prefill_median_ms"]=normal_ms
    manifest["breakdown_path"]=str(profile_dir/"fig3_jupiter_breakdown.csv")
    write_json(root/"manifest.json",manifest)
    print(f"[JUPITER_FIG3] DONE. Plot ready: {root/'fig3_jupiter_paper_row.csv'}",flush=True)


def main():
    args=parser().parse_args()
    host,port=check_paths(args,not args.analyze_only)
    root=Path(args.out_dir).resolve()
    if args.dry_run:
        print("[JUPITER_FIG3] DRY_RUN_PASS: baseline=Jupiter original Layer Pipeline")
        print(f"[JUPITER_FIG3] PC={args.pc_layers} Phone={40-args.pc_layers}, "
              f"ubatch={args.ubatch} window={args.window} RPC={args.rpc}")
        print("[JUPITER_FIG3] command=",cli(args,"<contents-of-prompt-file>"))
        return
    root.mkdir(parents=True,exist_ok=True)
    manifest_file=root/"manifest.json"
    if args.analyze_only:
        if not manifest_file.is_file():
            raise RuntimeError("--analyze-only requires manifest.json from an earlier COMPLETE collection")
        manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("rpc")!=args.rpc:
            # Restore source experiment args instead of silently mixing settings.
            args.rpc=manifest["rpc"]
        args.pc_layers=manifest["pc_layers"]
        finalize(args,root,manifest)
        return
    if manifest_file.exists():
        if not args.resume:
            raise RuntimeError(f"{manifest_file} already exists. Use --resume "
                               "to avoid repeating already completed inference.")
        manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["rpc"]!=args.rpc or manifest["pc_layers"]!=args.pc_layers:
            raise ValueError("Cannot resume with different model partition or Phone endpoint")
    else:
        prompt=Path(args.prompt_file).read_text(encoding="utf-8-sig").rstrip("\r\n")
        manifest={
            "branch":BRANCH,
            "run_id":"jupiter-fig3-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "model":identity(Path(args.model)), "exe":identity(Path(args.exe)),
            "prompt_file":identity(Path(args.prompt_file)),
            "prompt_sha256":hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rpc":args.rpc,"pc_layers":args.pc_layers,
            "phone_layers":40-args.pc_layers,
            "context":args.context,"batch":args.batch,"ubatch":args.ubatch,
            "window":args.window,"threads":args.threads,
            "async_c2p":args.async_c2p,"local_graph_reuse":args.local_graph_reuse,
            "plan_file":args.plan_file,"sequence_plan":args.sequence_plan,
            "normal_repeats":args.normal_repeats,"warmups":args.warmups,
            "profile":None,"normal":[],"warmup":[],"analysis_status":"RUNNING",
            "comparison_note":"Jupiter remains Layer Pipeline; normal Prefill and Profile measured separately"
        }
        write_json(manifest_file,manifest)
    prompt=Path(args.prompt_file).read_text(encoding="utf-8-sig").rstrip("\r\n")
    command=cli(args,prompt)
    profile=root/"profile"
    profile.mkdir(exist_ok=True)
    # Reachability preflight WITHOUT launching model.
    print(f"[JUPITER_FIG3] RPC connect preflight {host}:{port}...",flush=True)
    try:
        with socket.create_connection((host,port),timeout=5): pass
    except OSError as ex:
        raise RuntimeError(f"Phone RPC unreachable BEFORE model run: {ex}") from ex
    print("[JUPITER_FIG3] RPC TCP reachable",flush=True)
    if not manifest.get("profile"):
        # Run the profiled trial first: an old/uninstrumented Phone server
        # is rejected by the tag/clock handshake before doing a full Prefill.
        res=launch(command,env_for(args,True,manifest["run_id"],profile),
                   root/"profile_run.log")
        manifest["profile"]=res
        write_json(manifest_file,manifest)
        required=["pc_prefill_windows.csv","pc_chunks.csv","pc_cuda_events.csv",
                  "pc_cpu_events.csv","pc_rpc_events.csv","pc_clock_sync.csv"]
        missing=[file for file in required if not (profile/file).is_file()]
        if missing:
            raise RuntimeError("PC profile missing required instrumentation: "+", ".join(missing)+
                               ". Logs preserved, do not proceed to formal runs.")
    for i in range(len(manifest["warmup"]),args.warmups):
        r=launch(command,env_for(args,False,manifest["run_id"],profile),
                 root/f"warmup_{i+1}.log")
        manifest["warmup"].append(r);write_json(manifest_file,manifest)
    for i in range(len(manifest["normal"]),args.normal_repeats):
        r=launch(command,env_for(args,False,manifest["run_id"],profile),
                 root/f"normal_{i+1}.log")
        manifest["normal"].append(r);write_json(manifest_file,manifest)
    if any(abs(r["tokens"]-manifest["profile"]["tokens"])>1
           for r in manifest["normal"]):
        raise RuntimeError("Normal and Profile token counts mismatch; raw runs preserved")
    finalize(args,root,manifest)


if __name__=="__main__":
    main()
