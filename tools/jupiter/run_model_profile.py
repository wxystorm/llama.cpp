#!/usr/bin/env python3
"""Profile measured PC/Phone boundaries and emit a Jupiter Model Partition plan."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import statistics
from model_boundary import choose_boundary

STAGE = re.compile(r"\[JUPITER_STAGE_PROFILE\].*?backend=(\S+).*?input_ms=([0-9.]+) compute_ms=([0-9.]+) wall_ms=([0-9.]+)")
BEGIN = re.compile(r"\[JUPITER_UBATCH\] event=BEGIN .*?prefix_tokens=(\d+) tokens=(\d+)")
END = re.compile(r"\[JUPITER_UBATCH\] event=QUEUED")
PLAN = re.compile(r"\[JUPITER_STAGE_PLAN\].*?pc_layers=0\.\.(\d+).*?phone_layers=(\d+)\.\.(\d+)")
MEM = re.compile(r"\b(\S+)\s+(?:model|compute|KV|output) buffer size\s*=\s*([0-9.]+) MiB")
HEADERS = ["pc_layers","phone_layers","repeat","pc_compute_ms","phone_compute_ms","pc_wall_ms","phone_wall_ms","pc_memory_mib","phone_memory_mib","status","reason","log"]

def parse_log(text, boundary, chunk):
    p = PLAN.findall(text)
    if not p or int(p[-1][0])+1 != boundary or int(p[-1][1]) != boundary:
        raise ValueError("runtime model boundary did not match requested boundary")
    phone_layers = int(p[-1][2])-boundary+1
    batches = []
    active = None
    stages = None
    def flush():
        nonlocal active, stages
        if active is not None:
            batches.append((active[0],active[1],stages))
        active,stages = None,None
    for line in text.splitlines():
        m = BEGIN.search(line)
        if m:
            flush()
            active = (int(m[1]),int(m[2]))
            stages = {"PC":[0.,0.,False],"PHONE":[0.,0.,False]}
        elif active is not None:
            m = STAGE.search(line)
            if m:
                backend = m[1]
                owner = "PHONE" if backend.startswith("RPC") else "PC"
                stages[owner][0] += float(m[3])
                stages[owner][1] += float(m[4])
                stages[owner][2] |= owner=="PHONE" or not backend.startswith("CPU")
            elif END.search(line):
                flush()
    flush()
    for idx,(prefix,tokens,_) in enumerate(batches):
        if prefix!=0 or tokens!=chunk:
            continue
        # A fixed-size profile chunk provides repeatable service samples;
        # exclude 2-token warmups and unrelated suffix batches.
        _,_,stage = batches[idx]
        if not stage["PC"][2] or not stage["PHONE"][2]:
            continue
        mem = {"PC":0.,"PHONE":0.}
        found = {"PC":False,"PHONE":False}
        for m in MEM.finditer(text):
            backend = m[1]
            if not backend.startswith(("RPC","CUDA","HIP","Vulkan","Metal")):
                continue
            owner = "PHONE" if backend.startswith("RPC") else "PC"
            mem[owner] += float(m[2])
            found[owner] = True
        return {
            "phone_layers":phone_layers,
            "pc_compute_ms":stage["PC"][0],
            "phone_compute_ms":stage["PHONE"][0],
            "pc_wall_ms":stage["PC"][1],
            "phone_wall_ms":stage["PHONE"][1],
            "pc_memory_mib": round(mem["PC"],3) if found["PC"] else "",
            "phone_memory_mib": round(mem["PHONE"],3) if found["PHONE"] else "",
        }
    raise ValueError(f"no complete profiled {chunk}-token chunk")

def main():
    ap=argparse.ArgumentParser(description="Jupiter two-stage measured model partition")
    for key in ("exe","model","rpc","prompt-file"):
        ap.add_argument("--"+key, required=True)
    ap.add_argument("--boundaries",default="20,22,24,25,26,28")
    ap.add_argument("--chunk",type=int,default=64)
    ap.add_argument("--repeats",type=int,default=2)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--pc-budget-mib",type=float)
    ap.add_argument("--phone-budget-mib",type=float)
    ap.add_argument("--metric",choices=["compute_ms","wall_ms"],default="compute_ms")
    ap.add_argument("--out-dir",default="jupiter-model-profile")
    ap.add_argument("--plan-output",default="jupiter-model.plan")
    args=ap.parse_args()
    boundaries=sorted(set(int(x) for x in args.boundaries.split(",")))
    if not boundaries or min(boundaries)<1 or args.chunk<2 or args.repeats<1:
        ap.error("invalid boundaries, chunk or repeats")
    exe,model,prompt_file=(Path(getattr(args,k)).resolve() for k in ("exe","model","prompt_file"))
    if not all(p.is_file() for p in (exe,model,prompt_file)):
        ap.error("missing exe/model/prompt-file")
    prompt=prompt_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    out=Path(args.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for boundary in boundaries:
        for repeat in range(1,args.repeats+1):
            log=out/f"pc{boundary}-r{repeat}.log"
            env=os.environ.copy()
            env.update(LLAMA_JUPITER_PREFILL="1",LLAMA_JUPITER_PC_LAYERS=str(boundary),
                       LLAMA_JUPITER_WINDOW="1",LLAMA_JUPITER_TRACE="1",
                       LLAMA_JUPITER_PROFILE_STAGE="1",LLAMA_JUPITER_Q_SWEEP="1",
                       LLAMA_JUPITER_ASYNC_C2P="0",LLAMA_JUPITER_LOGITS_TOP10="0",
                       LLAMA_JUPITER_SEQUENCE_PLAN=str(args.chunk))
            env.pop("LLAMA_JUPITER_PLAN_FILE",None)
            cmd=[str(exe),"-m",str(model),"--rpc",args.rpc,"--split-mode","layer",
                 "-ngl","all","-ub",str(args.chunk),"-b",str(args.chunk),
                 "-c",str(max(512,args.chunk)),"-n","1","-t",str(args.threads),
                 "--flash-attn","on","-lv","4",
                 "--no-conversation","--single-turn","--simple-io","-p",prompt]
            row=dict(pc_layers=boundary,repeat=repeat,status="FAILED",reason="",log=str(log))
            print(f"[JUPITER_MODEL_RUN] pc_layers={boundary} repeat={repeat}",flush=True)
            try:
                with log.open("w",encoding="utf-8") as f:
                    result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,
                                          stdout=f,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"llama-cli exit={result.returncode}")
                row.update(parse_log(log.read_text(encoding="utf-8",errors="replace"),boundary,args.chunk))
                row["status"]="OK"
                print(f"[JUPITER_MODEL_SAMPLE] pc={boundary} "
                      f"pc_ms={row['pc_compute_ms']:.3f} phone_ms={row['phone_compute_ms']:.3f}",flush=True)
            except (ValueError,OSError,RuntimeError) as exc:
                row["reason"]=str(exc)
                print(f"[JUPITER_MODEL_REJECT] pc={boundary} reason={exc}",flush=True)
            rows.append(row)
            with (out/"model-boundaries.csv").open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=HEADERS)
                writer.writeheader()
                writer.writerows(rows)
    best,feasible,failures=choose_boundary(rows,args.metric,args.pc_budget_mib,args.phone_budget_mib)
    dest=Path(args.plan_output)
    # Preserve any independently calibrated sequence entries.
    retained=[]
    if dest.is_file():
        retained=[line for line in dest.read_text(encoding="utf-8").splitlines()
                  if not line.startswith("model.pc_layers=")]
    dest.write_text("\n".join([f"model.pc_layers={best.pc_layers}",*retained])+"\n",encoding="utf-8")
    print(f"[JUPITER_MODEL_DP] pc_layers={best.pc_layers} phone_layers={best.phone_layers} "
          f"bottleneck_ms={best.bottleneck_ms:.3f} feasible={len(feasible)} "
          f"failed={failures} plan={dest}",flush=True)

if __name__=="__main__":
    main()
