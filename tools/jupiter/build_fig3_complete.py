#!/usr/bin/env python3
"""Strict Jupiter Fig.3 evidence aggregation, WITHOUT triggering inference.

Retains all raw overlapping intervals, their PC/Phone clock alignment, the
Jupiter chunk schedule, and a 100%-covered exclusive PRIORITY PROJECTION.
The projection is not proof of causal critical path, GPU occupancy, physical
idle or isolated RF transmission time.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

ACTIVITIES = (
    "Phone Compute", "PCIe Copy", "GPU Compute",
    "CPU Compute", "Wi-Fi Sync", "Waiting/Other"
)
COLORS = {
    "Phone Compute":"#3978AB","PCIe Copy":"#45A48B",
    "GPU Compute":"#8662B0","CPU Compute":"#BA9948",
    "Wi-Fi Sync":"#DC8858","Waiting/Other":"#A6AFBD",
}
RAW_CATEGORIES = set(ACTIVITIES) | {"Phone Prepare", "Host Prepare"}


def read_csv(path: Path, allow_empty: bool = False):
    if not path.exists():
        raise FileNotFoundError(f"Missing required source: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        data = list(reader)
    if not data and not allow_empty:
        raise ValueError(f"Empty evidence file: {path}")
    return data


def write_csv(path: Path, cols: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def segment(start: int, end: int, category: str, source: str, label: str):
    if end <= start:
        raise ValueError(f"Invalid non-positive interval {source}: {start}..{end}")
    if category not in RAW_CATEGORIES:
        raise ValueError(f"Unexpected activity category {category!r}")
    return dict(start=start,end=end,category=category,source=source,label=label)


def calibration(profile: Path, run_id: str, endpoint: str, max_ms: float):
    rows = [r for r in read_csv(profile/"pc_clock_sync.csv")
            if r["run_id"] == run_id and (not endpoint or r["endpoint"] == endpoint)]
    if len(rows) < 8:
        raise ValueError(f"Need >=8 four-timestamp phone clock samples for run {run_id}; got {len(rows)}")
    row = min(rows, key=lambda r: (int(r["uncertainty_ns"]), int(r["rtt_ns"])))
    bound_ms = int(row["uncertainty_ns"])/1e6
    if bound_ms > max_ms:
        raise ValueError(f"Clock uncertainty {bound_ms:.3f} ms > {max_ms:.3f} ms")
    return int(row["offset_phone_minus_pc_ns"]), bound_ms, row["endpoint"], len(rows)


def projection(records: list[dict], t0: int, t1: int):
    points = [(t0, "", 0),(t1,"",0)]
    for r in records:
        if r["category"] not in ACTIVITIES[:-1]:
            continue
        a,b = max(t0,r["start"]),min(t1,r["end"])
        if b <= a:
            continue
        points.append((a,r["category"],1))
        points.append((b,r["category"],-1))
    points.sort(key=lambda e:e[0])
    active = Counter()
    timeline = []
    overlap = 0
    previous = t0
    i = 0
    while i < len(points):
        at = points[i][0]
        if at > previous:
            occupied = [c for c in ACTIVITIES[:-1] if active[c] > 0]
            category = occupied[0] if occupied else "Waiting/Other"
            if len(occupied) > 1:
                overlap += at - previous
            if timeline and timeline[-1]["category"] == category and timeline[-1]["end"] == previous:
                timeline[-1]["end"] = at
            else:
                timeline.append(dict(start=previous,end=at,category=category))
        while i < len(points) and points[i][0] == at:
            _, cat, delta = points[i]
            if cat:
                active[cat] += delta
                if active[cat] < 0:
                    raise ValueError(f"Bad sweep count for {cat}")
            i += 1
        previous = at
    covered = sum(t["end"]-t["start"] for t in timeline)
    if covered != t1-t0 or timeline[0]["start"] != t0 or timeline[-1]["end"] != t1:
        raise ValueError("Exclusive time axis has uncovered intervals")
    return timeline, overlap


def preview(path: Path, events: list[dict], projected: list[dict],
            t0: int, t1: int, values: dict):
    width,left,right=1180,165,28
    usable = width-left-right
    y = {"Exclusive projection":86,"PC GPU":130,"Phone Compute":172,
         "PCIe Copy":214,"PC CPU":256,"RPC / Sync":298}
    mapping = {"PC GPU":"GPU Compute","Phone Compute":"Phone Compute",
               "PCIe Copy":"PCIe Copy","PC CPU":"CPU Compute",
               "RPC / Sync":"Wi-Fi Sync"}
    def x(t): return left+(t-t0)*usable/(t1-t0)
    svg=[
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 408">',
        '<rect width="1180" height="408" fill="white"/>',
        f'<text x="165" y="29" font-family="Arial" font-size="19" font-weight="bold" fill="#202D3B">Jupiter {html.escape(values["model_label"])} - Fig. 3 timing evidence</text>',
        f'<text x="165" y="49" font-size="12" fill="#69788A">'
        + html.escape("Profile window %.2f s; Phone clock uncertainty %.3f ms" %
                      (values["profile_envelope_ms"]/1000, values["clock_uncertainty_ms"]))+"</text>",
    ]
    for label,pos in y.items():
        svg.append(f'<text x="{left-11}" y="{pos+16}" text-anchor="end" font-size="12" fill="#344353">'
                   + html.escape(label) + "</text>")
        svg.append(f'<rect x="{left}" y="{pos}" width="{usable}" height="24" fill="#EFF2F5"/>')
    for r in projected:
        svg.append(f'<rect x="{x(r["start"]):.3f}" y="86" width="{max(.06,x(r["end"])-x(r["start"])):.3f}" height="24" fill="{COLORS[r["category"]]}"/>')
    for label,name in mapping.items():
        for r in events:
            if r["category"] != name: continue
            svg.append(f'<rect x="{x(r["start"]):.3f}" y="{y[label]}" width="{max(.06,x(r["end"])-x(r["start"])):.3f}" height="24" fill="{COLORS[name]}"/>')
    for tick in range(6):
        xx=left+usable*tick/5
        svg.append(f'<text x="{xx:.2f}" y="341" text-anchor="middle" font-size="11" fill="#637182">{(t1-t0)*tick/5/1e9:.1f}s</text>')
    xx=left
    for name in ACTIVITIES:
        key=name.lower().replace(" ","_").replace("/","_").replace("-","_")
        pct=values[key+"_pct"]
        ww=usable*pct/100
        if ww:
            svg.append(f'<rect x="{xx:.3f}" y="357" width="{ww:.3f}" height="23" fill="{COLORS[name]}"/>')
            if ww>63: svg.append(f'<text x="{xx+ww/2:.3f}" y="373" text-anchor="middle" font-size="11" fill="white">{pct:.1f}%</text>')
        xx+=ww
    svg.append('<text x="165" y="398" font-size="10" fill="#657282">Raw lanes overlap. Wi-Fi Sync includes RPC service/waits, not isolated wireless transfer. Remaining is not physical idle.</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg),encoding="utf-8")


def analyze(args):
    profile=Path(args.profile_dir).resolve()
    out=Path(args.out_dir or profile).resolve()
    out.mkdir(parents=True,exist_ok=True)
    windows=read_csv(profile/"pc_prefill_windows.csv")
    windows=[r for r in windows if int(r["tokens"])>1]
    if args.expected_tokens:
        windows=[r for r in windows if abs(int(r["tokens"])-args.expected_tokens) <= 1]
    if not windows:
        raise ValueError("No native PC Prefill start->synchronized completion window")
    if len(windows)!=1 and not args.allow_multiple_windows:
        raise ValueError(f"Expected exactly one recorded Prefill window; found {len(windows)}")
    chosen=max(windows,key=lambda r:int(r["end_ns"])-int(r["start_ns"]))
    t0,t1=int(chosen["start_ns"]),int(chosen["end_ns"])
    if t1<=t0: raise ValueError("Invalid Prefill boundaries")
    tok=int(chosen["tokens"])
    offset,clock_bound,endpoint,n_cal = calibration(
        profile,args.run_id,args.endpoint,args.max_clock_uncertainty_ms)

    sources=[]
    counts=Counter()
    for file in ("pc_cuda_events.csv","pc_cpu_events.csv","pc_rpc_events.csv"):
        for r in read_csv(profile/file,allow_empty=(file=="pc_cpu_events.csv")):
            cat=r["category"]
            a,b=int(r["start_ns"]),int(r["end_ns"])
            if a>=b or b<=t0 or a>=t1: continue
            sources.append(segment(max(a,t0),min(b,t1),cat,file,r.get("source","")))
            counts[file]+=1

    phone=read_csv(Path(args.phone_events))
    rows=[r for r in phone if r["run_id"]==args.run_id]
    if not rows: raise ValueError("Phone native CSV has no events for this RUN_ID")
    for r in rows:
        a=int(r["start_phone_ns"])-offset
        b=int(r["end_phone_ns"])-offset
        if a>=b or b<=t0 or a>=t1: continue
        sources.append(segment(max(a,t0),min(b,t1),r["category"],
                               "phone_native",f'{r["device"]}:{r["graph_uid"]}'))
        counts["phone_native_"+r["category"]]+=1

    chunk_rows=read_csv(profile/"pc_chunks.csv")
    chunks=[]
    for r in chunk_rows:
        a,b=int(r["start_ns"]),int(r["queued_ns"])
        if b<=t0 or a>=t1: continue
        if b<=a: raise ValueError("Invalid native Jupiter chunk timing")
        chunks.append(r)
        sources.append(segment(max(a,t0),min(b,t1),"Host Prepare",
                               "pc_chunks.csv",f'chunk={r["chunk"]} slot={r["slot"]}'))
    chunks.sort(key=lambda r:int(r["chunk"]))
    if not chunks: raise ValueError("No native chunk/slot scheduling witnesses")
    chunks_total=sum(int(r["tokens"]) for r in chunks)
    if abs(chunks_total-tok)>1:
        raise ValueError(f"Chunk token coverage {chunks_total} != Prefill {tok}")
    if not counts["pc_cuda_events.csv"] or not any(
            r["category"]=="GPU Compute" for r in sources):
        raise ValueError("No PC GPU device interval within actual Prefill")
    if not counts["pc_rpc_events.csv"]:
        raise ValueError("No PC RPC service windows within actual Prefill")
    if not counts["phone_native_Phone Compute"]:
        raise ValueError("No native Phone Compute intervals within actual Prefill")
    # CPU graph and PCIe activity may legitimately be zero for particular
    # execution paths, but their instrumented backends must exist.
    timeline,overlap=projection(sources,t0,t1)
    total=t1-t0
    exclusive=defaultdict(int)
    raw=defaultdict(int)
    for r in timeline: exclusive[r["category"]]+=r["end"]-r["start"]
    for r in sources: raw[r["category"]]+=r["end"]-r["start"]
    if sum(exclusive.values())!=total: raise ValueError("Projected coverage not 100%")

    if args.model_layers is None or args.pc_layers is None or not (1 <= args.pc_layers < args.model_layers):
        raise ValueError("Provide valid --model-layers and --pc-layers for the selected model")
    model_label = {40: "14B", 64: "32B"}.get(args.model_layers, f"{args.model_layers}L")
    values=dict(
        framework="Jupiter",run_id=args.run_id,
        model_layers=args.model_layers,model_label=model_label,
        normal_prefill_ms=round(args.normal_prefill_ms,3) if args.normal_prefill_ms is not None else "",
        profile_envelope_ms=round(total/1e6,3),
        tokens=tok,
        pc_layers=args.pc_layers if args.pc_layers is not None else "",
        phone_layers=args.model_layers-args.pc_layers,
        phone_backend="native RPC backend (record device at Phone; no HTP claim)",
        clock_endpoint=endpoint,clock_uncertainty_ms=round(clock_bound,4),
        clock_samples=n_cal,phone_compute_events=counts["phone_native_Phone Compute"],
        phone_prepare_events=counts["phone_native_Phone Prepare"],
        gpu_device_events=sum(r["category"]=="GPU Compute" for r in sources),
        pcie_device_events=sum(r["category"]=="PCIe Copy" for r in sources),
        pc_cpu_events=counts["pc_cpu_events.csv"],
        pc_rpc_events=counts["pc_rpc_events.csv"],
        chunk_events=len(chunks),
        chunk_tokens=chunks_total,
        raw_event_overlap_ms=round(overlap/1e6,3),
        full_coverage=1,required_sources_present=1,
        accounting_method="exclusive_priority_projection_NOT_causal_critical_path",
        communication_definition="RPC_sync_service_NOT_RF_wire_time",
        gpu_definition="CUDA_stream_intervals_NOT_kernel_utilization",
        waiting_definition="residual_NOT_physical_idle"
    )
    for c in ACTIVITIES:
        key=c.lower().replace(" ","_").replace("/","_").replace("-","_")
        values[key+"_ms"]=round(exclusive[c]/1e6,3)
        values[key+"_pct"]=round(100*exclusive[c]/total,4)
        values[key+"_raw_ms"]=round(raw[c]/1e6,3)
    write_csv(out/"fig3_jupiter_breakdown.csv",list(values),[values])
    rows=[dict(start_ms=round((r["start"]-t0)/1e6,5),
               end_ms=round((r["end"]-t0)/1e6,5),
               duration_ms=round((r["end"]-r["start"])/1e6,5),
               category=r["category"],source=r["source"],label=r["label"])
          for r in sorted(sources,key=lambda x:(x["start"],x["end"]))]
    write_csv(out/"fig3_jupiter_source_events.csv",
              ["start_ms","end_ms","duration_ms","category","source","label"], rows)
    projected=[dict(start_ms=round((r["start"]-t0)/1e6,5),
                    end_ms=round((r["end"]-t0)/1e6,5),
                    duration_ms=round((r["end"]-r["start"])/1e6,5),
                    category=r["category"]) for r in timeline]
    write_csv(out/"fig3_jupiter_timeline.csv",
              ["start_ms","end_ms","duration_ms","category"],projected)
    write_csv(out/"fig3_jupiter_chunks.csv",
              ["chunk","slot","prefix_tokens","tokens","start_ns","queued_ns"],chunks)
    preview(out/"fig3_jupiter_preview.svg",sources,timeline,t0,t1,values)
    (out/"fig3_jupiter_audit.json").write_text(json.dumps({
        "run_id":args.run_id,"pc_prefill_begin_ns":t0,"pc_prefill_end_ns":t1,
        "clock_offset_phone_minus_pc_ns":offset,"results":values,
        "category_priority":ACTIVITIES,"raw_phone_prepare_ms":raw["Phone Prepare"]/1e6,
        "raw_host_prepare_ms":raw["Host Prepare"]/1e6,
        "explanation":"overlap is observed, not added; Wi-Fi Sync is RPC roundtrip, not RF transport"
    },indent=2),encoding="utf-8")
    print(f"[JUPITER_FIG3] PASS run_id={args.run_id} tokens={tok} "
          f"envelope_ms={total/1e6:.3f} clock_bound_ms={clock_bound:.3f} "
          f"phone_graphs={counts['phone_native_Phone Compute']} "
          f"pc_gpu={values['gpu_device_events']} chunks={len(chunks)}")
    for name in ACTIVITIES:
        key=name.lower().replace(" ","_").replace("/","_").replace("-","_")
        print(f"  {name:>18s}: {values[key+'_ms']:>10.3f} ms "
              f"({values[key+'_pct']:>7.3f}%) raw={values[key+'_raw_ms']:.3f} ms")
    print("[JUPITER_FIG3] CAUTION: six-way exclusive PRIORITY projection; not "
          "causal critical path, GPU kernel busy time, or Wi-Fi-only wire time.")
    print(f"[JUPITER_FIG3] output={out}")
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-dir",required=True)
    p.add_argument("--phone-events",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--normal-prefill-ms",type=float)
    p.add_argument("--expected-tokens",type=int)
    p.add_argument("--pc-layers",type=int)
    p.add_argument("--model-layers",type=int,default=64)
    p.add_argument("--endpoint",default="")
    p.add_argument("--out-dir")
    p.add_argument("--max-clock-uncertainty-ms",type=float,default=20.0)
    p.add_argument("--allow-multiple-windows",action="store_true")
    analyze(p.parse_args())


if __name__=="__main__":
    main()
