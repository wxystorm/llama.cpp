#!/usr/bin/env python3
"""Profile-only four-backend whole-layer Ring planner (offline, not executable).

Cycle: PC_CUDA(optional) -> PC_CPU(optional) -> Phone_HTP(optional) -> Phone_OpenCL(optional).
NPU overflow spills a contiguous suffix into Phone OpenCL; do not reject
until both Phone memory budgets and the shared physical budget are checked.
Exact event DAG models per-device compute lanes, boundary copies, PC CUDA
2-bank H2D loads and Phone HTP 2-bank staging/repack with safe fences.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import heapq
import itertools
import json
import math
from pathlib import Path
import sys

MIB = 1024 * 1024

CLASSES = {"GPU", "CPU", "PHONE", "PHONE_GPU"}
REQUIRED_MIXED = (
    "phone_gpu_budget_mib", "phone_gpu_weight_layer_mib",
    "phone_gpu_scratch_mib",
    "phone_shared_budget_mib", "phone_local_mibps",
    "npu_host_stage_mibps", "npu_bank_repack_mibps",
    "npu_host_stage_budget_mib",
)
REQUIRED_BASE = (
    "gpu_budget_mib", "npu_budget_mib", "gpu_weight_layer_mib",
    "npu_weight_layer_mib", "gpu_h2d_mibps", "c2p_mibps",
    "p2c_mibps", "local_mibps", "gpu_scratch_mib",
    "npu_scratch_mib",
)


def calibration(raw, keys, optional_zero=()):
    out = {}
    for k in keys:
        if k not in raw:
            raise ValueError(f"MISSING_MEASURED_CALIBRATION:{k}")
        try:
            x = float(raw[k])
        except (ValueError, TypeError):
            raise ValueError(f"INVALID_CALIBRATION:{k}") from None
        if not math.isfinite(x) or (x < 0 if k in optional_zero or k.endswith("_scratch_mib") or k.endswith("_kv_layer_mib") else x <= 0):
            raise ValueError(f"INVALID_CALIBRATION:{k}")
        out[k] = x
    return out


@dataclass(frozen=True)
class Sample:
    device: str
    layers: int
    tokens: int
    kv: int
    setup_ms: float
    prepare_ms: float
    compute_ms: float
    terminal: bool


class StageProfile:
    def __init__(self, path: Path):
        self.rows: list[Sample] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            expected = ["device", "layers", "tokens", "kv_tokens", "setup_ms",
                        "upload_ms", "prepare_ms", "compute_ms", "terminal", "samples"]
            if reader.fieldnames != expected:
                raise ValueError("INVALID_REAL_STAGE_CSV_SCHEMA")
            for r in reader:
                if r["device"] not in CLASSES:
                    raise ValueError("INVALID_PROFILE_DEVICE:" + r["device"])
                nums = [float(r[k]) for k in ("setup_ms", "upload_ms", "prepare_ms", "compute_ms")]
                if any(not math.isfinite(n) or n < 0 for n in nums):
                    raise ValueError("INVALID_STAGE_COST")
                dep, x, kv, term, samples = (int(r[k]) for k in
                    ("layers", "tokens", "kv_tokens", "terminal", "samples"))
                if min(dep, x, samples) < 1 or kv < x or term not in (0, 1):
                    raise ValueError("INVALID_STAGE_SHAPE")
                self.rows.append(Sample(r["device"], dep, x, kv, nums[0],
                                        nums[1] + nums[2], nums[3], bool(term)))
        if not self.rows:
            raise ValueError("EMPTY_REAL_STAGE_PROFILE")
        self.devices = {r.device for r in self.rows}

    def estimate(self, dev: str, layers: int, tokens: int, kv: int, terminal: bool = False):
        if layers == 0:
            return 0., 0.
        rows = [r for r in self.rows if r.device == dev and r.layers == layers and
                r.terminal == terminal]
        if not rows and terminal and dev in ("PHONE", "PHONE_GPU"):
            rows = [r for r in self.rows if r.device == dev and
                    r.layers == layers and not r.terminal]
        # Model longer OpenCL Regions as sequential measured whole layers.
        # This is a marked P1 linear projection, not a measured Pn service.
        if not rows and dev == "PHONE_GPU" and layers > 1:
            one = self.estimate(dev, 1, tokens, kv, terminal)
            if one is not None:
                return (layers * one[0], layers * one[1])
        # Never extrapolate unmeasured multi-layer HTP services.
        if not rows and dev != "PHONE":
            depths = sorted({r.layers for r in self.rows
                             if r.device == dev and r.terminal == terminal})
            lo = max((d for d in depths if d < layers), default=None)
            hi = min((d for d in depths if d > layers), default=None)
            if lo is not None and hi is not None:
                a = self.estimate(dev, lo, tokens, kv, terminal)
                b = self.estimate(dev, hi, tokens, kv, terminal)
                if a is not None and b is not None:
                    frac = (layers - lo) / (hi - lo)
                    return tuple((1-frac)*p + frac*q for p,q in zip(a,b))
        if not rows:
            return None
        nearest = min(rows, key=lambda r: (abs(r.tokens-tokens), abs(r.kv-kv)))
        if tokens < nearest.tokens * .75 or tokens > nearest.tokens * 1.25:
            return None
        matching = [r for r in rows if r.tokens == nearest.tokens]
        closest = min(matching, key=lambda r: abs(r.kv-kv))
        if kv > max(r.kv for r in matching) * 2:
            return None
        return (closest.compute_ms, closest.prepare_ms)


@dataclass(frozen=True)
class Cycle:
    g: int
    c: int
    n: int
    a: int

    def __post_init__(self):
        if min(self.g, self.c, self.n, self.a) < 0 or self.layers == 0:
            raise ValueError("INVALID_CYCLE")

    @property
    def phone(self):
        return self.n + self.a

    @property
    def layers(self):
        return self.g + self.c + self.phone


@dataclass
class MemoryResult:
    fits: bool
    gpu_bank_mib: tuple[float, float]
    npu_bank_mib: tuple[float, float]
    gpu_pc_used_mib: float
    npu_used_mib: float
    phone_gpu_used_mib: float
    phone_shared_used_mib: float
    npu_host_stage_peak_mib: float


def memory(cycles, b, m):
    g_banks = [0., 0.]
    n_banks = [0., 0.]
    g_rank = n_rank = 0
    ng = nn = na = 0
    for c in cycles:
        if c.g:
            g_banks[g_rank % 2] = max(g_banks[g_rank % 2], c.g * b["gpu_weight_layer_mib"])
            g_rank += 1
            ng += c.g
        if c.n:
            n_banks[n_rank % 2] = max(n_banks[n_rank % 2], c.n * b["npu_weight_layer_mib"])
            n_rank += 1
            nn += c.n
        na += c.a
    # Host-staged GGUF weights are additional temporary Phone RAM, not
    # NPU device Bank allocations. Rank 0/1 are already primed before RUN.
    n_weight_regions = [c.n * b["npu_weight_layer_mib"] for c in cycles if c.n]
    host_stage_peak = max(n_weight_regions[2:], default=0.)
    pg = sum(g_banks) + ng*b["gpu_kv_layer_mib"] + (b["gpu_scratch_mib"] if ng else 0.)
    np = sum(n_banks) + nn*b["npu_kv_layer_mib"] + (b["npu_scratch_mib"] if nn else 0.)
    ap = na*m["phone_gpu_weight_layer_mib"] + na*m["phone_gpu_kv_layer_mib"] + (m["phone_gpu_scratch_mib"] if na else 0.)
    shared = np + ap + host_stage_peak
    fits = (pg <= b["gpu_budget_mib"] + 1e-9 and
            np <= b["npu_budget_mib"] + 1e-9 and
            ap <= m["phone_gpu_budget_mib"] + 1e-9 and
            host_stage_peak <= m["npu_host_stage_budget_mib"] + 1e-9 and
            shared <= m["phone_shared_budget_mib"] + 1e-9)
    return MemoryResult(fits, tuple(g_banks), tuple(n_banks), pg, np, ap, shared, host_stage_peak)


def split_phone(gcp, base, mix, max_npu):
    """Enumerate HTP prefixes and Adreno suffixes before memory pruning.

    No physical compute engine is required in every Cycle. N=0 and A=0
    are legitimate choices.
    """
    if max_npu < 0 or not gcp or any(
            min(g, cpu, phone) < 0 or g + cpu + phone == 0
            for g, cpu, phone in gcp):
        raise ValueError("INVALID_PHONE_SPLIT_GEOMETRY")
    ranges = [range(min(max_npu, p) + 1) for g,c,p in gcp]
    for ranks in itertools.product(*ranges):
        cycles = [Cycle(g, cpu, n, p-n) for (g,cpu,p), n in zip(gcp, ranks)]
        mem = memory(cycles, base, mix)
        if mem.fits:
            yield cycles, mem


def handoff_route(src: str, dst: str):
    """Conservative activation route across the next actual nonzero stage."""
    pc = ("G", "C")
    phone = ("N", "A")
    if src not in pc + phone or dst not in pc + phone:
        raise ValueError("INVALID_HANDOFF_BACKEND")
    if src == dst and src in pc:
        return "SAME_PC_BACKEND", None
    if src in pc and dst in pc:
        return ("PC_CPU_TO_CUDA", "pc_h2d") if src == "C" and dst == "G" else ("PC_LOCAL", "pc_local")
    if src in pc:
        return "C2P", "pc_to_phone"
    if dst in pc:
        return "P2C", "phone_to_pc"
    if src == "N" and dst == "A":
        return "HTP_TO_ADRENO", "phone_gpu"
    # Separate Phone Graphs cannot assume zero-copy, even on the same engine.
    return "PHONE_LOCAL", "phone_local"


def execution_stage_graph(cycles):
    """Ordered physical Regions/edges, without zero-depth phantom Regions.

    A schema=2 metadata contract, NOT an implemented N->A RPC runtime.
    """
    stages = []
    next_layer = 0
    for cycle_index, cycle in enumerate(cycles):
        for backend, depth in (("G", cycle.g), ("C", cycle.c),
                               ("N", cycle.n), ("A", cycle.a)):
            if depth == 0:
                continue
            if backend == "N" and depth != 1:
                raise ValueError("HTP_RUNTIME_P1_ONLY")
            stages.append({
                "region": len(stages),
                "cycle": cycle_index,
                "backend": backend,
                "first_layer": next_layer,
                "last_layer_exclusive": next_layer + depth,
            })
            next_layer += depth
    edges = []
    for source, target in zip(stages, stages[1:]):
        kind, lane = handoff_route(source["backend"], target["backend"])
        edges.append({
            "from_region": source["region"],
            "to_region": target["region"],
            "handoff": kind,
            "resource_lane": lane,
        })
    return stages, edges


@dataclass
class Node:
    name: str
    lane: str
    ms: float
    deps: set[int] = field(default_factory=set)


class Dag:
    def __init__(self):
        self.nodes: list[Node] = []

    def add(self, name, lane, ms, *deps):
        if not math.isfinite(ms) or ms < 0:
            raise ValueError("INVALID_DAG_SERVICE:" + name)
        nid = len(self.nodes)
        ds = set()
        for d in deps:
            if d is not None:
                if d < 0 or d >= nid:
                    raise ValueError("NON_TOPOLOGICAL_DAG:" + name)
                ds.add(d)
        self.nodes.append(Node(name, lane, ms, ds))
        return nid

    def run(self):
        size = len(self.nodes)
        pending = [len(n.deps) for n in self.nodes]
        successors = [[] for _ in self.nodes]
        for i, n in enumerate(self.nodes):
            for d in n.deps:
                successors[d].append(i)
        ready = {lane: [] for lane in {n.lane for n in self.nodes}}
        for i, n in enumerate(self.nodes):
            if not pending[i]:
                heapq.heappush(ready[n.lane], i)
        busy = {lane: False for lane in ready}
        events = []
        now = 0.
        done = 0
        busy_ms = {lane: 0. for lane in ready}
        starts = [0.] * size
        ends = [0.] * size
        while done < size:
            for lane in sorted(ready):
                if busy[lane] or not ready[lane]:
                    continue
                idx = heapq.heappop(ready[lane])
                busy[lane] = True
                starts[idx] = now
                ms = self.nodes[idx].ms
                busy_ms[lane] += ms
                heapq.heappush(events, (now + ms, idx))
            if not events:
                raise ValueError("DAG_DEADLOCK")
            now, idx = heapq.heappop(events)
            busy[self.nodes[idx].lane] = False
            ends[idx] = now
            done += 1
            for nxt in successors[idx]:
                pending[nxt] -= 1
                if pending[nxt] == 0:
                    heapq.heappush(ready[self.nodes[nxt].lane], nxt)
        return now, busy_ms, starts, ends


def dag_for(cycles, x, token_count, hidden, profiles, base, mix, head_ms):
    dag = Dag()
    chunks = [(k, min(x, token_count-k)) for k in range(0, token_count, x)]
    bytes_per_token = hidden * 4
    last_by_stage = {}
    last_gpu_region = []
    last_npu_region = []
    prev_boundary = [None] * len(chunks)
    for ci, cycle in enumerate(cycles):
        stages = [("G", cycle.g, "GPU", "cuda"),
                  ("C", cycle.c, "CPU", "cpu"),
                  ("N", cycle.n, "PHONE", "htp"),
                  ("A", cycle.a, "PHONE_GPU", "phone_gpu")]
        last_stage = next(s[0] for s in reversed(stages) if s[1])
        for letter, depth, backend, lane in stages:
            if not depth:
                continue
            rank = len(last_gpu_region) if letter == "G" else len(last_npu_region) if letter == "N" else -1
            bank_ready = None
            if letter == "G" and rank >= 2:
                weight_mib = depth*base["gpu_weight_layer_mib"]
                bank_ready = dag.add(f"G_BANK_H2D:{ci}", "pc_h2d",
                                     1000*weight_mib/base["gpu_h2d_mibps"],
                                     last_gpu_region[rank-2])
            if letter == "N" and rank >= 2:
                weight_mib = depth*base["npu_weight_layer_mib"]
                staged = dag.add(f"N_BANK_HOST_STAGE:{ci}", "htp_host_io",
                                 1000*weight_mib/mix["npu_host_stage_mibps"],
                                 last_npu_region[rank-2])
                bank_ready = dag.add(f"N_BANK_REPACK:{ci}", "htp",
                                     1000*weight_mib/mix["npu_bank_repack_mibps"],
                                     staged)
            this_last = None
            new_boundary = []
            for k, (begin, length) in enumerate(chunks):
                estimate = profiles.estimate(backend, depth, length, begin+length,
                                             terminal=(letter == last_stage and ci == len(cycles)-1))
                if estimate is None:
                    return None
                compute, prepare = estimate
                deps = [prev_boundary[k], last_by_stage.get((ci, letter)), bank_ready]
                task = dag.add(f"{letter}{ci}:chunk{k}", lane, compute+prepare, *deps)
                last_by_stage[(ci, letter)] = task
                this_last = task
                new_boundary.append(task)
            if letter == "G":
                last_gpu_region.append(this_last)
            if letter == "N":
                last_npu_region.append(this_last)
            following = next(((a,d) for a,d,_,_ in stages[stages.index((letter,depth,backend,lane))+1:]
                              if d > 0), None)
            if following is None and ci + 1 < len(cycles):
                following = ("G", cycles[ci+1].g)
            if following is None:
                prev_boundary = new_boundary
                continue
            next_letter = following[0]
            if letter == "G" and next_letter == "C":
                trans_lane, speed = "pc_local", base["local_mibps"]
            elif letter in ("G", "C") and next_letter in ("N", "A"):
                trans_lane, speed = "pc_to_phone", base["c2p_mibps"]
            elif letter == "N" and next_letter == "A":
                trans_lane, speed = "phone_gpu", mix["phone_local_mibps"]
            elif letter in ("N", "A") and next_letter == "G":
                trans_lane, speed = "phone_to_pc", base["p2c_mibps"]
            else:
                raise ValueError(f"UNSUPPORTED_EDGE:{letter}->{next_letter}")
            prev_boundary = [dag.add(f"{letter}->{next_letter}:{ci}:{k}", trans_lane,
                                     length*bytes_per_token/MIB/speed*1000,
                                     new_boundary[k])
                             for k, (_, length) in enumerate(chunks)]
    if not prev_boundary:
        return None
    terminal = next((s for s in ("A", "N") if getattr(cycles[-1], s.lower())), None)
    if terminal:
        prev_boundary = [dag.add(f"FINAL_P2C:{k}", "phone_to_pc",
                                 length*bytes_per_token/MIB/base["p2c_mibps"]*1000,
                                 prev_boundary[k])
                         for k,(_,length) in enumerate(chunks)]
    dag.add("PC_HEAD", "cpu", head_ms, *prev_boundary)
    wall, busy, start, end = dag.run()
    return {"wall_ms": wall, "nodes": len(dag.nodes),
            "busy_ms": busy, "bank_nodes": [n.name for n in dag.nodes if "BANK_" in n.name],
            "gpu_bank_h2d_ms": busy.get("pc_h2d", 0.),
            "npu_bank_stage_ms": busy.get("htp_host_io", 0.),
            "npu_bank_repack_ms": sum(n.ms for n in dag.nodes if n.name.startswith("N_BANK_REPACK"))}


def geometry(layers, max_cycles=4, max_c=4, max_phone=8):
    for nc in range(2, min(layers//3, max_cycles) + 1):
        widths = [layers//nc + (i < layers%nc) for i in range(nc)]
        for phone in range(2, min(max_phone, min(widths)-1)+1):
            for cpu in range(0, min(max_c, min(widths)-phone-1)+1):
                out = [(w-phone-cpu, cpu, phone) for w in widths]
                if min(g for g,_,_ in out) >= 1:
                    yield out


def layout(cycles):
    return ",".join(f"G{c.g},C{c.c},N{c.n},A{c.a}" for c in cycles)


def planner_backend_mapping(cycles):
    """The Plan owns physical Phone Region/Layer pairs, never Phone env.

    Region indices are assigned after omitting zero-width Regions, matching
    the full-stage runtime's contiguous stage indexing convention. These
    are plan metadata until the runtime supports N->A sub-stages.
    """
    region = 0
    first = 0
    htp = []
    opencl = []
    for c in cycles:
        for name, width in (("G",c.g),("C",c.c),("N",c.n),("A",c.a)):
            if width < 0:
                raise ValueError("NEGATIVE_PHYSICAL_BACKEND_DEPTH")
            if width == 0:
                continue
            if name == "N":
                if width != 1:
                    raise ValueError("HTP_RUNTIME_P1_ONLY")
                htp.append({"region":region,"layer":first})
            elif name == "A":
                opencl.append({"region":region,"first_layer":first,
                               "last_layer_exclusive":first+width})
            first += width
            region += 1
    return htp, opencl


def plan(model, profiles, base, mix, max_npu=1, max_cycles=4):
    if not {"CPU", "GPU", "PHONE", "PHONE_GPU"}.issubset(profiles.devices):
        raise ValueError("MIXED_PROFILE_NEEDS_CPU_GPU_PHONE_HTP_AND_PHONE_GPU_SAMPLES")
    if max_npu < 1:
        raise ValueError("INVALID_NPU_REGION_DEPTH")
    options = []
    diag = {"geometries": 0, "memory_rejected": 0,
            "stage_profile_rejected": 0, "dag_scored": 0,
            "npu_spill_geometries": 0}
    xs = [16,32,64,128]
    for gcp in geometry(model["layers"], max_cycles=max_cycles):
        diag["geometries"] += 1
        top_n = tuple(min(max_npu,p) for g,c,p in gcp)
        mem0 = memory([Cycle(g,c,n,p-n) for (g,c,p),n in zip(gcp,top_n)], base, mix)
        if not mem0.fits:
            diag["npu_spill_geometries"] += 1
        feasible = list(split_phone(gcp,base,mix,max_npu))
        if not feasible:
            diag["memory_rejected"] += 1
            continue
        for cycles, mem in feasible:
            for x in xs:
                result = dag_for(cycles,x,model["tokens"],model["hidden"],
                                 profiles,base,mix,mix.get("pc_head_ms",0.))
                if result is None:
                    diag["stage_profile_rejected"] += 1
                    continue
                diag["dag_scored"] += 1
                options.append((result["wall_ms"], layout(cycles), x, cycles, mem, result))
    if not options:
        raise ValueError("NO_FEASIBLE_MIXED_DAG:check_PHONE_GPU_profile_and_measured_memory_bandwidth")
    options.sort(key=lambda row:(row[0],row[1],row[2]))
    return options, diag


def run_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--profile-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-npu-region-layers", type=int, default=1)
    args = ap.parse_args(argv)
    from model_profile import gguf_info, tokenize
    model_path = args.model.resolve(strict=True)
    prompt = args.prompt.resolve(strict=True)
    info = gguf_info(model_path)
    folder = args.profile_dir.resolve(strict=True)
    manifest = json.loads((folder/"model_profile.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in (1,2) or manifest.get("model") != info:
        raise ValueError("MODEL_PROFILE_FINGERPRINT_MISMATCH")
    extra = json.loads((folder/"phone_mixed_calibration.json").read_text(encoding="utf-8"))
    if extra.get("schema") != 1 or extra.get("model_fingerprint") != info["fingerprint"]:
        raise ValueError("PHONE_MIXED_CALIBRATION_MODEL_MISMATCH")
    b = calibration(manifest["calibration"], REQUIRED_BASE)
    m = calibration(extra["calibration"], REQUIRED_MIXED)
    if "pc_head_ms" in extra["calibration"]:
        m.update(calibration(extra["calibration"], ("pc_head_ms",), ("pc_head_ms",)))
    token_count = tokenize(args.tokenizer.resolve(strict=True), model_path, prompt)
    # KV footprint is determined by model metadata and *this Prompt*, not
    # by the fixed-length Prompt that happened to collect the Profile.
    kv_mib = info["kv_bytes_per_token_per_layer"] * token_count / MIB
    b["gpu_kv_layer_mib"] = kv_mib
    b["npu_kv_layer_mib"] = kv_mib
    m["phone_gpu_kv_layer_mib"] = kv_mib
    # Mixed Stage samples live alongside the original HTP-only Profile;
    # never overwrite the production three-device CSV with PHONE_GPU rows.
    stage_name = extra.get("stage_csv", "real-stage-mixed.csv")
    if Path(stage_name).is_absolute() or ".." in Path(stage_name).parts:
        raise ValueError("INVALID_MIXED_STAGE_PROFILE_PATH")
    stage_file = (folder/stage_name).resolve(strict=True)
    profiles = StageProfile(stage_file)
    options, diag = plan({"layers":info["layers"],"tokens":token_count,
                          "hidden":info["hidden"]}, profiles, b, m,
                         max_npu=args.max_npu_region_layers)
    args.out.mkdir(parents=True,exist_ok=True)
    w, la, x, cycles, mem, dag = options[0]
    htp_map, opencl_map = planner_backend_mapping(cycles)
    record = {
        "schema": 2, "planner": "phone-four-backend-dual-bank-exact-dag-v1",
        "model_id": info["file"], "model_fingerprint":info["fingerprint"],
        "source_profile":str(stage_file), "source_mixed_calibration": str(folder/"phone_mixed_calibration.json"),
        "tokens":token_count, "layers":info["layers"], "chunk_tokens":x,
        "mixed_layout":la,
        "phone_npu_mapping":htp_map,
        "phone_gpu_mapping":opencl_map,
        "mapping_authority":"PC_PLANNER_RPC_NOT_PHONE_ENV",
        "cycles":[{"pc_gpu":c.g,"pc_cpu":c.c,"phone_npu":c.n,"phone_gpu":c.a} for c in cycles],
        "bank_model":{"pc_cuda":"two_bank_rank_k_minus_2_release_and_separate_h2d_lane",
                      "phone_npu":"two_bank_host_stage_plus_htp_repack_after_rank_k_minus_2",
                      "gpu_first_two_primed":True,"npu_first_two_primed":True},
        "memory":{"pc_gpu_bank_peak_mib":mem.gpu_bank_mib,
                  "npu_bank_peak_mib":mem.npu_bank_mib,
                  "pc_gpu_used_mib":mem.gpu_pc_used_mib,
                  "npu_used_mib":mem.npu_used_mib,
                  "phone_gpu_used_mib":mem.phone_gpu_used_mib,
                  "phone_shared_used_mib":mem.phone_shared_used_mib,
                  "npu_host_stage_peak_mib":mem.npu_host_stage_peak_mib},
        "dag":{"predicted_wall_ms":w,**dag},"search":diag,
        "phone_gpu_depth_cost":"measured_depth_or_linear_sum_of_measured_P1_services",
        "prediction_scope":"offline_measured_stage_plus_bank_transfer_model_no_runtime_validation",
        "runtime_compatible":False,
        "execution_blocker":"Runtime cannot execute an intra-cycle HTP->OpenCL contiguous mixed Phone stage; plan only",
    }
    (args.out/"plan.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    with (args.out/"candidates.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["rank","mixed_layout","chunk_tokens","predicted_wall_ms",
                         "pc_gpu_used_mib","npu_used_mib","phone_gpu_used_mib",
                         "phone_shared_used_mib","gpu_bank_h2d_ms",
                         "npu_bank_stage_ms","npu_bank_repack_ms"])
        for j,(ww,ll,xx,_,mm,dd) in enumerate(options[:100],start=1):
            writer.writerow([j,ll,xx,round(ww,3),round(mm.gpu_pc_used_mib,3),
                             round(mm.npu_used_mib,3),round(mm.phone_gpu_used_mib,3),
                             round(mm.phone_shared_used_mib,3),
                             round(dd["gpu_bank_h2d_ms"],3),
                             round(dd["npu_bank_stage_ms"],3),
                             round(dd["npu_bank_repack_ms"],3)])
    print(f"[PHONE_MIXED_DAG] layout={la} X={x} predicted_ms={w:.3f} "
          f"candidates={len(options)} spill_geometries={diag['npu_spill_geometries']} "
          f"status=PLAN_ONLY_RUNTIME_NOT_COMPATIBLE file={args.out/'plan.json'}")


if __name__ == "__main__":
    try:
        run_cli()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"[PHONE_MIXED_DAG] status=FAIL reason={exc}")
