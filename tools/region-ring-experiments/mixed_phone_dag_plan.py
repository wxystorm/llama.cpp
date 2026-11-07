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
    "phone_shared_budget_mib",
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
        # The F32 Host vector ownership moves without a second Host copy.
        # Adreno's measured PHONE_GPU upload_ms is already included in its
        # prepare cost; charging another transfer here double-counts upload.
        return "HTP_TO_ADRENO", None
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
    if x <= 0 or token_count <= 0 or hidden <= 0 or not cycles:
        raise ValueError("INVALID_MIXED_DAG_INPUT")
    # Flatten actual nonzero stages; next Cycle can start with C, N, or A.
    active = [(ci, name, depth, backend, lane)
              for ci, cycle in enumerate(cycles)
              for name, depth, backend, lane in (
                  ("G", cycle.g, "GPU", "cuda"),
                  ("C", cycle.c, "CPU", "cpu"),
                  ("N", cycle.n, "PHONE", "htp"),
                  ("A", cycle.a, "PHONE_GPU", "phone_gpu"))
              if depth]
    if not active:
        raise ValueError("EMPTY_MIXED_DAG")
    dag = Dag()
    chunks = [(k, min(x, token_count-k)) for k in range(0, token_count, x)]
    bytes_per_token = hidden * 4
    last_by_stage = {}
    last_gpu_region = []
    last_npu_region = []
    handoffs = set()
    prev_boundary = [None] * len(chunks)
    speed_by_lane = {
        "pc_h2d": base["gpu_h2d_mibps"],
        "pc_local": base["local_mibps"],
        "pc_to_phone": base["c2p_mibps"],
        "phone_to_pc": base["p2c_mibps"],
        "phone_local": mix.get("phone_local_mibps", 0.0),
    }
    for index, (ci, letter, depth, backend, lane) in enumerate(active):
        rank = (len(last_gpu_region) if letter == "G"
                else len(last_npu_region) if letter == "N" else -1)
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
        last = index + 1 == len(active)
        for k, (begin, length) in enumerate(chunks):
            estimate = profiles.estimate(
                backend, depth, length, begin+length,
                terminal=last and letter in ("N", "A"))
            if estimate is None:
                return None
            compute, prepare = estimate
            task = dag.add(f"{letter}{ci}:chunk{k}", lane, compute+prepare,
                           prev_boundary[k], last_by_stage.get((ci,letter)),
                           bank_ready)
            last_by_stage[(ci,letter)] = task
            this_last = task
            new_boundary.append(task)
        if letter == "G":
            last_gpu_region.append(this_last)
        if not last:
            next_letter = active[index+1][1]
            route, transfer_lane = handoff_route(letter, next_letter)
            handoffs.add(route)
            if transfer_lane is None:
                prev_boundary = new_boundary
            else:
                rate = speed_by_lane[transfer_lane]
                if rate <= 0.0:
                    # Unmeasured A->N Host copy cannot be priced as zero;
                    # the executable G-first N->A path is a vector move.
                    return None
                prev_boundary = [
                    dag.add(f"{letter}->{next_letter}:{ci}:{k}",
                            transfer_lane, length*bytes_per_token/MIB/rate*1000,
                            new_boundary[k])
                    for k, (_, length) in enumerate(chunks)]
        else:
            prev_boundary = new_boundary
        if letter == "N" and not last:
            # HTP keeps its Bank until the last local Host output exists.
            # The vector move to Adreno does not re-copy bytes, and GPU
            # upload is accounted for in PHONE_GPU stage prepare. No P2C
            # data or completion frame exists for this internal boundary.
            last_npu_region.append(prev_boundary[-1])
    final_backend = active[-1][1]
    if final_backend in ("N", "A"):
        prev_boundary = [
            dag.add(f"FINAL_P2C:{k}", "phone_to_pc",
                    length*bytes_per_token/MIB/base["p2c_mibps"]*1000,
                    prev_boundary[k])
            for k, (_, length) in enumerate(chunks)]
        handoffs.add("FINAL_P2C")
        if final_backend == "N":
            last_npu_region.append(prev_boundary[-1])
    elif final_backend == "G":
        # CUDA-to-CPU readback must be measured for the final CPU Head.
        rate = base.get("gpu_d2h_mibps", 0.)
        if rate <= 0:
            return None
        prev_boundary = [
            dag.add(f"FINAL_GPU_TO_CPU:{k}", "pc_d2h",
                    length*bytes_per_token/MIB/rate*1000,
                    prev_boundary[k])
            for k, (_, length) in enumerate(chunks)]
        handoffs.add("FINAL_GPU_TO_CPU")
    dag.add("PC_HEAD", "cpu", head_ms, *prev_boundary)
    wall, busy, start, end = dag.run()
    return {
        "wall_ms": wall, "nodes": len(dag.nodes), "busy_ms": busy,
        "handoffs": sorted(handoffs),
        "bank_nodes": [n.name for n in dag.nodes if "BANK_" in n.name],
        "gpu_bank_h2d_ms": sum(n.ms for n in dag.nodes
                               if n.name.startswith("G_BANK_H2D:")),
        "pc_cpu_cuda_h2d_ms": sum(n.ms for n in dag.nodes
                                  if n.name.startswith("C->G:")),
        "npu_bank_stage_ms": busy.get("htp_host_io", 0.),
        "npu_bank_repack_ms": sum(n.ms for n in dag.nodes
                                 if n.name.startswith("N_BANK_REPACK")),
    }


def geometry(layers, max_cycles=4, max_c=4, max_phone=8):
    """Bounded balanced Cycle seeds, allowing G/C/N/A to be absent."""
    for nc in range(2, min(layers, max_cycles) + 1):
        widths = [layers//nc + (i < layers%nc) for i in range(nc)]
        for phone in range(0, min(max_phone, min(widths))+1):
            for cpu in range(0, min(max_c, min(widths)-phone)+1):
                out = [(w-phone-cpu, cpu, phone) for w in widths]
                if min(g for g,_,_ in out) >= 0:
                    yield out


def layout(cycles):
    return ",".join(f"G{c.g},C{c.c},N{c.n},A{c.a}" for c in cycles)


def planner_backend_mapping(cycles):
    """The Plan owns Phone Region/Layer pairs, not Phone environment."""
    stages, _ = execution_stage_graph(cycles)
    htp = []
    opencl = []
    for stage in stages:
        if stage["backend"] == "N":
            htp.append({"region": stage["region"],
                        "layer": stage["first_layer"]})
        elif stage["backend"] == "A":
            opencl.append({
                "region": stage["region"],
                "first_layer": stage["first_layer"],
                "last_layer_exclusive": stage["last_layer_exclusive"],
            })
    return htp, opencl


def runtime_compatible_cycles(cycles):
    """Conservative executable APPLY_WIDE + Phone-local NPU->Adreno subset."""
    if not 2 <= len(cycles) <= 4:
        return False
    if any(c.g < 1 or c.phone < 1 or c.n > 1 for c in cycles):
        return False
    stages, edges = execution_stage_graph(cycles)
    if not stages or stages[-1]["backend"] not in ("N", "A"):
        return False
    htp = sum(s["backend"] == "N" for s in stages)
    adreno = sum(s["backend"] == "A" for s in stages)
    return (htp >= 2 and adreno >= 1 and
            any(e["handoff"] == "HTP_TO_ADRENO" for e in edges))


def plan(model, profiles, base, mix, max_npu=1, max_cycles=4):
    if not {"CPU", "GPU", "PHONE", "PHONE_GPU"}.issubset(profiles.devices):
        raise ValueError("MIXED_PROFILE_NEEDS_CPU_GPU_PHONE_HTP_AND_PHONE_GPU_SAMPLES")
    if max_npu < 0 or max_npu > 1:
        raise ValueError("HTP_RUNTIME_P1_ONLY")
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
    ap.add_argument("--runtime-compatible", action="store_true",
                    help="select only experimental adjacent HTP-to-Adreno executable plans")
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
    # CUDA-terminal plans need measured readback for the CPU Head.
    if "gpu_d2h_mibps" in manifest["calibration"]:
        b.update(calibration(manifest["calibration"], ("gpu_d2h_mibps",)))
    m = calibration(extra["calibration"], REQUIRED_MIXED)
    # Not required for adjacent HTP->Adreno: the F32 Host vector moves and
    # the OpenCL upload is charged once by the GPU stage Profile. Only
    # non-executable A->N topologies need an actual measured Host-copy rate.
    if "phone_local_mibps" in extra["calibration"]:
        m.update(calibration(extra["calibration"], ("phone_local_mibps",)))
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
    if args.runtime_compatible:
        options = [row for row in options if runtime_compatible_cycles(row[3])]
        if not options:
            raise ValueError("NO_RUNTIME_COMPATIBLE_FOUR_BACKEND_CANDIDATE")
    args.out.mkdir(parents=True,exist_ok=True)
    w, la, x, cycles, mem, dag = options[0]
    htp_map, opencl_map = planner_backend_mapping(cycles)
    execution_stages, execution_edges = execution_stage_graph(cycles)
    record = {
        "schema": 2, "planner": "phone-four-backend-dual-bank-exact-dag-v1",
        "model_id": info["file"], "model_fingerprint":info["fingerprint"],
        "source_profile":str(stage_file), "source_mixed_calibration": str(folder/"phone_mixed_calibration.json"),
        "tokens":token_count, "layers":info["layers"], "chunk_tokens":x,
        "mixed_layout":la,
        "phone_npu_mapping":htp_map,
        "phone_gpu_mapping":opencl_map,
        "execution_stages":execution_stages,
        "execution_edges":execution_edges,
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
        "runtime_compatible":runtime_compatible_cycles(cycles),
        "runtime_experimental":True,
        "runtime_requirements":"RPC_PATCH_31_DUAL_LANE_PLAN_CONTROLLED_TWO_NPU_BANKS_LOCAL_HANDOFF",
        "execution_blocker":None if runtime_compatible_cycles(cycles) else
            "Requires GPU-first Phone-final cycles, at least two HTP Regions and an adjacent HTP->Adreno edge",
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
          f"status={ 'RUNTIME_EXPERIMENTAL' if runtime_compatible_cycles(cycles) else 'PLAN_ONLY' } file={args.out/'plan.json'}")


if __name__ == "__main__":
    try:
        run_cli()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"[PHONE_MIXED_DAG] status=FAIL reason={exc}")
