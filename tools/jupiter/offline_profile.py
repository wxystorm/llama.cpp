#!/usr/bin/env python3
"""Reusable, layout-specific Jupiter offline profile on the Phone-owned FIFO.

Training inputs are deterministic synthetic chat prompts, not benchmark
prompts. Real KV prefixes are built by explicit native plans. Physical PC
and Phone are the two stages. Calibration may use up to 16 wire slots;
the PAPER DP always searches at most 4*2=8 chunks. No per-Prompt q refinement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

from jupiter_planner import QProfile, solve_sequence_table, sequence_points
from min_chunk_profile import select_min_chunk
from native_calibration import PHONE, PHONE_RECORD, BRIDGE, UBATCH, STAGE, profile_rows, validate_run
from resident_runtime import run_resident_cli
from run_phone_owned_multi_prompt import count_model_tokens

POLICY = "jupiter-offline-dp"
CONTEXT = 2048
DEVICES = 2
MAX_K = 4 * DEVICES
WIRE_K = 16
MIN_PROFILE_CHUNK = 8
BASE_X = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256)
BASE_Y = (0, 128, 256, 512, 768, 1024, 1280, 1536, 1792)
THRESHOLD = 0.90
VALIDATION_WARNING = 0.20
WORDS = ("The model reads a short text and predicts the next word. "
         "A local system can process data with several devices. "
         "This simple example tests memory and time for a common task. "
         "Each step uses the same basic rules in a different order.").split()
READY_FILES = ("min-chunk.json", "capacity.json", "stage2-samples.csv",
               "q-profile.csv", "q-samples.csv", "validation.csv",
               "validation-samples.csv", "validation.json", "sequence.plan",
               "plan-summary.csv")
DETAIL_FIELDS = (
    "x", "y", "repeat", "phase", "cpu_compute_ms", "cuda_compute_ms",
    "pc_local_input_ms", "pc_compute_ms", "pc_service_ms",
    "phone_upload_ms", "phone_htp_ms", "phone_copy_ms", "phone_service_ms",
    "bridge_ms", "post_bridge_ms", "admission_ms", "p2c_wait_ms",
    "q_ms", "log",
)
PROFILE_FIELDS = ("x", "y", "device", "samples", "input_ms",
                  "compute_ms", "wall_ms", "latency_ms")
PLAN_FIELDS = ("tokens", "chunks", "k", "b", "bottleneck_ms",
               "predicted_ms", "measured_coordinates", "status")
ADMISSION = re.compile(r"phase=PHONE_AUTO_PUSH run=\d+ chunk=(\d+) tokens=\d+ admission_ms=([0-9.]+)")
P2C = re.compile(r"phase=PHONE_AUTO_P2C_DONE run=\d+ chunk=(\d+) tokens=\d+ phone_compute_ms=[0-9.]+ p2c_wait_ms=([0-9.]+)")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    return digest_bytes(Path(path).read_bytes())


def digest_config(data):
    return digest_bytes(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False).encode("utf-8"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def file_id(path):
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def profile_directory(args, output, layout):
    explicit = getattr(args, "offline_profile_dir", None)
    return Path(explicit).resolve() if explicit else Path(output) / "offline-profile" / f"l{layout:02d}"


def check_options(args):
    if not 256 <= args.max_chunk <= 512 or args.max_chunk % 8:
        raise ValueError("offline DP covers 2048 tokens in at most 8 chunks; "
                         "--max-chunk must be a multiple of 8 in [256,512]")
    if args.stage2_repeats < 1 or args.q_repeats < 1:
        raise ValueError("offline profile repeats must be positive")


def hardware_config(args, layout):
    check_options(args)
    code = Path(__file__).resolve().parent
    pc_bin = Path(args.resident_server).resolve().parent
    libs = sorted(p for p in pc_bin.iterdir() if p.is_file() and
                  p.suffix.lower() in (".dll", ".so", ".dylib") and
                  p.name.lower().startswith(("ggml", "llama")))
    # The Phone binary/firmware and thermal conditions are not queryable via
    # this protocol. Users changing them must choose a new profile tag/path.
    return {
        "format": 1, "policy": POLICY, "backend": "native-autonomous",
        "model": file_id(args.model), "model_size": args.model_size,
        "model_layers": args.model_layers, "cpu_layers": args.cpu_layers,
        "pc_layers": layout, "cuda_layers": layout - args.cpu_layers,
        "phone_layers": args.model_layers - layout, "sessions": args.sessions,
        "rpc": args.rpc, "profile_tag": getattr(args, "profile_tag", ""),
        "server": file_id(args.resident_server), "exe": file_id(args.exe),
        "tokenizer": file_id(args.tokenizer), "pc_libraries": [file_id(p) for p in libs],
        "context": CONTEXT, "batch": CONTEXT, "threads": 8, "window": 2,
        "physical_devices": DEVICES, "max_dp_chunks": MAX_K,
        "max_calibration_chunks": WIRE_K, "max_chunk": args.max_chunk,
        "stage2_repeats": args.stage2_repeats, "q_repeats": args.q_repeats,
        "stage2_x": x_grid(args.max_chunk), "history_grid": list(BASE_Y),
        "threshold": THRESHOLD, "validation_warning": VALIDATION_WARNING,
        "cost": "max(PC_PREFIX_LOCAL_WALL,PHONE_LOCAL_WALL)",
        "source_sha256": {name: digest_file(code / name) for name in (
            "offline_profile.py", "jupiter_planner.py", "min_chunk_profile.py",
            "native_calibration.py", "resident_runtime.py",
            "run_phone_owned_multi_prompt.py")},
        "pc_runtime_env": {k: v for k, v in sorted(os.environ.items()) if
            k.startswith(("GGML_CUDA_", "CUDA_VISIBLE_DEVICES", "GGML_RPC_",
                          "LLAMA_JUPITER_PHONE_OWNED_PERSIST",
                          "LLAMA_JUPITER_RPC_FORCE_GRAPH_BUILD"))},
    }


def open_profile(args, layout, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = hardware_config(args, layout)
    manifest = {"signature": digest_config(config), "config": config}
    path = root / "hardware-config.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f"offline hardware/profile configuration differs: {path}; "
                             "use a new --offline-profile-dir")
        if not (root / "profile-ready.json").exists() and not args.resume:
            raise ValueError(f"incomplete offline profile {root}; use --resume")
    else:
        if any(root.iterdir()):
            raise ValueError(f"offline profile directory has no manifest: {root}")
        write_json(path, manifest)
    return manifest


def feasible(total, minimum, maximum, cap=MAX_K):
    return total >= minimum and math.ceil(total / maximum) <= min(cap, total // minimum)


def balanced_fill(total, minimum, maximum):
    if total == 0:
        return []
    count = (total + maximum - 1) // maximum
    if count < 1 or total // count < minimum:
        raise ValueError(f"cannot fill {total} tokens in [{minimum},{maximum}]")
    base, rem = divmod(total, count)
    return [base + (i < rem) for i in range(count)]


def calibration_plan(total, points, maximum, minimum=MIN_PROFILE_CHUNK, cap=WIRE_K):
    """One explicit real-KV run may measure several nonoverlapping (x,y)."""
    cuts, pos = [], 0
    for x, y in sorted(points, key=lambda p: p[1]):
        if not minimum <= x <= maximum or y < pos or y + x > total:
            raise ValueError(f"invalid/overlapping calibration point ({x},{y})")
        cuts.extend(balanced_fill(y - pos, minimum, maximum))
        cuts.append(x)
        pos = y + x
    cuts.extend(balanced_fill(total - pos, minimum, maximum))
    if not cuts or len(cuts) > cap or sum(cuts) != total:
        raise ValueError("calibration Plan exceeds native wire capacity")
    return cuts


def pack_points(points, total, maximum):
    """Greedily share full native Prefills; preserve every target exactly."""
    group, jobs = [], []
    for point in sorted(set(points), key=lambda p: p[1]):
        try:
            calibration_plan(total, group + [point], maximum)
        except ValueError:
            if not group:
                raise
            jobs.append((calibration_plan(total, group, maximum), group))
            group = [point]
            calibration_plan(total, group, maximum)
        else:
            group.append(point)
    if group:
        jobs.append((calibration_plan(total, group, maximum), group))
    return jobs


def x_grid(maximum):
    return sorted(set([x for x in BASE_X if x <= maximum] + [maximum]))


def training_jobs(b, maximum):
    jobs = []
    for x in sorted(set([b] + [v for v in x_grid(maximum) if v >= b])):
        ys = sorted(set([y for y in BASE_Y if y + x <= CONTEXT] + [CONTEXT - x]))
        jobs.extend(pack_points([(x, y) for y in ys], CONTEXT, maximum))
    return jobs


def holdout_points(b, maximum):
    xs = sorted(set([b] + [v for v in x_grid(maximum) if v >= b]))
    midpoints = [(a + z) // 2 for a, z in zip(xs, xs[1:]) if z - a > 1]
    ys = (64, 384, 640, 896, 1152, 1408, 1664)
    # Independent, off-grid points; none is used to build the training table.
    return [(x, ys[i % len(ys)]) for i, x in enumerate(midpoints)
            if ys[i % len(ys)] + x + MIN_PROFILE_CHUNK <= CONTEXT]


def exact_prompt(args, root, total):
    path = root / "prompts" / f"synthetic-{total}.txt"
    meta = path.with_suffix(".json")
    if meta.is_file():
        info = read_json(meta)
        if info != {"tokens": total, "sha256": digest_file(path)}:
            raise ValueError(f"synthetic prompt fingerprint mismatch: {path}")
        return path.read_text(encoding="utf-8")
    # Each common word normally encodes as one token; count with the real
    # chat template, not an assumed token/word ratio. Try a deterministic
    # correction loop and fail closed if exact length cannot be obtained.
    n = max(1, total - 8)
    tried = set()
    for _ in range(24):
        if n in tried or n <= 0:
            break
        tried.add(n)
        text = " ".join(WORDS[i % len(WORDS)] for i in range(n))
        actual = count_model_tokens(Path(args.tokenizer), Path(args.model), text)
        if actual == total:
            atomic_text(path, text)
            write_json(meta, {"tokens": total, "sha256": digest_file(path)})
            print(f"[JUPITER_OFFLINE_PROMPT] tokens={total} status=EXACT", flush=True)
            return text
        n += total - actual
    # Repeated simple word is a deterministic alternative when BPE merges
    # the more varied corpus in a way that skips the desired length.
    n = max(1, total - 8)
    tried.clear()
    for _ in range(24):
        if n in tried or n <= 0:
            break
        tried.add(n)
        text = " ".join(["system"] * n)
        actual = count_model_tokens(Path(args.tokenizer), Path(args.model), text)
        if actual == total:
            atomic_text(path, text)
            write_json(meta, {"tokens": total, "sha256": digest_file(path)})
            print(f"[JUPITER_OFFLINE_PROMPT] tokens={total} status=EXACT", flush=True)
            return text
        n += total - actual
    raise ValueError(f"could not build an exact {total}-token chat calibration input")


def detailed_rows(text, cuts, cpu_layers):
    """Physical-stage accounting; no communication/Head time double-counting."""
    validate_run(text, sum(cuts), cuts)
    rows = profile_rows(text, metric="wall_ms", require_cpu=cpu_layers > 0,
                        pc_prefix_only=True)
    by_xy = {}
    for r in rows:
        by_xy.setdefault((r["x"], r["y"]), []).append(r)
    phone = {int(m[2]): tuple(map(float, m.groups()[6:]))
             for m in PHONE_RECORD.finditer(text)}
    bridge = {int(m[1]): float(m[3]) for m in BRIDGE.finditer(text)}
    admission = {int(m[1]): float(m[2]) for m in ADMISSION.finditer(text)}
    p2c = {int(m[1]): float(m[2]) for m in P2C.finditer(text)}
    # Existing split-level profile cannot distinguish housekeeping inside
    # the tagged split from prefix math. Retain the local residual there,
    # and separately report all splits AFTER the native bridge (Head path).
    post = {}
    current, seen_bridge = None, False
    for line in text.splitlines():
        m = UBATCH.search(line)
        if m:
            if m[1] == "BEGIN":
                current, seen_bridge = int(m[2]), False
            else:
                current = None
            continue
        m = STAGE.search(line)
        if current is not None and m:
            if int(m[5]):
                seen_bridge = True
            elif seen_bridge:
                post[current] = post.get(current, 0.0) + float(m[4])
    if set(phone) != set(range(len(cuts))) or set(bridge) != set(phone):
        raise ValueError("missing native chunk service diagnostics")
    if set(admission) != set(phone) or set(p2c) != {len(cuts)-1}:
        raise ValueError("missing native PUSH/P2C timing diagnostics")
    result = []
    for chunk, (x, y) in enumerate(sequence_points(cuts)):
        items = by_xy[(x, y)]
        cpu = sum(r["compute_ms"] for r in items if r["device"].startswith("CPU"))
        cuda = sum(r["compute_ms"] for r in items if r["device"].startswith("CUDA"))
        pc = [r for r in items if r["device"] != PHONE]
        pc_wall = sum(r["wall_ms"] for r in pc)
        pc_input = sum(r["input_ms"] for r in pc)
        upload, htp, copy, phone_wall = phone[chunk]
        if not all(math.isfinite(v) and v > 0 for v in (cuda, pc_wall, htp, phone_wall)):
            raise ValueError("missing positive accelerator/physical stage time")
        result.append({
            "x": x, "y": y, "cpu_compute_ms": cpu, "cuda_compute_ms": cuda,
            "pc_local_input_ms": pc_input, "pc_compute_ms": cpu + cuda,
            "pc_service_ms": pc_wall, "phone_upload_ms": upload,
            "phone_htp_ms": htp, "phone_copy_ms": copy, "phone_service_ms": phone_wall,
            "bridge_ms": bridge[chunk], "post_bridge_ms": post.get(chunk, 0.0),
            "admission_ms": admission[chunk], "p2c_wait_ms": p2c.get(chunk, ""),
            "q_ms": max(pc_wall, phone_wall),
        })
    return result


class Measurements:
    def __init__(self, args, layout, root, runtime_env):
        self.args, self.layout, self.root = args, layout, Path(root)
        self.runtime_env = runtime_env
        self.prompts = {}

    def prompt(self, total):
        if total not in self.prompts:
            self.prompts[total] = exact_prompt(self.args, self.root, total)
        return self.prompts[total]

    def run(self, phase, cuts, repeat):
        total = sum(cuts)
        prompt = self.prompt(total)
        job = {"phase": phase, "cuts": list(cuts), "repeat": repeat,
               "prompt_sha256": digest_bytes(prompt.encode("utf-8"))}
        key = digest_config(job)[:24]
        root = self.root / "requests" / phase / key
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "sample.json"
        if checkpoint.is_file():
            record = read_json(checkpoint)
            if record["job"] != job or not re.fullmatch(r"attempt-\d+", record["attempt"]):
                raise ValueError(f"offline sample identity mismatch: {checkpoint}")
            log = root / record["attempt"] / "run.log"
            sidecar = log.with_suffix(".resident.json")
            if digest_file(log) != record["trace_sha256"] or digest_file(sidecar) != record["resident_sha256"]:
                raise ValueError(f"offline sample changed: {log}")
            print(f"[JUPITER_OFFLINE_SAMPLE] phase={phase} chunks={cuts} repeat={repeat} status=REUSE", flush=True)
        else:
            i = 1
            while (root / f"attempt-{i:03d}").exists():
                i += 1
            attempt = root / f"attempt-{i:03d}"
            attempt.mkdir()
            log = attempt / "run.log"
            sidecar = log.with_suffix(".resident.json")
            env = os.environ.copy()
            for k in ("PLAN_FILE", "LOGITS_DUMP_FILE", "STAGE_DUMP_DIR", "STAGE_DUMP_TOKENS"):
                env.pop("LLAMA_JUPITER_" + k, None)
            env.update({
                "LLAMA_JUPITER_PREFILL": "1",
                "LLAMA_JUPITER_PC_LAYERS": str(self.layout),
                "LLAMA_JUPITER_CPU_LAYERS": str(self.args.cpu_layers),
                "LLAMA_JUPITER_HTP_SESSIONS": str(self.args.sessions),
                "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(self.args.sessions),
                "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": str(CONTEXT),
                "LLAMA_JUPITER_PHONE_OWNED_NATIVE": "1",
                "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "1",
                "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "0",
                "LLAMA_JUPITER_WINDOW": "2",
                "LLAMA_JUPITER_GRAPH_PREBUILD": "0",
                "LLAMA_JUPITER_LOCAL_GRAPH_REUSE": "0",
                "LLAMA_JUPITER_ASYNC_C2P": "0",
                "LLAMA_JUPITER_TRACE": "1",
                "LLAMA_JUPITER_PROFILE_STAGE": "1",
                "LLAMA_JUPITER_Q_SWEEP": "1",
                "LLAMA_JUPITER_TIMELINE": "0",
                "LLAMA_JUPITER_SCHED_ALLOC_TIMING": "0",
                "LLAMA_JUPITER_LOGITS_HASH": "0",
                "LLAMA_JUPITER_SEQUENCE_PLAN": ",".join(map(str, cuts)),
            })
            env.update(self.runtime_env)
            command = [str(Path(self.args.exe).resolve()), "-m", str(Path(self.args.model).resolve()),
                       "--rpc", self.args.rpc, "--split-mode", "layer", "-ngl", "all",
                       "-c", str(CONTEXT), "-b", str(CONTEXT), "-ub", str(self.args.max_chunk),
                       "-t", "8", "-n", "1", "--flash-attn", "on", "-lv", "3",
                       "--no-conversation", "--single-turn", "--simple-io", "-p", prompt]
            print(f"[JUPITER_OFFLINE_SAMPLE] phase={phase} chunks={cuts} repeat={repeat} log={log} status=RUN", flush=True)
            rc = run_resident_cli(command, env, log)
            if rc:
                raise RuntimeError(f"offline Native sample failed; see {log}. "
                                   "No invalid sample accepted; resume after device recovery.")
        result = read_json(sidecar)
        text = log.read_text(encoding="utf-8", errors="replace")
        rid = result.get("id", "")
        if (result.get("status") != "PASS" or result.get("tokens") != total or
                result.get("calibration_backend") != "native-autonomous" or
                result.get("logits_validation") != "FINITE_ONLY" or
                result.get("logits_finite") is not True or
                not re.fullmatch(r"[0-9a-f]{32}", rid) or
                f"[JUPITER_RESIDENT_RESET] id={rid} status=PASS" not in text or
                f"[JUPITER_RESIDENT] phase=END id={rid} status=PASS" not in text):
            raise ValueError(f"offline sample lacks verified Native completion/reset: {log}")
        rows = detailed_rows(text, cuts, self.args.cpu_layers)
        if not checkpoint.exists():
            write_json(checkpoint, {"job": job, "attempt": log.parent.name,
                                   "trace_sha256": digest_file(log),
                                   "resident_sha256": digest_file(sidecar)})
        return [dict(row, repeat=repeat, phase=phase, log=str(log)) for row in rows]

    def points(self, phase, cuts, targets, repeats):
        wanted = set(targets)
        result = []
        for repeat in range(1, repeats + 1):
            rows = [r for r in self.run(phase, cuts, repeat) if (r["x"], r["y"]) in wanted]
            if {(r["x"], r["y"]) for r in rows} != wanted:
                raise ValueError("offline sample lacks intended coordinates")
            result.extend(rows)
        return result


def choose_b(samples, maximum, repeats):
    rows = [{"status": "OK", "chunk_tokens": r["x"],
             "pc_compute_ms": r["cuda_compute_ms"], "phone_compute_ms": r["phone_htp_ms"]}
            for r in samples]
    raw, curve, status = select_min_chunk(rows, threshold=THRESHOLD, min_samples=repeats)
    legal = [item for item in curve if all(feasible(n, item.tokens, maximum)
                                          for n in range(item.tokens, CONTEXT+1))]
    if not legal:
        raise ValueError("no uniform measured b is compatible with the paper's k cap")
    if raw not in legal:
        saturated = [v for v in legal if v.minimum_relative >= THRESHOLD]
        chosen = saturated[0] if saturated else max(
            legal, key=lambda v: (v.minimum_relative, -v.tokens))
        status = "FEASIBILITY_CONSTRAINED_" + (
            "BOTH_STAGES_SATURATED" if saturated else "NO_COMMON_PLATEAU")
    else:
        chosen = raw
    return {
        "b": chosen.tokens, "min_chunk": chosen.tokens, "raw_b": raw.tokens,
        "status": status, "method": "accelerator_throughput_proxy_not_occupancy",
        "threshold": THRESHOLD, "max_chunk": maximum, "max_dp_chunks": MAX_K,
        "candidates": [dict(chunk_tokens=v.tokens, samples=v.samples,
                            cuda_compute_ms=v.pc_ms, phone_htp_ms=v.phone_ms,
                            cuda_tokens_per_s=v.pc_tokens_per_s,
                            phone_tokens_per_s=v.phone_tokens_per_s,
                            cuda_relative=v.pc_relative, phone_relative=v.phone_relative,
                            feasible_for_all_lengths=v in legal) for v in curve],
    }


def aggregate_rows(samples):
    grouped = {}
    for row in samples:
        grouped.setdefault((row["x"], row["y"]), []).append(row)
    result = []
    for (x, y), items in sorted(grouped.items()):
        med = lambda key: statistics.median(r[key] for r in items)
        for name, inp, comp, wall in (
                ("PC", med("pc_local_input_ms"), med("pc_compute_ms"), med("pc_service_ms")),
                (PHONE, med("phone_upload_ms") + med("phone_copy_ms"),
                 med("phone_htp_ms"), med("phone_service_ms"))):
            result.append(dict(x=x, y=y, device=name, samples=len(items),
                               input_ms=inp, compute_ms=comp, wall_ms=wall, latency_ms=wall))
    return result


def make_all_plans(profile, b, maximum):
    solved = solve_sequence_table(profile, CONTEXT, DEVICES, b, maximum, MAX_K)
    plans, summary = {}, []
    for total in range(2, CONTEXT+1):
        if total < b:
            # Explicit practical exception for inputs shorter than the
            # efficiency floor, NOT a relaxed DP candidate for long inputs.
            cuts = [total]
            bottleneck = profile.q(total, 0)
            predicted = DEVICES * bottleneck
            status = "SHORT_SINGLE_CHUNK"
        else:
            if total not in solved:
                raise ValueError(f"no paper-constrained partition for N={total}, b={b}")
            sol = solved[total]
            cuts, bottleneck, predicted = sol.chunks, sol.bottleneck_ms, sol.predicted_ms
            status = "OFFLINE_INTERPOLATED_DP"
        if (sum(cuts) != total or not 1 <= len(cuts) <= MAX_K or
                any(not 1 <= c <= maximum for c in cuts) or
                (total >= b and min(cuts) < b)):
            raise ValueError(f"invalid offline plan N={total}")
        plans[total] = cuts
        summary.append(dict(tokens=total, chunks=",".join(map(str, cuts)),
                            k=len(cuts), b=b, bottleneck_ms=bottleneck,
                            predicted_ms=predicted,
                            measured_coordinates=sum(profile.is_measured(x, y)
                                                     for x, y in sequence_points(cuts)),
                            status=status))
    return plans, summary


def load_ready(root, manifest):
    root = Path(root)
    path = root / "profile-ready.json"
    if not path.is_file():
        return None
    ready = read_json(path)
    if (ready.get("signature") != manifest["signature"] or
            ready.get("status") != "READY" or
            set(ready.get("sha256", {})) != set(READY_FILES)):
        raise ValueError(f"invalid offline profile completion: {path}")
    for name, expected in ready["sha256"].items():
        if digest_file(root / name) != expected:
            raise ValueError(f"offline profile artifact changed: {root/name}")
    cfg = manifest["config"]
    plan = {}
    placement = {}
    for raw in (root / "sequence.plan").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, val = line.split("=", 1)
        if key.startswith("sequence."):
            n = int(key.split(".", 1)[1])
            if n in plan:
                raise ValueError("duplicate offline sequence length")
            plan[n] = [int(s) for s in val.split(",")]
        else:
            placement[key] = int(val)
    b = int(read_json(root / "min-chunk.json")["b"])
    if (placement != {"model.pc_layers": cfg["pc_layers"], "model.cpu_layers": cfg["cpu_layers"]} or
            set(plan) != set(range(2, CONTEXT+1)) or
            b != ready["b"] or not 1 <= b <= cfg["max_chunk"]):
        raise ValueError("offline plan placement/coverage mismatch")
    for n, cuts in plan.items():
        if (not 1 <= len(cuts) <= MAX_K or sum(cuts) != n or
                any(c < 1 or c > cfg["max_chunk"] for c in cuts) or
                (n >= b and min(cuts) < b) or (n < b and cuts != [n])):
            raise ValueError(f"invalid offline plan length {n}")
    return {"root": root, "b": b, "plans": plan, "ready": ready,
            "ready_sha256": digest_file(path)}


def build_profile(args, layout, root, manifest, runtime_env):
    root = Path(root)
    m = Measurements(args, layout, root, runtime_env)
    maximum = args.max_chunk
    # Verify maximum-width execution at the actual 2048-token context
    # under the paper cap before spending time on the training grid.
    capacity = balanced_fill(CONTEXT, MIN_PROFILE_CHUNK, maximum)
    if len(capacity) > MAX_K:
        raise ValueError("2048-token capacity incompatible with the paper's k cap")
    print(f"[JUPITER_OFFLINE_CAPACITY] chunks={capacity} status=RUN", flush=True)
    capacity_rows = m.run("capacity", capacity, 1)
    # If maximum does not divide the context, explicitly test max width too.
    if maximum not in capacity:
        capacity_rows.extend(m.run("capacity", calibration_plan(CONTEXT, [(maximum, 0)], maximum), 1))
    write_json(root / "capacity.json", {"status": "PASS", "tokens": CONTEXT,
                                       "chunks": capacity, "max_chunk": maximum,
                                       "logs": sorted({r["log"] for r in capacity_rows})})
    stage2 = []
    total = max(512, 2*maximum)
    for x in x_grid(maximum):
        cuts = calibration_plan(total, [(x, 0)], maximum)
        stage2.extend(m.points("stage2", cuts, [(x, 0)], args.stage2_repeats))
    b_info = choose_b(stage2, maximum, args.stage2_repeats)
    b = b_info["b"]
    write_json(root / "min-chunk.json", b_info)
    write_csv(root / "stage2-samples.csv", DETAIL_FIELDS, stage2)
    print(f"[JUPITER_OFFLINE_B] b={b} raw_b={b_info['raw_b']} status={b_info['status']}", flush=True)

    samples = []
    jobs = training_jobs(b, maximum)
    expected = {point for _, points in jobs for point in points}
    for i, (cuts, points) in enumerate(jobs, 1):
        print(f"[JUPITER_OFFLINE_GRID] {i}/{len(jobs)} points={points}", flush=True)
        samples.extend(m.points("grid", cuts, points, args.q_repeats))
    observed = {}
    for row in samples:
        p = (row["x"], row["y"])
        observed[p] = observed.get(p, 0) + 1
    if set(observed) != expected or any(n != args.q_repeats for n in observed.values()):
        raise ValueError("incomplete reusable q grid")
    write_csv(root / "q-samples.csv", DETAIL_FIELDS, samples)
    table = aggregate_rows(samples)
    write_csv(root / "q-profile.csv", PROFILE_FIELDS, table)
    profile = QProfile(table, metric="wall_ms")

    # Report errors on an independent off-grid set; never silently declare
    # interpolated plans exact or iterate on benchmark Prompt contents.
    validation, held_rows = [], []
    held = holdout_points(b, maximum)
    if not held:
        raise ValueError("no independent interpolation validation coordinates")
    for x, y in held:
        if (x, y) in expected:
            raise ValueError("holdout overlaps training grid")
        cuts = calibration_plan(CONTEXT, [(x, y)], maximum)
        held_rows.extend(m.points("validation", cuts, [(x, y)], args.q_repeats))
    held_profile = QProfile(aggregate_rows(held_rows), metric="wall_ms")
    for x, y in held:
        estimate, actual = profile.q(x, y), held_profile.q(x, y)
        if not held_profile.is_measured(x, y) or actual <= 0:
            raise ValueError("missing independent validation q")
        validation.append(dict(x=x, y=y, predicted_ms=estimate, measured_ms=actual,
                               abs_error_ms=abs(estimate-actual),
                               relative_error=abs(estimate-actual)/actual))
    write_csv(root / "validation.csv",
              ("x", "y", "predicted_ms", "measured_ms", "abs_error_ms", "relative_error"),
              validation)
    write_csv(root / "validation-samples.csv", DETAIL_FIELDS, held_rows)
    errors = [r["relative_error"] for r in validation]
    report = {"points": len(errors), "median_relative_error": statistics.median(errors),
              "max_relative_error": max(errors), "warning_threshold": VALIDATION_WARNING,
              "status": "INTERPOLATION_WARNING" if max(errors) > VALIDATION_WARNING else "WITHIN_DIAGNOSTIC_THRESHOLD",
              "interpretation": "Off-grid check only; not a guarantee of global interpolation accuracy."}
    write_json(root / "validation.json", report)
    print(f"[JUPITER_OFFLINE_VALIDATION] {json.dumps(report)}", flush=True)

    print(f"[JUPITER_OFFLINE_DP] lengths=2..{CONTEXT} devices={DEVICES} k_max={MAX_K} b={b} status=RUN", flush=True)
    plans, summary = make_all_plans(profile, b, maximum)
    atomic_text(root / "sequence.plan",
                "# Jupiter offline interpolated DP; 2 physical stages, k<=8\n" +
                f"model.pc_layers={layout}\nmodel.cpu_layers={args.cpu_layers}\n" +
                "".join(f"sequence.{n}={','.join(map(str,cuts))}\n" for n, cuts in plans.items()))
    write_csv(root / "plan-summary.csv", PLAN_FIELDS, summary)
    write_json(root / "profile-ready.json", {
        "signature": manifest["signature"], "status": "READY", "b": b,
        "physical_devices": DEVICES, "max_dp_chunks": MAX_K,
        "lengths": len(plans), "measured_q_coordinates": len(expected),
        "validation_status": report["status"],
        "sha256": {name: digest_file(root / name) for name in READY_FILES},
    })
    return load_ready(root, manifest)
