#!/usr/bin/env python3
"""Model-bound, real-stage calibration bundle and offline-only planning.

The PLAN command never performs profiling, RPC, or model inference.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
from statistics import median

MIB = 1024 * 1024
TYPES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
         7: 1, 10: 8, 11: 8, 12: 8}
FIELDS = re.compile(r"([a-z_]+)=([^\s]+)")
REQUIRED_META = ("general.architecture", "qwen3.block_count",
                 "qwen3.embedding_length", "qwen3.attention.head_count",
                 "qwen3.attention.head_count_kv")


def gguf_info(path):
    with path.open("rb") as f:
        def exact(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF header")
            return b

        def u32():
            return struct.unpack("<I", exact(4))[0]

        def u64():
            return struct.unpack("<Q", exact(8))[0]

        def string():
            n = u64()
            if n > 16 * MIB:
                raise ValueError("invalid GGUF string length")
            return exact(n).decode("utf-8")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                typ, count = u32(), u64()
                if typ in TYPES:
                    f.seek(count * TYPES[typ], 1)
                elif typ == 8:
                    for _ in range(count):
                        f.seek(u64(), 1)
                else:
                    raise ValueError("unsupported GGUF array type")
                return None
            if t not in TYPES:
                raise ValueError(f"unsupported GGUF value type {t}")
            data = exact(TYPES[t])
            if t in (6, 12):
                return struct.unpack("<f" if t == 6 else "<d", data)[0]
            return int.from_bytes(data, "little", signed=t in (1, 3, 5, 11))

        if exact(4) != b"GGUF":
            raise ValueError("not a GGUF model")
        version, tensor_count, kv_count = u32(), u64(), u64()
        if version not in (2, 3) or kv_count > 100000:
            raise ValueError("unsupported GGUF version or metadata count")
        meta = {}
        for _ in range(kv_count):
            key, typ = string(), u32()
            item = value(typ)
            if key in REQUIRED_META or key in (
                "qwen3.attention.key_length", "qwen3.attention.value_length"):
                meta[key] = item
        if any(key not in meta for key in REQUIRED_META):
            raise ValueError("missing Qwen3 Dense metadata; HTP P1 requires Qwen3")
        if meta["general.architecture"] != "qwen3":
            raise ValueError("HTP P1 profile currently supports Qwen3 Dense only")
        heads = meta["qwen3.attention.head_count"]
        hidden = meta["qwen3.embedding_length"]
        key_dim = meta.get("qwen3.attention.key_length", hidden // heads)
        val_dim = meta.get("qwen3.attention.value_length", hidden // heads)
        layers = meta["qwen3.block_count"]
        kv_heads = meta["qwen3.attention.head_count_kv"]
        if not 8 <= layers <= 256 or min(hidden, heads, kv_heads, key_dim, val_dim) <= 0:
            raise ValueError("invalid model dimensions")
        return {"arch": "qwen3", "layers": layers, "hidden": hidden,
                "kv_bytes_per_token_per_layer": 2 * kv_heads * (key_dim + val_dim),
                "file": path.name, "bytes": path.stat().st_size,
                "fingerprint": fingerprint(path)}


def fingerprint(path):
    h = hashlib.sha256()
    n = path.stat().st_size
    h.update(str(n).encode("ascii"))
    with path.open("rb") as f:
        h.update(f.read(MIB))
        if n > MIB:
            f.seek(max(0, n - MIB))
            h.update(f.read(MIB))
    return h.hexdigest()


def probe_layouts(layers):
    cycles = 4 if layers >= 24 else 3
    widths = [layers // cycles + (i < layers % cycles) for i in range(cycles)]
    if min(widths) < 4:
        raise ValueError("model has too few layers for HTP P1 probes")
    layouts = []
    for bias in ("gpu", "balanced", "cpu"):
        items = []
        for width in widths:
            rest = width - 1
            gpu = (max(2, rest - 2) if bias == "gpu" else
                   max(2, rest // 2) if bias == "balanced" else 2)
            gpu = min(gpu, rest - 1)
            items.extend((f"G{gpu}", f"C{rest - gpu}", "P1"))
        layout = ",".join(items)
        if layout not in layouts:
            layouts.append(layout)
    region_ids = [3 * i + 2 for i in range(cycles)]
    phone_layers = []
    off = 0
    for width in widths:
        off += width
        phone_layers.append(off - 1)
    return {"layouts": layouts, "phone_regions": ",".join(map(str, region_ids)),
            "phone_layers": ",".join(map(str, phone_layers)),
            "baseline": layouts[0], "chunks": [16, 32, 64, 128]}


def tags(line):
    return dict(FIELDS.findall(line))


def all_logs(root):
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in root.rglob("engine.stderr.log"))


def throughput(lines, label):
    points = []
    for line in lines.splitlines():
        if f"[HYBRID_PROFILE] {label} " not in line:
            continue
        d = tags(line)
        try:
            size, ms = int(d["bytes"]), float(d["ms"])
            if size >= 65536 and ms > 0 and math.isfinite(ms):
                points.append((size, size / MIB / (ms / 1000)))
        except (KeyError, ValueError):
            continue
    if not points:
        raise ValueError(f"missing measured transfer calibration: {label}")
    points.sort(reverse=True)
    return round(median(v for _, v in points[:4]), 4)


def calibration(stage_csv, pc_text, phone_text):
    with stage_csv.open(newline="", encoding="utf-8-sig") as f:
        points = list(csv.DictReader(f))
    classes = {p["device"] for p in points if float(p["compute_ms"]) > 0}
    if not {"CPU", "GPU", "PHONE"}.issubset(classes) or any(
            p["device"] == "PHONE" and int(p["layers"]) != 1 for p in points):
        raise ValueError("real Stage CSV must include CPU/GPU/HTP P1 measurements")
    if "[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK" not in phone_text:
        raise ValueError("missing real Phone HTP Stage events in Phone log")
    bank, kv = None, None
    for line in phone_text.splitlines():
        d = tags(line)
        if "[NPU_BANK_MEMORY] event=ALLOCATED" in line and d.get("status") == "PASS":
            bank = d
        if "[NPU_BANK_MEMORY] event=KV_BOUND" in line and d.get("status") == "PASS":
            if kv is None or int(d.get("kv_bytes", 0)) > int(kv.get("kv_bytes", 0)):
                kv = d
    if bank is None or kv is None:
        raise ValueError("Phone log has no completed HTP dual-Bank allocation and KV_BOUND")
    mem = None
    weight = None
    for line in pc_text.splitlines():
        if "[HYBRID_PROFILE] pc_memory free=" in line:
            mem = tags(line)
        if "[HYBRID_PROFILE] layer_weight_bytes count=" in line:
            weight = tags(line)
        elif "[HYBRID_PROFILE] attn_weight_bytes_per_layer=" in line and weight is None:
            weight = {"max": tags(line).get("total_weight_bytes_per_layer", "0")}
    if mem is None or weight is None:
        raise ValueError("missing HYBRID_PROFILE memory/weight data; first probe must enable profiling")
    match = re.search(r"gpu_memory free=(\d+)", pc_text)
    if not match or int(match.group(1)) == 0:
        raise ValueError("GPU free memory not measured")
    gpu_free = int(match.group(1))
    bank_a, bank_b = int(bank["a_bytes"]), int(bank["b_bytes"])
    kv_bytes, graph_bytes = int(kv["kv_bytes"]), int(kv["graph_bytes"])
    weight_bytes = int(weight["max"])
    if min(bank_a, bank_b, weight_bytes, gpu_free) <= 0:
        raise ValueError("incomplete measured HTP/GPU weight and memory sizes")
    return {"gpu_budget_mib": round(gpu_free * 0.90 / MIB, 4),
            "npu_budget_mib": round((bank_a + bank_b + kv_bytes + graph_bytes) / MIB, 4),
            "gpu_weight_layer_mib": round(weight_bytes / MIB, 5),
            "npu_weight_layer_mib": round(max(bank_a, bank_b) / MIB, 5),
            "npu_scratch_mib": round(graph_bytes / MIB, 5),
            "gpu_scratch_mib": 0,
            "gpu_h2d_mibps": throughput(pc_text, "gpu_weight_h2d"),
            "c2p_mibps": throughput(pc_text, "pc_to_phone"),
            "p2c_mibps": throughput(pc_text, "phone_to_pc"),
            "local_mibps": throughput(pc_text, "gpu_to_pc")}


def tokenize(exe, model, prompt):
    p = subprocess.run([str(exe), "-m", str(model), "-f", str(prompt),
                        "--no-escape", "--show-count", "--log-disable"],
                       text=True, capture_output=True, check=True)
    m = re.search(r"Total number of tokens:\s*(\d+)", p.stdout)
    if not m or int(m.group(1)) < 1:
        raise ValueError("llama-tokenize did not return a positive token count")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("inspect", "finalize", "plan"):
        p = sub.add_parser(cmd)
        p.add_argument("--model", type=Path, required=True)
        if cmd == "inspect":
            continue
        p.add_argument("--profile-dir", type=Path, required=True)
        if cmd == "finalize":
            p.add_argument("--phone-log", type=Path, required=True)
            p.add_argument("--stage-csv", type=Path, required=True)
        else:
            p.add_argument("--prompt", type=Path, required=True)
            p.add_argument("--tokenizer", type=Path, required=True)
            p.add_argument("--planner", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--banked-confirmed", action="store_true")
    args = ap.parse_args()
    model = args.model.resolve(strict=True)
    info = gguf_info(model)
    if args.cmd == "inspect":
        print(json.dumps({"model": info, "probe": probe_layouts(info["layers"])}, ensure_ascii=False))
        return
    folder = args.profile_dir.resolve()
    manifest = folder / "model_profile.json"
    if args.cmd == "finalize":
        stages = args.stage_csv.resolve(strict=True)
        phone_log = args.phone_log.resolve(strict=True)
        pc_text = all_logs(folder / "runs")
        phone_text = phone_log.read_text(encoding="utf-8", errors="replace")
        if info["file"] not in phone_text:
            raise ValueError("Phone log does not identify the same GGUF filename")
        cal = calibration(stages, pc_text, phone_text)
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / "real-stage-profile.csv"
        if stages != dst:
            shutil.copy2(stages, dst)
        data = {"schema": 1, "profile_type": "QWEN3_DENSE_HTP_DUAL_BANK_REAL_STAGE",
                "model": info, "probe": probe_layouts(info["layers"]),
                "stage_csv": dst.name, "calibration": cal,
                "notes": "HTP P1 only; NPU budget equals observed Bank+KV+Graph allocation. "
                         "GPU scratch not separately measured; network comes from PC Hybrid telemetry."}
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[MODEL_PROFILE] status=SAVED model={info['file']} file={manifest}")
        return
    if not manifest.is_file():
        raise FileNotFoundError(f"No model Profile: {manifest}; run profile separately")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("model") != info:
        raise ValueError("Profile/model mismatch; do not reuse this Profile with another GGUF")
    if data.get("profile_type") != "QWEN3_DENSE_HTP_DUAL_BANK_REAL_STAGE":
        raise ValueError("invalid or incompatible Profile backend")
    csv_path = (folder / data["stage_csv"]).resolve(strict=True)
    tokens = tokenize(args.tokenizer.resolve(strict=True), model,
                      args.prompt.resolve(strict=True))
    cal = data["calibration"]
    options = ["--profile", str(csv_path), "--out", str(args.out.resolve()),
               "--model-id", model.name, "--layers", str(info["layers"]),
               "--hidden", str(info["hidden"]), "--tokens", str(tokens),
               "--baseline", data["probe"]["baseline"]]
    for key in ("gpu_budget_mib", "npu_budget_mib", "gpu_weight_layer_mib",
                "npu_weight_layer_mib", "npu_scratch_mib", "gpu_scratch_mib",
                "gpu_h2d_mibps", "c2p_mibps", "p2c_mibps", "local_mibps"):
        options += ["--" + key.replace("_", "-"), str(cal[key])]
    kv = info["kv_bytes_per_token_per_layer"] * tokens / MIB
    for key in ("gpu", "npu"):
        options += [f"--{key}-kv-layer-mib", str(kv)]
    options += ["--max-npu-region-layers", "1", "--npu-banked-runtime-confirmed",
                "1" if args.banked_confirmed else "0"]
    print(f"[OFFLINE_PROFILE_ONLY] model={model.name} tokens={tokens} profile={manifest}", flush=True)
    subprocess.run([str(args.planner.resolve(strict=True)), *options], check=True)
    out_json = args.out.resolve() / "plan.json"
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    plan["model_profile_fingerprint"] = info["fingerprint"]
    plan["model_profile_file"] = str(manifest)
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[OFFLINE_PROFILE_ONLY] plan={out_json}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        raise SystemExit(f"[MODEL_PROFILE] status=FAIL reason={exc}")
