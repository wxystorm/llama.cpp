#!/usr/bin/env python3

import argparse
import csv
import itertools
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def switches(pattern):
    return sum(a != b for a, b in zip(pattern, pattern[1:]))


def layer_patterns(n_layers, max_switches, include_single_device=False):
    values = []
    for bits in itertools.product(("pc", "phone"), repeat=n_layers):
        if not include_single_device and len(set(bits)) == 1:
            continue
        if switches(bits) > max_switches:
            continue
        values.append(bits)

    # Prefer simpler / canonical layouts first so a partially interrupted run
    # still leaves useful results.
    values.sort(
        key=lambda x: (
            switches(x),
            0 if x[0] == "pc" else 1,
            x.count("phone"),
            x,
        )
    )
    return values


def parse_lt_run(stdout):
    line = None
    for raw in stdout.splitlines():
        if raw.startswith("[LT_RUN]") and "status=PASS" in raw:
            line = raw.strip()
    if line is None:
        return None

    result = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value

    try:
        result["wall_ms"] = float(result["wall_ms"])
        result["tok_per_s"] = float(result["tok_per_s"])
        result["checksum"] = float(result["checksum"])
        result["tokens"] = int(result["tokens"])
        result["chunks"] = int(result["chunks"])
    except (KeyError, ValueError):
        return None

    return result


def candidate_key(c):
    if c["mode"] == "layer":
        return (
            "layer",
            c["layers"],
            c["xt"],
            c["owners"],
        )
    return (
        "tensor",
        c["layers"],
        c["xt"],
        c["first_owner"],
        c["ratio"],
    )


def candidate_label(c):
    if c["mode"] == "layer":
        return f"owners={c['owners']}"
    return (
        f"first={c['first_owner']} "
        f"ratio={c['ratio']:.4f}"
    )


def build_command(args, candidate):
    cmd = [
        args.bench,
        "--connect", args.endpoint,
        "--model", args.model,
        "--prompt-file", args.prompt_file,
        "--mode", candidate["mode"],
        "--layers", str(candidate["layers"]),
        "--chunk-tokens", str(candidate["xt"]),
        "--threads", str(args.threads),
        "--owner-burst", str(args.owner_burst),
        "--max-owner-ahead", str(args.max_owner_ahead),
    ]

    if candidate["mode"] == "layer":
        cmd += [
            "--layer-owners",
            candidate["owners"],
        ]
    else:
        cmd += [
            "--first-owner",
            candidate["first_owner"],
            "--pc-ratio",
            f"{candidate['ratio']:.8f}",
        ]

    return cmd


def run_once(args, candidate, run_index):
    cmd = build_command(args, candidate)

    if args.dry_run:
        print("DRY:", subprocess.list2cmdline(cmd))
        return {
            "candidate": candidate,
            "run_index": run_index,
            "ok": False,
            "dry_run": True,
        }

    last = None
    for attempt in range(args.connect_retries + 1):
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )

        parsed = parse_lt_run(proc.stdout)
        if proc.returncode == 0 and parsed is not None:
            return {
                "candidate": candidate,
                "run_index": run_index,
                "ok": True,
                "returncode": proc.returncode,
                "wall_ms": parsed["wall_ms"],
                "tok_per_s": parsed["tok_per_s"],
                "checksum": parsed["checksum"],
                "tokens": parsed["tokens"],
                "chunks": parsed["chunks"],
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        last = (proc, parsed)
        transient_connect = (
            "connect failed" in proc.stderr.lower()
            or "connect failed" in proc.stdout.lower()
        )
        if transient_connect and attempt < args.connect_retries:
            time.sleep(args.connect_retry_delay)
            continue
        break

    proc, _ = last
    return {
        "candidate": candidate,
        "run_index": run_index,
        "ok": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def aggregate(raw_runs):
    grouped = defaultdict(list)
    for run in raw_runs:
        if run.get("ok"):
            grouped[candidate_key(run["candidate"])].append(run)

    rows = []
    for _, runs in grouped.items():
        c = runs[0]["candidate"]
        walls = [r["wall_ms"] for r in runs]
        tps = [r["tok_per_s"] for r in runs]
        checksums = [r["checksum"] for r in runs]

        row = dict(c)
        row.update(
            runs=len(runs),
            median_ms=statistics.median(walls),
            min_ms=min(walls),
            max_ms=max(walls),
            median_tok_per_s=statistics.median(tps),
            median_checksum=statistics.median(checksums),
            checksum_span=max(checksums) - min(checksums),
        )
        rows.append(row)

    return rows


def choose_top(rows, mode, layers, k):
    selected = [
        r for r in rows
        if r["mode"] == mode and r["layers"] == layers
    ]
    selected.sort(key=lambda r: r["median_ms"])
    return selected[:k]


def checksum_rel(a, b):
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) / scale


def write_raw(path, raw_runs):
    fields = [
        "mode", "layers", "xt", "owners",
        "first_owner", "ratio", "run_index",
        "ok", "returncode", "wall_ms",
        "tok_per_s", "checksum", "tokens", "chunks",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for run in raw_runs:
            c = run["candidate"]
            out.writerow({
                "mode": c["mode"],
                "layers": c["layers"],
                "xt": c["xt"],
                "owners": c.get("owners", ""),
                "first_owner": c.get("first_owner", ""),
                "ratio": c.get("ratio", ""),
                "run_index": run["run_index"],
                "ok": int(bool(run.get("ok"))),
                "returncode": run.get("returncode", ""),
                "wall_ms": run.get("wall_ms", ""),
                "tok_per_s": run.get("tok_per_s", ""),
                "checksum": run.get("checksum", ""),
                "tokens": run.get("tokens", ""),
                "chunks": run.get("chunks", ""),
            })


def write_aggregate(path, rows):
    fields = [
        "mode", "layers", "xt", "owners",
        "first_owner", "ratio", "runs",
        "median_ms", "min_ms", "max_ms",
        "median_tok_per_s", "median_checksum",
        "checksum_span",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                r["layers"],
                r["mode"],
                r["median_ms"],
            ),
        ):
            out.writerow({
                key: row.get(key, "")
                for key in fields
            })


def build_summary(rows, layer_counts, checksum_rel_tol):
    summary = []

    for n in layer_counts:
        layer_rows = choose_top(rows, "layer", n, 1)
        tensor_rows = choose_top(rows, "tensor", n, 1)
        if not layer_rows or not tensor_rows:
            continue

        layer = layer_rows[0]
        tensor = tensor_rows[0]

        rel = checksum_rel(
            layer["median_checksum"],
            tensor["median_checksum"],
        )

        layer_ms = layer["median_ms"]
        tensor_ms = tensor["median_ms"]

        if tensor_ms < layer_ms:
            winner = "TENSOR"
            speedup = layer_ms / tensor_ms
        else:
            winner = "LAYER"
            speedup = tensor_ms / layer_ms

        summary.append({
            "layers": n,
            "winner": winner,
            "speedup": speedup,
            "best_layer_ms": layer_ms,
            "best_layer_xt": layer["xt"],
            "best_layer_layout": layer.get("owners", ""),
            "best_tensor_ms": tensor_ms,
            "best_tensor_xt": tensor["xt"],
            "best_tensor_ratio": tensor.get("ratio", ""),
            "best_tensor_first_owner": tensor.get("first_owner", ""),
            "checksum_rel_diff": rel,
            "checksum_check": (
                "PASS" if rel <= checksum_rel_tol else "WARN"
            ),
        })

    return summary


def write_summary(path, rows):
    fields = [
        "layers", "winner", "speedup",
        "best_layer_ms", "best_layer_xt",
        "best_layer_layout",
        "best_tensor_ms", "best_tensor_xt",
        "best_tensor_ratio", "best_tensor_first_owner",
        "checksum_rel_diff", "checksum_check",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def print_summary(summary):
    print()
    print("=== Layer vs Tensor crossover ===")
    print(
        "layers  winner   speedup   layer_ms   layer_xt   "
        "tensor_ms  tensor_xt  ratio   checksum"
    )
    for row in summary:
        print(
            f"{row['layers']:>6}  "
            f"{row['winner']:<7}  "
            f"{row['speedup']:>7.3f}x  "
            f"{row['best_layer_ms']:>9.3f}  "
            f"{row['best_layer_xt']:>8}  "
            f"{row['best_tensor_ms']:>9.3f}  "
            f"{row['best_tensor_xt']:>9}  "
            f"{str(row['best_tensor_ratio']):>6}  "
            f"{row['checksum_check']}"
        )
        print(
            f"        layer={row['best_layer_layout']}  "
            f"tensor_first={row['best_tensor_first_owner']}  "
            f"checksum_rel={row['checksum_rel_diff']:.6g}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search real-model Layer-vs-Tensor crossover using the "
            "standalone PC/Phone runtime."
        )
    )
    parser.add_argument("--bench", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)

    parser.add_argument(
        "--layers",
        default="2,4,6,8",
        help="layer counts to search",
    )
    parser.add_argument(
        "--chunk-tokens",
        default="64,128",
        help="chunk sizes to search",
    )
    parser.add_argument(
        "--tensor-ratios",
        default="0.20,0.33,0.50,0.67,0.80",
        help="same physical PC FFN ratio used in C2P/P2C",
    )
    parser.add_argument(
        "--tensor-first-owners",
        default="pc",
        help="comma-separated: pc,phone",
    )
    parser.add_argument(
        "--layer-max-switches",
        type=int,
        default=1,
        help=(
            "maximum PC<->Phone ownership boundaries in Layer layouts; "
            "1 searches the canonical single-boundary pipeline"
        ),
    )
    parser.add_argument(
        "--include-single-device",
        action="store_true",
        help="also search all-PC and all-Phone Layer layouts",
    )

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--owner-burst", type=int, default=1)
    parser.add_argument("--max-owner-ahead", type=int, default=1)

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="coarse runs per candidate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="candidates per mode/layer count to stabilize",
    )
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=3,
        help="total runs for top candidates",
    )

    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--connect-retries", type=int, default=20)
    parser.add_argument("--connect-retry-delay", type=float, default=0.25)
    parser.add_argument("--settle-delay", type=float, default=0.10)
    parser.add_argument("--checksum-rel-tol", type=float, default=5e-3)

    parser.add_argument(
        "--output-prefix",
        default="layer-tensor-search",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    layer_counts = parse_csv_ints(args.layers)
    chunk_sizes = parse_csv_ints(args.chunk_tokens)
    ratios = parse_csv_floats(args.tensor_ratios)
    first_owners = parse_csv_strings(args.tensor_first_owners)

    if not layer_counts or not chunk_sizes or not ratios:
        parser.error("search lists must not be empty")
    if args.layer_max_switches < 0:
        parser.error("--layer-max-switches must be >= 0")
    if args.repeats <= 0 or args.final_repeats <= 0:
        parser.error("repeat counts must be > 0")
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")
    if any(x not in ("pc", "phone") for x in first_owners):
        parser.error("--tensor-first-owners accepts only pc,phone")

    candidates = []

    for n in layer_counts:
        patterns = layer_patterns(
            n,
            args.layer_max_switches,
            args.include_single_device,
        )
        for xt in chunk_sizes:
            for pattern in patterns:
                candidates.append({
                    "mode": "layer",
                    "layers": n,
                    "xt": xt,
                    "owners": ",".join(pattern),
                })

            for first_owner in first_owners:
                for ratio in ratios:
                    candidates.append({
                        "mode": "tensor",
                        "layers": n,
                        "xt": xt,
                        "first_owner": first_owner,
                        "ratio": ratio,
                    })

    print(
        f"search candidates={len(candidates)} "
        f"layers={layer_counts} xt={chunk_sizes}"
    )
    print(
        "Phone worker should be running with --serve before this script."
    )

    raw_runs = []
    failures_path = Path(
        args.output_prefix + "-failures.log"
    )

    if failures_path.exists():
        failures_path.unlink()

    total = len(candidates) * args.repeats
    done = 0

    for candidate in candidates:
        for rep in range(args.repeats):
            done += 1
            print(
                f"[{done}/{total}] "
                f"{candidate['mode'].upper()} "
                f"L={candidate['layers']} XT={candidate['xt']} "
                f"{candidate_label(candidate)}",
                flush=True,
            )

            run = run_once(args, candidate, rep)
            raw_runs.append(run)

            if run.get("ok"):
                print(
                    f"  wall={run['wall_ms']:.3f} ms "
                    f"tps={run['tok_per_s']:.3f} "
                    f"checksum={run['checksum']:.9f}",
                    flush=True,
                )
            elif not args.dry_run:
                print(
                    f"  FAIL rc={run.get('returncode')}",
                    flush=True,
                )
                with failures_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        "\n=== "
                        + repr(candidate)
                        + " ===\n"
                    )
                    f.write(
                        run.get("stdout", "")
                    )
                    f.write("\n--- STDERR ---\n")
                    f.write(
                        run.get("stderr", "")
                    )
                    f.write("\n")

            time.sleep(args.settle_delay)

    if args.dry_run:
        return 0

    coarse = aggregate(raw_runs)

    # Stabilize the best few Layer and Tensor candidates for every N.
    rerun_candidates = []
    for n in layer_counts:
        for mode in ("layer", "tensor"):
            for row in choose_top(
                    coarse,
                    mode,
                    n,
                    args.top_k):
                c = {
                    "mode": row["mode"],
                    "layers": row["layers"],
                    "xt": row["xt"],
                }
                if mode == "layer":
                    c["owners"] = row["owners"]
                else:
                    c["first_owner"] = row["first_owner"]
                    c["ratio"] = row["ratio"]
                rerun_candidates.append(c)

    for candidate in rerun_candidates:
        key = candidate_key(candidate)
        existing = sum(
            1
            for run in raw_runs
            if run.get("ok")
            and candidate_key(run["candidate"]) == key
        )
        needed = max(
            0,
            args.final_repeats - existing,
        )

        for extra in range(needed):
            print(
                f"[stabilize] "
                f"{candidate['mode'].upper()} "
                f"L={candidate['layers']} XT={candidate['xt']} "
                f"{candidate_label(candidate)} "
                f"run={existing + extra + 1}/{args.final_repeats}",
                flush=True,
            )
            run = run_once(
                args,
                candidate,
                existing + extra,
            )
            raw_runs.append(run)
            if run.get("ok"):
                print(
                    f"  wall={run['wall_ms']:.3f} ms "
                    f"tps={run['tok_per_s']:.3f}",
                    flush=True,
                )
            time.sleep(args.settle_delay)

    rows = aggregate(raw_runs)
    summary = build_summary(
        rows,
        layer_counts,
        args.checksum_rel_tol,
    )

    raw_path = args.output_prefix + "-runs.csv"
    aggregate_path = args.output_prefix + "-candidates.csv"
    summary_path = args.output_prefix + "-summary.csv"

    write_raw(raw_path, raw_runs)
    write_aggregate(aggregate_path, rows)
    write_summary(summary_path, summary)

    print_summary(summary)
    print()
    print(f"raw runs:   {raw_path}")
    print(f"candidates: {aggregate_path}")
    print(f"summary:    {summary_path}")
    if failures_path.exists():
        print(f"failures:   {failures_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
