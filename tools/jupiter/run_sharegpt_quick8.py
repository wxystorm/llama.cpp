#!/usr/bin/env python3
"""Select the first two ShareGPT40 prompts per group, then run Jupiter DP.

Selection is from the *existing* 40-prompt dataset, not the raw ShareGPT JSON.
The source is never modified. The downstream runner probes actual token counts,
calibrates q(x,y), refines a measured DP plan, and benchmarks DP only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

GROUPS = (128, 256, 512, 1024)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def select_first(source: Path, per_group: int) -> list[dict]:
    if not source.is_dir():
        raise ValueError(f"source directory not found: {source}")
    if per_group < 1:
        raise ValueError("--per-group must be positive")
    picked = []
    for group in GROUPS:
        for index in range(1, per_group + 1):
            name = f"prompt_{group}_{index:02d}.txt"
            file = source / name
            if not file.is_file():
                raise ValueError(f"missing expected prompt: {file}")
            if not file.read_text(encoding="utf-8-sig").strip():
                raise ValueError(f"prompt is empty: {file}")
            picked.append({
                "group": group, "index": index, "prompt": name,
                "source": str(file.resolve()), "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            })
    return picked


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", default=r"E:\llama\datasets\sharegpt\sharegpt40")
    p.add_argument("--out-dir", default="jupiter-sharegpt-quick8-dp")
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--exe", default=r".\build-jupiter\bin\Release\llama-cli.exe")
    p.add_argument("--model", default=r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf")
    p.add_argument("--rpc", default="10.39.52.114:10139")
    p.add_argument("--pc-layers", type=int, default=24)
    p.add_argument("--min-chunk", type=int, default=96)
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--ubatch", type=int, default=128)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--q-repeats", type=int, default=1)
    p.add_argument("--refine-repeats", type=int, default=1)
    p.add_argument("--bench-repeats", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=1)
    p.add_argument("--graph-prebuild", action="store_true")
    p.add_argument("--strict-refine", action="store_true",
                   help="require exact-q DP reoptimization convergence; slower")
    p.add_argument("--dry-run", action="store_true",
                   help="show selected source prompts without copying or running")
    p.add_argument("--sample-only", action="store_true",
                   help="copy selected prompts and write manifest, then exit")
    return p


def make_identity(args: argparse.Namespace, picked: list[dict]) -> dict:
    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    return {
        "version": 1, "selection": picked,
        "execution": {
            "exe": str(exe), "model": str(model), "rpc": args.rpc,
            "pc_layers": args.pc_layers,
            "min_chunk": args.min_chunk, "max_chunk": args.max_chunk,
            "ubatch": args.ubatch, "batch": args.batch,
            "context": args.context, "threads": args.threads,
            "window": args.window, "q_repeats": args.q_repeats,
            "refine_repeats": args.refine_repeats,
            "bench_repeats": args.bench_repeats,
            "max_rounds": args.max_rounds,
            "graph_prebuild": args.graph_prebuild,
            "strict_refine": args.strict_refine,
        },
    }


def prepare_sample(out: Path, picked: list[dict], identity: dict) -> Path:
    manifest = out / "quick8-config.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(
                f"existing run configuration differs: {manifest}; "
                "use a separate --out-dir for another model or settings"
            )
    prompt_dir = out / "sampled_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for entry in picked:
        dest = prompt_dir / entry["prompt"]
        if dest.exists():
            if file_sha256(dest) != entry["sha256"]:
                raise ValueError(f"sampled file has changed: {dest}")
        else:
            shutil.copy2(entry["source"], dest)
            if file_sha256(dest) != entry["sha256"]:
                raise RuntimeError(f"file copy checksum mismatch: {dest}")
    with (out / "sample-manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group", "index", "prompt", "source", "bytes", "sha256"
        ])
        writer.writeheader()
        writer.writerows(picked)
    manifest.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt_dir


def build_command(args: argparse.Namespace, prompt_dir: Path, out: Path) -> list[str]:
    runner = Path(__file__).resolve().with_name("run_sharegpt40_dp.py")
    if not runner.is_file():
        raise ValueError(f"missing Jupiter runner beside this script: {runner}")
    cmd = [
        sys.executable, str(runner),
        "--prompts-dir", str(prompt_dir),
        "--out-dir", str(out / "dp"),
        "--groups", ",".join(str(g) for g in GROUPS),
        "--exe", str(Path(args.exe).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--rpc", args.rpc, "--pc-layers", str(args.pc_layers),
        "--min-chunk", str(args.min_chunk),
        "--max-chunk", str(args.max_chunk),
        "--ubatch", str(args.ubatch), "--batch", str(args.batch),
        "--context", str(args.context), "--threads", str(args.threads),
        "--window", str(args.window),
        "--q-repeats", str(args.q_repeats),
        "--refine-repeats", str(args.refine_repeats),
        "--bench-repeats", str(args.bench_repeats),
        "--max-rounds", str(args.max_rounds),
        "--resume", "--only-incomplete",
    ]
    if args.graph_prebuild:
        cmd.append("--graph-prebuild")
    if not args.strict_refine:
        cmd.append("--budgeted-refine")
    return cmd


def main() -> int:
    args = make_parser().parse_args()
    if not (0 < args.min_chunk <= args.max_chunk <= args.ubatch <= args.batch
            <= args.context):
        raise ValueError("require 0 < min-chunk <= max-chunk <= ubatch <= batch <= context")
    if min(args.pc_layers, args.threads, args.window, args.q_repeats,
           args.refine_repeats, args.bench_repeats, args.max_rounds) < 1:
        raise ValueError("pc-layers, threads, window and repeats must be positive")
    source = Path(args.source_dir).resolve()
    picked = select_first(source, args.per_group)
    print(f"[JUPITER_QUICK8_SELECT] groups={GROUPS} per_group={args.per_group} "
          f"total={len(picked)}", flush=True)
    for item in picked:
        print(f"  {item['prompt']} ({item['bytes']} bytes)", flush=True)
    if args.dry_run:
        return 0
    for label, path in (("--exe", args.exe), ("--model", args.model)):
        if not Path(path).is_file() and not args.sample_only:
            raise ValueError(f"{label} not found: {path}")
    out = Path(args.out_dir).resolve()
    prompt_dir = prepare_sample(out, picked, make_identity(args, picked))
    print(f"[JUPITER_QUICK8_SAMPLE] path={prompt_dir} files={len(picked)}", flush=True)
    if args.sample_only:
        return 0
    cmd = build_command(args, prompt_dir, out)
    print("[JUPITER_QUICK8_RUN] " + subprocess.list2cmdline(cmd), flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[JUPITER_QUICK8_DONE] rc={rc} "
          f"results={out / 'dp' / 'dp-results.csv'} "
          f"groups={out / 'dp' / 'dp-groups.csv'}", flush=True)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[JUPITER_QUICK8_ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("[JUPITER_QUICK8_STOP] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
