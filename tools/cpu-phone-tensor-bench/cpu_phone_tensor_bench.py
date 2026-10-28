#!/usr/bin/env python3
"""
Real x-layer CPU/Phone Layer-vs-Tensor benchmark.

For every requested x this program starts llama-cli with
LLAMA_HYBRID_EXPERIMENT_N_LAYERS=x.  The model loader therefore creates and
loads only blk.0 ... blk.(x-1); blk.x ... blk.(N-1) are never materialized.

Two measured planners are compared:

Layer planner:
    The x active layers are PC CPU Full or Phone Full.  It enumerates a single
    contiguous Phone block (including all-CPU and all-Phone) and keeps the
    measured winner.

Tensor planner:
    All x active layers are alternating directional Tensor:
        C2P, P2C, C2P, P2C, ...
    It searches C2P/P2C FFN PC shares and Tensor chunk size, then keeps the
    measured winner.

Every candidate is a real llama-cli prompt evaluation over PC CPU + Phone RPC.
Local CUDA is excluded with --device RPC0 (configurable).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*([0-9]+)\s*tokens",
    re.IGNORECASE,
)
LAYER_SUM_RE = re.compile(
    r"\[META_LAYER_SUM\]\s+layer=(\d+).*?total=([0-9]+(?:\.[0-9]+)?)\s*ms"
)
TRUNCATE_RE = re.compile(
    r"\[HYBRID_EXPERIMENT_TRUNCATE\].*?original_layers=(\d+).*?active_layers=(\d+)"
)


@dataclass(frozen=True)
class Candidate:
    family: str
    x_layers: int
    name: str
    policy: Tuple[str, ...]
    c2p_ratio: float = 1.0 / 3.0
    p2c_ratio: float = 1.0 / 3.0
    tensor_chunk: int = 0


@dataclass
class Result:
    family: str
    x_layers: int
    name: str
    c2p_ratio: float
    p2c_ratio: float
    tensor_chunk: int
    prompt_ms: Optional[float]
    prompt_tokens: Optional[int]
    active_layer_ms: Optional[float]
    original_layers: Optional[int]
    loaded_layers: Optional[int]
    returncode: int
    reps_ok: int
    log_files: List[str]
    policy: str

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and self.prompt_ms is not None
            and self.loaded_layers == self.x_layers
        )


def parse_int_set(spec: str) -> List[int]:
    out: List[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid integer range: {item}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError(f"invalid integer range: {item}")
            out.extend(range(start, stop + 1, step))
        else:
            out.append(int(item))
    values = sorted(set(out))
    if not values or any(v <= 0 for v in values):
        raise ValueError("integer set must contain positive values")
    return values


def parse_float_set(spec: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in spec.split(",") if x.strip()))
    if not values or any(not (0.0 < x < 1.0) for x in values):
        raise ValueError("ratios must be strictly between 0 and 1")
    return values


def sanitize_env(base: Dict[str, str]) -> Dict[str, str]:
    env = dict(base)
    exact = {
        "LLAMA_HYBRID_REGION_RING_V1",
        "LLAMA_HYBRID_REGION_RING_EXEC",
        "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM",
        "LLAMA_HYBRID_REGION_RING_LAYOUT",
        "LLAMA_HYBRID_REGION_RING_PLAN_MODE",
        "LLAMA_HYBRID_REGION_BOUNDARY_V1",
        "LLAMA_HYBRID_FORCE_PHONE_PRIMARY",
        "LLAMA_HYBRID_PHONE_PRIMARY_EXEC",
        "LLAMA_HYBRID_EXPERIMENT_POLICY",
        "LLAMA_HYBRID_EXPERIMENT_N_LAYERS",
        "LLAMA_HYBRID_EXPERIMENT_TENSOR_CHUNK_TOKENS",
        "GGML_META_REGION_C2P_SELF_CHECK",
        "GGML_META_AB_TRACE_LAYER",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_RPC",
        "LLAMA_ARG_N_GPU_LAYERS",
    }
    prefixes = (
        "LLAMA_HYBRID_FIXED_",
        "LLAMA_HYBRID_REGION_RING_PLAN_",
    )
    for key in list(env):
        if key in exact or any(key.startswith(prefix) for prefix in prefixes):
            env.pop(key, None)

    # PC Full always means CPU in this benchmark.
    env["LLAMA_HYBRID_GPU_PC_LAYERS"] = "0"
    env["LLAMA_HYBRID_PC_PCT"] = "100"
    env["LLAMA_HYBRID_PHONE_LAYERS"] = "0"
    env["LLAMA_HYBRID_PC_LAYOUT"] = "tail"
    return env


def layer_candidates(x_layers: int, max_phone_layers: Optional[int]) -> Iterable[Candidate]:
    yield Candidate(
        family="layer",
        x_layers=x_layers,
        name="all_cpu",
        policy=tuple(["C"] * x_layers),
    )

    max_len = x_layers if max_phone_layers is None else min(x_layers, max_phone_layers)
    for phone_len in range(1, max_len + 1):
        for offset in range(0, x_layers - phone_len + 1):
            modes = ["C"] * x_layers
            modes[offset : offset + phone_len] = ["P"] * phone_len
            yield Candidate(
                family="layer",
                x_layers=x_layers,
                name=f"phone_{offset}_{offset + phone_len}",
                policy=tuple(modes),
            )


def tensor_candidates(
    x_layers: int,
    ratios: Sequence[float],
    tensor_chunks: Sequence[int],
    independent_ratios: bool,
) -> Iterable[Candidate]:
    policy = tuple("C2P" if i % 2 == 0 else "P2C" for i in range(x_layers))
    pairs = (
        ((a, b) for a in ratios for b in ratios)
        if independent_ratios
        else ((r, r) for r in ratios)
    )
    for c2p, p2c in pairs:
        for chunk in tensor_chunks:
            yield Candidate(
                family="tensor",
                x_layers=x_layers,
                name=f"alt_r{c2p:.3f}_{p2c:.3f}_xt{chunk}",
                policy=policy,
                c2p_ratio=c2p,
                p2c_ratio=p2c,
                tensor_chunk=chunk,
            )


def candidate_key(candidate: Candidate, args: argparse.Namespace) -> str:
    payload = {
        "candidate": asdict(candidate),
        "model": str(Path(args.model).resolve()),
        "rpc": args.rpc,
        "phone_device": args.phone_device,
        "ctx": args.ctx,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "threads": args.threads,
        "prompt_file": str(Path(args.prompt_file).resolve()) if args.prompt_file else None,
        "prompt": args.prompt,
        "extra": args.extra,
        "extra_env": args.extra_env,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()[:12]


def build_command(args: argparse.Namespace) -> List[str]:
    # --rpc must appear before --device so RPC0 is registered before the device
    # list is parsed.  Supplying only RPC0 excludes the local CUDA device.
    cmd = [
        args.exe,
        "-m", args.model,
        "--rpc", args.rpc,
        "--device", args.phone_device,
        "--split-mode", "tensor",
        "-fit", "off",
        "--no-mmap",
        "--flash-attn", args.flash_attn,
        "-c", str(args.ctx),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-n", "1",
        "--seed", str(args.seed),
        "--no-conversation",
        "-lv", str(args.log_verbosity),
    ]
    if args.prompt_file:
        cmd += ["-f", args.prompt_file]
    else:
        cmd += ["-p", args.prompt]
    cmd += args.extra
    return cmd


def parse_run_output(
    text: str, x_layers: int
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int], Optional[int]]:
    prompt_matches = list(PROMPT_RE.finditer(text))
    prompt_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    if prompt_matches:
        m = prompt_matches[-1]
        prompt_ms = float(m.group(1))
        prompt_tokens = int(m.group(2))

    layer_total: Dict[int, float] = {}
    for m in LAYER_SUM_RE.finditer(text):
        layer_total[int(m.group(1))] = float(m.group(2))
    active_layer_ms = (
        sum(layer_total[i] for i in range(x_layers))
        if all(i in layer_total for i in range(x_layers))
        else None
    )

    trunc = TRUNCATE_RE.search(text)
    original_layers = int(trunc.group(1)) if trunc else None
    loaded_layers = int(trunc.group(2)) if trunc else None
    return prompt_ms, prompt_tokens, active_layer_ms, original_layers, loaded_layers


def run_candidate(
    candidate: Candidate,
    args: argparse.Namespace,
    reps: int,
    out_dir: Path,
) -> Result:
    key = candidate_key(candidate, args)
    prompt_times: List[float] = []
    active_layer_times: List[float] = []
    prompt_tokens: Optional[int] = None
    original_layers: Optional[int] = None
    loaded_layers: Optional[int] = None
    logs: List[str] = []
    final_rc = 0

    for rep in range(reps):
        env = sanitize_env(os.environ)
        env["LLAMA_HYBRID_EXPERIMENT_N_LAYERS"] = str(candidate.x_layers)
        env["LLAMA_HYBRID_EXPERIMENT_POLICY"] = ",".join(candidate.policy)
        env["LLAMA_HYBRID_REGION_C2P_PC_RATIO"] = f"{candidate.c2p_ratio:.9f}"
        env["LLAMA_HYBRID_REGION_P2C_PC_RATIO"] = f"{candidate.p2c_ratio:.9f}"
        if candidate.tensor_chunk > 0:
            env["LLAMA_HYBRID_EXPERIMENT_TENSOR_CHUNK_TOKENS"] = str(
                candidate.tensor_chunk
            )

        for item in args.extra_env:
            if "=" not in item:
                raise ValueError(f"--extra-env must be KEY=VALUE: {item}")
            k, v = item.split("=", 1)
            env[k] = v

        cmd = build_command(args)
        log_path = out_dir / (
            f"{candidate.family}-x{candidate.x_layers}-{key}-r{rep}.log"
        )
        logs.append(str(log_path))

        print(
            f"[BENCH_RUN] family={candidate.family} x={candidate.x_layers} "
            f"name={candidate.name} c2p={candidate.c2p_ratio:.3f} "
            f"p2c={candidate.p2c_ratio:.3f} xt={candidate.tensor_chunk} "
            f"rep={rep + 1}/{reps}",
            flush=True,
        )

        if args.dry_run:
            print("  ENV LLAMA_HYBRID_EXPERIMENT_N_LAYERS=" + str(candidate.x_layers))
            print("  ENV LLAMA_HYBRID_EXPERIMENT_POLICY=" + ",".join(candidate.policy))
            print("  CMD " + shlex.join(cmd))
            prompt_times.append(float("inf"))
            loaded_layers = candidate.x_layers
            continue

        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        text = proc.stdout
        log_path.write_text(text, encoding="utf-8", errors="replace")
        final_rc = proc.returncode

        (
            prompt_ms,
            tokens,
            active_ms,
            original,
            loaded,
        ) = parse_run_output(text, candidate.x_layers)

        original_layers = original
        loaded_layers = loaded
        if (
            proc.returncode != 0
            or prompt_ms is None
            or loaded_layers != candidate.x_layers
        ):
            print(
                f"[BENCH_FAIL] rc={proc.returncode} prompt_ms={prompt_ms} "
                f"loaded_layers={loaded_layers} expected={candidate.x_layers} "
                f"log={log_path}",
                flush=True,
            )
            final_rc = proc.returncode if proc.returncode != 0 else 2
            break

        prompt_times.append(prompt_ms)
        prompt_tokens = tokens
        if active_ms is not None:
            active_layer_times.append(active_ms)
        print(
            f"[BENCH_RESULT] prompt_ms={prompt_ms:.3f} "
            f"active_layer_ms={active_ms if active_ms is not None else 'NA'} "
            f"loaded={loaded_layers}/{original_layers}",
            flush=True,
        )

    return Result(
        family=candidate.family,
        x_layers=candidate.x_layers,
        name=candidate.name,
        c2p_ratio=candidate.c2p_ratio,
        p2c_ratio=candidate.p2c_ratio,
        tensor_chunk=candidate.tensor_chunk,
        prompt_ms=statistics.median(prompt_times) if prompt_times else None,
        prompt_tokens=prompt_tokens,
        active_layer_ms=(
            statistics.median(active_layer_times) if active_layer_times else None
        ),
        original_layers=original_layers,
        loaded_layers=loaded_layers,
        returncode=final_rc,
        reps_ok=len(prompt_times),
        log_files=logs,
        policy=",".join(candidate.policy),
    )


def choose_best(results: Sequence[Result]) -> Optional[Result]:
    ok = [r for r in results if r.ok]
    return min(ok, key=lambda r: r.prompt_ms) if ok else None


def write_csv(path: Path, results: Sequence[Result]) -> None:
    fields = [
        "family", "x_layers", "name", "c2p_ratio", "p2c_ratio",
        "tensor_chunk", "prompt_ms", "prompt_tokens", "active_layer_ms",
        "original_layers", "loaded_layers", "returncode", "reps_ok",
        "policy", "log_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for result in results:
            row = asdict(result)
            row["log_files"] = ";".join(result.log_files)
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measured truncated x-layer CPU-vs-Phone Layer/Tensor benchmark"
    )
    p.add_argument("--exe", required=True, help="path to llama-cli(.exe)")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True, help="phone RPC endpoint")
    p.add_argument(
        "--phone-device", default="RPC0",
        help="registered Phone device name; default RPC0",
    )
    p.add_argument(
        "--layers", default="1:8",
        help="actual model layer counts x to load, e.g. 1:8 or 2,4,6,8",
    )
    p.add_argument(
        "--ratios", default="0.20,0.33,0.50",
        help="PC FFN shares searched for C2P/P2C",
    )
    p.add_argument(
        "--tensor-chunks", default="32,64,128",
        help="Tensor prefill chunk sizes searched by the Tensor planner",
    )
    p.add_argument(
        "--same-ratio", action="store_true",
        help="force C2P and P2C to use the same ratio candidate",
    )
    p.add_argument("--max-phone-layers", type=int, default=None)
    p.add_argument("--search-reps", type=int, default=1)
    p.add_argument("--winner-reps", type=int, default=2)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--flash-attn", default="on")
    p.add_argument("--log-verbosity", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-file")
    prompt.add_argument("--prompt")
    p.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="extra arguments appended verbatim to llama-cli",
    )
    p.add_argument(
        "--extra-env", action="append", default=[],
        help="extra environment KEY=VALUE; may be repeated",
    )
    p.add_argument("--out", default="cpu-phone-tensor-bench-results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    xs = parse_int_set(args.layers)
    ratios = parse_float_set(args.ratios)
    tensor_chunks = parse_int_set(args.tensor_chunks)
    if args.search_reps <= 0 or args.winner_reps <= 0:
        p.error("rep counts must be positive")
    if args.max_phone_layers is not None and args.max_phone_layers < 0:
        p.error("--max-phone-layers must be >= 0")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Result] = []
    summary: List[Dict[str, object]] = []

    for x in xs:
        print(f"\n[BENCH_X_BEGIN] x={x} actual_loaded_transformer_layers={x}")

        layer_cs = list(layer_candidates(x, args.max_phone_layers))
        tensor_cs = list(
            tensor_candidates(
                x,
                ratios,
                tensor_chunks,
                independent_ratios=not args.same_ratio,
            )
        )

        layer_results = [
            run_candidate(c, args, args.search_reps, out_dir) for c in layer_cs
        ]
        tensor_results = [
            run_candidate(c, args, args.search_reps, out_dir) for c in tensor_cs
        ]
        all_results.extend(layer_results)
        all_results.extend(tensor_results)

        search_layer = choose_best(layer_results)
        search_tensor = choose_best(tensor_results)
        if search_layer is None or search_tensor is None:
            print(
                f"[BENCH_X_FAIL] x={x} "
                f"layer_ok={search_layer is not None} "
                f"tensor_ok={search_tensor is not None}"
            )
            continue

        layer_winner = next(
            c for c in layer_cs
            if c.name == search_layer.name
        )
        tensor_winner = next(
            c for c in tensor_cs
            if c.name == search_tensor.name
            and abs(c.c2p_ratio - search_tensor.c2p_ratio) < 1e-9
            and abs(c.p2c_ratio - search_tensor.p2c_ratio) < 1e-9
            and c.tensor_chunk == search_tensor.tensor_chunk
        )

        best_layer = run_candidate(
            layer_winner, args, args.winner_reps, out_dir
        )
        best_tensor = run_candidate(
            tensor_winner, args, args.winner_reps, out_dir
        )
        all_results.extend([best_layer, best_tensor])

        speedup = (
            best_layer.prompt_ms / best_tensor.prompt_ms
            if best_layer.ok and best_tensor.ok and best_tensor.prompt_ms
            else None
        )
        winner = (
            "tensor" if speedup is not None and speedup > 1.0 else "layer"
        )
        row = {
            "x_layers": x,
            "layer_name": best_layer.name,
            "layer_policy": best_layer.policy,
            "layer_prompt_ms": best_layer.prompt_ms,
            "tensor_name": best_tensor.name,
            "tensor_policy": best_tensor.policy,
            "tensor_c2p_ratio": best_tensor.c2p_ratio,
            "tensor_p2c_ratio": best_tensor.p2c_ratio,
            "tensor_chunk": best_tensor.tensor_chunk,
            "tensor_prompt_ms": best_tensor.prompt_ms,
            "tensor_speedup_vs_layer": speedup,
            "winner": winner,
        }
        summary.append(row)
        print(
            f"[BENCH_X_WINNER] x={x} "
            f"layer={best_layer.policy} {best_layer.prompt_ms:.3f}ms "
            f"tensor={best_tensor.policy} "
            f"R={best_tensor.c2p_ratio:.3f}/{best_tensor.p2c_ratio:.3f} "
            f"XT={best_tensor.tensor_chunk} {best_tensor.prompt_ms:.3f}ms "
            f"speedup={speedup:.4f} winner={winner}"
        )

    write_csv(out_dir / "all-results.csv", all_results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n[BENCH_SUMMARY]")
    for row in summary:
        print(
            f"x={row['x_layers']:>2} "
            f"layer={row['layer_prompt_ms']:.3f}ms "
            f"tensor={row['tensor_prompt_ms']:.3f}ms "
            f"speedup={row['tensor_speedup_vs_layer']:.4f} "
            f"winner={row['winner']}"
        )
    print(f"results: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"[BENCH_FATAL] timeout: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"[BENCH_FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
