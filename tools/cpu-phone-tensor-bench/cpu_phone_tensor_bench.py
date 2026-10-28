#!/usr/bin/env python3
"""
CPU vs Phone full-layer / directional-Tensor benchmark.

The benchmark keeps all model layers outside a configurable experiment window
on PC CPU.  It searches two independent families inside the window:

  Layer planner:
      one contiguous PHONE_FULL block, PC_FULL everywhere else.

  Tensor planner:
      every layer in the window is Tensor and ownership alternates
      C2P, P2C, C2P, P2C, ...

Each candidate is a real llama-cli run.  The winner is selected by measured
prompt-eval wall time, not by a simulator.

This tool deliberately disables Region Ring / Hybrid Auto.  It relies on
LLAMA_HYBRID_EXPERIMENT_POLICY, an explicit per-layer policy added for this
benchmark branch.
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
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*([0-9]+)\s*tokens",
    re.IGNORECASE,
)
LAYER_SUM_RE = re.compile(
    r"\[META_LAYER_SUM\]\s+layer=(\d+).*?total=([0-9]+(?:\.[0-9]+)?)\s*ms"
)


@dataclass(frozen=True)
class Candidate:
    family: str
    x_layers: int
    name: str
    policy: Tuple[str, ...]
    c2p_ratio: float = 1.0 / 3.0
    p2c_ratio: float = 1.0 / 3.0


@dataclass
class Result:
    family: str
    x_layers: int
    name: str
    c2p_ratio: float
    p2c_ratio: float
    prompt_ms: Optional[float]
    prompt_tokens: Optional[int]
    window_layer_ms: Optional[float]
    returncode: int
    reps_ok: int
    log_files: List[str]
    policy: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.prompt_ms is not None


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
        raise ValueError("--layers must contain positive integers")
    return values


def parse_float_set(spec: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in spec.split(",") if x.strip()))
    if not values or any(not (0.0 < x < 1.0) for x in values):
        raise ValueError("--ratios must contain values strictly between 0 and 1")
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
        "GGML_META_REGION_C2P_SELF_CHECK",
        "GGML_META_AB_TRACE_LAYER",
    }
    prefixes = (
        "LLAMA_HYBRID_FIXED_",
        "LLAMA_HYBRID_REGION_RING_PLAN_",
    )
    for key in list(env):
        if key in exact or any(key.startswith(prefix) for prefix in prefixes):
            env.pop(key, None)

    # The experiment policy is the only placement source of truth.
    env["LLAMA_HYBRID_GPU_PC_LAYERS"] = "0"
    env["LLAMA_HYBRID_PC_PCT"] = "100"
    env["LLAMA_HYBRID_PHONE_LAYERS"] = "0"
    env["LLAMA_HYBRID_PC_LAYOUT"] = "tail"
    return env


def apply_window(
    model_layers: int,
    start_layer: int,
    x_layers: int,
    window_modes: Sequence[str],
) -> Tuple[str, ...]:
    if len(window_modes) != x_layers:
        raise ValueError("window mode count does not match x_layers")
    end = start_layer + x_layers
    if start_layer < 0 or end > model_layers:
        raise ValueError(
            f"window [{start_layer},{end}) exceeds model layer count {model_layers}"
        )
    modes = ["C"] * model_layers
    modes[start_layer:end] = list(window_modes)
    return tuple(modes)


def layer_candidates(
    model_layers: int,
    start_layer: int,
    x_layers: int,
    max_phone_layers: Optional[int],
) -> Iterable[Candidate]:
    # One contiguous Phone block is sufficient for the first experiment:
    # adding another Phone block adds two ownership handoffs without changing
    # the homogeneous per-layer compute backend.
    yield Candidate(
        family="layer",
        x_layers=x_layers,
        name="all_cpu",
        policy=apply_window(model_layers, start_layer, x_layers, ["C"] * x_layers),
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
                policy=apply_window(model_layers, start_layer, x_layers, modes),
            )


def tensor_candidates(
    model_layers: int,
    start_layer: int,
    x_layers: int,
    ratios: Sequence[float],
    independent_ratios: bool,
) -> Iterable[Candidate]:
    modes = ["C2P" if i % 2 == 0 else "P2C" for i in range(x_layers)]
    policy = apply_window(model_layers, start_layer, x_layers, modes)

    if independent_ratios:
        pairs = ((a, b) for a in ratios for b in ratios)
    else:
        pairs = ((r, r) for r in ratios)

    for c2p, p2c in pairs:
        yield Candidate(
            family="tensor",
            x_layers=x_layers,
            name=f"alt_c2p_p2c_r{c2p:.3f}_{p2c:.3f}",
            policy=policy,
            c2p_ratio=c2p,
            p2c_ratio=p2c,
        )


def candidate_key(candidate: Candidate, args: argparse.Namespace) -> str:
    payload = {
        "family": candidate.family,
        "x": candidate.x_layers,
        "policy": candidate.policy,
        "c2p": candidate.c2p_ratio,
        "p2c": candidate.p2c_ratio,
        "model": str(Path(args.model).resolve()),
        "rpc": args.rpc,
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
    cmd = [
        args.exe,
        "-m", args.model,
        "--rpc", args.rpc,
        "--split-mode", "tensor",
        "-fit", "off",
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


def parse_run_output(text: str, start_layer: int, x_layers: int) -> Tuple[Optional[float], Optional[int], Optional[float]]:
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

    window_ms: Optional[float] = None
    wanted = range(start_layer, start_layer + x_layers)
    if all(layer in layer_total for layer in wanted):
        window_ms = sum(layer_total[layer] for layer in wanted)

    return prompt_ms, prompt_tokens, window_ms


def run_candidate(
    candidate: Candidate,
    args: argparse.Namespace,
    reps: int,
    out_dir: Path,
) -> Result:
    key = candidate_key(candidate, args)
    prompt_times: List[float] = []
    window_times: List[float] = []
    prompt_tokens: Optional[int] = None
    logs: List[str] = []
    final_rc = 0

    for rep in range(reps):
        env = sanitize_env(os.environ)
        env["LLAMA_HYBRID_EXPERIMENT_POLICY"] = ",".join(candidate.policy)
        env["LLAMA_HYBRID_REGION_C2P_PC_RATIO"] = f"{candidate.c2p_ratio:.9f}"
        env["LLAMA_HYBRID_REGION_P2C_PC_RATIO"] = f"{candidate.p2c_ratio:.9f}"

        for item in args.extra_env:
            if "=" not in item:
                raise ValueError(f"--extra-env must be KEY=VALUE: {item}")
            k, v = item.split("=", 1)
            env[k] = v

        cmd = build_command(args)
        log_path = out_dir / f"{candidate.family}-x{candidate.x_layers}-{key}-r{rep}.log"
        logs.append(str(log_path))

        print(
            f"[BENCH_RUN] family={candidate.family} x={candidate.x_layers} "
            f"name={candidate.name} c2p={candidate.c2p_ratio:.3f} "
            f"p2c={candidate.p2c_ratio:.3f} rep={rep + 1}/{reps}",
            flush=True,
        )

        if args.dry_run:
            print("  ENV LLAMA_HYBRID_EXPERIMENT_POLICY=" + ",".join(candidate.policy))
            print("  CMD " + shlex.join(cmd))
            prompt_times.append(float("inf"))
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

        prompt_ms, tokens, window_ms = parse_run_output(
            text, args.start_layer, candidate.x_layers
        )
        if proc.returncode != 0 or prompt_ms is None:
            print(
                f"[BENCH_FAIL] rc={proc.returncode} prompt_ms={prompt_ms} "
                f"log={log_path}",
                flush=True,
            )
            final_rc = proc.returncode if proc.returncode != 0 else 2
            break

        prompt_times.append(prompt_ms)
        prompt_tokens = tokens
        if window_ms is not None:
            window_times.append(window_ms)
        print(
            f"[BENCH_RESULT] prompt_ms={prompt_ms:.3f} "
            f"window_layer_ms={window_ms if window_ms is not None else 'NA'}",
            flush=True,
        )

    return Result(
        family=candidate.family,
        x_layers=candidate.x_layers,
        name=candidate.name,
        c2p_ratio=candidate.c2p_ratio,
        p2c_ratio=candidate.p2c_ratio,
        prompt_ms=statistics.median(prompt_times) if prompt_times else None,
        prompt_tokens=prompt_tokens,
        window_layer_ms=statistics.median(window_times) if window_times else None,
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
        "prompt_ms", "prompt_tokens", "window_layer_ms", "returncode",
        "reps_ok", "policy", "log_files",
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
        description="Measured CPU-vs-Phone Layer/Tensor planner benchmark"
    )
    p.add_argument("--exe", required=True, help="path to llama-cli(.exe)")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True, help="phone RPC endpoint, e.g. 192.168.1.20:50052")
    p.add_argument("--model-layers", type=int, default=48)
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument(
        "--layers", default="1:8",
        help="x-layer values to test, e.g. 1:8 or 2,4,6,8",
    )
    p.add_argument(
        "--ratios", default="0.20,0.33,0.50",
        help="PC FFN share candidates for directional Tensor",
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
    if args.model_layers <= 0:
        p.error("--model-layers must be positive")
    if args.start_layer < 0:
        p.error("--start-layer must be >= 0")
    if any(args.start_layer + x > args.model_layers for x in xs):
        p.error("one of the requested --layers windows exceeds --model-layers")
    if args.search_reps <= 0 or args.winner_reps <= 0:
        p.error("rep counts must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Result] = []
    summary: List[Dict[str, object]] = []

    for x in xs:
        print(f"\n[BENCH_X_BEGIN] x={x} window=[{args.start_layer},{args.start_layer + x})")

        layer_results = [
            run_candidate(c, args, args.search_reps, out_dir)
            for c in layer_candidates(
                args.model_layers,
                args.start_layer,
                x,
                args.max_phone_layers,
            )
        ]
        tensor_results = [
            run_candidate(c, args, args.search_reps, out_dir)
            for c in tensor_candidates(
                args.model_layers,
                args.start_layer,
                x,
                ratios,
                independent_ratios=not args.same_ratio,
            )
        ]
        all_results.extend(layer_results)
        all_results.extend(tensor_results)

        best_layer = choose_best(layer_results)
        best_tensor = choose_best(tensor_results)
        if best_layer is None or best_tensor is None:
            print(
                f"[BENCH_X_FAIL] x={x} "
                f"layer_ok={best_layer is not None} tensor_ok={best_tensor is not None}"
            )
            continue

        # Re-run each winner to reduce one-shot noise.
        layer_candidate = next(
            c for c in layer_candidates(
                args.model_layers, args.start_layer, x, args.max_phone_layers
            )
            if c.name == best_layer.name
        )
        tensor_candidate = next(
            c for c in tensor_candidates(
                args.model_layers, args.start_layer, x, ratios,
                independent_ratios=not args.same_ratio
            )
            if c.name == best_tensor.name
            and abs(c.c2p_ratio - best_tensor.c2p_ratio) < 1e-9
            and abs(c.p2c_ratio - best_tensor.p2c_ratio) < 1e-9
        )

        best_layer = run_candidate(
            layer_candidate, args, args.winner_reps, out_dir
        )
        best_tensor = run_candidate(
            tensor_candidate, args, args.winner_reps, out_dir
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
            "layer_prompt_ms": best_layer.prompt_ms,
            "tensor_name": best_tensor.name,
            "tensor_c2p_ratio": best_tensor.c2p_ratio,
            "tensor_p2c_ratio": best_tensor.p2c_ratio,
            "tensor_prompt_ms": best_tensor.prompt_ms,
            "tensor_speedup_vs_layer": speedup,
            "winner": winner,
        }
        summary.append(row)
        print(
            f"[BENCH_X_WINNER] x={x} layer_ms={best_layer.prompt_ms:.3f} "
            f"tensor_ms={best_tensor.prompt_ms:.3f} "
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
