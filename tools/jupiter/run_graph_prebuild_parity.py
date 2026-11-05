#!/usr/bin/env python3
"""Compare Jupiter graph-prebuild against the unmodified path.

Diagnostic only: LLAMA_JUPITER_LOGITS_HASH forces a final synchronization, so
prefill times from these trials must NOT be used as formal A/B throughput.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

HASH_RE = re.compile(
    r"\\[JUPITER_LOGITS_HASH\\].*?tokens=(\\d+).*?outputs=(\\d+)"
    r".*?top1_id=(-?\\d+).*?digest=([0-9a-fA-F]+)"
)

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "model", "rpc", "prompt-file", "plan", "target-tokens"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--pc-layers", type=int, default=24)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--out-dir", default="jupiter-prebuild-parity")
    return p

def extract(log: Path, target: int) -> tuple[int, int, str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = [m for m in HASH_RE.finditer(text) if int(m.group(1)) == target]
    if not matches:
        raise ValueError(f"missing prompt logits digest for {target} tokens in {log}")
    m = matches[-1]
    return int(m.group(2)), int(m.group(3)), m.group(4).lower()

def main() -> int:
    args = parser().parse_args()
    target = int(args.target_tokens)
    out = Path(args.out_dir).resolve()
    script = Path(__file__).with_name("run_prefill_bench.py")
    baseline = {}
    for variant in ("baseline", "prebuild"):
        dest = out / variant
        cmd = [
            sys.executable, str(script),
            "--exe", args.exe, "--model", args.model, "--rpc", args.rpc,
            "--prompt-file", args.prompt_file, "--plan", args.plan,
            "--target-tokens", str(target), "--pc-layers", str(args.pc_layers),
            "--modes", "default,dp", "--repeats", "1",
            "--batch", str(args.batch), "--context", str(args.context),
            "--ubatch", str(args.ubatch), "--threads", str(args.threads),
            "--window", str(args.window), "--timing-token-tolerance", "1",
            "--out-dir", str(dest),
        ]
        if variant == "prebuild":
            cmd.append("--graph-prebuild")
        env = os.environ.copy()
        env["LLAMA_JUPITER_LOGITS_HASH"] = "1"
        print(f"[JUPITER_PREBUILD_PARITY] variant={variant} event=RUN", flush=True)
        subprocess.run(cmd, env=env, check=True)
        baseline[variant] = {
            mode: extract(dest / f"{mode}-r1.log", target)
            for mode in ("default", "dp")
        }
    ok = True
    for mode in ("default", "dp"):
        original = baseline["baseline"][mode]
        prepared = baseline["prebuild"][mode]
        match = original == prepared
        ok &= match
        print(f"[JUPITER_PREBUILD_PARITY] mode={mode} "
              f"baseline={original} prebuild={prepared} "
              f"status={'PASS' if match else 'FAIL'}", flush=True)
    print("[JUPITER_PREBUILD_PARITY_DONE] status=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[JUPITER_PREBUILD_PARITY_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
