#!/usr/bin/env python3
"""Compare TOKEN LOGITS TOP 10 blocks from two llama-cli logs.

The script is intentionally text-only: it finds the first decode step where the
Top-1 token diverges, and separately reports the first Top-1 logit drift beyond
the requested absolute tolerance.

Usage:
    python tools/compare_token_logits.py c2p.log baseline.log
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

DECODED_RE = re.compile(
    r"^decoded=(?P<step>\\d+)\\s+source=(?P<source>\\S+)\\s+"
    r"count=(?P<count>\\d+)\\s+zero=(?P<zero>\\d+)\\s+"
    r"nonfinite=(?P<nonfinite>\\d+)\\s*$"
)
TOP_RE = re.compile(
    r"^top\\[(?P<rank>\\d+)\\]:\\s+token=(?P<token>-?\\d+)\\s+"
    r"logit=(?P<logit>[-+0-9.eE]+)\\s*$"
)


def parse(path: Path):
    steps = {}
    current = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = DECODED_RE.match(line)
        if m:
            current = int(m.group("step"))
            steps[current] = {
                "source": m.group("source"),
                "count": int(m.group("count")),
                "zero": int(m.group("zero")),
                "nonfinite": int(m.group("nonfinite")),
                "tops": {},
            }
            continue
        m = TOP_RE.match(line)
        if m and current is not None:
            steps[current]["tops"][int(m.group("rank"))] = (
                int(m.group("token")),
                float(m.group("logit")),
            )
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path, help="A: C2P run log")
    p.add_argument("b", type=Path, help="B: known-correct baseline log")
    p.add_argument("--logit-atol", type=float, default=1e-3)
    p.add_argument("--show", type=int, default=8,
                   help="number of leading steps to print even when equal")
    args = p.parse_args()

    a = parse(args.a)
    b = parse(args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("No shared decoded=N TOKEN LOGITS blocks found.")

    first_token_diff = None
    first_logit_drift = None
    first_health_diff = None

    for step in shared:
        ah = a[step]
        bh = b[step]
        at = ah["tops"].get(0)
        bt = bh["tops"].get(0)
        health_same = (
            ah["zero"] == bh["zero"] and
            ah["nonfinite"] == bh["nonfinite"] and
            ah["count"] == bh["count"]
        )
        if not health_same and first_health_diff is None:
            first_health_diff = step

        if at is None or bt is None:
            status = "MISSING_TOP1"
        elif at[0] != bt[0]:
            status = "TOKEN_DIFF"
            if first_token_diff is None:
                first_token_diff = step
        else:
            delta = abs(at[1] - bt[1])
            if delta > args.logit_atol:
                status = "LOGIT_DRIFT"
                if first_logit_drift is None:
                    first_logit_drift = step
            else:
                status = "MATCH"

        if step < args.show or status != "MATCH":
            a_text = "none" if at is None else f"{at[0]}@{at[1]:.6f}"
            b_text = "none" if bt is None else f"{bt[0]}@{bt[1]:.6f}"
            print(
                f"decoded={step:>3} {status:<12} "
                f"A={a_text:<22} B={b_text:<22} "
                f"A(zero/nonfinite)={ah['zero']}/{ah['nonfinite']} "
                f"B={bh['zero']}/{bh['nonfinite']}"
            )

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print()
    print(f"shared_steps={len(shared)} only_A={len(only_a)} only_B={len(only_b)}")
    if first_token_diff is not None:
        print(f"FIRST TOP1 TOKEN DIVERGENCE: decoded={first_token_diff}")
    else:
        print("No Top-1 token divergence in shared decoded steps.")
    if first_logit_drift is not None:
        print(
            "FIRST TOP1 LOGIT DRIFT: "
            f"decoded={first_logit_drift} atol={args.logit_atol:g}"
        )
    else:
        print("No Top-1 logit drift beyond tolerance before token divergence.")
    if first_health_diff is not None:
        print(f"FIRST LOGIT HEALTH DIFFERENCE: decoded={first_health_diff}")

    return 1 if first_token_diff is not None else 0


if __name__ == "__main__":
    sys.exit(main())
