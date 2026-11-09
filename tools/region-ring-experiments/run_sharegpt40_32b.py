#!/usr/bin/env python3
"""Qwen3-32B ShareGPT40 prefill-only Region Ring sweep.

Fixed 64-layer layout by default:
    G8,P1,G9,P1,G8,P1,G9,P1,G8,P1,G8,P1,G8

GPU Bank A peak: G8; Bank B peak: G9. Uses the same tested ShareGPT40
runner as the 14B script: real GGUF token counts, minimum safe 256-aligned
Context per prompt, 1 generated token, fixed 128-token Chunks, no Profile,
independent XC/XG/XP and per-prompt overrides, resume and archived reruns.

For 32B the default is memory-mapped GGUF weights, not --no-mmap: eagerly
loading the full model into host RAM can exceed the PC's physical memory.
Pass --no-mmap explicitly if testing that tradeoff. Phone RPC must have
the correct Qwen3-32B model and HTP Region configuration available.
"""
from __future__ import annotations

import sys

from run_sharegpt40_14b import main as shared_main

LAYOUT = "G8,P1,G9,P1,G8,P1,G9,P1,G8,P1,G8,P1,G8"
LAYERS = 64


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    for flag in ("--model-layers", "--profile-chunks", "--update-profile"):
        if any(arg == flag or arg.startswith(flag + "=") for arg in user_args):
            raise SystemExit(
                f"{flag} is unavailable in the 32B fixed-layout sweep: "
                "model depth is always 64 and Chunk Profile is skipped; "
                "adjust --xc/--xg/--xp or --chunk-overrides instead.")
    return shared_main(
        user_args,
        description=__doc__,
        defaults={"layout": LAYOUT, "model_layers": LAYERS, "mmap": True},
    )


if __name__ == "__main__":
    raise SystemExit(main())
