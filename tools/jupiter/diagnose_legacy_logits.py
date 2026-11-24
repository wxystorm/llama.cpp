#!/usr/bin/env python3
"""Strict one-Prompt legacy RPC numeric diagnostic; never a benchmark.

Runs the same CLI and initialization path used by Stage2. Optional forced
graph rebuild and per-layer dumps isolate stale binding vs numerical errors.
"""
from __future__ import annotations

import argparse
import array
import math
import os
from pathlib import Path
import re
import sys

from run_min_chunk_profile import calibration_env, make_cli
from run_q_sweep import infer_runtime_prompt_tokens, stream_process

HASH = re.compile(r"\[JUPITER_LOGITS_HASH\] tokens=(\d+)\b")
STAGE = re.compile(r"\[JUPITER_STAGE_DUMP\] stage=(\S+) part=(\d+) tokens=(\d+) nonfinite=(\d+)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="build-phone-owned/bin/Release/llama-cli.exe")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--pc-layers", type=int, required=True)
    p.add_argument("--cpu-layers", type=int, required=True)
    p.add_argument("--sessions", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--target-tokens", type=int, required=True)
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", type=int, choices=(1, 2), default=1)
    p.add_argument("--chunks", default=None, help="optional exact plan; otherwise natural UBatch split")
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--stage-dump", action="store_true")
    p.add_argument("--detail-layers", default=None,
                   help="comma-separated layer ids; -1 includes final RMSNorm/head; implies stage dump")
    p.add_argument("--audit-copies", action="store_true",
                   help="compare source/destination host-visible F32 bytes for cross-session copies")
    p.add_argument("--force-host-copy", action="store_true",
                   help="diagnostic PC-staged GET/SET for eligible cross-session activations; implies audit")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.detail_layers:
        args.stage_dump = True
    if args.force_host_copy:
        args.audit_copies = True
    if not 0 <= args.cpu_layers < args.pc_layers:
        raise ValueError("require nonempty PC CUDA prefix")
    if not 2 <= args.target_tokens <= min(args.context, args.batch):
        raise ValueError("invalid target tokens")
    if not 1 <= args.max_chunk <= args.batch:
        raise ValueError("invalid max chunk")
    if args.chunks:
        cuts = [int(s) for s in args.chunks.split(",")]
        if sum(cuts) != args.target_tokens or any(not 1 <= x <= args.max_chunk for x in cuts):
            raise ValueError("invalid exact chunk plan")
    for name in (args.exe, args.model, args.prompt_file):
        if not Path(name).is_file():
            raise FileNotFoundError(name)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig").rstrip("\r\n")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    logits_path = out / "logits.f32"
    log_path = out / "legacy.log"
    env = calibration_env(args.pc_layers, args.cpu_layers)
    for key in list(env):
        if key.startswith("LLAMA_JUPITER_RESIDENT_"):
            env.pop(key)
    for key in ("LLAMA_JUPITER_STAGE_DUMP_DIR", "LLAMA_JUPITER_STAGE_DUMP_TOKENS",
                "LLAMA_JUPITER_STAGE_DUMP_DETAIL_LAYERS"):
        env.pop(key, None)
    env.update({
        "LLAMA_JUPITER_PHONE_OWNED_NATIVE": "0",
        "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "0",
        "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "0",
        "LLAMA_JUPITER_HTP_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(args.sessions),
        "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": str(args.context),
        "LLAMA_JUPITER_WINDOW": str(args.window),
        "LLAMA_JUPITER_PROFILE_STAGE": "0",
        "LLAMA_JUPITER_LOCAL_GRAPH_REUSE": "0",
        "LLAMA_JUPITER_RPC_FORCE_GRAPH_BUILD": str(int(args.force_build)),
        "LLAMA_JUPITER_RPC_AUDIT_SESSION_COPY": str(int(args.audit_copies)),
        "LLAMA_JUPITER_RPC_FORCE_HOST_SESSION_COPY": str(int(args.force_host_copy)),
        "LLAMA_JUPITER_LOGITS_HASH": "1",
        "LLAMA_JUPITER_LOGITS_DUMP_FILE": str(logits_path),
    })
    if args.chunks:
        env["LLAMA_JUPITER_SEQUENCE_PLAN"] = args.chunks
    if args.stage_dump:
        env["LLAMA_JUPITER_STAGE_DUMP_DIR"] = str(out / "stages")
        env["LLAMA_JUPITER_STAGE_DUMP_TOKENS"] = str(args.target_tokens)
        if args.detail_layers:
            env["LLAMA_JUPITER_STAGE_DUMP_DETAIL_LAYERS"] = args.detail_layers
    cmd = make_cli(args, prompt)
    cmd[cmd.index("-lv") + 1] = "4"
    print(f"[LEGACY_DIAG] tokens={args.target_tokens} window={args.window} "
          f"ubatch={args.max_chunk} force_build={int(args.force_build)} "
          f"stage_dump={int(args.stage_dump)} audit_copies={int(args.audit_copies)} "
          f"force_host_copy={int(args.force_host_copy)} detail_layers={args.detail_layers} "
          f"log={log_path}", flush=True)
    rc = stream_process(cmd, env, log_path)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for line in log.splitlines():
        if ("[JUPITER_GRAPH_BINDING]" in line or "[JUPITER_LOGITS_HASH]" in line
                or "[JUPITER_RPC_COPY_AUDIT]" in line):
            print(line)
    if args.detail_layers:
        details = [line for line in log.splitlines() if "[JUPITER_OP_DIAG]" in line]
        bad_ops = [line for line in details if re.search(r"\bnonfinite=[1-9]\d*\b", line)]
        print(f"[LEGACY_DIAG_OPS] records={len(details)} nonfinite_records={len(bad_ops)}")
        if bad_ops:
            first = details.index(bad_ops[0])
            print("\n".join(details[max(0, first-8):first+4]))
    if args.stage_dump:
        records = list(STAGE.finditer(log))
        print(f"[LEGACY_DIAG_STAGE] records={len(records)}")
        for m in records:
            print(f"stage={m[1]} part={m[2]} tokens={m[3]} nonfinite={m[4]}")
        bad = next((m for m in records if int(m[4]) > 0), None)
        if bad:
            print(f"[LEGACY_DIAG_FIRST_BAD] stage={bad[1]} part={bad[2]} nonfinite={bad[4]}")
    hashes = list(HASH.finditer(log))
    if rc != 0 or not logits_path.is_file() or not hashes or int(hashes[-1][1]) != args.target_tokens:
        raise RuntimeError(f"no verified full-prompt logits; CLI exit={rc}, see {log_path}")
    actual, _ = infer_runtime_prompt_tokens(log_path)
    if actual != args.target_tokens:
        raise RuntimeError(f"actual prompt tokens {actual} != {args.target_tokens}")
    raw = logits_path.read_bytes()
    if len(raw) == 0 or len(raw) % 4:
        raise RuntimeError("invalid F32 logits file")
    a = array.array("f")
    a.frombytes(raw)
    finite = sum(math.isfinite(v) for v in a)
    nan = sum(math.isnan(v) for v in a)
    inf = sum(math.isinf(v) for v in a)
    zero = sum(v == 0.0 for v in a)
    ok = finite == len(a) and zero != len(a)
    print(f"[LEGACY_DIAG_RESULT] total={len(a)} finite={finite} nan={nan} inf={inf} "
          f"zero={zero} status={'PASS' if ok else 'FAIL'} log={log_path}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[LEGACY_DIAG_ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
