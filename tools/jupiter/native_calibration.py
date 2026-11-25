#!/usr/bin/env python3
"""Strict calibration trace for the Phone-owned autonomous execution path.

The Phone reports completed local service times once, with the final P2C
hidden state. Do not time PC waiting as CPU compute or synthesize RPC stages.
"""
from __future__ import annotations

import math
import os
import re

ENV = "LLAMA_JUPITER_RESIDENT_NATIVE_CALIBRATION"
BACKEND = "native-autonomous"
PHONE = "PHONE_NATIVE"
FIELDS = ("x", "y", "latency_ms", "device", "input_ms", "compute_ms", "wall_ms")
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
UBATCH = re.compile(
    r"\[JUPITER_UBATCH\] event=(BEGIN|QUEUED) chunk=(\d+) slot=\d+ "
    r"prefix_tokens=(\d+) tokens=(\d+)")
PREPARE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\] phase=PREPARE_BEGIN run=(\d+) "
    r"tokens=(\d+) chunk=\d+ sessions=(\d+) ctx=\d+ first_layer=\d+ layers=(\d+)")
COMPLETE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\] phase=COMPLETE tokens=(\d+) "
    r"chunks=(\d+) output=REAL_LOGITS final_head=PC_CUDA status=PASS")
PUSH = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\] phase=PHONE_AUTO_PUSH run=(\d+) "
    r"chunk=(\d+) tokens=(\d+)")
PLAN = re.compile(r"\[JUPITER_SEQUENCE_PLAN\].*?total=(\d+).*?chunks=([\d,]+)")
PHONE_RECORD = re.compile(
    r"\[JUPITER_NATIVE_PHONE_PROFILE\] run=(\d+) chunk=(\d+) prefix=(\d+) "
    r"tokens=(\d+) layers=(\d+) sessions=(\d+) upload_ms=" + NUMBER +
    r" htp_ms=" + NUMBER + r" copy_ms=" + NUMBER + r" wall_ms=" + NUMBER)
BRIDGE = re.compile(
    r"\[JUPITER_NATIVE_BRIDGE_PROFILE\] chunk=(\d+) tokens=(\d+) bridge_ms=" + NUMBER)
STAGE = re.compile(
    r"\[JUPITER_STAGE_PROFILE\].*?backend=(\S+).*?input_ms=" + NUMBER +
    r" compute_ms=" + NUMBER + r" wall_ms=" + NUMBER + r" native_bridge=(\d+)")
MARKER = re.compile(r"\[JUPITER_NATIVE_CALIBRATION\] mode=PHONE_AUTONOMOUS profile=([01])")


def active(env=None):
    return (os.environ if env is None else env).get(ENV) == "1"


def is_phone(stage):
    return stage == PHONE or stage.startswith("RPC")


def validate_run(text, expected_tokens=None, expected_cuts=None):
    """Validate ALL chunks, local service coverage and no PC Graph fallback."""
    markers, prepares, completes, plans = (list(p.finditer(text)) for p in
                                           (MARKER, PREPARE, COMPLETE, PLAN))
    if any(len(items) != 1 for items in (markers, prepares, completes, plans)):
        raise ValueError("native calibration missing unique mode/prepare/plan/completion")
    if re.search(r"\[RPC_GRAPH\]|cmd=GRAPH_(?:COMPUTE|RECOMPUTE)", text):
        raise ValueError("PC-authored RPC Graph appeared in native calibration")
    run, total, sessions, layers = map(int, prepares[0].groups())
    cuts = list(map(int, plans[0][2].split(",")))
    if (int(plans[0][1]) != total or sum(cuts) != total or
            not 1 <= len(cuts) <= 16 or min(cuts) <= 0 or
            tuple(map(int, completes[0].groups())) != (total, len(cuts))):
        raise ValueError("native calibration plan/completion mismatch")
    if expected_tokens is not None and total != expected_tokens:
        raise ValueError("native calibration token count mismatch")
    if expected_cuts and cuts != list(expected_cuts):
        raise ValueError("native calibration exact cuts mismatch")
    if not 1 <= sessions <= 4 or layers <= 0:
        raise ValueError("invalid native prepare geometry")
    expected = []
    prefix = 0
    for chunk, tokens in enumerate(cuts):
        expected.append((chunk, prefix, tokens))
        prefix += tokens
    for phase in ("BEGIN", "QUEUED"):
        seen = [tuple(map(int, m.groups()[1:])) for m in UBATCH.finditer(text)
                if m[1] == phase]
        if seen != expected:
            raise ValueError(f"native calibration incomplete {phase} sequence")
    if (text.count("phase=PHONE_AUTO_START") != 1 or
            text.count("phase=PHONE_AUTO_P2C_DONE") != 1 or
            [tuple(map(int, m.groups())) for m in PUSH.finditer(text)] !=
                [(run, chunk, tokens) for chunk, _, tokens in expected]):
        raise ValueError("native autonomous PUSH/P2C sequence mismatch")
    records = list(PHONE_RECORD.finditer(text))
    if len(records) != len(cuts):
        raise ValueError("native Phone service footer missing Chunk records")
    phone = {}
    for m, (chunk, prefix, tokens) in zip(records, expected):
        observed = tuple(map(int, m.groups()[:6]))
        if observed != (run, chunk, prefix, tokens, layers, min(sessions, layers)):
            raise ValueError("native Phone service geometry mismatch")
        upload, htp, copy, wall = map(float, m.groups()[6:])
        service = upload + htp + copy
        if (not all(math.isfinite(v) and v >= 0 for v in (upload, htp, copy, wall))
                or htp <= 0 or wall <= 0 or service > wall + 0.02):
            raise ValueError("invalid native Phone-local service time")
        phone[chunk] = (tokens, prefix, service, wall)
    return {"run": run, "tokens": total, "cuts": cuts, "phone": phone,
            "profile": markers[0][1] == "1"}


def profile_rows(text, metric="compute_ms", min_tokens=1, require_cpu=False,
                 pc_prefix_only=False):
    """Optionally keep PC prefix service only, excluding the post-bridge Head.

    The tagged CPU split may include local staging/housekeeping alongside
    the bridge; subtract the measured callback, retain its local residual.
    All later splits belong to the post-bridge output path, not the prefix.
    """
    data = validate_run(text)
    if not data["profile"]:
        raise ValueError("native q extraction requires synchronized stage profiling")
    current = None
    stages = {}
    bridge_ms = None
    bridge_count = 0
    rows = []
    completed = []
    for line in text.splitlines():
        m = UBATCH.search(line)
        if m and m[1] == "BEGIN":
            if current is not None:
                raise ValueError("unfinished native PC Chunk")
            current = tuple(map(int, m.groups()[1:]))
            stages, bridge_ms, bridge_count = {}, None, 0
            continue
        if current is None:
            continue
        chunk, prefix, tokens = current
        m = BRIDGE.search(line)
        if m:
            if tuple(map(int, m.groups()[:2])) != (chunk, tokens) or bridge_ms is not None:
                raise ValueError("native bridge timer does not match Chunk")
            bridge_ms = float(m[3])
            continue
        m = STAGE.search(line)
        if "[JUPITER_STAGE_PROFILE]" in line and not m:
            raise ValueError("malformed native PC stage record")
        if m:
            device = m[1]
            if not (device.startswith("CPU") or device.startswith("CUDA")):
                raise ValueError("unexpected PC backend in native calibration: " + device)
            inp, compute, wall = map(float, m.groups()[1:4])
            if (not all(math.isfinite(v) and v >= 0 for v in (inp, compute, wall))
                    or inp + compute > wall + 0.02):
                raise ValueError("invalid native PC stage duration")
            tagged = int(m[5])
            if tagged:
                if tagged != 1 or not device.startswith("CPU") or bridge_ms is None or bridge_count:
                    raise ValueError("native CPU bridge split attribution mismatch")
                if not math.isfinite(bridge_ms) or bridge_ms < 0 or bridge_ms > compute + 0.02:
                    raise ValueError("native bridge duration exceeds tagged CPU split")
                compute = max(0.0, compute - bridge_ms)
                wall = max(0.0, wall - bridge_ms)
                bridge_count += 1
            elif pc_prefix_only and bridge_count:
                continue
            entry = stages.setdefault(device, [0.0, 0.0, 0.0])
            for i, value in enumerate((inp, compute, wall)):
                entry[i] += value
            continue
        m = UBATCH.search(line)
        if m and m[1] == "QUEUED":
            if (tuple(map(int, m.groups()[1:])) != current or bridge_count != 1 or
                    not any(d.startswith("CUDA") and v[1] > 0 for d, v in stages.items()) or
                    (require_cpu and not any(d.startswith("CPU") and v[1] > 0 for d, v in stages.items()))):
                raise ValueError("incomplete native PC accelerator/bridge stage coverage")
            phone_tokens, phone_prefix, service, phone_wall = data["phone"][chunk]
            if (phone_tokens, phone_prefix) != (tokens, prefix):
                raise ValueError("native Phone/PC Chunk mismatch")
            # The Phone is ONE serial superstage. Its compute_ms is local
            # service (all layer HTP compute + local upload/read), not raw
            # DSP kernels, TCP time, or the PC custom-op wait.
            stages[PHONE] = [0.0, service, phone_wall]
            if tokens >= min_tokens:
                for device, values in sorted(stages.items()):
                    row = dict(x=tokens, y=prefix, device=device,
                               input_ms=values[0], compute_ms=values[1], wall_ms=values[2])
                    row["latency_ms"] = row[metric]
                    rows.append(row)
            completed.append(chunk)
            current = None
    if current is not None or completed != list(range(len(data["cuts"]))) or not rows:
        raise ValueError("native PC profiling did not finish all Chunks")
    return rows
