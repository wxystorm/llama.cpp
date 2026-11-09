#!/usr/bin/env python3
"""Measured fixed-layout G/P Chunk-only planning, no layer-placement search.

The calibration is a set of real fixed-layout Ring runs at distinct Chunk
widths. The event DAG models per-region FIFO, interval dependencies, GPU/Phone
lanes and the two asynchronous GPU Bank refills. Predictions are estimates,
not substitutes for actual prefill_ms measurements.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import heapq
import json
import math
from pathlib import Path
import re
from statistics import median

LAYOUT = "G9,P1,G9,P1,G9,P1,G10"
REGIONS = (('G', 9), ('P', 1), ('G', 9), ('P', 1),
           ('G', 9), ('P', 1), ('G', 10))
GRID = (64, 96, 128)
TAIL_FRACTION = 0.25
REFILLS = (4, 6)


def effective_width(n_tokens: int, x: int) -> int:
    """Use a slightly wider *uniform* width to remove <=25% tiny tail.

    This preserves Phone-owned Plan v2/v3's implicit chunk index/offset ABI.
    Example: N=513, X=128 => Xeff=129 => 129,129,129,126.
    The number of chunks is the same as merging the 1-token tail backward.
    """
    if n_tokens <= 0 or x <= 0:
        raise ValueError('positive token count and chunk width required')
    complete, tail = divmod(n_tokens, x)
    threshold = x // 4
    if complete == 0 or tail == 0 or tail > threshold:
        return x
    new_x = (n_tokens + complete - 1) // complete
    return new_x if new_x <= x + threshold else x


def intervals(n_tokens: int, x: int) -> list[tuple[int, int]]:
    width = effective_width(n_tokens, x)
    return [(start, min(start + width, n_tokens))
            for start in range(0, n_tokens, width)]


def as_number(v) -> float | None:
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def duration(row: dict, before: str, after: str) -> float | None:
    a, b = as_number(row.get(before)), as_number(row.get(after))
    if a is not None and b is not None and b > a:
        return b - a
    return None


def load_trace(folder: Path) -> list[dict]:
    path = folder / 'metrics.jsonl'
    if not path.is_file():
        return []
    events = []
    for line in path.read_text('utf-8', errors='replace').splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                events.append(item)
        except ValueError:
            continue
    offsets = [i for i, item in enumerate(events)
               if item.get('type') == 'measurement_start']
    return events[offsets[-1] + 1:] if offsets else events


def extract_probe(folder: Path, x: int) -> dict:
    result = json.loads((folder / 'result.json').read_text('utf-8'))
    if result.get('status') != 'PASS':
        raise ValueError(f'probe not PASS: {folder}')
    events = load_trace(folder)
    trace = [row for row in events if row.get('type') == 'region_ring_chunk_trace']
    if not trace:
        raise ValueError(f'probe has no Chunk traces: {folder}')
    samples: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    pushes: dict[int, list[float]] = defaultdict(list)
    for row in trace:
        dev = row.get('device')
        layers = int(row.get('layer_end', 0)) - int(row.get('layer_begin', 0))
        tokens = int(row.get('tokens', 0) or 0)
        if dev not in ('GPU', 'PHONE') or layers <= 0 or tokens <= 0:
            continue
        if dev == 'GPU':
            elapsed = duration(row, 'host_execute_begin_ms', 'host_execute_end_ms')
            if elapsed is None:
                elapsed = duration(row, 'compute_range_begin_ms', 'compute_range_end_ms')
        else:
            complete = as_number(row.get('complete_ms'))
            claim = as_number(row.get('claim_ms'))
            recv = as_number(row.get('push_end_ms'))
            start = max(x for x in (claim, recv) if x is not None) if (claim is not None or recv is not None) else None
            elapsed = complete - start if start is not None and complete is not None else None
            push = duration(row, 'push_begin_ms', 'push_end_ms')
            if push is not None:
                pushes[tokens].append(push)
        if elapsed is not None and elapsed > 0:
            samples[(dev, layers, tokens)].append(elapsed)
    if not any(k[0] == 'GPU' for k in samples) or not any(k[0] == 'PHONE' for k in samples):
        raise ValueError(f'probe lacks measured GPU/Phone stage times: {folder}')
    if not pushes:
        raise ValueError(f'probe has no measured Phone PUSH timings: {folder}')
    bank = {}
    text = (folder / 'engine.stderr.log').read_text('utf-8', errors='replace')
    for m in re.finditer(r'\[REGION_RING_GPU_BANK_MOVER\] event=REFILL_DONE ([^\r\n]+)', text):
        data = dict(re.findall(r'(\w+)=([^\s]+)', m.group(1)))
        if data.get('region') in ('4', '6') and 'wall_ms' in data:
            bank[data['region']] = float(data['wall_ms'])
    return {
        'width': x,
        'prefill_tokens': int(result['actual_prefill_tokens']),
        'wall_ms': float(result['prefill_ms']),
        'stages': [dict(device=dev, layers=depth, tokens=tokens,
                        ms=round(median(times), 4), samples=len(times))
                   for (dev, depth, tokens), times in sorted(samples.items())],
        'push': [dict(tokens=tokens, ms=round(median(times), 4), samples=len(times))
                 for tokens, times in sorted(pushes.items())],
        'bank_refill_ms': bank,
        'probe_dir': str(folder),
    }


def median_interpolate(points: list[tuple[int, float]], tokens: int) -> float:
    if not points:
        raise ValueError('missing measured profile anchor')
    grouped: dict[int, list[float]] = defaultdict(list)
    for size, elapsed in points:
        if size > 0 and elapsed > 0:
            grouped[size].append(elapsed)
    if not grouped:
        raise ValueError('no positive measured costs')
    values = sorted((size, median(v)) for size, v in grouped.items())
    if len(values) == 1:
        size, ms = values[0]
        return max(0.05, ms * tokens / size)
    if tokens <= values[0][0]:
        (x0, y0), (x1, y1) = values[:2]
    elif tokens >= values[-1][0]:
        (x0, y0), (x1, y1) = values[-2:]
    else:
        (x0, y0), (x1, y1) = next(
            (a, b) for a, b in zip(values, values[1:])
            if a[0] <= tokens <= b[0])
    predicted = y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return max(0.05, predicted)


def service(profile: dict, device: str, layers: int, tokens: int) -> float:
    points = [(int(p['tokens']), float(p['ms']))
              for probe in profile['probes']
              for p in probe['stages']
              if p['device'] == device and int(p['layers']) == layers]
    if not points:
        raise ValueError(f'no {device} {layers}-layer measured profile')
    return median_interpolate(points, tokens)


def push_time(profile: dict, tokens: int) -> float:
    points = [(int(row['tokens']), float(row['ms']))
              for probe in profile['probes'] for row in probe['push']]
    return median_interpolate(points, tokens) if points else 0.0


def dag_simulate(profile: dict, tokens: int, xg: int, xp: int) -> dict:
    """Event DAG: interval overlap, stage-order KV, G/P lanes, two Bank refills."""
    if tokens <= 0:
        raise ValueError('positive token count required')
    all_intervals = [intervals(tokens, xg if dev == 'G' else xp)
                     for dev, _ in REGIONS]
    tasks: list[dict] = []
    by_region: list[list[int]] = []
    for region, ((dev, layers), chunks) in enumerate(zip(REGIONS, all_intervals)):
        indices = []
        for chunk, (begin, end) in enumerate(chunks):
            idx = len(tasks)
            indices.append(idx)
            tasks.append(dict(id=idx, region=region, chunk=chunk,
                              dev=dev, layers=layers, begin=begin, end=end,
                              cost=service(profile, 'GPU' if dev == 'G' else 'PHONE',
                                           layers, end - begin)))
        by_region.append(indices)
    # Every task requires its preceding same-region task, plus every producer
    # interval that overlaps its own token range. P input PUSH is charged once
    # after all upstream overlapping producer intervals are complete.
    downstream = [[] for _ in tasks]
    remaining = [0] * len(tasks)
    same_end = [0.0] * len(tasks)
    input_end = [0.0] * len(tasks)
    for region, items in enumerate(by_region):
        for i, tid in enumerate(items):
            if i:
                previous = items[i - 1]
                downstream[previous].append((tid, 'same'))
                remaining[tid] += 1
            if region:
                dest = tasks[tid]
                for src in by_region[region - 1]:
                    prod = tasks[src]
                    if min(prod['end'], dest['end']) > max(prod['begin'], dest['begin']):
                        downstream[src].append((tid, 'input'))
                        remaining[tid] += 1
    ready = {0}
    lane_free = {'G': 0.0, 'P': 0.0}
    running = []
    bank_ready = {0: 0.0, 2: 0.0}
    bank_jobs = []
    mover_end = 0.0
    t = 0.0
    finished = 0
    end_times = [0.0] * len(tasks)
    while finished < len(tasks):
        # Dispatch each available lane at current time; no unready task may
        # reserve future capacity, otherwise it could block an earlier arrival.
        for lane in ('G', 'P'):
            if lane_free[lane] > t + 1e-8:
                continue
            eligible = []
            for tid in ready:
                task = tasks[tid]
                if task['dev'] != lane:
                    continue
                region = task['region']
                if lane == 'G' and region not in bank_ready:
                    continue
                needed = max(same_end[tid], input_end[tid] +
                             (push_time(profile, task['end'] - task['begin'])
                              if lane == 'P' else 0.0))
                if needed <= t + 1e-8 and (lane != 'G' or bank_ready[region] <= t + 1e-8):
                    eligible.append(tid)
            if eligible:
                chosen = min(eligible, key=lambda tid: (tasks[tid]['region'], tasks[tid]['chunk']))
                ready.remove(chosen)
                job = tasks[chosen]
                end = t + job['cost']
                lane_free[lane] = end
                heapq.heappush(running, (end, 'task', chosen))
        if not running:
            if ready:
                possible = []
                for tid in ready:
                    job = tasks[tid]
                    gate = bank_ready.get(job['region']) if job['dev'] == 'G' else 0.0
                    if gate is not None:
                        possible.append(max(same_end[tid], input_end[tid] +
                                            (push_time(profile, job['end'] - job['begin'])
                                             if job['dev'] == 'P' else 0.0), gate))
                if possible and min(possible) > t:
                    t = min(possible)
                    continue
            raise ValueError('DAG deadlock or missing Bank refill')
        next_event = running[0][0]
        # A Phone PUSH can complete while GPU still computes a later Chunk.
        # Wake at that time instead of incorrectly idling until GPU finishes.
        for tid in ready:
            job = tasks[tid]
            if lane_free[job['dev']] > t + 1e-8:
                continue
            gate = bank_ready.get(job['region']) if job['dev'] == 'G' else 0.0
            if gate is None:
                continue
            needed = max(same_end[tid], input_end[tid] +
                         (push_time(profile, job['end'] - job['begin'])
                          if job['dev'] == 'P' else 0.0), gate)
            if t + 1e-8 < needed < next_event - 1e-8:
                next_event = needed
        if next_event < running[0][0] - 1e-8:
            t = next_event
            continue
        t = running[0][0]
        while running and running[0][0] <= t + 1e-8:
            end, kind, tid = heapq.heappop(running)
            if kind == 'mover':
                bank_ready[tid] = end
                continue
            task = tasks[tid]
            end_times[tid] = end
            finished += 1
            for dst, edge in downstream[tid]:
                remaining[dst] -= 1
                if edge == 'same':
                    same_end[dst] = max(same_end[dst], end)
                else:
                    input_end[dst] = max(input_end[dst], end)
                if remaining[dst] == 0:
                    ready.add(dst)
            # A refill is eligible only when ALL Chunks of its source GPU
            # region have drained. Bank A: G0 -> G4, B: G2 -> G6.
            if task['dev'] == 'G' and task['chunk'] == len(by_region[task['region']]) - 1:
                destination = {0: 4, 2: 6}.get(task['region'])
                if destination is not None:
                    vals = [float(p['bank_refill_ms'].get(str(destination), 0))
                            for p in profile['probes']
                            if str(destination) in p.get('bank_refill_ms', {})]
                    if not vals:
                        raise ValueError('missing measured GPU Bank refill timing')
                    cost = median(vals)
                    begin = max(end, mover_end)
                    mover_end = begin + cost
                    heapq.heappush(running, (mover_end, 'mover', destination))
                    bank_jobs.append(dict(region=destination, begin_ms=round(begin, 3),
                                          end_ms=round(mover_end, 3)))
    # This is only the compute/transfer DAG, not startup model load or PC Head.
    return dict(predicted_dag_ms=round(t, 3), xg=xg, xp=xp,
                effective_xg=effective_width(tokens, xg),
                effective_xp=effective_width(tokens, xp),
                tasks=len(tasks), bank_jobs=bank_jobs)


def choose(profile: dict, tokens: int, candidates: tuple[int, ...] = GRID) -> dict:
    options = []
    for xg in candidates:
        for xp in candidates:
            try:
                options.append(dag_simulate(profile, tokens, xg, xp))
            except ValueError:
                continue
    if not options:
        raise ValueError('no valid G/P chunk candidates; check profile timings')
    options.sort(key=lambda item: (item['predicted_dag_ms'],
                                   abs(item['xg'] - item['xp']),
                                   -item['xg'], -item['xp']))
    selected = dict(options[0])
    selected['alternatives'] = options[1:4]
    selected['plan_tokens'] = tokens
    selected['layout_fixed'] = LAYOUT
    selected['tail_policy'] = 'widen_uniform_if_tail_le_quarter'
    selected['model'] = 'measured_event_dag'
    return selected


def manifest_tokens(manifest: Path, name: str) -> int | None:
    if not manifest.is_file():
        return None
    with manifest.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if any(str(row.get(key, '')).strip().lower().endswith(name.lower())
                   for key in ('prompt_file', 'file', 'filename', 'name', 'output_file')):
                for key in ('actual_tokens', 'token_count', 'n_tokens',
                            'tokens', 'length_tokens', 'prompt_tokens'):
                    try:
                        val = int(row.get(key, ''))
                        if val > 0:
                            return val
                    except (TypeError, ValueError):
                        continue
    return None
