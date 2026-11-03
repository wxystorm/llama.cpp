#!/usr/bin/env python3
"""Compare measured Jupiter and Region Ring Prefill results without rerunning.

Inputs:
- Jupiter Stage-5 suite-results.csv and case_dir/configuration.json files.
- Region Ring run_prompt_sweep.py summary.csv (the exact run being compared).

Never join solely on a prompt_N filename. Verify original file SHA256,
actual tokenizer/Pefill token count, and GGUF filename+bytes first.
Show an indicative latency ratio for matched inputs; label stricter
parameter-matched comparisons separately (batch, ubatch, context, threads,
n_predict). CPU/CUDA/Phone vs CUDA/Phone remains an architectural contrast.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
from typing import Any


OUTPUT_FIELDS = [
    "prompt", "status", "strict_comparable", "parameter_differences",
    "jupiter_status", "region_status", "jupiter_tokens", "region_tokens",
    "jupiter_pc_layers", "jupiter_min_chunk", "jupiter_dp_chunks",
    "region_layout", "region_planner", "region_commit",
    "region_gpu_layers", "region_cpu_layers", "region_phone_layers",
    "jupiter_default_ms", "jupiter_dp_ms", "region_prefill_ms",
    "jupiter_default_tps", "jupiter_dp_tps", "region_tps",
    "region_vs_jupiter_dp", "region_vs_jupiter_default",
    "jupiter_samples", "region_samples", "jupiter_case_dir",
    "region_run_dir", "notes",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def file_basename(path: str) -> str:
    return PureWindowsPath(path).name.lower()


def parse_number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        v = float(value)
        return v if 0 < v < float("inf") else None
    except (ValueError, TypeError):
        return None


def per_prompt(rows: list[dict[str, str]], field: str, label: str
               ) -> dict[str, dict[str, str]]:
    mapped = {}
    for row in rows:
        name = file_basename(row.get(field, ""))
        if not name:
            raise ValueError(f"{label} CSV contains a row without {field}")
        if name in mapped:
            raise ValueError(
                f"{label} has duplicate prompt {name!r}. Select one actual "
                "experiment run; do not silently cherry-pick the faster row."
            )
        mapped[name] = row
    return mapped


def read_case_config(suite_row: dict[str, str]) -> dict[str, Any]:
    case_dir = suite_row.get("case_dir", "")
    if not case_dir:
        return {}
    file = Path(case_dir) / "configuration.json"
    if not file.is_file():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def same_model(cfg: dict, ring: dict[str, str]) -> bool:
    model = cfg.get("model") or {}
    if not isinstance(model, dict):
        return False
    j_name = file_basename(str(model.get("path") or ""))
    r_name = file_basename(ring.get("model", ""))
    j_size = str(model.get("bytes") or "")
    r_size = str(ring.get("model_size_bytes") or "")
    return bool(j_name and r_name and j_name == r_name and
                j_size and r_size and j_size == r_size)


def parameter_differences(cfg: dict, region: dict[str, str]) -> list[str]:
    # Fields with common meanings only. Hardware/partition is deliberately
    # different and is recorded as a method distinction, not a config error.
    fields = ("batch", "ubatch", "context", "threads")
    differences = []
    for key in fields:
        left = cfg.get(key)
        right = region.get(key)
        if left is None or right in (None, ""):
            differences.append(f"{key}:UNKNOWN")
        elif str(left) != str(right):
            differences.append(f"{key}:{left}/{right}")
    rr_predict = region.get("n_predict")
    if rr_predict in (None, ""):
        differences.append("n_predict:UNKNOWN")
    elif str(rr_predict) != "1":
        differences.append(f"n_predict:1/{rr_predict}")
    return differences


def compare(jupiter: list[dict[str, str]],
            region: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_j = per_prompt(jupiter, "prompt", "Jupiter")
    by_r = per_prompt(region, "prompt_file", "Region Ring")
    results: list[dict[str, Any]] = []

    def order_key(name: str) -> tuple[int, int, str]:
        import re
        match = re.fullmatch(r"prompt_(\d+)\.txt", name)
        return (0, int(match.group(1)), name) if match else (1, 0, name)

    for prompt in sorted(set(by_j) | set(by_r), key=order_key):
        j = by_j.get(prompt, {})
        rr = by_r.get(prompt, {})
        cfg = read_case_config(j) if j else {}
        row: dict[str, Any] = {
            "prompt": prompt,
            "jupiter_status": j.get("status", "MISSING"),
            "region_status": rr.get("status", "MISSING"),
            "jupiter_tokens": j.get("actual_tokens", ""),
            "region_tokens": rr.get("actual_prefill_tokens", ""),
            "jupiter_pc_layers": j.get("pc_layers", ""),
            "jupiter_min_chunk": j.get("min_chunk", ""),
            "jupiter_dp_chunks": j.get("dp_chunks", ""),
            "region_layout": rr.get("runtime_layout", ""),
            "region_planner": rr.get("planner_mode", ""),
            "region_commit": rr.get("commit", ""),
            "region_gpu_layers": rr.get("gpu_layers", ""),
            "region_cpu_layers": rr.get("cpu_layers", ""),
            "region_phone_layers": rr.get("phone_layers", ""),
            "jupiter_default_ms": j.get("default_median_ms", ""),
            "jupiter_dp_ms": j.get("dp_median_ms", ""),
            "region_prefill_ms": rr.get("prefill_ms", ""),
            "jupiter_default_tps": j.get("default_tok_s", ""),
            "jupiter_dp_tps": j.get("dp_tok_s", ""),
            "region_tps": rr.get("prefill_tps", ""),
            "jupiter_samples": j.get("sample_count", ""),
            "region_samples": 1 if rr else "",
            "jupiter_case_dir": j.get("case_dir", ""),
            "region_run_dir": rr.get("run_dir", ""),
            "strict_comparable": 0,
            "parameter_differences": "",
            "notes": "",
        }
        reason = ""
        if not j:
            reason = "NO_JUPITER_RESULT"
        elif not rr:
            reason = "NO_REGION_RESULT"
        elif j.get("status") != "PASS":
            reason = "JUPITER_NOT_PASS"
        elif rr.get("status") != "PASS":
            reason = "REGION_NOT_PASS"
        elif not cfg:
            reason = "NO_JUPITER_CASE_CONFIG"
        elif not cfg.get("prompt_sha256") or not rr.get("prompt_sha256"):
            reason = "MISSING_PROMPT_HASH"
        elif cfg["prompt_sha256"].lower() != rr["prompt_sha256"].lower():
            reason = "PROMPT_BYTES_DIFFER"
        elif (not j.get("actual_tokens") or
              not rr.get("actual_prefill_tokens") or
              str(j["actual_tokens"]) != str(rr["actual_prefill_tokens"])):
            reason = "TOKEN_COUNT_DIFFERS"
        elif not same_model(cfg, rr):
            reason = "MODEL_FILENAME_OR_SIZE_DIFFERS"
        else:
            jm = parse_number(j.get("dp_median_ms"))
            jdefault = parse_number(j.get("default_median_ms"))
            rm = parse_number(rr.get("prefill_ms"))
            if jm is None or jdefault is None or rm is None:
                reason = "MISSING_PREFILL_MEASUREMENT"
            else:
                diffs = parameter_differences(cfg, rr)
                row["parameter_differences"] = ";".join(diffs)
                row["strict_comparable"] = int(not diffs)
                row["region_vs_jupiter_dp"] = round(jm / rm, 4)
                row["region_vs_jupiter_default"] = round(jdefault / rm, 4)
                reason = ("MATCHED_STRICT_PARAMS" if not diffs
                          else "MATCHED_INPUTS_CONFIG_DIFF")
                row["notes"] = (
                    "System-level architecture comparison; Jupiter uses "
                    "CUDA/Phone layer split, Region Ring uses CPU/CUDA/Phone "
                    "and tensor/region scheduling. One Region Ring run "
                    "is not a stable statistical estimate."
                )
        row["status"] = reason
        results.append(row)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jupiter", type=Path, required=True,
                        help="jupiter-prompt-suite*/suite-results.csv")
    parser.add_argument("--region", type=Path, required=True,
                        help="one Region Ring sweep summary.csv")
    parser.add_argument("--out", type=Path, default=Path("jupiter-vs-ring.csv"))
    args = parser.parse_args()

    results = compare(csv_rows(args.jupiter), csv_rows(args.region))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows([
            {key: row.get(key, "") for key in OUTPUT_FIELDS}
            for row in results
        ])
    for row in results:
        ratio = row.get("region_vs_jupiter_dp")
        txt = f" region_vs_dp={ratio:.4f}x" if ratio else ""
        print(
            f"[JUPITER_RING_COMPARE] prompt={row['prompt']} "
            f"status={row['status']} "
            f"jupiter_tokens={row['jupiter_tokens']} "
            f"region_tokens={row['region_tokens']}{txt}"
        )
    matched = sum(row["status"].startswith("MATCHED_") for row in results)
    strict = sum(row["strict_comparable"] for row in results)
    print(f"[JUPITER_RING_COMPARE_SUM] matched={matched} strict={strict} "
          f"total={len(results)} output={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
