#!/usr/bin/env python3
"""Preflight missing exact-length calibration prompts without changing a Profile's identity.

Use the existing tokenizer, cached inputs, hardware manifest and b/grid.
The original generator is preferred; deterministic alternative text is
used only if that generator cannot reach the requested length. No RPC,
measurement, configuration, or completed-sample checkpoint is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import offline_profile as offline


def alternative_prompt(count, total):
    """Bounded search; accept ONLY an exact chat-tokenizer count."""
    measured = {}
    def measure(text):
        if text not in measured:
            measured[text] = count(text)
        return measured[text]

    # Changing lexical units changes BPE/whitespace boundary behavior.
    units = ("hello", "cat", "test", "a", "one", "x", "0",
             "The device works.", "Data and memory.")
    suffixes = (".", "!", "?", ",", " x", " a", " 1",
                " end", " final.", " a b", " 1 2")
    tried_suffix = set()
    for unit in units:
        n = max(1, total - 8)
        tried_n = set()
        near = []
        for _ in range(12):
            if n in tried_n or not 1 <= n <= max(32, 2 * total + 32):
                break
            tried_n.add(n)
            text = " ".join([unit] * n)
            actual = measure(text)
            if actual == total:
                return text
            near.append((abs(total - actual), actual, n, text))
            # Estimate the number of units needed for the remaining tokens.
            slope = max(1.0, (actual - 8) / n)
            delta = total - actual
            step = max(1, round(abs(delta) / slope))
            n += step if delta > 0 else -step

        # A repeated word may only reach even (or odd) lengths. Adjust
        # a small suffix or the neighbouring repeat counts instead.
        for _, _, near_n, _ in sorted(near)[:2]:
            for k in (near_n, near_n - 1, near_n + 1, near_n - 2):
                if k < 1:
                    continue
                prefix = " ".join([unit] * k)
                actual = measure(prefix)
                if actual == total:
                    return prefix
                if abs(actual - total) > 8:
                    continue
                for tail in suffixes:
                    text = prefix + tail
                    if text in tried_suffix:
                        continue
                    tried_suffix.add(text)
                    if measure(text) == total:
                        return text
    raise ValueError(
        f"could not construct exact {total}-token input after "
        f"{len(measured)} alternative tokenizer checks")


def prepare_one(args, root, total):
    path = root / "prompts" / f"synthetic-{total}.txt"
    meta = path.with_suffix(".json")
    if meta.is_file():
        # Preserve existing inputs and their original fingerprints.
        offline.exact_prompt(args, root, total)
        print(f"[PROMPT_PREFLIGHT] tokens={total} status=REUSE", flush=True)
        return
    def count(text):
        return offline.count_model_tokens(
            Path(args.tokenizer), Path(args.model), text)
    if path.exists():
        # A prior interruption could leave a text file without metadata.
        # Never silently replace it.
        text = path.read_text(encoding="utf-8")
        if count(text) != total:
            raise ValueError(f"unverified existing prompt: {path}")
        method = "RECOVER_EXACT_TEXT"
    else:
        try:
            offline.exact_prompt(args, root, total)
            return
        except ValueError as exc:
            expected = f"could not build an exact {total}-token chat calibration input"
            if str(exc) != expected:
                raise
        print(f"[PROMPT_PREFLIGHT] tokens={total} status=ALTERNATIVE_SEARCH",
              flush=True)
        text = alternative_prompt(count, total)
        method = "ALTERNATIVE"
    if count(text) != total:
        raise ValueError(f"final chat-tokenizer verification failed: {total}")
    offline.atomic_text(path, text)
    offline.write_json(meta, {"tokens": total, "sha256": offline.digest_file(path)})
    print(f"[PROMPT_PREFLIGHT] tokens={total} method={method} status=EXACT",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--tokens", nargs="+", type=int,
                        help="only preflight specified lengths; default: all remaining grid and holdout lengths")
    opts = parser.parse_args()
    root = opts.profile_dir.resolve()
    manifest = offline.read_json(root / "hardware-config.json")
    cfg = manifest["config"]
    if cfg.get("synthetic_input") == "RAW_PREFIX":
        raise ValueError("RAW_PREFIX profile constructs exact inputs during resident calibration; no preflight needed")
    if offline.digest_config(cfg) != manifest["signature"]:
        raise ValueError("invalid hardware-config.json signature")
    if (root / "profile-ready.json").exists():
        raise ValueError("Profile already READY; no repair required")
    for key in ("model", "tokenizer"):
        if offline.file_id(cfg[key]["path"]) != cfg[key]:
            raise ValueError(f"{key} identity differs from the existing Profile")
    code = Path(__file__).resolve().parent
    for name, sha in cfg["source_sha256"].items():
        if offline.digest_file(code / name) != sha:
            raise ValueError(f"Profile source fingerprint differs: {name}")
    args = SimpleNamespace(model=cfg["model"]["path"],
                           tokenizer=cfg["tokenizer"]["path"])
    if opts.tokens:
        totals = sorted(set(opts.tokens))
    else:
        b = int(offline.read_json(root / "min-chunk.json")["b"])
        maximum = int(cfg["max_chunk"])
        totals = sorted({sum(cuts) for cuts, _ in offline.training_jobs(b, maximum)}
                        | {x + y for x, y in offline.holdout_points(b, maximum)})
    if not totals or any(t < 2 or t > offline.CONTEXT for t in totals):
        raise ValueError("invalid calibration token lengths")
    print(f"[PROMPT_PREFLIGHT] profile={root} lengths={totals}", flush=True)
    for total in totals:
        prepare_one(args, root, total)
    print("[PROMPT_PREFLIGHT] status=PASS; resume original experiment", flush=True)


if __name__ == "__main__":
    main()
