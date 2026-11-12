#!/usr/bin/env python3
"""Model-tokenized, per-prompt Jupiter context selection for ShareGPT sweeps."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

MIN_CONTEXT = 1024
CONTEXT_PADDING = 64
CONTEXT_ALIGN = 256
N_PREDICT = 1


def context_for_tokens(token_count: int, predict: int = N_PREDICT,
                       minimum: int = MIN_CONTEXT,
                       padding: int = CONTEXT_PADDING,
                       fixed: int | None = None) -> int:
    """Match the Region Ring 14B policy: 256-aligned KV with a safety margin."""
    if token_count <= 0 or predict < 1 or minimum <= 0 or padding < 0:
        raise ValueError("invalid token count or context parameters")
    required = token_count + predict + padding
    if fixed is not None:
        if fixed < required:
            raise ValueError(
                f"fixed context {fixed} cannot fit {token_count} prompt + "
                f"{predict} predicted + {padding} safety tokens "
                f"(requires >= {required})"
            )
        return fixed
    return max(minimum, ((required + CONTEXT_ALIGN - 1) // CONTEXT_ALIGN)
               * CONTEXT_ALIGN)


def tokenizer_path(exe: str | Path, override: str | Path | None = None) -> Path:
    """Find llama-tokenize next to llama-cli, unless explicitly overridden."""
    if override is not None:
        return Path(override).expanduser().resolve()
    binary = "llama-tokenize.exe" if os.name == "nt" else "llama-tokenize"
    return Path(exe).expanduser().resolve().with_name(binary)


def model_token_counts(model: str | Path, tokenizer: str | Path,
                       prompts: list[Path], cache_file: Path | None = None,
                       persist: bool = True) -> dict[str, int]:
    """Tokenize with the selected GGUF and cache by model/binary/prompt hash.

    The llama-cli prefill probe remains authoritative for the actual DP batch.
    This count is for choosing a safe context *before* loading the model.
    """
    model = Path(model).expanduser().resolve()
    tokenizer = Path(tokenizer).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    if not tokenizer.is_file():
        raise FileNotFoundError(
            f"GGUF tokenizer not found: {tokenizer}. Build llama-tokenize, "
            "pass --tokenizer-exe, or use --context for a fixed context."
        )
    model_stat, tokenizer_stat = model.stat(), tokenizer.stat()
    signature = {
        "model": str(model), "model_bytes": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "tokenizer": str(tokenizer), "tokenizer_bytes": tokenizer_stat.st_size,
        "tokenizer_mtime_ns": tokenizer_stat.st_mtime_ns,
        "flags": ["--ids", "--show-count", "--no-escape", "--log-disable"],
    }
    cache: dict = {}
    if cache_file is not None and cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    items = cache.get("prompts", {}) if cache.get("signature") == signature else {}
    if not isinstance(items, dict):
        items = {}
    counts: dict[str, int] = {}
    for prompt in prompts:
        prompt = Path(prompt).resolve()
        digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
        key = prompt.name.lower()
        previous = items.get(key, {})
        count = (previous.get("tokens")
                 if isinstance(previous, dict)
                 and previous.get("prompt_sha256") == digest else None)
        if type(count) is not int or count <= 0:
            command = [
                str(tokenizer), "-m", str(model), "-f", str(prompt),
                "--ids", "--show-count", "--no-escape", "--log-disable",
            ]
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"llama-tokenize failed for {prompt.name} "
                    f"(code {result.returncode}): {result.stderr[-700:]}"
                )
            matches = re.findall(
                r"Total number of tokens:\s*(\d+)", result.stdout,
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"llama-tokenize missing unique count for "
                    f"{prompt.name}: {result.stdout[-300:]}"
                )
            count = int(matches[0])
            if count <= 0:
                raise ValueError(
                    f"llama-tokenize reported {count} tokens for {prompt.name}"
                )
            print(f"[JUPITER_TOKENS] prompt={prompt.name} model={count}",
                  flush=True)
        counts[key] = count
        items[key] = {"prompt_sha256": digest, "tokens": count}
        if persist and cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temp = cache_file.with_name(cache_file.name + ".tmp")
            temp.write_text(
                json.dumps({"signature": signature, "prompts": items},
                           indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, cache_file)
    return counts
