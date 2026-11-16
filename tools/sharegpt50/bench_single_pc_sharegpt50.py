#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone PC-only llama.cpp ShareGPT50 prefill benchmark.

The default fast path keeps one llama-server process alive per GGUF model,
posts each prompt to /completion with cache_prompt=False, and captures real
prompt processing metrics. No additional Python packages are needed.

Fallback: if llama-server is unavailable, use llama-cli/llama-completion
(one process/model load per prompt, much slower).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

GROUPS = (0, 128, 256, 512, 1024)
FIELDS = (
    "timestamp_utc", "key", "mode", "model", "model_path", "prompt",
    "group", "index", "repeat", "batch", "ubatch", "ctx", "threads",
    "threads_batch", "ngl", "status", "prompt_tokens", "prompt_ms",
    "prefill_tps", "cache_n", "predicted_tokens", "truncated",
    "wall_ms", "log_file", "error",
)
MODEL_FILENAME = {
    "14B": "Qwen3-14B-Q4_K_M.gguf",
    "24B": "Mistral-Small-3.2-24B-Q6_K.gguf",
    "32B": "Qwen3-32B-Q4_K_M.gguf",
}
MODEL_PATTERN = {
    "14B": ("qwen3", "14b", "q4_k_m"),
    "24B": ("mistral", "24b", "q6_k"),
    "32B": ("qwen3", "32b", "q4_k_m"),
}


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def fail(message: str):
    raise RuntimeError(message)


def model_signature(path: Path) -> dict:
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def discover_models(args) -> dict[str, Path]:
    overrides = {"14B": args.model_14b, "24B": args.model_24b, "32B": args.model_32b}
    candidates = None
    models = {}
    for name in args.models:
        if overrides[name]:
            path = Path(overrides[name]).expanduser()
            if not path.is_file():
                fail(f"{name} GGUF 不存在: {path}")
            models[name] = path.resolve()
            continue
        root = Path(args.models_root).expanduser()
        if not root.is_dir():
            fail(f"找不到模型目录 {root}。请传 --models-root 或 --model-{name.lower()}。")
        if candidates is None:
            print(f"[SCAN] 查找 GGUF: {root}", flush=True)
            candidates = [p for p in root.rglob("*.gguf") if p.is_file()]
        exact = [p for p in candidates if p.name.lower() == MODEL_FILENAME[name].lower()]
        matches = exact or [p for p in candidates if all(k in p.name.lower() for k in MODEL_PATTERN[name])]
        if not matches:
            fail(f"未找到 {name} (期望 {MODEL_FILENAME[name]})。请传 --model-{name.lower()} 完整路径。")
        if len(matches) != 1:
            show = "\n".join(f"  {x}" for x in matches[:12])
            fail(f"{name} 有 {len(matches)} 个候选，无法确定所用模型；请传 --model-{name.lower()}：\n{show}")
        models[name] = matches[0].resolve()
    return models


def select_exe(args):
    root = Path(args.bin_dir).expanduser()
    ext = ".exe" if os.name == "nt" else ""
    names = {
        "server": ["llama-server" + ext],
        "cli": ["llama-cli" + ext, "llama-completion" + ext],
    }
    preferred = [args.mode] if args.mode != "auto" else ["server", "cli"]
    for mode in preferred:
        for item in names[mode]:
            exe = root / item
            if exe.is_file():
                return mode, exe.resolve()
    fail(f"{root} 找不到 llama-server / llama-cli / llama-completion。"
         " 请用 --bin-dir 指向编译产物的 bin/Release 目录。")


def prompt_list(args):
    items = []
    missing = []
    for group in args.groups:
        for index in range(1, args.per_group + 1):
            p = Path(args.prompt_dir).expanduser() / f"prompt_{group}_{index:02d}.txt"
            if not p.is_file():
                missing.append(str(p))
            else:
                items.append((group, index, p.resolve()))
    if missing:
        fail("Prompt 缺失，请核对目录和命名：\n" + "\n".join(missing[:25]))
    return items


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig" if not existed else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not existed:
            writer.writeheader()
        writer.writerow(item)
        f.flush()


def config_key(args, mode, exe, model_name, model_path, prompt_bytes, group, index, repeat, ngl):
    info = {
        "v": 4,
        "mode": mode,
        "exe": model_signature(exe),
        "model": model_name,
        "gguf": model_signature(model_path),
        "group": group,
        "index": index,
        "repeat": repeat,
        "prompt_sha256": sha256(prompt_bytes),
        "batch": args.batch,
        "ubatch": args.ubatch,
        "ctx": args.ctx,
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "ngl": ngl,
        "cli_verbosity": args.cli_verbosity if mode == "cli" else None,
    }
    return sha256(json.dumps(info, sort_keys=True, ensure_ascii=False).encode("utf-8"))[:24]


def normalize_ngl(value: str) -> str:
    """Accept llama.cpp -ngl all, or an explicit nonnegative GPU layer count."""
    v = str(value).strip().lower()
    if v == "all":
        return "all"
    try:
        num = int(v)
    except ValueError:
        fail(f"GPU offload 层数必须是 all 或非负整数，收到：{value!r}")
    if num < 0:
        fail(f"GPU offload 层数不能小于 0，收到：{value!r}")
    return str(num)


def request_json(method, url, data=None, timeout=10.0):
    raw = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
    return json.loads(payload.decode("utf-8-sig", errors="replace"))


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


class ServerProcess:
    def __init__(self, args, exe, model_path, ngl, log_path):
        self.args = args
        self.exe = exe
        self.model_path = model_path
        self.ngl = ngl
        self.log_path = log_path
        self.process = None
        self.log = None
        self.url = f"http://127.0.0.1:{args.port}"

    def __enter__(self):
        if not port_available(self.args.port):
            fail(f"端口 {self.args.port} 已占用。指定 --port 另一个空闲端口，避免连接到别的服务。")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        cmd = [
            str(self.exe), "-m", str(self.model_path),
            "-c", str(self.args.ctx), "-b", str(self.args.batch),
            "-ub", str(self.args.ubatch), "-ngl", str(self.ngl),
            "-t", str(self.args.threads), "-tb", str(self.args.threads_batch),
            "-np", "1", "--host", "127.0.0.1", "--port", str(self.args.port),
        ]
        print("[SERVER] " + subprocess.list2cmdline(cmd), flush=True)
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            env = os.environ.copy()
            # Single-PC baseline: do not inherit RPC targets from a prior phone experiment.
            for k in ("LLAMA_ARG_RPC", "GGML_RPC_SERVERS"):
                env.pop(k, None)
            self.process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self.log,
                                            stderr=subprocess.STDOUT, env=env, **kw)
            deadline = time.monotonic() + self.args.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    fail(f"llama-server 启动退出 (code={self.process.returncode})，日志：{self.log_path}\n"
                         + tail_log(self.log_path))
                try:
                    health = request_json("GET", self.url + "/health", timeout=2)
                    if health.get("status") in ("ok", "ready") or health.get("status") is None:
                        print(f"[SERVER] READY model={self.model_path.name} ngl={self.ngl}", flush=True)
                        return self
                except (urllib.error.URLError, ValueError, OSError, TimeoutError):
                    pass
                time.sleep(1)
            fail(f"llama-server 启动超时 ({self.args.startup_timeout}s)，日志：{self.log_path}\n"
                 + tail_log(self.log_path))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log:
            self.log.close()
        self.process = None
        return False

    def completion(self, text):
        request = {
            "prompt": text, "n_predict": 1, "ignore_eos": True,
            "cache_prompt": False, "stream": False, "seed": 42,
            "temperature": 0.0,
        }
        return request_json("POST", self.url + "/completion", request,
                            timeout=self.args.request_timeout)


def tail_log(path, n=25):
    try:
        with path.open("rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8000))
            return "\n".join(f.read().decode("utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def server_result(server, prompt_text):
    t0 = time.monotonic()
    try:
        payload = server.completion(prompt_text)
        wall_ms = (time.monotonic() - t0) * 1000
        timing = payload.get("timings") or {}
        if "prompt_n" not in timing or "prompt_ms" not in timing:
            return {"status": "FAIL", "wall_ms": wall_ms,
                    "error": "服务器返回缺少 timings.prompt_n/prompt_ms: " + json.dumps(payload, ensure_ascii=False)[:1000]}
        tokens = int(timing["prompt_n"])
        cache = int(timing.get("cache_n", 0))
        ms = float(timing["prompt_ms"])
        truncated = bool(payload.get("truncated", False))
        predicted = int(timing.get("predicted_n", 0))
        tps = tokens * 1000 / ms if ms > 0 else 0
        if truncated:
            fail("TRUNCATED：上下文溢出，不能计入均值；请调大 --ctx")
        if cache != 0:
            fail(f"CACHE_REUSED：cache_n={cache}，不能计入独立 Prefill 均值")
        if tokens <= 0 or ms <= 0 or not math.isfinite(tps):
            fail(f"INVALID_TIMING prompt_n={tokens}, prompt_ms={ms}")
        return {"status": "PASS", "prompt_tokens": tokens, "prompt_ms": round(ms, 4),
                "prefill_tps": round(tps, 5), "cache_n": cache,
                "predicted_tokens": predicted, "truncated": truncated,
                "wall_ms": round(wall_ms, 3)}
    except Exception as exc:
        return {"status": "FAIL", "wall_ms": round((time.monotonic() - t0) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}"}


# Compatible with llama_perf_context_print and older llama_print_timings format.
CLI_PERF = re.compile(
    r"prompt\s+eval\s+time\s*=\s*([\d,]+(?:\.\d+)?)\s*ms\s*/\s*(\d+)\s*tokens?"
    r"(?:\s*\(\s*([\d,]+(?:\.\d+)?)\s*tokens?\s*per\s*second\s*\))?",
    flags=re.I,
)


def cli_result(args, exe, model_path, ngl, prompt_path, log_path):
    cmd = [str(exe), "-m", str(model_path), "-f", str(prompt_path),
           "-n", "1", "-c", str(args.ctx), "-b", str(args.batch),
           "-ub", str(args.ubatch), "-ngl", str(ngl), "-t", str(args.threads),
           "-tb", str(args.threads_batch)]
    # `-n 1` plus stdin=DEVNULL keeps this a one-shot prefill test.
    # Do not add `-st`: in some llama.cpp builds it controls special-token output.
    if args.cli_verbosity is not None:
        cmd.extend(["-lv", str(args.cli_verbosity)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        with log_path.open("wb") as output:
            env = os.environ.copy()
            for k in ("LLAMA_ARG_RPC", "GGML_RPC_SERVERS"):
                env.pop(k, None)
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=output,
                                  stderr=subprocess.STDOUT, timeout=args.request_timeout, env=env)
        wall_ms = (time.monotonic() - t0) * 1000
        logs = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            fail(f"llama-cli exit code={proc.returncode}: " + tail_log(log_path, n=8))
        matches = CLI_PERF.findall(logs)
        if not matches:
            fail("找不到 prompt eval time；请检查所用 llama-cli/llama-completion 是否输出原生计时")
        ms_str, n_str, _ = matches[-1]
        ms = float(ms_str.replace(",", ""))
        tokens = int(n_str)
        tps = tokens * 1000 / ms if ms > 0 else 0
        if tokens <= 0 or ms <= 0 or not math.isfinite(tps):
            fail(f"INVALID_TIMING: n={tokens}, ms={ms}")
        # Identify explicit context truncation/overflow warnings where emitted.
        if re.search(r"(?:prompt|context).{0,50}(?:truncat|exceed|too long)", logs, re.I):
            fail("POSSIBLE_TRUNCATION：日志出现上下文截断/超限提示，检查 log")
        return {"status": "PASS", "prompt_tokens": tokens, "prompt_ms": round(ms, 4),
                "prefill_tps": round(tps, 5), "cache_n": 0,
                "predicted_tokens": 1, "truncated": False, "wall_ms": round(wall_ms, 3)}
    except Exception as exc:
        return {"status": "FAIL", "wall_ms": round((time.monotonic() - t0) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}"}


def make_summary(args, csv_path, out, selected_models, items, keys):
    # The latest row per unique prompt/repeat/config key is authoritative.
    latest = {}
    for row in read_csv(csv_path):
        if row.get("key") in keys:
            latest[row["key"]] = row
    group_rows = []
    model_rows = []
    for name in selected_models:
        all_success = []
        for group in args.groups:
            expected_keys = [key for key, (m, g) in keys.items() if m == name and g == group]
            vals = [latest[k] for k in expected_keys if k in latest and latest[k].get("status") == "PASS"]
            group_result = summarize_values(vals)
            group_rows.append({"model": name, "group": group, "expected": len(expected_keys),
                               "pass": len(vals), "fail_or_missing": len(expected_keys) - len(vals),
                               **group_result})
            all_success.extend(vals)
        model_result = summarize_values(all_success)
        expect = len([k for k, (m, _) in keys.items() if m == name])
        model_rows.append({"model": name, "expected": expect, "pass": len(all_success),
                           "fail_or_missing": expect - len(all_success), **model_result})
    save_dict_csv(out / "summary_by_group.csv", group_rows)
    save_dict_csv(out / "summary_by_model.csv", model_rows)
    print("\n========== GROUP MEAN (arithmetic tok/s) ==========")
    print(f"{'model':<7} {'group':<6} {'PASS':>7} {'mean tok/s':>13} {'weighted tok/s':>16}")
    for r in group_rows:
        print(f"{r['model']:<7} {r['group']:<6} "
              f"{str(r['pass']) + '/' + str(r['expected']):>7} "
              f"{r['mean_tps']:>13} {r['weighted_tps']:>16}")
    print("---------- MODEL MEAN ----------")
    for r in model_rows:
        print(f"{r['model']}: PASS {r['pass']}/{r['expected']} | mean={r['mean_tps']} tok/s | "
              f"weighted={r['weighted_tps']} tok/s")
    print(f"CSV: {out / 'summary_by_group.csv'}")


def summarize_values(rows):
    if not rows:
        return {"mean_tps": "", "weighted_tps": "", "mean_prompt_ms": "",
                "total_tokens": 0, "total_prompt_ms": 0}
    tps = [float(r["prefill_tps"]) for r in rows]
    ms = [float(r["prompt_ms"]) for r in rows]
    toks = [int(r["prompt_tokens"]) for r in rows]
    total_ms = sum(ms)
    return {"mean_tps": round(mean(tps), 4),
            "weighted_tps": round(sum(toks) * 1000 / total_ms, 4) if total_ms else "",
            "mean_prompt_ms": round(mean(ms), 4), "total_tokens": sum(toks),
            "total_prompt_ms": round(total_ms, 4)}


def save_dict_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="单机 llama.cpp：ShareGPT50 五组 Prefill 基线 (默认每组前5条)")
    p.add_argument("--prompt-dir", default=r"E:\llama\datasets\sharegpt\sharegpt50_short")
    p.add_argument("--models-root", default=r"E:\llama\models", help="自动寻找3个GGUF的根目录，可递归")
    p.add_argument("--model-14b", help="指定 14B GGUF 的完整路径，优先于自动搜索")
    p.add_argument("--model-24b", help="指定 24B GGUF 的完整路径")
    p.add_argument("--model-32b", help="指定 32B GGUF 的完整路径")
    p.add_argument("--models", nargs="+", choices=("14B", "24B", "32B"), default=["14B", "24B", "32B"])
    p.add_argument("--bin-dir", default=r"E:\llama\llama-official\build\bin\Release",
                   help="原生 llama.cpp 的 bin/Release 目录；自动优先 llama-server，缺少则用 llama-cli")
    p.add_argument("--mode", choices=("auto", "server", "cli"), default="auto")
    p.add_argument("--out", default=r"E:\llama\results\llama_pc_sharegpt50_official_ngl_all")
    p.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=list(GROUPS))
    p.add_argument("--per-group", type=int, default=5, help="先运行前5条；改为10则自动补齐50条")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, choices=(512, 2048), default=512)
    p.add_argument("--ctx", type=int, default=3000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--threads-batch", type=int, default=8)
    p.add_argument("--ngl", default="all",
                   help="统一 GPU offload 层数：all（默认）或非负整数；OOM 时可手动调整")
    p.add_argument("--ngl14b", default=None, help="14B 单独覆盖 --ngl，例如 12 或 all")
    p.add_argument("--ngl24b", default=None, help="24B 单独覆盖 --ngl，例如 8 或 all")
    p.add_argument("--ngl32b", default=None, help="32B 单独覆盖 --ngl，例如 10 或 all")
    p.add_argument("--cli-verbosity", type=int, default=3,
                   help="CLI 模式传给 -lv 的日志级别，匹配你的 llama-cli -lv 3；设为 -1 可关闭")
    p.add_argument("--port", type=int, default=18089)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--no-warmup", action="store_true", help="server 默认做1次不计入成绩的预热")
    p.add_argument("--no-resume", action="store_true", help="重跑已PASS的样本，summary只取最近一次")
    p.add_argument("--dry-run", action="store_true", help="只检查路径/配置，不实际测试")
    args = p.parse_args()
    if not 1 <= args.per_group <= 10 or args.repeats < 1:
        fail("--per-group 必须在1..10之间，--repeats >=1")
    if args.ubatch > args.batch:
        fail("物理 ubatch 不应大于 batch")
    if args.cli_verbosity is not None and args.cli_verbosity < 0:
        args.cli_verbosity = None
    if len(set(args.groups)) != len(args.groups) or len(set(args.models)) != len(args.models):
        fail("groups/models 不要重复")
    mode, exe = select_exe(args)
    items = prompt_list(args)
    models = discover_models(args)
    out = Path(args.out).expanduser().resolve()
    ngls = {"14B": normalize_ngl(args.ngl14b if args.ngl14b is not None else args.ngl),
            "24B": normalize_ngl(args.ngl24b if args.ngl24b is not None else args.ngl),
            "32B": normalize_ngl(args.ngl32b if args.ngl32b is not None else args.ngl)}
    print(f"[CONFIG] mode={mode} exe={exe}")
    print(f"[CONFIG] inputs={args.prompt_dir} groups={args.groups} per_group={args.per_group} "
          f"repeats={args.repeats} | total cases={len(items) * args.repeats * len(models)}")
    print(f"[CONFIG] b={args.batch} ub={args.ubatch} ctx={args.ctx} threads={args.threads}/{args.threads_batch}")
    if "tensor-v2" in str(exe).lower() or "region-ring" in str(exe).lower():
        print("[NOTE] 当前可能是改造版 llama.cpp。论文原生单机基线建议用干净上游版本编译出的 bin。")
    for name, path in models.items():
        print(f"[MODEL] {name} ngl={ngls[name]} {path}")
    if any(v == "all" for v in ngls.values()):
        print("[WARN] -ngl all 将尝试尽量把模型层放在 GPU。6GiB 显存无法完整装入这些 14B/24B/32B GGUF 权重，"
              "可能 OOM 或触发共享 GPU 内存导致明显降速；脚本不会静默降低 -ngl。", flush=True)
    if args.dry_run:
        print("[DRY-RUN] 检查通过，未启动模型。")
        return
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.csv"
    existing = {r.get("key"): r for r in read_csv(result_path)}
    planned_keys = {}
    # Build the complete expected set before inference, so an interrupted run
    # shows honest PASS/expected counts for all three models.
    for name, model_path in models.items():
        for group, index, prompt_path in items:
            raw = prompt_path.read_bytes()
            for repeat in range(1, args.repeats + 1):
                k = config_key(args, mode, exe, name, model_path, raw, group, index, repeat, ngls[name])
                planned_keys[k] = (name, group)
    interrupted = False
    try:
        for name, model_path in models.items():
            ngl = ngls[name]
            pending = []
            for group, index, prompt_path in items:
                raw = prompt_path.read_bytes()
                try:
                    content = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    fail(f"Prompt 不是 UTF-8：{prompt_path}")
                if not content.strip():
                    fail(f"Prompt 为空：{prompt_path}")
                for repeat in range(1, args.repeats + 1):
                    key = config_key(args, mode, exe, name, model_path, raw, group, index, repeat, ngl)
                    planned_keys[key] = (name, group)
                    if not args.no_resume and existing.get(key, {}).get("status") == "PASS":
                        print(f"[SKIP] {name} {prompt_path.name} rep={repeat} PASS already")
                        continue
                    pending.append((group, index, prompt_path, content, repeat, key))
            if not pending:
                print(f"[MODEL] {name} 全部已 PASS，跳过启动")
                continue
            server_log = out / "logs" / f"server_{name}.log"
            print(f"\n[MODEL] {name} remaining={len(pending)}", flush=True)
            if mode == "server":
                with ServerProcess(args, exe, model_path, ngl, server_log) as server:
                    if not args.no_warmup:
                        warm = server_result(server, "Hello world. " * 20)
                        if warm["status"] != "PASS":
                            print("[WARN] warmup 失败：" + str(warm.get("error", "")), flush=True)
                    for model_index, (group, index, prompt_path, content, repeat, key) in enumerate(pending, 1):
                        result = server_result(server, content)
                        row = pack_row(args, mode, name, model_path, prompt_path, group,
                                       index, repeat, key, ngl, result, server_log, out)
                        append_csv(result_path, row)
                        existing[key] = row
                        print_case(model_index, len(pending), row)
            else:
                for model_index, (group, index, prompt_path, content, repeat, key) in enumerate(pending, 1):
                    log_path = out / "logs" / name / f"{prompt_path.stem}_rep{repeat}.log"
                    result = cli_result(args, exe, model_path, ngl, prompt_path, log_path)
                    row = pack_row(args, mode, name, model_path, prompt_path, group,
                                   index, repeat, key, ngl, result, log_path, out)
                    append_csv(result_path, row)
                    existing[key] = row
                    print_case(model_index, len(pending), row)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPTED] Ctrl+C：已写出的 PASS 下次会自动跳过。", flush=True)
    finally:
        if planned_keys and out.is_dir():
            make_summary(args, result_path, out, models.keys(), items, planned_keys)
    if interrupted:
        sys.exit(130)


def pack_row(args, mode, name, model_path, prompt_path, group, index, repeat,
             key, ngl, result, log_path, output_path):
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "timestamp_utc": stamp(), "key": key, "mode": mode, "model": name,
        "model_path": str(model_path), "prompt": prompt_path.name,
        "group": group, "index": index, "repeat": repeat, "batch": args.batch,
        "ubatch": args.ubatch, "ctx": args.ctx, "threads": args.threads,
        "threads_batch": args.threads_batch, "ngl": ngl,
        "log_file": relative_or_abs(log_path, output_path),
    })
    row.update(result)
    return row


def print_case(i, total, row):
    if row["status"] == "PASS":
        print(f"  [{i:02d}/{total:02d}] {row['model']:<3} {row['prompt']:<21} "
              f"PASS tokens={row['prompt_tokens']:<5} "
              f"prefill={float(row['prefill_tps']):.2f} tok/s "
              f"({float(row['prompt_ms']):.1f} ms)", flush=True)
    else:
        print(f"  [{i:02d}/{total:02d}] {row['model']:<3} {row['prompt']:<21} "
              f"FAIL {str(row['error'])[:170]}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print("[ERROR] " + str(err), file=sys.stderr)
        sys.exit(1)