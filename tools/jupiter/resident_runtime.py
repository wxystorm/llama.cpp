#!/usr/bin/env python3
"""Strict local resident backend for the existing Jupiter CLI runners."""
from __future__ import annotations

import array
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from native_calibration import active as native_calibration_active, validate_run

PREFIX = "LLAMA_JUPITER_RESIDENT_"
FLAGS = ("TRACE", "PROFILE_STAGE", "Q_SWEEP", "TIMELINE",
         "SCHED_ALLOC_TIMING", "LOGITS_HASH")
READY = re.compile(rb"\[JUPITER_RESIDENT\] phase=READY port=(\d+) native=(\d+) ctx=(\d+) batch=(\d+) ubatch=(\d+)")
CLI_VALUES = {"-m", "--rpc", "--split-mode", "-ngl", "-ub", "-b", "-c",
              "-n", "-t", "--flash-attn", "-lv", "-p"}
CLI_SWITCHES = {"--no-conversation", "--single-turn", "--simple-io",
                "--no-show-timings", "--no-warmup"}


def enabled(env):
    return bool(env.get(PREFIX + "URL"))


def http_json(url, key, payload, timeout):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False,
                                                   allow_nan=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def parse_cli(command):
    values = {}
    i = 1
    while i < len(command):
        key = command[i]
        if key in CLI_SWITCHES:
            i += 1
            continue
        if key not in CLI_VALUES or i + 1 == len(command) or key in values:
            raise ValueError(f"unsupported/duplicate resident option: {key}")
        values[key] = command[i+1]
        i += 2
    if set(values) != CLI_VALUES:
        raise ValueError(f"missing resident options: {CLI_VALUES-set(values)}")
    for key, required in (("-n", "1"), ("-ngl", "all"),
                          ("--split-mode", "layer"), ("--flash-attn", "on")):
        if values[key] != required:
            raise ValueError(f"resident requires {key} {required}")
    return values


def explicit_cuts(env, config):
    direct = env.get("LLAMA_JUPITER_SEQUENCE_PLAN", "")
    plan_file = env.get("LLAMA_JUPITER_PLAN_FILE", "")
    if direct and plan_file:
        raise ValueError("ambiguous resident plan sources")
    if direct:
        return [int(s) for s in direct.split(",")]
    if not plan_file:
        return []
    sequences, placement = [], {}
    for line in Path(plan_file).read_text(encoding="utf-8-sig").splitlines():
        text = line.split("#", 1)[0].strip()
        if "=" not in text:
            continue
        key, value = (s.strip() for s in text.split("=", 1))
        if key in ("model.pc_layers", "model.cpu_layers"):
            placement[key] = int(value)
        elif key.startswith("sequence."):
            cuts = [int(s) for s in value.split(",")]
            if sum(cuts) != int(key.removeprefix("sequence.")):
                raise ValueError("plan token sum mismatch")
            sequences.append(cuts)
    if (placement.get("model.pc_layers", config["pc_layers"]) != config["pc_layers"]
            or placement.get("model.cpu_layers", 0) != config["cpu_layers"]
            or len(sequences) != 1):
        raise ValueError("resident requires one exact sequence for the loaded layout")
    return sequences[0]


def request_payload(command, env, config, request_id):
    cli = parse_cli(command)
    calibration = bool(config.get("calibration", False))
    if calibration != native_calibration_active(env):
        raise ValueError("resident calibration namespace mismatch")
    for key, expected in (("-m", config["model"]), ("--rpc", config["rpc"]),
                          ("-b", str(config["batch"])), ("-c", str(config["context"])),
                          ("-t", str(config["threads"]))):
        actual = str(Path(cli[key]).resolve()) if key == "-m" else cli[key]
        if actual != expected:
            raise ValueError(f"resident immutable configuration mismatch: {key}")
    for key, expected in (("PC_LAYERS", config["pc_layers"]),
                          ("CPU_LAYERS", config["cpu_layers"]),
                          ("HTP_SESSIONS", config["sessions"]),
                          ("HTP_ACTIVE_SESSIONS", config["sessions"]),
                          ("HTP_CONTEXT_TOKENS", config["context"]),
                          ("PHONE_OWNED_NATIVE", int(config["native"])),
                          ("WINDOW", config["window"])):
        if int(env.get("LLAMA_JUPITER_" + key, str(expected))) != expected:
            raise ValueError(f"resident immutable configuration mismatch: {key}")
    ubatch = int(cli["-ub"])
    if not 1 <= ubatch <= config["ubatch"]:
        raise ValueError("requested ubatch exceeds resident capacity")
    if not config["native"] and ubatch != config["ubatch"]:
        raise ValueError("legacy calibration must use the loaded UBatch")
    for flag in ("GRAPH_PREBUILD", "LOCAL_GRAPH_REUSE", "ASYNC_C2P", "PHONE_OWNED_ASYNC"):
        if env.get("LLAMA_JUPITER_" + flag, "0") != "0":
            raise ValueError(f"resident does not support {flag}=1")
    if env.get("LLAMA_JUPITER_STAGE_DUMP_DIR"):
        raise ValueError("resident does not support intermediate stage dumps")
    cuts = explicit_cuts(env, config)
    if any(c <= 0 or c > ubatch for c in cuts):
        raise ValueError("invalid resident chunk width")
    profile = env.get("LLAMA_JUPITER_PROFILE_STAGE", "0") == "1"
    if config["native"] and (len(cuts) > 16 or (not cuts and (not calibration or profile))):
        raise ValueError("native resident requires an exact supported DP plan (except token probe)")
    if config["native"] and profile and not calibration:
        raise ValueError("use the separate native calibration worker for stage profiling")
    auto = env.get("LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS", "0") == "1"
    if calibration and (not config["native"] or not auto or env.get("LLAMA_JUPITER_LOGITS_DUMP_FILE")):
        raise ValueError("native calibration requires autonomous mode and no logits export")
    if auto and not config["native"]:
        raise ValueError("autonomous request sent to legacy worker")
    return {"id": request_id, "prompt": cli["-p"], "chunks": cuts,
            "expected_tokens": sum(cuts), "autonomous": auto,
            "window": config["window"], "calibration": calibration,
            "flags": {k: env.get("LLAMA_JUPITER_" + k, "0") == "1" for k in FLAGS},
            "return_logits": bool(env.get("LLAMA_JUPITER_LOGITS_DUMP_FILE"))}


def request_log(path, offset, request_id):
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
    begin = f"[JUPITER_RESIDENT] phase=BEGIN id={request_id} status=RUN".encode()
    end = f"[JUPITER_RESIDENT] phase=END id={request_id} status=".encode()
    first = data.find(begin)
    last = data.find(end, max(0, first))
    if first < 0 or last < 0:
        raise RuntimeError("missing exact resident request log boundaries")
    stop = data.find(b"\n", last)
    if stop < 0:
        raise RuntimeError("incomplete resident end marker")
    return data[first:stop+1]


def timeout_marker(server_log):
    return Path(server_log).with_suffix(".timeout.json")


def is_request_timeout(exc):
    return isinstance(exc, TimeoutError) or (
        isinstance(exc, urllib.error.URLError) and
        isinstance(exc.reason, TimeoutError))


def record_request_timeout(server_log, offset, request_id, log_path, exc):
    # Keep partial traces SEPARATE from the canonical successful sample log.
    # A partial trace can contain a valid-looking plan but no completed run.
    archive = log_path.with_name(log_path.stem + ".timeout-" + request_id + ".log")
    with server_log.open("rb") as stream:
        stream.seek(offset)
        archive.write_bytes(stream.read())
    marker = timeout_marker(server_log)
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps({
        "reason": "HTTP_REQUEST_TIMEOUT", "id": request_id,
        "sample_log": str(log_path.resolve()), "partial_trace": str(archive.resolve()),
        "error": str(exc),
    }, indent=2) + "\n", encoding="utf-8")
    temp.replace(marker)


def run_resident_cli(command, env, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"")
    server_log = Path(env[PREFIX + "LOG"])
    request_in_flight = False
    try:
        if timeout_marker(server_log).is_file():
            # Never enqueue on a worker with an unresolved timed-out request.
            raise RuntimeError("resident request already timed out; restart required")
        request_id = uuid.uuid4().hex
        config = json.loads(env[PREFIX + "CONFIG"])
        payload = request_payload(list(command), env, config, request_id)
        offset = server_log.stat().st_size
        t0 = time.perf_counter()
        request_in_flight = True
        status, result = http_json(env[PREFIX + "URL"] + "/jupiter/prefill",
                                   env[PREFIX + "KEY"], payload,
                                   float(env[PREFIX + "TIMEOUT"]))
        request_in_flight = False
        wall_ms = (time.perf_counter() - t0)*1000
        if status != 200:
            try:
                log_path.write_bytes(request_log(server_log, offset, request_id))
            except RuntimeError:
                pass
            raise RuntimeError(f"HTTP {status}: {result.get('error', 'unknown error')}")
        trace = request_log(server_log, offset, request_id)
        log_path.write_bytes(trace)
        if (result.get("status") != "PASS" or result.get("id") != request_id or
                result.get("metric") != "DECODE_TO_LOGITS" or
                f"[JUPITER_RESIDENT_RESET] id={request_id} status=PASS".encode() not in trace or
                f"[JUPITER_RESIDENT] phase=END id={request_id} status=PASS".encode() not in trace):
            raise RuntimeError("invalid resident completion or missing reset")
        if payload["expected_tokens"] and result["tokens"] != payload["expected_tokens"]:
            raise RuntimeError("resident actual token count differs from exact DP plan")
        if payload["calibration"]:
            validate_run(trace.decode("utf-8", errors="replace"), result["tokens"], payload["chunks"])
            result["calibration_backend"] = "native-autonomous"
        timing_only = bool(config.get("legacy_timing_only", False)
                           and not config["native"] and not payload["return_logits"])
        if result.get("logits_validation") != ("TIMING_ONLY" if timing_only else "FINITE_ONLY"):
            raise RuntimeError("resident logits validation policy mismatch")
        if not timing_only and result.get("logits_finite") is not True:
            raise RuntimeError("resident requires finite logits")
        if timing_only and result.get("logits_finite") is not True:
            print(f"[JUPITER_RESIDENT_TIMING_ONLY] nan={result.get('logits_nan')} "
                  f"inf={result.get('logits_inf')} accuracy=UNVERIFIED", flush=True)
        logits = result.pop("logits", None)
        if payload["return_logits"]:
            if not isinstance(logits, list) or not logits or not all(math.isfinite(v) for v in logits):
                raise RuntimeError("missing/nonfinite resident F32 logits")
            dest = Path(env["LLAMA_JUPITER_LOGITS_DUMP_FILE"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as stream:
                array.array("f", logits).tofile(stream)
            result["logits_count"] = len(logits)
        result["request_wall_ms"] = wall_ms
        result["execution_backend"] = "resident-server"
        log_path.with_suffix(".resident.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for line in trace.decode("utf-8", errors="replace").splitlines():
            if "[JUPITER_SEQUENCE_PLAN]" in line or "prompt eval time" in line:
                print(line, flush=True)
        return 0
    except Exception as exc:
        with log_path.open("ab") as stream:
            stream.write(f"\n[JUPITER_RESIDENT_ERROR] {exc}\nserver_log={server_log}\n".encode("utf-8"))
        print(f"[JUPITER_RESIDENT_ERROR] {exc}; see {log_path}", flush=True)
        if request_in_flight and is_request_timeout(exc) and env.get(PREFIX + "RECOVER_TIMEOUT") == "1":
            record_request_timeout(server_log, offset, request_id, log_path, exc)
            print(f"[JUPITER_RESIDENT_TIMEOUT] id={request_id} "
                  f"timeout_seconds={env[PREFIX + 'TIMEOUT']} sample={log_path}", flush=True)
            # Stage2 catches ordinary RuntimeError and proceeds to the next
            # sample; exit the child immediately instead. Completed samples
            # have already been checkpointed. The suite owns the restart.
            raise SystemExit(75)
        return 1


class ResidentServer:
    """One owned model process, immutable placement and scheduler window."""
    def __init__(self, args, layout, native, output, ubatch, *, calibration=False):
        if calibration and not native:
            raise ValueError("calibration requires native Phone-owned worker")
        self.config = dict(model=str(Path(args.model).resolve()), rpc=args.rpc,
                           context=2048, batch=2048, ubatch=ubatch,
                           threads=8, pc_layers=layout, cpu_layers=args.cpu_layers,
                           sessions=args.sessions, native=native, window=2 if native else 1,
                           calibration=calibration, legacy_timing_only=bool(not native and
                               getattr(args, "skip_legacy_logits_check", False)))
        self.exe = Path(args.resident_server).resolve()
        self.log = Path(output) / "resident-runtime" / (
            ("native-calibration" if calibration else "native" if native else "legacy")
            + "-" + uuid.uuid4().hex + ".log")
        self.key = uuid.uuid4().hex + uuid.uuid4().hex
        self.timeout = (max(1800, args.timeout_seconds) if native else
                        getattr(args, "legacy_request_timeout", 120))
        self.start_timeout = getattr(args, "resident_start_timeout", 1800)
        self.proc = self.url = None

    def __enter__(self):
        cfg = self.config
        env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
        env.update({
            PREFIX + "SERVER": "1",
            PREFIX + "NATIVE_CALIBRATION": str(int(cfg["calibration"])),
            PREFIX + "LEGACY_TIMING_ONLY": str(int(cfg["legacy_timing_only"])),
            "LLAMA_JUPITER_PREFILL": "1",
            "LLAMA_JUPITER_PC_LAYERS": str(cfg["pc_layers"]),
            "LLAMA_JUPITER_CPU_LAYERS": str(cfg["cpu_layers"]),
            "LLAMA_JUPITER_HTP_SESSIONS": str(cfg["sessions"]),
            "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": str(cfg["sessions"]),
            "LLAMA_JUPITER_HTP_CONTEXT_TOKENS": str(cfg["context"]),
            "LLAMA_JUPITER_PHONE_OWNED_NATIVE": str(int(cfg["native"])),
            "LLAMA_JUPITER_PHONE_OWNED_ENDPOINT": cfg["rpc"],
            "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "0",
            "LLAMA_JUPITER_PHONE_OWNED_ASYNC": "0",
            "LLAMA_JUPITER_WINDOW": str(cfg["window"]),
            **{"LLAMA_JUPITER_" + flag: "0" for flag in FLAGS},
        })
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.exe), "-m", cfg["model"], "--rpc", cfg["rpc"],
               "--split-mode", "layer", "-ngl", "all", "-c", str(cfg["context"]),
               "-b", str(cfg["batch"]), "-ub", str(cfg["ubatch"]),
               "-t", str(cfg["threads"]), "--flash-attn", "on", "--no-warmup",
               "-lv", "4", "--host", "127.0.0.1", "--port", "0", "--api-key", self.key]
        options = dict(stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        print(f"[JUPITER_RESIDENT_START] native={int(cfg['native'])} log={self.log}", flush=True)
        try:
            with self.log.open("wb") as stream:
                self.proc = subprocess.Popen(cmd, stdout=stream, **options)
            deadline = time.monotonic() + self.start_timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"resident exited {self.proc.returncode}; see {self.log}")
                with self.log.open("rb") as stream:
                    stream.seek(max(0, self.log.stat().st_size-65536))
                    match = READY.search(stream.read())
                if match:
                    port, native, ctx, batch, ubatch = map(int, match.groups())
                    if (native, ctx, batch, ubatch) != (int(cfg["native"]), cfg["context"],
                                                        cfg["batch"], cfg["ubatch"]):
                        raise RuntimeError("resident loaded different Context geometry")
                    self.url = f"http://127.0.0.1:{port}"
                    status, health = http_json(self.url + "/health", self.key, None, 10)
                    if status != 200 or not health.get("resident"):
                        raise RuntimeError("not the dedicated resident endpoint")
                    if (health.get("legacy_timing_only") != cfg["legacy_timing_only"] or
                            health.get("native_calibration") != cfg["calibration"]):
                        raise RuntimeError("resident validation/calibration policy mismatch; rebuild llama-server")
                    print(f"[JUPITER_RESIDENT_READY] native={native} port={port}", flush=True)
                    return self
                time.sleep(0.25)
            raise RuntimeError(f"resident startup timeout; see {self.log}")
        except BaseException:
            self.close()
            raise

    def child_env(self):
        if not self.url or self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("resident server is not ready")
        return {PREFIX + "URL": self.url, PREFIX + "KEY": self.key,
                PREFIX + "LOG": str(self.log.resolve()),
                PREFIX + "CONFIG": json.dumps(self.config),
                PREFIX + "TIMEOUT": str(self.timeout),
                PREFIX + "RECOVER_TIMEOUT": str(int(not self.config["native"])),
                PREFIX + "NATIVE_CALIBRATION": str(int(self.config["calibration"]))}

    def close(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        if self.url:
            try:
                http_json(self.url + "/jupiter/shutdown", self.key, {}, 10)
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
                return
            except subprocess.TimeoutExpired:
                pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"[JUPITER_RESIDENT_STOP] forced=1 log={self.log}; check Phone RPC state", flush=True)

    def __exit__(self, *_exc):
        self.close()
