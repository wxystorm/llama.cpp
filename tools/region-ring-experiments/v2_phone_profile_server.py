#!/usr/bin/env python3
"""LEGACY standalone Python V2 Profile sidecar for regression tests only.

The production Android ggml-rpc-server (4.0.35+) invokes the C++ native
profiler *in-process* on port 50052 and NEVER runs this Python module.
This file remains solely to reproduce old sidecar tests and data formats.
Do not deploy it to Phone or start a 50177 listener for a GCNA sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import secrets
import socketserver
import subprocess
import sys
from typing import Any

from v2_profile_wire import (
    DEFAULT_PORT, MAX_FRAME, ProfileProtocolError, SOURCE, WIRE_VERSION,
    decode, encode, validate_sample, validate_shape,
)
from v2_profile_transfer import phone_serve_transfer, validate_transfer


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def backend_library_hashes(exe: Path) -> dict[str, str]:
    # The native executable hash alone is insufficient: updating an HTP
    # DSP .so or OpenCL backend without recompiling the executable changes
    # the measured kernels. Resolve the local deployment library dir first.
    search = [exe.parent.parent / "lib"]
    search.extend(Path(p) for p in
                  os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                  if p)
    names = ("libggml-htp-v79.so", "libggml-hexagon.so",
             "libggml-opencl.so", "libggml-cpu.so", "libggml.so")
    result = {}
    for name in names:
        found = next((base / name for base in search
                      if (base / name).is_file()), None)
        result[name] = sha256(found) if found else "NOT_FOUND"
    return result


def property_value(name: str) -> str:
    try:
        proc = subprocess.run(["getprop", name], capture_output=True,
                              text=True, timeout=2, check=False)
        return proc.stdout.strip()[:128] if proc.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


class Service:
    def __init__(self, model: Path, exe: Path, *, token: str = "",
                 timeout: int = 300):
        self.model = model.expanduser().resolve(strict=True)
        self.exe = exe.expanduser().resolve(strict=True)
        if not self.model.is_file() or not self.exe.is_file():
            raise ValueError("PHONE_MODEL_AND_PROFILER_MUST_BE_FILES")
        self.token = token
        self.timeout = timeout
        if timeout <= 0:
            raise ValueError("PHONE_PROFILE_TIMEOUT_MUST_BE_POSITIVE")
        self.model_sha256 = sha256(self.model)
        self.exe_sha256 = sha256(self.exe)
        device = {
            "machine": platform.machine(),
            "product": property_value("ro.product.model"),
            "soc": property_value("ro.soc.model"),
            "abi": property_value("ro.product.cpu.abi"),
            "os_build": property_value("ro.build.fingerprint"),
            "vendor_build": property_value("ro.vendor.build.fingerprint"),
        }
        identity = "|".join(str(x) for x in device.values())
        self.device_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self.device = device
        self.backend_library_sha256 = backend_library_hashes(self.exe)
        # The integrated sampler has three independently updateable pieces:
        # the native graph profiler, its Python control worker, and the
        # inference RPC server hosting command 57. All are part of the
        # measurement provenance, not just the native profiler executable.
        self.control_sha256 = sha256(Path(__file__).resolve())
        rpc_server_exe = self.exe.parent / "ggml-rpc-server"
        self.rpc_server_sha256 = (
            sha256(rpc_server_exe) if rpc_server_exe.is_file()
            else "NOT_FOUND")
        runtime = {"exe_sha256": self.exe_sha256,
                   "library_sha256": self.backend_library_sha256,
                   "device_id": self.device_id,
                   "control_sha256": self.control_sha256,
                   "rpc_server_sha256": self.rpc_server_sha256}
        self.runtime_sha256 = hashlib.sha256(
            json.dumps(runtime, sort_keys=True).encode("utf-8")).hexdigest()

    def hello(self) -> dict[str, Any]:
        return {"version": WIRE_VERSION, "status": "OK",
                "model_sha256": self.model_sha256,
                "exe_sha256": self.exe_sha256,
                "runtime_sha256": self.runtime_sha256,
                "control_sha256": self.control_sha256,
                "rpc_server_sha256": self.rpc_server_sha256,
                "library_sha256": self.backend_library_sha256,
                "device_id": self.device_id,
                "device": self.device,
                "backends": ["N", "A"],
                "transport": "V2_PHONE_PROFILE_SIDECAR"}

    def authenticate(self, request: dict[str, Any]) -> None:
        if self.token:
            supplied = request.get("token")
            if not isinstance(supplied, str) or not secrets.compare_digest(
                    supplied, self.token):
                raise ProfileProtocolError("UNAUTHORIZED_PROFILE_REQUEST")

    def authorize_transfer(self, request: dict[str, Any]) -> None:
        self.authenticate(request)
        if request.get("model_sha256") != self.model_sha256:
            raise ProfileProtocolError("PHONE_PC_GGUF_SHA256_MISMATCH")
        validate_transfer(request)

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            self.authenticate(request)
            action = request.get("action")
            if action == "hello":
                return self.hello()
            if action != "profile":
                raise ProfileProtocolError("UNKNOWN_PROFILE_ACTION")
            if request.get("model_sha256") != self.model_sha256:
                raise ProfileProtocolError("PHONE_PC_GGUF_SHA256_MISMATCH")
            shape = validate_shape(request.get("shape"))
            command = [
                str(self.exe), "--model", str(self.model),
                "--backend", shape["backend"],
                "--layers", str(shape["layers"]),
                "--chunk", str(shape["chunk"]),
                "--kv", str(shape["kv_tokens"]),
                "--warmups", str(shape["warmups"]),
                "--repeats", str(shape["repeats"]),
                "--threads", str(shape["threads"]),
                "--relative-rmse", str(shape["max_relative_rmse"]),
                "--max-abs-error", str(shape["max_abs_error"]),
                "--bank-repeats", str(shape["bank_repeats"]),
                "--host-stage-cap-mib", str(shape["host_stage_cap_mib"]),
            ]
            if shape["verify_cpu"]:
                command.append("--verify-cpu")
            # Explicit argv, no shell; accepts no client-provided paths.
            try:
                run = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ProfileProtocolError("PHONE_NATIVE_PROFILE_TIMEOUT") from exc
            except OSError as exc:
                raise ProfileProtocolError(
                    "PHONE_NATIVE_EXEC_FAILED:" + str(exc)) from exc
            reader = csv.DictReader(io.StringIO(run.stdout))
            rows = list(reader)
            if len(rows) != 1:
                raise ProfileProtocolError(
                    "PHONE_NATIVE_NO_VALID_RESULT exit=" +
                    str(run.returncode) + " stderr=" + run.stderr[-1500:])
            sample = validate_sample(rows[0], shape)
            numeric_pass = (
                float(sample["relative_rmse"]) <= shape["max_relative_rmse"] and
                float(sample["max_abs"]) <= shape["max_abs_error"])
            if shape["verify_cpu"]:
                if sample["parity"] == "PASS" and not numeric_pass:
                    raise ProfileProtocolError(
                        "PHONE_NATIVE_PARITY_PASS_CONTRADICTS_MEASURED_ERROR")
                if sample["parity"] == "FAIL" and numeric_pass:
                    raise ProfileProtocolError(
                        "PHONE_NATIVE_PARITY_FAIL_CONTRADICTS_MEASURED_ERROR")
            if run.returncode == 0 and sample["parity"] != "FAIL" and (
                    not shape["verify_cpu"] or numeric_pass):
                state = "PASS"
            elif run.returncode == 2 and sample["parity"] == "FAIL" and (
                    shape["verify_cpu"] and not numeric_pass):
                state = "PARITY_FAIL"
            else:
                raise ProfileProtocolError(
                    "PHONE_NATIVE_EXIT_PARITY_MISMATCH exit=" +
                    str(run.returncode) + " parity=" + sample["parity"])
            return {
                "version": WIRE_VERSION,
                "status": state,
                "sample": sample,
                "model_sha256": self.model_sha256,
                "exe_sha256": self.exe_sha256,
                "runtime_sha256": self.runtime_sha256,
                "device_id": self.device_id,
                "diagnostics": run.stderr[-2500:],
            }
        except (ProfileProtocolError, ValueError) as exc:
            return {"version": WIRE_VERSION, "status": "ERROR",
                    "error": str(exc)[:3000]}


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(self.server.rpc_timeout)
        try:
            raw = self.rfile.readline(MAX_FRAME + 1)
            request = decode(raw)
            if request.get("action") == "transfer":
                self.server.service.authorize_transfer(request)
                phone_serve_transfer(request, self.rfile, self.connection)
                return
            response = self.server.service.dispatch(request)
        except (ProfileProtocolError, ValueError, OSError) as exc:
            response = {"version": WIRE_VERSION, "status": "ERROR",
                        "error": str(exc)[:3000]}
        try:
            self.wfile.write(encode(response))
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass


def serve_stdio(service: Service) -> int:
    """Internal worker for the *existing* ggml-rpc port 50052.

    The RPC server owns the network listener and forwards only bounded JSON
    frames through pipes. No second socket or exposed arbitrary shell command.
    stdout is reserved exclusively for protocol frames; logging uses stderr.
    """
    print("[V2_PHONE_PROFILE] transport=RPC50052_STDIO status=READY",
          file=sys.stderr, flush=True)
    while True:
        raw = sys.stdin.buffer.readline(MAX_FRAME + 1)
        if not raw:
            return 0
        try:
            request = decode(raw)
            # The independent TCP throughput diagnostic is not a valid
            # substitute for production ggml-rpc tensor transport.
            if request.get("action") == "transfer":
                raise ProfileProtocolError("USE_PRODUCTION_RPC_FOR_TRANSFERS")
            response = service.dispatch(request)
            if response.get("transport") == "V2_PHONE_PROFILE_SIDECAR":
                response["transport"] = "V2_PHONE_PROFILE_RPC50052"
        except (OSError, ValueError, ProfileProtocolError) as exc:
            response = {"version": WIRE_VERSION, "status": "ERROR",
                        "error": str(exc)[:3000]}
        sys.stdout.buffer.write(encode(response))
        sys.stdout.buffer.flush()


class Server(socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 4

    def __init__(self, address: tuple[str, int], service: Service):
        self.service = service
        self.rpc_timeout = float(service.timeout) + 30
        super().__init__(address, Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--stdio", action="store_true",
                        help="serve JSON over inherited pipes, no listening port")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=os.getenv("V2_PROFILE_TOKEN", ""))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not args.stdio:
        if not 1 <= args.port <= 65535:
            parser.error("INVALID_PROFILE_PORT")
        if args.bind not in ("127.0.0.1", "localhost", "::1") and \
                len(args.token) < 16:
            parser.error("NON_LOOPBACK_PHONE_PROFILE_REQUIRES_TOKEN_16_CHARS")
    try:
        service = Service(args.model, args.exe, token=args.token,
                          timeout=args.timeout)
        if args.stdio:
            return serve_stdio(service)
        print("[V2_PHONE_PROFILE] model_sha256=" + service.model_sha256,
              "exe_sha256=" + service.exe_sha256,
              "runtime_sha256=" + service.runtime_sha256,
              "device_id=" + service.device_id, flush=True)
        with Server((args.bind, args.port), service) as server:
            print(f"[V2_PHONE_PROFILE] listening={args.bind}:{args.port} "
                  f"protocol={WIRE_VERSION} status=READY", flush=True)
            server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("[V2_PHONE_PROFILE] status=STOPPED", flush=True)
        return 0
    except (OSError, ValueError) as exc:
        print("[V2_PHONE_PROFILE] status=FAIL reason=" + str(exc),
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
