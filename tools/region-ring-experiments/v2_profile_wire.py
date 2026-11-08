#!/usr/bin/env python3
"""Versioned, bounded V2 compute-Profile control protocol.

Only independent N/A Stage probes are accepted on the Phone. This is
a dedicated Profile transport, NOT the legacy ggml-rpc inference port.
Use USB adb forward tcp:50177 tcp:50177 with the Phone listener bound
to localhost, or a trusted LAN and a shared token.
"""
from __future__ import annotations

import json
import math
import socket
from typing import Any

WIRE_VERSION = 1
MAX_FRAME = 256 * 1024
DEFAULT_PORT = 50177
SOURCE = "NATIVE_SINGLE_BACKEND_GGUF_PREFIX"


class ProfileProtocolError(RuntimeError):
    pass


def parse_endpoint(value: str) -> tuple[str, int]:
    if not isinstance(value, str) or not value.strip():
        raise ProfileProtocolError("PHONE_ENDPOINT_REQUIRED")
    host, delimiter, port_str = value.rpartition(":")
    if not delimiter or not host or not port_str.isdecimal():
        raise ProfileProtocolError("PHONE_ENDPOINT_EXPECTED_HOST_PORT")
    port = int(port_str)
    if not 1 <= port <= 65535:
        raise ProfileProtocolError("PHONE_ENDPOINT_PORT_OUT_OF_RANGE")
    if any(c.isspace() or ord(c) < 32 for c in host):
        raise ProfileProtocolError("PHONE_ENDPOINT_INVALID_HOST")
    return host, port


def encode(payload: dict[str, Any]) -> bytes:
    raw = (json.dumps(payload, separators=(",", ":"), allow_nan=False) +
           "\n").encode("utf-8")
    if len(raw) > MAX_FRAME:
        raise ProfileProtocolError("PROFILE_WIRE_FRAME_TOO_LARGE")
    return raw


def decode(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_FRAME or not raw.endswith(b"\n"):
        raise ProfileProtocolError("PROFILE_WIRE_FRAME_INCOMPLETE_OR_TOO_LARGE")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ProfileProtocolError("PROFILE_WIRE_INVALID_JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != WIRE_VERSION:
        raise ProfileProtocolError("PROFILE_WIRE_VERSION_OR_SHAPE_MISMATCH")
    return payload


def receive(sock: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while len(chunks) <= MAX_FRAME:
        data = sock.recv(min(65536, MAX_FRAME + 1 - len(chunks)))
        if not data:
            break
        if b"\n" in data:
            chunks.extend(data[:data.index(b"\n") + 1])
            break
        chunks.extend(data)
    return decode(bytes(chunks))


def call(endpoint: str, action: str, payload: dict[str, Any] | None = None,
         *, token: str = "", timeout: float = 300.0) -> dict[str, Any]:
    if action not in ("hello", "profile"):
        raise ProfileProtocolError("INVALID_PROFILE_ACTION")
    host, port = parse_endpoint(endpoint)
    message = {"version": WIRE_VERSION, "action": action,
               "token": token, **(payload or {})}
    data = encode(message)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(data)
            result = receive(sock)
    except (OSError, TimeoutError) as exc:
        raise ProfileProtocolError(
            "PHONE_PROFILE_RPC_IO_FAILED:" + str(exc)) from exc
    if result.get("status") == "ERROR":
        raise ProfileProtocolError(
            "PHONE_PROFILE_RPC_REJECTED:" + str(result.get("error", "UNKNOWN")))
    return result


def positive_int(value: object, field: str, *, lower: int = 1,
                 upper: int = 100000) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ProfileProtocolError("INVALID_PROFILE_" + field)
    return value


def finite_threshold(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or \
            not 0 <= value <= 100:
        raise ProfileProtocolError("INVALID_PROFILE_" + field)
    return float(value)


def validate_shape(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileProtocolError("INVALID_PROFILE_SHAPE")
    backend = value.get("backend")
    if backend not in ("N", "A"):
        raise ProfileProtocolError("PHONE_PROFILE_ONLY_N_OR_A")
    result: dict[str, Any] = {
        "backend": backend,
        "layers": positive_int(value.get("layers"), "LAYERS", upper=32),
        "chunk": positive_int(value.get("chunk"), "CHUNK", upper=4096),
        "kv_tokens": positive_int(value.get("kv_tokens"), "KV", upper=16384),
        "warmups": positive_int(value.get("warmups"), "WARMUPS",
                                lower=0, upper=20),
        "repeats": positive_int(value.get("repeats"), "REPEATS", upper=100),
        "threads": positive_int(value.get("threads"), "THREADS", upper=64),
        "verify_cpu": value.get("verify_cpu") is True or backend == "N",
        "max_relative_rmse": finite_threshold(
            value.get("max_relative_rmse"), "REL_RMSE"),
        "max_abs_error": finite_threshold(
            value.get("max_abs_error"), "MAX_ABS"),
    }
    if result["kv_tokens"] < result["chunk"] or \
            result["kv_tokens"] % result["chunk"] != 0:
        raise ProfileProtocolError("INVALID_CHUNK_KV_GEOMETRY")
    return result


def validate_sample(row: dict[str, Any], shape: dict[str, Any]) -> dict[str, str]:
    if not isinstance(row, dict):
        raise ProfileProtocolError("PHONE_PROFILE_MISSING_NATIVE_ROW")
    row = {str(k): str(v) for k, v in row.items()}
    try:
        if row["backend"] != shape["backend"] or \
                int(row["layers"]) != shape["layers"] or \
                int(row["chunk"]) != shape["chunk"] or \
                int(row["kv_tokens"]) != shape["kv_tokens"] or \
                int(row["repeats"]) != shape["repeats"] or \
                row["source"] != SOURCE:
            raise ProfileProtocolError("PHONE_PROFILE_RESULT_SHAPE_MISMATCH")
        for key in ("compute_median_ms", "prepare_median_ms",
                    "weight_upload_ms", "htp_precompile_ms",
                    "relative_rmse", "max_abs"):
            n = float(row[key])
            if not math.isfinite(n) or n < 0:
                raise ProfileProtocolError("INVALID_NATIVE_TIMING_OR_PARITY:" + key)
        if float(row["compute_median_ms"]) <= 0:
            raise ProfileProtocolError("ZERO_NATIVE_COMPUTE_TIME")
        if shape["backend"] == "N" and row["parity"] not in ("PASS", "FAIL"):
            raise ProfileProtocolError("MISSING_MANDATORY_HTP_PARITY")
        if row.get("parity") not in ("PASS", "FAIL", "NOT_REQUESTED"):
            raise ProfileProtocolError("INVALID_NATIVE_PARITY")
    except (ValueError, KeyError) as exc:
        raise ProfileProtocolError("PHONE_PROFILE_MALFORMED_NATIVE_ROW") from exc
    return row
