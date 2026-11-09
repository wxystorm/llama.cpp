#!/usr/bin/env python3
"""Versioned, bounded native V2 N/A Profile control protocol.

Production control shares the existing ggml-rpc inference port 50052:
GGML RPC HELLO, then command 57 carrying one JSON profile frame.
Legacy standalone 50177 sidecar support is retained for isolated tests,
but is no longer used by the integrated sweep coordinator.
"""
from __future__ import annotations

import json
import math
import socket
import struct
from typing import Any

WIRE_VERSION = 1
MAX_FRAME = 256 * 1024
DEFAULT_PORT = 50177  # legacy standalone sidecar, no longer required
NATIVE_RPC_PORT = 50052
NATIVE_RPC_HELLO = 14
NATIVE_RPC_PROFILE = 57
NATIVE_RPC_VERSION = (4, 0, 35)
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


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ProfileProtocolError("PHONE_RPC50052_TRUNCATED_FRAME")
        buf.extend(chunk)
    return bytes(buf)


def _rpc50052_frame(sock: socket.socket) -> bytes:
    (size,) = struct.unpack("<Q", _recv_exact(sock, 8))
    if not 0 < size <= MAX_FRAME:
        raise ProfileProtocolError("PHONE_RPC50052_INVALID_FRAME_SIZE")
    return _recv_exact(sock, size)


def call_rpc50052(endpoint: str, data: bytes, *, timeout: float) -> dict[str, Any]:
    """Use ggml-rpc's HELLO + bounded command 57 on the *same* 50052 socket.

    Zero capability bytes explicitly opt out of RDMA so this Python client
    can use the ordinary TCP transport even if the Phone advertises RDMA.
    No new port, second network service, or user-supplied executable path.
    """
    host, port = parse_endpoint(endpoint)
    if port != NATIVE_RPC_PORT:
        raise ProfileProtocolError("EXPECTED_RPC50052_ENDPOINT")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(bytes([NATIVE_RPC_HELLO]) +
                         struct.pack("<Q", 24) + bytes(24))
            hello = _recv_exact(sock, 8)
            (hello_length,) = struct.unpack("<Q", hello)
            if hello_length != 28:
                raise ProfileProtocolError("PHONE_RPC50052_BAD_HELLO_SIZE")
            info = _recv_exact(sock, hello_length)
            if tuple(info[:3]) != NATIVE_RPC_VERSION:
                raise ProfileProtocolError(
                    "PHONE_RPC_VERSION_MISMATCH: expected=" +
                    ".".join(map(str, NATIVE_RPC_VERSION)) + " actual=" +
                    ".".join(map(str, info[:3])) +
                    "; rebuild PC and Phone from matching source")
            sock.sendall(bytes([NATIVE_RPC_PROFILE]) +
                         struct.pack("<Q", len(data)) + data)
            return decode(_rpc50052_frame(sock))
    except (OSError, TimeoutError) as exc:
        raise ProfileProtocolError(
            "PHONE_PROFILE_RPC50052_IO_FAILED:" + str(exc)) from exc


def call(endpoint: str, action: str, payload: dict[str, Any] | None = None,
         *, token: str = "", timeout: float = 300.0) -> dict[str, Any]:
    if action not in ("hello", "profile"):
        raise ProfileProtocolError("INVALID_PROFILE_ACTION")
    host, port = parse_endpoint(endpoint)
    message = {"version": WIRE_VERSION, "action": action,
               "token": token, **(payload or {})}
    data = encode(message)
    if port == NATIVE_RPC_PORT:
        result = call_rpc50052(endpoint, data, timeout=timeout)
        if result.get("status") == "ERROR":
            raise ProfileProtocolError(
                "PHONE_PROFILE_RPC_REJECTED:" +
                str(result.get("error", "UNKNOWN")))
        return result
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
        "bank_repeats": positive_int(
            value.get("bank_repeats", 0), "BANK_REPEATS", lower=0,
            upper=20),
        "host_stage_cap_mib": positive_int(
            value.get("host_stage_cap_mib", 1024),
            "HOST_STAGE_CAP_MIB", upper=8192),
    }
    if result["kv_tokens"] < result["chunk"]:
        raise ProfileProtocolError("INVALID_CHUNK_KV_GEOMETRY")
    if backend != "N" and result["bank_repeats"]:
        raise ProfileProtocolError("SAME_BANK_REPACK_ONLY_SUPPORTED_ON_N")
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
        if "token_begin" in row and int(row["token_begin"]) != (
                shape["kv_tokens"] - shape["chunk"]):
            raise ProfileProtocolError("WRONG_NATIVE_TOKEN_BEGIN")
        if "activation_bytes" in row and int(row["activation_bytes"]) <= 0:
            raise ProfileProtocolError("INVALID_NATIVE_ACTIVATION_BYTES")
        if shape.get("bank_repeats", 0):
            if float(row["same_bank_refill_median_ms"]) <= 0 or \
                    int(row["raw_weight_bytes"]) <= 0:
                raise ProfileProtocolError("MISSING_HTP_SAME_BANK_REFILL_SAMPLE")
        if shape["backend"] == "N" and row["parity"] not in ("PASS", "FAIL"):
            raise ProfileProtocolError("MISSING_MANDATORY_HTP_PARITY")
        if row.get("parity") not in ("PASS", "FAIL", "NOT_REQUESTED"):
            raise ProfileProtocolError("INVALID_NATIVE_PARITY")
    except (ValueError, KeyError) as exc:
        raise ProfileProtocolError("PHONE_PROFILE_MALFORMED_NATIVE_ROW") from exc
    return row
