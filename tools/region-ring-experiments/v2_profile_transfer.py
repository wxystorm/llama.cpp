#!/usr/bin/env python3
"""Bounded raw-TCP diagnostic benchmark for the V2 Phone Profile sidecar.

Unlike ordinary V2 JSON control messages, this sends actual binary payload
bytes in either direction. Its timings measure only the RECEIVING side's
TCP stream, not production ggml-rpc tensor serialization or Bank movement.
"""
from __future__ import annotations

import math
import os
import socket
import statistics
import time
from typing import Any, BinaryIO

from v2_profile_wire import (
    MAX_FRAME, ProfileProtocolError, WIRE_VERSION, decode, encode,
    parse_endpoint, positive_int,
)

MAX_BYTES = 32 * 1024 * 1024
BLOCK = os.urandom(64 * 1024)
SOURCE = "V2_TCP_SIDECAR_RX_STREAM"


def validate_transfer(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ProfileProtocolError("INVALID_TRANSFER_REQUEST")
    direction = request.get("direction")
    if direction not in ("PC_TO_PHONE", "PHONE_TO_PC"):
        raise ProfileProtocolError("INVALID_TRANSFER_DIRECTION")
    return {
        "direction": direction,
        "size_bytes": positive_int(request.get("size_bytes"), "TRANSFER_BYTES",
                                   upper=MAX_BYTES),
        "warmups": positive_int(request.get("warmups"), "TRANSFER_WARMUPS",
                                lower=0, upper=3),
        "repeats": positive_int(request.get("repeats"), "TRANSFER_REPEATS",
                                upper=16),
    }


def read_frame(reader: BinaryIO) -> dict[str, Any]:
    return decode(reader.readline(MAX_FRAME + 1))


def send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode({"version": WIRE_VERSION, **payload}))


def send_payload(sock: socket.socket, size: int) -> None:
    remaining = size
    while remaining:
        length = min(len(BLOCK), remaining)
        sock.sendall(memoryview(BLOCK)[:length])
        remaining -= length


def receive_payload(reader: BinaryIO, size: int) -> float:
    # read EXACTLY size bytes; Python socket buffering retains any later
    # framed messages without discarding bytes at frame boundaries.
    started = time.perf_counter()
    remaining = size
    while remaining:
        data = reader.read(min(64 * 1024, remaining))
        if not data:
            raise ProfileProtocolError("TRANSFER_PAYLOAD_TRUNCATED")
        remaining -= len(data)
    ms = (time.perf_counter() - started) * 1000
    return max(ms, 1e-9)


def phone_serve_transfer(request: dict[str, Any],
                         reader: BinaryIO, sock: socket.socket) -> None:
    cfg = validate_transfer(request)
    send_frame(sock, {"status": "READY", "action": "transfer", **cfg})
    total = cfg["warmups"] + cfg["repeats"]
    for sequence in range(total):
        if cfg["direction"] == "PC_TO_PHONE":
            recv_ms = receive_payload(reader, cfg["size_bytes"])
            send_frame(sock, {"status": "ACK", "iteration": sequence,
                              "receiver_ms": recv_ms})
        else:
            send_payload(sock, cfg["size_bytes"])
            ack = read_frame(reader)
            if ack.get("status") != "ACK" or \
                    ack.get("iteration") != sequence:
                raise ProfileProtocolError("TRANSFER_PC_ACK_MISMATCH")
    send_frame(sock, {"status": "DONE", "action": "transfer",
                      "iterations": total})


def transfer_call(endpoint: str, *, token: str, model_sha256: str,
                  direction: str, size_bytes: int, warmups: int = 1,
                  repeats: int = 5, timeout: float = 300.0) -> dict[str, Any]:
    shape = validate_transfer({"direction": direction, "size_bytes": size_bytes,
                               "warmups": warmups, "repeats": repeats})
    host, port = parse_endpoint(endpoint)
    request = {"version": WIRE_VERSION, "action": "transfer",
               "token": token, "model_sha256": model_sha256, **shape}
    samples: list[float] = []
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            with sock.makefile("rb") as reader:
                sock.sendall(encode(request))
                ready = read_frame(reader)
                if ready.get("status") == "ERROR":
                    raise ProfileProtocolError(
                        "TRANSFER_PHONE_REJECTED:" + str(ready.get("error")))
                if ready.get("status") != "READY" or \
                        ready.get("action") != "transfer" or \
                        any(ready.get(key) != value for key, value in shape.items()):
                    raise ProfileProtocolError("TRANSFER_PHONE_HANDSHAKE_MISMATCH")
                total = shape["warmups"] + shape["repeats"]
                for sequence in range(total):
                    if direction == "PC_TO_PHONE":
                        send_payload(sock, size_bytes)
                        ack = read_frame(reader)
                        if ack.get("status") != "ACK" or \
                                ack.get("iteration") != sequence:
                            raise ProfileProtocolError("TRANSFER_PHONE_ACK_MISMATCH")
                        recv_ms = float(ack.get("receiver_ms", 0))
                    else:
                        recv_ms = receive_payload(reader, size_bytes)
                        send_frame(sock, {"status": "ACK", "iteration": sequence})
                    if not math.isfinite(recv_ms) or recv_ms <= 0:
                        raise ProfileProtocolError("TRANSFER_INVALID_RECEIVER_TIME")
                    if sequence >= shape["warmups"]:
                        samples.append(recv_ms)
                final = read_frame(reader)
                if final.get("status") != "DONE" or \
                        final.get("action") != "transfer" or \
                        final.get("iterations") != total:
                    raise ProfileProtocolError("TRANSFER_INCOMPLETE_FINAL_RESPONSE")
    except (OSError, TimeoutError) as exc:
        raise ProfileProtocolError("TRANSFER_TCP_IO_FAILED:" + str(exc)) from exc
    samples.sort()
    median_ms = statistics.median(samples)
    p90_ms = samples[max(0, math.ceil(0.9 * len(samples)) - 1)]
    return {**shape, "receiver_median_ms": round(median_ms, 6),
            "receiver_p90_ms": round(p90_ms, 6),
            "receiver_min_ms": round(samples[0], 6),
            "receiver_max_ms": round(samples[-1], 6),
            "throughput_mib_s": round(
                size_bytes * 1000 / (median_ms * 1024 * 1024), 6),
            "source": SOURCE, "planner_eligible": False,
            "measurement": "unidirectional_stream_receiver_wall"}
