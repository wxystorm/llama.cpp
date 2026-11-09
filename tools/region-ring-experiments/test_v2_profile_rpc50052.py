"""Mock GGML RPC server tests for native V2 profile control on port 50052.

No model, Android device, or native libraries needed. The fixture mirrors the
HELLO framing and new command 57, but binds to an ephemeral port locally.
"""
from __future__ import annotations

import socket
import struct
import threading
import unittest
from unittest.mock import patch

import v2_profile_wire as wire


def recv_exact(sock: socket.socket, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        chunk = sock.recv(size - len(out))
        if not chunk:
            raise EOFError("MOCK_EARLY_CLOSE")
        out.extend(chunk)
    return bytes(out)


class MockRpc:
    def __init__(self, payload: dict, *, version=(4, 0, 36),
                 oversized=False):
        self.payload = payload
        self.version = version
        self.oversized = oversized
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.port = self.socket.getsockname()[1]
        self.error = None
        self.received = None
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.thread.join(timeout=2)
        self.socket.close()
        if self.error:
            raise self.error

    def serve(self):
        try:
            conn, _ = self.socket.accept()
            with conn:
                conn.settimeout(3)
                assert recv_exact(conn, 1) == bytes([14])
                assert struct.unpack("<Q", recv_exact(conn, 8)) == (24,)
                assert recv_exact(conn, 24) == bytes(24)
                hello = bytes(self.version) + bytes(25)
                conn.sendall(struct.pack("<Q", len(hello)) + hello)
                if self.version != wire.NATIVE_RPC_VERSION:
                    return
                assert recv_exact(conn, 1) == bytes([57])
                (size,) = struct.unpack("<Q", recv_exact(conn, 8))
                self.received = wire.decode(recv_exact(conn, size))
                raw = wire.encode(self.payload)
                if self.oversized:
                    conn.sendall(struct.pack("<Q", wire.MAX_FRAME + 1))
                else:
                    conn.sendall(struct.pack("<Q", len(raw)) + raw)
        except Exception as exc:
            self.error = exc


class NativeRpcPortTests(unittest.TestCase):
    def rpc(self, server, action="hello"):
        with patch.object(wire, "NATIVE_RPC_PORT", server.port):
            return wire.call(f"127.0.0.1:{server.port}", action)

    def test_hello_and_single_socket_dispatch(self):
        response = {"version": 1, "status": "OK",
                    "model_sha256": "a" * 64,
                    "transport": "V2_PHONE_PROFILE_RPC50052"}
        with MockRpc(response) as mock:
            self.assertEqual(self.rpc(mock), response)
        self.assertEqual(mock.received["action"], "hello")

    def test_rejected_profile_propagates_failure(self):
        with MockRpc({"version": 1, "status": "ERROR",
                      "error": "PHONE_PC_GGUF_SHA256_MISMATCH"}) as mock:
            with self.assertRaisesRegex(
                    wire.ProfileProtocolError, "PHONE_PC_GGUF_SHA256_MISMATCH"):
                self.rpc(mock, "profile")
        self.assertEqual(mock.received["action"], "profile")

    def test_version_mismatch_rejected_before_command(self):
        with MockRpc({"version": 1, "status": "OK"}, version=(4, 0, 33)) as mock:
            with self.assertRaisesRegex(
                    wire.ProfileProtocolError, "PHONE_RPC_VERSION_MISMATCH"):
                self.rpc(mock)

    def test_oversized_response_rejected(self):
        with MockRpc({"version": 1, "status": "OK"}, oversized=True) as mock:
            with self.assertRaisesRegex(
                    wire.ProfileProtocolError, "INVALID_FRAME_SIZE"):
                self.rpc(mock)


if __name__ == "__main__":
    unittest.main()
