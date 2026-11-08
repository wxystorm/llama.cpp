#!/usr/bin/env python3
"""No-device regression tests for V2 Phone N/A profiling control plane."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from v2_phone_profile_server import Server, Service
from v2_profile_wire import (
    ProfileProtocolError, WIRE_VERSION, call, decode, encode,
    validate_shape, validate_sample,
)


def shape(backend: str = "N", n: int = 2) -> dict:
    return {
        "backend": backend, "layers": n, "chunk": 64, "kv_tokens": 512,
        "warmups": 1, "repeats": 3, "threads": 4,
        "verify_cpu": backend == "N",
        "max_relative_rmse": 0.02, "max_abs_error": 0.5,
    }


def native_response(backend: str = "N", n: int = 2,
                    status: str = "PASS"):
    parity = status if status == "FAIL" else (
        "PASS" if backend == "N" else "NOT_REQUESTED")
    relative_rmse = 0.071 if status == "FAIL" else 0.010
    max_abs = 2.289 if status == "FAIL" else 0.140
    return SimpleNamespace(
        returncode=2 if status == "FAIL" else 0,
        stderr="HTP test diagnostics",
        stdout=("backend,layers,chunk,kv_tokens,repeats,graph_build_ms,"
                "weight_upload_ms,htp_precompile_ms,prepare_median_ms,"
                "compute_median_ms,weight_bytes,kv_bytes,"
                "graph_workspace_bytes,relative_rmse,max_abs,parity,source\n"
                f"{backend},{n},64,512,3,13.0,32.0,0.04,0.11,20.5,"
                f"4096,512,512,{relative_rmse},{max_abs},{parity},"
                "NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))


class ProfileWireTests(unittest.TestCase):
    def test_wire_rejects_bad_version_and_unbounded(self):
        with self.assertRaises(ProfileProtocolError):
            decode(b'{"version":2}\n')
        with self.assertRaises(ProfileProtocolError):
            decode(b"{}")
        with self.assertRaises(ProfileProtocolError):
            encode({"version": 1, "blob": "z" * (256 * 1024)})
        self.assertEqual(decode(encode({"version": WIRE_VERSION}))["version"],
                         WIRE_VERSION)

    def test_rejects_bad_phone_profile_shapes(self):
        self.assertTrue(validate_shape(shape())["verify_cpu"])
        p = shape()
        p["backend"] = "C"
        with self.assertRaises(ProfileProtocolError):
            validate_shape(p)
        p = shape()
        p["chunk"] = 100
        with self.assertRaises(ProfileProtocolError):
            validate_shape(p)
        p = shape()
        p["threads"] = "8"
        with self.assertRaises(ProfileProtocolError):
            validate_shape(p)
        p = shape("N")
        p["verify_cpu"] = False
        self.assertTrue(validate_shape(p)["verify_cpu"])


class PhoneServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "tiny.gguf"
        path.write_bytes(b"qwen3 tiny fixture")
        self.service = Service(path, Path(sys.executable), token="securelongtoken123",
                               timeout=5)

    def test_rejects_non_model_prefix_and_wrong_token_without_exec(self):
        data = {"version": WIRE_VERSION, "action": "profile",
                "token": "securelongtoken123",
                "model_sha256": self.service.model_sha256,
                "shape": shape()}
        with mock.patch("v2_phone_profile_server.subprocess.run") as process:
            bad = dict(data)
            bad["model_sha256"] = "0" * 64
            self.assertEqual(self.service.dispatch(bad)["status"], "ERROR")
            bad = dict(data)
            bad["shape"] = {**shape(), "backend": "G"}
            self.assertEqual(self.service.dispatch(bad)["status"], "ERROR")
            bad = dict(data)
            bad["token"] = "invalid"
            self.assertEqual(self.service.dispatch(bad)["status"], "ERROR")
            self.assertEqual(process.call_count, 0)

    def test_cpu_parity_enforced_and_raw_exec_no_shell(self):
        request = {"version": WIRE_VERSION, "action": "profile",
                   "token": "securelongtoken123",
                   "model_sha256": self.service.model_sha256,
                   "shape": {**shape(), "verify_cpu": False}}
        with mock.patch("v2_phone_profile_server.subprocess.run",
                        return_value=native_response()) as process:
            done = self.service.dispatch(request)
            self.assertEqual(done["status"], "PASS")
            self.assertEqual(done["sample"]["parity"], "PASS")
            cmd = process.call_args.args[0]
            self.assertIn("--verify-cpu", cmd)
            self.assertEqual(cmd[cmd.index("--model") + 1], str(self.service.model))
            self.assertFalse(process.call_args.kwargs.get("shell", False))

    def test_preserves_real_n4_timing_but_not_as_pass(self):
        request = {"version": WIRE_VERSION, "action": "profile",
                   "token": "securelongtoken123",
                   "model_sha256": self.service.model_sha256,
                   "shape": shape(n=4)}
        with mock.patch("v2_phone_profile_server.subprocess.run",
                        return_value=native_response(n=4, status="FAIL")):
            done = self.service.dispatch(request)
        self.assertEqual(done["status"], "PARITY_FAIL")
        self.assertEqual(done["sample"]["compute_median_ms"], "20.5")

    def test_phone_loopback_request_roundtrip(self):
        with Server(("127.0.0.1", 0), self.service) as server:
            port = server.server_address[1]
            worker = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01},
                daemon=True)
            worker.start()
            try:
                addr = f"127.0.0.1:{port}"
                hello = call(addr, "hello", token="securelongtoken123",
                             timeout=2)
                self.assertEqual(hello["model_sha256"],
                                 self.service.model_sha256)
                with mock.patch("v2_phone_profile_server.subprocess.run",
                                return_value=native_response()):
                    response = call(addr, "profile", {
                        "model_sha256": self.service.model_sha256,
                        "shape": shape(),
                    }, token="securelongtoken123", timeout=2)
                self.assertEqual(response["status"], "PASS")
                with self.assertRaises(ProfileProtocolError):
                    call(addr, "hello", token="wrong", timeout=2)
            finally:
                server.shutdown()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
