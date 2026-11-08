"""No-GGUF/no-device tests for V2 real-Stage sample orchestration."""
import argparse
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from run_v2_stage_matrix import cases, csv_positive, run_matrix


class MatrixTests(unittest.TestCase):
    def test_omits_invalid_chunk_kv_shapes_without_faking_rows(self):
        self.assertEqual(cases("GC", [1], [32, 64], [32, 48]),
                         [("G", 1, 32, 32), ("C", 1, 32, 32)])
        self.assertEqual(csv_positive("4,2,4,1", "L"), [1, 2, 4])
        with self.assertRaises(ValueError):
            cases("X", [1], [32], [128])

    def test_real_result_only_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fake_model = base / "qwen.gguf"
            fake_model.write_bytes(b"identifiable original qwen GGUF placeholder")
            exe = Path(sys.executable)
            folder = base / "results"
            args = argparse.Namespace(
                exe=exe, model=fake_model, out=folder,
                backends="GC", layers="2", chunks="64", kvs="512",
                warmups=1, repeats=2, threads=2, timeout=1,
                verify_cpu=False, max_relative_rmse=0.02, max_abs_error=0.5,
                update=False, dry_run=False)
            def completed(command, **_):
                backend = command[command.index("--backend") + 1]
                repeats = command[command.index("--repeats") + 1]
                return mock.Mock(
                    returncode=0, stderr="", stdout=(
                        "backend,layers,chunk,kv_tokens,repeats,"
                        "graph_build_ms,weight_upload_ms,htp_precompile_ms,"
                        "prepare_median_ms,compute_median_ms,weight_bytes,"
                        "kv_bytes,graph_workspace_bytes,relative_rmse,"
                        "max_abs,parity,source\n"
                        f"{backend},2,64,512,{repeats},1.0,2.0,0,0.1,3.0,"
                        "100,100,100,0,0,NOT_REQUESTED,"
                        "NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))
            with mock.patch("run_v2_stage_matrix.host_device_id",
                            return_value="h" * 64), \
                 mock.patch("subprocess.run", side_effect=completed) as run:
                self.assertEqual(run_matrix(args), 0)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run_matrix(args), 0)
                self.assertEqual(run.call_count, 2)  # verified resume hit
                args.repeats = 3
                self.assertEqual(run_matrix(args), 0)
                self.assertEqual(run.call_count, 4)
            self.assertTrue((folder / "stage-profile-v2.csv").exists())
            self.assertTrue((folder / "G-L2-X64-KV512.json").exists())

    def test_npu_never_accepts_missing_parity(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model = base / "m.gguf"
            model.write_bytes(b"fake model")
            args = argparse.Namespace(
                exe=Path(sys.executable), model=model,
                out=base / "out", backends="N", layers="2", chunks="64",
                kvs="512", warmups=0, repeats=1, threads=2,
                timeout=1, verify_cpu=False,
                max_relative_rmse=0.02, max_abs_error=0.5,
                update=False, dry_run=False)
            response = mock.Mock(returncode=0, stderr="", stdout=(
                "backend,layers,chunk,kv_tokens,repeats,parity,source\n"
                "N,2,64,512,1,NOT_REQUESTED,"
                "NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))
            with mock.patch("run_v2_stage_matrix.host_device_id",
                            return_value="h" * 64), \
                 mock.patch("subprocess.run", return_value=response) as run:
                self.assertEqual(run_matrix(args), 1)
                self.assertIn("--verify-cpu", run.call_args.args[0])
            self.assertEqual(
                __import__("json").loads(
                    (args.out / "N-L2-X64-KV512.json").read_text("utf-8")
                )["status"], "FAIL")
            self.assertTrue((args.out / "N-L2-X64-KV512.failed.json").exists())

    def test_pc_and_phone_gcna_dispatch_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model = base / "m.gguf"
            model.write_bytes(b"qwen3 fake for matching digest")
            args = argparse.Namespace(
                exe=Path(sys.executable), model=model, out=base / "profile",
                phone="127.0.0.1:50177", phone_token="", phone_timeout=10,
                backends="GCNA", layers="2", chunks="64", kvs="512",
                warmups=1, repeats=3, threads=4, timeout=2,
                verify_cpu=False, max_relative_rmse=0.02, max_abs_error=0.5,
                update=False, dry_run=False)
            info = {"status": "OK", "exe_sha256": "e" * 64,
                    "device_id": "d" * 64}
            def local_response(command, **_):
                backend = command[command.index("--backend") + 1]
                return mock.Mock(returncode=0, stderr="", stdout=(
                    "backend,layers,chunk,kv_tokens,repeats,"
                    "graph_build_ms,weight_upload_ms,htp_precompile_ms,"
                    "prepare_median_ms,compute_median_ms,weight_bytes,"
                    "kv_bytes,graph_workspace_bytes,relative_rmse,"
                    "max_abs,parity,source\n"
                    f"{backend},2,64,512,3,1,2,0,0.1,3,100,100,100,0,0,"
                    "NOT_REQUESTED,NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))
            def remote_response(endpoint, token, timeout, sha, phone, shape):
                row = {"backend": shape["backend"], "layers": "2",
                       "chunk": "64", "kv_tokens": "512", "repeats": "3",
                       "graph_build_ms": "1", "weight_upload_ms": "1",
                       "htp_precompile_ms": "0.04",
                       "prepare_median_ms": "0.1",
                       "compute_median_ms": "2.0", "weight_bytes": "100",
                       "kv_bytes": "100", "graph_workspace_bytes": "100",
                       "relative_rmse": "0.01", "max_abs": "0.1",
                       "parity": ("PASS" if shape["backend"] == "N"
                                  else "NOT_REQUESTED"),
                       "source": "NATIVE_SINGLE_BACKEND_GGUF_PREFIX"}
                return "PASS", row, ""
            with mock.patch("run_v2_stage_matrix.host_device_id",
                            return_value="h" * 64), \
                 mock.patch("run_v2_stage_matrix.hello_phone", return_value=info), \
                 mock.patch("run_v2_stage_matrix.phone_remote",
                            side_effect=remote_response) as remote, \
                 mock.patch("subprocess.run", side_effect=local_response) as local:
                self.assertEqual(run_matrix(args), 0)
                self.assertEqual(remote.call_count, 2)
                self.assertEqual(local.call_count, 2)
                self.assertEqual(run_matrix(args), 0)
                self.assertEqual(remote.call_count, 2)
                self.assertEqual(local.call_count, 2)
            import csv
            with (args.out / "stage-profile-v2.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["backend"] for r in rows}, set("GCNA"))
            self.assertEqual(len(rows), 4)
            import json
            unified = json.loads((args.out / "profile.json").read_text())
            self.assertEqual(unified["counts"]["pass"], 4)
            self.assertTrue(unified["compute_complete"])
            self.assertFalse(unified["planner_ready"])
            self.assertEqual(
                {item["key"]["backend"] for item in unified["compute_samples"]},
                set("GCNA"))

    def test_phone_unavailable_does_not_block_pc_gc_and_will_resume(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model = base / "model.gguf"
            model.write_bytes(b"tiny qwen model identifier")
            args = argparse.Namespace(
                exe=Path(sys.executable), model=model, out=base / "out",
                phone="127.0.0.1:50177", phone_token="", phone_timeout=2,
                backends="GCNA", layers="2", chunks="64", kvs="512",
                warmups=1, repeats=2, threads=2, timeout=2,
                verify_cpu=False, max_relative_rmse=0.02, max_abs_error=0.5,
                update=False, dry_run=False)
            def fake_native(command, **_):
                backend = command[command.index("--backend") + 1]
                return mock.Mock(
                    returncode=0, stderr="", stdout=(
                        "backend,layers,chunk,kv_tokens,repeats,graph_build_ms,"
                        "weight_upload_ms,htp_precompile_ms,prepare_median_ms,"
                        "compute_median_ms,weight_bytes,kv_bytes,"
                        "graph_workspace_bytes,relative_rmse,max_abs,parity,source\n"
                        f"{backend},2,64,512,2,1,1,0,0.1,2,100,100,100,"
                        "0,0,NOT_REQUESTED,NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))
            with mock.patch("run_v2_stage_matrix.host_device_id",
                            return_value="h" * 64), \
                 mock.patch("run_v2_stage_matrix.hello_phone",
                            side_effect=RuntimeError("phone offline")), \
                 mock.patch("subprocess.run", side_effect=fake_native) as native:
                self.assertEqual(run_matrix(args), 1)
                self.assertEqual(native.call_count, 2)
            import json
            self.assertEqual(
                json.loads((args.out / "G-L2-X64-KV512.json").read_text())["status"],
                "PASS")
            self.assertEqual(
                json.loads((args.out / "C-L2-X64-KV512.json").read_text())["status"],
                "PASS")
            self.assertEqual(
                json.loads((args.out / "N-L2-X64-KV512.json").read_text())["status"],
                "FAIL")
            self.assertEqual(
                json.loads((args.out / "A-L2-X64-KV512.json").read_text())["status"],
                "FAIL")

    def test_remote_n4_parity_fail_never_enters_pass_csv(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model = base / "m.gguf"
            model.write_bytes(b"qwen3 fake")
            args = argparse.Namespace(
                exe=None, model=model, out=base / "out",
                phone="127.0.0.1:50177", phone_token="", phone_timeout=10,
                backends="N", layers="4", chunks="64", kvs="512",
                warmups=1, repeats=3, threads=4, timeout=2,
                verify_cpu=False, max_relative_rmse=0.02, max_abs_error=0.5,
                update=False, dry_run=False)
            info = {"status": "OK", "exe_sha256": "e" * 64,
                    "device_id": "d" * 64}
            sample = {"backend": "N", "layers": "4", "chunk": "64",
                      "kv_tokens": "512", "repeats": "3",
                      "relative_rmse": "0.071", "max_abs": "2.289",
                      "compute_median_ms": "41.2",
                      "parity": "FAIL",
                      "source": "NATIVE_SINGLE_BACKEND_GGUF_PREFIX"}
            with mock.patch("run_v2_stage_matrix.hello_phone", return_value=info), \
                 mock.patch("run_v2_stage_matrix.phone_remote",
                            return_value=("PARITY_FAIL", sample, "")):
                self.assertEqual(run_matrix(args), 1)
            self.assertFalse((args.out / "stage-profile-v2.csv").exists())
            self.assertTrue(
                (args.out / "stage-profile-v2-parity-fail.csv").exists())
            import json
            record = json.loads((args.out / "N-L4-X64-KV512.json").read_text())
            self.assertEqual(record["status"], "PARITY_FAIL")
            self.assertEqual(record["sample"]["compute_median_ms"], "41.2")
            unified = json.loads((args.out / "profile.json").read_text())
            self.assertFalse(unified["compute_complete"])
            self.assertFalse(unified["planner_ready"])
            self.assertEqual(unified["counts"]["pass"], 0)
            self.assertEqual(unified["counts"]["parity_fail"], 1)
            self.assertEqual(unified["parity_fail_samples"][0]["status"],
                             "PARITY_FAIL")


if __name__ == "__main__":
    unittest.main()
