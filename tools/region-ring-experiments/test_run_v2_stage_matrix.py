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
                        "backend,layers,chunk,kv_tokens,repeats,parity,source\n"
                        f"{backend},2,64,512,{repeats},NOT_REQUESTED,"
                        "NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n"))
            with mock.patch("subprocess.run", side_effect=completed) as run:
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
            with mock.patch("subprocess.run", return_value=response) as run:
                self.assertEqual(run_matrix(args), 1)
                self.assertIn("--verify-cpu", run.call_args.args[0])
            self.assertFalse((args.out / "N-L2-X64-KV512.json").exists())
            self.assertTrue((args.out / "N-L2-X64-KV512.failed.json").exists())


if __name__ == "__main__":
    unittest.main()
