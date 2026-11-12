#!/usr/bin/env python3
"""Regression tests for the Android/Termux Jupiter HTP launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("start_phone_htp.sh")


class StartPhoneHtpScriptTests(unittest.TestCase):
    def test_script_exists_and_uses_unix_line_endings(self):
        data = SCRIPT.read_bytes()
        self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", data, "Termux launcher must use LF, not CRLF")
        self.assertTrue(data.endswith(b"\n"))

    def test_adsp_library_path_is_single_dsp_search_directory(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('export ADSP_LIBRARY_PATH="$root/lib"', source)
        self.assertNotIn('export ADSP_LIBRARY_PATH="$root/lib:$root/bin:', source)

    def test_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed on this test host")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_arguments_print_usage(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed on this test host")
        result = subprocess.run(
            [bash, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_device_list_matches_session_count(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed on this test host")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            (root / "lib" / "libggml-htp-v79.so").touch()
            model = root / "test.gguf"
            model.touch()
            server = root / "bin" / "ggml-rpc-server"
            server.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'MOCK_NDEV=%s\\n' \"$GGML_HEXAGON_NDEV\"\n"
                "printf 'MOCK_ARGS=%s\\n' \"$*\"\n",
                encoding="utf-8",
            )
            server.chmod(0o755)
            base_env = os.environ.copy()
            base_env["LLAMA_JUPITER_PHONE_BIN_ROOT"] = str(root)
            base_env.pop("GGML_HEXAGON_NDEV", None)
            base_env.pop("GGML_HEXAGON_MBUF", None)
            base_env.pop("GGML_HEXAGON_VMEM", None)
            for sessions in (1, 2, 3, 4):
                with self.subTest(sessions=sessions):
                    env = dict(base_env, GGML_HEXAGON_NDEV=str(sessions))
                    completed = subprocess.run(
                        [bash, str(SCRIPT), str(model), "10140"],
                        env=env, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    devices = ",".join(f"HTP{i}" for i in range(sessions))
                    self.assertIn(f"MOCK_NDEV={sessions}", completed.stdout)
                    self.assertIn(f"--device {devices} --host", completed.stdout)
                    self.assertIn(f"devices={devices} sessions={sessions}", completed.stdout)
            default_run = subprocess.run(
                [bash, str(SCRIPT), str(model), "10140"],
                env=base_env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(default_run.returncode, 0, default_run.stderr)
            self.assertIn("MOCK_NDEV=4", default_run.stdout)
            self.assertIn("--device HTP0,HTP1,HTP2,HTP3 --host", default_run.stdout)
            for invalid in ("0", "5", "xyz"):
                with self.subTest(invalid=invalid):
                    bad_run = subprocess.run(
                        [bash, str(SCRIPT), str(model), "10140"],
                        env=dict(base_env, GGML_HEXAGON_NDEV=invalid),
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(bad_run.returncode, 2)
                    self.assertIn("GGML_HEXAGON_NDEV must", bad_run.stderr)

    def test_invalid_port_is_rejected_before_model_lookup(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed on this test host")
        result = subprocess.run(
            [bash, str(SCRIPT), "/nonexistent/model.gguf", "10x40"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid RPC port", result.stderr)


if __name__ == "__main__":
    unittest.main()
