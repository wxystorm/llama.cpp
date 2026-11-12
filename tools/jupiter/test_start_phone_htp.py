#!/usr/bin/env python3
"""Regression tests for the Android/Termux Jupiter HTP launcher."""

from __future__ import annotations

import shutil
import subprocess
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
