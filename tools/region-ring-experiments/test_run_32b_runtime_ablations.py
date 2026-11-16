"""Offline regression checks for the 32B Runtime ablation wrapper.

Run from the repository root:
  python -m unittest discover -s tools/region-ring-experiments -p "test_run_32b_runtime_ablations.py" -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "run_32b_runtime_ablations.py"

STUB = """
from types import SimpleNamespace
LONG_HTP_KV_ANCHORS = (2048, 3072)
def verify_independent_profile(source, kv): return 1
def get_args(argv=None):
    assert '--auto-plan' in argv and '--htp-only' in argv
    return SimpleNamespace(auto_plan=True, htp_only=True, locked_layout=None)
def main(argv=None):
    args = get_args(argv)
    assert args.locked_layout
    assert '--env' in argv
    assert 'LLAMA_REGION_RING_GPU_BANK_COUNT=2' in argv
    print('[STUB] ' + args.locked_layout)
    return 0
"""


class RuntimeAblationTests(unittest.TestCase):
    def test_topology_optin_present(self):
        header = (ROOT / "src" / "llama-region-ring.h").read_text(encoding="utf-8")
        self.assertIn("LLAMA_REGION_RING_ALLOW_ADJACENT_GPU", header)
        self.assertIn("widths = {8, 8, 8, 8, 24, 8}", header)
        self.assertIn("if (exact && next_layer == n_layer) { return true; }", header)

    def fake_checkout(self, root: Path) -> Path:
        folder = root / "tools" / "region-ring-experiments"
        folder.mkdir(parents=True)
        shutil.copy2(RUNNER, folder / RUNNER.name)
        (folder / "run_32b_all40.py").write_text(STUB, encoding="utf-8")
        return folder

    def test_dry_run_each_case(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = self.fake_checkout(root)
            cmd = [
                sys.executable, str(folder / RUNNER.name),
                "--exe", str(root / "llama.exe"),
                "--model", str(root / "model.gguf"),
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(root / "prompts"),
                "--out", str(root / "out"),
                "--cases", "all", "--dry-run", "--smoke",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[STUB] G8,G8,G8,G8,P24,G8", result.stdout)
            self.assertIn("[STUB] G8,P6,G8,P6,G8,P6,G8,P6,G8", result.stdout)
            self.assertIn("prefetch=0", result.stdout)
            self.assertIn("drain_first=0", result.stdout)

    def test_audit_rejects_wrong_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = self.fake_checkout(root)
            sys.path.insert(0, str(folder))
            try:
                spec = importlib.util.spec_from_file_location("test_ablation_runner", folder / RUNNER.name)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                case = root / "out"
                case.mkdir()
                result_file = case / "result.json"
                result_file.write_text(json.dumps({"env_overrides": {}}), encoding="utf-8")
                with (case / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "group", "prompt", "status", "plan_layout", "plan_source",
                        "plan_applied", "gpu_bank_streaming", "result_file",
                    ])
                    writer.writeheader()
                    writer.writerow({
                        "group": 128, "prompt": "prompt_128_01.txt",
                        "status": "PASS", "plan_layout": "G32,P24,G8",
                        "plan_source": "forced", "plan_applied": "True",
                        "gpu_bank_streaming": "False", "result_file": str(result_file),
                    })
                self.assertTrue(module.audit_case(case, "no_multicycle", {128}))
            finally:
                sys.path.remove(str(folder))


if __name__ == "__main__":
    unittest.main()
