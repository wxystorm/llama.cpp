#!/usr/bin/env python3
"""Offline runner tests: no real GGUF, RPC server or GPU required."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name('run_32b_all40.py')
MOCK = '''#!/usr/bin/env python3
import json,os,pathlib,sys,time
args=sys.argv
prompt=pathlib.Path(args[args.index('-f')+1]).name
config=pathlib.Path(os.environ['LLAMA_REGION_RING_METRICS_FILE'])
mode=(config.parent.parent.parent.parent.parent / 'behavior.txt')
behavior=mode.read_text() if mode.exists() else ''
once=(config.parent.parent.parent.parent.parent / ('attempted_'+prompt))
if behavior=='fail_once' and prompt=='prompt_128_01.txt' and not once.exists():
    once.write_text('1'); sys.exit(1)
if behavior=='no_metrics' and prompt=='prompt_128_01.txt': sys.exit(0)
if behavior=='wrong_layout' and prompt=='prompt_128_01.txt':
  with config.open('w') as f:
    for x in [{'type':'measurement_start'}, {'type':'region_ring','complete':1,'status':0,'layout':'G40,P24'}, {'type':'perf','prefill_tokens':167,'prefill_ms':42}]:
      f.write(json.dumps(x)+'\\n')
  sys.exit(0)
if behavior=='timeout' and prompt=='prompt_128_01.txt': time.sleep(2)
with config.open('w') as f:
  for x in [{'type':'measurement_start'}, {'type':'region_ring','complete':1,'status':0,'layout':'G8,P6,G8,P6,G8,P6,G8,P6,G8'}, {'type':'perf','prefill_tokens':167,'prefill_ms':42,'prefill_tps':3976.2}]:
    f.write(json.dumps(x)+'\\n')
'''


@unittest.skipIf(os.name == 'nt', 'mock #! executable uses POSIX shebang')
class All40Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.out = root / 'results'
        prompts = root / 'prompts'
        prompts.mkdir()
        self.prompts = prompts
        for group in (128, 256, 512, 1024):
            for i in range(1, 11):
                (prompts / f'prompt_{group}_{i:02}.txt').write_text(f'content {group} {i}')
        with (prompts / 'manifest.csv').open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['file', 'tokens_tokenizer', 'length_bin'])
            writer.writerow(['prompt_128_01.txt', 167, '128-255'])
            writer.writerow(['prompt_1024_10.txt', 1855, '1024-2000'])
        self.exe = root / 'mock_engine.py'
        self.exe.write_text(MOCK)
        self.exe.chmod(0o755)
        self.model = root / 'model.gguf'
        self.model.write_bytes(b'fake model')
        self.root = root
        self.common = ['--exe', str(self.exe), '--model', str(self.model),
                       '--rpc', '127.0.0.1:50052', '--prompt-dir', str(prompts),
                       '--repo-root', str(root), '--out', str(self.out),
                       '--max-retries', '0']

    def invoke(self, *options):
        return subprocess.run([sys.executable, str(RUNNER), *self.common, *options],
                              capture_output=True, text=True, timeout=30)

    def load(self, name):
        trials = sorted((self.out/'runs'/Path(name).stem).glob('attempt-*/result.json'))
        return [json.loads(path.read_text()) for path in trials]

    def test_fail_then_resume_without_repeating_pass(self):
        (self.root / 'behavior.txt').write_text('fail_once')
        self.assertEqual(self.invoke('--limit','2').returncode, 1)
        self.assertEqual([r['status'] for r in self.load('prompt_128_01.txt')], ['FAIL'])
        self.assertEqual(self.invoke('--limit','2').returncode, 0)
        self.assertEqual([r['status'] for r in self.load('prompt_128_01.txt')], ['FAIL','PASS'])
        self.assertEqual(len(self.load('prompt_128_02.txt')), 1)
        self.assertEqual(self.invoke('--limit','3').returncode, 0)
        self.assertEqual(len(self.load('prompt_128_02.txt')), 1)
        counts = json.loads((self.out/'summary.json').read_text())['counts']
        self.assertEqual((counts['PASS'], counts['PENDING']), (3, 37))

    def test_config_mismatch_rejected_without_mutation(self):
        self.assertEqual(self.invoke('--limit','1').returncode, 0)
        result = self.invoke('--limit','1','--batch','512','--ubatch','512')
        self.assertEqual(result.returncode, 2)
        self.assertIn('different configuration', result.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)

    def test_rerun_all_preserves_old_pass(self):
        self.assertEqual(self.invoke('--limit','1').returncode, 0)
        self.assertEqual(self.invoke('--limit','1','--rerun-all').returncode, 0)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 2)
        result = self.invoke('--limit','2','--retry-failed-only')
        self.assertEqual(result.returncode, 1)  # unrun prompt_128_02 stays PENDING
        self.assertEqual(len(self.load('prompt_128_02.txt')), 0)

    def test_no_metrics_never_pass(self):
        (self.root/'behavior.txt').write_text('no_metrics')
        self.assertEqual(self.invoke('--limit','1').returncode, 1)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['status'], 'NO_METRICS')

    def test_timeout_marks_failure_and_keeps_logs(self):
        (self.root/'behavior.txt').write_text('timeout')
        self.assertEqual(self.invoke('--limit','1','--timeout','0.1').returncode, 1)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['status'], 'TIMEOUT')
        self.assertTrue((self.out/'runs/prompt_128_01/attempt-0001/engine.stderr.log').exists())

    def test_fixed_ring_and_native_tail_policy(self):
        self.assertEqual(self.invoke('--limit', '1').returncode, 0)
        row = self.load('prompt_128_01.txt')[0]
        env = row['env_overrides']
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_SKIP_PROFILE'], '1')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_PLANNER'], 'apply-wide')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT'],
                         'G8,P6,G8,P6,G8,P6,G8,P6,G8')
        for flag in ('FORCE_XC', 'FORCE_XG', 'FORCE_XP', 'CHUNK_TOKENS'):
            self.assertEqual(env['LLAMA_HYBRID_REGION_RING_'+flag], '128')
        self.assertEqual(env['LLAMA_REGION_RING_COALESCE_TINY_TAIL'], '1')
        config = json.loads((self.out/'run-config.json').read_text())
        self.assertEqual(config['planner_search'], 'SKIPPED_FORCED_LAYOUT')
        self.assertTrue(config['tiny_tail_coalesce'])
        self.assertEqual(row['layout'], config['fixed_layout'])
        self.assertEqual(row['context'], 1024)

    def test_wrong_runtime_layout_is_not_a_pass(self):
        (self.root/'behavior.txt').write_text('wrong_layout')
        self.assertEqual(self.invoke('--limit', '1').returncode, 1)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['status'], 'WRONG_LAYOUT')

    def test_fixed_configuration_cannot_be_overridden(self):
        for opt in ('--force-layout', '--skip-profile', '--baseline-chunk-tokens'):
            self.assertNotEqual(self.invoke(opt).returncode, 0)
        attempt = self.invoke('--env', 'LLAMA_HYBRID_REGION_RING_FORCE_XG=64')
        self.assertEqual(attempt.returncode, 2)
        self.assertIn('cannot override fixed Ring settings', attempt.stderr)
        self.assertFalse(self.out.exists())

    def test_manifest_context_and_dry_run_no_output(self):
        dry = self.invoke('--only-prompt','prompt_1024_10.txt','--dry-run')
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn('ctx=2304', dry.stdout)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.invoke('--only-prompt','prompt_1024_10.txt').returncode, 0)
        result = self.load('prompt_1024_10.txt')[-1]
        self.assertEqual(result['prompt_tokens_hint'], 1855)
        self.assertEqual(result['context'], 2304)


if __name__ == '__main__':
    unittest.main()
