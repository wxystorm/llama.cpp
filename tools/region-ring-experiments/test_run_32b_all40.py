#!/usr/bin/env python3
"""Offline runner tests: no real GGUF, RPC server or GPU required."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER = Path(__file__).with_name('run_32b_all40.py')
MOCK = '''#!/usr/bin/env python3
import json,os,pathlib,sys,time
args=sys.argv
prompt=pathlib.Path(args[args.index('-f')+1]).name
config=pathlib.Path(os.environ['LLAMA_REGION_RING_METRICS_FILE'])
root=config.parent.parent.parent.parent if config.parent.parent.name=='warmup' else config.parent.parent.parent.parent.parent
mode=root / 'behavior.txt'
behavior=mode.read_text() if mode.exists() else ''
once=root / ('attempted_'+prompt)
if behavior=='fail_once' and prompt=='prompt_128_01.txt' and not once.exists():
    once.write_text('1'); sys.exit(1)
if behavior=='no_metrics' and prompt=='prompt_128_01.txt': sys.exit(0)
if behavior=='wrong_layout' and prompt=='prompt_128_01.txt':
  with config.open('w') as f:
    for x in [{'type':'measurement_start'}, {'type':'region_ring','complete':1,'status':0,'layout':'G40,P24'}, {'type':'perf','prefill_tokens':167,'prefill_ms':42}]:
      f.write(json.dumps(x)+'\\n')
  sys.exit(0)
if behavior=='timeout' and prompt=='prompt_128_01.txt': time.sleep(2)
layout=os.environ['LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT']
xg=int(os.environ['LLAMA_HYBRID_REGION_RING_FORCE_XG'])
force=os.environ.get('LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH')=='1'
tokens=167
active=(tokens>=2*xg or force) and tokens>1
ring={'type':'region_ring','complete':1,'status':0,'tokens':tokens,
      'layout':layout,'xc':int(os.environ['LLAMA_HYBRID_REGION_RING_FORCE_XC']),
      'xg':xg,'xp':int(os.environ['LLAMA_HYBRID_REGION_RING_FORCE_XP']),
      'gpu_bank_streaming':active,'gpu_bank_min_tokens':2*xg,
      'gpu_bank_force_small_batch':force,
      'gpu_bank_small_batch_override_applied':force and tokens<2*xg and active,
      'bank_refills':3 if active else 0,
      'bank_h2d_ms':400 if active else 0,'bank_gpu_wait_ms':120 if active else 0,
      'bank_mover_wall_ms':500 if active else 0,'gpu_weight_stage_ms':100}
if behavior=='old_engine': ring.pop('gpu_bank_force_small_batch')
if behavior=='inactive_mover': ring['gpu_bank_streaming']=False
if os.environ.get('LLAMA_REGION_RING_PHONE_EARLY_WEIGHT_RPC'):
  print('[REGION_RING_PHONE_EARLY_WEIGHTS] event=START phase=PC_MODEL_LOAD phone_regions=4 status=RUNNING', file=sys.stderr)
  if behavior=='preload_fail':
    print('[REGION_RING_PHONE_EARLY_WEIGHTS] event=FAIL reason=REMOTE_PRELOAD_UNSUPPORTED', file=sys.stderr)
    print('[REGION_RING_PHONE_EARLY_WEIGHTS] event=JOIN phase=PC_MODEL_LOAD status=FAIL', file=sys.stderr)
  elif behavior!='preload_unverified':
    for r in (1,3,5,7):
      print(f'[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION region={r} first_layer={r} layers=5 wall_ms=4 status=PASS', file=sys.stderr)
    print('[REGION_RING_PHONE_EARLY_WEIGHTS] event=DONE regions=4 wall_ms=12.4 phase=PC_MODEL_LOAD status=PASS', file=sys.stderr)
    print('[REGION_RING_PHONE_EARLY_WEIGHTS] event=JOIN phase=PC_MODEL_LOAD status=PASS', file=sys.stderr)
with config.open('w') as f:
  for x in [{'type':'measurement_start'}, ring, {'type':'perf','prefill_tokens':167,'prefill_ms':42,'prefill_tps':3976.2}]:
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
                       '--max-retries', '0',
                       '--inter-prompt-delay-seconds', '0',
                       '--no-phone-weight-preload', '--skip-warmup']

    def invoke(self, *options):
        return subprocess.run([sys.executable, str(RUNNER), *self.common, *options],
                              capture_output=True, text=True, timeout=30)

    def invoke_new(self, *options):
        common = [arg for arg in self.common
                  if arg not in ('--no-phone-weight-preload', '--skip-warmup')]
        return subprocess.run([sys.executable, str(RUNNER), *common, *options],
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
        self.assertEqual(row['context'], 256)
        self.assertEqual(row['command'][row['command'].index('-n') + 1], '1')
        self.assertEqual(row['command'][row['command'].index('-c') + 1], '256')

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

    def test_manifest_fallback_contexts_are_compact(self):
        for name, expected in (
            ('prompt_128_02.txt', 512),  # upper bound 255 + 1 + 64
            ('prompt_256_02.txt', 768),  # upper bound 511 + 1 + 64
            ('prompt_512_02.txt', 1280),
            ('prompt_1024_02.txt', 2304),
        ):
            dry = self.invoke('--only-prompt', name, '--dry-run')
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn(f'ctx={expected}', dry.stdout)
            self.assertIn(' -n 1 ', dry.stdout)

    def test_prediction_and_context_change_do_not_mix_previous_results(self):
        self.assertEqual(self.invoke('--limit', '1').returncode, 0)
        old = self.invoke('--limit', '1', '--n-predict', '32')
        self.assertEqual(old.returncode, 2)
        self.assertIn('different configuration', old.stderr)
        old = self.invoke('--limit', '1', '--context-padding', '256')
        self.assertEqual(old.returncode, 2)
        self.assertIn('different configuration', old.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)

    def test_custom_layout_and_chunks_pass_and_rerun_one(self):
        layout = 'G7,P6,G8,P6,G8,P6,G8,P6,G9'
        options = ('--layout', ' g7, p6, G8,P6,G8,P6,G8,P6,G9 ',
                   '--chunk', '64', '--xc', '48', '--xg', '96', '--xp', '128')
        r = self.invoke(*options, '--limit', '2')
        self.assertEqual(r.returncode, 0, r.stderr)
        row = self.load('prompt_128_01.txt')[0]
        self.assertEqual(row['status'], 'PASS')
        self.assertEqual(row['layout'], layout)
        self.assertEqual(row['plan_layout'], layout)
        self.assertEqual((row['plan_xc'], row['plan_xg'], row['plan_xp']),
                         (48, 96, 128))
        self.assertEqual((row['runtime_xc'], row['runtime_xg'],
                          row['runtime_xp']), (48, 96, 128))
        env = row['env_overrides']
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS'], '64')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_XC'], '48')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_XG'], '96')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_XP'], '128')
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_SKIP_PROFILE'], '1')
        cfg = json.loads((self.out / 'run-config.json').read_text())
        self.assertEqual(cfg['fixed_layout'], layout)
        self.assertEqual((cfg['fixed_xc'], cfg['fixed_xg'], cfg['fixed_xp']),
                         (48, 96, 128))
        r = self.invoke(*options, '--only-prompt', 'prompt_128_01.txt',
                        '--rerun-all')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 2)
        self.assertEqual(len(self.load('prompt_128_02.txt')), 1)
        self.assertEqual(self.invoke(*options, '--only-prompt',
                         'prompt_128_01.txt').returncode, 0)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 2)

    def test_config_changed_with_manual_parameters_is_refused(self):
        self.assertEqual(self.invoke('--limit', '1').returncode, 0)
        for flag, value in (('--layout', 'G7,P6,G8,P6,G8,P6,G8,P6,G9'),
                            ('--chunk', '64'), ('--xp', '96')):
            r = self.invoke('--limit', '1', flag, value)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('different configuration', r.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)

    def test_reject_invalid_manual_layouts_and_chunks(self):
        for flag, value in (('--layout', 'G8,P6'),
                            ('--layout', 'G64'),
                            ('--layout', 'G8,G8,P48'),
                            ('--layout', 'P1,G63'),
                            ('--chunk', '256'), ('--xg', '7'), ('--xp', '0')):
            r = self.invoke('--limit', '1', flag, value)
            self.assertEqual(r.returncode, 2, (flag, value, r.stderr))
        self.assertFalse(self.out.exists())

    def test_auto_plan_rejects_manual_override(self):
        for flag, value in (('--layout', 'G8,P56'), ('--chunk', '64'),
                            ('--xc', '64'), ('--xp', '128')):
            r = self.invoke('--auto-plan', flag, value)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('cannot be combined', r.stderr)
        self.assertFalse(self.out.exists())

    def test_c0_is_canonicalized(self):
        r = self.invoke('--layout', 'G8,C0,P56', '--limit', '1')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['layout'], 'G8,P56')

    def test_xg96_force_short_mover_and_metrics(self):
        out = self.root/'force-short-out'
        options = ('--out', str(out), '--xg', '96',
                   '--force-small-batch-mover', '--limit', '1')
        result = self.invoke(*options)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[GPU_BANK_FORCE_SHORT] active=True', result.stdout)
        row = json.loads(next((out/'runs/prompt_128_01').glob('attempt-*/result.json')).read_text())
        self.assertEqual(row['status'], 'PASS')
        self.assertTrue(row['gpu_bank_force_small_batch'])
        self.assertTrue(row['gpu_bank_streaming'])
        self.assertTrue(row['gpu_bank_small_batch_override_applied'])
        self.assertEqual(row['gpu_bank_min_tokens'], 192)
        self.assertEqual(row['bank_refills'], 3)
        self.assertEqual(row['bank_h2d_ms'], 400)
        self.assertEqual(row['bank_gpu_wait_ms'], 120)
        self.assertEqual(row['env_overrides'][
            'LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH'], '1')
        self.assertTrue(json.loads((out/'run-config.json').read_text())[
            'force_small_batch_mover'])
        with (out/'summary.csv').open(encoding='utf-8-sig') as stream:
            summary = next(csv.DictReader(stream))
        self.assertEqual(summary['gpu_bank_streaming'], 'True')
        self.assertEqual(summary['bank_refills'], '3')
        # Same settings can force rerun without overwriting attempt-0001.
        again = self.invoke(*options, '--only-prompt', 'prompt_128_01.txt',
                            '--rerun-all')
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(len(list((out/'runs/prompt_128_01').glob(
            'attempt-*/result.json'))), 2)

    def test_xg96_mover_off_stays_on_demand(self):
        result = self.invoke('--xg', '96', '--limit', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        row = self.load('prompt_128_01.txt')[-1]
        self.assertEqual(row['status'], 'PASS')
        self.assertFalse(row['gpu_bank_streaming'])
        self.assertEqual(row['gpu_bank_min_tokens'], 192)
        self.assertEqual(row['bank_refills'], 0)
        self.assertNotIn('LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH',
                         row['env_overrides'])

    def test_force_mover_old_engine_or_inactive_rejected(self):
        (self.root/'behavior.txt').write_text('old_engine')
        options = ('--xg', '96', '--force-small-batch-mover', '--limit', '1')
        old = self.invoke(*options)
        self.assertEqual(old.returncode, 1)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['status'],
                         'MOVER_UNVERIFIED')
        (self.root/'behavior.txt').write_text('inactive_mover')
        inactive = self.invoke(*options)
        self.assertEqual(inactive.returncode, 1)
        self.assertEqual(self.load('prompt_128_01.txt')[-1]['status'],
                         'MOVER_INACTIVE')

    def test_force_mover_requires_streaming_and_distinct_config(self):
        for opts in (('--streaming', 'off'),
                     ('--auto-plan',),
                     ('--layout', 'G8,P56')):
            result = self.invoke('--force-small-batch-mover', *opts,
                                 '--limit', '1')
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.invoke('--xg', '96', '--limit', '1').returncode, 0)
        changed = self.invoke('--xg', '96', '--force-small-batch-mover',
                              '--limit', '1')
        self.assertEqual(changed.returncode, 2)
        self.assertIn('different configuration', changed.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)

    def test_force_mover_default_output_is_isolated(self):
        args = list(self.common)
        index = args.index('--out')
        del args[index:index + 2]
        cmd = [sys.executable, str(RUNNER), *args,
               '--force-small-batch-mover', '--limit', '1']
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        forced = list(self.root.glob('32b-*-force-short-mover*-all40'))
        self.assertEqual(len(forced), 1)
        self.assertTrue((forced[0]/'run-config.json').exists())
        self.assertFalse((self.root/'32b-g8p6-x128-n1-all40').exists())

    def test_inter_prompt_cooldown_only_between_executed_prompts(self):
        options = ('--inter-prompt-delay-seconds', '0.15')
        begin = time.monotonic()
        first = self.invoke('--limit', '3', *options)
        duration = time.monotonic() - begin
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout.count('[INTER_PROMPT_PAUSE]'), 2)
        self.assertIn('after=prompt_128_01.txt before=prompt_128_02.txt', first.stdout)
        self.assertIn('after=prompt_128_02.txt before=prompt_128_03.txt', first.stdout)
        self.assertGreaterEqual(duration, 0.28)
        self.assertEqual(json.loads((self.out / 'run-config.json').read_text())[
            'inter_prompt_delay_seconds'], 0.15)
        # First three PASS prompts skip; one new execution has no pause.
        resume = self.invoke('--limit', '4', *options)
        self.assertEqual(resume.returncode, 0, resume.stderr)
        self.assertEqual(resume.stdout.count('[INTER_PROMPT_PAUSE]'), 0)
        self.assertEqual(len(self.load('prompt_128_04.txt')), 1)

    def test_inter_prompt_pause_excludes_retries(self):
        (self.root / 'behavior.txt').write_text('fail_once')
        run = self.invoke('--limit', '2', '--max-retries', '1',
                          '--retry-delay', '0',
                          '--inter-prompt-delay-seconds', '0.10')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([v['status'] for v in self.load('prompt_128_01.txt')],
                         ['FAIL', 'PASS'])
        self.assertEqual(run.stdout.count('[INTER_PROMPT_PAUSE]'), 1)

    def test_boost_bank_mover_and_no_mmap_are_signature_safe(self):
        options = ('--xg', '96', '--force-small-batch-mover',
                   '--boost-bank-mover', '--no-mmap', '--limit', '1')
        run = self.invoke(*options)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('bank_mover_priority=ABOVE_NORMAL', run.stdout)
        self.assertIn('no_mmap=1', run.stdout)
        row = self.load('prompt_128_01.txt')[0]
        self.assertEqual(row['status'], 'PASS')
        self.assertIn('--no-mmap', row['command'])
        self.assertEqual(row['env_overrides']['LLAMA_REGION_RING_PRIORITY_BANK_MOVER'],
                         '1')
        cfg = json.loads((self.out / 'run-config.json').read_text())
        self.assertTrue(cfg['no_mmap'])
        self.assertEqual(cfg['bank_mover_priority'], 'ABOVE_NORMAL')
        self.assertEqual(cfg['effective_runtime_env'][
            'LLAMA_REGION_RING_PRIORITY_BANK_MOVER'], '1')
        mismatch = self.invoke('--xg', '96', '--force-small-batch-mover',
                               '--no-mmap', '--limit', '1')
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn('different configuration', mismatch.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)

    def test_cooldown_defaults_and_preflight_validation(self):
        common = list(self.common)
        i = common.index('--inter-prompt-delay-seconds')
        del common[i:i + 2]
        dry = subprocess.run([sys.executable, str(RUNNER), *common,
                              '--dry-run', '--limit', '2'],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn('between_prompts=5s', dry.stdout)
        for invalid in ('-1', 'nan', 'inf'):
            bad = self.invoke('--inter-prompt-delay-seconds', invalid)
            self.assertEqual(bad.returncode, 2, (invalid, bad.stderr))
        priority_override = self.invoke(
            '--env', 'LLAMA_REGION_RING_PRIORITY_BANK_MOVER=1')
        self.assertEqual(priority_override.returncode, 2)
        self.assertFalse(self.out.exists())

    def test_default_boost_nommap_output_separates_from_baseline(self):
        common = list(self.common)
        i = common.index('--out')
        del common[i:i + 2]
        run = subprocess.run(
            [sys.executable, str(RUNNER), *common,
             '--boost-bank-mover', '--no-mmap', '--limit', '1'],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        boosted = list(self.root.glob('32b-*-boost-mover-nommap-pause0s-all40'))
        self.assertEqual(len(boosted), 1)
        self.assertTrue((boosted[0] / 'run-config.json').exists())
        self.assertFalse((self.root / '32b-g8p6-x128-n1-all40').exists())

    def test_previous_no_cooldown_signature_can_resume_at_zero_only(self):
        self.assertEqual(self.invoke('--limit', '1').returncode, 0)
        path = self.out / 'run-config.json'
        legacy = json.loads(path.read_text())
        legacy.pop('inter_prompt_delay_seconds')
        path.write_text(json.dumps(legacy))
        resume = self.invoke('--limit', '2')
        self.assertEqual(resume.returncode, 0, resume.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)
        self.assertEqual(len(self.load('prompt_128_02.txt')), 1)
        self.assertNotIn('inter_prompt_delay_seconds', json.loads(path.read_text()))
        changed = self.invoke('--limit', '2', '--inter-prompt-delay-seconds', '5')
        self.assertEqual(changed.returncode, 2)
        self.assertIn('different configuration', changed.stderr)

    def test_model_load_preload_then_unmeasured_warmup_and_resume(self):
        first = self.invoke_new('--limit', '2')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('[WARMUP] event=START', first.stdout)
        self.assertIn('[WARMUP] event=DONE status=PASS', first.stdout)
        warm = sorted((self.out / 'warmup').glob('attempt-*/result.json'))
        self.assertEqual(len(warm), 1)
        row = json.loads(warm[0].read_text())
        self.assertTrue(row['warmup'])
        self.assertEqual(row['early_weights_status'], 'PASS')
        self.assertEqual(row['early_weights_regions'], 4)
        self.assertEqual(row['early_weights_wall_ms'], 12.4)
        self.assertEqual(row['prompt'], 'prompt_128_01.txt')
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)
        measured = self.load('prompt_128_01.txt')[0]
        self.assertEqual(measured['early_weights_status'], 'PASS')
        self.assertEqual(measured['env_overrides'][
            'LLAMA_REGION_RING_PHONE_HTP_ONLY'], '1')
        self.assertEqual(measured['env_overrides'][
            'LLAMA_REGION_RING_PHONE_EARLY_WEIGHT_RPC'], '127.0.0.1:50052')
        cfg = json.loads((self.out / 'run-config.json').read_text())
        self.assertIn('PC_MODEL_LOAD', cfg['phone_weight_preload'])
        self.assertTrue(cfg['warmup']['enabled'])
        with (self.out / 'summary.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(r['status'] == 'PASS' for r in rows), 2)

        # A completely finished selection should not warm again.
        done = self.invoke_new('--limit', '2')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('reason=NO_MEASURED_WORK', done.stdout)
        self.assertEqual(len(list((self.out / 'warmup').glob(
            'attempt-*/result.json'))), 1)
        # Resume with more work: warm up again; phone may have rebooted.
        resume = self.invoke_new('--limit', '3')
        self.assertEqual(resume.returncode, 0, resume.stderr)
        self.assertEqual(len(list((self.out / 'warmup').glob(
            'attempt-*/result.json'))), 2)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)
        self.assertEqual(len(self.load('prompt_128_03.txt')), 1)

    def test_preload_fail_or_missing_ack_prevents_formal_run(self):
        for i, (behavior, expected) in enumerate(
                (('preload_fail', 'PRELOAD_FAILED'),
                 ('preload_unverified', 'PRELOAD_UNVERIFIED'))):
            with self.subTest(behavior=behavior):
                out = self.root / f"preload-error-{i}"
                (self.root / 'behavior.txt').write_text(behavior)
                result = self.invoke_new('--out', str(out), '--limit', '1')
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('no measured Prompt was started', result.stderr)
                first = json.loads(next(
                    (out / 'warmup').glob('attempt-*/result.json')).read_text())
                self.assertEqual(first['status'], expected)
                self.assertFalse((out / 'runs').exists())
                self.assertFalse((out / 'summary.csv').read_text().count(
                    'PRELOAD_FAILED'))
                (self.root / 'behavior.txt').unlink()
                # A failed warm-up must never poison the run directory.
                recovered = self.invoke_new('--out', str(out), '--limit', '1')
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                self.assertEqual(len(list((out / 'warmup').glob(
                    'attempt-*/result.json'))), 2)
                self.assertTrue((out / 'runs/prompt_128_01').exists())

    def test_preload_can_run_without_warmup(self):
        args = [arg for arg in self.common
                if arg != '--no-phone-weight-preload']
        proc = subprocess.run(
            [sys.executable, str(RUNNER), *args, '--limit', '1'],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse((self.out / 'warmup').exists())
        self.assertEqual(self.load('prompt_128_01.txt')[0][
            'early_weights_status'], 'PASS')

    def test_warmup_choice_and_config_isolation(self):
        out = self.root / 'new-out'
        proc = self.invoke_new('--out', str(out), '--only-group', '1024',
                               '--limit', '1', '--warmup-prompt',
                               'prompt_128_02.txt')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        warm = json.loads(next((out / 'warmup').glob(
            'attempt-*/result.json')).read_text())
        self.assertEqual(warm['prompt'], 'prompt_128_02.txt')
        self.assertTrue((out / 'runs/prompt_1024_01/attempt-0001/result.json').exists())
        self.assertFalse((out / 'runs/prompt_128_02').exists())
        self.assertEqual(self.invoke_new(
            '--out', str(out), '--only-group', '1024', '--limit', '1',
            '--warmup-prompt', 'prompt_128_01.txt').returncode, 2)

    def test_new_defaults_do_not_mix_with_previous_results(self):
        self.assertEqual(self.invoke('--limit', '1').returncode, 0)
        mismatch = self.invoke_new('--limit', '1')
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn('different configuration', mismatch.stderr)
        self.assertEqual(len(self.load('prompt_128_01.txt')), 1)
        self.assertFalse((self.out / 'warmup').exists())

    def test_preload_flags_require_consistent_mode(self):
        self.assertEqual(self.invoke_new(
            '--warmup-prompt', 'unknown.txt').returncode, 2)
        self.assertEqual(self.invoke_new(
            '--warmup-prompt', 'prompt_128_02.txt',
            '--skip-warmup').returncode, 2)
        self.assertEqual(self.invoke_new(
            '--env', 'LLAMA_REGION_RING_PHONE_EARLY_WEIGHT_RPC=1').returncode, 2)
        self.assertEqual(self.invoke_new(
            '--env', 'LLAMA_REGION_RING_PHONE_HTP_ONLY=1').returncode, 2)
        self.assertEqual(self.invoke_new(
            '--log-verbosity', '3').returncode, 2)
        self.assertEqual(self.invoke_new(
            '--rpc', '127.0.0.1:50052,127.0.0.1:50053').returncode, 2)
        self.assertFalse(self.out.exists())

    def test_manifest_context_and_dry_run_no_output(self):
        dry = self.invoke('--only-prompt','prompt_1024_10.txt','--dry-run')
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn('ctx=2048', dry.stdout)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.invoke('--only-prompt','prompt_1024_10.txt').returncode, 0)
        result = self.load('prompt_1024_10.txt')[-1]
        self.assertEqual(result['prompt_tokens_hint'], 1855)
        self.assertEqual(result['context'], 2048)
        self.assertEqual(result['command'][result['command'].index('-n') + 1], '1')


if __name__ == '__main__':
    unittest.main()
