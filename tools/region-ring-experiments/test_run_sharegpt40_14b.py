import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run_sharegpt40_14b as mod

class FakePopen:
    calls = []
    def __init__(self, command, cwd, env, stdin, stdout, stderr):
        self.__class__.calls.append((command, env))
    def wait(self): return 0
    def poll(self): return 0
    def kill(self): pass

def read_result(out, stem):
    return json.loads((out / stem / "result.json").read_text())

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root/'llama-completion.exe'; self.exe.touch()
        self.model = self.root/'Qwen3-14B-Q4_K_M.gguf'; self.model.write_bytes(b'model')
        self.prompts = self.root/'sharegpt40'; self.prompts.mkdir()
        for p in ['prompt_128_01.txt', 'prompt_128_02.txt', 'prompt_512_01.txt','prompt_1024_02.txt']:
            (self.prompts/p).write_text('hello '+p)
        (self.prompts/'manifest.csv').write_text(
            'file,length_bin,sample_in_bin,actual_tokens\n'
            'prompt_128_01.txt,128-255,1,167\n'
            'prompt_128_02.txt,128-255,2,171\n'
            'prompt_512_01.txt,512-1024,1,587\n'
            'prompt_1024_02.txt,1024-2000,2,1500\n')
        self.out = self.root/'results'
        self.base = ['--exe', str(self.exe), '--model', str(self.model), '--rpc', 'local:50052',
                     '--prompt-dir', str(self.prompts), '--out', str(self.out), '--delay', '0',
                     '--token-source', 'manifest']
        FakePopen.calls.clear()
        self.patch = patch.object(mod.subprocess, 'Popen', FakePopen)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.git_patch = patch.object(mod, 'git_head', return_value='test-commit')
        self.git_patch.start(); self.addCleanup(self.git_patch.stop)
        def fake_result_row(args, folder, prompt, hint, context, code, elapsed, commit):
            import hashlib
            return {
                'prompt_file': prompt.name,
                'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
                'actual_prefill_tokens': hint,
                'actual_decode_tokens': 1,
                'prefill_ms': 5.0 if code == 0 else None,
                'prefill_tps': 321.1 if code == 0 else None,
                'decode_ms': 10.0,
                'decode_tps': 3.5,
                'exit_code': code,
                'status': 'PASS' if code == 0 else 'FAIL',
                'commit': commit,
                'run_dir': str(folder),
            }
        self.row_patch = patch.object(mod, 'result_row', side_effect=fake_result_row)
        self.row_patch.start(); self.addCleanup(self.row_patch.stop)

    def rows(self):
        with (self.out/'summary.csv').open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def test_resume_and_force_rerun_preserve_history(self):
        self.assertEqual(mod.main(self.base+['--limit', '2']), 0)
        self.assertEqual(len(FakePopen.calls), 2)
        rows=self.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['status'] for r in rows], ['PASS','PASS','PENDING','PENDING'])
        self.assertEqual(mod.main(self.base), 0)
        self.assertEqual(len(FakePopen.calls), 4)
        self.assertEqual([r['status'] for r in self.rows()], ['PASS']*4)
        self.assertEqual(mod.main(self.base), 0)
        self.assertEqual(len(FakePopen.calls), 4)
        self.assertEqual(mod.main(self.base+['--rerun','prompt_512_01']), 0)
        self.assertEqual(len(FakePopen.calls), 5)
        self.assertEqual(json.loads((self.out/'prompt_512_01'/'result.json').read_text())['attempt'], 2)
        self.assertTrue((self.out/'history'/'prompt_512_01'/'attempt-001'/'result.json').exists())
        self.assertEqual(len(self.rows()), 4)
        cmd,env=FakePopen.calls[-1]
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT'], mod.LAYOUT)
        self.assertEqual(env['LLAMA_HYBRID_REGION_RING_FORCE_XG'], '128')
        self.assertEqual(cmd[cmd.index('-ub')+1], '2048')
        self.assertEqual(cmd[cmd.index('-c')+1], '1024')
        self.assertEqual(cmd[cmd.index('-n')+1], '1')
        self.assertFalse((self.out / 'profile').exists())
        self.assertIn('--no-mmap',cmd)

    def test_changed_prompt_automatically_retried(self):
        mod.main(self.base)
        self.assertEqual(len(FakePopen.calls), 4)
        (self.prompts/'prompt_128_02.txt').write_text('modified input')
        mod.main(self.base)
        self.assertEqual(len(FakePopen.calls), 5)
        self.assertTrue((self.out/'history'/'prompt_128_02'/'attempt-001').exists())

    def test_below_and_only(self):
        mod.main(self.base)
        self.assertEqual(len(FakePopen.calls), 4)
        mod.main(self.base+['--only','prompt_128_02'])
        self.assertEqual(len(FakePopen.calls),4)
        mod.main(self.base+['--only','prompt_128_02','--rerun-below','330'])
        self.assertEqual(len(FakePopen.calls),5)
        self.assertTrue((self.out/'history'/'prompt_128_02'/'attempt-001').exists())

    def test_failed_attempt_is_retried(self):
        # A failed run is not considered completed and is retried automatically.
        original = mod.result_row
        def fail_once(args, folder, prompt, group, context, code, elapsed, commit):
            row = original(args, folder, prompt, group, context, code, elapsed, commit)
            if not (self.out / "history" / prompt.stem).exists():
                row["status"] = "FAIL"
            return row
        with patch.object(mod, "result_row", side_effect=fail_once):
            self.assertEqual(mod.main(self.base + ["--only", "prompt_128_01"]), 1)
        self.assertEqual(mod.main(self.base + ["--only", "prompt_128_01"]), 0)
        self.assertEqual(len(FakePopen.calls), 2)
        self.assertEqual(read_result(self.out, "prompt_128_01")["attempt"], 2)
        self.assertTrue((self.out / "history" / "prompt_128_01" / "attempt-001").exists())

    def test_config_guard_and_dry_run(self):
        self.assertEqual(mod.main(self.base+['--dry-run']), 0)
        self.assertFalse(self.out.exists())
        mod.main(self.base+['--limit', '1'])
        with self.assertRaises(SystemExit):
            mod.main(self.base+['--context','3072'])
        self.assertEqual(len(FakePopen.calls),1)


    def test_context_auto_uses_real_manifest_counts(self):
        self.assertEqual(mod.main(self.base), 0)
        by_name = {Path(cmd[cmd.index("-f") + 1]).name:
                   (int(cmd[cmd.index("-c") + 1]), int(cmd[cmd.index("-n") + 1]))
                   for cmd, env in FakePopen.calls}
        self.assertEqual(by_name["prompt_128_01.txt"], (1024, 1))
        self.assertEqual(by_name["prompt_512_01.txt"], (1024, 1))
        # 1500 + 1 predicted + 64 safety = 1565; round to 1792.
        self.assertEqual(by_name["prompt_1024_02.txt"], (1792, 1))
        plan = json.loads((self.out/"prompt_1024_02"/"plan.json").read_text())
        self.assertEqual(plan["context"], 1792)
        self.assertEqual(plan["plan_tokens"], 1500)
        self.assertEqual(plan["layout_fixed"], "G9,P1,G9,P1,G9,P1,G10")
        self.assertEqual(plan["model"], "fixed_manual_no_profile")
        self.assertEqual(plan["xp"], 128)
        self.assertFalse((self.out/"profile").exists())

    def test_context_boundaries_and_fixed_overflow(self):
        self.assertEqual(mod.context_for_tokens(1855), 2048)
        self.assertEqual(mod.context_for_tokens(1013), 1280)
        self.assertEqual(mod.context_for_tokens(1900), 2048)
        self.assertEqual(mod.context_for_tokens(1984), 2304)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            mod.context_for_tokens(1855, fixed=1856)
        with self.assertRaises(SystemExit):
            mod.main(self.base + ["--context", "1024",
                                  "--only", "prompt_1024_02"])
        self.assertFalse(FakePopen.calls)

    def test_manual_overrides_and_independent_chunks(self):
        overrides = self.root/"chunks.json"
        overrides.write_text(json.dumps({
            "groups": {"1024": {"xg": 112, "xp": 96}},
            "prompts": {"prompt_1024_02": {"xp": 64}}
        }))
        self.assertEqual(mod.main(self.base + [
            "--xc", "128", "--xg", "128", "--xp", "128",
            "--chunk-overrides", str(overrides)]), 0)
        cmd, env = FakePopen.calls[-1]
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"],
                         "G9,P1,G9,P1,G9,P1,G10")
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XC"], "128")
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XG"], "112")
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XP"], "64")
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_PLAN_TOKENS"], "1500")
        self.assertEqual(cmd[cmd.index("-n") + 1], "1")
        previous_count = len(FakePopen.calls)
        with self.assertRaises(SystemExit):
            mod.main(self.base + ["--chunk-overrides", str(overrides),
                                  "--xp", "96"])
        self.assertEqual(len(FakePopen.calls), previous_count)

    def test_default_dry_run_reports_context_without_writing(self):
        output = self.root/"new-dry-run"
        with patch("sys.stdout", new_callable=__import__('io').StringIO) as capture:
            self.assertEqual(mod.main([
                "--exe", str(self.exe), "--model", str(self.model),
                "--rpc", "local:50052", "--prompt-dir", str(self.prompts),
                "--out", str(output), "--token-source", "manifest",
                "--dry-run", "--limit", "4", "--delay", "0"]), 0)
        self.assertFalse(output.exists())
        self.assertIn("context=1792", capture.getvalue())
        self.assertIn("profile=SKIPPED", capture.getvalue())
        self.assertFalse(FakePopen.calls)

    def test_reject_invalid_layout_or_override(self):
        with self.assertRaisesRegex(ValueError, "invalid layout"):
            mod.validate_layout("G8,P2,G8,P2,G8,P2,G9")
        with self.assertRaisesRegex(ValueError, "invalid layout"):
            mod.validate_layout("P2,G8,P2,G8,P2,G8,G10")
        over = self.root / "invalid-chunks.json"
        over.write_text('{"groups":{"1024":{"xp":0}}}')
        with self.assertRaisesRegex(ValueError, "integer"):
            mod.load_chunk_overrides(over)

    def test_model_tokenizer_cache_and_invalidation(self):
        tokenizer = self.root / 'llama-tokenize.exe'
        tokenizer.touch()
        options = SimpleNamespace(
            tokenizer_exe=tokenizer,
            model=self.model,
            out=self.out,
            root=self.root,
            update_tokens=False,
        )
        self.out.mkdir()
        prompts = [(128, self.prompts/'prompt_128_01.txt'),
                   (128, self.prompts/'prompt_128_02.txt')]
        result = SimpleNamespace(returncode=0,
                                 stdout='[1,2,3]\nTotal number of tokens: 167\n',
                                 stderr='')
        with patch.object(mod.subprocess, 'run', return_value=result) as p:
            counts = mod.model_token_counts(options, prompts)
            self.assertEqual(counts['prompt_128_01.txt'], 167)
            self.assertEqual(p.call_count, 2)
            self.assertIn('--no-escape', p.call_args.args[0])
            self.assertIn('--show-count', p.call_args.args[0])
        self.assertTrue((self.out/'prompt-tokens.json').is_file())
        with patch.object(mod.subprocess, 'run', side_effect=AssertionError('not cached')):
            mod.model_token_counts(options, prompts)
        (self.prompts/'prompt_128_01.txt').write_text('changed')
        with patch.object(mod.subprocess, 'run', return_value=result) as p:
            mod.model_token_counts(options, prompts)
            self.assertEqual(p.call_count, 1)
        options.update_tokens=True
        with patch.object(mod.subprocess, 'run', return_value=result) as p:
            mod.model_token_counts(options, prompts)
            self.assertEqual(p.call_count, 2)

    def test_profile_cached_once_and_chunk_dag(self):
        planned_args = list(self.base) + ['--profile-chunks', '--layout', mod.chunk_dag.LAYOUT]
        def fake_extract(folder, x):
            return {
                'width': x, 'stages': [
                    dict(device='GPU', layers=9, tokens=x, ms=9 + x * 0.1),
                    dict(device='GPU', layers=10, tokens=x, ms=10 + x * 0.1),
                    dict(device='PHONE', layers=1, tokens=x, ms=2 + x * 0.08),
                ], 'push': [dict(tokens=x, ms=0.5 + x * 0.01)],
                'bank_refill_ms': {'4': 200.0, '6': 220.0},
            }
        with patch.object(mod.chunk_dag, 'extract_probe', side_effect=fake_extract):
            self.assertEqual(mod.main(planned_args + ['--limit', '1']), 0)
        self.assertEqual(len(FakePopen.calls), len(mod.chunk_dag.GRID) + 1)  # expanded profile + real run
        self.assertTrue((self.out / 'profile' / 'chunk-profile.json').is_file())
        plan = json.loads((self.out / 'prompt_128_01' / 'plan.json').read_text())
        self.assertEqual(plan['layout_fixed'], mod.chunk_dag.LAYOUT)
        self.assertEqual(plan['plan_tokens'], 167)
        self.assertEqual(plan['token_source'], 'manifest')
        self.assertEqual(plan['candidate_count'], 49)
        self.assertEqual(plan['candidate_grid'], list(mod.chunk_dag.GRID))
        self.assertIn(plan['xg'], mod.chunk_dag.GRID)
        self.assertIn(plan['xp'], mod.chunk_dag.GRID)
        self.assertEqual(mod.main(planned_args), 0)
        self.assertEqual(len(FakePopen.calls), len(mod.chunk_dag.GRID) + 4)
        self.assertEqual(mod.main(planned_args), 0)
        self.assertEqual(len(FakePopen.calls), len(mod.chunk_dag.GRID) + 4)


    def test_expanded_profile_reuses_old_successful_widths(self):
        """A previous three-width calibration shouldn't repeat 64/96/128."""
        prompt = self.prompts / 'prompt_512_01.txt'
        old = (64, 96, 128)
        for width in old:
            folder = self.out / 'profile' / 'probes' / f'x{width}' / prompt.stem
            folder.mkdir(parents=True)
            (folder / 'result.json').write_text(json.dumps({
                'status': 'PASS',
                'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
            }))
        options = SimpleNamespace(out=self.out, model=self.model,
                                  exe=self.exe, rpc='local:50052',
                                  context=4096, root=self.root,
                                  update_profile=False)
        def fake_extract(folder, x):
            return {
                'width': x,
                'stages': [
                    dict(device='GPU', layers=9, tokens=x, ms=10.0),
                    dict(device='GPU', layers=10, tokens=x, ms=11.0),
                    dict(device='PHONE', layers=1, tokens=x, ms=5.0)],
                'push': [dict(tokens=x, ms=2.0)],
                'bank_refill_ms': {'4': 200.0, '6': 220.0},
            }
        with patch.object(mod.chunk_dag, 'extract_probe', side_effect=fake_extract):
            profile = mod.ensure_chunk_profile(options, [(512, prompt)], 'test-commit')
        self.assertEqual(
            {x['width'] for x in profile['probes']}, set(mod.chunk_dag.GRID))
        self.assertEqual(len(FakePopen.calls), len(mod.chunk_dag.GRID) - len(old))
        self.assertTrue((self.out/'profile'/'chunk-profile.json').is_file())

    def test_completed_prompt_replans_only_if_widths_change(self):
        planned_args = list(self.base) + ['--profile-chunks', '--layout', mod.chunk_dag.LAYOUT]
        def fake_extract(folder, x):
            return {
                'width': x,
                'stages': [
                    dict(device='GPU', layers=9, tokens=x, ms=9 + x * .1),
                    dict(device='GPU', layers=10, tokens=x, ms=10 + x * .1),
                    dict(device='PHONE', layers=1, tokens=x, ms=2 + x * .08)],
                'push': [dict(tokens=x, ms=.5 + x * .01)],
                'bank_refill_ms': {'4': 200.0, '6': 220.0},
            }
        with patch.object(mod.chunk_dag, 'extract_probe', side_effect=fake_extract):
            self.assertEqual(mod.main(planned_args + ['--limit', '1']), 0)
        original_calls = len(FakePopen.calls)
        planfile = self.out/'prompt_128_01'/'plan.json'
        original_plan = json.loads(planfile.read_text())
        older_plan = dict(original_plan)
        older_plan.pop('candidate_grid')
        older_plan.pop('candidate_count')
        planfile.write_text(json.dumps(older_plan))
        self.assertEqual(mod.main(planned_args + ['--only', 'prompt_128_01']), 0)
        self.assertEqual(len(FakePopen.calls), original_calls)
        self.assertEqual(json.loads(planfile.read_text())['candidate_count'], 49)
        # A newly chosen different GPU width must archive and re-run.
        new_width = next(x for x in mod.chunk_dag.GRID if x != original_plan['xg'])
        def different_plan(profile, tokens):
            changed = dict(original_plan)
            changed['xg'] = new_width
            changed['candidate_count'] = 49
            return changed
        with patch.object(mod.chunk_dag, 'choose', side_effect=different_plan):
            self.assertEqual(mod.main(planned_args + ['--only', 'prompt_128_01']), 0)
        self.assertEqual(len(FakePopen.calls), original_calls + 1)
        self.assertTrue((self.out/'history'/'prompt_128_01'/
                         'attempt-001'/'result.json').exists())
        self.assertEqual(read_result(self.out, 'prompt_128_01')['xg'], new_width)

if __name__ == '__main__':
    unittest.main()
