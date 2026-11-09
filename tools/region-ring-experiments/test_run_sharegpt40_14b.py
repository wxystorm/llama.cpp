import csv
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
                     '--prompt-dir', str(self.prompts), '--out', str(self.out), '--delay', '0', '--fixed-chunks']
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
                'actual_decode_tokens': 32,
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
        self.assertEqual(cmd[cmd.index('-c')+1], '4096')
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
        planned_args = list(self.base)
        planned_args.remove('--fixed-chunks')
        planned_args += ['--token-source', 'manifest']
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
        self.assertEqual(len(FakePopen.calls), 4)  # 3 probes + 1 real prompt
        self.assertTrue((self.out / 'profile' / 'chunk-profile.json').is_file())
        plan = json.loads((self.out / 'prompt_128_01' / 'plan.json').read_text())
        self.assertEqual(plan['layout_fixed'], mod.chunk_dag.LAYOUT)
        self.assertEqual(plan['plan_tokens'], 167)
        self.assertEqual(plan['token_source'], 'manifest')
        self.assertIn(plan['xg'], mod.chunk_dag.GRID)
        self.assertIn(plan['xp'], mod.chunk_dag.GRID)
        self.assertEqual(mod.main(planned_args), 0)
        self.assertEqual(len(FakePopen.calls), 7)
        self.assertEqual(mod.main(planned_args), 0)
        self.assertEqual(len(FakePopen.calls), 7)

if __name__ == '__main__':
    unittest.main()
