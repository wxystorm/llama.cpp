import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import chunk_dag as m


def synthetic_profile():
    data = []
    for x in m.GRID:
        stages = []
        for layers in (9, 10):
            stages.append(dict(device='GPU', layers=layers, tokens=x,
                               ms=layers * 0.9 + x * 0.06))
        stages.append(dict(device='PHONE', layers=1, tokens=x,
                           ms=3.0 + x * 0.085))
        data.append(dict(width=x, stages=stages,
                         push=[dict(tokens=x, ms=0.5 + x * 0.01)],
                         bank_refill_ms={'4': 200, '6': 220}))
    return {'probes': data}


class DagTests(unittest.TestCase):
    def test_tail_coalescence(self):
        self.assertEqual(m.effective_width(513, 128), 129)
        self.assertEqual(
            m.intervals(513, 128),
            [(0, 129), (129, 258), (258, 387), (387, 513)])
        self.assertEqual(m.effective_width(511, 128), 128)
        self.assertEqual(m.effective_width(257, 128), 129)
        self.assertEqual(m.intervals(257, 128),
                         [(0, 129), (129, 257)])
        for tokens in range(1, 2200):
            for width in (32, 64, 96, 128, 160):
                chunks = m.intervals(tokens, width)
                self.assertEqual(chunks[0][0], 0)
                self.assertEqual(chunks[-1][1], tokens)
                self.assertTrue(all(a[1] == b[0]
                                    for a, b in zip(chunks, chunks[1:])))
                self.assertTrue(all(
                    0 < b - a <= width + width // 4 for a, b in chunks))

    def test_dag(self):
        p = synthetic_profile()
        for n in (64, 128, 129, 192, 256, 257, 511, 512,
                  513, 1024, 1500, 2000):
            for xg, xp in ((64, 64), (64, 128), (128, 64),
                           (128, 128), (96, 128)):
                result = m.dag_simulate(p, n, xg, xp)
                self.assertGreater(result['predicted_dag_ms'], 0)
                self.assertEqual(result['tasks'], sum(
                    len(m.intervals(n, xg if d == 'G' else xp))
                    for d, _ in m.REGIONS))
                self.assertEqual(len(result['bank_jobs']), 2)

    def test_manifest_exact_tokens_with_sharegpt_headers(self):
        # Reproduce ShareGPT40's file/length_bin/sample_in_bin/actual_tokens
        # manifest. Some extractors omit .txt or abbreviate actual_tokens.
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.csv'
            manifest.write_text(
                'file,length_bin,sample_in_bin,actual_tokens,sharegpt_id,prompt_sha256\n'
                'prompt_128_01,128-255,1,167,abc,checksum\n'
                'prompt_128_02.txt,128-255,2,171,def,checksum\n'
                'prompt_128_03.txt,128-255,3,173,ghi,checksum\n',
                encoding='utf-8-sig')
            self.assertEqual(m.manifest_tokens(manifest, 'prompt_128_01.txt'), 167)
            self.assertEqual(m.manifest_tokens(manifest, 'prompt_128_02.txt'), 171)
            self.assertEqual(m.manifest_tokens(manifest, 'prompt_128_03.txt'), 173)
            with self.assertRaisesRegex(ValueError, 'not found'):
                m.manifest_tokens(manifest, 'prompt_128_04.txt')
            with self.assertRaises(FileNotFoundError):
                m.manifest_tokens(Path(directory) / 'missing.csv', 'prompt_128_01.txt')
            manifest.write_text(
                ' FILE ,Length Bin,Actual_Toks\n'
                '"E:\\\\sharegpt\\\\prompt_512_05.txt",512-1024,771\n',
                encoding='utf-8')
            self.assertEqual(m.manifest_tokens(manifest, 'prompt_512_05.txt'), 771)
            manifest.write_text(
                'file,length_bin,actual_tokens\n'
                'prompt_128_01.txt,128-255,not-a-token\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'non-integer'):
                m.manifest_tokens(manifest, 'prompt_128_01.txt')
            manifest.write_text(
                'file,length_bin,actual_tokens\n'
                'prompt_128_01.txt,128-255,167\n'
                'prompt_128_01.txt,128-255,168\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                m.manifest_tokens(manifest, 'prompt_128_01.txt')

    def test_extended_chunk_grid(self):
        self.assertEqual(m.GRID, (32, 48, 64, 80, 96, 112, 128))
        self.assertEqual(len(m.GRID), 7)
        # Cover every GPU/Phone combination under a non-aligned tail.
        p = synthetic_profile()
        for xg in m.GRID:
            for xp in m.GRID:
                estimate = m.dag_simulate(p, 513, xg, xp)
                self.assertGreater(estimate['predicted_dag_ms'], 0)
                self.assertEqual(estimate['xg'], xg)
                self.assertEqual(estimate['xp'], xp)

    def test_select_chunks_only(self):
        selected = m.choose(synthetic_profile(), 513)
        self.assertIn(selected['xg'], m.GRID)
        self.assertIn(selected['xp'], m.GRID)
        self.assertEqual(selected['candidate_grid'], list(m.GRID))
        self.assertEqual(selected['candidate_count'], 49)
        self.assertEqual(selected['plan_tokens'], 513)
        self.assertEqual(selected['layout_fixed'], m.LAYOUT)


if __name__ == '__main__':
    unittest.main()
