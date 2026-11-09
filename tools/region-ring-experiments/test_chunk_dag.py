import unittest
import chunk_dag as m


def synthetic_profile():
    data = []
    for x in (64, 96, 128):
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

    def test_select_chunks_only(self):
        selected = m.choose(synthetic_profile(), 513)
        self.assertIn(selected['xg'], m.GRID)
        self.assertIn(selected['xp'], m.GRID)
        self.assertEqual(selected['plan_tokens'], 513)
        self.assertEqual(selected['layout_fixed'], m.LAYOUT)


if __name__ == '__main__':
    unittest.main()
