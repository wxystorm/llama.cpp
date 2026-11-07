"""Compact G/C/N/A Profile is a real small model, not 40-layer CPU filler."""
import unittest

from gcna_probe_model import describe

class GCNAProbeTests(unittest.TestCase):
    def test_default_layout_and_absolute_region_map(self):
        p = describe("G7,C1,N1,A1", 3)
        self.assertEqual(p["layers"], 30)
        self.assertEqual(p["runtime_layout"],
                         ",".join(["G7,C1,P1,P1"]*3))
        self.assertEqual(p["npu_regions"], [2,6,10])
        self.assertEqual(p["adreno_regions"], [3,7,11])
        self.assertEqual(p["sample_layers"],
                         {"G":21,"C":3,"N":3,"A":3})
        self.assertEqual(p["original_layer_indices"], list(range(30)))

    def test_skip_optional_cpu(self):
        p = describe("G7,N1,A1",3)
        self.assertEqual(p["runtime_layout"],
                         "G7,P1,P1,G7,P1,P1,G7,P1,P1")
        self.assertEqual(p["npu_regions"], [1,4,7])

    def test_reject_unsupported_shape(self):
        for pattern,cycles in [
            ("G7,C1,N2,A1",3),
            ("G7,C1,A1,N1",3),
            ("G7,C1,N1,A1,G1",3),
            ("G7,C1,N1,A1",1),
            ("G7,C1,N1,A1",5),
            ("G0,C1,N1,A1",3),
            ("C1,N1,A1",3),
            ("G7,C1,N1",3),
            ("G7,C1,A1",3),
            ("G9,C1,N1,A1",3),
        ]:
            with self.subTest(pattern=pattern, cycles=cycles):
                with self.assertRaises(ValueError):
                    describe(pattern, cycles)

if __name__ == "__main__":
    unittest.main()
