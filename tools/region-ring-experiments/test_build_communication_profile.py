"""Regression for real measured HYBRID_PROFILE transfer curve extraction."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from build_communication_profile import extract, LABELS


class CommunicationProfileTests(unittest.TestCase):
    def test_aggregates_real_samples_without_fabricating_links(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            case = root / "run-1" / "engine.stderr.log"
            case.parent.mkdir(parents=True)
            lines = []
            for name in LABELS:
                lines += [
                    f"[HYBRID_PROFILE] {name} bytes=1048576 ms=2.000 confirmed_wall_ms=3.000",
                    f"[HYBRID_PROFILE] {name} bytes=1048576 ms=4.000 confirmed_wall_ms=5.000",
                    f"[HYBRID_PROFILE] {name} bytes=65536 ms=1.000 confirmed_wall_ms=1.500",
                ]
            lines.append("[HYBRID_PROFILE] phone_to_pc bytes=100 ms=0.001")
            case.write_text("\n".join(lines), encoding="utf-8")
            rows = extract([root])
            self.assertEqual(len(rows), 8)
            p = next(r for r in rows if r["path"] == "pc_to_phone"
                     and r["bytes"] == 1048576)
            self.assertEqual(p["sample_count"], 2)
            self.assertEqual(p["median_ms"], 3)
            self.assertEqual(p["wall_ms"], 4)
            self.assertAlmostEqual(p["effective_mibps"], 1000 / 3, places=4)
            self.assertFalse(any(r["bytes"] == 100 for r in rows))
            self.assertTrue(all(r["origin"] ==
                                "REAL_HYBRID_PROFILE_TRANSFER_PROBE"
                                for r in rows))


if __name__ == "__main__":
    unittest.main()
