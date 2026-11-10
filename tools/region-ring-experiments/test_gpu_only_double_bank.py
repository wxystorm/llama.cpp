"""Deterministic timeline regression for the GPU-only double-bank validator."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gpu_only_verify", Path(__file__).with_name("verify_gpu_only_double_bank.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def events(overlap=True):
    items = [{"type": "measurement_start"}]
    items.append({
        "type": "region_ring", "ubatch": 5, "tokens": 512,
        "layout": "G8,G8,G8,G8", "gpu_bank_streaming": True,
        "complete": True, "status": 0, "cpu_tasks": 0,
        "phone_tasks": 0, "gpu_tasks": 16, "bank_refills": 2,
    })
    # rank0 = A, rank1 = B; after draining rank0, upload rank2 into A
    for region in range(4):
        items.append({
            "type": "region_ring_chunk_trace", "ubatch": 5,
            "device": "GPU", "region": region, "chunk": 0,
            "complete_ms": 10 + region * 12,
            "lane_compute_begin_ms": region * 12 + 2,
            "lane_compute_end_ms": region * 12 + 9,
        })
    for target in (2, 3):
        source = target - 2
        begin = 14 + source * 12 if overlap else 200 + source * 12
        items.append({
            "type": "region_ring_bank_trace", "ubatch": 5,
            "region": target, "source_region": source, "status": 1,
            "upload_bytes": 1024,
            "enqueue_ms": 11 + source * 12,
            "h2d_begin_ms": begin, "h2d_end_ms": begin + 5,
        })
    return items


class GPUOnlyBankTimingTests(unittest.TestCase):
    def test_valid_overlap(self):
        result = mod.verify(events(), 4)
        self.assertEqual(result["refills"], 2)
        self.assertGreater(result["overlapped_refills"], 0)

    def test_no_overlap_is_reported_not_falsified(self):
        result = mod.verify(events(False), 4)
        self.assertEqual(result["overlapped_refills"], 0)

    def test_cpu_compute_is_rejected(self):
        rows = events()
        rows[1]["cpu_tasks"] = 1
        with self.assertRaisesRegex(ValueError, "CPU/Phone"):
            mod.verify(rows, 4)

    def test_wrong_bank_source_is_rejected(self):
        rows = events()
        rows[-1]["source_region"] = 0
        with self.assertRaisesRegex(ValueError, "parity"):
            mod.verify(rows, 4)

    def test_refill_before_drain_is_rejected(self):
        rows = events()
        rows[-1]["enqueue_ms"] = 1
        with self.assertRaisesRegex(ValueError, "before"):
            mod.verify(rows, 4)


if __name__ == "__main__":
    unittest.main()
