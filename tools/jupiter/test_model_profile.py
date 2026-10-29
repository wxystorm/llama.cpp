"""Offline tests for Jupiter model-boundary profiler and minimax selector."""
import unittest
from model_boundary import choose_boundary
from run_model_profile import parse_log

def sample_log(boundary=25):
    data = [
        "0.1 I [JUPITER_STAGE_PLAN] mode=ENV pc_device=CUDA0 pc_layers=0..24 phone_device=RPC0[phone] phone_layers=25..39 output=PHONE",
        "llama_model_load_tensors: CUDA0 model buffer size = 4500.00 MiB",
        "llama_model_load_tensors: RPC0[phone] model buffer size = 2800.00 MiB",
        "llama_context: CUDA0 compute buffer size = 150.00 MiB",
        "llama_context: RPC0[phone] compute buffer size = 100.00 MiB",
    ]
    for tokens in (2,64):
        data += [
            f"[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens={tokens} t_us=1",
            f"[JUPITER_STAGE_PROFILE] split=0 backend=CUDA0 nodes=10 inputs=0 input_ms=1 compute_ms={tokens} wall_ms={tokens+1}",
            f"[JUPITER_STAGE_PROFILE] split=1 backend=RPC0[phone] nodes=10 inputs=1 input_ms=3 compute_ms={tokens*2} wall_ms={tokens*2+3}",
            f"[JUPITER_UBATCH] event=QUEUED chunk=0 slot=0 prefix_tokens=0 tokens={tokens} t_us=2",
        ]
    return "\n".join(data)

class ModelProfileTests(unittest.TestCase):
    def test_excludes_warmup_and_reads_memory(self):
        d=parse_log(sample_log(),25,64)
        self.assertEqual(d["phone_layers"],15)
        self.assertEqual(d["pc_compute_ms"],64)
        self.assertEqual(d["phone_compute_ms"],128)
        self.assertEqual(d["pc_memory_mib"],4650)
        self.assertEqual(d["phone_memory_mib"],2900)

    def test_rejects_wrong_boundary(self):
        with self.assertRaisesRegex(ValueError,"boundary"):
            parse_log(sample_log(),24,64)

    def test_selects_measured_minimax(self):
        rows=[]
        for boundary,pc,phone in ((23,60,110),(25,85,90),(27,130,50)):
            rows.append(dict(pc_layers=str(boundary),phone_layers=str(40-boundary),
                pc_compute_ms=str(pc),phone_compute_ms=str(phone),
                pc_wall_ms=str(pc+10),phone_wall_ms=str(phone+10),
                pc_memory_mib="4800",phone_memory_mib="3200",status="OK"))
        best,feasible,fail=choose_boundary(rows)
        self.assertEqual(best.pc_layers,25)
        self.assertEqual(len(feasible),3)
        self.assertEqual(fail,0)

    def test_budget_filters_and_missing_fails(self):
        rows=[
            dict(pc_layers="25",phone_layers="15",pc_compute_ms="10",phone_compute_ms="10",
                 pc_wall_ms="11",phone_wall_ms="11",pc_memory_mib="5500",
                 phone_memory_mib="3000",status="OK"),
            dict(pc_layers="23",phone_layers="17",pc_compute_ms="20",phone_compute_ms="25",
                 pc_wall_ms="21",phone_wall_ms="26",pc_memory_mib="4400",
                 phone_memory_mib="3100",status="OK"),
        ]
        best,_,_=choose_boundary(rows,pc_budget_mib=5000)
        self.assertEqual(best.pc_layers,23)
        rows[1]["phone_memory_mib"]=""
        with self.assertRaisesRegex(ValueError,"no measured feasible"):
            choose_boundary(rows,pc_budget_mib=5000,phone_budget_mib=4000)

if __name__=="__main__":
    unittest.main()
