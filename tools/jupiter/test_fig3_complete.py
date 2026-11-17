#!/usr/bin/env python3
"""Synthetic end-to-end Jupiter Fig3 analyzer test; NO model or RPC required."""
import csv
import subprocess
import sys
import tempfile
import socket
import struct
import threading
import time
import run_fig3_complete as runner
from pathlib import Path


def csv_file(p,header,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_new_rpc_protocol():
    """Verify uncompressed HELLO, TAG and CLOCK exactly without a Phone."""
    errors=[]
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1",0))
        listener.listen(1)
        port=listener.getsockname()[1]
        def serve():
            try:
                with listener.accept()[0] as conn:
                    def exact(n):
                        data=b""
                        while len(data)<n:
                            more=conn.recv(n-len(data))
                            if not more: raise RuntimeError("test peer closed")
                            data+=more
                        return data
                    def request(cmd,size):
                        got,got_size=struct.unpack("<BQ",exact(9))
                        assert (got,got_size)==(cmd,size), (got,got_size)
                        return exact(got_size) if got_size else b""
                    assert request(14,24)==bytes(24)
                    conn.sendall(struct.pack("<Q",28)+bytes((1,0,0,0))+bytes(24))
                    tag=request(21,96)
                    assert tag.startswith(b"fixture-run")
                    conn.sendall(struct.pack("<Q",1)+b"\x01")
                    for _ in range(8):
                        request(22,0)
                        now=time.monotonic_ns()
                        conn.sendall(struct.pack("<Qqq",16,now,now+1000))
            except Exception as exc:
                errors.append(exc)
        thread=threading.Thread(target=serve,daemon=True)
        thread.start()
        runner.phone_profile_preflight("127.0.0.1",port,"fixture-run")
        thread.join(timeout=5)
        assert not errors, errors
        assert not thread.is_alive()


def main():
    script=Path(__file__).with_name("build_fig3_complete.py")
    with tempfile.TemporaryDirectory(prefix="jupiter_fig3_test_") as temp:
        root=Path(temp); profile=root/"profile"; profile.mkdir()
        offset=500_000_000; origin=1_200_000_000_000
        ms=lambda n:origin+n*1_000_000
        run="jupiter-fixture-001"
        csv_file(profile/"pc_prefill_windows.csv",
                 ["start_ns","end_ns","tokens"],[(ms(0),ms(130),100)])
        cs=["run_id","endpoint","offset_phone_minus_pc_ns","uncertainty_ns",
            "rtt_ns","t0_pc_ns","t3_pc_ns"]
        csv_file(profile/"pc_clock_sync.csv",cs,[
            (run,"phone:10139",offset,100000+i*1000,400000+i*1000,ms(0),ms(0)+400000)
            for i in range(8)])
        header=["category","start_ns","end_ns","source","device","bytes"]
        csv_file(profile/"pc_cuda_events.csv",header,[
            ("GPU Compute",ms(0),ms(42),"CUDA_DIRECT_GRAPH",0,0),
            ("PCIe Copy",ms(37),ms(51),"D2H_ASYNC",0,10240),
        ])
        csv_file(profile/"pc_cpu_events.csv",header,[])
        csv_file(profile/"pc_rpc_events.csv",header,[
            ("Wi-Fi Sync",ms(12),ms(100),"RPC_ROUNDTRIP",20,128)])
        csv_file(profile/"pc_chunks.csv",
                 ["chunk","slot","prefix_tokens","tokens","start_ns","queued_ns"],[
            (0,0,0,40,ms(0),ms(1)),
            (1,1,40,32,ms(42),ms(44)),
            (2,0,72,28,ms(84),ms(85)),
        ])
        phone=root/"phone.csv"
        csv_file(phone,["run_id","category","start_phone_ns","end_phone_ns","device","graph_uid","nodes"],[
            (run,"Phone Prepare",ms(15)+offset,ms(20)+offset,0,1,64),
            (run,"Phone Compute",ms(20)+offset,ms(83)+offset,0,1,64),
        ])
        cmd=[sys.executable,str(script),"--profile-dir",str(profile),
             "--phone-events",str(phone),"--run-id",run,"--pc-layers","24",
             "--normal-prefill-ms","117","--expected-tokens","100"]
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode:
            raise AssertionError(result.stderr+"\n"+result.stdout)
        with (profile/"fig3_jupiter_breakdown.csv").open(encoding="utf-8") as f:
            v=next(csv.DictReader(f))
        assert float(v["profile_envelope_ms"])==130.0
        assert v["full_coverage"]=="1" and v["required_sources_present"]=="1"
        assert v["phone_compute_events"]=="1" and v["chunk_events"]=="3"
        keys=["phone_compute","pcie_copy","gpu_compute","cpu_compute",
              "wi_fi_sync","waiting_other"]
        total=sum(float(v[k+"_pct"]) for k in keys)
        assert abs(total-100)<0.001, total
        with (profile/"fig3_jupiter_timeline.csv").open(encoding="utf-8") as f:
            line=list(csv.DictReader(f))
        assert line[0]["start_ms"]=="0.0"
        assert line[-1]["end_ms"]=="130.0"
        for a,b in zip(line,line[1:]):
            assert abs(float(a["end_ms"])-float(b["start_ms"]))<0.0001
        (profile/"pc_cuda_events.csv").unlink()
        fail=subprocess.run(cmd,capture_output=True,text=True)
        assert fail.returncode!=0
        assert "pc_cuda_events.csv" in fail.stderr
        test_new_rpc_protocol()
        print("[JUPITER_FIG3_TEST] PASS: 8 clock samples, native phone run ID, GPU/CPU/PCIe/RPC, chunks, 100% coverage, strict missing-source rejection, real HELLO/TAG/CLOCK packet protocol")


if __name__=="__main__":
    main()
