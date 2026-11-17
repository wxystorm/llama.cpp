#!/usr/bin/env python3
"""Termux monotonic-clock responder. Run alongside PRIMA Rank 1.

Wire format: client sends PING newline; server captures t1 on receipt and
t2 immediately before sending: "<t1>,<t2>\n". Clock is time.monotonic_ns().
"""
import argparse
import socket
import time

def serve(bind: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind, port))
        server.listen(16)
        print(f"[PRIMA_FIG3_CLOCK] listening {bind}:{port}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(3.0)
                data = bytearray()
                while len(data) < 16 and not data.endswith(b"\n"):
                    part = conn.recv(16 - len(data))
                    if not part:
                        break
                    data.extend(part)
                if data != b"PING\n":
                    continue
                t1 = time.monotonic_ns()
                t2 = time.monotonic_ns()
                conn.sendall(f"{t1},{t2}\n".encode("ascii"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=19139)
    args = ap.parse_args()
    serve(args.host, args.port)

if __name__ == "__main__":
    main()
