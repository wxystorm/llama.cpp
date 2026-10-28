#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys

RUN_RE = re.compile(
    r"\[LT_RUN\].*?wall_ms=([0-9.]+).*?checksum=([-+0-9.eE]+).*?status=([A-Z]+)"
)


def rel_diff(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def run_case(args, name, mode, layers, extra):
    cmd = [
        args.bench,
        "--connect", args.endpoint,
        "--model", args.model,
        "--prompt-file", args.prompt_file,
        "--mode", mode,
        "--layers", str(layers),
        "--chunk-tokens", str(args.chunk_tokens),
        "--threads", str(args.threads),
    ]
    cmd.extend(extra)
    if args.trace:
        cmd.append("--trace")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout,
    )

    combined = proc.stdout + "\n" + proc.stderr
    match = RUN_RE.search(combined)

    result = {
        "name": name,
        "rc": proc.returncode,
        "ok": False,
        "wall_ms": None,
        "checksum": None,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    if match is not None:
        result["wall_ms"] = float(match.group(1))
        result["checksum"] = float(match.group(2))
        result["ok"] = (
            proc.returncode == 0 and
            match.group(3) == "PASS"
        )

    return result


def print_case(result, reference=None, tol=5e-3):
    if not result["ok"]:
        print(
            f"{result['name']:<28} "
            f"RUN_FAIL rc={result['rc']}"
        )
        return

    suffix = ""
    if reference is not None and reference.get("ok"):
        rel = rel_diff(
            reference["checksum"],
            result["checksum"],
        )
        suffix = (
            f" rel={rel:.6g} "
            f"{'PASS' if rel <= tol else 'FAIL'}"
        )

    print(
        f"{result['name']:<28} "
        f"wall={result['wall_ms']:.3f} ms "
        f"checksum={result['checksum']:.12f}"
        f"{suffix}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Small numerical correctness matrix for the standalone "
            "Layer/Tensor PC/Phone runtime."
        )
    )
    parser.add_argument("--bench", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pc-ratio",
        type=float,
        default=0.333333,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "keep per-layer [LT_NUMERIC] and "
            "[LT_CONT_COPY_CHECK] diagnostics"
        ),
    )
    args = parser.parse_args()

    cases = [
        ("L1 PC", "layer", 1, [
            "--layer-owners", "pc",
        ]),
        ("L1 Phone", "layer", 1, [
            "--layer-owners", "phone",
        ]),
        ("L2 all PC", "layer", 2, [
            "--layer-owners", "pc,pc",
        ]),
        ("L2 PC->Phone", "layer", 2, [
            "--layer-owners", "pc,phone",
        ]),
        ("L2 Phone->PC", "layer", 2, [
            "--layer-owners", "phone,pc",
        ]),
        ("L2 all Phone", "layer", 2, [
            "--layer-owners", "phone,phone",
        ]),
        ("L2 Tensor", "tensor", 2, [
            "--first-owner", "pc",
            "--pc-ratio",
            f"{args.pc_ratio:.8f}",
        ]),
        ("L4 all PC", "layer", 4, [
            "--layer-owners", "pc,pc,pc,pc",
        ]),
        ("L4 PC,PC,Phone,Phone", "layer", 4, [
            "--layer-owners", "pc,pc,phone,phone",
        ]),
        ("L4 Phone,Phone,PC,PC", "layer", 4, [
            "--layer-owners", "phone,phone,pc,pc",
        ]),
        ("L4 Tensor", "tensor", 4, [
            "--first-owner", "pc",
            "--pc-ratio",
            f"{args.pc_ratio:.8f}",
        ]),
    ]

    results = {}

    print(
        f"correctness matrix XT={args.chunk_tokens} "
        f"pc_ratio={args.pc_ratio:.6f} "
        f"tol={args.tol:.6g}"
    )
    print(
        "Phone worker must already be running with --serve."
    )

    for name, mode, layers, extra in cases:
        result = run_case(
            args,
            name,
            mode,
            layers,
            extra,
        )
        results[name] = result

        if layers == 1:
            reference = results.get("L1 PC")
        elif layers == 2:
            reference = results.get("L2 all PC")
        else:
            reference = results.get("L4 all PC")

        print_case(
            result,
            None if name in (
                "L1 PC",
                "L2 all PC",
                "L4 all PC",
            ) else reference,
            args.tol,
        )

        if not result["ok"]:
            if result["stderr"]:
                print(
                    result["stderr"].rstrip(),
                    file=sys.stderr,
                )
            return 2

    failures = []

    for name, result in results.items():
        if name in (
            "L1 PC",
            "L2 all PC",
            "L4 all PC",
        ):
            continue

        layers = (
            1 if name.startswith("L1")
            else 2 if name.startswith("L2")
            else 4
        )

        reference = (
            results["L1 PC"]
            if layers == 1
            else results["L2 all PC"]
            if layers == 2
            else results["L4 all PC"]
        )

        if rel_diff(
                reference["checksum"],
                result["checksum"]) > args.tol:
            failures.append(name)

    print()

    if failures:
        print(
            "CORRECTNESS FAIL: " +
            ", ".join(failures)
        )
        print(
            "Re-run a failing case with --trace and inspect "
            "the first divergent [LT_NUMERIC] layer plus "
            "[LT_CONT_COPY_CHECK]."
        )
        return 3

    print("CORRECTNESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
