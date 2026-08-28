#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

CAPTURE_SHA256 = "7b9a6daeedb96872e1dff14bd7d395d5758fa6bb06b01acd06a1adbbac796a4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hash-pinned TP8 H200 CPU-offload capture adapter")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    capture_source = source / "capture_b300.py"
    if sha256_file(capture_source) != CAPTURE_SHA256:
        raise SystemExit("capture source SHA-256 differs")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    path = out / "capture_b300.py"
    text = path.read_text()
    replacements = [
        (
            '        kv_cache_memory_bytes=args.kv_cache_bytes,',
            '        kv_cache_memory_bytes=args.kv_cache_bytes,\n        cpu_offload_gb=args.cpu_offload_gb,',
            "LLM CPU offload",
        ),
        (
            '        "cpu_offload_gb": 0,',
            '        "cpu_offload_gb": args.cpu_offload_gb,',
            "offload receipt",
        ),
        (
            '    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)',
            '    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)\n'
            '    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)',
            "offload CLI",
        ),
        (
            '        f"loading BF16 source TP{CAPTURE_TP}, no CPU offload/low-memory mode; "',
            '        f"loading BF16 source TP{CAPTURE_TP}, cpu_offload_gb={args.cpu_offload_gb}; "',
            "offload log",
        ),
        (
            'parser = argparse.ArgumentParser(description="GLM-5.2 TP8 BF16 RAM capture for B300")',
            'parser = argparse.ArgumentParser(description="GLM TP8 BF16 RAM capture with optional H200 CPU offload")',
            "description",
        ),
        (
            '        if tuple(status["capability"]) != (10, 3):',
            '        if tuple(status["capability"]) not in {(10, 3), (9, 0)}:',
            "H200 capability admission",
        ),
        (
            '''                f"rank {status['rank']}: expected B300 capability (10, 3), "''',
            '''                f"rank {status['rank']}: expected B300/H200 capability, "''',
            "capability error",
        ),
    ]
    for old, new, label in replacements:
        text = replace_once(text, old, new, label)
    path.write_text(text)
    receipt = out / "H200_OFFLOAD_ADAPTER.txt"
    receipt.write_text(
        f"source_capture_sha256={CAPTURE_SHA256}\n"
        f"patched_capture_sha256={sha256_file(path)}\n"
        "capture_tp=8\n"
        "cpu_offload=explicit-per-gpu\n"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
