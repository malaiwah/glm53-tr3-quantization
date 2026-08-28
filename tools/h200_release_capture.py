#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from release_gate import canonical_sha256


def run(command: list[str], env: dict | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-load reduced-calibration TP8 H200 BF16 capture and export worker windows")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--capture-adapter", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, default=Path("/dev/shm/glm53-main-capture"))
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=131_072)
    parser.add_argument("--cpu-offload-gb", type=float, default=70.0)
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise SystemExit("revision must be a full SHA")
    source_receipt = json.loads(args.source_receipt.read_text())
    if source_receipt.get("complete") is not True or source_receipt.get("revision") != args.revision:
        raise RuntimeError("source receipt is incomplete or revision-mismatched")
    if (args.source / "revision.txt").read_text().strip() != args.revision:
        raise RuntimeError("source tree revision differs")
    started = time.time()
    env = dict(
        os.environ,
        HF_HOME=str(args.source.parent / "hf-cache"),
        PATH=f"/usr/local/cuda-13.0/bin:{os.environ.get('PATH', '')}",
        LD_LIBRARY_PATH=(
            f"/usr/local/cuda-13.0/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        ),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_LEVEL="NVL",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        VLLM_USE_DEEP_GEMM="0",
        VLLM_MOE_USE_DEEP_GEMM="0",
        VLLM_DEEP_GEMM_WARMUP="skip",
        B300_HOST_RAM_RESERVE_BYTES="0",
        B300_RAMFS_RESERVE_BYTES=str(2 << 30),
    )
    run([
        str(args.python), str(args.capture_adapter / "capture_b300.py"),
        "--plan", "--src", str(args.source), "--corpus", str(args.corpus),
        "--plan-file", str(args.plan), "--target-tokens", str(args.target_tokens),
        "--log", str(args.receipt.with_suffix(".plan.log")),
    ], env)
    plan = json.loads(args.plan.read_text())
    if int(plan.get("total_tokens") or 0) > args.target_tokens * 1.25:
        raise RuntimeError("reduced capture plan exceeded token allowance")
    if args.capture_dir.exists():
        shutil.rmtree(args.capture_dir)
    run([
        str(args.python), str(args.capture_adapter / "capture_b300.py"),
        "--capture", "--src", str(args.source), "--corpus", str(args.corpus),
        "--plan-file", str(args.plan), "--capture-dir", str(args.capture_dir),
        "--layers", "3-77", "--cpu-offload-gb", str(args.cpu_offload_gb),
        "--kv-cache-bytes", str(1 << 30), "--max-num-seqs", "4",
        "--min-post-load-free-bytes", str(16 << 30),
        "--max-window-layers", "75", "--verify-engine", "--fresh",
        "--log", str(args.receipt.with_suffix(".capture.log")),
    ], env)
    run([
        str(args.python), str(Path(__file__).resolve().with_name("export_capture_windows.py")),
        "--capture-dir", str(args.capture_dir), "--plan", str(args.plan),
        "--out", str(args.export_root), "--revision", args.revision,
        "--layers", "3-77", "--window-size", "8",
    ], env)
    shutil.rmtree(args.capture_dir)
    body = {
        "schema": "glm53-h200-release-capture/1",
        "revision": args.revision,
        "capture_fingerprint": plan["capture_fingerprint"],
        "tokens_per_layer": plan["total_tokens"],
        "cpu_offload_gb_per_gpu": args.cpu_offload_gb,
        "capture_tp": 8,
        "layers": [3, 77],
        "export_root": str((args.export_root / args.revision).resolve()),
        "elapsed_seconds": time.time() - started,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
