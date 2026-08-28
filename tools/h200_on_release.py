#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GLM-5.3 BF16 then run H200 capture")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    project = args.root / "release/glm53-main-release-smoke"
    download = subprocess.run([
        str(project / "tools/on_release.py"),
        "--repo", args.repo,
        "--revision", args.revision,
        "--token-file", str(args.token_file),
        "--hf", str(args.hf),
        "--root", str(args.root),
        "--max-workers", "8",
    ], check=False)
    if download.returncode:
        return download.returncode
    slug = args.repo.replace("/", "--")
    source = args.root / f"models/downloads/{slug}"
    receipt = args.root / f"receipts/downloads/{slug}/SOURCE_DOWNLOAD_COMPLETE.json"
    adapter = args.root / "work/h200-offload-adapter"
    capture = subprocess.run([
        str(args.root / "venvs/h200-vllm/bin/python"),
        str(project / "tools/h200_release_capture.py"),
        "--source", str(source),
        "--source-receipt", str(receipt),
        "--revision", args.revision,
        "--capture-adapter", str(adapter),
        "--python", str(args.root / "venvs/h200-vllm/bin/python"),
        "--corpus", str(adapter / "calibration/reap_recall_calib.jsonl"),
        "--plan", str(args.root / "work/glm53-reduced-capture-plan.json"),
        "--export-root", str(args.root / "capture-export"),
        "--receipt", str(args.root / "receipts/h200-release-capture.json"),
        "--target-tokens", "131072",
        "--cpu-offload-gb", "70",
    ], check=False)
    return capture.returncode

if __name__ == "__main__":
    raise SystemExit(main())
