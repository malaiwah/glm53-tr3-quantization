#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def model_info(repo: str, token: str) -> dict:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "glm53-release-download/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def notify(message: str, title: str, priority: str = "default") -> None:
    url, token = os.environ.get("NTFY_URL"), os.environ.get("NTFY_TOKEN")
    if not url:
        return
    headers = {"Title": title, "Priority": priority}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=message.encode(), method="POST", headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            pass
    except Exception as exc:
        print(f"notification failed: {exc}", flush=True)


def local_progress(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts]
    return len(files), sum(path.stat().st_size for path in files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/jl_fs"))
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise SystemExit("release revision must be a full commit SHA")
    token = args.token_file.read_text().strip()
    root = args.root.resolve()
    slug = args.repo.replace("/", "--")
    model_root = (args.dest or root / f"models/downloads/{slug}").resolve()
    receipts = root / f"receipts/downloads/{slug}"
    receipts.mkdir(parents=True, exist_ok=True)
    info = model_info(args.repo, token)
    if info.get("sha") != args.revision:
        raise SystemExit(f"Hub revision moved before pin: {info.get('sha')} != {args.revision}")
    siblings = sorted(
        row["rfilename"] for row in info.get("siblings", [])
        if isinstance(row, dict) and isinstance(row.get("rfilename"), str)
    )
    remote_manifest = {
        "schema": "hf-source-remote-manifest/1",
        "repo": args.repo,
        "revision": args.revision,
        "files": siblings,
        "api_payload_sha256": canonical_sha256(info),
        "used_storage": info.get("usedStorage"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    remote_manifest["receipt_sha256"] = canonical_sha256(remote_manifest)
    atomic_json(receipts / "remote-manifest.json", remote_manifest)
    expected_bytes = int(info.get("usedStorage") or 2_000_000_000_000)
    free = shutil.disk_usage(root).free
    if free < expected_bytes + 1_500_000_000_000:
        raise SystemExit(f"disk gate: free={free}, source={expected_bytes}, reserve=1500000000000")

    metadata = [name for name in siblings if name in {
        "config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
        "LICENSE", "README.md",
    }]
    env = dict(os.environ, HF_TOKEN=token, HF_XET_HIGH_PERFORMANCE="1",
               HF_HOME=str(root / f"cache/hf-downloads/{slug}"))
    model_root.mkdir(parents=True, exist_ok=True)
    if metadata:
        subprocess.run(
            [str(args.hf), "download", args.repo, *metadata, "--revision", args.revision,
             "--local-dir", str(model_root)], check=True, env=env,
        )
    (model_root / "revision.txt").write_text(args.revision + "\n")
    preflight_rc = None
    if not args.skip_preflight:
        preflight = root / "release/glm53-main-release-smoke/tools/preflight.py"
        preflight_out = receipts / "topology-preflight.json"
        preflight_result = subprocess.run(
            [str(Path(args.hf).parent / "python"), str(preflight), "--candidate", str(model_root),
             "--out", str(preflight_out)], check=False, env=env,
        )
        preflight_rc = preflight_result.returncode
    notify(
        f"Pinned {args.repo}@{args.revision}; topology preflight rc={preflight_rc}; full download starting.",
        f"{args.repo} metadata pinned", "high",
    )

    log_path = receipts / "download.log"
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(args.hf), "download", args.repo, "--revision", args.revision,
             "--local-dir", str(model_root), "--max-workers", str(args.max_workers)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        last_heartbeat = started
        while True:
            try:
                rc = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                now = time.time()
                if now - last_heartbeat >= 600:
                    count, size = local_progress(model_root)
                    notify(
                        f"files={count}/{len(siblings)}; local={size/1e9:.1f} GB; elapsed={(now-started)/60:.0f} min",
                        f"{args.repo} download progress",
                    )
                    last_heartbeat = now
    if rc != 0:
        notify(f"Source download failed rc={rc}; see {log_path}", f"{args.repo} download FAILED", "urgent")
        return rc

    actual = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.name != "revision.txt"
    }
    missing = sorted(set(siblings) - actual)
    extra = sorted(actual - set(siblings))
    if missing or extra:
        raise SystemExit(f"download census mismatch: missing={missing[:8]} extra={extra[:8]}")
    count, size = local_progress(model_root)
    receipt = {
        "schema": "hf-source-download/1",
        "repo": args.repo,
        "revision": args.revision,
        "files": len(actual),
        "bytes": size,
        "remote_manifest_sha256": remote_manifest["receipt_sha256"],
        "topology_preflight_rc": preflight_rc,
        "elapsed_seconds": time.time() - started,
        "complete": True,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_json(receipts / "SOURCE_DOWNLOAD_COMPLETE.json", receipt)
    notify(
        f"{args.repo}@{args.revision} complete: {size/1e12:.3f} TB in {receipt['elapsed_seconds']/60:.1f} min",
        f"{args.repo} download COMPLETE", "urgent",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
