#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from release_gate import canonical_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sealed tmpfs captures as atomic quant-worker windows")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layers", default="3-77")
    parser.add_argument("--window-size", type=int, default=8)
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise SystemExit("revision must be a full SHA")
    layers = parse_layers(args.layers)
    destination = args.out / args.revision
    results = []
    for offset in range(0, len(layers), args.window_size):
        group = layers[offset : offset + args.window_size]
        window = f"{group[0]}-{group[-1]}"
        final = destination / window
        if (final / "READY.json").is_file():
            results.append(window)
            continue
        incoming = destination / f"{window}.new-{os.getpid()}"
        if incoming.exists():
            shutil.rmtree(incoming)
        capture_out = incoming / "capture"
        capture_out.mkdir(parents=True)
        for layer in group:
            source = args.capture_dir / f"layer_{layer:03d}"
            manifest = json.loads((source / "layer_manifest.json").read_text())
            if int(manifest.get("layer", -1)) != layer:
                raise RuntimeError(f"layer {layer}: capture manifest differs")
            shutil.copytree(source, capture_out / source.name)
        shutil.copy2(args.plan, incoming / "capture_plan.json")
        files = {}
        for path in sorted(incoming.rglob("*")):
            if path.is_file():
                files[path.relative_to(incoming).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        body = {
            "schema": "glm53-capture-window-export/1",
            "window": window,
            "files": files,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ready = {**body, "receipt_sha256": canonical_sha256(body)}
        (incoming / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        incoming.replace(final)
        results.append(window)
        print(json.dumps({"window": window, "bytes": sum(row["bytes"] for row in files.values())}), flush=True)
    print(json.dumps({"revision": args.revision, "windows": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
