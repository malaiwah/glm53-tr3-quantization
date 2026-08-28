#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
from pathlib import Path

from release_gate import canonical_sha256


def tensor_signs(path: Path, key: str) -> tuple[bytes, list[int]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        metadata = header[key]
        if metadata["dtype"] != "F16":
            raise RuntimeError(f"{key}: expected F16")
        start, stop = metadata["data_offsets"]
        handle.seek(8 + header_size + start)
        payload = handle.read(stop - start)
    signs = bytes((payload[index + 1] >> 7) for index in range(0, len(payload), 2))
    return signs, metadata["shape"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract only proven shared-H signs from published layer shards")
    parser.add_argument("--layer-file", action="append", required=True, metavar="L=PATH")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    layers = {}
    for item in args.layer_file:
        layer_text, path_text = item.split("=", 1)
        layer, path = int(layer_text), Path(path_text)
        gate_rows = []
        for projection in ("gate_proj", "up_proj"):
            for rank in range(4):
                key = f"model.layers.{layer}.mlp.experts.shared_h.{projection}.rank{rank}.suh"
                signs, shape = tensor_signs(path, key)
                if shape not in ([6144], [1, 6144]):
                    raise RuntimeError(f"{key}: shape {shape}")
                gate_rows.append(signs)
        if len(set(gate_rows)) != 1:
            raise RuntimeError(f"layer {layer}: gate/up H signs differ")
        down = {}
        for rank in range(4):
            key = f"model.layers.{layer}.mlp.experts.shared_h.down_proj.rank{rank}.svh"
            signs, shape = tensor_signs(path, key)
            if shape not in ([6144], [1, 6144]):
                raise RuntimeError(f"{key}: shape {shape}")
            down[str(rank)] = {
                "base64_u8_signbit": base64.b64encode(signs).decode(),
                "sha256": hashlib.sha256(signs).hexdigest(),
            }
        gate = gate_rows[0]
        layers[str(layer)] = {
            "gate_up": {
                "base64_u8_signbit": base64.b64encode(gate).decode(),
                "sha256": hashlib.sha256(gate).hexdigest(),
            },
            "down": down,
        }
    body = {
        "schema": "glm53-shared-h-sign-template/1",
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "hidden_size": 6144,
        "layers": layers,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".new-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"layers": len(layers), "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
