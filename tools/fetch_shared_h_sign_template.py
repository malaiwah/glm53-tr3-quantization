#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from release_gate import canonical_sha256


def parse_layers(value: str) -> list[int]:
    layers = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def range_get(url: str, start: int, stop: int, attempts: int = 4) -> bytes:
    expected = stop - start + 1
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{stop}",
                    "User-Agent": "glm53-shared-h-sign-template/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) != expected:
                raise RuntimeError(f"range {start}-{stop}: got {len(payload)}, expected {expected}")
            return payload
        except Exception as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"range request failed: {last}")


def fetch_layer(repo: str, revision: str, layer: int) -> tuple[str, dict]:
    filename = f"model-layer-{layer:03d}.safetensors"
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    prefix = range_get(url, 0, 7)
    header_size = struct.unpack("<Q", prefix)[0]
    header_bytes = range_get(url, 8, 7 + header_size)
    header = json.loads(header_bytes)
    keys = [
        *(f"model.layers.{layer}.mlp.experts.shared_h.gate_proj.rank{rank}.suh" for rank in range(4)),
        *(f"model.layers.{layer}.mlp.experts.shared_h.up_proj.rank{rank}.suh" for rank in range(4)),
        *(f"model.layers.{layer}.mlp.experts.shared_h.down_proj.rank{rank}.svh" for rank in range(4)),
    ]
    for key in keys:
        metadata = header[key]
        if metadata["dtype"] != "F16" or metadata["shape"] not in ([6144], [1, 6144]):
            raise RuntimeError(f"{key}: unexpected {metadata['dtype']} {metadata['shape']}")
    starts = [header[key]["data_offsets"][0] for key in keys]
    stops = [header[key]["data_offsets"][1] for key in keys]
    payload_start, payload_stop = min(starts), max(stops)
    if payload_stop - payload_start != 12 * 6144 * 2:
        raise RuntimeError(f"layer {layer}: shared tensors are not one contiguous payload")
    data_start = 8 + header_size
    shared_payload = range_get(
        url, data_start + payload_start, data_start + payload_stop - 1
    )

    def signs_for(key: str) -> bytes:
        start, stop = header[key]["data_offsets"]
        relative = start - payload_start
        tensor = shared_payload[relative : relative + stop - start]
        return bytes((tensor[index + 1] >> 7) for index in range(0, len(tensor), 2))

    gate_rows = [signs_for(key) for key in keys[:8]]
    if len(set(gate_rows)) != 1:
        raise RuntimeError(f"layer {layer}: gate/up shared signs differ")
    gate = gate_rows[0]
    down = {}
    for rank, key in enumerate(keys[8:]):
        signs = signs_for(key)
        down[str(rank)] = {
            "base64_u8_signbit": base64.b64encode(signs).decode(),
            "sha256": hashlib.sha256(signs).hexdigest(),
        }
    return str(layer), {
        "gate_up": {
            "base64_u8_signbit": base64.b64encode(gate).decode(),
            "sha256": hashlib.sha256(gate).hexdigest(),
        },
        "down": down,
        "source_file": filename,
        "source_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Range-fetch all proven shared-H sign rows without downloading model shards")
    parser.add_argument("--repo", default="willfalco/GLM-5.2-EXL3-TR3-3.42bpw")
    parser.add_argument("--revision", default="a350292cb2038f2c31732569a711a89e5d72fd46")
    parser.add_argument("--layers", default="3-78")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda layer: fetch_layer(args.repo, args.revision, layer), layers))
    layer_map = dict(results)
    body = {
        "schema": "glm53-shared-h-sign-template/1",
        "source_repo": args.repo,
        "source_revision": args.revision,
        "hidden_size": 6144,
        "layers": layer_map,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval": "HTTP range: safetensors header plus contiguous shared-H payload only",
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".new-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"layers": len(layer_map), "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
