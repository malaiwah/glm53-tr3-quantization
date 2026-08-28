#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENCODER_SHA256 = "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"
EXLLAMA_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
EXLLAMA_VERSION = "0.0.43"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def arch_for(capability: tuple[int, int]) -> str:
    major, minor = capability
    if major == 12:
        return "12.0a"
    if major == 10:
        return "10.0"
    return f"{major}.{minor}"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the exact GLM-5.2 v3.1 calibrated Trellis core at K3 and K4"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.bits or any(bits not in (3, 4) for bits in args.bits):
        raise SystemExit("--bits must contain only 3 and/or 4")

    root = args.root.resolve()
    encoder_dir = root / "upstream/brandon-glm52-v31/calibration_encoder"
    encoder_path = encoder_dir / "encode_tr3_v31.py"
    bootstrap_path = encoder_dir / "bootstrap_ext_b300.py"
    exllama_root = root / "upstream/exllamav3-v0.0.43"
    out = (args.out or root / "receipts/gpu-codec-smoke.json").resolve()
    log_path = out.with_suffix(".log")
    out.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    receipt = {
        "schema": "glm53-mainline-gpu-codec-smoke/1",
        "passed": False,
        "bits": list(args.bits),
        "encoder_sha256": sha256_file(encoder_path),
        "exllamav3_commit": git_head(exllama_root),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if receipt["encoder_sha256"] != ENCODER_SHA256:
        errors.append("production encoder SHA-256 differs")
    if receipt["exllamav3_commit"] != EXLLAMA_COMMIT:
        errors.append("exllamav3 source commit differs")
    try:
        installed = importlib.metadata.version("exllamav3").split("+", 1)[0]
    except importlib.metadata.PackageNotFoundError:
        installed = None
    receipt["exllamav3_version"] = installed
    if installed != EXLLAMA_VERSION:
        errors.append(f"installed exllamav3 is {installed!r}, expected {EXLLAMA_VERSION}")

    try:
        import ninja  # noqa: F401
        import torch
    except ImportError as exc:
        errors.append(f"GPU dependency unavailable: {exc}")
        torch = None
    if torch is not None:
        if not torch.cuda.is_available():
            errors.append("torch CUDA is unavailable")
        else:
            capability = tuple(torch.cuda.get_device_capability(0))
            arch = arch_for(capability) if args.arch == "auto" else args.arch
            receipt["environment"] = {
                "torch": torch.__version__,
                "torch_cuda": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "arch_list": arch,
            }
    if errors:
        receipt["errors"] = errors
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True))
        return 1

    build_dir = (args.build_dir or root / "work/torch_extensions/exllamav3_ext").resolve()
    os.environ["B300_DISK_RESERVE_BYTES"] = os.environ.get("B300_DISK_RESERVE_BYTES", "0")
    os.environ["EXL3_B300_BUILD_DIR"] = str(build_dir)
    os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "16")
    bootstrap = load_module("glm53_release_bootstrap", bootstrap_path)
    bootstrap.ARCH_LIST = receipt["environment"]["arch_list"]
    bootstrap.DEFAULT_BUILD_DIR = build_dir
    extension_source = exllama_root / "exllamav3/exllamav3_ext"
    if not (extension_source / "bindings.cpp").is_file() or not (
        extension_source / "quant/quantize.cu"
    ).is_file():
        raise RuntimeError(f"pinned extension sources are incomplete: {extension_source}")
    bootstrap.find_source_root = lambda: extension_source

    started = time.monotonic()
    bit_results = []
    try:
        with log_path.open("w") as log_handle, contextlib.redirect_stdout(
            Tee(sys.stdout, log_handle)
        ), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
            extension = bootstrap.build(verbose=False)
            functional = bootstrap._functional_smoke(extension, torch)
            print("extension functional smoke:", json.dumps(functional, sort_keys=True))
            for bits in args.bits:
                bit_started = time.monotonic()
                module_name = f"glm53_release_encoder_k{bits}"
                encoder = load_module(module_name, encoder_path)
                encoder.BITS = bits
                encoder.KEEP_NVFP4 = 0
                encoder._lazy_torch = lambda: (torch, extension)
                print(f"=== BEGIN K{bits} CALIBRATED CODEC SMOKE ===")
                encoder.smoke(argparse.Namespace())
                elapsed = time.monotonic() - bit_started
                print(f"=== END K{bits} CALIBRATED CODEC SMOKE ({elapsed:.2f}s) ===")
                bit_results.append({"bits": bits, "passed": True, "elapsed_seconds": elapsed})
                sys.modules.pop(module_name, None)
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        receipt.update({
            "passed": True,
            "errors": [],
            "extension": getattr(extension, "_b300_bootstrap", {}),
            "extension_functional_smoke": functional,
            "bit_results": bit_results,
            "log_sha256": sha256_file(log_path),
            "elapsed_seconds": time.monotonic() - started,
        })
    except Exception as exc:
        receipt["errors"] = [f"{type(exc).__name__}: {exc}"]
        if log_path.is_file():
            receipt["log_sha256"] = sha256_file(log_path)
        raise
    finally:
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        staging = out.with_suffix(out.suffix + ".tmp")
        staging.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        staging.replace(out)
    print(json.dumps({
        "passed": receipt["passed"],
        "bits": args.bits,
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "receipt": str(out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
