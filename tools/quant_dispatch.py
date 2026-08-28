#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from release_gate import canonical_sha256

WINDOWS = ("3-10", "11-18", "19-26", "27-34", "35-42", "43-50", "51-58", "59-66", "67-74", "75-77")


def atomic_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)


def sealed_state(path: Path, state: dict) -> None:
    body = {key: value for key, value in state.items() if key != "receipt_sha256"}
    atomic_json(path, {**body, "receipt_sha256": canonical_sha256(body)}, mode=0o600)


def parse_layers(window: str) -> list[int]:
    start, stop = map(int, window.split("-", 1))
    return list(range(start, stop + 1))


def run(command: list[str], *, env: dict | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, env=env, text=True, capture_output=capture)


def resume_node(args, node: dict) -> dict:
    env = dict(os.environ, JL_API_KEY=args.jarvis_key_file.read_text().strip())
    command = [
        args.jarvis_python,
        str(args.tools / "jl_resume_ssh.py"),
        "--machine-id", str(node["machine_id"]),
        "--alias", node["alias"],
        "--user", "root",
        "--gpu", "RTX-PRO6000",
        "--num-gpus", "4",
        "--spot",
        "--storage", "100",
        "--fs-id", str(args.fs_id),
    ]
    result = run(command, env=env, capture=True)
    updated = json.loads(result.stdout)
    return {**node, "machine_id": int(updated["machine_id"]), "status": "Running"}


def pause_node(args, node: dict) -> None:
    if node.get("status") != "Running":
        return
    env = dict(os.environ, JL_API_KEY=args.jarvis_key_file.read_text().strip())
    try:
        run([args.jl, "pause", str(node["machine_id"]), "--yes", "--json"], env=env)
    finally:
        node["status"] = "Paused"


def remote(alias: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", alias, command], check=check, text=True)


def encode_command(args, node: dict, profile: str, window: str, capture_plan: Path) -> list[str]:
    bits = "3" if profile == "k3" else "4"
    work = args.work_root / f"{profile}-shared-h"
    return [
        "ssh", node["alias"], "env",
        f"TR3_BITS={bits}", "EXL3_ARCH_LIST=12.0",
        f"EXLLAMAV3_EXT_PREBUILT={args.prebuilt_extension}",
        f"B300_DISK_RESERVE_BYTES={args.disk_reserve_bytes}",
        str(args.python), str(args.adapter / "encode_b300.py"),
        "--encode", "--base-encoder", str(args.adapter / "encode_tr3_v31.py"),
        "--src", str(args.source), "--work", str(work),
        "--layers", window, "--workers", "4", "--gpus", "4",
        "--capture-dir", "/dev/shm/glm53-production-capture",
        "--capture-manifest", str(capture_plan),
        "--min-routed", "1024", "--out-scales", "auto", "--lockstep", "12",
    ]


def verify_done(args, profile: str, layers: list[int]) -> None:
    for layer in layers:
        done_path = args.work_root / f"{profile}-shared-h" / f"layer-{layer:03d}.done.json"
        done = json.loads(done_path.read_text())
        if (
            done.get("rotation_layout") != "shared_h_v1"
            or int(done.get("tensor_count", -1)) != 9_228
            or int(done.get("source_expert_tensor_count", -1)) != 768
            or done.get("keep_nvfp4") != []
        ):
            raise RuntimeError(f"{profile} layer {layer} receipt did not close shared-H contract")

def assemble_uniform(args, bits: int, capture_manifest: Path, output: Path) -> None:
    if (output / "MANIFEST.sha256").is_file():
        return
    env = dict(os.environ, TR3_BITS=str(bits))
    run([
        args.assembly_python,
        str(args.adapter / "encode_b300.py"),
        "--assemble",
        "--base-encoder", str(args.adapter / "encode_tr3_v31.py"),
        "--src", str(args.source),
        "--work", str(args.work_root / f"k{bits}-shared-h"),
        "--out", str(output),
        "--capture-manifest", str(capture_manifest),
        "--io-workers", "8",
    ], env=env)
    if not (output / "MANIFEST.sha256").is_file():
        raise RuntimeError(f"K{bits} assembly produced no manifest")

def start_private_upload(args, profile: str, repository: str, output: Path):
    log_path = args.receipt.parent / f"upload-{profile}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    env = dict(os.environ, HF_TOKEN=args.hf_token_file.read_text().strip())
    process = subprocess.Popen([
        args.hf, "upload-large-folder", repository, str(output),
        "--repo-type", "model", "--private", "--num-workers", "4", "--no-bars",
    ], env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return process, handle

def build_artifact_manifest(args, profile: str, output: Path) -> Path:
    manifest = args.receipt.parent / "artifacts" / f"{profile}.json"
    run([
        args.assembly_python, str(args.tools / "build_artifact_manifest.py"),
        "--root", str(output),
        "--source-revision", args.revision,
        "--profile", profile,
        "--out", str(manifest),
    ])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume two 4-GPU Jarvis workers only when sealed capture windows are ready")
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--jarvis-key-file", type=Path, required=True)
    parser.add_argument("--jarvis-python", required=True)
    parser.add_argument("--jl", required=True)
    parser.add_argument("--fs-id", type=int, default=3423)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--prebuilt-extension", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--assembly-python", required=True)
    parser.add_argument("--tier-bitmap", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--disk-reserve-bytes", type=int, default=274_877_906_944)
    parser.add_argument("--max-runtime-hours", type=float, default=16)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--k3-repo", default="malaiwah/GLM-5.3-TR3-3bpw")
    parser.add_argument("--k4-repo", default="malaiwah/GLM-5.3-TR3-4bpw")
    parser.add_argument("--mixed-repo", default="malaiwah/GLM-5.3-TR3-3.42bpw")
    args = parser.parse_args()
    args.tools = args.tools.resolve()
    if len(args.revision) != 40:
        raise SystemExit("revision must be a full SHA")
    deadline = time.time() + args.max_runtime_hours * 3600
    contract = (args.adapter / "UNIFORM_ADAPTER.txt").read_text()
    if "rotation_layout=shared_h_v1" not in contract:
        raise SystemExit("adapter is not sealed shared_h_v1")
    while not args.source_receipt.is_file() or not (args.source / "revision.txt").is_file():
        if time.time() >= deadline:
            raise SystemExit("source download did not close before dispatcher deadline")
        time.sleep(args.poll_seconds)
    source_receipt = json.loads(args.source_receipt.read_text())
    if source_receipt.get("complete") is not True or source_receipt.get("revision") != args.revision:
        raise SystemExit("source download receipt is incomplete or revision-mismatched")
    if (args.source / "revision.txt").read_text().strip() != args.revision:
        raise SystemExit("source tree revision differs")
    nodes_state = json.loads(args.nodes.read_text())
    nodes = nodes_state["nodes"]
    if {node["profile"] for node in nodes} != {"k3", "k4"}:
        raise SystemExit("node contract must contain one K3 and one K4 worker")
    cleanup_armed = {"value": False}

    def cleanup() -> None:
        if not cleanup_armed["value"]:
            return
        cleanup_armed["value"] = False
        for node in nodes:
            try:
                pause_node(args, node)
            except Exception as exc:
                print(f"pause failed for {node['profile']}: {exc}", flush=True)
        nodes_state["nodes"] = nodes
        nodes_state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sealed_state(args.nodes, nodes_state)

    def stop(signum, _frame) -> None:
        raise SystemExit(f"quant dispatcher received signal {signum}")

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    first = args.capture_root / args.revision / WINDOWS[0] / "READY.json"
    while not first.is_file():
        if time.time() >= deadline:
            raise SystemExit("first capture window did not arrive before dispatcher deadline")
        time.sleep(args.poll_seconds)
    resumed = []
    try:
        # Aliases share one SSH config file, so resume serially to keep updates atomic.
        for index, node in enumerate(nodes):
            nodes[index] = resume_node(args, node)
            resumed.append(nodes[index])
            nodes_state["nodes"] = nodes
            nodes_state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            sealed_state(args.nodes, nodes_state)
        cleanup_armed["value"] = True
        for node in nodes:
            remote(node["alias"], f"SKIP_CUDA_TOOLKIT=1 {shlex.quote(str(args.tools / 'prepare_jarvis_node.sh'))} quant-{node['profile']}")
            gpu_count = subprocess.run(
                ["ssh", node["alias"], "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if gpu_count != "4":
                raise RuntimeError(f"{node['profile']} worker exposes {gpu_count} GPUs, expected 4")

        completed_windows = []
        for window in WINDOWS:
            ready = args.capture_root / args.revision / window / "READY.json"
            while not ready.is_file():
                if time.time() >= deadline:
                    raise SystemExit("quant runtime cap reached while waiting for capture")
                time.sleep(args.poll_seconds)
            window_root = ready.parent
            capture_plan = window_root / "capture_plan.json"
            copy_processes = []
            for node in nodes:
                copy_command = (
                    "rm -rf /dev/shm/glm53-production-capture && "
                    "mkdir -p /dev/shm/glm53-production-capture && "
                    f"rsync -a {shlex.quote(str(window_root / 'capture'))}/ /dev/shm/glm53-production-capture/"
                )
                copy_processes.append(subprocess.Popen(["ssh", node["alias"], copy_command]))
            copy_codes = [process.wait() for process in copy_processes]
            if copy_codes != [0, 0]:
                raise RuntimeError(f"capture copy failed for window {window}: {copy_codes}")
            encoders = [
                subprocess.Popen(encode_command(args, node, node["profile"], window, capture_plan))
                for node in nodes
            ]
            codes = [process.wait() for process in encoders]
            if codes != [0, 0]:
                raise RuntimeError(f"window {window} encoders exited {codes}")
            layers = parse_layers(window)
            verify_done(args, "k3", layers)
            verify_done(args, "k4", layers)
            for node in nodes:
                remote(node["alias"], "rm -rf /dev/shm/glm53-production-capture")
            completed_windows.append(window)
            print(f"QUANT_WINDOW_COMPLETE {window}", flush=True)
        cleanup()
        args.outputs_root.mkdir(parents=True, exist_ok=True)
        capture_manifest = (
            args.capture_root / args.revision / WINDOWS[0] / "capture_plan.json"
        )
        flat_k3 = args.outputs_root / "GLM-5.3-TR3-3bpw"
        flat_k4 = args.outputs_root / "GLM-5.3-TR3-4bpw"
        mixed_output = args.outputs_root / "GLM-5.3-TR3-3.42bpw"
        uploads = []
        assemble_uniform(args, 3, capture_manifest, flat_k3)
        k3_manifest = build_artifact_manifest(args, "flat-k3", flat_k3)
        uploads.append(start_private_upload(args, "k3", args.k3_repo, flat_k3))
        assemble_uniform(args, 4, capture_manifest, flat_k4)
        k4_manifest = build_artifact_manifest(args, "flat-k4", flat_k4)
        uploads.append(start_private_upload(args, "k4", args.k4_repo, flat_k4))
        mixed_work = args.work_root / "mixed-3.42-shared-h"
        mixed_receipt = args.receipt.parent / "mixed-parts.json"
        run([
            args.assembly_python, str(args.tools / "mixed_materialize.py"),
            "--adapter", str(args.adapter),
            "--base-encoder", str(args.adapter / "encode_tr3_v31.py"),
            "--k3-work", str(args.work_root / "k3-shared-h"),
            "--k4-work", str(args.work_root / "k4-shared-h"),
            "--tier-bitmap", str(args.tier_bitmap),
            "--out", str(mixed_work),
            "--layers", "3-77",
            "--receipt", str(mixed_receipt),
        ])
        mixed_checkpoint_receipt = args.receipt.parent / "mixed-checkpoint.json"
        if not (mixed_output / "MANIFEST.sha256").is_file():
            run([
                args.assembly_python, str(args.tools / "assemble_mixed_checkpoint.py"),
                "--adapter", str(args.adapter),
                "--base-encoder", str(args.adapter / "encode_tr3_v31.py"),
                "--source", str(args.source),
                "--capture-manifest", str(capture_manifest),
                "--mixed-work", str(mixed_work),
                "--mixed-receipt", str(mixed_receipt),
                "--out", str(mixed_output),
                "--io-workers", "8",
                "--receipt", str(mixed_checkpoint_receipt),
            ])
        mixed_manifest = build_artifact_manifest(args, "mixed-3.42", mixed_output)
        uploads.append(
            start_private_upload(args, "mixed-3.42", args.mixed_repo, mixed_output)
        )
        upload_codes = []
        for process, handle in uploads:
            upload_codes.append(process.wait())
            handle.close()
        if upload_codes != [0, 0, 0]:
            raise RuntimeError(f"private uploads exited {upload_codes}")
        body = {
            "schema": "glm53-core-quant-dispatch/1",
            "revision": args.revision,
            "rotation_layout": "shared_h_v1",
            "profiles": ["k3", "k4", "mixed-3.42"],
            "windows": completed_windows,
            "layers": list(range(3, 78)),
            "workers_paused": True,
            "outputs": {
                "k3": str(flat_k3),
                "k4": str(flat_k4),
                "mixed_3_42": str(mixed_output),
            },
            "artifact_manifests": {
                "k3": str(k3_manifest),
                "k4": str(k4_manifest),
                "mixed_3_42": str(mixed_manifest),
            },
            "private_uploads": {
                "k3": args.k3_repo,
                "k4": args.k4_repo,
                "mixed_3_42": args.mixed_repo,
            },
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(args.receipt, {**body, "receipt_sha256": canonical_sha256(body)})
        return 0
    except Exception:
        cleanup_armed["value"] = bool(resumed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
