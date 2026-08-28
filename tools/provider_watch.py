#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from provider_funding import runpod_balance, vast_balance
from release_gate import canonical_sha256, release_revision


def atomic_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)


def run(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr).strip()[-2000:],
    }

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
    with urllib.request.urlopen(request, timeout=15):
        pass




def start_quant_dispatcher(args, revision: str) -> None:
    if args.quant_receipt.is_file():
        return
    session = "glm53-quant-dispatch"
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        return
    command = [
        sys.executable, str(args.tools / "quant_dispatch.py"),
        "--tools", str(args.tools),
        "--nodes", str(args.quant_nodes),
        "--jarvis-key-file", str(args.jarvis_key_file),
        "--jarvis-python", args.jarvis_python,
        "--jl", args.jl,
        "--source", str(args.source_root),
        "--source-receipt", str(args.source_receipt),
        "--capture-root", str(args.capture_root),
        "--revision", revision,
        "--adapter", str(args.quant_adapter),
        "--python", str(args.quant_python),
        "--prebuilt-extension", str(args.quant_prebuilt),
        "--work-root", str(args.quant_work_root),
        "--receipt", str(args.quant_receipt),
        "--assembly-python", args.assembly_python,
        "--tier-bitmap", str(args.tier_bitmap),
        "--outputs-root", str(args.outputs_root),
        "--hf", args.hf,
        "--hf-token-file", str(args.hf_token_file),
    ]
    log = args.receipts / "quant-dispatch.log"
    shell_command = f"exec {shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log))}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command],
        check=True,
    )



def start_controller(args, provider: str, private_state: Path, revision: str) -> None:
    start_quant_dispatcher(args, revision)
    controller_receipt = args.receipts / "capture-controller.json"
    if controller_receipt.is_file():
        return
    session = "glm53-capture-controller"
    active = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    if active:
        return
    command = [
        sys.executable, str(args.tools / "capture_controller.py"),
        "--provider", provider,
        "--private-state", str(private_state),
        "--runpod-token-file", str(args.runpod_token_file),
        "--vastai", args.vastai,
        "--ssh-key", str(args.ssh_key),
        "--adapter-archive", str(args.adapter_archive),
        "--adapter-sha256", str(args.adapter_sha256),
        "--worker-script", str(args.tools / "capture_worker.sh"),
        "--hf-token-file", str(args.hf_token_file),
        "--revision", revision,
        "--capture-root", str(args.capture_root),
        "--receipt", str(controller_receipt),
        "--max-runtime-hours", "7" if provider == "runpod" else "8",
    ]
    log = args.receipts / "capture-controller.log"
    shell_command = f"exec {shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log))}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command], check=True)


def launch_if_ready(args, revision: str) -> dict:
    selection_path = args.private_dir / "provider-selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        if selection.get("revision") != revision:
            raise RuntimeError("provider selection belongs to another revision")
        start_controller(args, selection["provider"], Path(selection["private_state"]), revision)
        return {"selected": selection["provider"], "existing": True}
    topology = args.topology_receipt
    if not topology.is_file():
        return {"selected": None, "reason": "topology receipt pending"}
    # RunPod count-8 inventory is still monitored, but its official template is
    # not the pinned Gilded Gnosis runtime and a custom-image SSH path has not
    # been qualified. Do not spend on an unbootstrappable capture host.
    attempts = [{
        "provider": "runpod",
        "returncode": None,
        "output": "launch disabled until Gilded Gnosis custom-image SSH is qualified",
    }]
    result = run([
        sys.executable, str(args.tools / "vast_launch_b300.py"),
        "--watch-state", str(args.watch_state),
        "--topology-receipt", str(topology),
        "--onstart", str(args.tools / "prepare_vast_capture.sh"),
        "--receipt", str(args.receipts / "vast-launch.json"),
        "--private-state", str(args.private_dir / "vast-instance.json"),
        "--vastai", args.vastai,
        "--launch",
    ])
    attempts.append({"provider": "vast", **result})
    if result["returncode"] == 0:
        selection = {
            "schema": "glm53-provider-selection-private/1",
            "provider": "vast",
            "revision": revision,
            "private_state": str(args.private_dir / "vast-instance.json"),
            "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(selection_path, selection, mode=0o600)
        start_controller(args, "vast", Path(selection["private_state"]), revision)
        return {"selected": "vast", "attempts": attempts}
    return {"selected": None, "reason": "all guarded launch paths failed", "attempts": attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll B300 providers and launch one guarded capture host after release topology closure")
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--watch-state", type=Path, required=True)
    parser.add_argument("--topology-receipt", type=Path, required=True)
    parser.add_argument("--runpod-token-file", type=Path, required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--vastai", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--adapter-archive", type=Path, required=True)
    parser.add_argument("--adapter-sha256", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--quant-nodes", type=Path, required=True)
    parser.add_argument("--jarvis-key-file", type=Path, required=True)
    parser.add_argument("--jarvis-python", required=True)
    parser.add_argument("--jl", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--quant-adapter", type=Path, required=True)
    parser.add_argument("--quant-python", type=Path, required=True)
    parser.add_argument("--quant-prebuilt", type=Path, required=True)
    parser.add_argument("--quant-work-root", type=Path, required=True)
    parser.add_argument("--quant-receipt", type=Path, required=True)
    parser.add_argument("--assembly-python", required=True)
    parser.add_argument("--tier-bitmap", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--target-epoch", type=float, default=1787929200)
    parser.add_argument("--far-interval", type=float, default=300)
    parser.add_argument("--near-interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.tools = args.tools.resolve()
    args.receipts.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.private_dir, 0o700)
    last_notification = 0.0
    while True:
        secure = run([
            sys.executable, str(args.tools / "runpod_b300_preflight.py"),
            "--token-file", str(args.runpod_token_file),
            "--out", str(args.receipts / "runpod-secure.json"), "--cloud", "SECURE",
        ])
        community = run([
            sys.executable, str(args.tools / "runpod_b300_preflight.py"),
            "--token-file", str(args.runpod_token_file),
            "--out", str(args.receipts / "runpod-community.json"), "--cloud", "COMMUNITY",
        ])
        vast = run([
            sys.executable, str(args.tools / "vast_b300_preflight.py"),
            "--out", str(args.receipts / "vast-live.json"),
            "--private-state", str(args.private_dir / "vast-inventory.json"),
            "--vastai", args.vastai,
        ])
        try:
            rp_funding = runpod_balance(args.runpod_token_file)
        except Exception as exc:
            rp_funding = {"error": str(exc)}
        try:
            vast_funding = vast_balance(args.vastai)
        except Exception as exc:
            vast_funding = {"error": str(exc)}
        revision = release_revision(args.watch_state) if args.watch_state.is_file() else None
        launch = None
        if revision:
            try:
                launch = launch_if_ready(args, revision)
            except Exception as exc:
                launch = {"selected": None, "error": str(exc)}
        remaining = args.target_epoch - time.time()
        interval = args.near_interval if remaining <= 3600 else args.far_interval
        body = {
            "schema": "glm53-provider-watch/1",
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "release_revision": revision,
            "topology_receipt_ready": args.topology_receipt.is_file(),
            "runpod_secure": secure,
            "runpod_community": community,
            "vast": vast,
            "funding": {
                "runpod_one_hour_ready": float(rp_funding.get("clientBalance") or 0) >= 64,
                "runpod_spend_limit_ready": float(rp_funding.get("spendLimit") or 0) >= 64,
                "vast_campaign_funded": float(vast_funding.get("balance") or 0) + float(vast_funding.get("credit") or 0) >= 420,
            },
            "launch": launch,
            "next_poll_seconds": interval,
        }
        receipt = {**body, "receipt_sha256": canonical_sha256(body)}
        atomic_json(args.state, receipt)
        print(json.dumps({
            "utc": body["checked_utc"], "release": revision, "funding": body["funding"],
            "launch": launch, "next": interval,
        }, sort_keys=True), flush=True)
        now = time.time()
        if now - last_notification >= 600:
            try:
                notify(
                    f"release={revision or 'pending'}; "
                    f"runpod8={secure['returncode'] == 0 or community['returncode'] == 0}; "
                    f"vast8={vast['returncode'] == 0}; funding={body['funding']}; "
                    f"launch={launch}",
                    "GLM-5.3 campaign status",
                    "high" if revision else "default",
                )
            except Exception as exc:
                print(f"notification failed: {exc}", flush=True)
            last_notification = now
        if args.once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
