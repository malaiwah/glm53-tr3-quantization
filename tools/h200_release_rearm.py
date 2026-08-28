#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

from jarvislabs import Client


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def notify(message: str, title: str, priority: str = "default") -> None:
    url = os.environ.get("NTFY_URL")
    if not url:
        return
    headers = {"Title": title, "Priority": priority}
    token = os.environ.get("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=message.encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15):
            pass
    except Exception as exc:
        print(f"notification failed: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume the Jarvis H200 campaign near release, supervise capture, then pause it"
    )
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--fs-id", type=int, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--resume-epoch", type=float, required=True)
    parser.add_argument("--jarvis-key-file", type=Path, required=True)
    parser.add_argument("--jarvis-python", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--notify-env-file", type=Path, required=True)
    parser.add_argument("--bridge-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=14_400)
    parser.add_argument("--bridge-wait-seconds", type=float, default=7_200)
    args = parser.parse_args()

    key = args.jarvis_key_file.read_text().strip()
    if not key:
        raise SystemExit("Jarvis API key is empty")
    os.environ["JL_API_KEY"] = key
    client = Client()
    try:
        instance = client.instances.get(args.machine_id)
        if instance.status == "Running":
            client.instances.pause(args.machine_id)
            print(
                json.dumps({"event": "paused-until-release", "machine_id": args.machine_id}),
                flush=True,
            )
        elif instance.status != "Paused":
            raise RuntimeError(
                f"Jarvis machine {args.machine_id} cannot be scheduled from state {instance.status}"
            )
    finally:
        client.close()

    delay = max(0.0, args.resume_epoch - time.time())
    print(json.dumps({"event": "scheduled", "resume_in_seconds": round(delay, 1)}), flush=True)
    while delay > 0:
        time.sleep(min(30.0, delay))
        delay = max(0.0, args.resume_epoch - time.time())

    resume = subprocess.run(
        [
            str(args.jarvis_python),
            str(args.project / "tools/jl_resume_ssh.py"),
            "--machine-id", str(args.machine_id),
            "--alias", args.alias,
            "--user", "root",
            "--fs-id", str(args.fs_id),
            "--gpu", "H200",
            "--num-gpus", "8",
            "--spot",
            "--timeout", "600",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    resume_lines = [line for line in resume.stdout.splitlines() if line.strip().startswith("{")]
    if not resume_lines:
        raise RuntimeError(f"resume helper returned no receipt: {resume.stdout}")
    for source, destination in (
        (args.hf_token_file, "/root/.hf_token"),
        (args.notify_env_file, "/root/.notify_env"),
    ):
        subprocess.run(
            ["scp", str(source), f"{args.alias}:{destination}"],
            check=True,
        )
    subprocess.run(
        ["ssh", args.alias, "chmod", "600", "/root/.hf_token", "/root/.notify_env"],
        check=True,
    )
    resumed = json.loads(resume_lines[-1])
    machine_id = int(resumed["machine_id"])
    notify(
        f"Jarvis H200 machine {machine_id} resumed; release watcher armed.",
        "GLM-5.3 H200 capture resumed",
        "high",
    )
    print(
        json.dumps({"event": "release-watcher-starting", "machine_id": machine_id}),
        flush=True,
    )


    budget_log = args.receipt.with_suffix(".budget.log")
    budget_log.parent.mkdir(parents=True, exist_ok=True)
    with budget_log.open("a") as budget_output:
        budget = subprocess.Popen(
            [
                str(args.jarvis_python),
                str(args.project / "tools/jarvis_budget_watch.py"),
                "--machine-id", str(machine_id),
                "--max-seconds", str(args.max_seconds),
                "--receipt", str(args.receipt.with_suffix(".budget.json")),
            ],
            stdout=budget_output,
            stderr=subprocess.STDOUT,
        )

        remote_command = (
            "source /root/.notify_env; "
            "exec /usr/bin/python3 "
            "/home/jl_fs/release/glm53-main-release-smoke/tools/release_watcher.py "
            "--repo zai-org/GLM-5.3 --token-file /root/.hf_token "
            "--state-dir /home/jl_fs/receipts/release-watch-h200 "
            "--on-release /home/jl_fs/release/glm53-main-release-smoke/tools/h200_on_release.py "
            "--hf /home/jl_fs/venvs/h200-vllm/bin/hf --root /home/jl_fs "
            "--near-interval 60 --far-interval 60 --heartbeat 600"
        )
        watcher = subprocess.run(
            ["ssh", args.alias, "/bin/bash", "-lc", shlex.quote(remote_command)],
            check=False,
        )
        bridge_ready = False
        if watcher.returncode == 0:
            deadline = time.time() + args.bridge_wait_seconds
            while time.time() < deadline:
                if args.bridge_receipt.is_file():
                    bridge_ready = True
                    break
                if budget.poll() is not None:
                    break
                time.sleep(30)

        client = Client()
        try:
            instance = client.instances.get(machine_id)
            if instance.status == "Running":
                client.instances.pause(machine_id)
        finally:
            client.close()
        try:
            budget.wait(timeout=120)
        except subprocess.TimeoutExpired:
            budget.terminate()
            budget.wait(timeout=30)

    body = {
        "schema": "glm53-h200-release-rearm/1",
        "machine_id": machine_id,
        "resume_epoch": args.resume_epoch,
        "watcher_returncode": watcher.returncode,
        "bridge_ready": bridge_ready,
        "max_seconds": args.max_seconds,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.receipt, body)
    if watcher.returncode == 0 and bridge_ready:
        notify("H200 capture exported and bridged; campaign machine paused.", "GLM-5.3 H200 capture COMPLETE", "urgent")
        return 0
    notify(json.dumps(body, sort_keys=True), "GLM-5.3 H200 capture stopped", "urgent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
