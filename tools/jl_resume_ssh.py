#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path



def update_alias(config: Path, alias: str, hostname: str, user: str) -> None:
    config.parent.mkdir(parents=True, exist_ok=True)
    lines = config.read_text().splitlines() if config.is_file() else []
    block_lines = [
        f"Host {alias}",
        f"    HostName {hostname}",
        f"    User {user}",
        "    StrictHostKeyChecking no",
        "    UserKnownHostsFile /dev/null",
        "    ControlMaster auto",
        "    ControlPath ~/.ssh/cm-%C",
        "    ControlPersist 15m",
        "    ServerAliveInterval 30",
        "    ServerAliveCountMax 4",
    ]
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == f"Host {alias}"),
        None,
    )
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block_lines)
    else:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("Host ")
            ),
            len(lines),
        )
        lines[start:end] = block_lines + [""]
    text = "\n".join(lines).rstrip() + "\n"
    tmp = config.with_suffix(".tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o600)
    tmp.replace(config)
    os.chmod(config.parent, 0o700)


def ssh_ready(alias: str, timeout: float) -> None:
    subprocess.run(
        ["ssh", "-O", "exit", alias],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        master = subprocess.run(
            ["ssh", "-MNf", alias],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if master.returncode == 0:
            result = subprocess.run(
                ["ssh", alias, "echo", "SSH_OK"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            last = result.stdout.strip()
            if result.returncode == 0 and last == "SSH_OK":
                return
        else:
            last = master.stdout.strip()
        time.sleep(2)
    raise RuntimeError(f"SSH did not become ready for {alias}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume Jarvis, refresh SSH alias, start ControlMaster, require SSH_OK"
    )
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--fs-id", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vcpus", type=int)
    parser.add_argument("--ram", type=int)
    parser.add_argument("--gpu")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--storage", type=int)
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh/config")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not os.environ.get("JL_API_KEY"):
        raise SystemExit("JL_API_KEY is required")
    from jarvislabs import Client
    from jarvislabs.regions import region_base_url

    client = Client()
    instance = client.instances.get(args.machine_id)
    if instance.status == "Paused":
        if args.cpu:
            if args.vcpus is None or args.ram is None:
                raise SystemExit("CPU resume requires --vcpus and --ram")
            payload = {
                "machine_id": args.machine_id,
                "num_cpus": 1,
                "hdd": args.storage or instance.storage_gb,
                "name": instance.name,
                "vcpus": args.vcpus,
                "ram_gb": args.ram,
            }
            if args.fs_id is not None:
                payload["fs_id"] = args.fs_id
            if instance.vpc_id:
                payload["vpc_id"] = instance.vpc_id
            response = client._transport.request(
                "POST", "templates/vm/cpu/resume", json=payload,
                base_url=region_base_url(instance.region),
            )
            machine_id = int(response["machine_id"])
            deadline = time.time() + args.timeout
            while True:
                try:
                    instance = client.instances.get(machine_id)
                    if instance.status == "Running":
                        break
                except Exception:
                    pass
                if time.time() >= deadline:
                    raise RuntimeError(f"CPU VM {machine_id} did not reach Running")
                time.sleep(2)
        else:
            instance = client.instances.resume(
                args.machine_id,
                gpu_type=args.gpu,
                num_gpus=args.num_gpus,
                storage=args.storage,
                fs_id=args.fs_id,
                is_spot=args.spot,
            )
            machine_id = instance.machine_id
    elif instance.status == "Running":
        machine_id = instance.machine_id
    else:
        raise RuntimeError(f"instance {args.machine_id} has non-resumable state {instance.status}")
    if not instance.public_ip:
        instance = client.instances.get(machine_id)
    if not instance.public_ip:
        raise RuntimeError(f"instance {machine_id} has no public IP")
    update_alias(args.ssh_config, args.alias, instance.public_ip, args.user)
    ssh_ready(args.alias, args.timeout)
    result = {
        "machine_id": machine_id,
        "alias": args.alias,
        "public_ip": instance.public_ip,
        "status": instance.status,
        "fs_id": instance.fs_id,
        "ssh": "SSH_OK",
    }
    print(json.dumps(result, sort_keys=True))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
