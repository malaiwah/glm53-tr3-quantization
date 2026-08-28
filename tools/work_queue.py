#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "glm53-uniform-parts-work-queue/1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(state: dict) -> dict:
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    return {**body, "state_sha256": canonical_sha256(body)}


def verify(state: dict) -> None:
    claimed = state.get("state_sha256")
    if state.get("schema") != SCHEMA or not isinstance(claimed, str):
        raise RuntimeError("invalid work-queue schema/seal")
    if seal(state)["state_sha256"] != claimed:
        raise RuntimeError("work-queue state seal differs")
    units = state.get("units")
    if not isinstance(units, dict) or not units:
        raise RuntimeError("work queue has no units")
    pending = state.get("pending")
    active = state.get("active")
    completed = state.get("completed")
    if not isinstance(pending, list) or not isinstance(active, dict) or not isinstance(completed, dict):
        raise RuntimeError("work-queue collections invalid")
    locations = pending + [row["unit"] for row in active.values()] + list(completed)
    if sorted(locations) != sorted(units) or len(locations) != len(set(locations)):
        raise RuntimeError("work unit exists in zero or multiple states")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@contextmanager
def locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load(path: Path) -> dict:
    state = json.loads(path.read_text())
    verify(state)
    return state


def parse_layers(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def cmd_init(args) -> int:
    units = {
        f"{profile}-layer-{layer:03d}": {"profile": profile, "layer": layer}
        for profile in args.profiles
        for layer in parse_layers(args.layers)
    }
    state = seal({
        "schema": SCHEMA,
        "contract_sha256": args.contract_sha256,
        "units": units,
        "pending": sorted(units),
        "active": {},
        "completed": {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with locked(args.state):
        if args.state.exists():
            current = load(args.state)
            if current["contract_sha256"] != args.contract_sha256 or current["units"] != units:
                raise RuntimeError("existing queue contract or units differ")
            state = current
        else:
            atomic_json(args.state, state)
    print(json.dumps({"units": len(units), "state_sha256": state["state_sha256"]}))
    return 0


def cmd_claim(args) -> int:
    with locked(args.state):
        state = load(args.state)
        existing = state["active"].get(args.worker)
        if existing:
            result = existing
        else:
            candidates = [
                unit for unit in state["pending"]
                if args.profile is None or state["units"][unit]["profile"] == args.profile
            ]
            if not candidates:
                print(json.dumps({"worker": args.worker, "unit": None}))
                return 0
            unit = candidates[0]
            state["pending"].remove(unit)
            result = {
                "unit": unit,
                "nonce": uuid.uuid4().hex,
                "claimed_unix": time.time(),
                "claimed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            state["active"][args.worker] = result
            atomic_json(args.state, seal(state))
    print(json.dumps({"worker": args.worker, **result}, sort_keys=True))
    return 0


def cmd_complete(args) -> int:
    receipt = args.receipt.resolve()
    if not receipt.is_file() or receipt.is_symlink():
        raise RuntimeError(f"completion receipt missing or symlinked: {receipt}")
    with locked(args.state):
        state = load(args.state)
        claim = state["active"].get(args.worker)
        if not claim or claim["unit"] != args.unit or claim["nonce"] != args.nonce:
            raise RuntimeError("completion does not match active claim")
        state["active"].pop(args.worker)
        state["completed"][args.unit] = {
            "worker": args.worker,
            "receipt": str(receipt),
            "receipt_sha256": sha256_file(receipt),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(args.state, seal(state))
    print(json.dumps({"unit": args.unit, "complete": True}, sort_keys=True))
    return 0


def cmd_requeue(args) -> int:
    cutoff = time.time() - args.max_age
    requeued = []
    with locked(args.state):
        state = load(args.state)
        for worker, claim in list(state["active"].items()):
            if float(claim["claimed_unix"]) > cutoff:
                continue
            state["active"].pop(worker)
            state["pending"].append(claim["unit"])
            requeued.append({"worker": worker, "unit": claim["unit"]})
        state["pending"].sort()
        if requeued:
            atomic_json(args.state, seal(state))
    print(json.dumps({"requeued": requeued}, sort_keys=True))
    return 0


def cmd_status(args) -> int:
    state = load(args.state)
    print(json.dumps({
        "pending": len(state["pending"]),
        "active": state["active"],
        "completed": len(state["completed"]),
        "total": len(state["units"]),
        "state_sha256": state["state_sha256"],
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--contract-sha256", required=True)
    init.add_argument("--profiles", nargs="+", default=["k3", "k4"])
    init.add_argument("--layers", default="3-77")
    init.set_defaults(func=cmd_init)
    claim = sub.add_parser("claim")
    claim.add_argument("--state", type=Path, required=True)
    claim.add_argument("--worker", required=True)
    claim.add_argument("--profile")
    claim.set_defaults(func=cmd_claim)
    complete = sub.add_parser("complete")
    complete.add_argument("--state", type=Path, required=True)
    complete.add_argument("--worker", required=True)
    complete.add_argument("--unit", required=True)
    complete.add_argument("--nonce", required=True)
    complete.add_argument("--receipt", type=Path, required=True)
    complete.set_defaults(func=cmd_complete)
    requeue = sub.add_parser("requeue")
    requeue.add_argument("--state", type=Path, required=True)
    requeue.add_argument("--max-age", type=float, default=1800)
    requeue.set_defaults(func=cmd_requeue)
    status = sub.add_parser("status")
    status.add_argument("--state", type=Path, required=True)
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    if hasattr(args, "contract_sha256") and len(args.contract_sha256) != 64:
        raise SystemExit("contract SHA-256 must have 64 hex characters")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
