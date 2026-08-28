from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_TOPOLOGY = {
    "first_k_dense_replace": 3,
    "hidden_size": 6144,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_nextn_predict_layers": 1,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def release_revision(path: Path) -> str | None:
    target = json.loads(path.read_text()).get("targets", {}).get("zai-org/GLM-5.3", {})
    revision = target.get("revision")
    if target.get("released") is True and isinstance(revision, str) and len(revision) == 40:
        return revision
    return None


def topology_gate(path: Path) -> str:
    receipt = json.loads(path.read_text())
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise RuntimeError("topology receipt seal differs")
    if receipt.get("metadata_passed") is not True or receipt.get("errors") != []:
        raise RuntimeError("topology metadata did not pass cleanly")
    observed = receipt.get("observed_topology", {})
    drift = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED_TOPOLOGY.items()
        if observed.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"release topology differs: {drift}")
    if receipt.get("candidate_mode") == "glm52-rehearsal":
        raise RuntimeError("GLM-5.2 rehearsal receipt cannot authorize GLM-5.3 rental")
    return claimed
