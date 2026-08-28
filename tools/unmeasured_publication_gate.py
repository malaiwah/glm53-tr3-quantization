#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from publication_gate import load_sealed, verify_artifacts
from release_gate import canonical_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize an explicitly UNMEASURED first payload without making fidelity claims")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = load_sealed(
        args.artifact_manifest, "glm53-publish-artifact-manifest/1"
    )
    root = args.artifact_root.resolve()
    verify_artifacts(root, artifact)
    card = (root / "README.md").read_text()
    required = (
        "UNMEASURED FIRST RELEASE — NOT QUALIFIED",
        "qualification_status: unmeasured",
        "matches or beats any other quantization.",
    )
    if any(marker not in card for marker in required):
        raise RuntimeError("model card lacks mandatory unmeasured warnings")
    config = json.loads((root / "config.json").read_text())
    if config.get("hybrid_tr3_tail", {}).get("rotation_layout") != "shared_h_v1":
        raise RuntimeError("artifact does not declare shared_h_v1")
    source_revision = artifact.get("source_revision")
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise RuntimeError("artifact source revision is invalid")
    body = {
        "schema": "glm53-publication-authorization/1",
        "hf_repo": args.hf_repo,
        "source_revision": source_revision,
        "profile": artifact.get("profile"),
        "artifact_manifest_sha256": artifact["receipt_sha256"],
        "qualification_status": "unmeasured",
        "fidelity_claims_authorized": False,
        "public_flip_authorized": True,
        "authorized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "authorized": True,
        "qualification_status": "unmeasured",
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
