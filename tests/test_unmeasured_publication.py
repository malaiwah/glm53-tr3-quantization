from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
REVISION = "a" * 40


def run_tool(name: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOLS / name), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def rewrite_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (root / "MANIFEST.sha256").write_text("".join(rows))


class UnmeasuredPublicationTests(unittest.TestCase):
    def make_artifact(self, temporary: str) -> tuple[Path, Path, Path, Path]:
        base = Path(temporary)
        source = base / "source"
        artifact = base / "artifact"
        source.mkdir()
        artifact.mkdir()
        (source / "LICENSE").write_text("GLM-5.3 License\n")
        (artifact / "config.json").write_text(json.dumps({
            "hybrid_tr3_tail": {
                "rotation_layout": "shared_h_v1",
                "source_format": "BF16",
            }
        }))
        (artifact / "calibration_manifest.json").write_text(json.dumps({
            "total_tokens": 131_072,
            "corpus_sha256": "b" * 64,
        }))
        (artifact / "model.safetensors").write_bytes(b"physical payload")
        artifact_manifest = base / "artifact-manifest.json"
        authorization = base / "authorization.json"
        return source, artifact, artifact_manifest, authorization

    def write_card_and_manifest(self, source: Path, artifact: Path, manifest: Path) -> None:
        run_tool(
            "write_unmeasured_card.py",
            "--artifact", str(artifact),
            "--source", str(source),
            "--source-revision", REVISION,
            "--profile", "flat-k3",
            "--repo", "malaiwah/GLM-5.3-TR3-3bpw",
        )
        run_tool(
            "build_artifact_manifest.py",
            "--root", str(artifact),
            "--source-revision", REVISION,
            "--profile", "flat-k3",
            "--out", str(manifest),
        )

    def test_exact_unmeasured_artifact_is_authorized_without_fidelity_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, artifact, manifest, authorization = self.make_artifact(temporary)
            self.write_card_and_manifest(source, artifact, manifest)
            run_tool(
                "unmeasured_publication_gate.py",
                "--artifact-root", str(artifact),
                "--artifact-manifest", str(manifest),
                "--hf-repo", "malaiwah/GLM-5.3-TR3-3bpw",
                "--out", str(authorization),
            )
            receipt = json.loads(authorization.read_text())
            self.assertEqual(receipt["qualification_status"], "unmeasured")
            self.assertTrue(receipt["public_flip_authorized"])
            self.assertFalse(receipt["fidelity_claims_authorized"])

    def test_missing_warning_is_rejected_after_exact_manifest_reseal(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, artifact, manifest, authorization = self.make_artifact(temporary)
            self.write_card_and_manifest(source, artifact, manifest)
            card = artifact / "README.md"
            card.write_text(card.read_text().replace(
                "UNMEASURED FIRST RELEASE — NOT QUALIFIED",
                "FIRST RELEASE",
            ))
            rewrite_manifest(artifact)
            run_tool(
                "build_artifact_manifest.py",
                "--root", str(artifact),
                "--source-revision", REVISION,
                "--profile", "flat-k3",
                "--out", str(manifest),
            )
            result = run_tool(
                "unmeasured_publication_gate.py",
                "--artifact-root", str(artifact),
                "--artifact-manifest", str(manifest),
                "--hf-repo", "malaiwah/GLM-5.3-TR3-3bpw",
                "--out", str(authorization),
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mandatory unmeasured warnings", result.stderr)
            self.assertFalse(authorization.exists())


if __name__ == "__main__":
    unittest.main()
