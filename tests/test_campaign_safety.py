from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import capture_controller
import mixed_materialize
import release_gate
import work_queue


class CampaignSafetyTests(unittest.TestCase):
    def test_topology_gate_rejects_rehearsal_and_tamper(self):
        body = {
            "metadata_passed": True,
            "errors": [],
            "candidate_mode": "glm53-release",
            "observed_topology": dict(release_gate.EXPECTED_TOPOLOGY),
        }
        receipt = {**body, "receipt_sha256": release_gate.canonical_sha256(body)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(release_gate.topology_gate(path), receipt["receipt_sha256"])
            receipt["candidate_mode"] = "glm52-rehearsal"
            rehearsal_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            receipt["receipt_sha256"] = release_gate.canonical_sha256(rehearsal_body)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, "rehearsal"):
                release_gate.topology_gate(path)
            receipt["candidate_mode"] = "glm53-release"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, "seal"):
                release_gate.topology_gate(path)

    def test_shared_h_mixed_key_census_is_physical_contract(self):
        shared, local = mixed_materialize.expected_keys(3)
        self.assertEqual(len(shared), 12)
        self.assertEqual(len(local), 9_216)
        self.assertEqual(len(shared | local), 9_228)
        expert_zero = {key for key in local if ".experts.0." in key}
        self.assertEqual(len(expert_zero), 36)

    def test_capture_window_rejects_same_size_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "capture" / "x.bin"
            payload.parent.mkdir()
            payload.write_bytes(b"sealed-capture")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            body = {
                "schema": "glm53-capture-window-export/1",
                "window": "3-10",
                "files": {"capture/x.bin": {"bytes": payload.stat().st_size, "sha256": digest}},
                "created_utc": "2026-08-28T00:00:00Z",
            }
            receipt = {**body, "receipt_sha256": release_gate.canonical_sha256(body)}
            (root / "READY.json").write_text(json.dumps(receipt))
            capture_controller.verify_window(root)
            payload.write_bytes(b"x" * payload.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                capture_controller.verify_window(root)

    def test_work_queue_rejects_duplicate_or_lost_unit(self):
        body = {
            "schema": work_queue.SCHEMA,
            "contract_sha256": "a" * 64,
            "units": {"k3-layer-003": {"profile": "k3", "layer": 3}},
            "pending": ["k3-layer-003"],
            "active": {},
            "completed": {},
        }
        state = work_queue.seal(body)
        work_queue.verify(state)
        state["active"] = {"worker": {"unit": "k3-layer-003"}}
        state = work_queue.seal(state)
        with self.assertRaisesRegex(RuntimeError, "multiple states"):
            work_queue.verify(state)

    def test_sign_template_covers_main_and_mtp_layers(self):
        template = json.loads(
            (ROOT / "baselines" / "willfalco-3.42" / "shared_h_sign_template.json").read_text()
        )
        claimed = template.pop("receipt_sha256")
        self.assertEqual(release_gate.canonical_sha256(template), claimed)
        self.assertEqual(sorted(map(int, template["layers"])), list(range(3, 79)))
        for row in template["layers"].values():
            self.assertEqual(len(row["down"]), 4)


if __name__ == "__main__":
    unittest.main()
