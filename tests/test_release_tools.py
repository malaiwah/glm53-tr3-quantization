from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch = load("watch_matrix")
codec = load("run_codec_smoke")


class ReleaseToolTests(unittest.TestCase):
    def test_release_requires_config_index_weights_and_full_revision(self):
        payload = {
            "sha": "a" * 40,
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "model.safetensors.index.json"},
                {"rfilename": "model-00001.safetensors"},
            ],
        }
        self.assertTrue(watch.is_released(payload))
        payload["siblings"].pop()
        self.assertFalse(watch.is_released(payload))
        payload["siblings"].append({"rfilename": "model-00001.safetensors"})
        payload["sha"] = "main"
        self.assertFalse(watch.is_released(payload))

    def test_blackwell_architecture_mapping(self):
        self.assertEqual(codec.arch_for((12, 0)), "12.0a")
        self.assertEqual(codec.arch_for((10, 3)), "10.0")
        self.assertEqual(codec.arch_for((9, 0)), "9.0")

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            watch.atomic_json(path, {"value": 1})
            watch.atomic_json(path, {"value": 2})
            self.assertEqual(path.read_text(), '{\n  "value": 2\n}\n')
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
