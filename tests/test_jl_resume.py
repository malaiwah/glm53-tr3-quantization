from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jl_resume_under_test", ROOT / "tools/jl_resume_ssh.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SSHAliasTests(unittest.TestCase):
    def test_updating_one_alias_preserves_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            config.write_text(
                "Host worker-a\n    HostName 10.0.0.1\n\n"
                "Host worker-b\n    HostName 10.0.0.2\n\n"
                "Host runner\n    HostName 10.0.0.3\n"
            )
            MODULE.update_alias(config, "runner", "10.0.0.9", "ubuntu")
            text = config.read_text()
            self.assertIn("Host worker-a\n", text)
            self.assertIn("Host worker-b\n", text)
            self.assertIn("Host runner\n    HostName 10.0.0.9\n", text)
            self.assertNotIn("HostName 10.0.0.3", text)
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
