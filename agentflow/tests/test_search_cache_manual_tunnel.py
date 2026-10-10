"""Exercise the actual Bash tunnel entrypoint with a dummy ssh executable."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ManualTunnelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="search-tunnel-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name) / "repo with spaces"
        self.scripts = self.repo / "train-roma"
        self.scripts.mkdir(parents=True)
        for name in ("open_search_cache_tunnel.sh", "search_cache_bootstrap.py"):
            shutil.copy2(ROOT / "train-roma" / name, self.scripts / name)
        self.key = self.repo / "pem" / "dummy identity.pem"
        self.key.parent.mkdir()
        self.key.write_text("dummy fixture, not a private key")
        self.key.chmod(0o600)
        self.output = Path(self.directory.name) / "ssh-argv.json"
        binary_dir = Path(self.directory.name) / "bin"
        binary_dir.mkdir()
        ssh = binary_dir / "ssh"
        ssh.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['TEST_SSH_ARGV_OUTPUT'], 'w') as output:\n"
            "    json.dump(sys.argv[1:], output)\n"
        )
        ssh.chmod(0o700)
        self.env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ.get("PATH", ""),
                        TEST_SSH_ARGV_OUTPUT=str(self.output), HOME=self.directory.name)
        self.env.pop("SEARCH_CACHE_SSH_IDENTITY_FILE", None)

    def run_tunnel(self, identity=None):
        if identity is not None:
            self.env["SEARCH_CACHE_SSH_IDENTITY_FILE"] = identity
        return subprocess.run(["bash", str(self.scripts / "open_search_cache_tunnel.sh")],
                              cwd=self.directory.name, env=self.env, capture_output=True,
                              text=True, timeout=10, check=False)

    def test_relative_identity_with_spaces_is_repo_rooted_and_one_argv_item(self):
        result = self.run_tunnel("pem/dummy identity.pem")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.output.read_text())
        self.assertEqual(args[args.index("-i") + 1], str(self.key.resolve()))
        self.assertIn("IdentitiesOnly=yes", args)
        self.assertIn("BatchMode=yes", args)
        self.assertIn("StrictHostKeyChecking=yes", args)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)

    def test_empty_identity_preserves_default_selection(self):
        result = self.run_tunnel()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.output.read_text())
        self.assertNotIn("-i", args)
        self.assertNotIn("IdentitiesOnly=yes", args)

    def test_invalid_or_permissive_key_is_rejected_without_running_ssh(self):
        for identity in ("pem/missing.pem", "pem"):
            result = self.run_tunnel(identity)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())
        self.key.chmod(0o644)
        result = self.run_tunnel("pem/dummy identity.pem")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("chmod 600", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
