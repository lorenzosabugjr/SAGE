"""
Unit tests for utils/benchmark_artifacts.py.

Run with: python -m unittest tests.test_benchmark_artifacts
"""

import io
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.benchmark_artifacts import (
    TeeTextIO,
    copy_config,
    create_run_dir,
    get_git_commit,
)


class CreateRunDirTests(unittest.TestCase):
    def test_creates_timestamp_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = create_run_dir(root, timestamp="2026-07-04 23-18-42")
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(run_dir, root / "2026-07-04 23-18-42")

    def test_collision_appends_suffix(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ts = "2026-07-04 23-18-42"
            first = create_run_dir(root, timestamp=ts)
            second = create_run_dir(root, timestamp=ts)
            third = create_run_dir(root, timestamp=ts)

            self.assertEqual(first, root / ts)
            self.assertEqual(second, root / f"{ts}_02")
            self.assertEqual(third, root / f"{ts}_03")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertTrue(third.is_dir())


class CopyConfigTests(unittest.TestCase):
    def test_copy_preserves_content_and_basename(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "my_config.yaml"
            content = b"list_dims: [2]\nfoo: bar\n"
            src.write_bytes(content)

            run_dir = root / "run"
            run_dir.mkdir()

            dest = copy_config(src, run_dir)

            self.assertEqual(dest.name, "my_config.yaml")
            self.assertEqual(dest.parent, run_dir)
            self.assertEqual(dest.read_bytes(), content)


class TeeTextIOTests(unittest.TestCase):
    def test_write_mirrors_to_all_streams(self):
        stream_a = io.StringIO()
        stream_b = io.StringIO()
        tee = TeeTextIO(stream_a, stream_b)

        tee.write("hello\n")
        tee.write("world\n")
        tee.flush()

        self.assertEqual(stream_a.getvalue(), "hello\nworld\n")
        self.assertEqual(stream_b.getvalue(), "hello\nworld\n")
        self.assertEqual(stream_a.getvalue(), stream_b.getvalue())


class GetGitCommitTests(unittest.TestCase):
    def test_returns_nonempty_string(self):
        commit = get_git_commit()
        self.assertIsInstance(commit, str)
        self.assertTrue(len(commit) > 0)


if __name__ == "__main__":
    unittest.main()
