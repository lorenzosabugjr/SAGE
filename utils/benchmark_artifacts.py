"""
Run artifact helpers for benchmark scripts.

Provides a timestamped run directory under a fixed results root, a raw
config snapshot, a tee'd stdout log, and git commit resolution, so that
each benchmark invocation is self-contained and reproducible.
"""

import contextlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class TeeTextIO:
    """A writable text stream that mirrors writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def create_run_dir(results_root, timestamp=None) -> Path:
    """Create results_root/<timestamp>/, appending _02, _03, ... on collision.

    Never overwrites an existing run directory.
    """
    results_root = Path(results_root)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    run_dir = results_root / timestamp
    suffix = 2
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = results_root / f"{timestamp}_{suffix:02d}"
            suffix += 1


def copy_config(config_path, run_dir) -> Path:
    """Copy the raw config file into run_dir, preserving basename and bytes."""
    config_path = Path(config_path)
    dest = Path(run_dir) / config_path.name
    shutil.copy2(config_path, dest)
    return dest


def get_git_commit() -> str:
    """Return the current git commit hash, or "unknown" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


@contextlib.contextmanager
def tee_stdout(log_file):
    """Redirect stdout so it is written to both the terminal and log_file."""
    tee = TeeTextIO(sys.stdout, log_file)
    with contextlib.redirect_stdout(tee):
        yield
