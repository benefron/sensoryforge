"""Where this SensoryForge came from: its git sha (spec §7.4).

In a checkout (an editable install, or a worktree on ``PYTHONPATH``) the sha
is ``git rev-parse HEAD``. In a pip install from git -- pressure-simulation's
pinned ``pip install "sensoryforge @ git+file:///...@<sha>"`` -- pip records
the commit in PEP 610's ``direct_url.json``. Otherwise it is ``"unknown"``.
"""

from __future__ import annotations

import functools
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _direct_url_sha() -> Optional[str]:
    try:
        text = metadata.distribution("sensoryforge").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def read_source_info(package_dir: Path = _PACKAGE_DIR) -> Dict[str, Any]:
    """``{"sha", "dirty", "source"}`` for the SensoryForge package in ``package_dir``.

    Args:
        package_dir: The ``sensoryforge`` package directory (default: this one).

    Returns:
        ``source`` is ``"git"`` (``dirty``: tracked files modified),
        ``"direct_url"`` (``dirty`` False) or ``"unknown"`` (``sha`` "unknown",
        ``dirty`` None).
    """
    root = Path(package_dir).resolve().parent
    if (root / ".git").exists():
        sha = _git(root, "rev-parse", "HEAD")
        if sha:
            status = _git(root, "status", "--porcelain", "--untracked-files=no")
            return {"sha": sha, "dirty": bool(status), "source": "git"}
    sha = _direct_url_sha()
    if sha:
        return {"sha": sha, "dirty": False, "source": "direct_url"}
    return {"sha": "unknown", "dirty": None, "source": "unknown"}


@functools.lru_cache(maxsize=1)
def _cached() -> Dict[str, Any]:
    return read_source_info()


def source_info() -> Dict[str, Any]:
    """This process's SensoryForge sha (read once, cached); a fresh dict each call."""
    return dict(_cached())
