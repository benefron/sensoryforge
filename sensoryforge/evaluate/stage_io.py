"""Small helpers the declared and compare stages share (report contract, section 4).

Both stages write ``<stage>.json`` and ``<stage>_entries.jsonl`` the way the audit does:
shortest round-trip floats, no NaN or infinity, rows already sorted by the caller, and
the rows file's sha256 recorded in the ``.json``. This module is outside the audit's
import wall.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


def timestamp(created_at: Optional[str] = None) -> str:
    """The UTC time to write as ``created_at``.

    Args:
        created_at: A timestamp to use as given (tests pin it); default now.

    Returns:
        ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return created_at or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def producer() -> Dict[str, str]:
    """The ``producer`` object: SensoryForge's version and source sha."""
    import sensoryforge
    from sensoryforge.provenance import source_info

    return {
        "sensoryforge_version": sensoryforge.__version__,
        "sensoryforge_sha": source_info()["sha"],
    }


def file_sha256(path: Path) -> str:
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> Tuple[int, str]:
    """Write rows as JSON lines.

    Args:
        path: The ``_entries.jsonl`` file.
        rows: The rows, already in entry order.

    Returns:
        ``(number of rows, sha256 of the file)``.

    Raises:
        ValueError: If a row holds NaN or infinity.
    """
    digest = hashlib.sha256()
    n = 0
    with open(path, "wb") as fh:
        for row in rows:
            data = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
            digest.update(data)
            fh.write(data)
            n += 1
    return n, digest.hexdigest()


def write_json(path: Path, content: Dict[str, Any]) -> None:
    """Write a stage's ``.json`` (indented, no NaN or infinity)."""
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def clean(value: Any) -> Any:
    """JSON-safe copy of a value: numpy scalars become numbers, NaN becomes null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
