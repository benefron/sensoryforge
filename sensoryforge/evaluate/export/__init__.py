"""Static export of an evaluation data set for sharing (roadmap A12).

``export_snapshot`` records what the local server would answer for the dashboard, the
gallery, the checks and a few chosen draws, and writes it as a folder of HTML pages that
open from disk with no server. The pages are the server's own; a small shim answers
their requests from the recording. See ``docs/user_guide/evaluation_export.md``.

The names are loaded on first use so that importing a submodule (the CLI's argument
list) does not pull in the server.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ExportOptions", "ExportResult", "export_snapshot", "request_key"]


def __getattr__(name: str) -> Any:
    if name in ("ExportOptions", "ExportResult", "export_snapshot"):
        from sensoryforge.evaluate.export import build

        return getattr(build, name)
    if name == "request_key":
        from sensoryforge.evaluate.export import snapshot

        return snapshot.request_key
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
