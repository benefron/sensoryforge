"""Run every documentation example script and assert it succeeds (H5).

Mirrors the pattern in ``tests/integration/test_examples_smoke.py`` (run each
shipped artefact as a real invocation, assert exit code 0), applied to
standalone Python examples under ``docs/examples/`` instead of YAML configs.
Each file is executed as a subprocess (not exec'd in-process) so a script
that mutates global state (e.g. registering a component) can't leak into
other tests, and so a crash produces a real traceback in captured output.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DOCS_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "examples"
EXAMPLE_SCRIPTS = sorted(DOCS_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_docs_example_runs(script_path):
    """Every docs/examples/*.py must execute successfully as a script."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script_path.name} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_docs_examples_dir_is_not_empty():
    """Guard against this discovering zero files and silently passing."""
    assert EXAMPLE_SCRIPTS, f"No .py files found under {DOCS_EXAMPLES_DIR}"
