"""Run every documentation example script and assert it succeeds (H5).

Mirrors the pattern in ``tests/integration/test_examples_smoke.py`` (run each
shipped artefact as a real invocation, assert exit code 0), applied to
standalone Python examples under ``docs/examples/`` instead of YAML configs.
Each file is executed as a subprocess (not exec'd in-process) so a script
that mutates global state (e.g. registering a component) can't leak into
other tests, and so a crash produces a real traceback in captured output.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_EXAMPLES_DIR = REPO_ROOT / "docs" / "examples"
EXAMPLE_SCRIPTS = sorted(DOCS_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_docs_example_runs(script_path):
    """Every docs/examples/*.py must execute successfully as a script.

    The subprocess is given this checkout explicitly (F-053). Running a script
    directly puts the script's own directory on ``sys.path[0]``, not the
    repository root, so ``import sensoryforge`` falls through to whatever the
    environment's editable install points at -- which, with git worktrees in
    play, can be a different checkout entirely. Setting ``cwd`` and
    ``PYTHONPATH`` makes the example import the package it ships beside,
    without the example itself doing any path surgery.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(REPO_ROOT)]
    )
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"{script_path.name} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_docs_examples_dir_is_not_empty():
    """Guard against this discovering zero files and silently passing."""
    assert EXAMPLE_SCRIPTS, f"No .py files found under {DOCS_EXAMPLES_DIR}"


def test_docs_example_subprocess_imports_this_checkout():
    """The example subprocess must import the package beside it (F-053).

    Guards the fix in :func:`test_docs_example_runs`: without an explicit
    ``cwd``/``PYTHONPATH``, a script run directly resolves ``sensoryforge``
    through the environment's editable install, which may point at another
    checkout (a git worktree, a second clone). That silently tests the wrong
    code and the suite still passes, so this asserts the resolved path.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(REPO_ROOT)]
    )
    result = subprocess.run(
        [sys.executable, "-c", "import sensoryforge; print(sensoryforge.__file__)"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip()).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"docs examples would import {resolved}, not this checkout at "
        f"{REPO_ROOT}. The environment's editable install points elsewhere."
    )


def test_docs_examples_do_not_patch_sys_path():
    """No shipped example may manipulate sys.path.

    An example is a thing users copy. Path surgery in one teaches a pattern
    that is wrong outside this repository, and here it would only paper over
    F-053, which the harness fixes properly.
    """
    offenders = [p.name for p in EXAMPLE_SCRIPTS if "sys.path" in p.read_text()]
    assert not offenders, (
        f"these docs examples manipulate sys.path: {offenders}. The test "
        "harness puts the repository root on PYTHONPATH; examples must not."
    )
