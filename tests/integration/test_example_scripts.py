"""Run every script in ``examples/scripts/`` headless (D4).

The notebook runs under nbmake in CI and the two recipe scripts have their own
tests; these two had none, and ``example_pipeline.py`` had stopped working
without anyone noticing. Each runs in a scratch folder with the Agg backend,
so ``plt.show()`` opens nothing and any file it writes lands there.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "examples" / "scripts").glob("*.py"))


def test_the_scripts_are_found():
    assert [p.name for p in SCRIPTS] == [
        "example_pipeline.py",
        "generalized_pipeline_demo.py",
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_an_example_script_runs(script, tmp_path):
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(
            filter(None, [str(REPO), os.environ.get("PYTHONPATH")])
        ),
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
