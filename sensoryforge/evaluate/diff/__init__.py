"""Run-to-run diff: what moved between two runs of the same data set (A10).

``python -m sensoryforge.evaluate.diff BASELINE NEW`` writes ``diff.json``. See
``docs/developer_guide/diff_stage.md`` and the report contract, section 14.
"""

from sensoryforge.evaluate.diff.arrays import compare_arrays
from sensoryforge.evaluate.diff.entries import compare_entry
from sensoryforge.evaluate.diff.run import (
    DIFF_ENTRIES_FILE,
    DIFF_FILE,
    DIFF_FORMAT,
    run_diff,
)

__all__ = [
    "DIFF_ENTRIES_FILE",
    "DIFF_FILE",
    "DIFF_FORMAT",
    "compare_arrays",
    "compare_entry",
    "run_diff",
]
