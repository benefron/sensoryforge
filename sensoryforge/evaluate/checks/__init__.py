"""Reference-free checks: tests of the simulator that need no declaration (A8).

``python -m sensoryforge.evaluate.checks DATASET`` writes ``checks.json``. See
``docs/developer_guide/checks_stage.md`` and the report contract, section 13.
"""

from sensoryforge.evaluate.checks.base import (
    Check,
    CheckContext,
    CheckPack,
    CheckResult,
    CheckSkipped,
    Failure,
)
from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
from sensoryforge.evaluate.checks.run import CHECKS_FILE, CHECKS_FORMAT, run_checks

__all__ = [
    "CHECKS_FILE",
    "CHECKS_FORMAT",
    "CHECK_PACK_REGISTRY",
    "Check",
    "CheckContext",
    "CheckPack",
    "CheckResult",
    "CheckSkipped",
    "Failure",
    "run_checks",
]
