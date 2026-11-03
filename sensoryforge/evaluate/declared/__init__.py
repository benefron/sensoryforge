"""The declared stage (roadmap A3, report contract section 8).

Importing this package registers the built-in adapter (``tactile``) in
``DECLARED_ADAPTER_REGISTRY``: each adapter module registers itself when imported. The
adapters import the world package, so they are imported here and never from
``sensoryforge.evaluate.registry``, which the audit imports.
"""

from sensoryforge.evaluate.declared.base import Declaration, DeclaredAdapter
from sensoryforge.evaluate.declared.run import (
    DECLARED_FILE,
    ENTRIES_FILE,
    choose_adapter,
    run_declared,
)
from sensoryforge.evaluate.registry import DECLARED_ADAPTER_REGISTRY

# The adapter registers itself when its module is imported.
import sensoryforge.evaluate.tactile.declared  # noqa: E402,F401

__all__ = [
    "DECLARED_FILE",
    "DECLARED_ADAPTER_REGISTRY",
    "ENTRIES_FILE",
    "Declaration",
    "DeclaredAdapter",
    "choose_adapter",
    "run_declared",
]
