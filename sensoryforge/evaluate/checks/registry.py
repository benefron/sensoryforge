"""The registry of check packs (report contract, section 13).

``CHECK_PACK_REGISTRY`` holds the packs ``run_checks`` can run: ``core`` and the first
modality's pack are built in; a plugin registers its own through the
``sensoryforge.components`` entry-point group. It is separate from
``METRIC_PACK_REGISTRY`` because the checks stage may import the engine and the world,
which the audit's import wall forbids.
"""

from __future__ import annotations

from sensoryforge.registry import ComponentRegistry

CHECK_PACK_REGISTRY = ComponentRegistry("CHECK_PACK_REGISTRY")


def register_builtin_check_packs() -> None:
    """Register the packs that ship with SensoryForge (idempotent)."""
    from sensoryforge.evaluate.checks.core_pack import CoreChecks
    from sensoryforge.evaluate.tactile.checks import TactileChecks

    CHECK_PACK_REGISTRY.register("core", CoreChecks)
    CHECK_PACK_REGISTRY.register("tactile", TactileChecks)


register_builtin_check_packs()
