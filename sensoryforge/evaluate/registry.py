"""The evaluation seams (report contract, section 10).

``METRIC_PACK_REGISTRY`` holds the packs the audit can run; ``core`` and the first
modality's pack are built in. A pack is registered by name and found from a plugin
through the ``sensoryforge.components`` entry-point group, whose callable registers it
here. The frame-renderer and declared-adapter registries arrive with the tasks that need
them (A5 and A3).

This module is inside the audit's import wall: it never imports ``sensoryforge.world``.
"""

from __future__ import annotations

from sensoryforge.registry import ComponentRegistry

METRIC_PACK_REGISTRY = ComponentRegistry("METRIC_PACK_REGISTRY")


def register_builtin_packs() -> None:
    """Register the packs that ship with SensoryForge (idempotent)."""
    from sensoryforge.evaluate.audit.core_pack import CorePack
    from sensoryforge.evaluate.tactile.metrics import TactilePack

    METRIC_PACK_REGISTRY.register("core", CorePack)
    METRIC_PACK_REGISTRY.register("tactile", TactilePack)


register_builtin_packs()
