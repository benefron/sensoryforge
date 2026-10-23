"""The evaluation seams (report contract, section 10).

``METRIC_PACK_REGISTRY`` holds the packs the audit can run; ``core`` and the first
modality's pack are built in. A pack is registered by name and found from a plugin
through the ``sensoryforge.components`` entry-point group, whose callable registers it
here. ``DECLARED_ADAPTER_REGISTRY`` holds the declared adapters; it is filled by
``sensoryforge.evaluate.declared`` (the adapters import the world package, which this
module must not). The frame-renderer registry arrives with A5.

This module is inside the audit's import wall: it never imports ``sensoryforge.world``.
"""

from __future__ import annotations

from sensoryforge.registry import ComponentRegistry

METRIC_PACK_REGISTRY = ComponentRegistry("METRIC_PACK_REGISTRY")
DECLARED_ADAPTER_REGISTRY = ComponentRegistry("DECLARED_ADAPTER_REGISTRY")


def register_builtin_packs() -> None:
    """Register the packs that ship with SensoryForge (idempotent)."""
    from sensoryforge.evaluate.audit.core_pack import CorePack
    from sensoryforge.evaluate.tactile.metrics import TactilePack

    METRIC_PACK_REGISTRY.register("core", CorePack)
    METRIC_PACK_REGISTRY.register("tactile", TactilePack)


register_builtin_packs()
