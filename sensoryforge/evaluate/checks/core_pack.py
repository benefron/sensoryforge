"""The core check pack: checks that assume no modality."""

from __future__ import annotations

from typing import Any, Dict, List

from sensoryforge.evaluate.checks.base import Check, CheckPack
from sensoryforge.evaluate.checks.data_checks import (
    Finite,
    Readable,
    ReadoutValues,
    TimeGrid,
)
from sensoryforge.evaluate.checks.engine_checks import (
    Determinism,
    EmptyStimulus,
    InputFloor,
    MirrorSymmetry,
    RerunMatches,
    Saturation,
    SingleSensor,
    Voltage,
)
from sensoryforge.evaluate.checks.world_checks import PrefixStable


class CoreChecks(CheckPack):
    """The checks every data set gets (``--packs core``)."""

    name = "core"

    def to_dict(self) -> Dict[str, Any]:
        """The pack's parameters (none)."""
        return {}

    def checks(self) -> List[Check]:
        """The core checks: data, then probes, then re-runs, then the sampler."""
        return [
            Readable(),
            Finite(),
            ReadoutValues(),
            TimeGrid(),
            Determinism(),
            MirrorSymmetry(),
            EmptyStimulus(),
            SingleSensor(),
            Saturation(),
            RerunMatches(),
            InputFloor(),
            Voltage(),
            PrefixStable(),
        ]
