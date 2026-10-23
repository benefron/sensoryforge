"""The tactile check pack: reference-free checks that use touch vocabulary.

``tactile.rate.monotonic``: more pressure never lowers a tonic afferent's rate. Inside
the checks stage (outside the audit's import wall); it imports the engine, not the
world.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from sensoryforge.evaluate.checks import probe as pb
from sensoryforge.evaluate.checks.base import (
    Check,
    CheckContext,
    CheckPack,
    CheckResult,
    CheckSkipped,
    Failure,
)

#: Hold levels, as fractions of the data's own peak frame value.
LEVELS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)

#: The filter name of a slowly adapting (tonic) afferent's population.
TONIC_FILTER = "sa"


class MonotonicRate(Check):
    """A tonic afferent's spike count over a hold does not fall as pressure rises."""

    id = "tactile.rate.monotonic"
    pack = "tactile"
    scope = "probe"
    title = "More pressure never lowers a tonic afferent's rate"
    what = (
        "A Gaussian indentation at the grid's centre is held at "
        + ", ".join(f"{x:g}" for x in LEVELS)
        + " of the data's peak value, noise off. For every neuron of each tonic (SA) "
        "population, the spike count over the hold must not fall from one level to the "
        "next by more than the tolerance (default 0)."
    )

    def __init__(self, tolerance: float = 0.0) -> None:
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.tolerance = float(tolerance)

    def run(self, ctx: CheckContext) -> CheckResult:
        cfg = pb.quiet(pb.base_config(ctx))
        tonic = [
            p.name
            for p in cfg.populations
            if str(p.filter_method).lower() == TONIC_FILTER
        ]
        if not tonic:
            raise CheckSkipped(
                f"no population uses the {TONIC_FILTER!r} (tonic) filter"
            )
        probe = pb.build(ctx, cfg)
        peak = ctx.peak()
        t = probe.n_frames
        ii, jj = np.meshgrid(
            np.arange(probe.rows), np.arange(probe.cols), indexing="ij"
        )
        sigma = max(1.0, 0.15 * min(probe.rows, probe.cols))
        blob = np.exp(
            -((ii - (probe.rows - 1) / 2) ** 2 + (jj - (probe.cols - 1) / 2) ** 2)
            / (2 * sigma**2)
        ).astype(np.float32)
        hold = slice(t // 4, 3 * t // 4)
        counts: Dict[str, List[np.ndarray]] = {name: [] for name in tonic}
        for level in LEVELS:
            frames = np.zeros((t, probe.rows, probe.cols), dtype=np.float32)
            frames[hold] = blob * np.float32(level * peak)
            if probe.channels > 1:
                frames = np.repeat(frames[:, None], probe.channels, axis=1)
            out = pb.run(probe, frames, ctx.seed)
            for name in tonic:
                key = pb.readout_of(out[name])
                if key not in ("spikes", "events"):
                    continue
                counts[name].append(np.abs(out[name][key][hold]).sum(axis=0))
        failures: List[Failure] = []
        numbers: Dict[str, Any] = {}
        tested = 0
        fired = 0.0
        for name, per_level in counts.items():
            if len(per_level) != len(LEVELS):
                continue
            stack = np.stack(per_level)  # [levels, N]
            fired += float(stack.sum())
            drops = stack[:-1] - stack[1:]  # positive where the count fell
            worst = float(drops.max())
            tested += stack.shape[1]
            numbers[f"{name}.mean_count_by_level"] = [
                round(float(x), 6) for x in stack.mean(axis=1)
            ]
            numbers[f"{name}.largest_fall"] = worst
            bad = np.flatnonzero((drops > self.tolerance).any(axis=0))
            for n in bad:
                k = int(np.argmax(drops[:, n]))
                failures.append(
                    Failure(
                        f"{name} neuron {int(n)}",
                        f"count {stack[k, n]:g} at {LEVELS[k]:g} of the peak, "
                        f"{stack[k + 1, n]:g} at {LEVELS[k + 1]:g}",
                    )
                )
        if not tested:
            raise CheckSkipped("no tonic population has a spike readout")
        if not fired:
            raise CheckSkipped(
                "no tonic neuron fired at any level, so nothing was tested"
            )
        return CheckResult(
            "fail" if failures else "pass",
            tested=tested,
            failed=len(failures),
            failures=failures,
            numbers=numbers,
        )


class TactileChecks(CheckPack):
    """The tactile pack.

    Args:
        tolerance: Spikes a neuron's count may fall between two levels.
    """

    name = "tactile"

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def to_dict(self) -> Dict[str, Any]:
        """The pack's parameters, recorded under ``params`` in ``checks.json``."""
        return {"tolerance": self.tolerance}

    def checks(self) -> List[Check]:
        """The pack's checks."""
        return [MonotonicRate(self.tolerance)]
