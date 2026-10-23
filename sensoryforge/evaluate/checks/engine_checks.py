"""Checks that run the simulator: probes through the engine and entry re-runs.

*Probes* (``determinism``, ``mirror_symmetry``, ``empty_stimulus``, ``single_sensor``,
``saturation``) build the engine from the data set's own bundle config and feed it a
synthetic stimulus scaled by the data's largest frame value. *Re-runs*
(``rerun_matches``, ``input_floor``, ``voltage``) take a few entries spread over the
data set and run each population's neuron stage again from its stored ``filtered``
with the code a run uses (the same function the draw browser's voltage trace uses).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from sensoryforge.evaluate.audit.reader import ReadError
from sensoryforge.evaluate.checks import probe as pb
from sensoryforge.evaluate.checks.base import (
    Check,
    CheckContext,
    CheckResult,
    CheckSkipped,
    Failure,
)

#: How much larger than the data's own peak the saturation probe's stimulus is.
SATURATION_FACTOR = 1000.0


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not a.size:
        return float("nan")
    with np.errstate(invalid="ignore"):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def _differences(
    a: Dict[str, Dict[str, np.ndarray]], b: Dict[str, Dict[str, np.ndarray]]
) -> List[str]:
    """``population.array`` names whose values differ between two runs."""
    out = []
    for name in a:
        for key in a[name]:
            if key not in b.get(name, {}) or not _same(a[name][key], b[name][key]):
                out.append(f"{name}.{key}")
    return out


# --------------------------------------------------------------------- probes


class Determinism(Check):
    """The same configuration and seed give the same arrays, bit for bit."""

    id = "core.run.determinism"
    pack = "core"
    scope = "probe"
    title = "Same seed, same arrays"
    what = (
        "The engine runs the same probe stimulus twice with one seed, on one engine "
        "and on a freshly built one. Every readout, drive and filtered array must be "
        "identical. A fresh build differs legitimately only where a population's "
        "wiring is unseeded; that is reported as a note, not a failure."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        probe = pb.build(ctx, pb.base_config(ctx))
        amp = pb.amplitude(ctx, probe)
        frames = pb.pattern(probe, amp)
        first = pb.run(probe, frames, ctx.seed)
        again = pb.run(probe, frames, ctx.seed)
        failures = [
            Failure("same engine, second run", f"{name} differs")
            for name in _differences(first, again)
        ]
        note = None
        fresh = pb.run(pb.build(ctx, probe.config), frames, ctx.seed)
        diff = _differences(first, fresh)
        if diff:
            unseeded = [p.name for p in probe.config.populations if p.seed is None]
            if unseeded:
                note = (
                    f"a freshly built engine differs ({', '.join(diff[:6])}); "
                    "populations without a seed draw their wiring afresh at each "
                    f"build: {', '.join(unseeded)}"
                )
            else:
                failures += [
                    Failure("fresh engine", f"{name} differs") for name in diff
                ]
        return CheckResult(
            "fail" if failures else "pass",
            tested=2,
            failed=len(failures),
            failures=failures,
            numbers={
                "amplitude": amp,
                "arrays_compared": sum(len(v) for v in first.values()),
                "total_events": sum(pb.events(v) for v in first.values()),
            },
            note=note,
        )


class MirrorSymmetry(Check):
    """A mirrored stimulus on a symmetric grid gives the mirrored response."""

    id = "core.run.mirror_symmetry"
    pack = "core"
    scope = "probe"
    title = "Mirrored stimulus, mirrored response"
    what = (
        "Each population is replaced by one neuron per sensor on the same grid (same "
        "neuron, filter and gain, no noise). An asymmetric stimulus and its mirror "
        "image "
        "along each grid axis must give mirrored readouts, drive and filtered arrays, "
        "bit for bit."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        cfg = pb.quiet(pb.symmetric(pb.base_config(ctx)))
        probe = pb.build(ctx, cfg)
        amp = pb.amplitude(ctx, probe)
        frames = pb.pattern(probe, amp)
        base = pb.run(probe, frames, ctx.seed)
        failures: List[Failure] = []
        numbers: Dict[str, Any] = {}
        events = sum(pb.events(v) for v in base.values())
        if not events and not any(pb.readout_of(v) == "state" for v in base.values()):
            raise CheckSkipped(
                "the probe stimulus evoked no spikes, so there is nothing to mirror"
            )
        tested = 0
        # frames are [T, (C,) rows, cols]; a folded population array is
        # [T, k, rows, cols]
        for label, f_axis, axis in (
            ("mirror_x (rows)", frames.ndim - 2, 2),
            ("mirror_y (columns)", frames.ndim - 1, 3),
        ):
            mirrored = pb.run(probe, np.flip(frames, f_axis).copy(), ctx.seed)
            worst = 0.0
            for name, arrays in base.items():
                for key, value in arrays.items():
                    a, b = pb.fold(value, probe), pb.fold(mirrored[name][key], probe)
                    tested += 1
                    if a is None or b is None:
                        failures.append(
                            Failure(
                                label,
                                f"{name}.{key}: {value.shape[1]} neurons do "
                                f"not fold onto the {probe.rows} x {probe.cols} grid",
                            )
                        )
                        continue
                    flipped = np.flip(a, axis)
                    worst = max(
                        worst, 0.0 if _same(flipped, b) else _max_abs(flipped, b)
                    )
                    if not _same(flipped, b):
                        failures.append(
                            Failure(
                                label,
                                f"{name}.{key}: largest difference "
                                f"{_max_abs(flipped, b):.6g}",
                            )
                        )
            numbers[f"largest_difference_{'x' if axis == 2 else 'y'}"] = worst
        numbers["total_events"] = events
        numbers["amplitude"] = amp
        return CheckResult(
            "fail" if failures else "pass",
            tested=tested,
            failed=len(failures),
            failures=failures,
            numbers=numbers,
        )


class EmptyStimulus(Check):
    """No stimulus gives finite arrays and, with noise off, no spikes."""

    id = "core.run.empty_stimulus"
    pack = "core"
    scope = "probe"
    title = "Empty stimulus"
    what = (
        "An all-zero stimulus, noise off, must give finite arrays and no spikes or "
        "events. A model that fires with no input fails this on purpose: it is worth "
        "knowing."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        probe = pb.build(ctx, pb.quiet(pb.base_config(ctx)))
        frames = pb.pattern(probe, 0.0)
        out = pb.run(probe, frames, ctx.seed)
        failures = []
        for name, arrays in out.items():
            bad = [k for k, v in arrays.items() if v.size and not np.isfinite(v).all()]
            if bad:
                failures.append(Failure(name, f"non-finite: {', '.join(bad)}"))
            n = pb.events(arrays)
            if n:
                failures.append(Failure(name, f"{n:g} spikes or events with no input"))
        return CheckResult(
            "fail" if failures else "pass",
            tested=len(out),
            failed=len(failures),
            failures=failures,
            numbers={"total_events": sum(pb.events(v) for v in out.values())},
        )


class SingleSensor(Check):
    """One active sensor gives finite arrays that reach a population."""

    id = "core.run.single_sensor"
    pack = "core"
    scope = "probe"
    title = "A single active sensor"
    what = (
        "One sensor at the data's peak value for the middle half of the run, "
        "noise off. "
        "Every array must be finite, and at least one population's drive must be "
        "non-zero (the sensor reaches the neurons)."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        probe = pb.build(ctx, pb.quiet(pb.base_config(ctx)))
        frames = pb.pattern(probe, 0.0)
        t = probe.n_frames
        centre = (probe.rows // 2, probe.cols // 2)
        frames[(slice(t // 4, 3 * t // 4),) + (Ellipsis, *centre)] = ctx.peak()
        out = pb.run(probe, frames, ctx.seed)
        failures = []
        for name, arrays in out.items():
            bad = [k for k, v in arrays.items() if v.size and not np.isfinite(v).all()]
            if bad:
                failures.append(Failure(name, f"non-finite: {', '.join(bad)}"))
        reached = sum(float(np.abs(v["drive"]).sum()) for v in out.values())
        if not reached:
            failures.append(
                Failure("all populations", "the sensor's stimulus gave no drive")
            )
        return CheckResult(
            "fail" if failures else "pass",
            tested=len(out),
            failed=len(failures),
            failures=failures,
            numbers={
                "total_events": sum(pb.events(v) for v in out.values()),
                "total_abs_drive": reached,
            },
        )


class Saturation(Check):
    """A stimulus far above the data's peak still gives finite, bounded arrays."""

    id = "core.run.saturation"
    pack = "core"
    scope = "probe"
    title = "Saturating stimulus"
    what = (
        f"The probe stimulus at {SATURATION_FACTOR:g} times the data's peak value, "
        "noise off. Every array must be finite and no population may count more "
        "spikes in a bin "
        "than it has integration steps."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        probe = pb.build(ctx, pb.quiet(pb.base_config(ctx)))
        out = pb.run(probe, pb.pattern(probe, SATURATION_FACTOR * ctx.peak()), ctx.seed)
        sim = probe.config.simulation
        steps = float(sim.dt_ms) / float(sim.integrate_dt_ms)
        failures = []
        most = 0.0
        for name, arrays in out.items():
            bad = [k for k, v in arrays.items() if v.size and not np.isfinite(v).all()]
            if bad:
                failures.append(Failure(name, f"non-finite: {', '.join(bad)}"))
            key = pb.readout_of(arrays)
            if key in ("spikes", "events") and arrays[key].size:
                top = float(np.abs(arrays[key]).max())
                most = max(most, top)
                if top > steps + 1e-6:
                    failures.append(
                        Failure(
                            name, f"{top:g} counts in one bin, at most {steps:g} steps"
                        )
                    )
        return CheckResult(
            "fail" if failures else "pass",
            tested=len(out),
            failed=len(failures),
            failures=failures,
            numbers={"most_counts_in_a_bin": most, "steps_per_bin": steps},
        )


# -------------------------------------------------------------------- re-runs


def _targets(ctx: CheckContext) -> List[Tuple[str, str, str]]:
    """``(entry, population, readout)`` for the sampled entries."""
    if "targets" not in ctx.cache:
        out = []
        for entry in ctx.sampled_entries():
            try:
                info = ctx.reader.peek(entry)
            except (ReadError, KeyError):
                continue
            out += [(entry, name, readout) for name, readout, _n in info.populations]
        ctx.cache["targets"] = out
    return ctx.cache["targets"]


def _rerun(ctx: CheckContext, entry: str, pop: str, readout: str, clip: bool) -> Any:
    """A population's re-simulation, cached; ``None`` when it gives no voltage."""
    key = ("rerun", entry, pop, clip)
    if key not in ctx.cache:
        from sensoryforge.evaluate.server.resim import ResimError, resimulate

        try:
            path = ctx.reader.bundle_path(entry)
            ctx.cache[key] = resimulate(path, pop, readout, clip_input=clip)
        except (ResimError, ReadError, KeyError):
            ctx.cache[key] = None
    return ctx.cache[key]


def _need_targets(ctx: CheckContext) -> List[Tuple[str, str, str]]:
    targets = _targets(ctx)
    if not targets:
        raise CheckSkipped("no entry ran, so there is nothing to re-run")
    return targets


class RerunMatches(Check):
    """The neuron stage, run again from the stored input, reproduces the readout."""

    id = "core.run.rerun_matches"
    pack = "core"
    scope = "entry"
    title = "Neuron stage reproduces the stored readout"
    what = (
        "For a few entries spread over the data set, each population's neuron stage "
        "runs again from its stored filtered input with the code a run uses and must "
        "give the "
        "stored spike, event or state arrays exactly. A bundle written on another "
        "platform can differ in the last float step and fail here."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        failures, tested = [], 0
        worst = 0.0
        for entry, pop, readout in _need_targets(ctx):
            re = _rerun(ctx, entry, pop, readout, False)
            if re is None:
                continue
            tested += 1
            v = re.verification
            worst = max(worst, float(v.get("max_abs_difference") or 0.0))
            if not v["matches"]:
                failures.append(
                    Failure(
                        f"{entry} / {pop}",
                        f"{v['n_differing']} of {v['n_units']} neurons differ "
                        f"(largest {v.get('max_abs_difference')})",
                        entry,
                    )
                )
        if not tested:
            raise CheckSkipped("no population's neuron stage could be re-run")
        return CheckResult(
            "fail" if failures else "pass",
            tested=tested,
            failed=len(failures),
            failures=failures,
            numbers={"largest_difference": worst},
        )


class InputFloor(Check):
    """A declared input floor acts as a clip of the neuron's input."""

    id = "core.run.input_floor"
    pack = "core"
    scope = "entry"
    title = "Declared input floor is applied"
    what = (
        "Where a population declares an input floor, running its neuron stage with the "
        "floor must equal running it on the stored filtered input clipped at the floor "
        "with no floor: the neuron never sees a current below it."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        failures, tested = [], 0
        for entry, pop, readout in _need_targets(ctx):
            plain = _rerun(ctx, entry, pop, readout, False)
            if plain is None or plain.verification.get("input_floor_ma") is None:
                continue
            clipped = _rerun(ctx, entry, pop, readout, True)
            if clipped is None:
                continue
            tested += 1
            floor = float(plain.verification["input_floor_ma"])
            if not _same(plain.voltages, clipped.voltages):
                failures.append(
                    Failure(
                        f"{entry} / {pop}",
                        f"voltage differs by up to "
                        f"{_max_abs(plain.voltages, clipped.voltages):.6g} mV "
                        f"with the floor at {floor:g} mA",
                        entry,
                    )
                )
        if not tested:
            raise CheckSkipped("no population declares an input floor")
        return CheckResult(
            "fail" if failures else "pass",
            tested=tested,
            failed=len(failures),
            failures=failures,
        )


class Voltage(Check):
    """Re-simulated membrane voltage is finite and, if limits are given, inside them."""

    id = "core.run.voltage"
    pack = "core"
    scope = "entry"
    title = "Membrane voltage is finite and in range"
    what = (
        "The re-run neuron stage's voltage must be finite. When limits are given "
        "(--voltage-bounds LOW,HIGH, in mV) it must also stay inside them; none are "
        "assumed."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        failures, tested = [], 0
        low, high = float("inf"), float("-inf")
        for entry, pop, readout in _need_targets(ctx):
            re = _rerun(ctx, entry, pop, readout, False)
            if re is None:
                continue
            tested += 1
            v = re.voltages
            if not np.isfinite(v).all():
                failures.append(
                    Failure(
                        f"{entry} / {pop}",
                        f"{int((~np.isfinite(v)).sum())} non-finite voltages",
                        entry,
                    )
                )
                continue
            low, high = min(low, float(v.min())), max(high, float(v.max()))
            if ctx.bounds_mv is not None and (
                v.min() < ctx.bounds_mv[0] or v.max() > ctx.bounds_mv[1]
            ):
                failures.append(
                    Failure(
                        f"{entry} / {pop}",
                        f"voltage {float(v.min()):.4g} to {float(v.max()):.4g} mV, "
                        f"outside {ctx.bounds_mv[0]:g} to {ctx.bounds_mv[1]:g} mV",
                        entry,
                    )
                )
        if not tested:
            raise CheckSkipped("no population gives a membrane voltage to look at")
        note = (
            None
            if ctx.bounds_mv
            else "no limits were given, so only finiteness was checked"
        )
        return CheckResult(
            "fail" if failures else "pass",
            tested=tested,
            failed=len(failures),
            failures=failures,
            numbers={"lowest_mv": low, "highest_mv": high},
            note=note,
        )
