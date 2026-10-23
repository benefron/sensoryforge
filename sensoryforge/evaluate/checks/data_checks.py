"""Checks that read every entry's data through the blind reader (no engine needed).

One pass over the entries serves all four checks: ``readable`` (the index says the
entry ran and its bundle opens), ``finite`` (no NaN or infinity in any array),
``readout_values`` (spike and event counts are non-negative whole numbers) and
``time_grid`` (frame times rise by the frame step and every population has one row per
frame).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sensoryforge.evaluate.audit.reader import EntryView, ReadError
from sensoryforge.evaluate.checks.base import (
    Check,
    CheckContext,
    CheckResult,
    Failure,
)

#: Relative tolerance on the spacing of frame times (they are float64 multiples of dt).
TIME_TOLERANCE = 1e-6

_PASS_KEYS = ("readable", "finite", "readout_values", "time_grid")


def _array_problems(view: EntryView) -> List[str]:
    """Names of the arrays that hold NaN or infinity."""
    bad = []
    named = [("frames", view.frames), ("time", view.time)]
    for name, pop in view.populations.items():
        named += [
            (f"{name}.{pop.readout}", pop.data),
            (f"{name}.drive", pop.drive),
            (f"{name}.filtered", pop.filtered),
        ]
    for label, array in named:
        if array.size and not np.isfinite(array).all():
            n = int((~np.isfinite(array)).sum())
            bad.append(f"{label} has {n} non-finite values")
    return bad


def _readout_problems(view: EntryView) -> List[str]:
    bad = []
    for name, pop in view.populations.items():
        if pop.readout not in ("spikes", "events") or not pop.data.size:
            continue
        data = np.asarray(pop.data, dtype=np.float64)
        finite = data[np.isfinite(data)]
        if pop.readout == "spikes" and (finite < 0).any():
            bad.append(f"{name}: {int((finite < 0).sum())} negative spike counts")
        if (finite != np.rint(finite)).any():
            bad.append(f"{name}: {pop.readout} are not whole numbers")
    return bad


def _time_problems(view: EntryView) -> List[str]:
    bad = []
    n = len(view.time)
    if view.frames.shape[0] != n:
        bad.append(f"{view.frames.shape[0]} frames but {n} times")
    for name, pop in view.populations.items():
        for label, array in (("readout", pop.data), ("drive", pop.drive)):
            if array.shape[0] != n:
                bad.append(f"{name} {label} has {array.shape[0]} rows for {n} frames")
    if n > 1 and np.isfinite(view.time).all():
        steps = np.diff(view.time)
        off = np.abs(steps - view.dt) > TIME_TOLERANCE * max(view.dt, 1e-12)
        if off.any():
            first = int(np.flatnonzero(off)[0])
            bad.append(
                f"{int(off.sum())} steps differ from dt={view.dt:g} "
                f"(first at frame {first + 1}: {steps[first]:g})"
            )
    return bad


def data_pass(ctx: CheckContext) -> Dict[str, Tuple[int, List[Failure]]]:
    """Read the entries once and sort out the four checks' findings.

    Args:
        ctx: The context; the result is cached on it.

    Returns:
        For each of ``readable``, ``finite``, ``readout_values`` and ``time_grid``:
        ``(entries tried, failures)``.
    """
    if "data_pass" in ctx.cache:
        return ctx.cache["data_pass"]
    tried = {k: 0 for k in _PASS_KEYS}
    found: Dict[str, List[Failure]] = {k: [] for k in _PASS_KEYS}
    rows = ctx.reader.rows
    if ctx.limit is not None:
        rows = rows[: ctx.limit]
    for row in rows:
        tried["readable"] += 1
        if row.status != "ok":
            why = row.error or (
                "no row in the batch index" if row.status is None else ""
            )
            found["readable"].append(
                Failure(
                    row.entry,
                    f"status {row.status or 'unknown'}: {why}".strip(": "),
                    row.entry,
                )
            )
            continue
        try:
            view = ctx.reader.read_entry(row.entry)
        except ReadError as exc:
            found["readable"].append(Failure(row.entry, str(exc), row.entry))
            continue
        for key, problems in (
            ("finite", _array_problems(view)),
            ("readout_values", _readout_problems(view)),
            ("time_grid", _time_problems(view)),
        ):
            tried[key] += 1
            if problems:
                found[key].append(Failure(row.entry, "; ".join(problems), row.entry))
    out = {k: (tried[k], found[k]) for k in _PASS_KEYS}
    ctx.cache["data_pass"] = out
    return out


class _DataCheck(Check):
    scope = "entry"
    pack = "core"
    key = ""

    def run(self, ctx: CheckContext) -> CheckResult:
        tried, failures = data_pass(ctx)[self.key]
        if not tried:
            return CheckResult("skipped", note="the data set has no entries")
        return CheckResult(
            "fail" if failures else "pass",
            tested=tried,
            failed=len(failures),
            failures=failures,
        )


class Readable(_DataCheck):
    """Every entry ran and its bundle opens."""

    id = "core.data.readable"
    key = "readable"
    title = "Every entry ran and reads"
    what = (
        "The batch index lists the entry with status ok and its bundle opens under the "
        "blind reader's rules."
    )


class Finite(_DataCheck):
    """No NaN or infinity in a frame, time, drive, filtered or readout array."""

    id = "core.data.finite"
    key = "finite"
    title = "No NaN or infinity"
    what = "Every stored array of every entry that read holds only finite numbers."


class ReadoutValues(_DataCheck):
    """Spike and event counts are whole numbers (spike counts non-negative)."""

    id = "core.data.readout_values"
    key = "readout_values"
    title = "Spike and event counts are whole numbers"
    what = (
        "Spike counts are non-negative whole numbers per bin; signed event counts are "
        "whole numbers."
    )


class TimeGrid(_DataCheck):
    """Frame times rise by the frame step and every array has one row per frame."""

    id = "core.data.time_grid"
    key = "time_grid"
    title = "Time grid and array lengths"
    what = (
        "Frame times rise by the frame step dt (to a relative 1e-6), and the frames "
        "and each population's readout and drive have one row per frame."
    )
