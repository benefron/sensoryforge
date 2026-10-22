"""The ``core`` metric pack (report contract, section 6): modality-neutral statistics.

Frame statistics per channel (``core.frames.*``), each population's input and response
(``core.input.*``, ``core.response.*``), the entry's run time and two data-set counts.
The arithmetic is in :mod:`frame_metrics` and :mod:`response_metrics`; this class names
the metrics, gives them units and orders the values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sensoryforge.evaluate.audit import frame_metrics, response_metrics
from sensoryforge.evaluate.audit.units import unit_of
from sensoryforge.evaluate.pack import (
    DatasetRow,
    MetricPack,
    metric_definition,
)

#: ``(name, unit expression, value type, is a time measured on the frame grid)``.
_FRAME_METRICS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("n_frames", "count", "integer", False),
    ("dt", "time", "number", False),
    ("duration", "time", "number", True),
    ("peak", "frame", "number", False),
    ("trough", "frame", "number", False),
    ("mean", "frame", "number", False),
    ("rms", "frame", "number", False),
    ("negative_fraction", "1", "number", False),
    ("onset_time", "time", "number", True),
    ("offset_time", "time", "number", True),
    ("active_time", "time", "number", True),
    ("active_fraction", "1", "number", False),
    ("bouts", "count", "integer", False),
    ("bout_mean", "time", "number", True),
    ("gap_mean", "time", "number", True),
    ("gap_max", "time", "number", True),
    ("footprint_mean", "1", "number", False),
    ("footprint_max", "1", "number", False),
    ("centroid_x", "space", "number", False),
    ("centroid_y", "space", "number", False),
    ("centroid_z", "space", "number", False),
    ("spread", "space", "number", False),
    ("path_length", "space", "number", False),
    ("centroid_speed", "space/time", "number", False),
    ("direction", "deg", "number", False),
    ("dominant_frequency", "1/time", "number", False),
    ("digest", "", "string", False),
)

_INPUT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("drive_peak", "drive"),
    ("drive_mean", "drive"),
    ("filtered_max", "filtered"),
    ("filtered_min", "filtered"),
)

_SPIKES, _EVENTS, _STATE = "spikes", "events", "state"
_ALL = (_SPIKES, _EVENTS, _STATE)

#: ``(name, unit expression, value type, readouts it applies to)``.
_RESPONSE_METRICS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("n_units", "count", "integer", _ALL),
    ("count", "count", "integer", (_SPIKES,)),
    ("rate_mean", "1/time", "number", (_SPIKES,)),
    ("rate_active", "1/time", "number", (_SPIKES,)),
    ("rate_quiet", "1/time", "number", (_SPIKES,)),
    ("active_unit_fraction", "1", "number", (_SPIKES, _EVENTS)),
    ("latency", "time", "number", (_SPIKES, _EVENTS)),
    ("isi_cv", "1", "number", (_SPIKES,)),
    ("events_on", "count", "integer", (_EVENTS,)),
    ("events_off", "count", "integer", (_EVENTS,)),
    ("state_mean", "state", "number", (_STATE,)),
    ("state_std", "state", "number", (_STATE,)),
    ("state_min", "state", "number", (_STATE,)),
    ("state_max", "state", "number", (_STATE,)),
    ("digest", "", "string", _ALL),
)

_AXES = ("x", "y", "z")
_RUN_SOURCE = "index/task_*.jsonl:seconds"


class CorePack(MetricPack):
    """Modality-neutral frame and response statistics.

    Args:
        zero_tolerance: A frame value is active when ``|x| > zero_tolerance`` (frame
            unit; default 0, exactly non-zero).
        histogram_bins: Equal-width histogram bins of every summary.

    Raises:
        ValueError: If ``zero_tolerance`` is negative or ``histogram_bins`` is below 1.
    """

    name = "core"
    version = "1"
    dataset_inputs = ("core.frames.digest", "core.frames.bouts")

    def __init__(self, zero_tolerance: float = 0.0, histogram_bins: int = 32) -> None:
        if zero_tolerance < 0:
            raise ValueError(f"zero_tolerance must be >= 0, got {zero_tolerance}")
        if histogram_bins < 1:
            raise ValueError(f"histogram_bins must be >= 1, got {histogram_bins}")
        self.zero_tolerance = float(zero_tolerance)
        self.histogram_bins = int(histogram_bins)

    def to_dict(self) -> Dict[str, Any]:
        """The pack's parameters (``zero_tolerance``, ``histogram_bins``)."""
        return {
            "zero_tolerance": self.zero_tolerance,
            "histogram_bins": self.histogram_bins,
        }

    @classmethod
    def get_param_spec(cls) -> List[Any]:
        """The pack's parameters as ``ParamSpec`` objects."""
        from sensoryforge.stimuli.base import ParamSpec

        return [
            ParamSpec(
                "zero_tolerance",
                dtype="float",
                default=0.0,
                min_val=0.0,
                unit="frame unit",
                help="A frame value counts as active when its magnitude exceeds this.",
            ),
            ParamSpec(
                "histogram_bins",
                dtype="int",
                default=32,
                min_val=1,
                help="Equal-width bins of every summary's histogram.",
            ),
        ]

    # ------------------------------------------------------------------ catalogue

    def metrics(
        self, layout: Any, units: Mapping[str, Any], dt: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """The pack's definitions in catalogue order (contract, section 6).

        Args:
            layout: The shared layout; it fixes which ``centroid_*`` metrics exist.
            units: The resolved units by quantity.
            dt: The data set's frame step (the resolution of the times measured on the
                frame grid); ``None`` when the bundles differ.

        Returns:
            One definition per metric id.
        """
        out: List[Dict[str, Any]] = []
        for name, expr, vtype, on_grid in _FRAME_METRICS:
            if name.startswith("centroid_") and name[-1] in _AXES:
                if _AXES.index(name[-1]) >= layout.dims:
                    continue
            out.append(
                metric_definition(
                    f"core.frames.{name}",
                    unit_of(expr, units) if expr else None,
                    scope="channel",
                    value_type=vtype,
                    resolution=None,
                    summarised=vtype != "string",
                )
            )
            if on_grid:
                out[-1]["resolution"] = dt
        for name, expr in _INPUT_METRICS:
            out.append(
                metric_definition(
                    f"core.input.{name}", unit_of(expr, units), scope="population"
                )
            )
        for name, expr, vtype, applies in _RESPONSE_METRICS:
            out.append(
                metric_definition(
                    f"core.response.{name}",
                    unit_of(expr, units) if expr else None,
                    scope="population",
                    value_type=vtype,
                    applies_to=None if applies == _ALL else applies,
                    summarised=vtype != "string",
                )
            )
        out.append(
            metric_definition(
                "core.run.seconds",
                {"symbol": "s", "origin": "data", "source": _RUN_SOURCE},
                scope="entry",
            )
        )
        for name in ("silent_entries", "duplicate_frames"):
            out.append(
                metric_definition(
                    f"core.dataset.{name}",
                    {"symbol": "count", "origin": "derived"},
                    scope="entry",
                    value_type="integer",
                    level="dataset",
                    summarised=False,
                )
            )
        return out

    # -------------------------------------------------------------------- measure

    def measure(self, view: Any) -> List[Dict[str, Any]]:
        """Measure one entry.

        Args:
            view: The entry's :class:`EntryView`.

        Returns:
            Values in catalogue order; frame metrics per channel, population metrics per
            population the metric applies to.
        """
        coords = view.layout.coords
        n_ch = view.frames.shape[1]
        per_channel = [
            frame_metrics.measure_channel(
                view.frames[:, c, :], coords, view.dt, self.zero_tolerance
            )
            for c in range(n_ch)
        ]
        active = np.logical_or.reduce([p["_active"] for p in per_channel])
        values: List[Dict[str, Any]] = []
        for name, _expr, _vtype, _grid in _FRAME_METRICS:
            for c, stats in enumerate(per_channel):
                if name not in stats:
                    break
                values.append(
                    _value(f"core.frames.{name}", stats[name], channel=c, n_ch=n_ch)
                )
        pops = {}
        for pop_name, pop in view.populations.items():
            stats = response_metrics.input_stats(pop.drive, pop.filtered)
            stats.update(
                response_metrics.response_stats(
                    pop.readout, pop.data, pop.attrs, active, view.dt
                )
            )
            pops[pop_name] = (pop.readout, stats)
        for name, _expr in _INPUT_METRICS:
            for pop_name, (_readout, stats) in pops.items():
                values.append(_value(f"core.input.{name}", stats[name], pop=pop_name))
        for name, _expr, _vtype, applies in _RESPONSE_METRICS:
            for pop_name, (readout, stats) in pops.items():
                if readout in applies:
                    values.append(
                        _value(f"core.response.{name}", stats.get(name), pop=pop_name)
                    )
        values.append(_value("core.run.seconds", view.seconds))
        return values

    def measure_dataset(self, rows: Sequence[DatasetRow]) -> List[Dict[str, Any]]:
        """Count the silent entries and the entries that repeat an earlier one's frames.

        Args:
            rows: One row per measured entry, sorted by id, carrying
                ``core.frames.digest`` and ``core.frames.bouts`` (per channel).

        Returns:
            ``core.dataset.silent_entries`` and ``core.dataset.duplicate_frames``.
        """
        silent, duplicates = 0, 0
        seen = set()
        for row in rows:
            bouts = [
                v
                for k, v in sorted(row.values.items(), key=_by_channel)
                if k[0] == "core.frames.bouts"
            ]
            digests = tuple(
                v
                for k, v in sorted(row.values.items(), key=_by_channel)
                if k[0] == "core.frames.digest"
            )
            has_activity = any(b for b in bouts if b is not None)
            if not has_activity:
                silent += 1
                continue
            if row.noise_repeat != 0:
                continue
            if digests in seen:
                duplicates += 1
            seen.add(digests)
        return [
            {"id": "core.dataset.silent_entries", "value": silent},
            {"id": "core.dataset.duplicate_frames", "value": duplicates},
        ]


def _by_channel(item: Tuple[Any, Any]) -> Tuple[str, int]:
    key = item[0]
    return (key[0], -1 if key[1] is None else key[1])


def _value(
    metric_id: str,
    value: Any,
    channel: Optional[int] = None,
    n_ch: int = 1,
    pop: Optional[str] = None,
) -> Dict[str, Any]:
    """One value object; ``channel`` appears only when the data has several channels."""
    from sensoryforge.evaluate.audit.summary import finite_or_none

    out: Dict[str, Any] = {"id": metric_id, "value": finite_or_none(value)}
    if channel is not None and n_ch > 1:
        out["channel"] = channel
    if pop is not None:
        out["population"] = pop
    return out
