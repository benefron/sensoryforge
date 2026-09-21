"""Render a config's stimulus the way the CLI does, for the GUI v2 shell.

One renderer, one canvas: :func:`render_for_config` builds the same regular
canvas :func:`sensoryforge.stimuli.canvas.stimulus_canvas` gives
``sensoryforge/cli.py`` (so ``poisson``/``hex`` arrangements render, F-076)
and calls :func:`sensoryforge.stimuli.render.render_stimulus` on it with
exactly the parameters :class:`~sensoryforge.config.schema.StimulusConfig`
carries. The GUI therefore shows, and runs, the stimulus the CLI would
produce for the same YAML -- the point of the whole rewrite.

Shapes and units follow the rest of SensoryForge: the returned stimulus is
``[1, time, H, W]`` in mA, ``time_ms`` is ms, canvas coordinates are mm.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from sensoryforge.config.schema import SensoryForgeConfig, StimulusConfig
from sensoryforge.stimuli.canvas import StimulusCanvas, stimulus_canvas
from sensoryforge.stimuli.render import render_stimulus

#: Keys of ``StimulusConfig.to_dict()`` that are never stimulus constructor
#: parameters: they name the stimulus rather than shape it.
_NON_PARAM_KEYS = ("name", "type")


@dataclass
class RenderedStimulus:
    """A config's stimulus, ready for :meth:`SimulationEngine.run`.

    Attributes:
        stimulus: ``[1, time, H, W]`` (or ``[1, time, C, H, W]`` on a
            multi-channel grid) in mA -- the batch dimension the engine wants.
        time_ms: ``[time]`` sample times in ms.
        canvas: The :class:`~sensoryforge.stimuli.canvas.StimulusCanvas` the
            frames were drawn on, in mm, so a preview can label its axes.
        dropped: ``(field_name, value)`` pairs the stimulus constructor did
            not accept, in the order they were discarded.
        warning: A message naming only the *deliberately set* dropped fields
            (F-061), or ``None`` when every dropped field was still at its
            schema default and there is nothing worth saying.
    """

    stimulus: torch.Tensor
    time_ms: np.ndarray
    canvas: StimulusCanvas
    dropped: List[Tuple[str, Any]] = dataclasses.field(default_factory=list)
    warning: Optional[str] = None


def _is_schema_default(field_name: str, value: Any) -> bool:
    """Whether ``value`` is what :class:`StimulusConfig` would hold untouched.

    ``StimulusConfig.to_dict()`` carries every field the schema defines, most
    of which a given stimulus class knows nothing about. Discarding those is
    housekeeping. Discarding one the user actually set is a *different*
    stimulus, so the two cases are told apart here rather than warning about
    all of them and training the reader to ignore the warning.

    Args:
        field_name: The dropped keyword.
        value: The value it held.

    Returns:
        ``True`` when the field is unknown to the schema or still at its
        declared default.
    """
    for field in dataclasses.fields(StimulusConfig):
        if field.name != field_name:
            continue
        if field.default is not dataclasses.MISSING:
            return value == field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value == field.default_factory()  # type: ignore[misc]
        return False
    return True


def dropped_params_warning(
    stimulus_type: str, dropped: List[Tuple[str, Any]]
) -> Optional[str]:
    """The warning text for discarded stimulus settings, or ``None`` (F-061).

    Args:
        stimulus_type: The stimulus's registered name, for the message.
        dropped: ``(field_name, value)`` pairs the constructor rejected.

    Returns:
        A message naming only the fields whose value differs from the schema
        default, or ``None`` when every discarded field was untouched.
    """
    deliberate = [
        f"{key}={value!r}"
        for key, value in dropped
        if not _is_schema_default(key, value)
    ]
    if not deliberate:
        return None
    return (
        f"Stimulus {stimulus_type!r} does not accept {', '.join(deliberate)}; "
        "the value(s) you set were ignored and the stimulus ran without them."
    )


def _grid_for_stimulus(config: SensoryForgeConfig):
    """The grid whose extent the stimulus is rendered on.

    Args:
        config: The experiment.

    Returns:
        ``config.stimulus.target_layer``'s :class:`GridConfig` when that field
        names a grid, else the first grid.

    Raises:
        ValueError: If the config has no grids at all (nothing to render on),
            or ``target_layer`` names a grid that is not in the config --
            listing the names there are.
    """
    if not config.grids:
        raise ValueError(
            "cannot render a stimulus: the configuration has no grids. Add a "
            "receptor grid on the Sensors screen first."
        )
    target = config.stimulus.target_layer
    if not target:
        return config.grids[0]
    for grid in config.grids:
        if grid.name == target:
            return grid
    known = [grid.name for grid in config.grids]
    raise ValueError(
        f"stimulus target_layer {target!r} is not a grid in this "
        f"configuration; it has {known}"
    )


def _render_frames(
    config: SensoryForgeConfig, duration_ms: float, dt_ms: float
) -> Tuple[torch.Tensor, torch.Tensor, StimulusCanvas, List[Tuple[str, Any]]]:
    """Render ``config``'s stimulus; the one place rendering happens here.

    THE ONE PLACE STIMULUS RENDERING IS ISOLATED. This body is a copy of
    ``sensoryforge/gui/circuit/run.py::render_graph_stimulus`` (which the
    Circuit tab and ``sensoryforge/cli.py`` share the shape of). Task 0.6
    moves that body into the shared helper
    ``sensoryforge.stimuli.render.render_for_config(config, *, duration_ms,
    dt_ms) -> (stimulus, time_ms, canvas, dropped)``, whose return tuple is
    deliberately the one below: when it lands, this function's body becomes
    the single line ``return render_for_config(config, duration_ms=duration_ms,
    dt_ms=dt_ms)`` and nothing else in the GUI changes.

    Args:
        config: The experiment whose ``stimulus`` is rendered.
        duration_ms: Stimulus duration in ms.
        dt_ms: Record step in ms.

    Returns:
        ``(stimulus, time_ms, canvas, dropped)`` -- ``stimulus`` is
        ``[1, time, H, W]``, ``time_ms`` is ``[time]`` in ms, ``canvas`` is
        the render canvas in mm and ``dropped`` the ``(field, value)`` pairs
        the stimulus constructor rejected.

    Raises:
        ValueError: If the config has no grid to render on, or the stimulus
            type is unknown (raised by ``render_stimulus``).
    """
    grid_cfg = _grid_for_stimulus(config)
    device = config.simulation.device
    canvas = stimulus_canvas(grid_cfg, device=device)

    stim = config.stimulus
    # `StimulusConfig.to_dict()` carries every field the schema has
    # (administrative ones such as motion/composition_mode/channel included);
    # a given registered stimulus class's constructor accepts only its own
    # subset. Retry dropping whichever keyword the constructor just rejected,
    # the same way render.py's own envelope-key retry works, rather than
    # hard-coding a per-type field list.
    stimulus_params = {
        key: value
        for key, value in stim.to_dict().items()
        if key not in _NON_PARAM_KEYS
    }
    dropped: List[Tuple[str, Any]] = []
    while True:
        try:
            frames, time_ms = render_stimulus(
                stim.type,
                stimulus_params,
                canvas.xx,
                canvas.yy,
                dt_ms=dt_ms,
                duration_ms=duration_ms,
                device=device,
            )
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if match is None or match.group(1) not in stimulus_params:
                raise
            key = match.group(1)
            dropped.append((key, stimulus_params.pop(key)))

    return frames.unsqueeze(0), time_ms, canvas, dropped


def render_for_config(
    config: SensoryForgeConfig,
    *,
    duration_ms: float,
    dt_ms: Optional[float] = None,
) -> RenderedStimulus:
    """Render ``config``'s stimulus on its grid's canvas.

    Args:
        config: The experiment. Its ``stimulus``, its grid (``target_layer``
            or the first one) and its ``simulation.device`` are used.
        duration_ms: Stimulus duration in ms. ``round(duration_ms / dt_ms)``
            frames are produced, the same convention ``--duration`` uses.
        dt_ms: Record step in ms; defaults to ``config.simulation.dt_ms``.

    Returns:
        A :class:`RenderedStimulus`. Unlike ``render_graph_stimulus`` this
        does **not** emit a :class:`UserWarning` for dropped parameters -- the
        message is returned as :attr:`RenderedStimulus.warning` so the GUI can
        show it where the user is looking.

    Raises:
        ValueError: If ``duration_ms`` or the resolved ``dt_ms`` is not
            positive, if the config has no grid to render on, or if the
            stimulus type is not registered.

    Example:
        >>> rendered = render_for_config(config, duration_ms=200.0)  # doctest: +SKIP
        >>> rendered.stimulus.shape                                  # doctest: +SKIP
        torch.Size([1, 200, 80, 80])
    """
    step = config.simulation.dt_ms if dt_ms is None else float(dt_ms)
    if step <= 0:
        raise ValueError(f"dt_ms must be positive, got {step}")
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")

    stimulus, time_ms, canvas, dropped = _render_frames(config, duration_ms, step)
    return RenderedStimulus(
        stimulus=stimulus,
        time_ms=time_ms.detach().cpu().numpy(),
        canvas=canvas,
        dropped=dropped,
        warning=dropped_params_warning(config.stimulus.type, dropped),
    )
