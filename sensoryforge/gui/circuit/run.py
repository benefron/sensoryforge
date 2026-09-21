"""Shared graph-execution helpers (Phase 3, Wave Q, Q1).

:meth:`~sensoryforge.gui.tabs.circuit_tab.CircuitTab.run_graph` (Wave O, O4)
originally inlined "build a config from the flowchart, render its stimulus,
run :class:`~sensoryforge.core.simulation_engine.SimulationEngine`, optionally
write a bundle" as one method. Wave Q's batch sweep (Q1) needs to do exactly
that, once per swept value, so the config-building/render/run/bundle part is
pulled out here as :func:`run_graph_once` and both the tab and
:mod:`sensoryforge.gui.circuit.sweep` call it -- a refactor with no behaviour
change to ``CircuitTab.run_graph`` (its own tests, ``tests/unit/test_circuit_run.py``,
are unchanged and still pass).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyqtgraph.flowchart import Flowchart

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui.circuit.serialise import graph_to_config
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.stimuli.render import render_stimulus


def _dropped_params_warning(stimulus_type: str, dropped) -> Optional[str]:
    """The warning text for discarded stimulus settings, or ``None``.

    Args:
        stimulus_type: The stimulus's registered name, for the message.
        dropped: ``(field_name, value)`` pairs the constructor rejected.

    Returns:
        A message naming only the fields whose value the user had changed
        from the schema default, or ``None`` when every discarded field was
        untouched and there is nothing worth saying.
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


def _is_schema_default(field_name: str, value) -> bool:
    """Whether *value* is what ``StimulusConfig`` would hold untouched.

    ``StimulusConfig.to_dict()`` carries every field the schema defines,
    most of which a given stimulus class knows nothing about. Discarding
    those is housekeeping. Discarding one the user actually set is a
    changed stimulus, so the two cases are told apart here rather than
    warning about all of them and training the reader to ignore it.

    Args:
        field_name: The dropped keyword.
        value: The value it held.

    Returns:
        ``True`` when the field is unknown to the schema or still at its
        declared default.
    """
    import dataclasses

    from sensoryforge.config.schema import StimulusConfig

    for field in dataclasses.fields(StimulusConfig):
        if field.name != field_name:
            continue
        if field.default is not dataclasses.MISSING:
            return value == field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value == field.default_factory()  # type: ignore[misc]
        return False
    return True


def render_graph_stimulus(
    config: SensoryForgeConfig, duration_ms: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render ``config``'s stimulus on its first grid's receptor coordinates.

    Args:
        config: The reconstructed :class:`SensoryForgeConfig`.
        duration_ms: Stimulus duration in ms.

    Returns:
        ``(stimulus_tensor, frames)`` -- ``stimulus_tensor`` is
        ``frames.unsqueeze(0)`` (batch dimension added), matching what
        :class:`~sensoryforge.core.simulation_engine.SimulationEngine.run`
        expects.
    """
    if config.grids:
        grid_cfg = config.grids[0]
        canvas = stimulus_canvas(grid_cfg, device=config.simulation.device)
        xx, yy = canvas.xx, canvas.yy
    else:
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 40),
            torch.linspace(-1, 1, 40),
            indexing="ij",
        )

    stim = config.stimulus
    # StimulusConfig.to_dict() carries every field the schema has
    # (administrative ones like motion/composition_mode/channel included);
    # a given registered stimulus class's constructor only accepts its own
    # subset. Retry dropping whichever keyword the constructor just
    # rejected, the same way render.py's own envelope-key retry works,
    # rather than hard-coding a per-type field list here.
    stimulus_params = {
        k: v for k, v in stim.to_dict().items() if k not in ("name", "type")
    }
    dropped: List[Any] = []
    while True:
        try:
            frames, _ = render_stimulus(
                stim.type,
                stimulus_params,
                xx,
                yy,
                dt_ms=config.simulation.dt_ms,
                duration_ms=duration_ms,
                device=config.simulation.device,
            )
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if match is None or match.group(1) not in stimulus_params:
                raise
            key = match.group(1)
            dropped.append((key, stimulus_params.pop(key)))

    # Dropping a field the user never set is housekeeping; dropping one
    # they did set changes the stimulus they asked for, and doing that
    # silently is how a graph ends up describing a run that did not
    # happen. Warn for the second case only, so the message means
    # something when it appears.
    message = _dropped_params_warning(stim.type, dropped)
    if message is not None:
        warnings.warn(message, UserWarning, stacklevel=2)
    stimulus_tensor = frames.unsqueeze(0)
    return stimulus_tensor, frames


def run_graph_once(
    flowchart: Flowchart,
    *,
    duration_ms: float = 200.0,
    bundle_dir: Optional[str] = None,
) -> Tuple[SensoryForgeConfig, Dict[str, Any], torch.Tensor, float]:
    """Build a config from ``flowchart``, run it, and optionally write a bundle.

    Args:
        flowchart: The Circuit tab's live flowchart.
        duration_ms: Stimulus duration in ms.
        bundle_dir: When given, the run's bundle is written there via
            :class:`~sensoryforge.core.simulation_engine.SimulationEngine.run`
            (the Wave J writer) -- exactly as ``sensoryforge run --bundle`` does.
            ``None`` (the default) falls back to the graph's own
            ``RecordNode.output_dir`` (``config.metadata["record_output_dir"]``),
            so a graph with a configured ``RecordNode`` still writes a bundle
            with no caller-supplied path.

    Returns:
        ``(config, raw_results, frames, dt_ms)`` -- ``raw_results`` is the
        population-name -> result-dict mapping
        ``SimulationEngine.run(return_intermediates=True)`` returns.
    """
    config = graph_to_config(flowchart)
    if bundle_dir is None:
        bundle_dir = config.metadata.get("record_output_dir")
    stimulus_tensor, frames = render_graph_stimulus(config, duration_ms)

    engine = SimulationEngine(config)
    raw_results = engine.run(
        stimulus_tensor,
        return_intermediates=True,
        bundle_dir=bundle_dir,
        stimulus_config=config.stimulus.to_dict(),
    )
    return config, raw_results, frames, config.simulation.dt_ms
