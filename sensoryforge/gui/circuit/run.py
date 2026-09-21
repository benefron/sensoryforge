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

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from pyqtgraph.flowchart import Flowchart

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui.circuit.serialise import graph_to_config
from sensoryforge.stimuli.render import (
    render_for_config,
    dropped_params_warning,
    _is_stimulus_schema_default,
)

# Task 0.6 (F-061) moved the dropped-keyword-warning logic to
# sensoryforge.stimuli.render, shared with sensoryforge.cli. These two
# names stay importable from here (sensoryforge.gui.tabs.circuit_tab
# re-exports them, and tests/unit/test_circuit_dropped_params.py imports
# them from there) so nothing downstream needs to change.
_dropped_params_warning = dropped_params_warning
_is_schema_default = _is_stimulus_schema_default


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
    # The actual rendering (canvas selection, dispatch through
    # STIMULUS_REGISTRY, dropped-keyword retry) lives in
    # sensoryforge.stimuli.render.render_for_config (Task 0.6, F-061), shared
    # with sensoryforge.cli's `sensoryforge run` so both entry points render
    # a config's canonical stimulus: block identically.
    stimulus_tensor, _time_ms, _canvas, dropped = render_for_config(
        config, duration_ms=duration_ms, dt_ms=config.simulation.dt_ms
    )

    # Dropping a field the user never set is housekeeping; dropping one
    # they did set changes the stimulus they asked for, and doing that
    # silently is how a graph ends up describing a run that did not
    # happen. Warn for the second case only, so the message means
    # something when it appears.
    message = dropped_params_warning(config.stimulus.type, dropped)
    if message is not None:
        warnings.warn(message, UserWarning, stacklevel=2)
    frames = stimulus_tensor[0]
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
