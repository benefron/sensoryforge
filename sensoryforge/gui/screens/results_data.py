"""``ResultsView``: the one shape every Run & Results panel reads.

A live :class:`~sensoryforge.gui.session.RunResult` and a
:class:`~sensoryforge.io.bundle.Bundle` loaded from disk describe the same
run in two different shapes (a raw engine results dict plus a rendered
stimulus tensor, versus HDF5 datasets plus a receptive-field bank per
population). Every panel in ``screens/results.py`` reads only a
:class:`ResultsView`, built here from either source, so a panel never has to
know which one it is looking at.

Shapes and units follow the rest of SensoryForge: ``stimulus`` is
``[time, H, W]`` in mA, ``time_ms`` is ``[time]`` in ms, geometry is mm. A
population may be analog (a DSL model with no threshold): it carries
``state`` instead of ``spikes`` and :attr:`PopulationView.is_analog` is
``True`` -- panels must handle both, never assume spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui.session import RunResult
from sensoryforge.io.bundle import Bundle
from sensoryforge.stimuli.canvas import StimulusCanvas, stimulus_canvas


@dataclass
class PopulationView:
    """One population's data, whichever source it came from.

    Attributes:
        name: Population name.
        index: Position among the run's populations (0-based), used to pick
            a stable colour via ``theme.population_color(index, neuron_type)``.
        neuron_type: ``PopulationConfig.neuron_type`` (``"SA"``, ``"RA"``, ...).
        spikes: ``[T, N]`` sub-step spike counts, or ``None`` for an analog
            population.
        state: ``[T, N]`` analog readout trace, or ``None`` for a spiking
            population. Exactly one of ``spikes``/``state`` is not ``None``.
        drive: ``[T, N]`` innervation-weighted drive in mA, or ``None``.
        filtered: ``[T, N]`` post-filter current in mA, or ``None``.
        voltages: ``[T, N]`` membrane voltage in mV, or ``None`` (spiking
            populations only, and only when the run kept intermediates).
        neuron_centers: ``[N, 2]`` neuron centres in mm, or ``None``.
        receptor_coords: ``[M, 2]`` receptor positions in mm, or ``None``.
        weights: ``[N, M]`` receptive-field weights, or ``None``.
        state_var_name: For an analog population, the DSL model's state
            variable name (e.g. ``"v"``), read from ``PopulationConfig
            .dsl_config["state_vars"]``'s first key; ``"State"`` if that is
            not available. ``None`` for a spiking population.
    """

    name: str
    index: int
    neuron_type: str
    spikes: Optional[torch.Tensor]
    state: Optional[torch.Tensor]
    drive: Optional[torch.Tensor]
    filtered: Optional[torch.Tensor]
    voltages: Optional[torch.Tensor]
    neuron_centers: Optional[torch.Tensor]
    receptor_coords: Optional[torch.Tensor]
    weights: Optional[torch.Tensor]
    state_var_name: Optional[str] = None

    @property
    def is_analog(self) -> bool:
        """Whether this population has no spikes -- ``state`` in place of them."""
        return self.spikes is None and self.state is not None

    @property
    def n_neurons(self) -> int:
        """Neuron count, read from whichever readout tensor is present."""
        trace = self.spikes if self.spikes is not None else self.state
        if trace is None:
            return 0
        return int(trace.shape[-1])


@dataclass
class ResultsView:
    """One run, normalised for the Run & Results screen.

    Attributes:
        stimulus: ``[T, H, W]`` in mA -- the first channel only, for a
            multi-channel grid.
        time_ms: ``[T]`` sample times in ms.
        xlim: ``(x_min, x_max)`` mm the stimulus canvas spans.
        ylim: ``(y_min, y_max)`` mm the stimulus canvas spans.
        populations: One :class:`PopulationView` per population, in run order.
        label: What to show as the source of these results, e.g.
            ``"Viewing saved run 20260921-101500_demo"``.
    """

    stimulus: torch.Tensor
    time_ms: torch.Tensor
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    populations: List[PopulationView]
    label: str = ""

    def population(self, name: str) -> Optional[PopulationView]:
        """The population called ``name``, or ``None`` if there is none."""
        for pop in self.populations:
            if pop.name == name:
                return pop
        return None


def _first_channel(stimulus: torch.Tensor) -> torch.Tensor:
    """``[T, H, W]`` unchanged; ``[T, C, H, W]`` -> its first channel."""
    if stimulus.ndim == 4:
        return stimulus[:, 0]
    if stimulus.ndim == 3:
        return stimulus
    raise ValueError(
        f"expected a stimulus of shape [T, H, W] or [T, C, H, W], got "
        f"{list(stimulus.shape)}"
    )


def _squeeze_batch(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Drop a leading batch dim of size 1, or pass ``None`` through."""
    if t is None:
        return None
    if t.ndim >= 1 and t.shape[0] == 1:
        return t[0]
    return t


def _banks_from_config(config: SensoryForgeConfig) -> Dict[str, ReceptiveFieldBank]:
    """Rebuild every population's :class:`ReceptiveFieldBank` from a config.

    A live :class:`~sensoryforge.gui.session.RunResult` keeps the rendered
    stimulus and the raw results dict, but not the engine that built it, so
    the banks (needed for the neuron map and RF-weight panels) are rebuilt
    here on the CPU. Innervation is reproducible for a given seed (F-050),
    so this reproduces the run's own banks bit-for-bit.

    Args:
        config: The run's config snapshot.

    Returns:
        Population name -> bank, for every enabled population.
    """
    engine = SimulationEngine(config, device=torch.device("cpu"))
    return {pop["name"]: pop["bank"] for pop in engine.populations}


def _state_var_names(config: SensoryForgeConfig) -> Dict[str, str]:
    """Population name -> its DSL state variable's name, analog ones only.

    Read from ``PopulationConfig.dsl_config["state_vars"]``'s first key. A
    population with no ``dsl_config`` or no ``state_vars`` is left out (the
    caller falls back to ``"State"``).

    Args:
        config: The run's config snapshot.

    Returns:
        Population name -> state variable name.
    """
    names: Dict[str, str] = {}
    for pop in config.populations:
        dsl_config = getattr(pop, "dsl_config", None) or {}
        state_vars = dsl_config.get("state_vars") or {}
        if state_vars:
            names[pop.name] = next(iter(state_vars))
    return names


def _population_views(
    names_in_order: List[str],
    neuron_types: Dict[str, str],
    pop_results: Dict[str, Dict[str, torch.Tensor]],
    banks: Dict[str, ReceptiveFieldBank],
    state_var_names: Dict[str, str],
    *,
    squeeze: bool,
) -> List[PopulationView]:
    """Build one :class:`PopulationView` per name, in order.

    Args:
        names_in_order: Population names, in the order to present them.
        neuron_types: Name -> ``neuron_type``.
        pop_results: Name -> its readout dict (``spikes``/``state``, and
            optionally ``drive``/``filtered``/``voltages``).
        banks: Name -> its bank (from :func:`_banks_from_config`, or
            ``Bundle.banks``).
        state_var_names: Name -> DSL state variable name, from
            :func:`_state_var_names`.
        squeeze: Whether the tensors in ``pop_results`` still carry a leading
            batch dimension of size 1 (true for a live run's raw results,
            false for a bundle's, which are already unbatched).
    """
    views: List[PopulationView] = []
    for index, name in enumerate(names_in_order):
        raw = pop_results.get(name, {})
        maybe = (lambda t: _squeeze_batch(t)) if squeeze else (lambda t: t)
        bank = banks.get(name)
        state = maybe(raw.get("state"))
        views.append(
            PopulationView(
                name=name,
                index=index,
                neuron_type=neuron_types.get(name, "SA"),
                spikes=maybe(raw.get("spikes")),
                state=state,
                drive=maybe(raw.get("drive")),
                filtered=maybe(raw.get("filtered")),
                voltages=maybe(raw.get("voltages")),
                neuron_centers=bank.neuron_centers if bank is not None else None,
                receptor_coords=bank.receptor_coords if bank is not None else None,
                weights=bank.weights if bank is not None else None,
                state_var_name=(
                    state_var_names.get(name, "State") if state is not None else None
                ),
            )
        )
    return views


def _grid_for_stimulus(config: SensoryForgeConfig):
    """The :class:`GridConfig` the stimulus was rendered on (mirrors the CLI)."""
    if not config.grids:
        raise ValueError("config has no grids; nothing to build a canvas from")
    target = getattr(config.stimulus, "target_layer", None)
    return next(
        (g for g in config.grids if target and g.name == target), config.grids[0]
    )


def from_run_result(result: RunResult) -> ResultsView:
    """Build a :class:`ResultsView` from a live, in-memory run.

    Args:
        result: The finished run, as published by
            :meth:`~sensoryforge.gui.session.Session.set_results`.

    Returns:
        A :class:`ResultsView` over the run's stimulus and populations.
    """
    config = result.config_snapshot
    names_in_order = [pop.name for pop in config.populations if pop.enabled]
    neuron_types = {pop.name: pop.neuron_type for pop in config.populations}
    banks = _banks_from_config(config)
    state_var_names = _state_var_names(config)

    canvas: StimulusCanvas = result.canvas
    stimulus = _first_channel(_squeeze_batch(result.stimulus))
    time_ms = torch.as_tensor(result.time_ms)

    return ResultsView(
        stimulus=stimulus,
        time_ms=time_ms,
        xlim=tuple(canvas.xlim),
        ylim=tuple(canvas.ylim),
        populations=_population_views(
            names_in_order,
            neuron_types,
            result.results,
            banks,
            state_var_names,
            squeeze=True,
        ),
        label="Live results",
    )


def from_bundle(bundle: Bundle, *, label: str = "") -> ResultsView:
    """Build a :class:`ResultsView` from a loaded data bundle.

    Args:
        bundle: The bundle, as returned by
            :func:`sensoryforge.io.bundle.load_bundle`.
        label: What the screen should show as the results' source, e.g.
            ``"Viewing saved run 20260921-101500_demo"``.

    Returns:
        A :class:`ResultsView` over the bundle's stimulus and populations.

    Raises:
        ValueError: If the bundle's config has no grids (nothing to build a
            canvas from) or no stimulus was stored.
    """
    config = bundle.config
    names_in_order = list(bundle.banks.keys()) or [
        pop.name for pop in config.populations if pop.enabled
    ]
    neuron_types = {pop.name: pop.neuron_type for pop in config.populations}
    state_var_names = _state_var_names(config)

    grid_cfg = _grid_for_stimulus(config)
    canvas = stimulus_canvas(grid_cfg, device="cpu")

    if bundle.stimulus is None:
        raise ValueError("bundle has no stored stimulus to display")
    stimulus = _first_channel(bundle.stimulus)
    time_ms = (
        bundle.time_ms
        if bundle.time_ms is not None
        else torch.arange(stimulus.shape[0], dtype=torch.float32)
    )

    return ResultsView(
        stimulus=stimulus,
        time_ms=torch.as_tensor(time_ms),
        xlim=tuple(canvas.xlim),
        ylim=tuple(canvas.ylim),
        populations=_population_views(
            names_in_order,
            neuron_types,
            bundle.populations,
            bundle.banks,
            state_var_names,
            squeeze=False,
        ),
        label=label,
    )
