"""Synthetic probes through the engine, built from a data set's own bundle config.

A probe never reads a world: it takes the bundle's ``config.json["config"]`` (the
``SensoryForgeConfig`` the batch ran), optionally makes it quiet (all noise off) or
symmetric (receptive fields replaced by one deterministic neuron per sensor), and runs a
stimulus whose scale is the data's own largest frame value. Anything that stops a probe
from being built or run is a :class:`~sensoryforge.evaluate.checks.base.CheckSkipped`
with the reason.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sensoryforge.evaluate.checks.base import CheckContext, CheckSkipped

#: Errors that mean "this configuration cannot be built or run here".
ENGINE_ERRORS = (OSError, ValueError, KeyError, RuntimeError, TypeError, ImportError)

#: The probe's stimulus is raised from the data's peak by powers of two until the
#: populations count at least this many spikes or events, or up to ``MAX_DOUBLINGS``.
MIN_EVENTS = 20
MAX_DOUBLINGS = 10

#: Readout keys the engine returns for a population.
READOUT_KEYS = ("spikes", "events", "state")


@dataclass
class Probe:
    """A built engine and the geometry its stimuli need.

    Attributes:
        engine: The ``SimulationEngine``.
        config: The configuration it was built from.
        rows: Sensor rows of grid 0.
        cols: Sensor columns of grid 0.
        channels: Number of channels of grid 0 (stimuli are ``[1, T, C, H, W]`` when
            more than one).
        n_frames: Frames per probe stimulus.
    """

    engine: Any
    config: Any
    rows: int
    cols: int
    channels: int
    n_frames: int


def base_config(ctx: CheckContext) -> Any:
    """The data set's ``SensoryForgeConfig`` (from the first bundle that ran).

    Raises:
        CheckSkipped: If the configuration cannot be read.
    """
    from sensoryforge.config.schema import SensoryForgeConfig

    raw = ctx.bundle_config().get("config")
    if not isinstance(raw, dict):
        raise CheckSkipped("the first bundle's config.json holds no 'config' block")
    try:
        return SensoryForgeConfig.from_dict(raw)
    except ENGINE_ERRORS as exc:
        raise CheckSkipped(f"the bundle's config cannot be loaded ({exc})") from exc


def quiet(config: Any) -> Any:
    """A copy with every noise source off (sensor, membrane and receptor)."""
    cfg = copy.deepcopy(config)
    for pop in cfg.populations:
        pop.sensor_noise_std = 0.0
        pop.membrane_noise_std = 0.0
        pop.noise_std = 0.0
    cfg.simulation.receptor_noise_std = 0.0
    return cfg


def _grid(config: Any) -> Tuple[int, int, int]:
    """``(rows, cols, channels)`` of the single regular grid probes need."""
    if len(config.grids) != 1:
        raise CheckSkipped(f"probes need one grid, the config has {len(config.grids)}")
    grid = config.grids[0]
    if grid.arrangement != "grid" or grid.rows is None or grid.cols is None:
        raise CheckSkipped(
            f"probes need a regular grid, grid {grid.name!r} is {grid.arrangement!r}"
        )
    return int(grid.rows), int(grid.cols), max(1, len(grid.channels or ["value"]))


def symmetric(config: Any) -> Any:
    """A copy whose populations each read the sensors one for one.

    Every receptive field becomes the ``template`` builder with ``k = 1`` and unit
    weights on a lattice at the sensors' own pitch, so neuron ``n`` is sensor ``n`` and
    a mirrored stimulus has a mirrored response. The neuron, filter and gain settings
    stay as the data set had them.

    Raises:
        CheckSkipped: If the sensors are not a regular grid, or a population's
            receptive field cannot be replaced.
    """
    from sensoryforge.config.schema import RFBuilderConfig

    cfg = copy.deepcopy(config)
    _grid(cfg)
    spacing = float(cfg.grids[0].spacing)
    params = {
        "sigma_mm": spacing / 2.0,
        "pitch_mm": spacing,
        "k": 1,
        "normalize": "none",
    }
    for pop in cfg.populations:
        if getattr(pop, "target_layers", None):
            raise CheckSkipped(f"population {pop.name!r} reads a composite grid")
        if pop.inputs:
            for inp in pop.inputs:
                inp.rf = RFBuilderConfig(method="template", params=dict(params))
        else:
            pop.innervation_method = "template"
            pop.innervation_params = dict(params)
    return cfg


def build(ctx: CheckContext, config: Any) -> Probe:
    """Build the engine for a configuration.

    Args:
        ctx: The context (its reference bundle gives the number of frames).
        config: A ``SensoryForgeConfig``.

    Returns:
        The probe.

    Raises:
        CheckSkipped: If the engine cannot be built (the message names the error).
    """
    from sensoryforge.core.simulation_engine import SimulationEngine

    rows, cols, channels = _grid(config)
    info = ctx.reader.peek(ctx.ran_entries()[0])
    try:
        with warnings.catch_warnings():
            # the symmetric probe's template builder derives its own lattice
            warnings.filterwarnings(
                "ignore", message=".*derives its own neuron lattice.*"
            )
            engine = SimulationEngine(config)
    except ENGINE_ERRORS as exc:
        raise CheckSkipped(
            "the engine cannot be built from the bundle's config "
            f"({type(exc).__name__}: "
            f"{exc})"
        ) from exc
    return Probe(engine, config, rows, cols, channels, min(int(info.n_frames), 400))


def pattern(probe: Probe, amplitude: float) -> np.ndarray:
    """An asymmetric stimulus: a 2 x 2 block, then a weaker single sensor, then rest.

    Args:
        probe: The probe (its grid shape and frame count).
        amplitude: The block's value, in the frame's own unit.

    Returns:
        ``[T, rows, cols]`` float32, or ``[T, C, rows, cols]`` when the grid has
        channels (every channel carries the same pattern).
    """
    t, r, c = probe.n_frames, probe.rows, probe.cols
    frames = np.zeros((t, r, c), dtype=np.float32)
    i0, j0 = max(0, r // 4), max(0, c // 4)
    frames[t // 10 : 9 * t // 10, i0 : i0 + 2, j0 : j0 + 2] = amplitude
    frames[t // 3 : 2 * t // 3, min(r - 1, 3 * r // 4), min(c - 1, 2 * c // 3)] = (
        0.5 * amplitude
    )
    if probe.channels > 1:
        frames = np.repeat(frames[:, None], probe.channels, axis=1)
    return frames


def run(
    probe: Probe, frames: np.ndarray, seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the engine on ``frames`` and return each population's arrays.

    Args:
        probe: The probe.
        frames: The stimulus from :func:`pattern` (``[T, ...]``, no batch axis).
        seed: The run seed.

    Returns:
        ``{population: {readout, "drive", "filtered"}}`` with each array
        ``[T, N]`` (batch axis removed; the readout is under its own key).

    Raises:
        CheckSkipped: If the run raises one of :data:`ENGINE_ERRORS`.
    """
    import torch

    stim = torch.from_numpy(np.ascontiguousarray(frames))[None]
    try:
        with torch.no_grad():
            out = probe.engine.run(stim, return_intermediates=True, seed=seed)
    except ENGINE_ERRORS as exc:
        raise CheckSkipped(
            f"the engine failed on a probe ({type(exc).__name__}: {exc})"
        )
    return {
        name: {
            k: v[0].detach().cpu().numpy() for k, v in arrays.items() if k != "voltages"
        }
        for name, arrays in out.items()
    }


def readout_of(arrays: Dict[str, np.ndarray]) -> Optional[str]:
    """The readout key (``spikes``, ``events``, ``state``) of a population's arrays."""
    return next((k for k in READOUT_KEYS if k in arrays), None)


def events(arrays: Dict[str, np.ndarray]) -> float:
    """Total absolute spike or event count of a population (0 for an analog readout)."""
    key = readout_of(arrays)
    if key in ("spikes", "events"):
        return float(np.abs(arrays[key]).sum())
    return 0.0


def fold(array: np.ndarray, probe: Probe) -> Optional[np.ndarray]:
    """A population array ``[T, N]`` as ``[T, k, rows, cols]``, or ``None``.

    ``k`` is how many sensor lattices the neuron axis holds (one per concatenated
    input); ``None`` when ``N`` is not a multiple of ``rows * cols``.
    """
    t, n = array.shape
    m = probe.rows * probe.cols
    if n == 0 or n % m:
        return None
    return array.reshape(t, n // m, probe.rows, probe.cols)


def populations(probe: Probe) -> List[str]:
    """The population names, in the engine's order."""
    return [p["name"] for p in probe.engine.populations]


def amplitude(ctx: CheckContext, probe: Probe) -> float:
    """The probe pattern's amplitude: the data's peak, doubled until they respond.

    A stimulus at the data's own peak can evoke almost nothing on a coarse probe (a 2 x
    2 block for a few frames), which would leave a symmetry or determinism check
    comparing empty arrays. The amplitude therefore starts at the peak and doubles, up
    to ``2 ** MAX_DOUBLINGS`` times, until the populations together count
    :data:`MIN_EVENTS` spikes or events. This is a search for a usable scale, not a
    claim about the data; the amplitude used is reported in the check's numbers.

    Args:
        ctx: The context (its seed and the data's peak).
        probe: The probe.

    Returns:
        The amplitude in the frame's unit. If none reaches the target, the largest
        tried (the check then sees few events and says so).
    """
    amp = ctx.peak()
    for _ in range(MAX_DOUBLINGS + 1):
        out = run(probe, pattern(probe, amp), ctx.seed)
        if sum(events(v) for v in out.values()) >= MIN_EVENTS:
            return amp
        amp *= 2.0
    return amp / 2.0
