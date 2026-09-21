"""Single source of truth for filter and neuron defaults.

The GUI (``gui/tabs/spiking_tab.py``), :class:`~sensoryforge.core.simulation_engine.SimulationEngine`,
and the legacy adapter (``core/generalized_pipeline.py``) must all resolve identical
filter and neuron parameters for the same population config (see
``.claude/rules/engine-parity.md``, ledger F-026). This module is the one place
those defaults live; every caller routes through :func:`resolve_filter_params`
and :func:`resolve_neuron_params` instead of hard-coding its own copy.

This module has no Qt (or other GUI-framework) dependency so it can be
imported and tested from a plain, headless process.

D-Q1 (ledger F-030) is decided: the RA filter gain ``k3`` is 2.0,
matching both repos' ``RAFilterTorch`` class default, pressure-simulation's
``config/pipeline_config.yml`` and its decoder gain. It is resolver-owned
like every other filter parameter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sensoryforge.neurons.izhikevich import IZHIKEVICH_PRESETS

#: Run length when neither the caller nor the config names one, in ms.
DEFAULT_DURATION_MS: float = 1000.0


def resolve_duration_ms(
    configured: Optional[float], override: Optional[float] = None
) -> float:
    """The run length in ms: the override, else the config's, else the default.

    One rule for ``sensoryforge run`` (``--duration`` is the override), the
    GUI run bar, the Stimulus preview and the Batch screen, so the length a
    config records is the length every one of them runs.

    Args:
        configured: ``SimulationConfig.duration_ms`` (``None`` when unset).
        override: An explicit request (``--duration``), or ``None``.

    Returns:
        The duration in ms.

    Raises:
        ValueError: If the chosen value is not positive.
    """
    for value in (override, configured):
        if value is not None:
            duration = float(value)
            if duration <= 0:
                raise ValueError(f"duration must be positive, got {duration} ms")
            return duration
    return DEFAULT_DURATION_MS


#: Resolver-owned SA/RA filter defaults (D-015: tau_RA = 8 ms everywhere;
#: D-Q1: RA gain k3 = 2.0 everywhere).
FILTER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sa": {"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0},
    "ra": {"tau_RA": 8.0, "k3": 2.0},
}

#: F-004: which Izhikevich preset each population neuron type builds by
#: default. RA/RA-I (Meissner) populations get the fast-spiking preset for
#: parity with pressure-simulation; SA/SA-I (Merkel) and SA2 keep the
#: historical regular-spiking default.
NEURON_PRESET_BY_TYPE: Dict[str, str] = {"SA": "RS", "RA": "FS", "SA2": "RS"}

#: Non-preset Izhikevich defaults (unaffected by neuron type).
_IZHIKEVICH_BASE_DEFAULTS: Dict[str, float] = {"threshold": 30.0, "noise_std": 0.0}

#: Neuron integration step (ms, F-008): pressure-simulation's hard-coded
#: native Izhikevich step, and SimulationConfig.integrate_dt_ms's default.
#: Shared by SimulationConfig, SimulationEngine, and the GUI (both
#: spiking_tab.py's neuron construction and stimulus_tab.py's time-step
#: spinbox single-step, F-042) so a record step can only be set to a whole
#: multiple of this value from the GUI.
DEFAULT_INTEGRATE_DT_MS: float = 0.05


def resolve_filter_params(
    method: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve SA/RA filter parameters from :data:`FILTER_DEFAULTS`.

    Args:
        method: Filter method, "sa" or "ra" (case-insensitive).
        overrides: Population-specific ``filter_params`` overrides.

    Returns:
        Resolved parameter dict (defaults merged with ``overrides``). Does
        not include ``dt`` -- callers add the simulation dt separately.

    Raises:
        ValueError: If ``method`` is not "sa" or "ra".
    """
    key = method.lower()
    if key not in FILTER_DEFAULTS:
        raise ValueError(
            f"Unknown filter method {method!r}; expected one of {sorted(FILTER_DEFAULTS)}"
        )
    params = dict(FILTER_DEFAULTS[key])
    params.update(overrides or {})
    return params


def resolve_neuron_params(
    model_name: str, neuron_type: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve neuron model parameters, applying the F-004 RA->FS preset.

    For the Izhikevich model, always starts from a preset -- the explicit
    ``preset`` override if given, otherwise :data:`NEURON_PRESET_BY_TYPE` for
    ``neuron_type`` -- then applies any ``a``/``b``/``c``/``d`` overrides on
    top of it (matching ``IzhikevichNeuronTorch(preset=..., d=...)``
    semantics). The preset is expanded into concrete numeric values (not
    left as a ``"preset"`` key) so GUI widgets and engine construction see
    the same numbers either way. The result always contains ``a``, ``b``,
    ``c``, ``d`` and ``threshold`` -- overriding a single parameter (e.g.
    ``{"d": 4.0}`` on an RA population) never drops the rest of the preset
    (F-031: it used to, silently reverting the other three to the RS/type
    default and raising ``KeyError`` in the legacy adapter).

    Non-Izhikevich models pass ``overrides`` straight through: their
    defaults live only in the class constructor signature, which is already
    a single source of truth (verified to match ``gui/default_params.json``
    for AdEx and MQIF).

    Args:
        model_name: Neuron model name (e.g. "Izhikevich", "AdEx"; case-insensitive).
        neuron_type: Population neuron type (e.g. "SA", "RA", "SA2").
        overrides: Population-specific ``model_params`` overrides.

    Returns:
        Resolved parameter dict merged with ``overrides``.
    """
    overrides = dict(overrides or {})
    if model_name.lower() != "izhikevich":
        return overrides

    params = dict(_IZHIKEVICH_BASE_DEFAULTS)
    preset_name = overrides.get("preset")
    if preset_name is not None:
        if preset_name not in IZHIKEVICH_PRESETS:
            raise ValueError(
                f"Unknown Izhikevich preset {preset_name!r}; choose one of "
                f"{sorted(IZHIKEVICH_PRESETS)}"
            )
    else:
        preset_name = NEURON_PRESET_BY_TYPE.get((neuron_type or "").upper(), "RS")
    params.update(IZHIKEVICH_PRESETS[preset_name])

    params.update(overrides)
    params.pop("preset", None)
    return params
