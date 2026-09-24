"""Single source of truth for filter and neuron defaults.

The GUI's forms, :class:`~sensoryforge.core.simulation_engine.SimulationEngine`,
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

from sensoryforge.neurons.adex import ADEX_PRESETS
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
#: historical regular-spiking default. This table is Izhikevich-specific
#: (kept under its historical name -- other code and tests import it); the
#: per-model tables live in :data:`_PRESET_TABLES_BY_MODEL`.
NEURON_PRESET_BY_TYPE: Dict[str, str] = {"SA": "RS", "RA": "FS", "SA2": "RS"}

#: Which AdEx preset (:data:`sensoryforge.neurons.adex.ADEX_PRESETS`) each
#: population neuron type builds by default. RA/RA-I (Meissner) populations
#: get the phasic/adapting regime; SA/SA-I (Merkel) and SA2 get the tonic
#: regime -- the same SA/RA split as :data:`NEURON_PRESET_BY_TYPE`, just
#: against AdEx's own two named presets.
ADEX_PRESET_BY_TYPE: Dict[str, str] = {
    "SA": "SA1_tonic",
    "RA": "RA1_phasic",
    "SA2": "SA1_tonic",
}

#: Per-model preset table + neuron-type->preset map + fallback preset name +
#: human-readable label (for error messages), keyed by lowercased
#: ``neuron_model``. Shared by :func:`resolve_neuron_params` for every model
#: that resolves via a named preset (currently Izhikevich and AdEx); a model
#: absent from this table passes ``overrides`` straight through unchanged
#: (see :func:`resolve_neuron_params`).
_PRESET_TABLES_BY_MODEL: Dict[str, tuple] = {
    "izhikevich": (IZHIKEVICH_PRESETS, NEURON_PRESET_BY_TYPE, "RS", "Izhikevich"),
    "adex": (ADEX_PRESETS, ADEX_PRESET_BY_TYPE, "SA1_tonic", "AdEx"),
}

#: Non-preset Izhikevich defaults (unaffected by neuron type).
_IZHIKEVICH_BASE_DEFAULTS: Dict[str, float] = {"threshold": 30.0, "noise_std": 0.0}

#: Neuron integration step (ms, F-008): pressure-simulation's hard-coded
#: native Izhikevich step, and SimulationConfig.integrate_dt_ms's default.
#: Shared by SimulationConfig and SimulationEngine; the GUI rejects a record
#: step that is not a whole multiple of it (validation, F-042).
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
            f"Unknown filter method {method!r}; expected one of "
            f"{sorted(FILTER_DEFAULTS)}"
        )
    params = dict(FILTER_DEFAULTS[key])
    params.update(overrides or {})
    return params


def _resolve_preset_params(
    model_label: str,
    presets: Dict[str, Dict[str, Any]],
    preset_by_type: Dict[str, str],
    default_preset: str,
    neuron_type: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve one model's preset table against a neuron type + overrides.

    Shared by every preset-driven model in :func:`resolve_neuron_params`
    (Izhikevich, AdEx): starts from a preset -- the explicit ``preset``
    override if given, otherwise ``preset_by_type`` for ``neuron_type``,
    otherwise ``default_preset`` -- then applies ``overrides`` on top of the
    preset's expanded numeric values, so overriding one parameter (F-031)
    never drops the rest of the preset.

    Args:
        model_label: Human-readable model name for the error message (e.g.
            "Izhikevich", "AdEx").
        presets: The model's named preset table (e.g. ``IZHIKEVICH_PRESETS``).
        preset_by_type: Neuron-type -> preset-name map (e.g.
            ``NEURON_PRESET_BY_TYPE``).
        default_preset: Preset name used when ``neuron_type`` is unknown.
        neuron_type: Population neuron type (e.g. "SA", "RA", "SA2").
        overrides: Population-specific ``model_params`` overrides (may
            include an explicit ``"preset"`` key).

    Returns:
        The preset's parameters (expanded into concrete numbers, not a
        ``"preset"`` key) merged with ``overrides``.

    Raises:
        ValueError: If an explicit ``overrides["preset"]`` names an unknown
            preset.
    """
    preset_name = overrides.get("preset")
    if preset_name is not None:
        if preset_name not in presets:
            raise ValueError(
                f"Unknown {model_label} preset {preset_name!r}; choose one of "
                f"{sorted(presets)}"
            )
    else:
        preset_name = preset_by_type.get((neuron_type or "").upper(), default_preset)
    params = dict(presets[preset_name])
    params.update(overrides)
    params.pop("preset", None)
    return params


def resolve_neuron_params(
    model_name: str, neuron_type: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve neuron model parameters, applying each model's type preset.

    For Izhikevich (F-004: RA->FS) and AdEx (RA->``RA1_phasic``, SA/SA2->
    ``SA1_tonic``), always starts from a preset -- the explicit ``preset``
    override if given, otherwise the model's neuron-type->preset table for
    ``neuron_type`` -- then applies any parameter overrides on top of it
    (matching ``IzhikevichNeuronTorch(preset=..., d=...)`` /
    ``AdExNeuronTorch(preset=..., tau_w=...)`` semantics). The preset is
    expanded into concrete numeric values (not left as a ``"preset"`` key)
    so GUI widgets and engine construction see the same numbers either way.
    Overriding a single parameter (e.g. ``{"d": 4.0}`` on an RA Izhikevich
    population, or ``{"tau_w": 300.0}`` on an SA AdEx population) never
    drops the rest of the preset (F-031: for Izhikevich it used to,
    silently reverting the other three to the RS/type default and raising
    ``KeyError`` in the legacy adapter).

    Models not in :data:`_PRESET_TABLES_BY_MODEL` (e.g. MQIF, FA, SA, DSL)
    pass ``overrides`` straight through: their defaults live only in the
    class constructor signature, which is already a single source of truth.

    Args:
        model_name: Neuron model name (e.g. "Izhikevich", "AdEx"; case-insensitive).
        neuron_type: Population neuron type (e.g. "SA", "RA", "SA2").
        overrides: Population-specific ``model_params`` overrides.

    Returns:
        Resolved parameter dict merged with ``overrides``.

    Raises:
        ValueError: If an explicit ``overrides["preset"]`` names an unknown
            preset for a preset-driven model.
    """
    overrides = dict(overrides or {})
    key = model_name.lower()
    table = _PRESET_TABLES_BY_MODEL.get(key)
    if table is None:
        return overrides

    presets, preset_by_type, default_preset, model_label = table
    resolved = _resolve_preset_params(
        model_label,
        presets,
        preset_by_type,
        default_preset,
        neuron_type,
        overrides,
    )
    if key == "izhikevich":
        params = dict(_IZHIKEVICH_BASE_DEFAULTS)
        params.update(resolved)
        return params
    return resolved
