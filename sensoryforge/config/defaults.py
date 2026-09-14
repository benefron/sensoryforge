"""Single source of truth for filter and neuron defaults.

The GUI (``gui/tabs/spiking_tab.py``), :class:`~sensoryforge.core.simulation_engine.SimulationEngine`,
and the legacy adapter (``core/generalized_pipeline.py``) must all resolve identical
filter and neuron parameters for the same population config (see
``.claude/rules/engine-parity.md``, ledger F-026). This module is the one place
those defaults live; every caller routes through :func:`resolve_filter_params`
and :func:`resolve_neuron_params` instead of hard-coding its own copy.

This module has no Qt (or other GUI-framework) dependency so it can be
imported and tested from a plain, headless process.

Note (D-Q1, ledger F-030): the RA filter gain ``k3`` is intentionally
excluded from :data:`FILTER_DEFAULTS` — its value is undecided across
SensoryForge and pressure-simulation. Each caller keeps its own k3 default
(``RAFilterTorch``'s class default of 2.0 for the engine/legacy pipeline,
``gui/default_params.json``'s 100 for the GUI) until D-Q1 is answered.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sensoryforge.neurons.izhikevich import IZHIKEVICH_PRESETS

#: Resolver-owned SA/RA filter defaults (D-015: tau_RA = 8 ms everywhere).
#: k3 is deliberately absent -- see module docstring (D-Q1 pending).
FILTER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sa": {"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0},
    "ra": {"tau_RA": 8.0},
}

#: F-004: which Izhikevich preset each population neuron type builds by
#: default. RA/RA-I (Meissner) populations get the fast-spiking preset for
#: parity with pressure-simulation; SA/SA-I (Merkel) and SA2 keep the
#: historical regular-spiking default.
NEURON_PRESET_BY_TYPE: Dict[str, str] = {"SA": "RS", "RA": "FS", "SA2": "RS"}

#: Non-preset Izhikevich defaults (unaffected by neuron type).
_IZHIKEVICH_BASE_DEFAULTS: Dict[str, float] = {"threshold": 30.0, "noise_std": 0.0}


def resolve_filter_params(
    method: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve SA/RA filter parameters from :data:`FILTER_DEFAULTS`.

    Args:
        method: Filter method, "sa" or "ra" (case-insensitive).
        overrides: Population-specific ``filter_params`` overrides.

    Returns:
        Resolved parameter dict (defaults merged with ``overrides``). Does
        not include ``dt`` -- callers add the simulation dt separately, and
        does not include ``k3`` for "ra" -- see module docstring.

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

    For the Izhikevich model, resolves ``a``/``b``/``c``/``d`` from
    :data:`NEURON_PRESET_BY_TYPE` (unless the caller already supplies an
    explicit ``preset`` or any of ``a``/``b``/``c``/``d``, in which case that
    choice wins). The preset is expanded into concrete numeric values (not
    left as a ``"preset"`` key) so GUI widgets and engine construction see
    the same numbers either way.

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
    has_explicit = any(k in overrides for k in ("preset", "a", "b", "c", "d"))
    if has_explicit:
        preset_name = overrides.get("preset")
        if preset_name is not None:
            if preset_name not in IZHIKEVICH_PRESETS:
                raise ValueError(
                    f"Unknown Izhikevich preset {preset_name!r}; choose one of "
                    f"{sorted(IZHIKEVICH_PRESETS)}"
                )
            params.update(IZHIKEVICH_PRESETS[preset_name])
    else:
        preset_name = NEURON_PRESET_BY_TYPE.get((neuron_type or "").upper(), "RS")
        params.update(IZHIKEVICH_PRESETS[preset_name])

    params.update(overrides)
    params.pop("preset", None)
    return params
