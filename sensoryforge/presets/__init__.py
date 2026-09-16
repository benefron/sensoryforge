"""Shipped canonical-config presets (Phase 2, Wave K, K4).

A preset is a YAML fragment under this package -- data, not code -- read
through :mod:`importlib.resources` so it works from an installed wheel run
outside the repository, never a cwd-relative path (Phase 1b, F-014).

Two presets ship with SensoryForge:

- ``tactile_sa1_ra1``: the pressure-simulation recipe -- an 80x80 grid at
  0.15 mm, an SA population (regular-spiking preset, ``sa`` filter) and an
  RA population (fast-spiking preset, ``ra`` filter, k3 = 2.0), both using
  the ``template`` receptive-field builder with
  ``resolvable_distance_mm: 0.40``.
- ``tactile_stochastic_control``: the named control arm (D-019) -- identical
  except each population's receptive fields come from the ``gaussian``
  builder with ``use_distance_weights: false`` (pressure-simulation's
  uniform-random-weight "stochastic" innervation) instead of ``template``'s
  designed, analytic-Gaussian fields. There is no separate
  ``gaussian_stochastic`` builder registered under
  ``INNERVATION_REGISTRY`` -- D-019 (``docs_root/LEDGER.md``) settles the
  control arm as ``gaussian`` with ``use_distance_weights: false``, which
  ``GaussianInnervation`` already implements.

Example:
    >>> from sensoryforge.presets import list_presets, load_preset
    >>> list_presets()
    ['tactile_sa1_ra1', 'tactile_stochastic_control']
    >>> config = load_preset("tactile_sa1_ra1")
    >>> config["grids"][0]["rows"]
    80
"""

from __future__ import annotations

import importlib.resources
from typing import Any, Dict, List

import yaml

PRESET_DIR = importlib.resources.files("sensoryforge.presets")

# One-line description shown by `sensoryforge list-presets`.
_DESCRIPTIONS: Dict[str, str] = {
    "tactile_sa1_ra1": (
        "The pressure-simulation recipe: 80x80 grid at 0.15 mm, one SA and "
        "one RA population with template (designed) receptive fields at "
        "resolvable_distance_mm=0.40."
    ),
    "tactile_stochastic_control": (
        "Named control arm (D-019): identical to tactile_sa1_ra1 but with "
        "gaussian/use_distance_weights=false (stochastic, uniform-random "
        "weight) receptive fields instead of the designed template."
    ),
}


def list_presets() -> List[str]:
    """Return the names of every shipped preset (sorted, no ``.yml``).

    Returns:
        Sorted list of preset names, each loadable via :func:`load_preset`.
    """
    names = []
    for entry in PRESET_DIR.iterdir():
        if entry.name.endswith(".yml") and entry.is_file():
            names.append(entry.name[: -len(".yml")])
    return sorted(names)


def preset_description(name: str) -> str:
    """One-line description of a preset, for ``sensoryforge list-presets``.

    Args:
        name: A preset name from :func:`list_presets`.

    Returns:
        The description string, or an empty string if none is recorded.
    """
    return _DESCRIPTIONS.get(name, "")


def load_preset(name: str) -> Dict[str, Any]:
    """Load a preset's canonical config dict by name.

    Args:
        name: A preset name from :func:`list_presets` (without ``.yml``).

    Returns:
        The parsed canonical config dictionary (the same shape
        :meth:`~sensoryforge.config.schema.SensoryForgeConfig.from_dict`
        accepts).

    Raises:
        ValueError: If ``name`` is not a shipped preset, naming the
            available ones.
    """
    available = list_presets()
    if name not in available:
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")
    text = (PRESET_DIR / f"{name}.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Preset {name!r} did not parse to a dict")
    return data
