"""The SensoryForge data bundle: the contract with pressure-simulation and
learning pipelines (Phase 2, Wave J, F-013, F-011).

A bundle is a directory:

```
bundle_dir/
    config.json               # schema_version "2.0.0", kind "sensoryforge_bundle"
    population_01_<NAME>.pt   # ReceptiveFieldBank.save() output + grid_shape
    population_02_<NAME>.pt
    stimuli/
        stimulus.json
    neuron_modules/
        sensoryforge.json     # schema_version "1.0.0", kind "neuron_module" (J6)
    data.h5                   # /stimulus/frames, /time_ms, /populations/<name>/...
```

``config.json`` is a superset of pressure-simulation's ``1.0.0`` "mechanoreceptor
bundle" format (``kind: "mechanoreceptor_bundle"``): its viewer
(``GUIs/ebkf_viewer.py``) reads ``grid`` and ``populations[*].tensors`` unchanged.
See ``docs/development/handover/phase2_tasks.md`` section 2 for the exact fields
that loader reads, and ``docs/user_guide/bundles.md`` for the user-facing layout.

Coordinates and receptor ordering follow the rest of SensoryForge: ``(x, y)`` in
mm, receptor ``k = i * cols + j`` for a ``[rows, cols]`` grid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

import sensoryforge
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.rf_bank import ReceptiveFieldBank

SCHEMA_VERSION = "2.0.0"
BUNDLE_KIND = "sensoryforge_bundle"
_SCHEMA_MAJOR = SCHEMA_VERSION.split(".")[0]


def _safe_name(name: str) -> str:
    """Filesystem-safe population name: ``"SA #6"`` -> ``"SA_6"``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "population"


def _squeeze_batch(t: torch.Tensor, *, what: str) -> torch.Tensor:
    """Drop a leading batch dim of size 1; raise naming both shapes otherwise.

    Bundles hold one run (batch=1); ``write_bundle`` accepts the batched
    tensors :meth:`SimulationEngine.run` returns and stores the unbatched
    arrays.
    """
    if t.ndim >= 1 and t.shape[0] == 1:
        return t[0]
    raise ValueError(
        f"write_bundle: expected {what} with a leading batch dim of size 1, "
        f"got shape {list(t.shape)}"
    )


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised via ImportError test
        raise ImportError(
            "h5py is required to read/write SensoryForge data bundles. "
            "Install with: pip install -e '.[hdf5]' (or: pip install h5py)"
        ) from exc
    return h5py


@dataclass
class Bundle:
    """A loaded data bundle (:func:`load_bundle` return value).

    Attributes:
        config: The run's canonical config.
        banks: Population name -> :class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank`.
        stimulus: Stimulus frames ``[T, H, W]`` or ``[T, C, H, W]`` (float32),
            or ``None`` if the bundle has no stimulus dataset.
        time_ms: ``[T]`` time axis in ms, or ``None``.
        populations: Population name -> dict of tensors (``drive``,
            ``filtered``, and ``spikes`` or ``state``), each ``[T, N]``.
        meta: ``dt_ms``, ``integrate_dt_ms``, ``seed``, ``sensoryforge_version``,
            ``config_yaml``, ``provenance`` (parsed from ``provenance_json``),
            and ``config_json`` (the raw ``config.json`` dict).
    """

    config: SensoryForgeConfig
    banks: Dict[str, ReceptiveFieldBank]
    stimulus: Optional[torch.Tensor]
    time_ms: Optional[torch.Tensor]
    populations: Dict[str, Dict[str, torch.Tensor]]
    meta: Dict[str, Any] = field(default_factory=dict)


def _pop_grid_cfg(config: SensoryForgeConfig, pop_cfg: Any):
    """The :class:`GridConfig` a population targets (first grid if unset)."""
    target = pop_cfg.target_grid
    if target is not None:
        for g in config.grids:
            if g.name == target:
                return g
    if config.grids:
        return config.grids[0]
    return None


def write_bundle(
    bundle_dir: Union[str, Path],
    config: SensoryForgeConfig,
    engine: Any,
    results: Dict[str, Dict[str, torch.Tensor]],
    stimulus: torch.Tensor,
    *,
    stimulus_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Write a run as a data bundle.

    Args:
        bundle_dir: Destination directory. Created if missing.
        config: The run's canonical config (written whole into ``config.json``
            and, as YAML, into ``data.h5``'s ``/meta`` group).
        engine: The :class:`~sensoryforge.core.simulation_engine.SimulationEngine`
            that produced *results* -- duck-typed: needs ``.populations``, a
            list of ``{"name", "config", "bank"}`` dicts (as the engine
            builds them).
        results: :meth:`SimulationEngine.run` output, keyed by population
            name; each value must include ``"drive"`` and ``"filtered"``
            (i.e. the run that produced it used ``return_intermediates=True``)
            and either ``"spikes"`` or ``"state"``.
        stimulus: Stimulus tensor, ``[1, T, H, W]``/``[1, T, C, H, W]`` (batch
            of 1) or ``[T, H, W]``/``[T, C, H, W]`` (no batch dim).
        stimulus_config: The stimulus's own config dict, written to
            ``stimuli/stimulus.json`` (``{}`` if not given).
        seed: The run's seed, recorded as an HDF5 root attribute.
        overwrite: If ``False`` (default) and *bundle_dir* already exists and
            is non-empty, raise ``FileExistsError``.

    Returns:
        *bundle_dir* as a :class:`~pathlib.Path`.

    Raises:
        FileExistsError: If *bundle_dir* exists, is non-empty, and
            ``overwrite`` is ``False``.
        ValueError: If a population in *results* is missing ``"drive"``/
            ``"filtered"``, or a tensor has an unexpected batch dimension.
        ImportError: If ``h5py`` is not installed.
    """
    h5py = _require_h5py()

    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{bundle_dir} already exists and is not empty; pass overwrite=True"
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "stimuli").mkdir(exist_ok=True)

    pop_by_name = {p["name"]: p for p in engine.populations}

    # ------------------------------------------------------------------ #
    # population_NN_<NAME>.pt -- bank + grid_shape
    # ------------------------------------------------------------------ #
    pop_entries = []
    for idx, name in enumerate(results.keys(), start=1):
        pop = pop_by_name.get(name)
        if pop is None:
            raise ValueError(
                f"write_bundle: results has population {name!r} that "
                f"engine.populations does not (has: {sorted(pop_by_name)})"
            )
        pop_cfg = pop["config"]
        bank: ReceptiveFieldBank = pop["bank"]
        grid_cfg = _pop_grid_cfg(config, pop_cfg)
        if grid_cfg is not None and grid_cfg.rows is not None:
            rows = grid_cfg.rows
        else:
            rows = 1
        if grid_cfg is not None and grid_cfg.cols is not None:
            cols = grid_cfg.cols
        else:
            cols = bank.num_receptors
        tensor_name = f"population_{idx:02d}_{_safe_name(name)}.pt"
        torch.save(
            {
                "innervation_weights": bank.weights.detach().cpu().clone(),
                "neuron_centers": bank.neuron_centers.detach().cpu().clone(),
                "receptor_coords": bank.receptor_coords.detach().cpu().clone(),
                "provenance": dict(bank.provenance),
                "grid_shape": [int(rows), int(cols)],
            },
            bundle_dir / tensor_name,
        )
        weight_range = pop_cfg.weight_range or [0.05, 1.0]
        pop_entries.append(
            {
                "name": name,
                "neuron_type": pop_cfg.neuron_type,
                "color": list(pop_cfg.color),
                "parameters": {
                    "neurons_per_row": pop_cfg.neurons_per_row,
                    "connections_per_neuron": pop_cfg.connections_per_neuron,
                    "sigma_d_mm": pop_cfg.sigma_d_mm,
                    "weight_min": weight_range[0],
                    "weight_max": weight_range[1],
                    "seed": pop_cfg.seed,
                    "edge_offset": pop_cfg.edge_offset,
                },
                "tensors": tensor_name,
                "visible": pop_cfg.visible,
            }
        )

    # ------------------------------------------------------------------ #
    # config.json
    # ------------------------------------------------------------------ #
    primary_grid = config.grids[0] if config.grids else None
    grid_section = (
        {
            "rows": primary_grid.rows,
            "cols": primary_grid.cols,
            "spacing_mm": primary_grid.spacing,
            "center_mm": [primary_grid.center_x, primary_grid.center_y],
            "device": str(engine.device),
        }
        if primary_grid is not None
        else {}
    )
    config_json = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_created": datetime.now(timezone.utc).isoformat(),
        "sensoryforge_version": sensoryforge.__version__,
        "grid": grid_section,
        "populations": pop_entries,
        "config": config.to_dict(),
    }
    with open(bundle_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    # ------------------------------------------------------------------ #
    # stimuli/stimulus.json
    # ------------------------------------------------------------------ #
    with open(bundle_dir / "stimuli" / "stimulus.json", "w") as f:
        json.dump(stimulus_config or {}, f, indent=2)

    # ------------------------------------------------------------------ #
    # neuron_modules/sensoryforge.json (J6)
    #
    # Without this, pressure-simulation's viewer (GUIs/ebkf_viewer.py
    # _on_load_bundle) loads config.json and the population tensors fine,
    # but its "neuron module" combo box stays empty (it globs
    # neuron_modules/*.json) and its Run button never enables -- the
    # bundle displays but can never be encoded. _on_run only reads
    # "enabled", "name", "neuron_type", "filter_method", "noise_std",
    # "model_params" and "filter_params" from each population_configs
    # entry (it never reads "model" -- kept here as metadata only -- and
    # always overrides "input_gain" with its own spinboxes); it matches
    # entries to populations by exact "name" against config.json's
    # populations[*].name, so this uses the *raw* population name (not
    # the filesystem-safe one the .pt filenames use).
    # ------------------------------------------------------------------ #
    (bundle_dir / "neuron_modules").mkdir(exist_ok=True)
    neuron_module = {
        "schema_version": "1.0.0",
        "kind": "neuron_module",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stimulus": "stimulus.json",
        "device": str(engine.device),
        "population_configs": [
            {
                "name": name,
                "neuron_type": pop_by_name[name]["config"].neuron_type,
                "model": pop_by_name[name]["config"].neuron_model,
                "filter_method": pop_by_name[name]["config"].filter_method,
                "enabled": pop_by_name[name]["config"].enabled,
                "input_gain": pop_by_name[name]["config"].input_gain,
                "noise_std": pop_by_name[name]["config"].noise_std,
                "model_params": dict(pop_by_name[name]["config"].model_params or {}),
                "filter_params": dict(pop_by_name[name]["config"].filter_params or {}),
                "selected_neuron": 0,
            }
            for name in results
        ],
    }
    with open(bundle_dir / "neuron_modules" / "sensoryforge.json", "w") as f:
        json.dump(neuron_module, f, indent=2)

    # ------------------------------------------------------------------ #
    # data.h5
    # ------------------------------------------------------------------ #
    stim = stimulus
    if stim.ndim in (4, 5):
        stim = _squeeze_batch(stim, what="stimulus")
    elif stim.ndim not in (3, 4):
        raise ValueError(
            "write_bundle: stimulus must be [T, H, W], [T, C, H, W] (or the "
            f"same with a leading batch dim), got shape {list(stim.shape)}"
        )
    stim = stim.detach().cpu().to(torch.float32).numpy()
    T = stim.shape[0]

    dt_ms = config.simulation.dt_ms
    time_ms = (torch.arange(T, dtype=torch.float32) * dt_ms).numpy()

    with h5py.File(bundle_dir / "data.h5", "w") as f:
        f.attrs["dt_ms"] = dt_ms
        f.attrs["integrate_dt_ms"] = config.simulation.integrate_dt_ms
        f.attrs["seed"] = -1 if seed is None else int(seed)
        f.attrs["sensoryforge_version"] = sensoryforge.__version__

        stim_grp = f.create_group("stimulus")
        stim_grp.create_dataset(
            "frames", data=stim, compression="gzip", compression_opts=4
        )
        f.create_dataset("time_ms", data=time_ms)

        pops_grp = f.create_group("populations")
        for name, pop_results in results.items():
            if "drive" not in pop_results or "filtered" not in pop_results:
                raise ValueError(
                    f"write_bundle: population {name!r} results are missing "
                    "'drive'/'filtered' -- call engine.run(..., "
                    "return_intermediates=True)"
                )
            pop_grp = pops_grp.create_group(name)
            drive = _squeeze_batch(pop_results["drive"], what=f"{name} drive")
            filtered = _squeeze_batch(pop_results["filtered"], what=f"{name} filtered")
            pop_grp.create_dataset(
                "drive",
                data=drive.detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )
            pop_grp.create_dataset(
                "filtered",
                data=filtered.detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )
            if "spikes" in pop_results:
                spikes = _squeeze_batch(pop_results["spikes"], what=f"{name} spikes")
                pop_grp.create_dataset(
                    "spikes",
                    data=spikes.detach().cpu().to(torch.int16).numpy(),
                    compression="gzip",
                    compression_opts=4,
                )
            elif "state" in pop_results:
                state = _squeeze_batch(pop_results["state"], what=f"{name} state")
                pop_grp.create_dataset(
                    "state",
                    data=state.detach().cpu().numpy(),
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                raise ValueError(
                    f"write_bundle: population {name!r} results have "
                    "neither 'spikes' nor 'state'"
                )

        provenance = {
            name: dict(pop_by_name[name]["bank"].provenance) for name in results
        }
        meta_grp = f.create_group("meta")
        meta_grp.attrs["config_yaml"] = config.to_yaml()
        meta_grp.attrs["provenance_json"] = json.dumps(provenance)

    return bundle_dir


def load_bundle(bundle_dir: Union[str, Path]) -> Bundle:
    """Read a bundle written by :func:`write_bundle`.

    Args:
        bundle_dir: The bundle directory (containing ``config.json``).

    Returns:
        A :class:`Bundle`.

    Raises:
        ValueError: If ``config.json`` has no ``schema_version``, or its
            major version is not ``2``.
        FileNotFoundError: If ``config.json`` is missing.
        ImportError: If ``h5py`` is not installed and ``data.h5`` is present.
    """
    bundle_dir = Path(bundle_dir)
    config_path = bundle_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path}: no config.json in bundle")
    with open(config_path, "r") as f:
        config_json = json.load(f)

    schema_version = config_json.get("schema_version")
    if not schema_version:
        raise ValueError(f"{config_path}: missing 'schema_version'")
    major = str(schema_version).split(".")[0]
    if major != _SCHEMA_MAJOR:
        raise ValueError(
            f"{config_path}: schema_version {schema_version!r} is not "
            f"compatible with load_bundle (expects major version "
            f"{_SCHEMA_MAJOR}.x)"
        )

    config = SensoryForgeConfig.from_dict(config_json.get("config", {}))

    banks: Dict[str, ReceptiveFieldBank] = {}
    for pop_entry in config_json.get("populations", []):
        name = pop_entry["name"]
        tensor_path = bundle_dir / pop_entry["tensors"]
        banks[name] = ReceptiveFieldBank.load(tensor_path)

    stimulus = None
    time_ms = None
    populations: Dict[str, Dict[str, torch.Tensor]] = {}
    meta: Dict[str, Any] = {
        "config_json": config_json,
    }

    h5_path = bundle_dir / "data.h5"
    if h5_path.exists():
        h5py = _require_h5py()
        with h5py.File(h5_path, "r") as f:
            if "stimulus" in f and "frames" in f["stimulus"]:
                stimulus = torch.from_numpy(f["stimulus"]["frames"][()])
            if "time_ms" in f:
                time_ms = torch.from_numpy(f["time_ms"][()])
            for name, pop_grp in f.get("populations", {}).items():
                populations[name] = {
                    key: torch.from_numpy(pop_grp[key][()]) for key in pop_grp.keys()
                }
            meta["dt_ms"] = float(f.attrs.get("dt_ms", config.simulation.dt_ms))
            meta["integrate_dt_ms"] = float(
                f.attrs.get("integrate_dt_ms", config.simulation.integrate_dt_ms)
            )
            raw_seed = f.attrs.get("seed", -1)
            meta["seed"] = None if int(raw_seed) < 0 else int(raw_seed)
            meta["sensoryforge_version"] = f.attrs.get("sensoryforge_version")
            if "meta" in f:
                meta["config_yaml"] = f["meta"].attrs.get("config_yaml")
                prov_json = f["meta"].attrs.get("provenance_json")
                meta["provenance"] = json.loads(prov_json) if prov_json else {}

    return Bundle(
        config=config,
        banks=banks,
        stimulus=stimulus,
        time_ms=time_ms,
        populations=populations,
        meta=meta,
    )
