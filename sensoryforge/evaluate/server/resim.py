"""Membrane voltage by re-simulation (roadmap A6b, decided 2026-10-23, commit 82034fc).

A bundle stores a population's ``filtered`` (after gain and sensor noise, before the
input floor) and its readout, not the membrane voltage. :func:`resimulate` runs the
population's neuron again from the stored ``filtered`` with
:meth:`~sensoryforge.core.simulation_engine.SimulationEngine._run_neuron_stage`,
the very code a run uses, built from ``config.json`` (neuron parameters, record and integration
steps, the floor the run applied, the population's noise seed), and compares what it
produced with what the run stored. The comparison is the point: a re-run that does not
match is shown as such, never as a voltage the bundle vouches for.

This is a viewer module, not the blind audit; it imports torch only when it runs.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

#: Names of neurons listed as differing in a verification (the count is always exact).
MAX_LISTED_DIFFERENCES = 20

# The neuron stage reseeds the global torch RNG around a noisy neuron; two re-runs must
# not interleave with each other.
_RUN_LOCK = threading.Lock()


class ResimError(Exception):
    """A re-run that cannot be made (the message says why)."""


@dataclass
class Resimulation:
    """One population re-run.

    Attributes:
        voltages: Bin-end membrane voltage ``[T, N]`` float32, mV.
        verification: What the re-run is compared against and the result (see
            :func:`verify`), plus ``seconds``, ``bundle_sha`` and ``code_sha``.
    """

    voltages: np.ndarray
    verification: Dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        """Bytes held by the voltage array."""
        return int(self.voltages.nbytes)


def verify(stored: np.ndarray, rerun: np.ndarray, readout: str) -> Dict[str, Any]:
    """Compare a re-run readout with the stored one, neuron by neuron.

    Args:
        stored: The stored readout ``[T, N]`` (spike counts or signed event counts as
            integers, or a float state).
        rerun: The re-run's readout, same shape.
        readout: ``spikes``, ``events`` or ``state``.

    Returns:
        ``{matches, n_differing, n_units, first_differences, max_abs_difference}``;
        spikes and events are compared as integers, a state exactly. The largest
        absolute difference is over every sample.
    """
    if stored.shape != rerun.shape:
        return {
            "matches": False,
            "n_differing": int(stored.shape[1]) if stored.ndim == 2 else 0,
            "n_units": int(stored.shape[1]) if stored.ndim == 2 else 0,
            "first_differences": [],
            "max_abs_difference": None,
            "note": f"shapes differ: stored {list(stored.shape)}, re-run "
            f"{list(rerun.shape)}",
        }
    if readout in ("spikes", "events"):
        a = np.asarray(stored, dtype=np.int64)
        b = np.rint(np.asarray(rerun, dtype=np.float64)).astype(np.int64)
    else:
        a = np.asarray(stored, dtype=np.float32)
        b = np.asarray(rerun, dtype=np.float32)
    bad = (a != b).any(axis=0)
    # a NaN on both sides is the same value
    if readout == "state":
        bad &= ~(np.isnan(a) & np.isnan(b)).all(axis=0)
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    largest = float(np.nanmax(diff)) if diff.size and np.isfinite(diff).any() else 0.0
    ids = np.flatnonzero(bad)
    return {
        "matches": not bool(ids.size),
        "n_differing": int(ids.size),
        "n_units": int(a.shape[1]),
        "first_differences": [int(i) for i in ids[:MAX_LISTED_DIFFERENCES]],
        "max_abs_difference": largest,
    }


def verify_probes(
    stored: np.ndarray, ids: List[int], voltages: np.ndarray
) -> Dict[str, Any]:
    """Compare the stored voltage probes with the re-run's voltage at the same neurons.

    Args:
        stored: The bundle's ``voltage_probe`` ``[T, K]`` float32, mV.
        ids: The probed neuron ids (``K`` of them).
        voltages: The re-run's bin-end voltage ``[T, N]``, mV.

    Returns:
        ``{ids, matches, n_differing, max_abs_difference}``: ``matches`` means every
        sample is equal (float32, exactly; a NaN on both sides is the same value);
        ``n_differing`` counts probed neurons that differ somewhere.
    """
    if (
        stored.ndim != 2
        or stored.shape[1] != len(ids)
        or voltages.shape[0] != stored.shape[0]
    ):
        return {
            "ids": [int(i) for i in ids],
            "matches": False,
            "n_differing": len(ids),
            "max_abs_difference": None,
            "note": f"shapes differ: stored {list(stored.shape)}, re-run "
            f"{list(voltages.shape)} at {len(ids)} ids",
        }
    a = np.asarray(stored, dtype=np.float32)
    b = np.asarray(voltages[:, ids], dtype=np.float32)
    bad = (a != b) & ~(np.isnan(a) & np.isnan(b))
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(diff)
    return {
        "ids": [int(i) for i in ids],
        "matches": not bool(bad.any()),
        "n_differing": int(bad.any(axis=0).sum()),
        "max_abs_difference": float(diff[finite].max()) if finite.any() else 0.0,
    }


def _population_entry(config_json: Dict[str, Any], name: str) -> Dict[str, Any]:
    for entry in config_json.get("populations") or []:
        if entry.get("name") == name:
            return entry
    raise ResimError(f"config.json has no population {name!r}")


def resimulate(bundle_dir: Path, population: str, readout: str) -> Resimulation:
    """Re-run one population's neuron stage from its stored ``filtered``.

    Args:
        bundle_dir: The bundle folder (``config.json`` and ``data.h5``).
        population: The population name.
        readout: The stored readout dataset (``spikes``, ``events`` or ``state``).

    Returns:
        The voltages and the verification against the stored readout.

    Raises:
        ResimError: If the config or the arrays cannot be read, the population is
            unknown, or its neuron model returns no voltage (an analog readout is its
            own state, which is already stored).
    """
    try:
        import torch

        from sensoryforge.config.schema import SensoryForgeConfig
        from sensoryforge.core.simulation_engine import SimulationEngine, build_neuron
        from sensoryforge.provenance import source_info
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        raise ResimError(f"re-simulation needs the SensoryForge engine ({exc})")
    try:
        with open(Path(bundle_dir) / "config.json") as fh:
            config_json = json.load(fh)
        config = SensoryForgeConfig.from_dict(config_json["config"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ResimError(f"config.json cannot be read ({exc})") from exc
    pop_cfg = next((p for p in config.populations if p.name == population), None)
    if pop_cfg is None:
        raise ResimError(f"config.json has no population {population!r}")
    entry = _population_entry(config_json, population)
    floor = (entry.get("encoder") or {}).get("input_floor_ma")
    try:
        with h5py.File(Path(bundle_dir) / "data.h5", "r") as f:
            filtered = np.asarray(
                f[f"populations/{population}/filtered"][...], dtype=np.float32
            )
            stored = np.asarray(f[f"populations/{population}/{readout}"][...])
            bundle_sha = f.attrs.get("sensoryforge_sha")
            probe_key = f"populations/{population}/voltage_probe"
            probe = (
                np.asarray(f[probe_key][...], dtype=np.float32)
                if probe_key in f
                else None
            )
    except (OSError, KeyError, ValueError) as exc:
        raise ResimError(f"data.h5 cannot be read ({exc})") from exc
    if isinstance(bundle_sha, bytes):
        bundle_sha = bundle_sha.decode()

    sensor = pop_cfg.effective_sensor_noise_std()
    membrane = pop_cfg.effective_membrane_noise_std()
    notes: List[str] = []
    started = time.perf_counter()
    with _RUN_LOCK, torch.no_grad():
        try:
            neuron = build_neuron(pop_cfg, config.simulation, device="cpu")
        except (ValueError, TypeError) as exc:
            raise ResimError(f"the neuron cannot be built ({exc})") from exc
        generator = None
        if pop_cfg.noise_seed is not None and (sensor > 0 or membrane > 0):
            generator = torch.Generator(device="cpu").manual_seed(pop_cfg.noise_seed)
        elif membrane > 0:
            notes.append(
                "the neuron has membrane noise but no noise_seed, so the run drew it "
                "from the process's global generator and it cannot be reproduced"
            )
        out = SimulationEngine._run_neuron_stage(
            torch.from_numpy(filtered)[None],
            neuron,
            dt_ms=config.simulation.dt_ms,
            integrate_dt_ms=config.simulation.integrate_dt_ms,
            noise_generator=generator,
            input_floor=None if floor is None else float(floor),
        )
    seconds = time.perf_counter() - started
    if "voltages" not in out:
        raise ResimError(
            "this neuron model returns no membrane voltage "
            f"(its readout is {sorted(k for k in out)[0]})"
        )
    rerun = out[readout][0].numpy() if readout in out else None
    if rerun is None:
        raise ResimError(f"the re-run produced no {readout!r} to compare with")
    check = verify(stored, rerun, readout)
    check.update(
        {
            "readout": readout,
            "seconds": round(seconds, 3),
            "bundle_sha": bundle_sha,
            "code_sha": source_info()["sha"],
            "membrane_noise_std": membrane,
            "noise_seed": pop_cfg.noise_seed,
            "input_floor_ma": floor,
            "notes": notes,
        }
    )
    voltages = out["voltages"][0].numpy().astype(np.float32)
    probe_ids = (entry.get("voltage_probe") or {}).get("ids")
    if probe is not None and probe_ids is not None:
        check["probes"] = verify_probes(probe, list(probe_ids), voltages)
    return Resimulation(voltages, check)
