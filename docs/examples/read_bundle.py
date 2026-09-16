"""Worked example: write a data bundle, then read it back (Wave J, J5).

The smallest complete round trip through :mod:`sensoryforge.io.bundle`,
described in ``docs/user_guide/bundles.md``:

1. Build a tiny canonical config and a :class:`SimulationEngine`.
2. Run it with ``bundle_dir=`` set, so it writes a bundle to disk.
3. Read the bundle back with :func:`~sensoryforge.io.bundle.load_bundle` and
   inspect every piece: the config, the per-population
   :class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank`, the stimulus, and
   the per-population drive/filtered/spikes arrays.
4. Show the two ways to read ``data.h5`` directly (without ``load_bundle``):
   with ``h5py`` and as a flat ``pandas.DataFrame`` of one population's
   spike counts.

Run it directly: ``python docs/examples/read_bundle.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import load_bundle


def main() -> None:
    config = SensoryForgeConfig(
        grids=[GridConfig(name="Main Grid", arrangement="grid", rows=8, cols=8, spacing=0.15)],
        populations=[
            PopulationConfig(
                name="SA Population",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="gaussian",
                neurons_per_row=3,
                seed=1,
            ),
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / "bundle"

        # 1-2: run and write the bundle in one call.
        engine = SimulationEngine(config)
        stimulus = torch.rand(1, 20, 8, 8)
        engine.run(
            stimulus,
            bundle_dir=bundle_dir,
            stimulus_config={"type": "custom_random"},
            seed=1,
        )
        print(f"Bundle written to {bundle_dir}")

        # 3: read it back.
        bundle = load_bundle(bundle_dir)
        print(f"Loaded config: dt_ms={bundle.config.simulation.dt_ms}")

        bank = bundle.banks["SA Population"]
        print(
            f"SA Population bank: {bank.num_neurons} neurons, "
            f"{bank.num_receptors} receptors"
        )

        print(f"Stimulus shape: {tuple(bundle.stimulus.shape)}")
        print(f"time_ms: {bundle.time_ms[0].item()} .. {bundle.time_ms[-1].item()} ms")

        pop = bundle.populations["SA Population"]
        spike_counts = pop["spikes"]  # [T, N] int16 sub-step spike counts
        print(f"drive shape: {tuple(pop['drive'].shape)}")
        print(f"Total spikes (sum of counts): {int(spike_counts.sum())}")
        # Spikes are stored as PER-BIN COUNTS, not a binary raster -- a bin
        # can hold more than one sub-step spike (F-008). Use > 0 to recover
        # a binary raster:
        binary_raster = spike_counts > 0
        print(f"Bins with >=1 spike: {int(binary_raster.sum())}")

        # 4a: reading data.h5 directly with h5py (no load_bundle).
        with h5py.File(bundle_dir / "data.h5", "r") as f:
            dt_ms = f.attrs["dt_ms"]
            raw_spikes = f["populations"]["SA Population"]["spikes"][()]
            print(f"h5py: dt_ms={dt_ms}, spikes dtype={raw_spikes.dtype}")

        # 4b: as a flat pandas.DataFrame (optional dependency; skipped if
        # pandas is not installed).
        try:
            import pandas as pd
        except ImportError:
            print("pandas not installed; skipping DataFrame example")
        else:
            df = pd.DataFrame(
                spike_counts.numpy(),
                columns=[f"neuron_{i}" for i in range(spike_counts.shape[1])],
            )
            df.insert(0, "time_ms", bundle.time_ms.numpy())
            print(df.head())


if __name__ == "__main__":
    main()
