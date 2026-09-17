"""Worked example: an analog (non-spiking) DSL readout through the engine (N5).

The smallest complete example of the analog readout path described in
``docs/user_guide/analog_readouts.md``:

1. Define a leaky-integrator DSL model with no ``threshold``/``reset`` --
   an analog readout (N1).
2. Build a canonical config with ``neuron_model: dsl`` and that model's
   ``dsl_config``.
3. Run it through :class:`~sensoryforge.core.simulation_engine.SimulationEngine`
   and check the result carries ``"state"`` instead of ``"spikes"`` (N2, N3).
4. For contrast, add the same model *with* a threshold and confirm it
   spikes as usual in the same run.

Run it directly: ``python docs/examples/analog_dsl.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine


def main() -> None:
    grid = GridConfig(name="Grid", arrangement="grid", rows=12, cols=12, spacing=0.15)

    analog_population = PopulationConfig(
        name="Analog Population",
        neuron_type="SA",
        neuron_model="dsl",
        innervation_method="gaussian",
        neurons_per_row=4,
        connections_per_neuron=8,
        sigma_d_mm=0.3,
        dsl_config={
            "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
            "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
            "state_vars": {"v": -65.0},
            # No 'threshold'/'reset' -- readout="auto" (the default) infers
            # analog.
        },
    )

    spiking_population = PopulationConfig(
        name="Spiking Population",
        neuron_type="SA",
        neuron_model="dsl",
        innervation_method="gaussian",
        neurons_per_row=4,
        connections_per_neuron=8,
        sigma_d_mm=0.3,
        dsl_config={
            "equations": "dv/dt = (-(v - (-65.0)) + I) / 10.0",
            "threshold": "v >= -50.0",
            "reset": "v = -65.0",
            "state_vars": {"v": -65.0},
        },
    )

    config = SensoryForgeConfig(
        grids=[grid],
        populations=[analog_population, spiking_population],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=1.0),
    )

    engine = SimulationEngine(config)
    stimulus = torch.rand(30, 12, 12) * 300.0  # [time, height, width]
    results = engine.run(stimulus)

    analog_result = results["Analog Population"]
    assert "state" in analog_result
    assert "spikes" not in analog_result
    print(f"Analog Population state shape: {tuple(analog_result['state'].shape)}")

    spiking_result = results["Spiking Population"]
    assert "spikes" in spiking_result
    assert "state" not in spiking_result
    print(
        f"Spiking Population total spikes: "
        f"{int(spiking_result['spikes'].sum().item())}"
    )


if __name__ == "__main__":
    main()
