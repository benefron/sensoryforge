"""Regression for F-038 (task E6): seeded innervation must work on accelerators.

Before this fix, every seeded random draw in innervation.py paired a
CPU-only ``torch.Generator`` (the per-instance generator introduced for
F-006) with a tensor already moved to the target device, which PyTorch
refuses: "Expected a 'mps' device type for generator but found 'cpu'".
Fixed by always drawing on a CPU tensor, then moving the result to the
target device -- this also means a given seed's wiring is identical
across CPU/MPS/CUDA, which these tests assert directly.
"""

import pytest
import torch

from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import InnervationModule, FlatInnervationModule
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine

_ACCELERATOR = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else None)
)

pytestmark = pytest.mark.skipif(
    _ACCELERATOR is None, reason="no MPS or CUDA device available"
)


def test_innervation_module_weights_match_cpu_on_accelerator():
    cpu_gm = GridManager(grid_size=20, spacing=0.15, center=(0.0, 0.0), device="cpu")
    accel_gm = GridManager(
        grid_size=20, spacing=0.15, center=(0.0, 0.0), device=_ACCELERATOR
    )

    cpu_module = InnervationModule(
        neuron_type="SA",
        grid_manager=cpu_gm,
        neurons_per_row=3,
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=42,
    )
    accel_module = InnervationModule(
        neuron_type="SA",
        grid_manager=accel_gm,
        neurons_per_row=3,
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=42,
    )

    assert accel_module.innervation_weights.device.type == _ACCELERATOR
    torch.testing.assert_close(
        accel_module.innervation_weights.cpu(), cpu_module.innervation_weights
    )


def test_flat_innervation_module_weights_match_cpu_on_accelerator():
    torch.manual_seed(0)
    receptor_coords_cpu = torch.rand(100, 2) * 4.0 - 2.0

    cpu_module = FlatInnervationModule(
        neuron_type="SA",
        receptor_coords=receptor_coords_cpu,
        neurons_per_row=3,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=7,
        device="cpu",
    )
    accel_module = FlatInnervationModule(
        neuron_type="SA",
        receptor_coords=receptor_coords_cpu.to(_ACCELERATOR),
        neurons_per_row=3,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=7,
        device=_ACCELERATOR,
    )

    assert accel_module.innervation_weights.device.type == _ACCELERATOR
    torch.testing.assert_close(
        accel_module.innervation_weights.cpu(), cpu_module.innervation_weights
    )


def test_simulation_engine_runs_on_accelerator_with_seeded_population():
    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=6, cols=6, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="SA Pop",
                target_grid="grid",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="gaussian",
                neurons_per_row=2,
                seed=11,
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0, sigma=1.0),
        simulation=SimulationConfig(device=_ACCELERATOR, dt_ms=1.0),
    )
    engine = SimulationEngine(config)
    stimulus = torch.rand(20, 6, 6, device=_ACCELERATOR) * 10.0
    result = engine.run(stimulus)
    assert "SA Pop" in result
    assert result["SA Pop"]["spikes"].device.type == _ACCELERATOR
