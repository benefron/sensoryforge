"""Regression tests for ledger finding F-012.

``GeneralizedTactileEncodingPipeline._canonical_to_legacy_config`` used to set
``legacy["pipeline"]["grid_size"] = rows * cols``, and ``ReceptorGrid``/
``GridManager`` treat an int ``grid_size`` as a PER-SIDE count. A 20x20
canonical grid therefore allocated a 400x400 (160,000-receptor) lattice, and
the README's 80x80 quick-start example allocated 41M receptors — enough to
OOM a laptop. Two integration tests
(``test_gui_cli_parity::test_pipeline_accepts_canonical_config`` and
``test_regression_refactoring::test_canonical_config_loads_via_adapter``)
were observed exceeding 5 GB RSS and being killed by the OS during the
2026-09-14 publication-readiness audit.

The fix: the adapter now emits ``grid_size = (rows, cols)`` — a tuple, which
``ReceptorGrid.__init__`` already accepts and uses directly as ``(n_x, n_y)``.
"""

import pytest
import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)


def _canonical_config(rows: int, cols: int) -> dict:
    return {
        "grids": [
            {
                "name": "grid",
                "arrangement": "grid",
                "rows": rows,
                "cols": cols,
                "spacing": 0.15,
            }
        ],
        "populations": [
            {
                "name": "SA Pop",
                "target_grid": "grid",
                "neuron_type": "SA",
                "neurons_per_row": 4,
                "innervation_method": "gaussian",
                "connections_per_neuron": 4,
                "sigma_d_mm": 0.3,
                "filter_method": "none",
                "neuron_model": "izhikevich",
                "input_gain": 1.0,
                "noise_std": 0.0,
            }
        ],
        "simulation": {"dt": 0.1, "device": "cpu", "seed": 42},
    }


def test_adapter_emits_rows_cols_tuple_not_product():
    """The adapter's grid_size must be a (rows, cols) tuple, never rows*cols."""
    pipeline = GeneralizedTactileEncodingPipeline.__new__(
        GeneralizedTactileEncodingPipeline
    )
    canonical = _canonical_config(rows=20, cols=20)
    legacy = pipeline._canonical_to_legacy_config(canonical)

    grid_size = legacy["pipeline"]["grid_size"]
    assert grid_size == (20, 20), (
        f"grid_size={grid_size!r} — F-012 regression: adapter must emit "
        "(rows, cols), not rows*cols (which ReceptorGrid would treat as a "
        "per-side count, squaring the receptor count)"
    )


def test_canonical_20x20_allocates_400_receptors_not_160000():
    """A 20x20 canonical grid must allocate 400 receptors end-to-end.

    Before the F-012 fix this allocated a 400x400 = 160,000-receptor
    lattice. This test would hang/OOM under the old adapter for larger
    grids; 20x20 is kept small enough to still fail fast (assertion, not
    a timeout) if the regression reappears.
    """
    pipeline = GeneralizedTactileEncodingPipeline.from_config(
        _canonical_config(rows=20, cols=20)
    )

    n_x, n_y = pipeline.grid_manager.grid_size
    assert (n_x, n_y) == (20, 20)
    assert n_x * n_y == 400

    stimulus_tensor, _, _ = pipeline.generate_stimulus(
        stimulus_type="gaussian", amplitude=10.0, sigma=1.0, duration=5.0
    )
    assert stimulus_tensor.shape[-2:] == (20, 20)


def _canonical_neuron_count_config(
    neuron_rows: int, neuron_cols: int, grid_rows: int = 30, grid_cols: int = 30
) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(
                name="grid",
                arrangement="grid",
                rows=grid_rows,
                cols=grid_cols,
                spacing=0.15,
            )
        ],
        populations=[
            PopulationConfig(
                name="SA Pop",
                target_grid="grid",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="gaussian",
                neuron_rows=neuron_rows,
                neuron_cols=neuron_cols,
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0, sigma=1.0),
        simulation=SimulationConfig(device="cpu", dt=0.1),
    )


@pytest.mark.parametrize("rows,cols", [(4, 4), (3, 5)])
def test_adapter_does_not_square_neuron_counts(rows, cols):
    """Regression for F-025: the adapter must build rows*cols neurons, not (rows*cols)**2.

    Before the fix, ``_canonical_to_legacy_config`` wrote ``neuron_rows * neuron_cols``
    into ``neurons.sa_neurons``, which ``InnervationModule`` then squares again
    (treating it as per-row). A canonical 4x4 population built 256 neurons instead
    of 16; a 3x5 population built 225 instead of 15.
    """
    config = _canonical_neuron_count_config(rows, cols)
    pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
    assert pipeline.sa_innervation.num_neurons == rows * cols

    engine = SimulationEngine(config)
    engine_neurons = engine.populations[0]["innervation"].num_neurons
    assert engine_neurons == rows * cols
    assert pipeline.sa_innervation.num_neurons == engine_neurons, (
        "legacy pipeline and SimulationEngine must build identical neuron counts "
        "for the same canonical config"
    )


def test_legacy_config_mistaken_total_neuron_count_raises():
    """Regression for F-023: a legacy per-row key that reads like a total must fail fast.

    ``neurons.sa_neurons`` is per-row (squared by InnervationModule), so passing
    what looks like a total count (e.g. 100000, meaning "100000 SA neurons") would
    silently build a 100000x100000-neuron population and allocate a dense weight
    tensor far beyond any reasonable memory budget. This must raise instead.
    """
    config = {
        "pipeline": {"device": "cpu", "grid_size": 80, "spacing": 0.15},
        "neurons": {"sa_neurons": 100000, "dt": 0.1},
    }
    with pytest.raises(ValueError, match="sa_neurons"):
        GeneralizedTactileEncodingPipeline.from_config(config)
