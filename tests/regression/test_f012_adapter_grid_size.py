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

import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline


def _canonical_config(rows: int, cols: int) -> dict:
    return {
        "grids": [
            {"name": "grid", "arrangement": "grid", "rows": rows, "cols": cols, "spacing": 0.15}
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
    pipeline = GeneralizedTactileEncodingPipeline.__new__(GeneralizedTactileEncodingPipeline)
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
    pipeline = GeneralizedTactileEncodingPipeline.from_config(_canonical_config(rows=20, cols=20))

    n_x, n_y = pipeline.grid_manager.grid_size
    assert (n_x, n_y) == (20, 20)
    assert n_x * n_y == 400

    stimulus_tensor, _, _ = pipeline.generate_stimulus(
        stimulus_type="gaussian", amplitude=10.0, sigma=1.0, duration=5.0
    )
    assert stimulus_tensor.shape[-2:] == (20, 20)
