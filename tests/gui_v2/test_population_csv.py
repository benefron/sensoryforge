"""Receptive fields export as a CSV folder and import back unchanged.

The old Grid tab's CSV export/import (CLAUDE.md "CSV Population
Import/Export"): export now lives in core (``write_csv_folder``) with a
button on the Populations receptive-field bench; import is the ``imported``
RF builder, chosen in the Inputs card with the folder as its path.
"""

import json

import pytest

pytestmark = pytest.mark.gui

import torch  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.bench.rf_footprint import (  # noqa: E402
    RfFootprintBench,
    build_population_bank_for_config,
)
from sensoryforge.gui.session import Session  # noqa: E402


def _config(rows=12):
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=rows, cols=rows, spacing=0.15)],
        populations=[
            PopulationConfig(
                name="SA",
                target_grid="skin",
                innervation_method="gaussian",
                neurons_per_row=5,
                seed=3,
            )
        ],
    )


def _export(qtbot, config, folder):
    bench = RfFootprintBench(Session(config))
    qtbot.addWidget(bench)
    bench.set_population("SA")
    bench.choose_folder = lambda *a: str(folder)
    bench.export_rf_button.click()
    return bench


def test_the_bench_button_writes_the_four_files(qtbot, tmp_path):
    folder = tmp_path / "rf"
    _export(qtbot, _config(), folder)
    names = sorted(p.name for p in folder.iterdir())
    assert names == [
        "bank.pt",
        "innervation_weights.csv",
        "manifest.json",
        "neuron_positions.csv",
    ]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (manifest["num_neurons"], manifest["num_receptors"]) == (25, 144)


def test_export_then_import_builds_the_same_receptive_fields(qtbot, tmp_path):
    folder = tmp_path / "rf"
    config = _config()
    _export(qtbot, config, folder)
    original = build_population_bank_for_config(config, "SA")

    imported = _config()
    imported.populations[0].innervation_method = "imported"
    imported.populations[0].innervation_params = {"path": str(folder)}
    again = build_population_bank_for_config(imported, "SA")
    assert torch.allclose(again.weights, original.weights, atol=1e-6)
    assert torch.allclose(again.neuron_centers, original.neuron_centers, atol=1e-6)


def test_importing_onto_a_grid_of_another_size_is_refused(qtbot, tmp_path):
    folder = tmp_path / "rf"
    _export(qtbot, _config(rows=12), folder)
    other = _config(rows=10)
    other.populations[0].innervation_method = "imported"
    other.populations[0].innervation_params = {"path": str(folder)}
    with pytest.raises(ValueError, match="144"):
        build_population_bank_for_config(other, "SA")
