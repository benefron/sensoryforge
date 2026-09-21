"""Worked example: a plugin component gets a GUI form with no GUI code.

The Populations screen renders every receptive-field builder's parameters
from its ``get_param_spec()`` (``sensoryforge.gui.widgets.param_form``), so a
plugin builder registered with ``INNERVATION_REGISTRY`` appears in the RF
builder list with its own settings form, and editing a field writes into the
config that the engine runs. The same is true of filters, neuron models and
stimuli. See ``docs/extending/gui_forms.md``.

Run it directly: ``QT_QPA_PLATFORM=offscreen python
docs/examples/gui_plugin_param_form.py``. It is also executed by
``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5 import QtWidgets

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.innervation import BaseInnervation  # noqa: E402
from sensoryforge.gui.screens.populations_cards import InputsCard  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.widgets.param_form import ParamForm  # noqa: E402
from sensoryforge.registry import INNERVATION_REGISTRY  # noqa: E402
from sensoryforge.stimuli.base import ParamSpec  # noqa: E402


# 1. A plugin RF builder with three parameters, registered the way a
#    third-party package's entry point would register it.
class DemoInnervation(BaseInnervation):
    """A trivial builder: identity weights, three demo parameters."""

    def __init__(
        self,
        receptor_coords,
        neuron_centers,
        spread_mm: float = 0.3,
        k_neighbors: int = 8,
        normalize: str = "l2",
        device="cpu",
    ):
        super().__init__(receptor_coords, neuron_centers, device=device)
        self.spread_mm = float(spread_mm)
        self.k_neighbors = int(k_neighbors)
        self.normalize = str(normalize)

    def compute_weights(self, **kwargs) -> torch.Tensor:
        return torch.eye(self.num_neurons, self.num_receptors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread_mm": self.spread_mm,
            "k_neighbors": self.k_neighbors,
            "normalize": self.normalize,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "spread_mm",
                dtype="float",
                default=0.3,
                min_val=0.0,
                max_val=5.0,
                unit="mm",
                help="Demo spread.",
                group="Shape",
            ),
            ParamSpec(
                "k_neighbors",
                dtype="int",
                default=8,
                min_val=1,
                max_val=64,
                help="Demo neighbour count.",
                group="Shape",
            ),
            ParamSpec(
                "normalize",
                dtype="str",
                default="l2",
                choices=["l1", "l2", "none"],
                help="Demo normalisation.",
                group="Shape",
                advanced=True,
            ),
        ]


INNERVATION_REGISTRY.register("demo_plugin_rf", DemoInnervation)

# 2. A population that uses it, shown in the Populations screen's Inputs card.
config = SensoryForgeConfig(
    grids=[GridConfig(name="skin", rows=6, cols=6)],
    populations=[
        PopulationConfig(
            name="SA", target_grid="skin", innervation_method="demo_plugin_rf"
        )
    ],
)
session = Session(config)
card = InputsCard(session)
card.set_population("SA")
form = card.findChild(ParamForm)
assert form is not None, "no form was rendered for the plugin builder"

spread = form.widget_for("spread_mm")
assert isinstance(spread, QtWidgets.QDoubleSpinBox)
assert isinstance(form.widget_for("k_neighbors"), QtWidgets.QSpinBox)
assert isinstance(form.widget_for("normalize"), QtWidgets.QComboBox)
print(f"The plugin builder rendered 3 parameters (spread_mm = {spread.value()}).")

# 3. Editing a field writes into the config the engine runs.
spread.setValue(1.25)
assert session.config.populations[0].innervation_params["spread_mm"] == 1.25
print(
    "Edited spread_mm; the config now holds:",
    session.config.populations[0].innervation_params,
)
print("Done.")
