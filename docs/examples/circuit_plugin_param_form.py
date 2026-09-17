"""Worked example: extending the Circuit tab, using a plugin RF builder as the
model (Phase 3, Wave R, R2).

The smallest complete example of the extension path described in
``docs/extending/add_gui_node.md``:

1. A plugin RF builder (``DemoTemplateInnervation``) is registered with
   ``INNERVATION_REGISTRY`` and its ``get_param_spec()`` renders as a real
   settings form in the Circuit tab's inspector -- with no GUI code written
   for this component. This is the existing, plugin-only extension point.
2. A small stand-in for the change ``_build_visualisation`` in
   ``sensoryforge/gui/circuit/inspector.py`` would need for a *custom
   preview*, since that dispatch is not plugin-extensible today (disclosed
   in the extending guide, not hidden). This does not monkeypatch the
   shipped function -- it shows the shape an in-repo change would take,
   called directly.

Run it directly: ``QT_QPA_PLATFORM=offscreen python
docs/examples/circuit_plugin_param_form.py``. It is also executed by
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

from sensoryforge.config.schema import PopulationInput, RFBuilderConfig
from sensoryforge.core.innervation import BaseInnervation
from sensoryforge.gui.circuit.inspector import build_node_inspector
from sensoryforge.gui.circuit.nodes import RFBankNode
from sensoryforge.registry import INNERVATION_REGISTRY
from sensoryforge.stimuli.base import ParamSpec

# --------------------------------------------------------------------------- #
# 1. A plugin RF builder with three parameters, registered the way a
#    third-party package's entry point (or register_components.py, for an
#    in-repo builder) would register it.
# --------------------------------------------------------------------------- #


class DemoTemplateInnervation(BaseInnervation):
    """A trivial builder: identity weights, three demo parameters."""

    def __init__(self, receptor_coords, neuron_centers, device="cpu", **params):
        super().__init__(receptor_coords, neuron_centers, device=device)
        self.spread_mm = float(params.get("spread_mm", 0.3))
        self.k_neighbors = int(params.get("k_neighbors", 8))
        self.normalize = str(params.get("normalize", "l2"))

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
                choices=None,
                help="Demo receptive-field spread.",
                group="Shape",
                advanced=False,
            ),
            ParamSpec(
                "k_neighbors",
                dtype="int",
                default=8,
                min_val=1,
                max_val=64,
                unit=None,
                choices=None,
                help="Demo neighbour count.",
                group="Shape",
                advanced=False,
            ),
            ParamSpec(
                "normalize",
                dtype="str",
                default="l2",
                min_val=None,
                max_val=None,
                unit=None,
                choices=["l1", "l2", "none"],
                help="Demo weight normalisation.",
                group="Shape",
                advanced=True,
            ),
        ]


INNERVATION_REGISTRY.register("wave_r_demo_template", DemoTemplateInnervation)

node = RFBankNode("rf_demo")
node.pop_input = PopulationInput(
    grid="Main Grid", rf=RFBuilderConfig(method="wave_r_demo_template")
)

inspector_widget = build_node_inspector(node, expert_mode=False)


def _find_param_widgets(widget: QtWidgets.QWidget):
    direct = widget.property("param_widgets")
    if direct:
        return direct
    for child in widget.findChildren(QtWidgets.QWidget):
        found = child.property("param_widgets")
        if found:
            return found
    return None


param_widgets = _find_param_widgets(inspector_widget)
assert param_widgets is not None, "no param form was rendered for the plugin builder"
assert "spread_mm" in param_widgets and isinstance(
    param_widgets["spread_mm"], QtWidgets.QDoubleSpinBox
), "spread_mm should render as a float spin box"
assert "k_neighbors" in param_widgets and isinstance(
    param_widgets["k_neighbors"], QtWidgets.QSpinBox
), "k_neighbors should render as an int spin box"
assert "normalize" in param_widgets and isinstance(
    param_widgets["normalize"], QtWidgets.QComboBox
), "normalize has choices, so it should render as a combo box"

# The advanced param is built but hidden in Basic mode (Expert mode
# convention). The widgets are never shown (this script runs offscreen with
# no window on screen), so isVisible() would be False regardless of the
# setVisible() call above it in the ancestor chain -- isHidden() reflects
# the widget's own explicit hidden flag instead, which is what
# build_param_form actually sets.
assert param_widgets["normalize"].isHidden() is True
expert_widget = build_node_inspector(node, expert_mode=True)
expert_params = _find_param_widgets(expert_widget)
assert expert_params["normalize"].isHidden() is False

print(
    "Part 1: plugin RF builder 'wave_r_demo_template' rendered "
    f"{len(param_widgets)} parameters with no GUI code of its own "
    f"(spread_mm spin box default {param_widgets['spread_mm'].value()})."
)

# Editing a widget writes straight back onto the node's own config object --
# no separate GUI model.
param_widgets["spread_mm"].setValue(1.25)
assert node.pop_input.rf.params["spread_mm"] == 1.25
print(
    "Editing the spin box wrote back to node.pop_input.rf.params:",
    node.pop_input.rf.params,
)


# --------------------------------------------------------------------------- #
# 2. What a custom PREVIEW requires today: an in-repo change, not a plugin
#    hook. This function is shaped exactly like
#    sensoryforge.gui.circuit.inspector._build_visualisation, and is called
#    directly -- it stands in for what a contributor would add there, it
#    does not patch the shipped dispatch table.
# --------------------------------------------------------------------------- #


def _demo_visualisation(node_type: str, demo_node) -> QtWidgets.QWidget:
    """A stand-in for the in-repo change inspector._build_visualisation needs
    to show something other than the generic RFBank placeholder for a
    specific registered component."""
    if (
        node_type == "RFBank"
        and demo_node.pop_input.rf.method == "wave_r_demo_template"
    ):
        label = QtWidgets.QLabel(
            f"DemoTemplateInnervation preview: spread={demo_node.pop_input.rf.params.get('spread_mm')}mm, "
            f"k={demo_node.pop_input.rf.params.get('k_neighbors')}"
        )
        return label
    raise ValueError(f"no demo preview for {node_type!r}")


custom_preview = _demo_visualisation("RFBank", node)
assert isinstance(custom_preview, QtWidgets.QLabel)
print("Part 2 (in-repo shape, not a plugin hook):", custom_preview.text())

print("\nDone.")
