"""Shared component-contract checks for the plugin template's two components.

Runs the same checks used across SensoryForge's own contract-test sweep (G4)
via ``sensoryforge.testing.contracts.check_component`` -- the exact function
named by V1's spec, run against both the ``innervation`` and the
``processing`` kind (the latter wired into the shared checks as of Wave U).
"""

import torch

from sensoryforge.testing.contracts import check_component

from sensoryforge_plugin_template.rf_builder import RadialFalloffRFBuilder
from sensoryforge_plugin_template.processing import GainThresholdLayer


def test_radial_falloff_rf_builder_satisfies_component_contract():
    instance = RadialFalloffRFBuilder(
        receptor_coords=torch.rand(30, 2), neuron_centers=torch.rand(4, 2)
    )
    check_component("innervation", RadialFalloffRFBuilder, instance)


def test_gain_threshold_layer_satisfies_component_contract():
    instance = GainThresholdLayer(gain=2.0, threshold=0.1)
    check_component("processing", GainThresholdLayer, instance)
