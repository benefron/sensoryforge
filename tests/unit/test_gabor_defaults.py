"""Every Gabor entry point renders the same default geometry (D-407c639).

The named ``gabor`` type (``GaborTexture``), the builder's ``Stimulus.gabor()``,
``StaticStimulus``'s ``gabor`` kind and the two functions behind them all
default to sigma = wavelength = 1.0 mm, so a Gabor built by any route with no
geometry given renders the same pattern.
"""

import torch

from sensoryforge.stimuli.builder import StaticStimulus, Stimulus
from sensoryforge.stimuli.stimulus import gabor_texture_torch
from sensoryforge.stimuli.texture import GaborTexture, gabor_texture

AXIS = torch.linspace(-4.0, 4.0, 81)
XX, YY = torch.meshgrid(AXIS, AXIS, indexing="ij")


def _reference():
    return GaborTexture()(XX, YY)


def test_the_class_defaults_are_one_millimetre():
    gabor = GaborTexture()
    assert gabor.sigma == 1.0 and gabor.wavelength == 1.0


def test_the_functions_render_the_class_default():
    assert torch.allclose(gabor_texture(XX, YY), _reference())
    assert torch.allclose(gabor_texture_torch(XX, YY, 0.0, 0.0), _reference())


def test_the_builder_routes_render_the_class_default():
    assert torch.allclose(Stimulus.gabor()(XX, YY), _reference())
    assert torch.allclose(StaticStimulus("gabor", {})(XX, YY), _reference())
