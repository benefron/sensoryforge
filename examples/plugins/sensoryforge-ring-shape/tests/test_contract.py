"""The ring shape passes the world contract check and gives a hand-checkable value."""

import torch

from sensoryforge.testing.world_contracts import check_world_component

from sensoryforge_ring_shape.shape import register, ring


def test_ring_satisfies_the_world_contract():
    register()
    register()  # a second run is harmless
    check_world_component("shape", "ring")


def test_ring_peaks_on_its_radius():
    x = torch.tensor([0.0, 0.3, 0.6], dtype=torch.float64)
    y = torch.zeros_like(x)
    p = {"radius_mm": 0.3, "width_mm": 0.1}
    out = ring(x, y, p)
    assert out[1].item() == 1.0  # on the circle: the full profile
    expected = torch.exp(torch.tensor(-4.5, dtype=torch.float64))  # 0.3 mm off
    assert torch.allclose(out[0], expected) and torch.allclose(out[2], expected)
    assert out.dtype == x.dtype
