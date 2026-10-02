"""The world kernel: shapes equal layered's; pattern batches; motion; registries."""

import pytest
import torch

from sensoryforge.stimuli import layered
from sensoryforge.world import kernel

N = 33
XS = torch.linspace(-2.0, 2.0, N, dtype=torch.float64)
X, Y = torch.meshgrid(XS, XS, indexing="ij")

PARAMS = {
    "gaussian": [{"sigma_mm": 0.2}, {"sigma_mm": 0.7}],
    "disc": [
        {"diameter_mm": 1.0, "edge_mm": 0.2},
        {"diameter_mm": 0.6, "edge_mm": 0.0},
    ],
    "bar": [
        {"width_mm": 0.1, "length_mm": 0.0, "orientation_deg": 30.0},
        {
            "width_mm": 0.3,
            "length_mm": 1.0,
            "orientation_deg": 100.0,
            "profile": "flat",
        },
    ],
    "grating": [
        {"wavelength_mm": 0.5, "orientation_deg": 20.0, "phase_deg": 45.0},
        {"wavelength_mm": 0.9, "orientation_deg": 70.0, "signed": True},
        {"wavelength_mm": 0.7, "profile": "square", "duty": 0.3},
    ],
    "gabor": [
        {
            "sigma_mm": 0.5,
            "wavelength_mm": 0.4,
            "orientation_deg": 10.0,
            "phase_deg": 90.0,
        },
        {
            "sigma_mm": 0.8,
            "wavelength_mm": 0.7,
            "orientation_deg": 135.0,
            "signed": True,
        },
    ],
}


def _tensors(params):
    return {
        k: (
            torch.tensor([float(v)], dtype=torch.float64).view(1, 1, 1)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            else v
        )
        for k, v in params.items()
    }


@pytest.mark.parametrize("kind", sorted(PARAMS))
def test_vectorised_shapes_equal_layered(kind):
    for given in PARAMS[kind]:
        params = {**layered.defaults(layered.SHAPES[kind]), **given}
        want = layered._SHAPE_FUNCTIONS[kind](X, Y, params)
        got = kernel.SHAPE_KINDS[kind].fn(X, Y, _tensors(params))[0]
        torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_a_batch_of_parameters_equals_each_alone_bit_for_bit():
    sigmas = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64).view(3, 1, 1)
    gaussian = kernel.SHAPE_KINDS["gaussian"].fn
    batch = gaussian(X, Y, {"sigma_mm": sigmas})
    for i in range(3):
        alone = gaussian(X, Y, {"sigma_mm": sigmas[i : i + 1]})
        assert torch.equal(batch[i], alone[0])


def test_pattern_batch_pads_with_zero_scales_and_adds_placement():
    base = {"kind": "braille", **layered.defaults(layered.PATTERNS["braille"])}
    patterns = [
        {**base, "dots": "1", "x_mm": 0.5, "y_mm": 0.0},
        {**base, "dots": "123456", "x_mm": 0.0, "y_mm": -0.5},
    ]
    pos, scales = kernel.pattern_batch("braille", patterns, torch.float64, "cpu")
    assert pos.shape == (2, 6, 2) and scales.shape == (2, 6)
    assert scales[0].tolist() == [1.0, 0, 0, 0, 0, 0]
    assert scales[1].tolist() == [1.0] * 6
    for i, pattern in enumerate(patterns):
        want, _ = layered.pattern_positions(pattern)
        expected = torch.tensor(want, dtype=torch.float64)
        torch.testing.assert_close(pos[i, : len(want)], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "motion",
    [
        {"kind": "linear", "start": [0.0, 0.0], "end": [2.0, 1.0]},
        {"kind": "circular", "radius_mm": 1.0, "revolutions": 0.5, "start_deg": 10.0},
        {"kind": "path", "waypoints": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
    ],
)
def test_motion_offsets_follow_layered(motion):
    time_ms = torch.arange(11, dtype=torch.float64)
    timing = {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 10, "ramp_down_ms": 0}
    want = layered.motion_offsets({**motion, "span": "hold"}, timing, time_ms, 10.0)
    progress = (time_ms / 10.0).view(1, 11)
    got = kernel.motion_offsets(motion, progress)[0]
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_a_registered_shape_also_works_in_a_layered_stimulus():
    def ring(x, y, p):
        r = torch.sqrt(x**2 + y**2)
        return torch.exp(-((r - p["radius_mm"]) ** 2) / (2.0 * p["width_mm"] ** 2))

    specs = [
        layered._f("amplitude", 1.0, 0.0, 1.0e4),
        layered._f("radius_mm", 1.0, 0.0, 10.0, "mm"),
        layered._f("width_mm", 0.1, 0.001, 10.0, "mm"),
    ]
    kernel.register_shape("test_ring", ring, specs)
    try:
        layer = layered.default_layer()
        layer["shape"] = {"kind": "test_ring", "radius_mm": 1.0}
        layer["timing"] = {
            "onset_ms": 0,
            "ramp_up_ms": 0,
            "hold_ms": None,
            "ramp_down_ms": 0,
        }
        xx, yy = X.float(), Y.float()
        frames = layered.render_layers([layer], xx, yy, dt_ms=1.0, total_ms=2.0)
        want = ring(
            xx, yy, {"radius_mm": torch.tensor(1.0), "width_mm": torch.tensor(0.1)}
        )
        torch.testing.assert_close(frames[0], want, atol=1e-6, rtol=0)
        assert kernel.shape_specs("test_ring") == specs
    finally:
        kernel.SHAPE_KINDS.pop("test_ring")


def test_unknown_kinds_name_the_known_ones():
    with pytest.raises(ValueError, match="unknown shape kind 'blob'"):
        kernel.shape_specs("blob")
    with pytest.raises(ValueError, match="unknown pattern kind 'spiral'"):
        kernel.pattern_specs("spiral")
    with pytest.raises(ValueError, match="unknown modulation kind 'wobble'"):
        kernel.modulation_specs("wobble")
    assert set(kernel.MODULATION_KINDS) >= {"none", "sine", "pulses"}


def test_an_unknown_modulation_kind_in_a_layered_stimulus_is_a_value_error():
    layer = layered.default_layer()
    layer["timing"] = {
        "onset_ms": 0,
        "ramp_up_ms": 0,
        "hold_ms": None,
        "ramp_down_ms": 0,
    }
    layer["modulation"] = {"kind": "sinus"}
    with pytest.raises(ValueError, match="unknown modulation kind 'sinus'"):
        layered.render_layers([layer], X.float(), Y.float(), dt_ms=1.0, total_ms=2.0)
