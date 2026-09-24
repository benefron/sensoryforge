"""Every stimulus type defaults to a unit peak and renders no negative values.

D-0437899 (ledger F-083): the named ``gaussian``, ``texture``, ``moving`` and
``repeated_pattern`` types (and the legacy ``trapezoidal``/``step``/``ramp``
generators) defaulted to amplitude 30 while every other type used 1.0, and
``gabor``/``texture`` rendered negative pressure by default. Now every type
peaks at 1.0 when its amplitude is not set, and none goes below zero unless
asked to (``signed=True`` on the Gabor types).

Each type is rendered at its defaults through
:func:`sensoryforge.stimuli.render.render_for_config` for 500 ms at dt 1 ms,
on two grids at 0.15 mm:

* **40 x 40**, the default canvas: no value may be negative, and none may
  exceed 1.0 (by more than float rounding) -- a stimulus that still defaulted
  to 30 fails here.
* **41 x 41**, which has a receptor at the centre, so a stimulus centred on
  the origin is sampled at its true peak: the peak must be 1.0 within 1%.
  On 40 x 40 the nearest receptors are 0.075 mm off-centre in x and y, which
  loses up to 3% for the unit types and 25% for ``gabor`` (sigma 0.3 mm,
  wavelength 0.5 mm) -- a sampling loss, not an amplitude.

``repeated_pattern`` sums six overlapping copies, so its sum is not 1.0; its
per-copy amplitude is checked instead, by rendering one copy.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.register_components import register_all
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.presets import PRESETS, preset
from sensoryforge.stimuli.render import effective_defaults, render_for_config
from sensoryforge.stimuli.texture import GaborTexture

register_all()

DURATION_MS = 500.0
DT_MS = 1.0
SPACING_MM = 0.15

#: Legacy generator names render_stimulus knows besides the registry.
#: ``custom`` is left out: it has no default, it renders the caller's tensor.
LEGACY_NAMES = ["trapezoidal", "step", "ramp"]

#: The kinds StaticStimulus draws (``static``, and the children of
#: ``composite``/``timeline``); each is given nothing but its kind.
STATIC_KINDS = ["gaussian", "point", "edge", "gabor", "edge_grating"]

#: Types with no stimulus of their own: they are rendered with one child that
#: sets nothing, so the defaults under test are the child's.
NEEDS_CHILDREN = {"composite", "static", "timeline"}


def _child(kind: str = "gaussian") -> Dict[str, Any]:
    return {"class": "StaticStimulus", "stim_type": kind, "params": {}}


def _stimulus(name: str, kind: str = "gaussian") -> StimulusConfig:
    """A config block for ``name`` that sets no parameter of the stimulus."""
    stim = StimulusConfig(type=name)
    if name == "static":
        stim.params["stim_type"] = kind
    elif name == "composite":
        stim.stimuli = [_child(kind)]
    elif name == "timeline":
        stim.params["sub_stimuli"] = [
            {"stimulus": _child(kind), "onset_ms": 0.0, "duration_ms": DURATION_MS}
        ]
    return stim


def _render(stim: StimulusConfig, n: int) -> torch.Tensor:
    config = SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=n, cols=n, spacing=SPACING_MM)],
        populations=[],
        stimulus=stim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames, _, _, _ = render_for_config(
            config, duration_ms=DURATION_MS, dt_ms=DT_MS
        )
    return frames


def _cases():
    names = sorted(set(STIMULUS_REGISTRY.list_registered()) - {"repeated_pattern"})
    cases = []
    for name in names + LEGACY_NAMES:
        if name in NEEDS_CHILDREN:
            cases += [pytest.param(name, k, id=f"{name}-{k}") for k in STATIC_KINDS]
        else:
            cases.append(pytest.param(name, None, id=name))
    return cases


def test_every_registered_type_is_covered():
    """A newly registered type is picked up by the cases, not silently skipped."""
    covered = {c.values[0] for c in _cases()} | {"repeated_pattern"}
    assert set(STIMULUS_REGISTRY.list_registered()) <= covered


@pytest.mark.parametrize("name, kind", _cases())
def test_default_render_is_non_negative_and_never_exceeds_1(name, kind):
    frames = _render(_stimulus(name, kind or "gaussian"), 40)
    assert float(frames.min()) >= 0.0, (name, kind, float(frames.min()))
    assert float(frames.max()) <= 1.0 + 1e-5, (name, kind, float(frames.max()))


@pytest.mark.parametrize("name, kind", _cases())
def test_default_render_peaks_at_1_on_a_grid_with_a_centre_receptor(name, kind):
    frames = _render(_stimulus(name, kind or "gaussian"), 41)
    assert float(frames.max()) == pytest.approx(1.0, rel=0.01), (name, kind)
    assert float(frames.min()) >= 0.0, (name, kind)


def test_repeated_pattern_copies_each_default_to_amplitude_1():
    base = effective_defaults("repeated_pattern")["base_stimulus"]
    assert base["params"]["amplitude"] == 1.0

    one_copy = _stimulus("repeated_pattern")
    one_copy.params.update({"copies_x": 1, "copies_y": 1})
    assert float(_render(one_copy, 41).max()) == pytest.approx(1.0, rel=0.01)

    # The default is six overlapping unit copies: non-negative, and no more
    # than six times one copy (3.85 on 40 x 40, where they overlap).
    frames = _render(_stimulus("repeated_pattern"), 40)
    assert float(frames.min()) >= 0.0
    assert 1.0 < float(frames.max()) <= 6.0


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_element_has_amplitude_1(name):
    for layer in preset(name, DURATION_MS)["layers"]:
        assert layer["shape"]["amplitude"] == 1.0, (name, layer["shape"])


def test_schema_amplitude_default_matches_the_types():
    """An amplitude equal to the schema default counts as unset; it must render the same."""
    assert StimulusConfig().amplitude == 1.0


# ------------------------------------------------------------------ signed


@pytest.mark.parametrize("name", ["gabor", "texture"])
def test_signed_restores_the_zero_mean_form(name):
    stim = _stimulus(name)
    stim.params["signed"] = True
    frames = _render(stim, 41)
    assert float(frames.min()) < 0.0
    assert float(frames.max()) == pytest.approx(1.0, rel=0.01)


def test_the_non_negative_gabor_is_the_raised_cosine_under_the_window():
    """``amplitude * window * (1 + cos) / 2``: the mean of the two signed phases."""
    xx, yy = torch.meshgrid(
        torch.linspace(-2, 2, 81), torch.linspace(-2, 2, 81), indexing="ij"
    )
    kw = dict(amplitude=2.0, sigma=0.8, wavelength=0.6, orientation=0.4)
    raised = GaborTexture(**kw)(xx, yy)
    signed = GaborTexture(**kw, signed=True)(xx, yy)
    envelope = 2.0 * torch.exp(-(xx**2 + yy**2) / (2 * 0.8**2))
    assert torch.allclose(raised, 0.5 * (envelope + signed), atol=1e-6)
    assert float(raised.min()) >= 0.0


def test_signed_is_an_advanced_bool_defaulting_to_false():
    (spec,) = [s for s in GaborTexture.get_param_spec() if s.name == "signed"]
    assert spec.dtype == "bool"
    assert spec.default is False
    assert spec.advanced is True
    assert spec.help


@pytest.mark.parametrize("signed", [False, True])
def test_signed_round_trips_through_to_dict_and_from_config(signed):
    stim = GaborTexture(sigma=0.7, signed=signed)
    data = stim.to_dict()
    assert data["signed"] is signed
    rebuilt = GaborTexture.from_config(data)
    assert rebuilt.to_dict() == data
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 21), torch.linspace(-1, 1, 21), indexing="ij"
    )
    assert torch.equal(rebuilt(xx, yy), stim(xx, yy))
