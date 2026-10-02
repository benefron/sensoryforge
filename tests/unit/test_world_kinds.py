"""Class kinds: timelines, end times, the layered form of a draw, the registry."""

from pathlib import Path

import pytest

from sensoryforge.world.kinds import LayeredKind, register_class_kind
from sensoryforge.world.schema import load_world

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)


def _values(class_name, **overrides):
    spec = WORLD.class_spec(class_name)
    values = {name: axis.midpoint() for name, axis in spec.axes.items()}
    values.update(overrides)
    return spec, values


def test_timeline_and_end_of_a_two_contact_episode():
    spec, v = _values(
        "twice",
        delay_ms=5.0,
        touch_ms=10.0,
        hold_ms=20.0,
        release_ms=10.0,
        pause_ms=7.0,
    )
    kind = spec.kind_obj
    assert kind.end_ms(v) == 5.0 + 2 * 40.0 + 7.0
    phases = kind.timeline(v)
    assert [p[0] for p in phases] == [
        "quiet",
        "touch",
        "hold",
        "release",
        "pause",
        "touch",
        "hold",
        "release",
    ]
    assert phases[-1][2] == pytest.approx(92.0)


def test_to_layer_maps_the_episode_onto_layered_fields():
    spec, v = _values(
        "sliders",
        delay_ms=3.0,
        touch_ms=4.0,
        hold_ms=5.0,
        slide_ms=20.0,
        release_ms=6.0,
        speed_mm_per_ms=0.01,
        direction_deg=90.0,
        amplitude=0.7,
        x_mm=0.1,
        y_mm=-0.2,
    )
    layer = spec.kind_obj.to_layer(spec, v)
    assert layer["timing"] == {
        "onset_ms": 3.0,
        "ramp_up_ms": 4.0,
        "hold_ms": 5.0,
        "slide_ms": 20.0,
        "ramp_down_ms": 6.0,
        "contacts": 1,
        "pause_ms": 0.0,
    }
    assert layer["motion"]["kind"] == "linear" and layer["motion"]["span"] == "slide"
    assert layer["motion"]["end"] == pytest.approx([0.0, 0.2], abs=1e-12)
    assert layer["shape"]["amplitude"] == 0.7
    assert (layer["pattern"]["x_mm"], layer["pattern"]["y_mm"]) == (0.1, -0.2)
    assert layer["modulation"] == {"kind": "none"}


def test_a_declared_motion_is_kept_and_moves_during_the_slide():
    spec, v = _values("orbit")
    layer = spec.kind_obj.to_layer(spec, v)
    assert layer["motion"]["kind"] == "circular" and layer["motion"]["span"] == "slide"


def test_the_quiet_kind():
    spec = WORLD.class_spec("quiet")
    v = {"quiet_ms": 30.0}
    assert spec.kind_obj.end_ms(v) == 30.0
    assert spec.kind_obj.timeline(v) == [["quiet", 0.0, 30.0]]
    assert spec.kind_obj.to_layer(spec, v) is None


def test_class_kinds_cannot_be_registered_twice():
    with pytest.raises(ValueError, match="already registered"):
        register_class_kind(LayeredKind())
