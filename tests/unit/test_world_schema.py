"""World declarations: binding, defaults, precedence, identity, validation."""

import copy
import re
from pathlib import Path

import pytest
import yaml

from sensoryforge.world.schema import World, load_world

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
RAW = yaml.safe_load((WORLDS / "tactile_small.yml").read_text())


def _world(mutate=None):
    raw = copy.deepcopy(RAW)
    if mutate is not None:
        mutate(raw["world"])
    return World.from_dict(raw)


def test_the_fixture_world_loads():
    world = load_world(WORLDS / "tactile_small.yml")
    assert set(world.classes) == {
        "dots",
        "edges",
        "braille",
        "sliders",
        "orbit",
        "taps",
        "vibes",
        "twice",
        "quiet",
    }
    assert set(world.held_out) == {"gratings"}
    assert set(world.fixed) == {"braille_H", "wide_dot"}
    assert re.fullmatch(r"w-[0-9a-f]{12}", world.world_id)
    assert world.channels == ["value"]
    assert load_world(world) is world


def test_axes_bind_to_fields_and_constants_are_recorded():
    world = _world()
    dots = world.classes["dots"]
    assert dots.bindings["sigma_mm"] == ("shape", "sigma_mm")
    assert dots.bindings["x_mm"] == ("pattern", "x_mm")
    assert dots.bindings["hold_ms"] == ("episode", "hold_ms")
    assert dots.axes["slide_ms"].to_dict() == {"value": 0.0}
    assert dots.axes["contacts"].to_dict() == {"value": 1}
    assert world.classes["braille"].bindings["dots"] == ("pattern", "dots")
    assert world.classes["taps"].bindings["rate_hz"] == ("modulation", "rate_hz")
    assert set(world.classes["quiet"].axes) == {"quiet_ms"}


def test_world_defaults_skip_fields_a_class_lacks_or_fixes():
    def add(w):
        w["defaults"]["rate_hz"] = {"range": [1, 2]}
        w["defaults"]["sigma_mm"] = {"range": [0.2, 0.3]}

    world = _world(add)
    assert "rate_hz" not in world.classes["dots"].axes  # not modulated
    assert world.classes["taps"].axes["rate_hz"].hi == 80  # its own axis wins
    assert world.classes["dots"].axes["sigma_mm"].hi == 0.45  # its own axis wins
    assert "sigma_mm" not in world.classes["braille"].axes  # fixed in its layer
    assert "sigma_mm" not in world.classes["edges"].axes  # a bar has no sigma


def test_a_layer_fixed_amplitude_beats_the_world_default():
    def fix(w):
        w["classes"]["dots"]["layer"]["shape"]["amplitude"] = 2.0

    assert _world(fix).classes["dots"].axes["amplitude"].to_dict() == {"value": 2.0}


def test_ambiguous_names_need_a_dotted_path():
    def ambiguous(w):
        w["classes"]["edges"]["layer"]["pattern"] = {"kind": "random", "count": 3}

    with pytest.raises(ValueError, match="ambiguous.*shape.width_mm"):
        _world(ambiguous)

    def dotted(w):
        ambiguous(w)
        axes = w["classes"]["edges"]["axes"]
        axes["shape.width_mm"] = axes.pop("width_mm")

    edges = _world(dotted).classes["edges"]
    assert edges.bindings["shape.width_mm"] == ("shape", "width_mm")


def test_identity_ignores_the_description_but_not_values():
    base = _world().world_id
    assert _world(lambda w: w.update(description="other")).world_id == base
    widened = _world(
        lambda w: w["classes"]["dots"]["axes"]["sigma_mm"].update(range=[0.15, 0.46])
    )
    assert widened.world_id != base


def _never_touch(w):
    w["defaults"].update(touch_ms={"value": 0}, release_ms={"value": 0})
    w["classes"]["sliders"]["axes"].update(hold_ms={"value": 0}, slide_ms={"value": 0})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda w: w["held_out"]["gratings"].update(weight=1.0), "take no weight"),
        (lambda w: w["classes"]["dots"].update(kind="hologram"), "unknown class kind"),
        (
            lambda w: w["classes"]["dots"]["layer"].update(timing={"hold_ms": 5}),
            "episode axes",
        ),
        (
            lambda w: w["classes"]["dots"].update(channel="heat"),
            "not one of the world's channels",
        ),
        (_never_touch, "never touches"),
        (lambda w: w["fixed_draws"].update(bad={"class": "nope"}), "unknown class"),
        (
            lambda w: w["fixed_draws"].update(bad={"class": "dots", "radius_mm": 1}),
            "radius_mm",
        ),
        (
            lambda w: w["classes"].update(
                gratings={"layer": {"shape": {"kind": "grating"}}}
            ),
            "also classes",
        ),
        (
            lambda w: w["classes"]["dots"]["layer"]["shape"].update(colour=1),
            "has no fields",
        ),
        (
            lambda w: w["classes"]["dots"]["axes"].update(radius_mm={"range": [0, 1]}),
            "radius_mm.*known",
        ),
        (lambda w: w.update(colour="blue"), "unknown keys"),
        (
            lambda w: w["classes"]["orbit"]["layer"]["motion"].update(span="hold"),
            "drop 'span'",
        ),
        # Two names for one field would make the result depend on key order.
        (
            lambda w: w["classes"]["dots"]["axes"].update(
                {"shape.sigma_mm": {"range": [0.2, 0.3]}}
            ),
            "'shape.sigma_mm' and 'sigma_mm' both set shape.sigma_mm",
        ),
        (
            lambda w: w["defaults"].update({"pattern.x_mm": {"range": [0, 0.1]}}),
            "'pattern.x_mm' and 'x_mm' both set pattern.x_mm",
        ),
        (
            lambda w: w["fixed_draws"].update(
                bad={"class": "dots", "sigma_mm": 0.2, "shape.sigma_mm": 0.3}
            ),
            "'sigma_mm' is set twice",
        ),
    ],
)
def test_invalid_worlds_are_named(mutate, message):
    with pytest.raises(ValueError, match=message):
        _world(mutate)
