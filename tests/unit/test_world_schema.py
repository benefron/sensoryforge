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


# ------------------------------------------- field values checked at load time


def _axis(cls, name, spec, group="classes"):
    def mutate(w):
        w[group][cls].setdefault("axes", {})[name] = spec

    return mutate


def _fixed(**values):
    def mutate(w):
        w["fixed_draws"]["bad"] = values

    return mutate


def _never_touch_by_range(w):
    w["defaults"].update(touch_ms={"value": 0}, release_ms={"range": [0, 0]})
    w["classes"]["sliders"]["axes"].update(
        hold_ms={"range": [0.0, 0.0]}, slide_ms={"values": [0, 0.0]}
    )


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "mutate, message",
    [
        # B1: contacts is a whole number >= 1.
        (_axis("twice", "contacts", {"range": [1, 3]}), "twice.axes.contacts.*whole"),
        (_axis("twice", "contacts", {"value": 0}), "twice.axes.contacts"),
        (_axis("twice", "contacts", {"value": 1.5}), "twice.axes.contacts.*whole"),
        (_axis("twice", "contacts", {"value": 2.0}), "twice.axes.contacts.*whole"),
        (
            _axis("twice", "contacts", {"range": [0, 2], "int": True}),
            "twice.axes.contacts",
        ),
        (_axis("twice", "contacts", {"values": [1, 2.5]}), "twice.axes.contacts"),
        (_axis("twice", "contacts", {"values": [0, 1]}), "twice.axes.contacts"),
        (
            _axis("twice", "contacts", {"dist": "braille_cells"}),
            "twice.axes.contacts",
        ),
        # B2: every bound axis lies inside its field's domain.
        (
            _axis("dots", "sigma_mm", {"range": [0.0, 0.4]}),
            r"world\.classes\.dots\.axes\.sigma_mm.*domain",
        ),
        (_axis("dots", "hold_ms", {"range": [-5, 10]}), "dots.axes.hold_ms.*domain"),
        (
            _axis("edges", "orientation_deg", {"values": [0, 400]}),
            "edges.axes.orientation_deg.*domain",
        ),
        (_axis("dots", "amplitude", {"value": 2.0e4}), "dots.axes.amplitude"),
        (
            _axis("taps", "rate_hz", {"range": [20, 2.0e4], "dist": "log_uniform"}),
            "taps.axes.rate_hz",
        ),
        (
            _axis("edges", "profile", {"values": ["gaussian", "flatt"]}),
            "edges.axes.profile.*'flatt'",
        ),
        (
            lambda w: w["defaults"].update(delay_ms={"range": [-1, 5]}),
            r"world\.defaults\.delay_ms",
        ),
        (
            lambda w: w["classes"]["dots"]["layer"]["shape"].update(amplitude=-1.0),
            "dots.layer.*amplitude",
        ),
        (_axis("quiet", "quiet_ms", {"range": [-10, 20]}), "quiet.axes.quiet_ms"),
        # B3: a fixed draw may leave the range but not the domain.
        (_fixed(**{"class": "dots", "sigma_mm": 100.0}), "fixed_draws.bad.*sigma_mm"),
        (_fixed(**{"class": "twice", "contacts": 1.5}), "fixed_draws.bad.*contacts"),
        (_fixed(**{"class": "braille", "dots": 125}), "fixed_draws.bad.*dots.*text"),
        (_fixed(**{"class": "dots", "hold_ms": NAN}), "fixed_draws.bad.*hold_ms"),
        # B4: contact phases that can only be 0.
        (_never_touch_by_range, "never touches"),
        # B6: numeric fields need numbers; strings and bools need their type.
        (_axis("dots", "amplitude", {"value": "3e-1"}), "dots.axes.amplitude.*3.0e-1"),
        (_axis("dots", "sigma_mm", {"range": ["1e-1", 0.4]}), "sigma_mm.*number"),
        (_axis("dots", "hold_ms", {"values": ["10", "20"]}), "dots.axes.hold_ms"),
        (_axis("dots", "amplitude", {"value": True}), "dots.axes.amplitude"),
        (_axis("braille", "dots", {"values": [125, 14]}), "braille.axes.dots.*text"),
        (
            _axis("gratings", "signed", {"values": ["yes"]}, group="held_out"),
            "gratings.axes.signed.*true or false",
        ),
        (
            _axis("gratings", "signed", {"range": [0, 1]}, group="held_out"),
            "gratings.axes.signed",
        ),
        # B6: NaN and infinity, in ranges, constants, values and weights.
        (_axis("dots", "sigma_mm", {"range": [NAN, 0.4]}), "sigma_mm.*finite"),
        (_axis("dots", "hold_ms", {"value": INF}), "hold_ms.*finite"),
        (_axis("dots", "hold_ms", {"values": [10, NAN]}), "hold_ms.*finite"),
        (
            _axis("dots", "hold_ms", {"values": [10, 20], "weights": [1, NAN]}),
            "hold_ms.*weights.*finite",
        ),
        (lambda w: w["classes"]["dots"].update(weight=NAN), "dots.weight"),
        (lambda w: w["classes"]["dots"].update(weight="1"), "dots.weight"),
    ],
)
def test_field_values_are_checked_at_load(mutate, message):
    with pytest.raises(ValueError, match=message):
        _world(mutate)


@pytest.mark.parametrize(
    "spec", [{"value": 3}, {"range": [1, 3], "int": True}, {"values": [1, 2, 4]}]
)
def test_whole_contacts_load(spec):
    axis = _world(_axis("twice", "contacts", spec)).classes["twice"].axes["contacts"]
    assert axis.to_dict() == {
        **spec,
        **({"weights": [1.0] * 3} if "values" in spec else {}),
    }


def test_a_fixed_draw_outside_its_range_but_inside_its_domain_loads():
    world = _world(_fixed(**{"class": "dots", "sigma_mm": 49.0, "hold_ms": 0}))
    assert world.fixed_draw("bad").out_of_range == ("hold_ms", "sigma_mm")


def test_a_yaml_exponent_without_a_point_is_named(tmp_path):
    path = tmp_path / "w.yml"
    path.write_text(
        "world:\n"
        "  classes:\n"
        "    dots:\n"
        "      axes: {hold_ms: {value: 20}, amplitude: {value: 3e-1}}\n"
    )
    with pytest.raises(ValueError, match="amplitude: needs a number, got '3e-1'"):
        load_world(path)
    path.write_text(path.read_text().replace("3e-1", "3.0e-1"))
    assert load_world(path).classes["dots"].axes["amplitude"].value == 0.3


# ------------------------------------------- loading: YAML, mappings and names


def test_a_duplicated_class_key_in_yaml_is_refused(tmp_path):
    path = tmp_path / "w.yml"
    path.write_text(
        "world:\n"
        "  classes:\n"
        "    dots: {axes: {hold_ms: {value: 20}}}\n"
        "    dots: {axes: {hold_ms: {value: 30}}}\n"
    )
    with pytest.raises(ValueError, match="Duplicate key 'dots'"):
        load_world(path)


def _set(path, value):
    """A mutation that sets ``world[path[0]][path[1]]... = value``."""

    def mutate(w):
        target = w
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set(["classes"], ["dots", "edges"]), r"world\.classes: expected a mapping"),
        (_set(["held_out"], ["gratings"]), r"world\.held_out: expected a mapping"),
        (_set(["defaults"], [1, 2]), r"world\.defaults: expected a mapping"),
        (_set(["fixed_draws"], "braille_H"), r"world\.fixed_draws: expected a mapping"),
        (_set(["units"], ["mm", "ms"]), r"world\.units: expected a mapping"),
        (_set(["channels"], "value"), r"world\.channels: expected a list"),
        (_set(["classes", "dots"], 3), r"world\.classes\.dots: expected a mapping"),
        (
            _set(["classes", "dots", "axes"], ["sigma_mm"]),
            r"world\.classes\.dots\.axes: expected a mapping",
        ),
        (
            _set(["classes", "dots", "layer"], "gaussian"),
            r"world\.classes\.dots\.layer: expected a mapping",
        ),
        (
            _set(["classes", "dots", "layer", "shape"], [["kind", "gaussian"]]),
            r"world\.classes\.dots\.layer\.shape: expected a mapping",
        ),
        (
            _set(["classes", "braille", "layer", "pattern"], "braille"),
            r"world\.classes\.braille\.layer\.pattern: expected a mapping",
        ),
        (
            _set(["classes", "taps", "layer", "modulation"], 5),
            r"world\.classes\.taps\.layer\.modulation: expected a mapping",
        ),
        (
            _set(["classes", "orbit", "layer", "motion"], "circular"),
            r"world\.classes\.orbit\.layer\.motion: expected a mapping",
        ),
        (
            _set(["fixed_draws", "braille_H"], "braille"),
            r"world\.fixed_draws\.braille_H: expected a mapping",
        ),
        (
            _set(["classes", "dots", "axes", "sigma_mm"], [0.1, 0.2]),
            r"world\.classes\.dots\.axes\.sigma_mm: expected a mapping",
        ),
        # Names that become directories.
        (
            lambda w: w["classes"].update({"my dots": w["classes"].pop("dots")}),
            "'my dots'",
        ),
        (
            lambda w: w["classes"].update({"a/b": w["classes"].pop("dots")}),
            "'a/b'",
        ),
        (
            lambda w: w["held_out"].update({".hidden": w["held_out"].pop("gratings")}),
            "'.hidden'",
        ),
        (
            lambda w: w["fixed_draws"].update(
                {"../up": w["fixed_draws"].pop("wide_dot")}
            ),
            "'../up'",
        ),
    ],
)
def test_declarations_of_the_wrong_shape_are_named(mutate, message):
    with pytest.raises(ValueError, match=message):
        _world(mutate)


def test_names_may_use_letters_digits_and_dot_dash_underscore():
    def rename(w):
        w["classes"]["2nd-dots_v1.0"] = w["classes"].pop("dots")
        w["fixed_draws"]["wide_dot"]["class"] = "2nd-dots_v1.0"

    assert "2nd-dots_v1.0" in _world(rename).classes


def test_a_top_level_that_is_not_a_mapping_is_named():
    with pytest.raises(ValueError, match="a world is a mapping"):
        World.from_dict({"world": ["dots"]})
    with pytest.raises(ValueError, match="a world is a mapping"):
        World.from_dict(["dots"])
