"""``check_world_component``: each rule of each world registry kind, both ways.

A check that every shipped built-in passes (``tests/contract/
test_world_component_contracts.py``) proves nothing unless it also refuses a broken
registration, so each rule here has a registration that breaks it. Entries are put
in the registry tables under names of their own and the tables are restored after
each test; nothing here depends on how a repeated registration behaves.
"""

import numpy as np
import pytest
import torch

from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.testing.world_contracts import (
    available_world_kinds,
    check_world_component,
)
from sensoryforge.world import (
    ClassKind,
    distributions,
    kernel,
    kinds,
    register_distribution,
    register_modulation,
    register_pattern,
    register_shape,
)


@pytest.fixture(autouse=True)
def restore_registries():
    tables = (
        kernel.SHAPE_KINDS,
        kernel.PATTERN_KINDS,
        kernel.MODULATION_KINDS,
        distributions.DISTRIBUTIONS,
        kinds.CLASS_KINDS,
    )
    saved = [dict(table) for table in tables]
    yield
    for table, before in zip(tables, saved):
        table.clear()
        table.update(before)
    kernel._POSITION_CACHE.clear()


def spec(name, default, lo=None, hi=None, dtype="float", **kwargs):
    return ParamSpec(
        name, dtype=dtype, default=default, min_val=lo, max_val=hi, **kwargs
    )


AMPLITUDE = spec("amplitude", 1.0, 0.0, 1.0e4)
RADIUS = spec("radius_mm", 1.0, 0.0, 10.0, unit="mm")
WIDTH = spec("width_mm", 0.1, 0.001, 10.0, unit="mm")
RING_SPECS = [AMPLITUDE, RADIUS, WIDTH]


def ring(x, y, p):
    """The ring of docs/user_guide/worlds.md ("Extending")."""
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-((r - p["radius_mm"]) ** 2) / (2.0 * p["width_mm"] ** 2))


def put_shape(name, fn, specs, **kwargs):
    kernel.SHAPE_KINDS[name] = kernel.ShapeKind(fn=fn, specs=list(specs), **kwargs)


def fails(kind, name, match, example=None):
    with pytest.raises(AssertionError, match=match):
        check_world_component(kind, name, example)


# ---------------------------------------------------------------- the front door


def test_the_kinds_are_the_world_registries():
    assert available_world_kinds() == [
        "class_kind",
        "distribution",
        "modulation",
        "pattern",
        "shape",
    ]


def test_an_unknown_kind_or_name_is_a_value_error():
    with pytest.raises(ValueError, match="Unknown world kind 'sensor'"):
        check_world_component("sensor", "gaussian")
    for kind in available_world_kinds():
        with pytest.raises(ValueError, match="no .* 'not_registered'"):
            check_world_component(kind, "not_registered")


def test_the_documented_ring_shape_passes():
    register_shape("contract_ring", ring, RING_SPECS)
    check_world_component("shape", "contract_ring")


# ----------------------------------------------------------------------- shapes


def test_a_shape_without_an_amplitude_spec_fails():
    put_shape("contract_no_amp", ring, [RADIUS, WIDTH])
    fails("shape", "contract_no_amp", "must include 'amplitude'")


def test_a_shape_function_that_cannot_take_the_documented_arguments_fails():
    def reads_amplitude(x, y, p):  # the renderer takes amplitude out first
        return p["amplitude"] * ring(x, y, p)

    put_shape("contract_reads_amp", reads_amplitude, RING_SPECS)
    fails("shape", "contract_reads_amp", "fn.x, y, params. failed")


@pytest.mark.parametrize(
    "fn, match",
    [
        (lambda x, y, p: ring(x, y, p).sum(), "coordinates' shape"),
        (lambda x, y, p: ring(x, y, p).double(), "returned torch.float64"),
        (lambda x, y, p: list(ring(x, y, p)), "must return a tensor"),
        (lambda x, y, p: x / p["radius_mm"], "not finite.*radius_mm=0.0"),
    ],
    ids=["reduced", "dtype", "not_a_tensor", "not_finite_at_a_declared_end"],
)
def test_a_shape_that_misreturns_fails(fn, match):
    put_shape("contract_misreturn", fn, RING_SPECS)
    fails("shape", "contract_misreturn", match)


def test_a_shape_that_reduces_over_the_batch_is_not_batch_invariant():
    def batch_max(x, y, p):
        s = p["width_mm"].max()  # all draws then share the widest one's width
        return torch.exp(-(x**2 + y**2) / (2.0 * s**2))

    put_shape("contract_batch_max", batch_max, RING_SPECS)
    fails("shape", "contract_batch_max", "draw 0 alone differs|draw 1 alone differs")


def test_an_indenter_is_zero_at_depth_zero_and_has_no_depth_spec():
    def sphere(x, y, p):
        return torch.clamp(p["depth"] - (x**2 + y**2), min=0.0)

    put_shape("contract_sphere", sphere, [AMPLITUDE, RADIUS], indenter=True)
    check_world_component("shape", "contract_sphere")

    put_shape(
        "contract_always",
        lambda x, y, p: torch.ones_like(x),
        [AMPLITUDE],
        indenter=True,
    )
    fails("shape", "contract_always", "exactly 0 at depth 0")

    put_shape(
        "contract_depth_spec",
        sphere,
        [AMPLITUDE, spec("depth", 1.0, 0.0, 5.0)],
        indenter=True,
    )
    fails("shape", "contract_depth_spec", "cannot also have a spec named 'depth'")


def test_a_shape_whose_defaults_fail_its_own_check_fails():
    def refuse(values, path):
        raise ValueError(f"{path('radius_mm')}: no radius is allowed")

    put_shape("contract_refuses", ring, RING_SPECS, check=refuse)
    fails("shape", "contract_refuses", "defaults .* fail the shape's own check")


def test_a_shape_check_that_refuses_some_draws_is_respected():
    """A range end the shape's own check refuses is not a draw, so it is skipped."""

    def width_below_radius(values, path):
        if values["width_mm"].largest() >= values["radius_mm"].smallest():
            raise ValueError(f"{path('width_mm')}: must be below radius_mm")

    put_shape("contract_cross_field", ring, RING_SPECS, check=width_below_radius)
    check_world_component("shape", "contract_cross_field")


@pytest.mark.parametrize(
    "bad, match",
    [
        (spec("radius_mm", 20.0, 0.0, 10.0), "above max_val"),
        (spec("radius_mm", -1.0, 0.0, 10.0), "below min_val"),
        (spec("radius_mm", 2.5, 0, 10, dtype="int"), "is not a int"),
        (spec("radius_mm", 1, dtype="bool"), "is not a bool"),
        (spec("radius_mm", 1.0, 5.0, 2.0), "exceeds max_val"),
        (spec("radius_mm", "c", dtype="str", choices=["a", "b"]), "not one of choices"),
        (spec("radius_mm", {1, 2}), "is not a float"),
        (spec("radius_mm", float("nan"), 0.0, 10.0), "default .* is below|not finite"),
        (spec("radius_mm", 1.0, dtype="complex"), "dtype 'complex'"),
        (spec("kind", 1.0, 0.0, 10.0), "'kind' is the key"),
        (spec("width_mm", 0.1, 0.001, 10.0), "spec names .* repeated"),
    ],
    ids=[
        "above_max",
        "below_min",
        "int_default_float",
        "bool_default_int",
        "min_over_max",
        "choice_missing",
        "set_default",
        "nan_default",
        "dtype",
        "named_kind",
        "repeated_name",
    ],
)
def test_malformed_specs_fail(bad, match):
    put_shape("contract_bad_spec", ring, [AMPLITUDE, bad, WIDTH])
    fails("shape", "contract_bad_spec", match)


def test_a_non_json_default_fails():
    put_shape("contract_object", ring, [AMPLITUDE, spec("radius_mm", [object()])])
    fails("shape", "contract_object", "not finite JSON-plain")


# --------------------------------------------------------------------- patterns


def row(params):
    n = int(params["count"])
    return [(0.5 * i, 0.0) for i in range(n)], [1.0] * n


PLACED = [
    spec("count", 3, 1, 10, dtype="int"),
    spec("x_mm", 0.0, -10.0, 10.0, unit="mm"),
    spec("y_mm", 0.0, -10.0, 10.0, unit="mm"),
]


def test_a_placed_pattern_passes():
    register_pattern("contract_row", row, PLACED)
    check_world_component("pattern", "contract_row")
    register_pattern("contract_row_unplaced", row, PLACED[:1])
    check_world_component("pattern", "contract_row_unplaced")


def test_a_pattern_with_only_one_placement_coordinate_fails():
    register_pattern("contract_half_placed", row, PLACED[:2])
    fails("pattern", "contract_half_placed", "x_mm and y_mm place a pattern together")


def test_a_non_deterministic_pattern_fails():
    calls = []

    def drifting(params):
        calls.append(1)
        return [(0.01 * len(calls), 0.0)], [1.0]

    register_pattern("contract_drift", drifting, PLACED[:1])
    fails("pattern", "contract_drift", "pure function")


@pytest.mark.parametrize(
    "fn, match",
    [
        (lambda p: ([(0.0, 0.0)], [1.0, 2.0]), "1 positions but 2 scales"),
        (lambda p: ([(0.0, float("nan"))], [1.0]), "finite .x, y. in mm"),
        (lambda p: ([(0.0, 0.0, 0.0)], [1.0]), "finite .x, y. in mm"),
        (lambda p: ([(0.0, 0.0)], ["a"]), "scale must be a finite number"),
        (lambda p: [(0.0, 0.0)], "two lists"),
        (lambda p: (np.zeros((1, 2)), [1.0]), "two lists"),
        (lambda p: p["not_a_field"], "fn.params. failed"),
    ],
    ids=["lengths", "nan", "three_coordinates", "scale", "one_list", "array", "key"],
)
def test_a_pattern_that_misreturns_fails(fn, match):
    register_pattern("contract_misreturn_pattern", fn, PLACED[:1])
    fails("pattern", "contract_misreturn_pattern", match)


# ------------------------------------------------------------------ modulations

RATE = spec("rate_hz", 5.0, 0.001, 100.0, unit="Hz")
DEPTH = spec("depth", 1.0, 0.0, 1.0)


def beat(tc, p):
    phase = 2.0 * np.pi * p["rate_hz"] * tc / 1000.0
    return 1.0 - p["depth"] * 0.5 * (1.0 - torch.cos(phase))


def test_a_modulation_passes_and_no_modulation_needs_no_function():
    register_modulation("contract_beat", beat, [RATE, DEPTH])
    check_world_component("modulation", "contract_beat")
    register_modulation("contract_off", None, [])
    check_world_component("modulation", "contract_off")


@pytest.mark.parametrize(
    "fn, match",
    [
        (lambda tc, p: tc + 2.0, r"values in \[0, 1\]"),
        (lambda tc, p: beat(tc, p).sum(dim=1), "tc's shape"),
        (lambda tc, p: beat(tc, p) / (tc - tc), "not finite"),
        (lambda tc, p: beat(tc, p) * p["depth"].max(), "draw . alone differs"),
        (lambda tc, p: beat(tc, p).double(), "returned torch.float64"),
    ],
    ids=["range", "shape", "not_finite", "batch_max", "dtype"],
)
def test_a_modulation_that_misreturns_fails(fn, match):
    register_modulation("contract_misreturn_mod", fn, [RATE, DEPTH])
    fails("modulation", "contract_misreturn_mod", match)


def test_a_modulation_with_a_bad_function_or_spec_fails():
    kernel.MODULATION_KINDS["contract_not_callable"] = kernel.ModulationKind(
        fn="sine", specs=[]
    )
    fails("modulation", "contract_not_callable", "callable or None")
    register_modulation("contract_dup_spec", beat, [RATE, RATE])
    fails("modulation", "contract_dup_spec", "repeated")


# ---------------------------------------------------------------- distributions


def tenths(u, axis):
    return [int(v * 10) for v in u]


def require_scale(axis):
    if "scale" not in dict(axis.options):
        raise ValueError("needs a scale")


def test_a_distribution_passes_with_and_without_options():
    register_distribution("contract_tenths", tenths, lambda axis: list(range(10)))
    check_world_component("distribution", "contract_tenths")

    def scaled(u, axis):
        scale = dict(axis.options)["scale"]
        return [float(v * scale) for v in u]

    register_distribution(
        "contract_scaled",
        scaled,
        quantile=True,
        check=require_scale,
        bounds=lambda axis: (0.0, float(dict(axis.options)["scale"])),
    )
    check_world_component("distribution", "contract_scaled", {"scale": 3.0})
    # An example the distribution's own check refuses is reported as such.
    fails("distribution", "contract_scaled", "a world refuses the axis", {"bad": 1})
    # A check hook that raises anything but ValueError is reported, not propagated.
    distributions.DISTRIBUTIONS["contract_keyerror"] = distributions.Distribution(
        sample=scaled, check=lambda axis: dict(axis.options)["scale"]
    )
    fails("distribution", "contract_keyerror", "must raise ValueError", {"bad": 1})


@pytest.mark.parametrize(
    "sample, extra, match",
    [
        (lambda u, a: [(float(v), 1) for v in u], {}, "must be a str, int, float"),
        (lambda u, a: list(np.sort(u)), {}, "depends on the other draws"),
        (lambda u, a: [float("nan")] * len(u), {}, "must be finite"),
        (lambda u, a: [0.5], {}, "one value per u"),
        (lambda u, a: np.asarray(u), {}, "one value per u"),
        (tenths, {"support": lambda a: [0, 1, 2]}, "outside support"),
        (tenths, {"support": lambda a: []}, "non-empty list"),
        (
            lambda u, a: [float(v) for v in u],
            {"bounds": lambda a: (0.0, 0.5)},
            "outside bounds",
        ),
        (
            lambda u, a: [float(v) for v in u],
            {"bounds": lambda a: (1.0, 0.0)},
            "lo <= hi",
        ),
        (lambda u, a: ["a"] * len(u), {"quantile": True}, "quantile function"),
        (tenths, {"check": "yes"}, "check must be callable"),
    ],
    ids=[
        "tuple_value",
        "batch_dependent",
        "nan",
        "short",
        "array",
        "outside_support",
        "empty_support",
        "outside_bounds",
        "inverted_bounds",
        "quantile_of_text",
        "check_not_callable",
    ],
)
def test_a_distribution_that_breaks_a_rule_fails(sample, extra, match):
    distributions.DISTRIBUTIONS["contract_broken"] = distributions.Distribution(
        sample=sample, **extra
    )
    fails("distribution", "contract_broken", match)


def test_a_distribution_that_changes_between_calls_fails():
    calls = []

    def drifting(u, axis):
        calls.append(1)
        return [len(calls)] * len(u)

    register_distribution("contract_drifting", drifting)
    fails("distribution", "contract_drifting", "second call")


def test_a_distribution_example_must_be_a_mapping():
    register_distribution("contract_tenths_b", tenths)
    with pytest.raises(ValueError, match="mapping of its options"):
        check_world_component("distribution", "contract_tenths_b", [1, 2])


# ------------------------------------------------------------------ class kinds


def world(kind, **cls):
    return {
        "world": {
            "name": "contract",
            "modality": "tactile",
            "units": {"space": "mm", "time": "ms"},
            "defaults": {
                "touch_ms": {"range": [2, 4]},
                "hold_ms": {"range": [5, 9]},
                "release_ms": {"range": [2, 4]},
            },
            "classes": {
                "a": {
                    "kind": kind,
                    "layer": {"shape": {"kind": "gaussian"}},
                    "axes": {"sigma_mm": {"range": [0.1, 0.3]}},
                    **cls,
                }
            },
        }
    }


def put_kind(name, base=kinds.LayeredKind, **methods):
    cls = type(f"Contract{name.title()}", (base,), {"name": name, **methods})
    kinds.CLASS_KINDS[name] = cls()


def test_a_class_kind_that_extends_layered_passes():
    put_kind("contract_fine")
    check_world_component("class_kind", "contract_fine", world("contract_fine"))


def test_the_shipped_kinds_name_themselves_and_override_what_the_base_leaves():
    put_kind("contract_fine_b")
    kinds.CLASS_KINDS["other_key"] = kinds.CLASS_KINDS["contract_fine_b"]
    fails("class_kind", "other_key", "registered as 'other_key' but its name is")


def test_a_class_kind_must_override_the_methods_the_base_leaves_open():
    kinds.CLASS_KINDS["contract_bare"] = type(
        "Bare", (ClassKind,), {"name": "contract_bare"}
    )()
    fails("class_kind", "contract_bare", "does not override normalise_layer")
    kinds.CLASS_KINDS["contract_not_a_kind"] = object()
    fails("class_kind", "contract_not_a_kind", "must be a ClassKind instance")


def test_a_class_kind_needs_an_example_world_with_a_class_of_its_kind():
    put_kind("contract_fine_c")
    with pytest.raises(ValueError, match="example world"):
        check_world_component("class_kind", "contract_fine_c")
    with pytest.raises(ValueError, match="no class of kind 'contract_fine_c'"):
        check_world_component("class_kind", "contract_fine_c", world("layered"))
    fails(
        "class_kind",
        "contract_fine_c",
        "example world does not load",
        {"world": {"classes": 3}},
    )


def leaky_render(self, spec, draws, X, Y, times):
    out = kinds.LayeredKind.render_group(self, spec, draws, X, Y, times)
    return out + 0.5


def orphan_defaults(self, layer):
    return {**kinds.LayeredKind.builtin_defaults(self, layer), "orphan": 1.0}


def drifting_layer(self, layer, where):
    out = kinds.LayeredKind.normalise_layer(self, layer, where)
    out["background"] = out.get("background", 0.0) + 1.0
    return out


@pytest.mark.parametrize(
    "methods, match",
    [
        ({"render_group": leaky_render}, "exactly 0 before 0"),
        ({"end_ms": lambda self, values: -1.0}, "end_ms must be a finite number"),
        ({"timeline": lambda self, values: [["touch", 1.0, 1.0]]}, "timeline must be"),
        (
            {"timeline": lambda self, values: [["touch", 0.0, 10_000.0]]},
            "timeline must be",
        ),
        ({"resolve": lambda self, name, layer: name}, "does not load"),
        ({"builtin_defaults": orphan_defaults}, "does not load.*'orphan'"),
        ({"normalise_layer": drifting_layer}, "not a fixed point"),
        ({"domain": lambda self, ref, layer: 5}, "does not load"),
        ({"field_info": lambda self, ref, layer: None}, "does not load"),
        ({"end_ms": lambda self, values: values["not_a_field"]}, "end_ms.* raised"),
    ],
    ids=[
        "leaks_frames",
        "negative_end",
        "zero_length_phase",
        "phase_past_the_end",
        "resolve_returns_text",
        "default_that_does_not_resolve",
        "layer_changes_when_normalised_again",
        "domain_not_a_pair",
        "field_info_none",
        "end_ms_raises",
    ],
)
def test_a_class_kind_that_breaks_a_rule_fails(methods, match):
    put_kind("contract_broken_kind", **methods)
    fails(
        "class_kind",
        "contract_broken_kind",
        match,
        world("contract_broken_kind"),
    )
