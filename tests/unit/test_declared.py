"""The declared stage (roadmap A3, report contract section 8).

The rules of the tactile adapter are checked on a small world whose every class is
constant, so each declared number can be read off the world. The shipped test worlds are
re-derived independently from the manifest alone, and the shape-based values are checked
against the world's own renderer on a fine canvas.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sensoryforge.evaluate.declared import (
    DECLARED_ADAPTER_REGISTRY,
    DeclaredAdapter,
    choose_adapter,
    run_declared,
)
from sensoryforge.evaluate.declared.__main__ import main as declared_main
from sensoryforge.evaluate.tactile.declared import TactileAdapter, contact_intervals
from sensoryforge.world import load_world
from sensoryforge.world.dataset import build_dataset, load_dataset, write_dataset

REPO = Path(__file__).resolve().parents[2]
WORLDS = REPO / "tests" / "fixtures" / "worlds"
STAMP = "2026-10-22T14:03:11Z"

# Every class is constant, so a declared number is readable off this text.
RULES = """
world:
  name: rules
  modality: tactile
  units: {space: mm, time: ms}
  defaults:
    delay_ms: {value: 10}
    touch_ms: {value: 5}
    hold_ms: {value: 20}
    release_ms: {value: 5}
    speed_mm_per_ms: {value: 0.01}
    direction_deg: {value: 30}
    amplitude: {value: 0.8}
    x_mm: {value: 0.2}
    y_mm: {value: -0.1}
  classes:
    dot: {layer: {shape: {kind: gaussian, sigma_mm: 0.3}}}
    slide:
      layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
      axes: {slide_ms: {value: 20}}
    vibe:
      layer:
        shape: {kind: gaussian, sigma_mm: 0.3}
        modulation: {kind: sine, depth: 0.3}
      axes: {frequency_hz: {value: 100}}
    taps:
      layer:
        shape: {kind: gaussian, sigma_mm: 0.3}
        modulation: {kind: pulses, duty: 0.5, edge_ms: 1}
      axes: {rate_hz: {value: 40}}
    braille:
      layer:
        shape: {kind: gaussian, sigma_mm: 0.15}
        pattern: {kind: braille, dot_spacing_mm: 0.35}
      axes: {dots: {value: "125"}}
    grating:
      layer: {shape: {kind: grating}}
      axes: {wavelength_mm: {value: 0.5}, orientation_deg: {value: 0}}
    base: {layer: {shape: {kind: gaussian, sigma_mm: 0.3}, background: 0.1}}
    twice:
      layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
      axes: {contacts: {value: 2}, pause_ms: {value: 10}}
    orbit:
      layer:
        shape: {kind: gaussian, sigma_mm: 0.3}
        motion: {kind: circular, radius_mm: 0.2, revolutions: 1}
      axes: {slide_ms: {value: 20}}
    still: {kind: quiet, axes: {quiet_ms: {value: 30}}}
  fixed_draws:
    f_dot: {class: dot}
    f_slide: {class: slide}
    f_vibe: {class: vibe}
    f_taps: {class: taps}
    f_braille: {class: braille}
    f_grating: {class: grating}
    f_base: {class: base}
    f_twice: {class: twice}
    f_orbit: {class: orbit}
    f_still: {class: still}
"""


@pytest.fixture(scope="module")
def rules():
    return load_world(yaml.safe_load(RULES))


def _declare(world, name, duration=200.0):
    draw = world.fixed_draw(name)
    result = TactileAdapter().declare(
        {"draw": draw.to_dict(), "duration_ms": duration}, world
    )
    values = {v["id"]: v["value"] for v in result.values}
    omitted = dict(result.omitted)
    return values, omitted


def _short(mapping):
    return {k.split(".", 1)[1]: v for k, v in mapping.items()}


# ------------------------------------------------------------------- intervals


def _draw(*phases):
    return SimpleNamespace(timeline=[list(p) for p in phases])


def test_contact_intervals_join_phases_that_touch_and_drop_quiet_and_pauses():
    d = _draw(
        ("quiet", 0, 5),
        ("touch", 5, 8),
        ("hold", 8, 20),
        ("release", 20, 25),
        ("pause", 25, 30),
        ("touch", 30, 33),
    )
    assert contact_intervals([(0.0, d)], 100.0) == [(5, 25), (30, 33)]


def test_contact_intervals_are_clipped_to_the_duration():
    d = _draw(("quiet", 0, 5), ("touch", 5, 8), ("hold", 8, 20))
    assert contact_intervals([(0.0, d)], 12.0) == [(5, 12)]
    assert contact_intervals([(0.0, d)], 5.0) == []


def test_session_items_are_placed_at_their_start_and_merge_where_they_touch():
    a = _draw(("touch", 0, 10))
    b = _draw(("quiet", 0, 4), ("hold", 4, 9))
    out = contact_intervals([(0.0, a), (10.0, b), (40.0, a)], 45.0)
    assert out == [(0, 10), (14, 19), (40, 45)]
    # an item that starts exactly where the last one ends joins it
    assert contact_intervals([(0.0, a), (10.0, a)], 50.0) == [(0, 20)]


# ---------------------------------------------------------------- the rules


def test_a_still_unmodulated_dot_is_declared_in_full(rules):
    values, omitted = _declare(rules, "f_dot")
    assert _short(values) == {
        "frames.duration": 200.0,
        "frames.onset_time": 10.0,
        "frames.offset_time": 40.0,
        "frames.active_time": 30.0,
        "frames.active_fraction": 0.15,
        "frames.bouts": 1,
        "frames.peak": 0.8,
        "frames.centroid_x": 0.2,
        "frames.centroid_y": -0.1,
        "episode.rise_time": 5.0,
        "episode.plateau_time": 20.0,
        "episode.fall_time": 5.0,
    }
    assert _short(omitted) == {
        "frames.gap_mean": "fewer than two bouts",
        "frames.gap_max": "fewer than two bouts",
        "frames.direction": "no motion",
        "motion.speed": "no motion",
        "frames.dominant_frequency": "no modulation",
    }


def test_values_follow_the_catalogue_order(rules):
    values, omitted = _declare(rules, "f_dot")
    catalogue = [
        m["id"] for m in TactileAdapter().metrics(TactileAdapter().units(rules))
    ]
    assert list(values) == [i for i in catalogue if i in values]
    assert sorted({*values, *omitted}) == sorted(catalogue)


def test_a_slide_declares_direction_and_speed_but_not_a_centroid(rules):
    values, omitted = _declare(rules, "f_slide")
    v, o = _short(values), _short(omitted)
    assert v["frames.direction"] == 30.0
    assert v["motion.speed"] == 0.01
    assert v["episode.plateau_time"] == 40.0  # hold + slide
    assert v["frames.offset_time"] == 60.0
    assert o["frames.centroid_x"] == o["frames.centroid_y"] == "motion"


def test_a_circular_motion_has_no_direction_or_speed(rules):
    _values, omitted = _declare(rules, "f_orbit")
    o = _short(omitted)
    assert o["frames.direction"] == o["motion.speed"] == "motion not linear"
    assert o["frames.centroid_x"] == "motion"


def test_modulation_gives_the_frequency_and_leaves_out_level_and_phases(rules):
    vibe, vibe_omitted = _declare(rules, "f_vibe")
    taps, _ = _declare(rules, "f_taps")
    assert vibe["core.frames.dominant_frequency"] == pytest.approx(
        0.1
    )  # 100 Hz in 1/ms
    assert taps["core.frames.dominant_frequency"] == pytest.approx(0.04)
    for gone in ("frames.peak", "episode.rise_time", "episode.plateau_time"):
        assert _short(vibe_omitted)[gone] == "modulated"
    assert vibe["core.frames.centroid_x"] == 0.2  # modulation does not move it
    assert vibe["core.frames.onset_time"] == 10.0  # nor does it change the timeline


def test_several_shapes_leave_out_peak_and_centroid(rules):
    _values, omitted = _declare(rules, "f_braille")
    o = _short(omitted)
    assert o["frames.peak"] == o["frames.centroid_x"] == "several shapes"


def test_an_unbounded_shape_has_a_peak_but_no_position(rules):
    values, omitted = _declare(rules, "f_grating")
    assert values["core.frames.peak"] == 0.8
    o = _short(omitted)
    assert o["frames.centroid_x"] == o["frames.direction"] == "unbounded shape"


def test_a_background_adds_to_the_peak_and_spoils_the_centroid(rules):
    values, omitted = _declare(rules, "f_base")
    assert values["core.frames.peak"] == pytest.approx(0.9)
    assert _short(omitted)["frames.centroid_x"] == "background"


def test_two_contacts_give_two_bouts_and_the_gap_between(rules):
    values, _ = _declare(rules, "f_twice")
    assert values["core.frames.bouts"] == 2
    assert values["core.frames.gap_mean"] == values["core.frames.gap_max"] == 10.0
    assert values["core.frames.active_time"] == 60.0
    assert values["core.frames.offset_time"] == 80.0
    assert values["tactile.episode.plateau_time"] == 20.0  # per contact


def test_a_quiet_draw_has_no_onset_but_declares_no_activity(rules):
    values, omitted = _declare(rules, "f_still")
    assert values["core.frames.active_time"] == 0.0
    assert values["core.frames.bouts"] == 0
    assert _short(omitted)["frames.onset_time"] == "no contact"
    assert _short(omitted)["frames.peak"] == "no contact"


def test_a_short_duration_clips_the_timeline_and_the_phases(rules):
    values, omitted = _declare(rules, "f_dot", duration=25.0)
    assert values["core.frames.offset_time"] == 25.0
    assert values["core.frames.active_time"] == 15.0
    assert _short(omitted)["episode.fall_time"] == "truncated"
    values, omitted = _declare(rules, "f_dot", duration=12.0)
    assert _short(omitted)["frames.peak"] == "peak not reached"
    values, omitted = _declare(rules, "f_dot", duration=5.0)
    assert values["core.frames.bouts"] == 0
    assert _short(omitted)["frames.onset_time"] == "no contact"


def test_contacts_that_merge_leave_out_the_phases(rules):
    raw = yaml.safe_load(RULES)
    raw["world"]["classes"]["twice"]["axes"]["pause_ms"] = {"value": 0}
    world = load_world(raw)
    values, omitted = _declare(world, "f_twice")
    assert values["core.frames.bouts"] == 1
    assert _short(omitted)["episode.rise_time"] == "contacts merge"


def test_a_session_declares_its_timeline_and_nothing_about_shape(rules):
    from sensoryforge.world.sampling import Session

    draw = rules.fixed_draw("f_dot")
    sess = Session(
        world=rules,
        seed=0,
        index=0,
        session_seed=0,
        duration_ms=300.0,
        items=((0.0, draw), (100.0, draw)),
    )
    result = TactileAdapter().declare(
        {"draw": sess.to_dict(), "duration_ms": 300.0}, rules
    )
    values = {v["id"]: v["value"] for v in result.values}
    omitted = dict(result.omitted)
    assert values["core.frames.bouts"] == 2
    assert values["core.frames.onset_time"] == 10.0
    assert values["core.frames.gap_mean"] == 70.0
    assert values["core.frames.active_time"] == 60.0
    assert omitted["core.frames.peak"] == "session"
    assert omitted["tactile.episode.rise_time"] == "session"


def test_a_two_channel_world_declares_each_channel_on_its_own():
    raw = yaml.safe_load(RULES)
    raw["world"]["channels"] = ["a", "b"]
    raw["world"]["classes"]["slide"]["channel"] = "b"
    world = load_world(raw)
    adapter = TactileAdapter()
    assert adapter.channels(world) == 2
    result = adapter.declare(
        {"draw": world.fixed_draw("f_slide").to_dict(), "duration_ms": 200.0}, world
    )
    by = {(v["id"], v["channel"]): v["value"] for v in result.values}
    assert by[("core.frames.bouts", 0)] == 0  # channel a stays at 0
    assert by[("core.frames.bouts", 1)] == 1
    assert by[("core.frames.active_time", 0)] == 0.0
    assert ("core.frames.onset_time", 1) in by and (
        "core.frames.onset_time",
        0,
    ) not in by
    assert ("core.frames.peak", 0) not in by  # nothing touches channel a


# ------------------------------------------------------ the adapter as a component


def test_the_adapter_is_registered_and_round_trips():
    assert DECLARED_ADAPTER_REGISTRY.is_registered("tactile")
    cls = DECLARED_ADAPTER_REGISTRY.get_class("tactile")
    assert cls is TactileAdapter and issubclass(cls, DeclaredAdapter)
    assert cls.from_config(cls().to_dict()).to_dict() == cls().to_dict() == {}
    assert cls.get_param_spec() == []


def test_the_metric_catalogue_carries_the_audits_units(rules):
    adapter = TactileAdapter()
    units = adapter.units(rules)
    assert set(units) == {"time", "space", "frame", "drive", "filtered", "state"}
    assert units["time"]["symbol"] == "ms" and units["frame"] is None
    by = {m["id"]: m for m in adapter.metrics(units)}
    assert by["core.frames.onset_time"]["unit"]["symbol"] == "ms"
    assert by["tactile.motion.speed"]["unit"]["symbol"] == "mm/ms"
    assert by["core.frames.dominant_frequency"]["unit"]["symbol"] == "1/ms"
    assert by["core.frames.direction"]["unit"]["symbol"] == "deg"
    assert by["core.frames.peak"]["unit"] is None  # a world's amplitude is unitless
    assert all(m["basis"] and m["source"] for m in by.values())


def test_the_space_unit_a_world_names_reaches_the_declared_units():
    raw = yaml.safe_load(RULES)
    raw["world"]["units"] = {"space": "px", "time": "ms"}
    world = load_world(raw)
    adapter = TactileAdapter()
    by = {m["id"]: m for m in adapter.metrics(adapter.units(world))}
    assert by["tactile.motion.speed"]["unit"]["symbol"] == "px/ms"
    assert by["core.frames.centroid_x"]["unit"]["symbol"] == "px"


# ----------------------------------------------------------- shipped worlds


@pytest.fixture(
    scope="module", params=["tactile_small", "elements_v1_2", "every_element_v1_1"]
)
def shipped(request, tmp_path_factory):
    """A data set built from one of the example worlds, and its stage run."""
    root = tmp_path_factory.mktemp(request.param)
    world = load_world(WORLDS / f"{request.param}.yml")
    splits = {"train": {"n": 40, "noise_repeats": 2}}
    if world.held_out:
        splits["held_out"] = {"stratified": {"bins": 1, "per_bin": 1}}
    if world.sessions is not None:
        splits["sessions"] = {"n": 2, "duration_ms": 400}
    raw = {
        "dataset": {
            "name": "t",
            "world": str(WORLDS / f"{request.param}.yml"),
            "seed": 3,
            "duration_ms": 150,
            "splits": splits,
        }
    }
    path = root / "spec.yml"
    path.write_text(yaml.safe_dump(raw))
    spec = load_dataset(path)
    write_dataset(spec, build_dataset(spec), root / "data")
    declared = run_declared(root / "data", root / "out", created_at=STAMP)
    return root, declared


def _manifest(root):
    lines = (root / "data" / "manifest.jsonl").read_text().splitlines()
    return sorted((json.loads(x) for x in lines), key=lambda r: r["entry"])


def _rows(root):
    lines = (root / "out" / "declared_entries.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def _sweep(row):
    """Contact intervals by an event sweep over the manifest's own timeline records."""
    record, duration = row["draw"], row["duration_ms"]
    if record.get("sampling") == "session":
        placed = [(s, d["timeline"]) for s, d in record["items"]]
    else:
        placed = [(0.0, record["timeline"])]
    events = []
    for start, timeline in placed:
        for phase, a, b in timeline:
            if phase in ("touch", "hold", "slide", "release"):
                lo, hi = min(start + a, duration), min(start + b, duration)
                if hi > lo:
                    events += [(lo, 0), (hi, 1)]  # a start sorts before an end at a tie
    spans, depth, begin = [], 0, 0.0
    for t, kind in sorted(events):
        if kind == 0:
            if depth == 0:
                begin = t
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                spans.append((begin, t))
    return spans


def test_the_timeline_values_are_rederived_from_the_manifest_alone(shipped):
    root, _declared = shipped
    rows = _rows(root)
    manifest = _manifest(root)
    assert [r["entry"] for r in rows] == [m["entry"] for m in manifest]
    for out, row in zip(rows, manifest):
        v = {x["id"]: x["value"] for x in out["values"]}
        spans = _sweep(row)
        total = sum(b - a for a, b in spans)
        assert v["core.frames.duration"] == row["duration_ms"]
        assert v["core.frames.bouts"] == len(spans)
        assert v["core.frames.active_time"] == pytest.approx(total, abs=1e-9)
        assert v["core.frames.active_fraction"] == pytest.approx(
            total / row["duration_ms"], abs=1e-12
        )
        if spans:
            assert v["core.frames.onset_time"] == pytest.approx(spans[0][0], abs=1e-9)
            assert v["core.frames.offset_time"] == pytest.approx(spans[-1][1], abs=1e-9)
        else:
            assert "core.frames.onset_time" not in v
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        if gaps:
            assert v["core.frames.gap_mean"] == pytest.approx(np.mean(gaps), abs=1e-9)
            assert v["core.frames.gap_max"] == pytest.approx(max(gaps), abs=1e-9)
        else:
            assert "core.frames.gap_mean" not in v


def test_entries_that_share_a_draw_get_the_same_values(shipped):
    root, _ = shipped
    by_base = {}
    for row in _rows(root):
        base = row["entry"].rsplit(".n", 1)[0]
        by_base.setdefault(base, []).append(row["values"])
    shared = [v for v in by_base.values() if len(v) > 1]
    assert shared, "the data set should hold noise repeats"
    for group in shared:
        assert all(g == group[0] for g in group)


def test_the_files_validate_against_the_schemas_and_the_rows_are_hashed(shipped):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    root, declared = shipped
    schemas = REPO / "schemas" / "evaluate"
    docs = [json.loads(p.read_text()) for p in sorted(schemas.glob("*.schema.json"))]
    registry = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs]
    )
    base = docs[0]["$id"].rsplit("/", 1)[0] + "/declared.schema.json"
    top = jsonschema.Draft202012Validator({"$ref": base}, registry=registry)
    top.validate(json.loads((root / "out" / "declared.json").read_text()))
    row = jsonschema.Draft202012Validator(
        {"$ref": base + "#/$defs/entry_row"}, registry=registry
    )
    for r in _rows(root):
        row.validate(r)
    import hashlib

    digest = hashlib.sha256((root / "out" / "declared_entries.jsonl").read_bytes())
    assert declared["entries"]["sha256"] == digest.hexdigest()
    assert declared["entries"]["n"] == len(_manifest(root))


def test_summaries_are_the_audits_own_over_the_declared_rows(shipped):
    root, declared = shipped
    rows = _rows(root)
    assert declared["dataset"]["n_entries"] == len(rows)
    assert declared["dataset"]["n_failed"] == 0
    summary = {s["id"]: s for s in declared["dataset"]["summaries"]}
    ids = [m["id"] for m in declared["metrics"]]
    assert list(summary) == ids
    durations = summary["core.frames.duration"]
    assert durations["n"] == len(rows) and durations["n_null"] == 0
    assert durations["mean"] == pytest.approx(
        np.mean([_manifest(root)[i]["duration_ms"] for i in range(len(rows))])
    )
    active = [
        v["value"]
        for r in rows
        for v in r["values"]
        if v["id"] == "core.frames.active_time"
    ]
    assert summary["core.frames.active_time"]["mean"] == pytest.approx(np.mean(active))
    # a left-out value is a null in the summary, and the count names its reason
    onset = summary["core.frames.onset_time"]
    assert onset["n"] + onset["n_null"] == len(rows)
    gone = declared["omitted"].get("core.frames.onset_time", {})
    assert onset["n_null"] == sum(gone.values())
    assert set(declared["groupings"]) == {"class", "split"}


def test_two_runs_differ_only_in_created_at(shipped, tmp_path):
    root, declared = shipped
    again = run_declared(root / "data", tmp_path, created_at="2030-01-01T00:00:00Z")
    assert {**again, "created_at": STAMP, "inputs": declared["inputs"]} == {
        **declared,
        "inputs": declared["inputs"],
    }
    first = (root / "out" / "declared_entries.jsonl").read_bytes()
    assert (tmp_path / "declared_entries.jsonl").read_bytes() == first


# ----------------------------------------------- shape values against the renderer


def test_declared_peak_and_centroid_equal_what_the_renderer_draws(tmp_path):
    """Render each eligible draw on a fine canvas and compare with the declaration."""
    import torch

    from sensoryforge.world import Canvas, render

    raw = {
        "dataset": {
            "name": "t",
            "world": str(WORLDS / "tactile_small.yml"),
            "seed": 11,
            "duration_ms": 150,
            "splits": {"train": {"n": 60}},
        }
    }
    path = tmp_path / "spec.yml"
    path.write_text(yaml.safe_dump(raw))
    spec = load_dataset(path)
    write_dataset(spec, build_dataset(spec), tmp_path / "data")
    run_declared(tmp_path / "data", tmp_path / "out")
    rows = {
        json.loads(x)["entry"]: json.loads(x)
        for x in (tmp_path / "out" / "declared_entries.jsonl").read_text().splitlines()
    }
    canvas = Canvas.from_grid(401, 401, 0.01)  # +-2 mm, finer than any shape
    xs = canvas.xx.numpy().astype(np.float64)
    ys = canvas.yy.numpy().astype(np.float64)
    checked_peak = checked_centroid = 0
    for line in (tmp_path / "data" / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        v = {x["id"]: x["value"] for x in rows[entry["entry"]]["values"]}
        peak, cx = v.get("core.frames.peak"), v.get("core.frames.centroid_x")
        if peak is None and cx is None:
            continue
        from sensoryforge.world.sampling import Draw

        draw = Draw.from_dict(entry["draw"], spec.world)
        lo, hi = next((a, b) for p, a, b in draw.timeline if p == "hold")
        times = torch.linspace(lo, hi, 25, dtype=torch.float64)
        frames = render([draw], canvas, times, dtype=torch.float64)[0].numpy()
        frame = frames[np.abs(frames).sum(axis=(1, 2)).argmax()]
        if peak is not None:
            assert frame.max() == pytest.approx(peak, rel=5e-3), entry["entry"]
            checked_peak += 1
        if cx is not None:
            w = np.abs(frame)
            assert (w * xs).sum() / w.sum() == pytest.approx(cx, abs=0.01), entry[
                "entry"
            ]
            assert (w * ys).sum() / w.sum() == pytest.approx(
                v["core.frames.centroid_y"], abs=0.01
            )
            checked_centroid += 1
    assert checked_peak >= 10 and checked_centroid >= 10


# --------------------------------------------------------------- the stage and CLI


def test_bounds_state_the_axis_ranges_they_come_from(tmp_path):
    raw = {
        "dataset": {
            "name": "t",
            "world": str(WORLDS / "tactile_small.yml"),
            "seed": 1,
            "duration_ms": 100,
            "splits": {"train": {"n": 4}},
        }
    }
    (tmp_path / "s.yml").write_text(yaml.safe_dump(raw))
    spec = load_dataset(tmp_path / "s.yml")
    write_dataset(spec, build_dataset(spec), tmp_path / "d")
    declared = run_declared(tmp_path / "d", tmp_path / "o")
    bounds = {(b["label"], b["id"]): b for b in declared["bounds"]}
    dots_x = bounds[("dots", "core.frames.centroid_x")]
    assert (dots_x["lo"], dots_x["hi"], dots_x["source"]) == (-0.3, 0.3, "axis x_mm")
    peak = bounds[("dots", "core.frames.peak")]
    assert (peak["lo"], peak["hi"]) == (0.5, 1.0)
    onset = bounds[("braille", "core.frames.onset_time")]
    assert (onset["lo"], onset["hi"]) == (0.0, 20.0)
    assert ("sliders", "core.frames.centroid_x") not in bounds  # it slides
    assert ("taps", "core.frames.peak") not in bounds  # it is modulated
    assert [b["grouping"] for b in declared["bounds"]] == ["class"] * len(
        declared["bounds"]
    )


def test_choose_adapter_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="no registered declared adapter"):
        choose_adapter({"world": {"format": "someone-else/1"}})
    with pytest.raises(ValueError, match="does not accept"):
        choose_adapter({}, name="tactile")
    with pytest.raises(KeyError):
        choose_adapter({}, name="nope")


def test_a_world_that_does_not_match_its_id_is_refused(tmp_path):
    raw = {
        "dataset": {
            "name": "t",
            "world": str(WORLDS / "tactile_small.yml"),
            "seed": 1,
            "duration_ms": 100,
            "splits": {"train": {"n": 2}},
        }
    }
    (tmp_path / "s.yml").write_text(yaml.safe_dump(raw))
    spec = load_dataset(tmp_path / "s.yml")
    write_dataset(spec, build_dataset(spec), tmp_path / "d")
    meta = json.loads((tmp_path / "d" / "dataset.json").read_text())
    meta["world_id"] = "w-not-this-one"
    (tmp_path / "d" / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hashes to"):
        run_declared(tmp_path / "d", tmp_path / "o")


def test_the_command_line_writes_the_files_and_reports_errors(
    shipped, capsys, tmp_path
):
    root, _ = shipped
    code = declared_main([str(root / "data"), "--out", str(tmp_path / "cli")])
    assert code == 0
    assert (tmp_path / "cli" / "declared.json").is_file()
    assert "values left out" in capsys.readouterr().out
    assert declared_main([str(tmp_path / "missing")]) == 2
    assert declared_main([str(root / "data"), "--adapter", "nope"]) == 2
    assert declared_main([str(root / "data"), "--param", "oops"]) == 2
    assert "declared:" in capsys.readouterr().err


def test_the_registry_and_audit_stay_free_of_the_world_package():
    """Importing the seam registry loads neither the world nor any adapter."""
    code = (
        "import sys\n"
        "import sensoryforge.evaluate.registry as r\n"
        "import sensoryforge.evaluate.audit.run\n"
        "assert not r.DECLARED_ADAPTER_REGISTRY.list_registered()\n"
        "assert 'sensoryforge.world' not in sys.modules\n"
        "assert 'sensoryforge.evaluate.declared' not in sys.modules\n"
        "import sensoryforge.evaluate.declared\n"
        "assert r.DECLARED_ADAPTER_REGISTRY.list_registered() == ['tactile']\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
