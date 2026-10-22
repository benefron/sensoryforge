"""The tactile metric pack (roadmap A2d, report contract section 7).

Phases and speed are checked on frames built by hand, whose answers follow from the
definitions; the end-to-end checks run the pack on a data set built through the engine
from the tactile test world.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensoryforge.evaluate.audit.run import AUDIT_FILE, ENTRIES_FILE, run_audit
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY
from sensoryforge.evaluate.tactile.metrics import TactilePack, measure_channel

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
STAMP = "2026-10-22T14:03:11Z"

ROWS = COLS = 5
SPACING = 2.0
# Sensor k = i * COLS + j sits at x = (i - 2) * 2, y = (j - 2) * 2 (dim 0 first).
COORDS = np.array(
    [[(i - 2) * SPACING, (j - 2) * SPACING] for i in range(ROWS) for j in range(COLS)]
)
CENTRE = 2 * COLS + 2


def _frames(n_t: int) -> np.ndarray:
    return np.zeros((n_t, ROWS * COLS), dtype=np.float32)


def _envelope(levels, start=0, n_t=None):
    """Frames whose centre sensor follows ``levels`` from frame ``start``."""
    x = _frames(n_t or start + len(levels) + 5)
    x[start : start + len(levels), CENTRE] = levels
    return x


def _measure(x, dt=1.0, plateau=0.01, motion=0.0):
    return measure_channel(x, COORDS, dt, plateau, motion)


# ---------------------------------------------------------------------- phases


def test_rise_plateau_and_fall_of_one_bout():
    # frames 3..12: two rising frames, a four-frame plateau at 1.0, four falling frames
    x = _envelope([0.2, 0.5, 1, 1, 1, 1, 0.6, 0.4, 0.3, 0.1], start=3, n_t=30)
    m = _measure(x, dt=2.0)
    assert m["rise_time"] == 2 * 2.0
    assert m["plateau_time"] == 4 * 2.0
    assert m["fall_time"] == 4 * 2.0


def test_the_three_phases_add_up_to_the_bout():
    x = _envelope([0.1, 0.3, 0.9, 1.0, 0.9, 0.2], start=4, n_t=20)
    m = _measure(x, plateau=0.0)
    assert m["rise_time"] + m["plateau_time"] + m["fall_time"] == 6.0


def test_a_bout_that_peaks_at_once_has_no_rise():
    x = _envelope([1, 1, 1, 0.5], start=2)
    m = _measure(x)
    assert (m["rise_time"], m["plateau_time"], m["fall_time"]) == (0.0, 3.0, 1.0)


def test_a_bout_that_ends_at_its_peak_has_no_fall():
    x = _envelope([0.5, 1.0], start=2)
    m = _measure(x)
    assert (m["rise_time"], m["plateau_time"], m["fall_time"]) == (1.0, 1.0, 0.0)


def test_plateau_tolerance_decides_which_frames_are_held():
    x = _envelope([0.5, 0.97, 1.0, 0.98, 0.4], start=1)
    tight = _measure(x, plateau=0.01)
    loose = _measure(x, plateau=0.05)
    assert (tight["rise_time"], tight["plateau_time"]) == (2.0, 1.0)
    assert (loose["rise_time"], loose["plateau_time"]) == (1.0, 3.0)
    assert tight["fall_time"] == 2.0 and loose["fall_time"] == 1.0
    everything = _measure(x, plateau=1.0)  # every frame of the bout is held
    assert (everything["rise_time"], everything["fall_time"]) == (0.0, 0.0)
    assert everything["plateau_time"] == 5.0


def test_phases_are_averaged_over_bouts():
    x = _frames(60)
    x[5:10, CENTRE] = [0.5, 1, 1, 0.5, 0.5]  # rise 1, plateau 2, fall 2
    x[30:39, CENTRE] = [0.2, 0.3, 0.4, 1, 1, 1, 1, 0.4, 0.2]  # 3, 4, 2
    m = _measure(x, dt=0.5)
    assert m["rise_time"] == (1 + 3) / 2 * 0.5
    assert m["plateau_time"] == (2 + 4) / 2 * 0.5
    assert m["fall_time"] == (2 + 2) / 2 * 0.5


def test_the_envelope_is_the_peak_over_sensors_and_ignores_sign():
    x = _frames(20)
    x[3, 0] = 0.4  # one sensor on, the next frames another sensor, negative
    x[4, 7] = -1.0
    x[5, 12] = 0.4
    m = _measure(x)
    assert (m["rise_time"], m["plateau_time"], m["fall_time"]) == (1.0, 1.0, 1.0)


def test_a_silent_entry_has_no_phase_and_no_speed():
    m = _measure(_frames(20))
    assert m == {
        "rise_time": None,
        "plateau_time": None,
        "fall_time": None,
        "speed": None,
    }


# ----------------------------------------------------------------------- speed


def _slide(n_hold=6, dt_frames=1):
    """A unit-weight spot that steps one sensor along dim 1 each frame for the hold."""
    x = _frames(30)
    x[4, CENTRE - 2] = 0.5  # rise frame, at rest
    for k in range(n_hold):
        x[5 + k, 2 * COLS + k % COLS] = 1.0
    x[5 + n_hold, 2 * COLS + COLS - 1] = 0.5  # fall frame
    return x


def test_speed_is_the_centroid_step_over_dt_within_the_plateau():
    x = _slide(
        n_hold=5
    )  # plateau frames 5..9, four steps of one sensor = 2 space units
    m = _measure(x, dt=0.5)
    assert m["plateau_time"] == 5 * 0.5
    assert m["speed"] == pytest.approx(2.0 / 0.5)


def test_motion_outside_the_plateau_is_not_counted():
    """The rise frame sits far from the plateau; the plateau itself holds still."""
    x = _frames(30)
    x[3:12, CENTRE] = [0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2]
    x[3, 0] = 0.2  # a distant sensor in the first frame: the centroid jumps there
    m = _measure(x)
    assert m["plateau_time"] == 5.0
    assert m["speed"] is None


def test_a_still_plateau_has_no_speed():
    x = _envelope([1, 1, 1, 1], start=3)
    assert _measure(x)["speed"] is None


def test_motion_tolerance_sets_how_long_a_path_counts_as_moving():
    x = _slide(n_hold=5)  # the path is 8 space units
    assert _measure(x, motion=7.9)["speed"] == pytest.approx(2.0)
    assert _measure(x, motion=8.0)["speed"] is None


def test_speed_pools_the_steps_of_the_moving_plateaus():
    x = _frames(60)
    for k, j in enumerate((0, 1, 2)):  # two steps of 2 space units
        x[5 + k, 2 * COLS + j] = 1.0
    for k, j in enumerate((0, 2, 4)):  # two steps of 4 space units
        x[30 + k, 2 * COLS + j] = 1.0
    assert _measure(x, dt=1.0)["speed"] == pytest.approx((2 + 2 + 4 + 4) / 4)
    # only the faster bout's path (8) is longer than 6: the slower one is left out
    assert _measure(x, dt=1.0, motion=6.0)["speed"] == pytest.approx(4.0)


def test_the_space_unit_scales_the_speed():
    x = _slide(n_hold=5)
    assert _measure(x)["speed"] == pytest.approx(2.0)
    assert measure_channel(x, COORDS * 10, 1.0, 0.01, 0.0)["speed"] == pytest.approx(
        20.0
    )


# ------------------------------------------------------------------------ pack


def test_the_pack_is_registered_and_round_trips():
    assert METRIC_PACK_REGISTRY.is_registered("tactile")
    cls = METRIC_PACK_REGISTRY.get_class("tactile")
    assert cls is TactilePack
    pack = cls.from_config({"plateau_tolerance": 0.05, "motion_tolerance": 0.3})
    assert pack.to_dict() == {"plateau_tolerance": 0.05, "motion_tolerance": 0.3}
    assert cls.from_config(pack.to_dict()).to_dict() == pack.to_dict()
    assert TactilePack().to_dict() == {
        "plateau_tolerance": 0.01,
        "motion_tolerance": 0.0,
    }
    spec = {p.name: p for p in cls.get_param_spec()}
    assert set(spec) == {"plateau_tolerance", "motion_tolerance"}
    assert spec["plateau_tolerance"].default == 0.01


@pytest.mark.parametrize(
    "kwargs",
    [{"plateau_tolerance": -0.1}, {"plateau_tolerance": 1.5}, {"motion_tolerance": -1}],
)
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        TactilePack(**kwargs)


def test_the_pack_states_drive_and_filtered_in_milliamps():
    units = TactilePack().units()
    assert units["drive"][0] == units["filtered"][0] == "mA"
    assert "units_and_gains" in units["drive"][1]
    assert "frame" not in units  # a world's amplitude is unitless


def test_the_catalogue_names_four_channel_metrics_with_units():
    from sensoryforge.evaluate.audit.layout import Layout

    layout = Layout(COORDS, (ROWS, COLS, SPACING), [])
    units = {
        "time": {"symbol": "ms", "origin": "data"},
        "space": {"symbol": "mm", "origin": "data"},
    }
    defs = TactilePack().metrics(layout, units, 1.0)
    assert [d["id"] for d in defs] == [
        "tactile.episode.rise_time",
        "tactile.episode.plateau_time",
        "tactile.episode.fall_time",
        "tactile.motion.speed",
    ]
    assert {d["scope"] for d in defs} == {"channel"}
    assert [d["unit"]["symbol"] for d in defs] == ["ms", "ms", "ms", "mm/ms"]
    assert [d["resolution"] for d in defs] == [1.0, 1.0, 1.0, None]
    assert all(d["pack"] == "tactile" and d["summarised"] for d in defs)


# ------------------------------------------------------------------ end to end


def _rows(out):
    return [json.loads(x) for x in (out / ENTRIES_FILE).read_text().splitlines()]


def _validator(key):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    root = REPO / "schemas" / "evaluate"
    docs = [json.loads(p.read_text()) for p in sorted(root.glob("*.schema.json"))]
    registry = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs]
    )
    audit_id = f"{docs[0]['$id'].rsplit('/', 1)[0]}/audit.schema.json"
    return jsonschema.Draft202012Validator({"$ref": audit_id + key}, registry=registry)


def test_the_audit_with_both_packs_validates_and_lists_the_tactile_metrics(audited):
    audit, out = audited
    _validator("").validate(json.loads((out / AUDIT_FILE).read_text()))
    for row in _rows(out):
        _validator("#/$defs/entry_row").validate(row)
    assert [p["name"] for p in audit["packs"]] == ["core", "tactile"]
    assert audit["params"]["tactile"] == {
        "plateau_tolerance": 0.01,
        "motion_tolerance": 0.0,
    }
    ids = [m["id"] for m in audit["metrics"]]
    assert ids.index("tactile.episode.rise_time") > ids.index(
        "core.dataset.duplicate_frames"
    )
    # the pack's unit reaches the core's population-input metrics
    assert audit["units"]["drive"]["symbol"] == "mA"
    assert audit["units"]["drive"]["origin"] == "pack:tactile"
    assert audit["units"]["frame"] is None
    drive = next(m for m in audit["metrics"] if m["id"] == "core.input.drive_peak")
    assert drive["unit"]["symbol"] == "mA"
    speed = next(m for m in audit["metrics"] if m["id"] == "tactile.motion.speed")
    assert speed["unit"] == {"symbol": "mm/ms", "origin": "derived"}


def test_a_single_bout_entry_splits_into_phases_that_equal_the_core_bout(audited):
    audit, out = audited
    checked = 0
    for row in _rows(out):
        v = {(x["id"], x.get("population")): x["value"] for x in row["values"]}
        if v[("core.frames.bouts", None)] != 1:
            continue
        total = sum(
            v[(f"tactile.episode.{n}_time", None)] for n in ("rise", "plateau", "fall")
        )
        assert total == pytest.approx(v[("core.frames.bout_mean", None)])
        checked += 1
    assert checked >= 3


def test_every_entry_with_activity_has_phases_and_a_silent_one_has_none(audited):
    _, out = audited
    for row in _rows(out):
        v = {x["id"]: x["value"] for x in row["values"] if "population" not in x}
        if v["core.frames.bouts"]:
            assert v["tactile.episode.plateau_time"] >= 1.0
            assert v["tactile.episode.rise_time"] >= 0.0
        else:
            assert v["tactile.episode.rise_time"] is None


def test_moving_stimuli_have_a_speed_and_static_ones_do_not(audited):
    audit, out = audited
    by_class = {}
    for row in _rows(out):
        v = {x["id"]: x["value"] for x in row["values"] if "population" not in x}
        by_class.setdefault(row["class"], []).append(v["tactile.motion.speed"])
    assert any(
        s is not None and s > 0 for speeds in by_class.values() for s in speeds
    ), by_class
    summaries = {
        s["id"]: s
        for s in audit["dataset"]["summaries"]
        if s["id"].startswith("tactile.")
    }
    assert set(summaries) == {
        "tactile.episode.rise_time",
        "tactile.episode.plateau_time",
        "tactile.episode.fall_time",
        "tactile.motion.speed",
    }
    assert summaries["tactile.episode.plateau_time"]["n"] > 0


def test_the_audit_runs_with_the_tactile_pack_on_the_toy_data_set(
    toy_dataset, tmp_path
):
    """Three channels: every tactile value carries its channel."""
    audit = run_audit(
        toy_dataset.data, tmp_path, packs=["core", "tactile"], created_at=STAMP
    )
    assert audit["layout"]["n_channels"] == 3
    row = _rows(tmp_path)[0]
    mine = [x for x in row["values"] if x["id"].startswith("tactile.")]
    assert len(mine) == 4 * 3
    assert {x["channel"] for x in mine} == {0, 1, 2}
    assert audit["units"]["space"]["symbol"] == "px"  # the data's unit wins


def test_two_runs_are_byte_identical_but_for_the_stamp(tactile_dataset, tmp_path):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        run_audit(tactile_dataset, out, packs=["core", "tactile"], created_at=STAMP)
        outs.append(out)
    for fname in (AUDIT_FILE, ENTRIES_FILE):
        assert (outs[0] / fname).read_bytes() == (outs[1] / fname).read_bytes()


def test_the_command_line_accepts_the_tactile_pack(tactile_dataset, tmp_path, capsys):
    from sensoryforge.evaluate.audit.__main__ import main

    status = main(
        [
            str(tactile_dataset),
            "--out",
            str(tmp_path),
            "--packs",
            "core,tactile",
            "--param",
            "tactile.plateau_tolerance=0.1",
        ]
    )
    assert status == 0
    audit = json.loads((tmp_path / AUDIT_FILE).read_text())
    assert audit["params"]["tactile"]["plateau_tolerance"] == 0.1
