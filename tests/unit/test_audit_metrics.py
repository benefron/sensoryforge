"""Core metrics, the summariser and the pack interface (roadmap A2b and A2c).

Each frame metric is checked on frames built by hand, whose answer follows from the
definitions in ``docs/reference/report_contract.md`` section 6, not from running the
metric on random data.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sensoryforge.evaluate.audit import frame_metrics as fm
from sensoryforge.evaluate.audit import response_metrics as rm
from sensoryforge.evaluate.audit.core_pack import CorePack
from sensoryforge.evaluate.audit.layout import Layout
from sensoryforge.evaluate.audit.summary import ValueTable, summarise_values
from sensoryforge.evaluate.audit.units import unit_of
from sensoryforge.evaluate.pack import DatasetRow
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY

ROWS = COLS = 5
SPACING = 2.0
# Sensor k = i * COLS + j sits at x = (i - 2) * 2, y = (j - 2) * 2 (dim 0 first).
COORDS = np.array(
    [[(i - 2) * SPACING, (j - 2) * SPACING] for i in range(ROWS) for j in range(COLS)]
)


def _empty(n_t: int) -> np.ndarray:
    return np.zeros((n_t, ROWS * COLS), dtype=np.float32)


def _sensor(i: int, j: int) -> int:
    return i * COLS + j


def _measure(x, dt=1.0, tol=0.0):
    return fm.measure_channel(x, COORDS, dt, tol)


# ------------------------------------------------------------------------- frames


def test_two_bouts_timeline_and_counts():
    x = _empty(40)
    x[10:20, _sensor(2, 2)] = 0.5
    x[30:35, _sensor(2, 2)] = 0.5
    m = _measure(x, dt=2.0)
    assert m["n_frames"] == 40 and m["dt"] == 2.0 and m["duration"] == 80.0
    assert m["onset_time"] == 20.0  # frame 10 at dt 2
    assert m["offset_time"] == 70.0  # the end of frame 34
    assert m["active_time"] == 30.0 and m["active_fraction"] == 15 / 40
    assert m["bouts"] == 2
    assert m["bout_mean"] == 15.0 / 2 * 2.0  # (10 + 5) / 2 frames, times dt
    assert m["gap_mean"] == m["gap_max"] == 10 * 2.0
    assert m["footprint_mean"] == m["footprint_max"] == 1 / 25


def test_gaps_need_two_bouts_and_three_bouts_give_mean_and_max():
    x = _empty(30)
    x[2:4, 0] = 1.0
    assert _measure(x)["gap_mean"] is None and _measure(x)["gap_max"] is None
    x[6:8, 0] = 1.0  # gap of 2 frames
    x[14:16, 0] = 1.0  # gap of 6 frames
    m = _measure(x)
    assert (m["bouts"], m["gap_mean"], m["gap_max"]) == (3, 4.0, 6.0)


def test_silent_entry_is_null_where_undefined():
    m = _measure(_empty(20))
    assert m["bouts"] == 0 and m["active_time"] == 0 and m["active_fraction"] == 0
    for name in (
        "onset_time offset_time bout_mean gap_mean footprint_mean spread path_length "
        "centroid_x centroid_y centroid_speed direction dominant_frequency"
    ).split():
        assert m[name] is None, name
    assert m["peak"] == m["trough"] == m["mean"] == m["rms"] == 0.0


def test_amplitude_statistics_and_negative_fraction():
    x = _empty(4)
    x[1, 0] = 2.0
    x[2, 1] = -1.0
    m = _measure(x)
    assert (m["peak"], m["trough"]) == (2.0, -1.0)
    assert m["mean"] == pytest.approx(1.0 / x.size)
    assert m["rms"] == pytest.approx(np.sqrt(5.0 / x.size))
    assert m["negative_fraction"] == pytest.approx(1 / x.size)


def test_zero_tolerance_hides_small_values():
    x = _empty(10)
    x[3:6, 4] = 0.01
    x[6:8, 4] = 0.5
    assert _measure(x)["bouts"] == 1 and _measure(x)["active_time"] == 5
    m = _measure(x, tol=0.05)
    assert m["active_time"] == 2 and m["onset_time"] == 6.0
    # a value under the tolerance is not counted as negative either
    x[0, 0] = -0.01
    assert _measure(x, tol=0.05)["negative_fraction"] == 0.0
    assert _measure(x)["negative_fraction"] == pytest.approx(1 / x.size)


def test_centroid_spread_and_footprint_of_a_two_sensor_frame():
    x = _empty(5)
    x[1:4, _sensor(2, 2)] = 1.0  # at the origin
    x[1:4, _sensor(2, 4)] = 3.0  # at y = 4
    m = _measure(x)
    assert m["centroid_x"] == pytest.approx(0.0)
    assert m["centroid_y"] == pytest.approx(3.0)  # weights 1 and 3
    # rms distance from y = 3: sqrt((1 * 9 + 3 * 1) / 4)
    assert m["spread"] == pytest.approx(np.sqrt(3.0))
    assert m["footprint_mean"] == pytest.approx(2 / 25)
    assert "centroid_z" not in m  # the layout has two dimensions


@pytest.mark.parametrize(
    "di, dj, angle", [(1, 0, 0.0), (0, 1, 90.0), (-1, 0, 180.0), (0, -1, 270.0)]
)
def test_motion_speed_and_direction(di, dj, angle):
    # Dimension 0 is x: sensor index i moves along x, j along y.
    x = _empty(8)
    for k in range(4):
        x[2 + k, _sensor(2 + di * (k - 1), 2 + dj * (k - 1))] = 1.0
    m = _measure(x, dt=0.5)
    # three steps of one sensor spacing, each over one frame step
    assert m["path_length"] == pytest.approx(3 * SPACING)
    assert m["centroid_speed"] == pytest.approx(SPACING / 0.5)
    assert m["direction"] == pytest.approx(angle)


def test_path_does_not_cross_a_gap_between_bouts():
    x = _empty(12)
    x[1:3, _sensor(0, 2)] = 1.0
    x[8:10, _sensor(4, 2)] = 1.0  # a jump across the gap does not count as motion
    m = _measure(x)
    assert m["path_length"] == 0.0 and m["centroid_speed"] == 0.0
    # but the net displacement first-to-last still gives a direction
    assert m["direction"] == pytest.approx(0.0)  # x grows from -4 to +4


def test_static_frames_have_no_direction():
    x = _empty(6)
    x[1:5, _sensor(1, 3)] = 0.7
    assert _measure(x)["direction"] is None


def test_dominant_frequency_of_a_sine_bout():
    n, dt = 64, 0.5
    t = np.arange(n) * dt
    x = _empty(n + 10)
    # 8 cycles in the bout (n * dt = 32 ms): 0.25 per ms; offset so frames stay active
    x[5 : 5 + n, _sensor(2, 2)] = 2.0 + np.sin(2 * np.pi * 0.25 * t)
    m = _measure(x, dt=dt)
    assert m["dominant_frequency"] == pytest.approx(0.25)
    assert m["bouts"] == 1


def test_dominant_frequency_null_when_short_or_flat():
    x = _empty(20)
    x[2:5, 0] = 1.0  # 3 frames
    assert _measure(x)["dominant_frequency"] is None
    x[2:12, 0] = 1.0  # a flat plateau has no power once its mean is removed
    assert _measure(x)["dominant_frequency"] is None


def test_digest_is_sha256_of_little_endian_float32():
    x = _empty(3)
    x[1, 4] = 0.25
    expect = hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()
    assert fm.digest(x) == expect == _measure(x)["digest"]
    x[1, 4] = 0.5
    assert fm.digest(x) != expect


# ----------------------------------------------------------------------- responses


def test_spike_rates_and_active_unit_fraction():
    n_t, n = 20, 4
    active = np.zeros(n_t, bool)
    active[5:10] = True
    s = np.zeros((n_t, n), bool)
    s[6, 0] = s[8, 0] = s[9, 1] = True  # 3 spikes in the active window
    s[15, 2] = True  # 1 in the quiet window
    r = rm.response_stats("spikes", s, {}, active, dt=2.0)
    assert r["n_units"] == 4 and r["count"] == 4
    assert r["rate_mean"] == pytest.approx(4 / (4 * 20 * 2.0))
    assert r["rate_active"] == pytest.approx(3 / (4 * 5 * 2.0))
    assert r["rate_quiet"] == pytest.approx(1 / (4 * 15 * 2.0))
    assert r["active_unit_fraction"] == 3 / 4
    assert r["latency"] == (6 - 5) * 2.0


def test_empty_windows_and_no_onset_are_null():
    s = np.zeros((10, 2), bool)
    s[3, 0] = True
    all_active = np.ones(10, bool)
    r = rm.response_stats("spikes", s, {}, all_active, dt=1.0)
    assert r["rate_quiet"] is None and r["rate_active"] == pytest.approx(1 / 20)
    none_active = np.zeros(10, bool)
    r = rm.response_stats("spikes", s, {}, none_active, dt=1.0)
    assert r["rate_active"] is None and r["latency"] is None
    silent = rm.response_stats("spikes", np.zeros((10, 2), bool), {}, all_active, 1.0)
    assert silent["latency"] is None and silent["active_unit_fraction"] == 0.0


def test_latency_ignores_spikes_before_the_onset():
    active = np.zeros(30, bool)
    active[10:15] = True
    s = np.zeros((30, 3), bool)
    s[2, 0] = True  # before the stimulus
    s[13, 1] = True
    r = rm.response_stats("spikes", s, {}, active, dt=0.5)
    assert r["latency"] == 3 * 0.5
    assert r["count"] == 2


def _naive_isi_cv(spikes):
    cvs = []
    for u in range(spikes.shape[1]):
        t = np.flatnonzero(spikes[:, u])
        if t.size >= 3:
            d = np.diff(t).astype(float)
            cvs.append(d.std() / d.mean())
    return float(np.mean(cvs)) if cvs else None


def test_isi_cv_matches_a_loop_over_units():
    rng = np.random.default_rng(3)
    s = rng.random((200, 12)) < 0.08
    s[:, 0] = False  # silent unit
    s[:2, 1] = True
    s[2:, 1] = False  # two spikes: not enough
    assert rm.isi_cv(s) == pytest.approx(_naive_isi_cv(s), rel=1e-12)
    regular = np.zeros((50, 1), bool)
    regular[::5] = True
    assert rm.isi_cv(regular) == pytest.approx(0.0, abs=1e-12)
    assert rm.isi_cv(np.zeros((9, 3), bool)) is None


def test_event_readout_counts_by_sign():
    e = np.zeros((10, 3), np.int16)
    e[2, 0], e[3, 0], e[4, 1], e[7, 2] = 2, -1, -3, 1
    active = np.zeros(10, bool)
    active[2:6] = True
    r = rm.response_stats("events", e, {}, active, dt=1.0)
    assert (r["events_on"], r["events_off"]) == (3, 4)
    assert r["active_unit_fraction"] == 1.0 and r["latency"] == 0.0
    assert "count" not in r and "rate_mean" not in r


def test_state_readout_statistics():
    v = np.arange(12, dtype=np.float32).reshape(6, 2)
    r = rm.response_stats("state", v, {}, np.ones(6, bool), dt=1.0)
    assert (r["state_min"], r["state_max"]) == (0.0, 11.0)
    assert r["state_mean"] == pytest.approx(5.5)
    assert r["state_std"] == pytest.approx(v.std())
    assert "latency" not in r


def test_response_digest_is_over_the_stored_bytes():
    s = np.zeros((4, 2), bool)
    s[1, 1] = True
    r = rm.response_stats("spikes", s, {}, np.ones(4, bool), 1.0)
    assert r["digest"] == hashlib.sha256(s.tobytes()).hexdigest()


def test_input_statistics_keep_the_sign_of_filtered():
    d = np.array([[0.0, 1.0], [3.0, 0.0]], np.float32)
    f = np.array([[-2.0, 1.0], [0.5, 0.0]], np.float32)
    r = rm.input_stats(d, f)
    assert (r["drive_peak"], r["drive_mean"]) == (3.0, 1.0)
    assert (r["filtered_max"], r["filtered_min"]) == (1.0, -2.0)


# ----------------------------------------------------------------------- summaries


def test_summary_fields_follow_the_contract():
    v = np.array([1.0, 2.0, 2.0, 4.0, 10.0])
    s = summarise_values(v, n_null=2, bins=4, metric_id="core.x.y")
    assert (s["n"], s["n_null"]) == (5, 2)
    assert s["mean"] == pytest.approx(3.8) and s["std"] == pytest.approx(v.std())
    assert (s["min"], s["max"]) == (1.0, 10.0)
    assert s["q50"] == 2.0 and s["q25"] == 2.0 and s["q75"] == 4.0
    assert s["q05"] == pytest.approx(np.quantile(v, 0.05))
    h = s["histogram"]
    assert h["edges"] == [1.0, 3.25, 5.5, 7.75, 10.0]
    assert h["counts"] == [3, 1, 0, 1] and sum(h["counts"]) == 5


def test_summary_of_equal_values_has_one_bin_and_none_has_no_statistics():
    s = summarise_values(np.array([3.0, 3.0]), 0, 8, "core.x.y")
    assert s["histogram"] == {"edges": [3.0, 3.0], "counts": [2]}
    empty = summarise_values(np.array([]), 4, 8, "core.x.y", population="p")
    assert empty["n"] == 0 and empty["n_null"] == 4 and empty["population"] == "p"
    assert empty["mean"] is None and empty["histogram"] is None


def test_summary_scope_keys_and_bad_bins():
    s = summarise_values(np.array([1.0]), 0, 2, "core.x.y", channel=1)
    assert s["channel"] == 1 and "population" not in s
    with pytest.raises(ValueError, match="histogram_bins"):
        summarise_values(np.array([1.0]), 0, 0, "core.x.y")


def test_value_table_groups_failures_and_classless_entries():
    key = ("core.x.y", None, None)
    t = ValueTable([key], bins=2)
    t.add("train", "a", {key: 1.0})
    t.add("train", "a", {key: 3.0})
    t.add("test", "b", {key: None})  # a null value
    t.add("test", None, {key: 5.0})  # a session: no class
    t.add_failed("test", "b")
    dataset, by_class, by_split = t.groups()
    assert (dataset["n_entries"], dataset["n_failed"]) == (5, 1)
    d = dataset["summaries"][0]
    assert (d["n"], d["n_null"], d["mean"]) == (3, 1, 3.0)
    assert sorted(by_class) == ["a", "b"] and sorted(by_split) == ["test", "train"]
    assert by_class["a"]["summaries"][0]["mean"] == 2.0
    b = by_class["b"]
    assert (b["n_entries"], b["n_failed"]) == (2, 1)
    assert b["summaries"][0]["n"] == 0 and b["summaries"][0]["n_null"] == 1
    sp = by_split["test"]
    assert (sp["n_entries"], sp["n_failed"]) == (3, 1)  # classless entry counted here
    assert sp["summaries"][0]["n"] == 1


# ------------------------------------------------------------------- the pack API


def _layout(dims=2):
    coords = np.hstack([COORDS, np.zeros((len(COORDS), 1))])
    return Layout(coords[:, :dims].copy(), None, 1)


UNITS = {
    "time": {"symbol": "ms", "origin": "data", "source": "data.h5:dt_ms"},
    "space": {"symbol": "mm", "origin": "data", "source": "x"},
    "frame": None,
    "drive": {"symbol": "mA", "origin": "pack:x", "source": "y"},
    "filtered": None,
    "state": None,
}


def test_the_core_pack_is_registered_and_round_trips():
    cls = METRIC_PACK_REGISTRY.get_class("core")
    pack = cls.from_config({"zero_tolerance": 0.25, "histogram_bins": 7})
    assert pack.to_dict() == {"zero_tolerance": 0.25, "histogram_bins": 7}
    assert cls.from_config(pack.to_dict()).to_dict() == pack.to_dict()
    assert {p.name for p in cls.get_param_spec()} == set(pack.to_dict())
    assert METRIC_PACK_REGISTRY.get_class("CORE") is cls
    with pytest.raises(ValueError, match="zero_tolerance"):
        CorePack(zero_tolerance=-1)
    with pytest.raises(ValueError, match="histogram_bins"):
        CorePack(histogram_bins=0)


def test_catalogue_units_scope_and_resolution():
    defs = {d["id"]: d for d in CorePack().metrics(_layout(), UNITS, dt=0.5)}
    assert len(defs) == len(list(CorePack().metrics(_layout(), UNITS)))
    assert defs["core.frames.onset_time"]["resolution"] == 0.5
    assert defs["core.frames.peak"]["unit"] is None  # nobody states the frame unit
    assert defs["core.frames.centroid_speed"]["unit"]["symbol"] == "mm/ms"
    assert defs["core.frames.dominant_frequency"]["unit"]["symbol"] == "1/ms"
    assert defs["core.frames.direction"]["unit"]["symbol"] == "deg"
    assert defs["core.frames.digest"]["summarised"] is False
    assert defs["core.frames.digest"]["unit"] is None
    assert defs["core.input.drive_peak"]["unit"]["symbol"] == "mA"
    assert defs["core.response.latency"]["applies_to"] == ["spikes", "events"]
    assert "applies_to" not in defs["core.response.n_units"]
    assert defs["core.dataset.silent_entries"]["level"] == "dataset"
    assert defs["core.run.seconds"]["scope"] == "entry"
    assert "core.frames.centroid_z" not in defs
    three = {d["id"] for d in CorePack().metrics(_layout(3), UNITS)}
    assert "core.frames.centroid_z" in three
    ids = [d["id"] for d in CorePack().metrics(_layout(), UNITS)]
    assert ids.index("core.frames.n_frames") < ids.index("core.input.drive_peak")


def test_unit_of_composes_and_gives_null_when_a_part_is_unstated():
    assert unit_of("1/time", UNITS) == {"symbol": "1/ms", "origin": "derived"}
    assert unit_of("space/time", {**UNITS, "time": None}) is None
    assert unit_of("count", UNITS)["symbol"] == "count"
    assert unit_of("frame", UNITS) is None
    with pytest.raises(ValueError, match="unknown unit"):
        unit_of("furlong", UNITS)


def test_dataset_counts_silent_and_duplicate_entries():
    d1, d2 = "a" * 64, "b" * 64

    def row(entry, noise, bouts, digest):
        return DatasetRow(
            entry,
            "train",
            "c",
            0,
            noise,
            {
                ("core.frames.bouts", 0, None): bouts,
                ("core.frames.bouts", 1, None): 0,
                ("core.frames.digest", 0, None): digest,
                ("core.frames.digest", 1, None): d2,
            },
        )

    rows = [
        row("e0", 0, 1, d1),
        row("e1", 1, 1, d1),  # a noise repeat shares frames by design
        row("e2", 0, 1, d1),  # an earlier entry has these frames
        row("e3", 0, 0, d1),  # silent: not a duplicate
        row("e4", 0, 2, d2),
    ]
    out = {v["id"]: v["value"] for v in CorePack().measure_dataset(rows)}
    assert out == {
        "core.dataset.silent_entries": 1,
        "core.dataset.duplicate_frames": 1,
    }


def test_a_three_dimensional_layout_gets_a_z_centroid():
    coords = np.hstack([COORDS, np.tile([[0.0], [3.0]], (13, 1))[:25]])
    x = _empty(4)
    x[1:3, 0] = 1.0
    m = fm.measure_channel(x, coords, 1.0, 0.0)
    assert m["centroid_z"] == pytest.approx(coords[0, 2])
    assert m["direction"] is None  # no net x-y displacement
