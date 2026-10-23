"""The compare stage (roadmap A4, report contract section 9).

Pairing, distances and the stage are checked on small hand-built audit and declared
files whose answers follow from the definitions. The end-to-end checks compare the
audit and the declared side of a data set built through the engine, and re-derive
every difference from the two entry files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensoryforge.evaluate.audit.summary import ValueTable
from sensoryforge.evaluate.compare import (
    COMPARE_FILE,
    ENTRIES_FILE,
    format_distribution_table,
    format_entry_table,
    format_histograms,
    format_stat_table,
    ks_distance,
    pair_histogram,
    pair_values,
    run_compare,
    wasserstein_distance,
)
from sensoryforge.evaluate.stage_io import file_sha256, write_json, write_rows

REPO = Path(__file__).resolve().parents[2]
STAMP = "2026-10-22T14:03:11Z"


def _validator(key=""):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    root = REPO / "schemas" / "evaluate"
    docs = [json.loads(p.read_text()) for p in sorted(root.glob("*.schema.json"))]
    registry = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs]
    )
    ident = next(d["$id"] for d in docs if d["$id"].endswith("/compare.schema.json"))
    return jsonschema.Draft202012Validator({"$ref": ident + key}, registry=registry)


# ------------------------------------------------------------------ pairing


@pytest.mark.parametrize(
    "measured, declared, m_unit, d_unit, pairing",
    [
        (13.0, 12.4, "ms", "ms", "ok"),
        (13.0, 12.4, None, "ms", "unit_unchecked"),
        (13.0, 12.4, "ms", None, "unit_unchecked"),
        (13.0, 12.4, None, None, "unit_unchecked"),
        (13.0, 12.4, "ms", "s", "unit_mismatch"),
        (None, 12.4, "ms", "ms", "missing_measured"),
        (None, None, "ms", "ms", "missing_measured"),
        (13.0, None, "ms", "ms", "missing_declared"),
        ("abc", 12.4, "ms", "ms", "not_numeric"),
        (13.0, "abc", "ms", "ms", "not_numeric"),
        (True, 1.0, "ms", "ms", "not_numeric"),
    ],
)
def test_the_pairing_names_why_a_difference_could_not_be_computed(
    measured, declared, m_unit, d_unit, pairing
):
    assert pair_values(measured, declared, m_unit, d_unit)["pairing"] == pairing


def test_the_difference_is_measured_minus_declared_and_is_exact_for_integers():
    pair = pair_values(13.0, 12.5, "ms", "ms")
    assert pair["diff"] == 0.5 and pair["rel_diff"] == 0.5 / 12.5
    ints = pair_values(10**17 + 1, 10**17, "count", "count")
    assert ints["diff"] == 1 and isinstance(ints["diff"], int)


def test_a_declared_zero_has_no_relative_difference():
    pair = pair_values(0.25, 0.0, "ms", "ms")
    assert pair["diff"] == 0.25 and pair["rel_diff"] is None


def test_a_negative_declared_value_divides_by_its_magnitude():
    pair = pair_values(-1.0, -2.0, "mm", "mm")
    assert pair["diff"] == 1.0 and pair["rel_diff"] == 0.5


def test_no_difference_is_computed_when_the_pairing_is_not_ok_or_unchecked():
    for args in [(None, 1.0), (1.0, None), ("a", 1.0)]:
        pair = pair_values(*args, "ms", "ms")
        assert pair["diff"] is None and pair["rel_diff"] is None
    mismatch = pair_values(1.0, 2.0, "ms", "s")
    assert mismatch["diff"] is None and mismatch["measured"] == 1.0


# ------------------------------------------------------------------ distances


def test_ks_and_wasserstein_follow_their_definitions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 3.0, 4.0, 5.0])
    assert ks_distance(x, y) == 0.25
    assert wasserstein_distance(x, y) == 1.0
    assert ks_distance(x, x) == 0.0 and wasserstein_distance(x, x) == 0.0
    assert ks_distance(x, x + 100.0) == 1.0
    assert wasserstein_distance(x, x + 100.0) == 100.0


def test_the_distances_ignore_the_order_of_the_samples():
    x = np.array([4.0, 1.0, 3.0, 2.0])
    y = np.array([5.0, 2.0, 4.0, 3.0])
    assert ks_distance(x, y) == 0.25 and wasserstein_distance(x, y) == 1.0


def test_the_distances_refuse_what_they_cannot_measure():
    with pytest.raises(ValueError, match="at least one"):
        ks_distance(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match="equal size"):
        wasserstein_distance(np.array([1.0, 2.0]), np.array([1.0]))


def test_the_paired_histograms_share_their_edges():
    hist = pair_histogram(np.array([0.0, 1.0, 1.0]), np.array([1.0, 3.0, 4.0]), 4)
    assert hist["edges"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert hist["measured"] == [1, 2, 0, 0] and hist["declared"] == [0, 1, 0, 2]
    one = pair_histogram(np.array([2.0, 2.0]), np.array([2.0, 2.0]), 8)
    assert one == {"edges": [2.0, 2.0], "measured": [2], "declared": [2]}


# ------------------------------------------------------------------ the stage, by hand

#: entry, split, class, audit values, declared values
ROWS = [
    ("a/1", "train", "a", {"t": 10.0, "p": 1.0, "x0": 0.5, "x1": 2.0, "d": "q"}, None),
    ("a/2", "train", "a", {"t": None, "p": 2.0, "x0": 1.5, "x1": 1.0, "d": "r"}, None),
    ("b/1", "test", "b", {"t": 7.0, "p": 3.0, "x0": 0.0, "x1": 0.0, "d": "s"}, None),
    ("s/1", "sessions", None, None, None),  # the audit failed this entry
]
DECLARED = {
    "a/1": {"t": 8.5, "p": 0.5, "x0": 0.25, "x1": 2.0, "d": 7, "dur": 100.0},
    "a/2": {"t": 4.0, "p": 1.0, "x0": 1.5, "x1": 0.0, "d": 7, "dur": 100.0},
    "b/1": {"t": 7.0, "p": 4.0, "x0": 0.5, "x1": 0.0, "d": 7, "dur": 100.0},
    "s/1": {"t": 0.0, "p": 1.0, "x0": 0.0, "x1": 0.0, "d": 7, "dur": 100.0},
}
IDS = {
    "t": ("core.frames.onset_time", None),
    "p": ("core.frames.peak", None),
    "x0": ("core.frames.centroid_x", 0),
    "x1": ("core.frames.centroid_x", 1),
    "d": ("core.frames.digest", None),
    "dur": ("core.frames.duration", None),
}


def _value(short, value):
    metric_id, channel = IDS[short]
    out = {"id": metric_id}
    if channel is not None:
        out["channel"] = channel
    out["value"] = value
    return out


def _metric(metric_id, symbol, **extra):
    return {
        "id": metric_id,
        "unit": None if symbol is None else {"symbol": symbol, "origin": "test"},
        **extra,
    }


def _write_stage(folder, tag, metrics, rows, extra):
    folder.mkdir(parents=True, exist_ok=True)
    stem = tag.split("-")[1].split("/")[0]
    n, sha = write_rows(folder / f"{stem}_entries.jsonl", rows)
    content = {
        "format": tag,
        "metrics": metrics,
        "entries": {"file": f"{stem}_entries.jsonl", "n": n, "sha256": sha},
        **extra,
    }
    write_json(folder / f"{stem}.json", content)
    return folder / f"{stem}.json"


def _groups(rows, keys):
    table = ValueTable(keys, 4)
    for row in rows:
        values = {(v["id"], v.get("channel"), None): v["value"] for v in row["values"]}
        if row.get("failed"):
            table.add_failed(row["split"], row["class"])
        else:
            table.add(row["split"], row["class"], values)
    dataset, by_class, by_split = table.groups()
    return dataset, {"class": by_class, "split": by_split}


@pytest.fixture()
def by_hand(tmp_path):
    keys = [
        ("core.frames.onset_time", None, None),
        ("core.frames.peak", None, None),
        ("core.frames.centroid_x", 0, None),
        ("core.frames.centroid_x", 1, None),
        ("core.frames.digest", None, None),
    ]
    audit_rows, declared_rows = [], []
    for entry, split, cls, measured, _ in ROWS:
        label = {"entry": entry, "split": split, "class": cls}
        failed = measured is None
        audit_rows.append(
            {
                **label,
                "repeat": 0,
                "noise_repeat": 0,
                "bundle": entry,
                "values": (
                    []
                    if failed
                    else [_value(k, measured[k]) for k in ("t", "p", "x0", "x1", "d")]
                ),
                "failed": failed,
            }
        )
        declared_rows.append(
            {
                **label,
                "repeat": 0,
                "noise_repeat": 0,
                "values": [
                    _value(k, DECLARED[entry][k])
                    for k in ("dur", "t", "p", "x0", "x1", "d")
                ],
            }
        )
    a_dataset, a_groups = _groups(audit_rows, keys)
    for row in audit_rows:
        row.pop("failed")
    d_keys = keys[:4] + [("core.frames.duration", None, None)]
    d_dataset, d_groups = _groups(declared_rows, d_keys + [keys[4]])
    audit = _write_stage(
        tmp_path / "audit",
        "sensoryforge-audit/1",
        [
            _metric("core.frames.onset_time", "ms", resolution=1.0),
            _metric("core.frames.peak", None, resolution=None),
            _metric("core.frames.centroid_x", "mm", resolution=None),
            _metric("core.frames.digest", None, resolution=None),
            _metric("core.frames.rms", None, resolution=None),
        ],
        audit_rows,
        {"dataset": a_dataset, "groupings": a_groups},
    )
    declared = _write_stage(
        tmp_path / "declared",
        "sensoryforge-declared/1",
        [
            _metric("core.frames.duration", "s"),
            _metric("core.frames.onset_time", "ms"),
            _metric("core.frames.peak", None),
            _metric("core.frames.centroid_x", "mm"),
            _metric("core.frames.digest", "ms"),
            _metric("tactile.motion.speed", "mm/ms"),
        ],
        declared_rows,
        {"dataset": d_dataset, "groupings": d_groups},
    )
    return audit, declared, tmp_path


def _entry_rows(out):
    return [json.loads(x) for x in (out / ENTRIES_FILE).read_text().splitlines()]


def _pair(row, metric_id, channel=None):
    found = [
        p for p in row["pairs"] if p["id"] == metric_id and p.get("channel") == channel
    ]
    assert len(found) == 1
    return found[0]


def test_every_declared_value_is_paired_with_its_measured_one(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    rows = {r["entry"]: r for r in _entry_rows(root / "out")}
    assert list(rows) == ["a/1", "a/2", "b/1", "s/1"]
    onset = _pair(rows["a/1"], "core.frames.onset_time")
    assert (onset["measured"], onset["declared"], onset["diff"]) == (10.0, 8.5, 1.5)
    assert onset["rel_diff"] == 1.5 / 8.5 and onset["resolution"] == 1.0
    assert onset["pairing"] == "ok"
    # the audit states no unit for the peak: a difference, but not a checked one
    assert _pair(rows["a/1"], "core.frames.peak")["pairing"] == "unit_unchecked"
    # a channel pairs with the same channel only
    assert _pair(rows["a/1"], "core.frames.centroid_x", 0)["diff"] == 0.25
    assert _pair(rows["a/1"], "core.frames.centroid_x", 1)["diff"] == 0.0
    assert _pair(rows["a/2"], "core.frames.centroid_x", 1)["diff"] == 1.0
    # a measured null, and a failed entry
    assert _pair(rows["a/2"], "core.frames.onset_time")["pairing"] == "missing_measured"
    assert all(p["pairing"] == "missing_measured" for p in rows["s/1"]["pairs"])
    # known and different units; a string against a number
    assert _pair(rows["a/1"], "core.frames.duration")["pairing"] == "missing_measured"
    assert _pair(rows["a/1"], "core.frames.digest")["pairing"] == "not_numeric"
    assert result["difference"] == "measured - declared"


def test_a_declared_zero_leaves_the_relative_difference_empty(by_hand):
    audit, declared, root = by_hand
    run_compare(audit, declared, root / "out", created_at=STAMP)
    row = next(r for r in _entry_rows(root / "out") if r["entry"] == "b/1")
    pair = _pair(row, "core.frames.centroid_x", 1)
    assert pair["declared"] == 0.0 and pair["diff"] == 0.0
    assert pair["rel_diff"] is None


def test_the_unit_mismatch_keeps_both_values_and_no_difference(tmp_path, by_hand):
    audit, declared, root = by_hand
    # the audit calls the duration seconds here, the declared side milliseconds
    doc = json.loads(audit.read_text())
    doc["metrics"].append(_metric("core.frames.duration", "ms", resolution=1.0))
    audit.write_text(json.dumps(doc))
    rows = [
        json.loads(x)
        for x in (audit.parent / "audit_entries.jsonl").read_text().splitlines()
    ]
    for r in rows:
        if r["values"]:
            r["values"].append({"id": "core.frames.duration", "value": 0.1})
    n, sha = write_rows(audit.parent / "audit_entries.jsonl", rows)
    doc["entries"]["sha256"] = sha
    write_json(audit, doc)
    run_compare(audit, declared, root / "out", created_at=STAMP)
    pair = _pair(_entry_rows(root / "out")[0], "core.frames.duration")
    assert pair["pairing"] == "unit_mismatch"
    assert (pair["measured"], pair["declared"]) == (0.1, 100.0)
    assert pair["diff"] is None and pair["rel_diff"] is None


def test_the_counts_the_metric_list_and_the_unpaired_ids(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    counts = result["counts"]
    assert sum(counts.values()) == 4 * 6
    assert counts["not_numeric"] == 3 and counts["unit_unchecked"] == 3
    assert counts["unit_mismatch"] == 0
    by_id = {m["id"]: m for m in result["metrics"]}
    assert list(by_id) == [
        "core.frames.onset_time",
        "core.frames.peak",
        "core.frames.centroid_x",
        "core.frames.digest",
    ]
    assert by_id["core.frames.onset_time"] == {
        "id": "core.frames.onset_time",
        "unit": {"symbol": "ms"},
        "resolution": 1.0,
        "n_pairs": 2,
    }
    assert by_id["core.frames.peak"]["unit"] is None
    assert result["unpaired"] == {
        "measured_only": ["core.frames.rms"],
        "declared_only": ["core.frames.duration", "tactile.motion.speed"],
    }


def test_group_and_data_set_rows_pair_the_summaries_stat_by_stat(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    onset = {
        p["stat"]: p for p in result["dataset"] if p["id"] == "core.frames.onset_time"
    }
    assert list(onset) == [
        "n",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "max",
    ]
    # measured onset times 10 and 7, declared 8.5, 4, 7 and 0
    assert onset["n"]["measured"] == 2 and onset["n"]["declared"] == 4
    assert onset["n"]["diff"] == -2 and onset["mean"]["measured"] == 8.5
    assert onset["mean"]["declared"] == 19.5 / 4
    assert onset["mean"]["diff"] == 8.5 - 19.5 / 4
    assert onset["min"]["pairing"] == "ok"
    # a group the audit has but the failed entry empties: the stats it lacks are missing
    sessions = result["groupings"]["split"]["sessions"]
    mean = next(
        p
        for p in sessions
        if p["id"] == "core.frames.onset_time" and p["stat"] == "mean"
    )
    assert mean["measured"] is None and mean["pairing"] == "missing_measured"
    assert set(result["groupings"]["class"]) == {"a", "b"}
    assert set(result["groupings"]["split"]) == {"sessions", "test", "train"}


def test_group_summaries_cover_the_same_entries_on_both_sides(tmp_path):
    """The audit measures 3 entries; the declaration fixes the key for only 2."""
    key = ("core.frames.centroid_x", None, None)
    measured = {"e1": 1.0, "e2": 2.0, "e3": 9.0}
    declared = {"e1": 1.0, "e2": 2.0}  # e3's declaration leaves the centroid out
    label = {"split": "train", "class": "a", "repeat": 0, "noise_repeat": 0}
    audit_rows = [
        {"entry": e, **label, "bundle": e, "values": [{"id": key[0], "value": v}]}
        for e, v in measured.items()
    ]
    declared_rows = [
        (
            {"entry": e, **label, "values": [{"id": key[0], "value": declared[e]}]}
            if e in declared
            else {"entry": e, **label, "values": []}
        )
        for e in measured
    ]

    def groups(rows):
        table = ValueTable([key], 4)
        for row in rows:
            table.add(
                row["split"],
                row["class"],
                {(v["id"], None, None): v["value"] for v in row["values"]},
            )
        dataset, by_class, by_split = table.groups()
        return dataset, {"class": by_class, "split": by_split}

    a_data, a_groups = groups(audit_rows)
    d_data, d_groups = groups(declared_rows)
    audit = _write_stage(
        tmp_path / "audit",
        "sensoryforge-audit/1",
        [_metric(key[0], "mm", resolution=None)],
        audit_rows,
        {"dataset": a_data, "groupings": a_groups},
    )
    declared_json = _write_stage(
        tmp_path / "declared",
        "sensoryforge-declared/1",
        [_metric(key[0], "mm")],
        declared_rows,
        {"dataset": d_data, "groupings": d_groups},
    )
    result = run_compare(audit, declared_json, tmp_path / "out", created_at=STAMP)
    for pairs in (result["dataset"], result["groupings"]["class"]["a"]):
        by_stat = {p["stat"]: p for p in pairs}
        # e3's 9.0 is not in the measured summary: the two sets are the same two entries
        assert by_stat["n"]["measured"] == by_stat["n"]["declared"] == 2
        assert by_stat["mean"]["measured"] == by_stat["mean"]["declared"] == 1.5
        assert by_stat["max"]["diff"] == 0.0
        assert all(p["diff"] == 0 for p in pairs)


def test_a_group_the_audit_lacks_pairs_as_missing_measured(by_hand):
    audit, declared, root = by_hand
    doc = json.loads(audit.read_text())
    del doc["groupings"]["class"]["b"]
    write_json(audit, doc)
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    assert {p["pairing"] for p in result["groupings"]["class"]["b"]} == {
        "missing_measured"
    }


def test_distributions_cover_the_entries_whose_difference_was_computed(by_hand):
    audit, declared, root = by_hand
    result = run_compare(
        audit, declared, root / "out", created_at=STAMP, histogram_bins=4
    )
    dist = result["distributions"]
    assert dist["histogram_bins"] == 4
    onset = next(d for d in dist["dataset"] if d["id"] == "core.frames.onset_time")
    # a/1 (10 vs 8.5) and b/1 (7 vs 7)
    assert onset["n_pairs"] == 2
    assert onset["diff_mean"] == 0.75 and onset["diff_abs_mean"] == 0.75
    assert onset["ks"] == 0.5 and onset["wasserstein"] == 0.75
    assert onset["histogram"]["edges"] == [7.0, 7.75, 8.5, 9.25, 10.0]
    assert onset["histogram"]["measured"] == [1, 0, 0, 1]
    assert onset["histogram"]["declared"] == [1, 0, 1, 0]
    # the string metric has no pairs at all: still listed, with nothing to say
    digest = next(d for d in dist["dataset"] if d["id"] == "core.frames.digest")
    assert digest["n_pairs"] == 0 and digest["ks"] is None
    assert digest["histogram"] is None
    # a class holds its own entries only
    a_onset = next(d for d in dist["class"]["a"] if d["id"] == "core.frames.onset_time")
    assert a_onset["n_pairs"] == 1 and a_onset["diff_mean"] == 1.5
    assert {d["n_pairs"] for d in dist["split"]["test"]} <= {0, 1}
    # a session has no class: it is in the data set and its split only
    assert set(dist["class"]) == {"a", "b"}


def test_the_files_validate_and_are_hashed(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    _validator().validate(result)
    row_validator = _validator("#/$defs/entry_row")
    rows = _entry_rows(root / "out")
    for row in rows:
        row_validator.validate(row)
    assert result["entries"]["n"] == len(rows) == 4
    assert result["entries"]["sha256"] == file_sha256(root / "out" / ENTRIES_FILE)
    assert result["inputs"]["audit"]["sha256"] == file_sha256(audit)
    assert result["inputs"]["declared"]["path"] == "../declared/declared.json"


def test_two_runs_differ_only_in_created_at(by_hand):
    audit, declared, root = by_hand
    one = run_compare(audit, declared, root / "one", created_at="2026-01-01T00:00:00Z")
    two = run_compare(audit, declared, root / "two", created_at="2027-01-01T00:00:00Z")
    assert one.pop("created_at") != two.pop("created_at")
    assert one == two
    assert (root / "one" / ENTRIES_FILE).read_bytes() == (
        root / "two" / ENTRIES_FILE
    ).read_bytes()


def test_the_inputs_may_be_folders(by_hand):
    audit, declared, root = by_hand
    by_file = run_compare(audit, declared, root / "f", created_at=STAMP)
    by_folder = run_compare(audit.parent, declared.parent, root / "f", created_at=STAMP)
    assert by_file == by_folder


def test_a_tampered_entry_file_is_refused(by_hand):
    audit, declared, root = by_hand
    path = audit.parent / "audit_entries.jsonl"
    path.write_text(path.read_text().replace("10.0", "11.0"))
    with pytest.raises(ValueError, match="sha256"):
        run_compare(audit, declared, root / "out", created_at=STAMP)


def test_files_of_other_data_sets_or_formats_are_refused(by_hand):
    audit, declared, root = by_hand
    with pytest.raises(ValueError, match="expected 'sensoryforge-audit/1'"):
        run_compare(declared, declared, root / "out")
    with pytest.raises(FileNotFoundError):
        run_compare(root / "nowhere", declared, root / "out")
    doc = json.loads(declared.read_text())
    rows = [
        json.loads(x)
        for x in (declared.parent / "declared_entries.jsonl").read_text().splitlines()
    ]
    rows[1]["entry"] = "a/9"
    n, sha = write_rows(declared.parent / "declared_entries.jsonl", rows)
    doc["entries"]["sha256"] = sha
    write_json(declared, doc)
    with pytest.raises(ValueError, match="different entries"):
        run_compare(audit, declared, root / "out")
    with pytest.raises(ValueError, match="histogram_bins"):
        run_compare(audit, declared, root / "out", histogram_bins=0)


def test_the_printed_tables_show_the_stored_numbers_unrounded(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    stat = next(
        p
        for p in result["dataset"]
        if p["id"] == "core.frames.onset_time" and p["stat"] == "mean"
    )
    table = format_stat_table(
        result, "dataset", stats=["mean"], metrics=["core.frames.onset_time"]
    )
    assert repr(stat["diff"]) in table and repr(stat["rel_diff"]) in table
    assert repr(stat["measured"]) in table and repr(stat["declared"]) in table
    header, rule, row = table.splitlines()
    assert header.split() == [
        "metric",
        "stat",
        "measured",
        "declared",
        "diff",
        "rel_diff",
        "unit",
        "pairing",
    ]
    assert row.split()[-2:] == ["ms", "ok"]
    # a missing value is a dash, not a zero
    assert "-" in format_stat_table(result, "split:sessions", stats=["mean"]).split()
    entry = format_entry_table(root / "out" / COMPARE_FILE, "a/1")
    assert repr(1.5) in entry and "unit_unchecked" in entry
    assert "ks" in format_distribution_table(result, "class:a").splitlines()[0]
    histograms = format_histograms(result, "core.frames.onset_time")
    assert "n_pairs=2" in histograms and "#" in histograms


def test_the_tables_refuse_an_unknown_group_or_statistic(by_hand):
    audit, declared, root = by_hand
    result = run_compare(audit, declared, root / "out", created_at=STAMP)
    with pytest.raises(ValueError, match="no class"):
        format_stat_table(result, "class:zzz")
    with pytest.raises(ValueError, match="group must be"):
        format_stat_table(result, "bogus")
    with pytest.raises(ValueError, match="unknown statistic"):
        format_stat_table(result, "dataset", stats=["median"])
    with pytest.raises(ValueError, match="no entry"):
        format_entry_table(root / "out" / COMPARE_FILE, "nope")


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def compared(tactile_dataset, audited, tmp_path_factory):
    from sensoryforge.evaluate.declared import run_declared

    audit, audit_dir = audited
    out = tmp_path_factory.mktemp("tactile_compare")
    declared = run_declared(tactile_dataset, out / "declared", created_at=STAMP)
    result = run_compare(audit_dir, out / "declared", out / "compare", created_at=STAMP)
    return audit, declared, result, out


def test_the_comparison_of_a_real_data_set_validates(compared):
    _, _, result, out = compared
    _validator().validate(result)
    row_validator = _validator("#/$defs/entry_row")
    rows = [
        json.loads(x) for x in (out / "compare" / ENTRIES_FILE).read_text().splitlines()
    ]
    assert len(rows) == result["entries"]["n"] == 14
    for row in rows:
        row_validator.validate(row)


def test_every_difference_is_rederived_from_the_two_entry_files(compared, audited):
    _, audit_dir = audited
    _, _, result, out = compared
    measured = {
        r["entry"]: {
            (v["id"], v.get("channel"), v.get("population")): v["value"]
            for v in r["values"]
        }
        for r in map(
            json.loads, (audit_dir / "audit_entries.jsonl").read_text().splitlines()
        )
    }
    declared = [
        json.loads(x)
        for x in (out / "declared" / "declared_entries.jsonl").read_text().splitlines()
    ]
    rows = [
        json.loads(x) for x in (out / "compare" / ENTRIES_FILE).read_text().splitlines()
    ]
    total = 0
    for drow, crow in zip(declared, rows):
        assert crow["entry"] == drow["entry"] and len(crow["pairs"]) == len(
            drow["values"]
        )
        for dval, pair in zip(drow["values"], crow["pairs"]):
            m = measured[drow["entry"]].get((dval["id"], dval.get("channel"), None))
            assert pair["id"] == dval["id"] and pair["declared"] == dval["value"]
            assert pair["measured"] == m
            if pair["diff"] is not None:
                assert pair["diff"] == m - dval["value"]
                total += 1
            else:
                assert m is None or dval["value"] is None
    assert total == sum(result["counts"][k] for k in ("ok", "unit_unchecked")) > 50
    assert sum(result["counts"].values()) == sum(len(r["values"]) for r in declared)


def test_the_real_data_set_shows_the_expected_pairings(compared):
    _, _, result, _ = compared
    counts = result["counts"]
    assert counts["unit_mismatch"] == 0 and counts["not_numeric"] == 0
    # the frames carry no unit, so the peak is a difference nobody checked
    peak = next(p for p in result["dataset"] if p["id"] == "core.frames.peak")
    assert peak["pairing"] == "unit_unchecked"
    onset = next(
        p
        for p in result["dataset"]
        if p["id"] == "core.frames.onset_time" and p["stat"] == "q50"
    )
    assert onset["pairing"] == "ok"
    # frame times sit on the grid; the declared ones do not: the gap is about dt
    for metric in result["distributions"]["dataset"]:
        if metric["id"] == "core.frames.onset_time":
            assert 0 < metric["diff_abs_mean"] < 2.0
    assert "core.frames.rms" in result["unpaired"]["measured_only"]
    assert result["unpaired"]["declared_only"] == []


def test_the_command_line_on_real_files(compared, audited, tmp_path, capsys):
    from sensoryforge.evaluate.compare.__main__ import main

    _, audit_dir = audited
    _, _, _, out = compared
    target = tmp_path / "cli"
    code = main(
        [
            str(audit_dir),
            str(out / "declared"),
            "--out",
            str(target),
            "--stats",
            "mean,q50",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0 and (target / COMPARE_FILE).is_file()
    assert "core.frames.onset_time" in printed and "q50" in printed
    assert (
        main(
            [
                str(audit_dir),
                str(out / "declared"),
                "--out",
                str(target),
                "--distributions",
                "--table",
                "class:dots",
            ]
        )
        == 0
    )
    assert "wasserstein" in capsys.readouterr().out
    assert (
        main(
            [
                str(audit_dir),
                str(out / "declared"),
                "--out",
                str(target),
                "--table",
                "entry:sessions/000",
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                str(audit_dir),
                str(out / "declared"),
                "--out",
                str(target),
                "--table",
                "class:zzz",
            ]
        )
        == 2
    )
    assert "no class" in capsys.readouterr().err
