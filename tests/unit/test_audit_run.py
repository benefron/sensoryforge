"""The audit stage end to end (roadmap A2b and A2c): ``run_audit`` and its command line.

Runs on the toy non-tactile data set (``toy_dataset``): three colour channels on a
16 x 16 array, two spiking populations. The checks that need an independent answer (a
drift speed, a direction) compare the audit with what the fixture's constructor
documents.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.evaluate.audit.run import AUDIT_FILE, ENTRIES_FILE, run_audit
from sensoryforge.evaluate.audit.units import UnitConflict
from sensoryforge.evaluate.pack import MetricPack, metric_definition
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY

REPO = Path(__file__).resolve().parents[2]
STAMP = "2026-10-22T14:03:11Z"


@pytest.fixture(scope="module")
def audited(toy_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("audit_out")
    audit = run_audit(toy_dataset.data, out, created_at=STAMP)
    return audit, out


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


def test_audit_json_and_every_row_validate_against_the_schemas(audited):
    audit, out = audited
    _validator("").validate(json.loads((out / AUDIT_FILE).read_text()))
    row_schema = _validator("#/$defs/entry_row")
    rows = _rows(out)
    assert len(rows) == 12
    for row in rows:
        row_schema.validate(row)


def test_rows_cover_every_entry_sorted_and_the_file_is_hashed(audited):
    audit, out = audited
    rows = _rows(out)
    ids = [r["entry"] for r in rows]
    assert ids == sorted(ids) and audit["entries"]["n"] == 12
    assert (
        audit["entries"]["sha256"]
        == hashlib.sha256((out / ENTRIES_FILE).read_bytes()).hexdigest()
    )
    assert audit["entries"]["n_ok"] == 12 and audit["failures"] == []
    assert (
        list(audit)
        == [
            "format created_at producer inputs packs params layout populations units "
            "metrics entries failures dataset groupings dataset_values"
        ][0].split()
    )


def test_every_row_follows_the_catalogue_order(audited):
    audit, out = audited
    n_ch = audit["layout"]["n_channels"]
    pops = {p["name"]: p["readout"] for p in audit["populations"]}
    expected = []
    for m in audit["metrics"]:
        if m["level"] != "entry":
            continue
        if m["scope"] == "channel":
            expected += [(m["id"], c, None) for c in range(n_ch)]
        elif m["scope"] == "population":
            expected += [
                (m["id"], None, p)
                for p, r in pops.items()
                if "applies_to" not in m or r in m["applies_to"]
            ]
        else:
            expected.append((m["id"], None, None))
    for row in _rows(out):
        got = [(v["id"], v.get("channel"), v.get("population")) for v in row["values"]]
        assert got == expected


def test_summaries_cover_data_set_class_and_split(audited):
    audit, _ = audited
    assert audit["dataset"]["n_entries"] == 12 and audit["dataset"]["n_failed"] == 0
    assert set(audit["groupings"]["split"]) == {"train", "test"}
    assert len(audit["groupings"]["class"]) >= 2
    ids = {s["id"] for s in audit["dataset"]["summaries"]}
    assert "core.frames.peak" in ids and "core.frames.digest" not in ids
    assert "core.dataset.silent_entries" not in ids  # level dataset: no summary
    for cls in audit["groupings"]["class"].values():
        assert cls["n_entries"] >= 1
    assert sum(g["n_entries"] for g in audit["groupings"]["split"].values()) == 12


def test_toy_drift_is_measured_as_documented(audited):
    """The fixture drifts a blob at 0.02 to 0.06 px/ms in the channel of its class."""
    audit, out = audited
    speeds = [
        v["value"]
        for row in _rows(out)
        for v in row["values"]
        if v["id"] == "core.frames.centroid_speed" and v["value"] is not None
    ]
    assert speeds, "no entry moved"
    assert all(0.015 <= s <= 0.07 for s in speeds), speeds
    units = audit["units"]
    assert units["space"]["symbol"] == "px" and units["frame"] is None
    speed_def = next(
        m for m in audit["metrics"] if m["id"] == "core.frames.centroid_speed"
    )
    assert speed_def["unit"] == {"symbol": "px/ms", "origin": "derived"}
    # a blob of 1.5 to 2.5 px sits in one channel only: the others are silent
    by_entry = {}
    for row in _rows(out):
        for v in row["values"]:
            if v["id"] == "core.frames.bouts":
                by_entry.setdefault(row["entry"], []).append(v["value"])
    for entry, bouts in by_entry.items():
        assert sum(1 for b in bouts if b) <= 1, entry


def test_dataset_values_count_silent_entries(audited):
    audit, out = audited
    silent = sum(
        1
        for row in _rows(out)
        if all(v["value"] == 0 for v in row["values"] if v["id"] == "core.frames.bouts")
    )
    values = {v["id"]: v["value"] for v in audit["dataset_values"]}
    assert values["core.dataset.silent_entries"] == silent
    assert values["core.dataset.duplicate_frames"] == 0


def test_two_audits_differ_only_in_created_at(toy_dataset, tmp_path):
    a = run_audit(toy_dataset.data, tmp_path / "a", created_at="2026-01-01T00:00:00Z")
    b = run_audit(toy_dataset.data, tmp_path / "b", created_at="2027-02-02T00:00:00Z")
    assert a["created_at"] != b["created_at"]
    a.pop("created_at"), b.pop("created_at")
    assert a == b
    assert (tmp_path / "a" / ENTRIES_FILE).read_bytes() == (
        tmp_path / "b" / ENTRIES_FILE
    ).read_bytes()
    text_a = (tmp_path / "a" / AUDIT_FILE).read_text()
    text_b = (tmp_path / "b" / AUDIT_FILE).read_text()
    swap = lambda t, s: t.replace(s, "STAMP")  # noqa: E731
    assert swap(text_a, "2026-01-01T00:00:00Z") == swap(text_b, "2027-02-02T00:00:00Z")


def test_a_missing_bundle_fails_only_its_entry(toy_dataset, tmp_path):
    work = tmp_path / "data"
    shutil.copytree(toy_dataset.data, work)
    rows = [json.loads(x) for x in (work / "manifest.jsonl").read_text().splitlines()]
    gone = rows[3]["entry"]
    shutil.rmtree(work / gone)
    audit = run_audit(work, tmp_path / "out", created_at=STAMP)
    assert audit["entries"]["n"] == 12 and audit["entries"]["n_failed"] == 1
    assert audit["failures"][0]["entry"] == gone
    assert audit["failures"][0]["stage"] == "read"
    assert audit["dataset"]["n_failed"] == 1
    rows_out = {r["entry"]: r for r in _rows(tmp_path / "out")}
    assert len(rows_out) == 12 and rows_out[gone]["values"] == []
    peak = next(
        s for s in audit["dataset"]["summaries"] if s["id"] == "core.frames.peak"
    )
    assert peak["n"] + peak["n_null"] == 11  # a summary per channel; 11 entries read
    _validator("").validate(json.loads((tmp_path / "out" / AUDIT_FILE).read_text()))


class _Bomb(MetricPack):
    """A pack that cannot measure the entry whose id ends in 0001."""

    name = "bomb"
    version = "1"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("bomb.x.y", None, scope="entry")]

    def measure(self, view):
        if view.entry.endswith("0001"):
            raise ValueError("cannot measure")
        return [{"id": "bomb.x.y", "value": 1.0}]


def test_a_pack_that_raises_fails_that_entry_at_measure(toy_dataset, tmp_path):
    METRIC_PACK_REGISTRY.register("bomb", _Bomb)
    audit = run_audit(
        toy_dataset.data, tmp_path, packs=["core", "bomb"], created_at=STAMP
    )
    bad = [f for f in audit["failures"] if f["stage"] == "measure"]
    assert bad and all("cannot measure" in f["error"] for f in bad)
    assert all(f["entry"].endswith("0001") for f in bad)
    assert [p["name"] for p in audit["packs"]] == ["core", "bomb"]
    assert "bomb" in audit["params"]
    row = next(r for r in _rows(tmp_path) if r["entry"] == bad[0]["entry"])
    assert row["values"] == []


def test_user_units_fill_what_the_data_does_not_name(toy_dataset, tmp_path):
    audit = run_audit(
        toy_dataset.data, tmp_path, user_units={"frame": "kPa"}, created_at=STAMP
    )
    assert audit["units"]["frame"] == {
        "symbol": "kPa",
        "origin": "user",
        "source": "--units",
    }
    peak = next(m for m in audit["metrics"] if m["id"] == "core.frames.peak")
    assert peak["unit"]["symbol"] == "kPa"
    with pytest.raises(UnitConflict, match="space"):
        run_audit(toy_dataset.data, tmp_path / "x", user_units={"space": "mm"})


def test_zero_tolerance_changes_what_is_active(toy_dataset, tmp_path):
    base = run_audit(toy_dataset.data, tmp_path / "a", created_at=STAMP)
    strict = run_audit(
        toy_dataset.data,
        tmp_path / "b",
        params={"core": {"zero_tolerance": 0.6}},
        created_at=STAMP,
    )
    assert strict["params"]["core"]["zero_tolerance"] == 0.6

    def active(a):
        s = [
            x for x in a["dataset"]["summaries"] if x["id"] == "core.frames.active_time"
        ]
        return sum(x["mean"] for x in s)

    assert active(strict) < active(base)


def test_unknown_or_repeated_packs_and_stray_parameters_are_refused(
    toy_dataset, tmp_path
):
    with pytest.raises(KeyError):
        run_audit(toy_dataset.data, tmp_path, packs=["nope"])
    with pytest.raises(ValueError, match="twice"):
        run_audit(toy_dataset.data, tmp_path, packs=["core", "core"])
    with pytest.raises(ValueError, match="not run"):
        run_audit(toy_dataset.data, tmp_path, params={"other": {"a": 1}})
    with pytest.raises(FileNotFoundError):
        run_audit(tmp_path / "empty", tmp_path)


def test_the_audit_runs_in_a_clean_subprocess_from_the_folder_alone(
    toy_dataset, tmp_path
):
    """No repository checkout of the world, no environment: ``python -I`` and a path."""
    out = tmp_path / "cli"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "sys.argv = ['audit'] + sys.argv[2:];"
        "from sensoryforge.evaluate.audit.__main__ import main;"
        "rc = main();"
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[:2] == ['sensoryforge', 'world']);"
        "print('WORLD', bad); sys.exit(rc)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(REPO), str(toy_dataset.data),
         "--out", str(out), "--units", "frame=kPa", "--param", "core.histogram_bins=8"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "WORLD []" in done.stdout, done.stdout
    audit = json.loads((out / AUDIT_FILE).read_text())
    assert audit["entries"]["n"] == 12
    assert audit["params"]["core"]["histogram_bins"] == 8
    sample = audit["dataset"]["summaries"][0]
    assert len(sample["histogram"]["counts"]) <= 8
    _validator("").validate(audit)


def test_the_command_line_reports_errors_with_status_2(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "sensoryforge.evaluate.audit", str(tmp_path / "none")],
        capture_output=True, text=True, timeout=120, cwd=REPO,
    )  # fmt: skip
    assert done.returncode == 2 and "audit:" in done.stderr


def test_the_pack_example_in_the_docs_runs(toy_dataset, tmp_path):
    text = (REPO / "docs" / "developer_guide" / "audit_metrics.md").read_text()
    code = text.split("```python")[2].split("```")[0]
    namespace: dict = {}
    exec(compile(code, "audit_metrics.md", "exec"), namespace)  # noqa: S102
    audit = run_audit(
        toy_dataset.data, tmp_path, packs=["core", "peaks"], created_at=STAMP
    )
    ids = [m["id"] for m in audit["metrics"]]
    assert ids[-1] == "peaks.frames.n_sensors_over_half"
    s = next(
        x
        for x in audit["dataset"]["summaries"]
        if x["id"] == "peaks.frames.n_sensors_over_half"
    )
    assert s["n"] == 12 and s["max"] > 0
