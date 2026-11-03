"""Run-to-run diff (roadmap A10): a change that alters nothing reports zero differences.

A tactile data set is compared with an untouched copy (identical), with a copy whose
float values moved by one float32 step (equivalent: platform rounding is not flagged)
and with copies altered in one entry in a known place (different, with the first
diverging frame, the array and the metric that moved named).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from sensoryforge.evaluate.diff import compare_arrays, run_diff
from sensoryforge.evaluate.diff.__main__ import main
from sensoryforge.evaluate.diff.arrays import (
    DIFF_RTOL,
    DIFFERENT,
    EQUIVALENT,
    IDENTICAL,
)
from sensoryforge.evaluate.diff.metrics import metric_deltas
from sensoryforge.testing.golden import FLOAT32_GOLDEN_ATOL

STAMP = "2026-10-23T16:00:00Z"


def copy(src, tmp_path, name="new"):
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def entries(folder):
    return [
        json.loads(x)["entry"]
        for x in (Path(folder) / "manifest.jsonl").read_text().splitlines()
    ]


def edit(folder, entry, fn):
    with h5py.File(Path(folder) / entry / "data.h5", "r+") as f:
        fn(f)


def rows(out):
    return [
        json.loads(x)
        for x in (Path(out) / "diff_entries.jsonl").read_text().splitlines()
    ]


@pytest.fixture(scope="module")
def identical(tactile_dataset, tmp_path_factory):
    root = tmp_path_factory.mktemp("diff_identical")
    new = copy(tactile_dataset, root)
    doc = run_diff(tactile_dataset, new, root / "out", created_at=STAMP)
    return doc, root / "out", new


# ---------------------------------------------------------- a change that does nothing


def test_a_no_op_reports_zero_differences(identical):
    doc, out, _ = identical
    n = doc["entries"]["n"]
    assert doc["verdict"] == "identical"
    assert doc["counts"]["identical"] == n and sum(doc["counts"].values()) == n
    assert doc["first_divergence"] is None and doc["by_array"] == {}
    assert doc["most_divergent"] == [] and doc["notes"] == []
    assert all(m["n_changed"] == 0 and m["n_pairs"] > 0 for m in doc["metrics"])
    assert all(r["status"] == "identical" and r["arrays"] == [] for r in rows(out))
    assert (
        doc["inputs"]["atol"] == FLOAT32_GOLDEN_ATOL
    )  # the golden fixtures' tolerance


def test_the_files_are_deterministic_and_valid(identical, tactile_dataset, tmp_path):
    doc, out, new = identical
    run_diff(tactile_dataset, new, tmp_path, created_at=STAMP)
    assert (out / "diff.json").read_bytes() == (tmp_path / "diff.json").read_bytes()
    assert (out / "diff_entries.jsonl").read_bytes() == (
        tmp_path / "diff_entries.jsonl"
    ).read_bytes()
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    root = Path(__file__).resolve().parents[2] / "schemas" / "evaluate"
    registry = Registry()
    for path in root.glob("*.json"):
        schema = json.loads(path.read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
        registry = registry.with_resource(path.name, Resource.from_contents(schema))
    schema = json.loads((root / "diff.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema, registry=registry)
    validator.validate(doc)
    row_schema = {"$ref": "diff.schema.json#/$defs/entry_row"}
    row_validator = jsonschema.Draft202012Validator(row_schema, registry=registry)
    for row in rows(out):
        row_validator.validate(row)


def test_pairing_reads_each_sides_index_a_fixed_number_of_times(
    identical, tactile_dataset, tmp_path, monkeypatch
):
    # Rebuilding both entry dicts for every entry made pairing O(N^2) (8000 entries:
    # 5.9 s just to pair).
    from sensoryforge.evaluate.audit.reader import DatasetReader

    reads = {"n": 0}
    original = DatasetReader.rows

    def counted(self):
        reads["n"] += 1
        return original.fget(self)

    monkeypatch.setattr(DatasetReader, "rows", property(counted))
    run_diff(tactile_dataset, identical[2], tmp_path, metrics=False, created_at=STAMP)
    assert len(entries(tactile_dataset)) > 4 and reads["n"] <= 6


def test_the_command_line_exit_code_for_no_difference(
    identical, tactile_dataset, tmp_path, capsys
):
    _, _, new = identical
    code = main(
        [str(tactile_dataset), str(new), "--out", str(tmp_path / "o"), "--no-metrics"]
    )
    assert code == 0 and "verdict: identical" in capsys.readouterr().out


# ----------------------------------------------------------------- platform rounding


def test_one_float32_step_is_equivalent_not_different(tactile_dataset, tmp_path):
    new = copy(tactile_dataset, tmp_path)
    entry = entries(new)[0]

    def nudge(f):
        name = next(iter(f["populations"]))
        d = f[f"populations/{name}/filtered"]
        a = d[...]
        a[5, 0] = np.nextafter(a[5, 0], np.float32(np.inf))
        d[...] = a

    edit(new, entry, nudge)
    doc = run_diff(
        tactile_dataset, new, tmp_path / "o", metrics=False, created_at=STAMP
    )
    assert doc["verdict"] == "equivalent"
    assert doc["counts"]["equivalent"] == 1 and doc["counts"]["different"] == 0
    row = next(r for r in rows(tmp_path / "o") if r["entry"] == entry)
    assert row["status"] == "equivalent" and row["arrays"][0]["status"] == "equivalent"
    assert row["arrays"][0]["n_differing"] == 0 and row["arrays"][0]["max_abs_diff"] > 0
    assert doc["first_divergence"] is None  # nothing beyond the tolerance
    # with a zero tolerance the same step is a difference
    strict = run_diff(
        tactile_dataset, new, tmp_path / "s", atol=0.0, metrics=False, created_at=STAMP
    )
    assert strict["verdict"] == "different" and strict["first_divergence"]["step"] == 5


# ----------------------------------------------------------------- a real difference


@pytest.fixture(scope="module")
def changed(tactile_dataset, tmp_path_factory):
    root = tmp_path_factory.mktemp("diff_changed")
    new = copy(tactile_dataset, root)
    target = entries(new)[3]
    other = entries(new)[6]

    def spike(f):
        d = f["populations/sa/spikes"]
        a = d[...]
        a[40, 0] = a[40, 0] + 1
        d[...] = a

    def early_frame(f):
        d = f["stimulus/frames"]
        a = d[...]
        a[10] = a[10] + 0.5
        d[...] = a

    edit(new, target, spike)
    edit(new, other, early_frame)
    doc = run_diff(tactile_dataset, new, root / "out", created_at=STAMP)
    return doc, root / "out", target, other, new


def test_a_changed_spike_is_found_at_its_frame(changed):
    doc, out, target, other, _ = changed
    assert doc["verdict"] == "different" and doc["counts"]["different"] == 2
    row = next(r for r in rows(out) if r["entry"] == target)
    assert (
        row["status"] == "different"
        and row["first_step"] == 40
        and row["first_ms"] == 40.0
    )
    arr = row["arrays"][0]
    assert (
        arr["name"] == "sa.spikes"
        and arr["n_differing"] == 1
        and arr["max_abs_diff"] == 1.0
    )
    assert row["largest_abs_diff"] == 1.0 and row["structure"] == []


def test_the_first_divergence_is_the_earliest_frame_in_the_data_set(changed):
    doc, _, target, other, _ = changed
    assert doc["first_divergence"] == {"entry": other, "step": 10, "ms": 10.0}
    assert {r["entry"] for r in doc["most_divergent"]} == {target, other}
    assert set(doc["by_array"]) >= {"sa.spikes", "frames"}
    assert doc["by_array"]["sa.spikes"] == {"n_equivalent": 0, "n_different": 1}


def test_metric_deltas_name_the_metric_and_the_entry_that_moved(changed):
    doc, _, target, _, _ = changed
    count = next(
        m
        for m in doc["metrics"]
        if m["id"] == "core.response.count" and m["population"] == "sa"
    )
    assert count["n_changed"] == 1 and count["max_entry"] == target
    assert count["max_abs_delta"] == 1.0 and count["mean_b"] > count["mean_a"]
    peak = next(m for m in doc["metrics"] if m["id"] == "core.frames.peak")
    assert peak["n_pairs"] == doc["entries"]["n"]
    untouched = next(m for m in doc["metrics"] if m["id"] == "core.frames.dt")
    assert untouched["n_changed"] == 0


def test_the_command_line_exit_code_for_a_difference(
    changed, tactile_dataset, tmp_path, capsys
):
    _, _, target, _, new = changed
    code = main([str(tactile_dataset), str(new), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert code == 1 and "verdict: different" in out and "first divergence" in out
    assert target in out and "core.response.count" in out


def test_audits_can_be_reused(tactile_dataset, changed, tmp_path):
    doc, out, *_ = changed
    again = run_diff(
        tactile_dataset,
        changed[4],
        tmp_path,
        audit_baseline=out / "audit_baseline",
        audit_new=out / "audit_new",
        created_at=STAMP,
    )
    assert again["metrics"] == doc["metrics"]
    assert not (tmp_path / "audit_baseline").exists()


# ----------------------------------------------------------------- structure


def test_entries_on_one_side_and_unreadable_ones(tactile_dataset, tmp_path):
    new = copy(tactile_dataset, tmp_path)
    names = entries(new)
    lines = (new / "manifest.jsonl").read_text().splitlines()
    (new / "manifest.jsonl").write_text("\n".join(lines[1:]) + "\n")  # drop the first
    shutil.rmtree(new / names[2])  # a bundle that is gone on one side
    doc = run_diff(
        tactile_dataset, new, tmp_path / "o", metrics=False, created_at=STAMP
    )
    by = {r["entry"]: r["status"] for r in rows(tmp_path / "o")}
    assert by[names[0]] == "only_a" and by[names[2]] == "unreadable"
    assert doc["verdict"] == "different"
    assert doc["counts"]["only_a"] == 1 and doc["counts"]["unreadable"] == 1


def test_an_entry_that_failed_on_both_sides_is_not_a_difference(
    tactile_dataset, tmp_path
):
    a, b = copy(tactile_dataset, tmp_path, "a"), copy(tactile_dataset, tmp_path, "b")
    victim = entries(a)[1]
    for side in (a, b):
        for path in sorted((side / "index").glob("task_*.jsonl")):
            lines = [json.loads(x) for x in path.read_text().splitlines()]
            hit = False
            for line in lines:
                if line["entry"] == victim:
                    line["status"], line["error"], hit = "failed", "boom", True
            if hit:
                path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    doc = run_diff(a, b, tmp_path / "o", metrics=False, created_at=STAMP)
    assert doc["counts"]["both_failed"] == 1 and doc["verdict"] == "identical"
    # failed on one side only is a difference
    c = copy(tactile_dataset, tmp_path, "c")
    doc = run_diff(a, c, tmp_path / "o2", metrics=False, created_at=STAMP)
    assert doc["counts"]["unreadable"] == 1 and doc["verdict"] == "different"


def test_a_different_data_set_is_noted(tactile_dataset, tmp_path):
    new = copy(tactile_dataset, tmp_path)
    meta = json.loads((new / "dataset.json").read_text())
    meta["dataset_id"] = "d-other"
    (new / "dataset.json").write_text(json.dumps(meta))
    doc = run_diff(
        tactile_dataset, new, tmp_path / "o", metrics=False, created_at=STAMP
    )
    assert any("different ids" in n for n in doc["notes"])


def test_arguments_are_validated(tactile_dataset, tmp_path, capsys):
    with pytest.raises(ValueError):
        run_diff(tactile_dataset, tactile_dataset, tmp_path, atol=-1.0)
    with pytest.raises(FileNotFoundError):
        run_diff(tactile_dataset, tmp_path / "nothing", tmp_path / "o")
    assert (
        main(
            [
                str(tactile_dataset),
                str(tmp_path / "nothing"),
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
    assert "diff:" in capsys.readouterr().err


# ----------------------------------------------------------------- the array comparison


def test_compare_arrays_statuses_and_first_step():
    a = np.zeros((6, 3), dtype=np.float32)
    assert compare_arrays(a, a.copy())["status"] == IDENTICAL
    b = a.copy()
    b[2, 1] = 1e-8
    assert compare_arrays(a, b)["status"] == EQUIVALENT
    b[4, 2] = 0.5
    got = compare_arrays(a, b)
    assert (
        got["status"] == DIFFERENT
        and got["first_step"] == 4
        and got["n_differing"] == 1
    )
    assert compare_arrays(a, b, atol=1.0)["status"] == EQUIVALENT
    assert (
        compare_arrays(a + 1, a + 1 + 0.4, atol=0.0, rtol=0.5)["status"] == EQUIVALENT
    )


@pytest.mark.parametrize("value", [0.5, 3.0, 40.0])
def test_one_float32_step_is_equivalent_at_any_magnitude(value):
    # A Linux run rounds the last step of a drive or filtered current differently from
    # the macOS baseline; at 40 mA that step (3.8e-6) is far above the golden atol.
    a = np.full((4, 3), value, dtype=np.float32)
    b = a.copy()
    b[1, 2] = np.nextafter(np.float32(value), np.float32(np.inf))
    assert compare_arrays(a, b)["status"] == EQUIVALENT
    b[1, 2] = a[1, 2] * np.float32(1 + 2**-18)
    assert compare_arrays(a, b)["status"] == DIFFERENT
    deltas = metric_deltas(
        {"e": {("m", None, None): float(a[1, 2])}},
        {"e": {("m", None, None): float(np.nextafter(a[1, 2], np.float32(99)))}},
        ["e"],
        FLOAT32_GOLDEN_ATOL,
        DIFF_RTOL,
    )
    assert deltas[0]["n_changed"] == 0


def test_integer_readouts_are_compared_exactly():
    a = np.full((3, 2), 2**25, dtype=np.int64)
    b = a.copy()
    b[0, 0] += 1  # within 2 float32 steps of the value, but a count is a count
    assert compare_arrays(a, b)["status"] == DIFFERENT
    deltas = metric_deltas(
        {"e": {("n", None, None): 2**25}},
        {"e": {("n", None, None): 2**25 + 1}},
        ["e"],
        FLOAT32_GOLDEN_ATOL,
        DIFF_RTOL,
    )
    assert deltas[0]["n_changed"] == 1


def test_compare_arrays_nan_and_shapes():
    a = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert compare_arrays(a, a.copy())["status"] == IDENTICAL
    b = a.copy()
    b[1, 0] = np.nan
    got = compare_arrays(a, b)
    assert got["status"] == DIFFERENT and got["first_step"] == 1
    short = compare_arrays(np.zeros((5, 2)), np.zeros((3, 2)))
    assert short["status"] == DIFFERENT and short["first_step"] == 3
    assert short["shapes"] == [[5, 2], [3, 2]]
    assert compare_arrays(np.zeros((2, 2)), np.zeros(4))["status"] == DIFFERENT
    assert compare_arrays(np.zeros((0, 2)), np.zeros((0, 2)))["status"] == IDENTICAL


# ----------------------------------------------------------------- the real thing


def test_two_engine_runs_of_one_data_set_show_no_difference(tactile_dataset, tmp_path):
    """The done-when: running one data set again (same code, same seed) is a no-op."""
    import yaml

    from sensoryforge.io.design import load_design
    from sensoryforge.world.dataset import build_dataset, load_dataset, write_dataset
    from sensoryforge.world.runner import run_dataset

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    raw = {
        "dataset": {
            "name": "tiny",
            "world": str(fixtures / "worlds" / "tactile_small.yml"),
            "seed": 5,
            "duration_ms": 120,
            "splits": {
                "train": {"n": 4},
                "test": {"stratified": {"bins": 1, "per_bin": 1}},
                "sessions": {"n": 1, "duration_ms": 200},
            },
        }
    }
    path = tmp_path / "tiny.yml"
    path.write_text(yaml.safe_dump(raw))
    spec = load_dataset(path)
    data = tmp_path / "again"
    write_dataset(spec, build_dataset(spec), data)
    assert run_dataset(load_design(fixtures / "design_8x8"), spec, data)["failed"] == 0
    doc = run_diff(tactile_dataset, data, tmp_path / "o", created_at=STAMP)
    assert doc["verdict"] == "identical", doc["counts"]
    assert doc["counts"]["identical"] == doc["entries"]["n"] == 14
