"""The blind audit's reader layer, counts and metadata (roadmap task A2a)."""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.evaluate.audit import (
    DatasetReader,
    ReadError,
    UnitConflict,
    inventory,
    resolve_units,
)
from sensoryforge.evaluate.audit import allowlist
from sensoryforge.evaluate.audit.layout import centered_lattice
from sensoryforge.evaluate.audit.reader import EntryView, PopulationView
from sensoryforge.evaluate.audit.units import compose, data_units
from sensoryforge.world.render import Canvas

SENTINEL = "SENTINEL-DECLARATION-4f2a"


@pytest.fixture()
def data(toy_dataset, tmp_path):
    """A private copy of the toy data set's folder, safe to damage."""
    dest = tmp_path / "data"
    shutil.copytree(toy_dataset.data, dest, ignore=shutil.ignore_patterns(".partial"))
    return dest


def _strings(obj, seen=None):
    """Every string reachable in dataclasses, dicts, lists and tuples."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, np.ndarray):
        return
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from _strings(getattr(obj, f.name))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _strings(v)


# --------------------------------------------------------------------- layout
@pytest.mark.parametrize(
    "rows,cols,spacing,center",
    [(16, 16, 1.0, (0.0, 0.0)), (3, 5, 0.15, (0.5, -0.2)), (1, 4, 2.0, (0.0, 1.0))],
)
def test_lattice_equals_the_world_canvas(rows, cols, spacing, center):
    grid = GridConfig(
        name="g",
        arrangement="grid",
        rows=rows,
        cols=cols,
        spacing=spacing,
        center_x=center[0],
        center_y=center[1],
    )
    canvas = Canvas.from_grid_config(grid)
    want = torch.stack([canvas.xx.reshape(-1), canvas.yy.reshape(-1)], dim=1).numpy()
    got = centered_lattice(rows, cols, spacing, center)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_lattice_rejects_a_bad_grid():
    with pytest.raises(ValueError):
        centered_lattice(0, 3, 1.0)
    with pytest.raises(ValueError):
        centered_lattice(3, 3, 0.0)


# ---------------------------------------------------------------------- units
def test_data_units_prefer_the_block_to_a_suffix():
    assert data_units(None, True, True) == {
        "time": ("ms", "data.h5:dt_ms"),
        "space": ("mm", "config.json:grid.spacing_mm"),
    }
    named = data_units({"space": "px", "bogus": "x"}, True, True)
    assert named["space"] == ("px", "config.json:units.space")
    assert "bogus" not in named


def test_resolve_units_order_and_conflicts():
    data = {"time": ("ms", "data.h5:dt_ms")}
    pack = [("tactile", {"drive": ("mA", "units_and_gains.md"), "frame": ("N", "x")})]
    out = resolve_units(data, {"frame": "kPa"}, pack)
    assert out["time"] == {"symbol": "ms", "origin": "data", "source": "data.h5:dt_ms"}
    assert out["frame"]["symbol"] == "kPa" and out["frame"]["origin"] == "user"
    assert out["drive"]["origin"] == "pack:tactile"
    assert out["space"] is None and out["state"] is None
    with pytest.raises(UnitConflict, match="'ms'.*'s'"):
        resolve_units(data, {"time": "s"})
    with pytest.raises(UnitConflict):
        resolve_units(data, None, [("p", {"time": ("s", "doc")})])
    assert resolve_units(data, {"time": "ms"})["time"]["origin"] == "data"
    with pytest.raises(ValueError, match="unknown quantity"):
        resolve_units(data, {"length": "m"})


def test_compose():
    assert compose("mm", "ms") == "mm/ms"
    assert compose("1", "ms") == "1/ms"
    assert compose("mm", "1") == "mm"
    assert compose(None, "ms") is None and compose("mm", None) is None


# --------------------------------------------------------------- the allow-list
def test_allowlist_is_the_contract_table():
    """The contract's section 2, written out independently of the module."""
    assert set(allowlist.H5_ATTRS) == {
        "dt_ms",
        "integrate_dt_ms",
        "sensoryforge_version",
        "sensoryforge_sha",
    }
    assert set(allowlist.H5_EVENT_ATTRS) == {"signed", "polarity", "theta"}
    assert set(allowlist.INDEX_COLUMNS) == {
        "entry",
        "bundle",
        "status",
        "error",
        "seconds",
        "finished_at",
        "design_id",
        "sensoryforge_sha",
        "task",
    }
    assert set(allowlist.MANIFEST_LABELS) == {
        "entry",
        "split",
        "class",
        "repeat",
        "noise_repeat",
    }
    forbidden = {"draw", "bins", "probe", "seeds", "duration_ms", "truncated"}
    assert not forbidden & set(allowlist.MANIFEST_LABELS)
    assert "/meta" not in " ".join(allowlist.H5_DATASETS)
    assert not any(
        p.startswith(("config.stimulus", "design")) for p in allowlist.CONFIG_PATHS
    )


def test_a_view_carries_only_allowlisted_fields(toy_dataset):
    reader = DatasetReader(toy_dataset.data)
    view = reader.read_entry(reader.rows[0].entry)
    assert {f.name for f in dataclasses.fields(EntryView)} == {
        "entry",
        "split",
        "cls",
        "repeat",
        "noise_repeat",
        "frames",
        "time",
        "dt",
        "layout",
        "populations",
        "units",
        "seconds",
    }
    assert {f.name for f in dataclasses.fields(PopulationView)} == {
        "name",
        "readout",
        "data",
        "drive",
        "filtered",
        "attrs",
    }
    allowed_attrs = set(allowlist.H5_EVENT_ATTRS) | {"event_encoding"}
    for pop in view.populations.values():
        assert set(pop.attrs) <= allowed_attrs
    assert set(view.layout.summary()) == {
        "n_sensors",
        "dims",
        "n_channels",
        "channel_names",
        "bounds",
        "regular_grid",
    }


def test_nothing_excluded_reaches_a_view(data):
    """Declarations planted in every excluded place are not found in anything read."""
    with open(data / "manifest.jsonl", "r+", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
        for r in rows:
            r["draw"]["note"] = SENTINEL
            r["probe"] = SENTINEL
            r["bins"] = {"axis": SENTINEL}
            r["extra"] = SENTINEL
        fh.seek(0)
        fh.truncate()
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    (data / "dataset.json").write_text(json.dumps({"x": SENTINEL}))
    (data / "batch.json").write_text(json.dumps({"x": SENTINEL}))
    for bundle in data.glob("*/*/*/config.json"):
        cfg = json.loads(bundle.read_text())
        cfg["design"] = SENTINEL
        cfg["config"]["stimulus"] = {"type": SENTINEL}
        cfg["extra"] = SENTINEL
        cfg["config"]["grids"][0]["color"] = [SENTINEL]
        bundle.write_text(json.dumps(cfg))
        (bundle.parent / "stimuli" / "stimulus.json").write_text(SENTINEL)
        with h5py.File(bundle.parent / "data.h5", "r+") as f:
            f["meta"].attrs["config_yaml"] = SENTINEL
            f["meta"].attrs["extra"] = SENTINEL
    for pop_dir in data.glob("*/*/*/data.h5"):
        with h5py.File(pop_dir, "r+") as f:
            f["populations/luma/drive"].attrs["note"] = SENTINEL
            f["stimulus/frames"].attrs["note"] = SENTINEL
    reader = DatasetReader(data)
    inv = inventory(reader)
    assert inv.n_failed == 0
    seen = (
        list(_strings(reader.rows))
        + list(_strings(inv))
        + list(_strings(inv.header([], {})))
    )
    for row in reader.rows:
        seen += list(_strings(reader.read_entry(row.entry)))
    assert not [s for s in seen if SENTINEL in s]


def test_only_allowlisted_files_and_h5_keys_are_opened(data, monkeypatch):
    opened, keys = [], []

    class Recording(h5py.File):
        def __getitem__(self, name):
            keys.append(str(name))
            return super().__getitem__(name)

        def __contains__(self, name):
            keys.append(str(name))
            return super().__contains__(name)

    real_open = open

    def spy(file, *args, **kwargs):
        opened.append(Path(str(file)).as_posix())
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", Recording)
    monkeypatch.setattr("builtins.open", spy)
    reader = DatasetReader(data)
    inventory(reader)
    for row in reader.rows[:3]:
        reader.read_entry(row.entry)
    monkeypatch.undo()
    names = {Path(p).name for p in opened}
    assert names <= {"manifest.jsonl", "config.json"} | {
        p.name for p in (data / "index").glob("task_*.jsonl")
    }
    for banned in ("dataset.json", "batch.json", "stimulus.json", "sensoryforge.json"):
        assert banned not in names
    assert not any(".pt" in p for p in opened)
    assert keys and not any(k.lstrip("/").startswith("meta") for k in keys)


# --------------------------------------------------------------- reading entries
def test_read_entry_shapes_and_values(toy_dataset):
    reader = DatasetReader(toy_dataset.data)
    row = reader.rows[0]
    view = reader.read_entry(row.entry)
    with h5py.File(toy_dataset.data / row.entry / "data.h5", "r") as f:
        stored = np.asarray(f["stimulus/frames"])
        spikes = np.asarray(f["populations/luma/spikes"])
    assert view.frames.shape == (100, 3, 256)
    np.testing.assert_array_equal(view.frames, stored.reshape(100, 3, 256))
    assert view.dt == 1.0 and view.time.shape == (100,) and view.time[3] == 3.0
    assert list(view.populations) == ["luma", "chroma"]
    np.testing.assert_array_equal(view.populations["luma"].data, spikes)
    assert view.populations["chroma"].data.shape == (100, 32)
    assert view.populations["luma"].readout == "spikes"
    assert view.layout.channel_names == ("red", "green", "blue")
    assert not view.frames.flags.writeable
    with pytest.raises(ValueError):
        view.frames[0, 0, 0] = 1.0


def test_single_channel_frames_become_one_channel(data):
    """[T, H, W] frames read as [T, 1, M] with no channel names."""
    reader = DatasetReader(data)
    entry = reader.rows[0].entry
    bundle = data / entry
    cfg = json.loads((bundle / "config.json").read_text())
    cfg["config"]["grids"][0]["channels"] = None
    (bundle / "config.json").write_text(json.dumps(cfg))
    with h5py.File(bundle / "data.h5", "r+") as f:
        first = np.asarray(f["stimulus/frames"])[:, 0]
        del f["stimulus/frames"]
        f.create_dataset("stimulus/frames", data=first)
    info = DatasetReader(data).peek(entry)
    assert info.layout.n_channels == 1 and info.layout.channel_names is None
    view = DatasetReader(data).read_entry(entry)
    assert view.frames.shape == (100, 1, 256)
    np.testing.assert_array_equal(view.frames[:, 0], first.reshape(100, 256))


def test_labels_come_from_the_manifest(toy_dataset):
    reader = DatasetReader(toy_dataset.data)
    by_entry = {r.entry: r for r in reader.rows}
    assert [r.entry for r in reader.rows] == sorted(by_entry)
    assert len(by_entry) == len(toy_dataset.entries)
    row = by_entry["train/r0/00000"]
    assert (row.split, row.cls, row.repeat, row.noise_repeat) == (
        "train",
        "red_blob",
        0,
        0,
    )
    assert row.bundle == "train/r0/00000" and row.status == "ok"
    assert row.seconds is not None and row.seconds > 0


# ------------------------------------------------------------------- inventory
def test_inventory_counts_and_metadata(toy_dataset):
    inv = inventory(DatasetReader(toy_dataset.data))
    assert (inv.n_entries, inv.n_ok, inv.n_failed) == (12, 12, 0)
    counts = inv.counts()
    assert counts["by_split"] == {"test": 4, "train": 8}
    assert counts["by_class"] == {
        "blue_blob": 3,
        "dark": 1,
        "green_blob": 2,
        "red_blob": 6,
    }
    assert counts["by_split_class"]["test"] == {
        "blue_blob": 1,
        "dark": 1,
        "green_blob": 1,
        "red_blob": 1,
    }
    assert counts["n_repeats"] == 1 and counts["n_noise_repeats"] == 1
    assert inv.lengths() == {"dt": [1.0], "n_frames": {"100": 12}}
    assert inv.populations == (("luma", "spikes", 16), ("chroma", "spikes", 32))
    summary = inv.layout.summary()
    assert summary["n_sensors"] == 256 and summary["dims"] == 2
    assert summary["n_channels"] == 3
    assert summary["channel_names"] == ["red", "green", "blue"]
    assert summary["bounds"] == [[-7.5, 7.5], [-7.5, 7.5]]
    assert summary["regular_grid"] == {"rows": 16, "cols": 16, "spacing": 1.0}
    assert inv.units["time"] == {
        "symbol": "ms",
        "origin": "data",
        "source": "data.h5:dt_ms",
    }
    assert inv.units["space"]["symbol"] == "px"
    assert inv.units["space"]["source"] == "config.json:units.space"
    assert inv.units["frame"] is None and inv.units["drive"] is None
    assert len(inv.identifiers()["world_id"]) == 1
    assert inv.index_only == []


def test_header_validates_against_the_audit_schema(toy_dataset):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    root = Path(__file__).resolve().parents[2] / "schemas" / "evaluate"
    docs = {
        n: json.loads((root / n).read_text())
        for n in ("common.schema.json", "audit.schema.json")
    }
    registry = referencing.Registry().with_resources(
        [(d["$id"], referencing.Resource.from_contents(d)) for d in docs.values()]
    )
    inv = inventory(DatasetReader(toy_dataset.data))
    header = inv.header(
        [{"name": "core", "version": "1"}],
        {"core": {"zero_tolerance": 0.0, "histogram_bins": 32}},
        created_at="2026-10-22T14:03:11Z",
    )
    audit_id = docs["audit.schema.json"]["$id"]
    for key in (
        "format",
        "created_at",
        "producer",
        "inputs",
        "packs",
        "params",
        "layout",
        "populations",
        "units",
        "failures",
    ):
        schema = {"$ref": f"{audit_id}#/properties/{key}"}
        jsonschema.Draft202012Validator(schema, registry=registry).validate(header[key])
    entries = header["entries"]
    assert entries == {"n": 12, "n_ok": 12, "n_failed": 0}
    again = inv.header(
        [{"name": "core", "version": "1"}],
        {"core": {"zero_tolerance": 0.0, "histogram_bins": 32}},
        created_at="2026-10-22T14:03:11Z",
    )
    assert json.dumps(header, sort_keys=True) == json.dumps(again, sort_keys=True)


# --------------------------------------------------------------------- failures
def _rewrite_index(data, edit):
    path = next((data / "index").glob("task_*.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        edit(r)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_failures_are_recorded_by_stage(data):
    def edit(r):
        if r["entry"] == "train/r0/00001":
            r.update(status="failed", error="boom")

    _rewrite_index(data, edit)
    shutil.rmtree(data / "train/r0/00002")
    cfg_path = data / "train/r0/00003" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["grid"]["center_mm"] = [1.0, 0.0]
    cfg_path.write_text(json.dumps(cfg))
    (data / "train/r0/00004" / "data.h5").write_bytes(b"not an hdf5 file")
    inv = inventory(DatasetReader(data))
    got = {f["entry"]: (f["stage"], f["error"]) for f in inv.failures}
    assert got["train/r0/00001"] == ("batch", "boom")
    assert got["train/r0/00002"][0] == "read" and got["train/r0/00002"][1].startswith(
        "missing_bundle"
    )
    assert got["train/r0/00003"][1].startswith("layout_mismatch")
    assert got["train/r0/00004"][1].startswith("unreadable")
    assert inv.n_failed == 4 and inv.n_ok == 8 and inv.n_entries == 12
    assert inv.counts()["n_entries"] == 12  # failed entries still count
    assert [f["entry"] for f in inv.failures] == sorted(got)


def test_a_bundle_cannot_point_outside_the_data_set(data):
    _rewrite_index(
        data,
        lambda r: (
            r.update(bundle="../elsewhere") if r["entry"] == "train/r0/00005" else None
        ),
    )
    reader = DatasetReader(data)
    with pytest.raises(ReadError) as err:
        reader.read_entry("train/r0/00005")
    assert err.value.code in {"unreadable", "missing_bundle"}


def test_an_unlocatable_sensor_layout_is_refused(data):
    cfg_path = data / "train/r0/00006" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["config"]["grids"][0]["coords_file"] = "receptors.csv"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ReadError) as err:
        DatasetReader(data).peek("train/r0/00006")
    assert err.value.code == "layout_unknown"


def test_the_last_index_row_for_an_entry_wins(data):
    path = next((data / "index").glob("task_*.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    old = copy.deepcopy(rows[0])
    old.update(
        status="failed", error="first try", finished_at="2000-01-01T00:00:00+00:00"
    )
    path.write_text("\n".join(json.dumps(r) for r in [rows[0], old]) + "\n")
    reader = DatasetReader(data)
    assert {r.entry: r.status for r in reader.rows}[rows[0]["entry"]] == "ok"


def test_index_rows_without_a_manifest_entry_are_listed_not_counted(data):
    path = next((data / "index").glob("task_*.jsonl"))
    extra = json.loads(path.read_text().splitlines()[0])
    extra["entry"] = "ghost/0001"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(extra) + "\n")
    reader = DatasetReader(data)
    assert reader.index_only == ["ghost/0001"] and len(reader.rows) == 12


def test_manifest_problems_are_loud(data, tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetReader(tmp_path / "nothing")
    lines = (data / "manifest.jsonl").read_text().splitlines()
    (data / "manifest.jsonl").write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(ValueError, match="appears twice"):
        DatasetReader(data)
    (data / "manifest.jsonl").write_text("{not json\n")
    with pytest.raises(ValueError, match="not JSON"):
        DatasetReader(data)


def test_a_user_unit_that_contradicts_the_data_stops_the_audit(toy_dataset):
    reader = DatasetReader(toy_dataset.data, user_units={"space": "mm"})
    with pytest.raises(UnitConflict, match="'px'.*'mm'"):
        inventory(reader)
    ok = DatasetReader(toy_dataset.data, user_units={"frame": "a.u."})
    assert inventory(ok).units["frame"]["origin"] == "user"
