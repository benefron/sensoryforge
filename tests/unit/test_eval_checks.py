"""Reference-free checks (roadmap A8): every check passes on a good data set and fails
on a seeded fault.

A check that never fails proves nothing, so each one is run against the fault it is
meant to catch: an array corrupted in a copy of the data set, or the engine patched to
break one promise (determinism, symmetry, silence, finiteness, monotonicity, the input
floor).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.evaluate import checks as ck
from sensoryforge.evaluate.checks import data_checks, engine_checks, world_checks
from sensoryforge.evaluate.checks import probe as pb
from sensoryforge.evaluate.checks.base import CheckContext, CheckSkipped
from sensoryforge.evaluate.checks.core_pack import CoreChecks
from sensoryforge.evaluate.checks.run import build_check_packs, run_one
from sensoryforge.evaluate.stage_io import clean
from sensoryforge.evaluate.server import Api, DatasetService, Request
from sensoryforge.evaluate.server import api as api_module
from sensoryforge.evaluate.tactile.checks import MonotonicRate, TactileChecks

STAMP = "2026-10-23T12:00:00Z"
ALL_IDS = [c.id for c in CoreChecks().checks()] + ["tactile.rate.monotonic"]


@pytest.fixture(scope="module")
def baseline(tactile_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("checks_out")
    doc = ck.run_checks(
        tactile_dataset, out, packs=["core", "tactile"], sample=3, created_at=STAMP
    )
    return doc, out


@pytest.fixture()
def broken(tactile_dataset, tmp_path):
    """A private copy of the tactile data set that a test may damage."""
    dst = tmp_path / "data"
    shutil.copytree(tactile_dataset, dst)
    return dst


def ctx_for(folder, **kw):
    kw.setdefault("sample", 3)
    return CheckContext(Path(folder), **kw)


def first_bundle(folder):
    rows = [
        json.loads(line)
        for line in (Path(folder) / "manifest.jsonl").read_text().splitlines()
    ]
    return Path(folder) / rows[0]["entry"], rows[0]["entry"]


def patch_neuron_stage(monkeypatch, wrap):
    """Replace the engine's neuron stage by ``wrap(original)(...)``."""
    original = SimulationEngine._run_neuron_stage
    monkeypatch.setattr(
        SimulationEngine, "_run_neuron_stage", staticmethod(wrap(original))
    )


# ------------------------------------------------------------------ a good data set


def test_a_good_data_set_passes_every_check(baseline):
    doc, _ = baseline
    assert [c["id"] for c in doc["checks"]] == ALL_IDS
    bad = [
        (c["id"], c["status"], c["note"])
        for c in doc["checks"]
        if c["status"] != "pass"
    ]
    assert not bad, bad
    assert doc["counts"] == {"pass": len(ALL_IDS), "fail": 0, "skipped": 0, "error": 0}
    assert doc["entries_with_failures"] == []
    assert doc["format"] == ck.CHECKS_FORMAT


def test_the_file_is_deterministic_and_valid(baseline, tactile_dataset, tmp_path):
    doc, out = baseline
    again = tmp_path / "again"
    ck.run_checks(
        tactile_dataset, again, packs=["core", "tactile"], sample=3, created_at=STAMP
    )
    assert (out / "checks.json").read_bytes() == (again / "checks.json").read_bytes()
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    root = Path(__file__).resolve().parents[2] / "schemas" / "evaluate"
    registry = Registry()
    for path in root.glob("*.json"):
        schema = json.loads(path.read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
        registry = registry.with_resource(path.name, Resource.from_contents(schema))
    schema = json.loads((root / "checks.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(doc)


def test_numbers_are_the_raw_values_behind_a_verdict(baseline):
    by_id = {c["id"]: c for c in baseline[0]["checks"]}
    mirror = by_id["core.run.mirror_symmetry"]["numbers"]
    assert mirror["largest_difference_x"] == 0.0 and mirror["total_events"] > 0
    mono = by_id["tactile.rate.monotonic"]["numbers"]
    assert any(k.endswith("mean_count_by_level") for k in mono)
    assert (
        by_id["core.run.voltage"]["numbers"]["lowest_mv"]
        < by_id["core.run.voltage"]["numbers"]["highest_mv"]
    )


def test_the_toy_data_set_with_three_channels(toy_dataset, tmp_path):
    doc = ck.run_checks(toy_dataset.data, tmp_path, sample=2, created_at=STAMP)
    status = {c["id"]: c["status"] for c in doc["checks"]}
    assert set(status.values()) == {"pass"}, doc["checks"]
    mirror = next(c for c in doc["checks"] if c["id"] == "core.run.mirror_symmetry")
    assert mirror["numbers"]["total_events"] > 0 and mirror["tested"] > 0
    det = next(c for c in doc["checks"] if c["id"] == "core.run.determinism")
    assert "populations without a seed" in det["note"]  # the toy leaves wiring unseeded


# ------------------------------------------------------------------ data faults


def damage(folder, fn):
    bundle, entry = first_bundle(folder)
    with h5py.File(bundle / "data.h5", "r+") as f:
        fn(f)
    return entry


def run_data(folder, check_cls):
    return run_one(check_cls(), ctx_for(folder))


def test_nan_in_an_array_fails_finite_and_names_the_entry(broken):
    def poke(f):
        name = next(iter(f["populations"]))
        f[f"populations/{name}/filtered"][3, 0] = np.nan

    entry = damage(broken, poke)
    res = run_data(broken, data_checks.Finite)
    assert res.status == "fail" and res.failed == 1
    assert (
        res.failures[0].entry == entry
        and "filtered has 1 non-finite" in res.failures[0].detail
    )


def test_negative_and_fractional_counts_fail_readout_values(broken):
    def poke(f):
        name = next(iter(f["populations"]))
        data = f[f"populations/{name}/spikes"]
        a = data[...].astype(np.float64)
        a[2, 0] = -1.0
        a[4, 1] = 0.5
        data[...] = a.astype(data.dtype)

    try:
        entry = damage(broken, poke)
    except TypeError:
        pytest.skip("spikes are stored as booleans here")
    res = run_data(broken, data_checks.ReadoutValues)
    assert res.status == "fail" and res.failures[0].entry == entry


def test_a_shifted_time_axis_fails_time_grid(broken):
    def poke(f):
        t = f["time_ms"][...]
        t[5:] += 0.5
        f["time_ms"][...] = t

    entry = damage(broken, poke)
    res = run_data(broken, data_checks.TimeGrid)
    assert res.status == "fail" and res.failures[0].entry == entry
    assert "steps differ" in res.failures[0].detail


@pytest.mark.parametrize("dt", [0.1, 0.2, 1.0])
def test_the_writers_float32_time_axis_passes_time_grid(dt):
    # io/bundle.py stores float32 arange(T) * dt, whose steps round by up to one
    # float32 step of the largest time (2935 of 3000 steps at dt 0.1).
    n = 5000
    time = (np.arange(n, dtype=np.float32) * np.float32(dt)).astype(np.float32)
    view = SimpleNamespace(
        time=time, dt=dt, frames=np.zeros((n, 1, 1), np.float32), populations={}
    )
    assert data_checks._time_problems(view) == []
    time[100:] += np.float32(dt / 100)
    assert "steps differ" in data_checks._time_problems(view)[0]


def test_a_missing_bundle_fails_readable(broken):
    bundle, entry = first_bundle(broken)
    shutil.rmtree(bundle)
    res = run_data(broken, data_checks.Readable)
    assert res.status == "fail" and res.failures[0].entry == entry
    assert res.tested == res.failed + (res.tested - res.failed) and res.failed >= 1


def test_a_failed_index_row_fails_readable(broken):
    for path in (broken / "index").glob("task_*.jsonl"):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        lines[0]["status"], lines[0]["error"] = "failed", "boom"
        path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        break
    res = run_data(broken, data_checks.Readable)
    assert res.status == "fail" and "boom" in res.failures[0].detail


def test_a_whole_run_lists_the_failed_entries(broken, tmp_path):
    def poke(f):
        name = next(iter(f["populations"]))
        f[f"populations/{name}/drive"][0, 0] = np.inf

    entry = damage(broken, poke)
    doc = ck.run_checks(broken, tmp_path, sample=1, created_at=STAMP)
    assert doc["counts"]["fail"] >= 1
    assert {"entry": entry, "checks": ["core.data.finite"]} in doc[
        "entries_with_failures"
    ]


# ------------------------------------------------------------------ engine faults


def test_a_run_that_differs_on_its_second_pass_fails_determinism(
    tactile_dataset, monkeypatch
):
    original = SimulationEngine.run
    calls = {"n": 0}

    def drifting(self, stimulus, *a, **kw):
        out = original(self, stimulus, *a, **kw)
        calls["n"] += 1
        for arrays in out.values():  # every call drifts a little more than the last
            arrays["filtered"] = arrays["filtered"] + 1e-3 * calls["n"]
        return out

    monkeypatch.setattr(SimulationEngine, "run", drifting)
    res = engine_checks.Determinism().run(ctx_for(tactile_dataset))
    assert res.status == "fail"
    assert any("second run" in f.where for f in res.failures)


def test_a_fresh_build_that_differs_fails_on_seeded_populations(
    tactile_dataset, monkeypatch
):
    # An unseeded population elsewhere once turned every fresh-build difference into a
    # note (F-5607a62). Seed sa, leave ra unseeded, and make every rebuilt engine drift.
    original_config = pb.base_config

    def half_seeded(ctx):
        cfg = original_config(ctx)
        next(p for p in cfg.populations if p.name == "sa").seed = 3
        return cfg

    original = SimulationEngine.run
    engines = []

    def rebuilt_drifts(self, stimulus, *a, **kw):
        out = original(self, stimulus, *a, **kw)
        if self not in engines:
            engines.append(self)
        if self is not engines[0]:
            for arrays in out.values():
                arrays["filtered"] = arrays["filtered"] + 1e-3
        return out

    monkeypatch.setattr(pb, "base_config", half_seeded)
    monkeypatch.setattr(SimulationEngine, "run", rebuilt_drifts)
    res = engine_checks.Determinism().run(ctx_for(tactile_dataset))
    assert res.status == "fail"
    assert [f.detail for f in res.failures] == ["sa.filtered differs"]
    assert res.note.endswith(": ra")


def test_unseeded_populations_are_always_named(tactile_dataset):
    res = engine_checks.Determinism().run(ctx_for(tactile_dataset))
    assert res.status == "pass" and res.note.endswith(": ra, sa")


def test_position_dependent_input_fails_mirror_symmetry(tactile_dataset, monkeypatch):
    import torch

    def wrap(original):
        def biased(filtered, neuron, **kw):
            ramp = torch.linspace(0, 1, filtered.shape[-1]).expand_as(filtered)
            return original(filtered + 50.0 * ramp, neuron, **kw)

        return biased

    patch_neuron_stage(monkeypatch, wrap)
    res = engine_checks.MirrorSymmetry().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and {f.where for f in res.failures} >= {
        "mirror_x (rows)"
    }


def test_a_model_that_fires_on_its_own_fails_empty_stimulus(
    tactile_dataset, monkeypatch
):
    def wrap(original):
        def hot(filtered, neuron, **kw):
            return original(filtered + 400.0, neuron, **kw)

        return hot

    patch_neuron_stage(monkeypatch, wrap)
    res = engine_checks.EmptyStimulus().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and "with no input" in res.failures[0].detail


def test_nan_under_saturation_fails(tactile_dataset, monkeypatch):
    def wrap(original):
        def blows_up(filtered, neuron, **kw):
            out = original(filtered, neuron, **kw)
            if filtered.abs().max() > 500:
                out["voltages"] = out["voltages"] * float("nan")
                out["spikes"] = out["spikes"] * float("nan")
            return out

        return blows_up

    patch_neuron_stage(monkeypatch, wrap)
    res = engine_checks.Saturation().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and "non-finite" in res.failures[0].detail


def test_event_converters_without_a_refractory_period_are_not_bounded(
    converter_dataset,
):
    # level_crossing and sigma_delta emit floor(|d| / theta) events in one step by
    # design; the per-step spike bound does not apply to them.
    res = engine_checks.Saturation().run(ctx_for(converter_dataset(0.0)))
    assert res.status == "pass", res.failures
    assert res.numbers["count_bounds"] == {"ra": None, "sa": None}
    assert "ra" in res.note and "sa" in res.note


def test_event_converters_with_a_refractory_period_count_once_per_period(
    converter_dataset,
):
    res = engine_checks.Saturation().run(ctx_for(converter_dataset(0.25)))
    steps = res.numbers["steps_per_bin"]
    assert res.numbers["count_bounds"] == {"ra": 4.0, "sa": 4.0} and steps == 20.0
    assert res.status == "pass" and res.note is None
    assert 0 < res.numbers["most_counts_in_a_bin"] <= 4


def test_a_stimulus_that_never_reaches_the_neurons_fails_single_sensor(
    tactile_dataset, monkeypatch
):
    original = pb.run

    def silent(probe, frames, seed):
        return original(probe, np.zeros_like(frames), seed)

    monkeypatch.setattr(pb, "run", silent)
    res = engine_checks.SingleSensor().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and "no drive" in res.failures[-1].detail


def test_falling_rate_with_pressure_fails_monotonic(tactile_dataset, monkeypatch):
    def wrap(original):
        def inverted(filtered, neuron, **kw):
            return original(12.0 - filtered, neuron, **kw)

        return inverted

    patch_neuron_stage(monkeypatch, wrap)
    res = MonotonicRate().run(ctx_for(tactile_dataset))
    assert res.status == "fail", res
    assert "neuron" in res.failures[0].where and res.numbers


def test_a_tolerance_accepts_small_falls(tactile_dataset, monkeypatch):
    assert MonotonicRate(tolerance=1e9).run(ctx_for(tactile_dataset)).status == "pass"
    with pytest.raises(ValueError):
        MonotonicRate(tolerance=-1)


def test_a_stored_readout_that_cannot_be_reproduced_fails_rerun(broken):
    def poke(f):
        for name in f["populations"]:
            d = f[f"populations/{name}/spikes"]
            a = d[...]
            a[10, :] = a[10, :] + 1
            d[...] = a

    for bundle in [first_bundle(broken)[0]]:
        with h5py.File(bundle / "data.h5", "r+") as f:
            poke(f)
    res = engine_checks.RerunMatches().run(ctx_for(broken, sample=1))
    assert res.status == "fail" and res.failures[0].entry


def test_a_rerun_failure_carries_the_reruns_notes(broken, monkeypatch):
    from sensoryforge.evaluate.server import resim

    original = resim.resimulate

    def noted(*a, **kw):
        out = original(*a, **kw)
        out.verification["matches"] = False
        out.verification["notes"] = ["membrane noise has no noise_seed"]
        return out

    monkeypatch.setattr(resim, "resimulate", noted)
    res = engine_checks.RerunMatches().run(ctx_for(broken, sample=1))
    assert res.status == "fail"
    assert res.failures[0].detail.endswith("; membrane noise has no noise_seed")


def test_an_ignored_floor_fails_input_floor(tactile_dataset, monkeypatch):
    def wrap(original):
        def no_floor(filtered, neuron, **kw):
            kw["input_floor"] = None
            return original(filtered, neuron, **kw)

        return no_floor

    patch_neuron_stage(monkeypatch, wrap)
    res = engine_checks.InputFloor().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and "mA" in res.failures[0].detail


def test_voltage_bounds_are_checked_only_when_given(tactile_dataset):
    ok = engine_checks.Voltage().run(ctx_for(tactile_dataset))
    assert ok.status == "pass" and "no limits" in ok.note
    tight = engine_checks.Voltage().run(ctx_for(tactile_dataset, bounds_mv=(-1.0, 1.0)))
    assert tight.status == "fail" and "outside -1 to 1 mV" in tight.failures[0].detail
    wide = engine_checks.Voltage().run(ctx_for(tactile_dataset, bounds_mv=(-1e6, 1e6)))
    assert wide.status == "pass" and wide.note is None


def test_a_sampler_that_depends_on_n_fails_prefix_stable(tactile_dataset, monkeypatch):
    import sensoryforge.world.sampling as sampling

    original = sampling.sample

    def leaky(world, n=None, seed=0, **kw):
        draws = original(world, n, seed, **kw)
        if n == 2 * world_checks.N_DRAWS:
            draws[3] = dataclasses.replace(draws[3], draw_seed=draws[3].draw_seed + 1)
        return draws

    monkeypatch.setattr(sampling, "sample", leaky)
    res = world_checks.PrefixStable().run(ctx_for(tactile_dataset))
    assert res.status == "fail" and res.failures[0].where == "draw 3"


# ------------------------------------------------------------------ skips and errors


def test_a_config_the_engine_cannot_build_skips_the_probes(broken):
    bundle, _ = first_bundle(broken)
    cfg = json.loads((bundle / "config.json").read_text())
    for pop in cfg["config"]["populations"]:
        pop["innervation_params"] = {"path": "/no/such/file.npz"}
    (bundle / "config.json").write_text(json.dumps(cfg))
    res = run_one(engine_checks.Determinism(), ctx_for(broken))
    assert res.status == "skipped" and "cannot be built" in res.note


def test_a_data_set_without_a_world_skips_prefix_stable(broken):
    meta = json.loads((broken / "dataset.json").read_text())
    del meta["world"]
    (broken / "dataset.json").write_text(json.dumps(meta))
    res = run_one(world_checks.PrefixStable(), ctx_for(broken))
    assert res.status == "skipped" and "no usable world" in res.note


def test_all_zero_frames_leave_the_probes_without_a_scale(broken):
    def zero(f):
        f["stimulus/frames"][...] = 0

    with h5py.File(first_bundle(broken)[0] / "data.h5", "r+") as f:
        zero(f)
    ctx = ctx_for(broken, sample=1)
    with pytest.raises(CheckSkipped):
        ctx.peak()
    assert run_one(engine_checks.MirrorSymmetry(), ctx).status == "skipped"


def test_the_tactile_check_skips_without_a_tonic_population(broken):
    bundle, _ = first_bundle(broken)
    cfg = json.loads((bundle / "config.json").read_text())
    for pop in cfg["config"]["populations"]:
        pop["filter_method"] = "ra"
    (bundle / "config.json").write_text(json.dumps(cfg))
    res = run_one(MonotonicRate(), ctx_for(broken))
    assert res.status == "skipped" and "tonic" in res.note


def test_context_and_pack_arguments_are_validated(tactile_dataset):
    with pytest.raises(ValueError):
        CheckContext(tactile_dataset, sample=0)
    with pytest.raises(ValueError):
        CheckContext(tactile_dataset, bounds_mv=(5.0, 1.0))
    with pytest.raises(ValueError):
        CheckContext(tactile_dataset, limit=0)
    with pytest.raises(FileNotFoundError):
        CheckContext(tactile_dataset / "nothing")
    with pytest.raises(KeyError):
        build_check_packs(["nope"])
    with pytest.raises(ValueError):
        build_check_packs(["core", "core"])
    with pytest.raises(ValueError):
        build_check_packs(["core"], {"tactile": {}})
    assert isinstance(
        build_check_packs(["tactile"], {"tactile": {"tolerance": 2}})[0], TactileChecks
    )


def test_clean_makes_values_json_safe():
    out = clean(
        {"a": float("nan"), "b": np.float32(2.5), "c": [np.int64(3), float("inf")]}
    )
    assert out == {"a": None, "b": 2.5, "c": [3, None]}


def test_sampling_is_spread_evenly(tactile_dataset):
    ctx = ctx_for(tactile_dataset, sample=3)
    ran = ctx.ran_entries()
    picks = ctx.sampled_entries()
    assert len(picks) == 3 and picks[0] == ran[0] and picks[-1] == ran[-1]
    assert ctx_for(tactile_dataset, sample=100).sampled_entries() == ran


def test_limit_reads_fewer_entries(tactile_dataset):
    res = run_one(data_checks.Finite(), ctx_for(tactile_dataset, limit=2))
    assert res.tested == 2


# ------------------------------------------------------------------ the command line


def test_the_command_line_exit_codes(tactile_dataset, broken, tmp_path, capsys):
    from sensoryforge.evaluate.checks.__main__ import main

    assert (
        main(
            [
                str(tactile_dataset),
                "--out",
                str(tmp_path / "a"),
                "--sample",
                "1",
                "--quiet",
            ]
        )
        == 0
    )
    assert (tmp_path / "a" / "checks.json").is_file()
    assert main([str(tmp_path / "missing"), "--out", str(tmp_path / "b")]) == 2
    assert "checks:" in capsys.readouterr().err
    assert (
        main([str(tactile_dataset), "--out", str(tmp_path / "c"), "--param", "bad"])
        == 2
    )

    def poke(f):
        name = next(iter(f["populations"]))
        f[f"populations/{name}/filtered"][1, 0] = np.nan

    damage(broken, poke)
    code = main([str(broken), "--out", str(tmp_path / "d"), "--sample", "1"])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out and "core.data.finite" in out


# ------------------------------------------------------------------ the server stage


def test_the_server_runs_the_checks_stage_and_serves_the_page(
    tactile_dataset, tmp_path
):
    api = Api(DatasetService(tactile_dataset, stages=tmp_path))

    def get(target):
        return api.handle(Request.from_target("GET", target))

    assert get("/api/stage/checks").status == 404
    assert get("/api/stages").status == 200
    assert json.loads(get("/api/stages").body)["checks"]["present"] is False
    resp = api.handle(Request.from_target("POST", "/api/stages/checks/run"))
    assert resp.status == 202
    job = api.jobs.wait(json.loads(resp.body)["id"], timeout=300)
    assert job["state"] == "done", job
    doc = json.loads(get("/api/stage/checks").body)
    assert doc["format"] == ck.CHECKS_FORMAT and doc["counts"]["fail"] == 0
    assert json.loads(get("/api/stages").body)["checks"]["present"] is True
    page = get("/checks")
    assert page.status == 200 and b"checks_page.js" in page.body


# ----------------------------------------------------------------- the page's own logic

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_checks_js(body: str):
    path = Path(api_module.__file__).parent / "static" / "checks.js"
    script = (
        f"const C = require({json.dumps(str(path))});\n"
        f"const out = (() => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@needs_node
def test_verdicts_groups_and_links_of_the_checks_page():
    got = run_checks_js("""
        const mk = (id, pack, status, tested, failed, note, failures) =>
          ({ id, pack, status, tested, failed, note: note || null,
             failures: failures || [] });
        const checks = [
          mk("core.a.x", "core", "pass", 12, 0),
          mk("core.a.y", "core", "fail", 14, 3, null,
             [{ where: "t/e/1", entry: "t/e/1" },
              { where: "t/e/1 / ra", entry: "t/e/1" },
              { where: "probe" }]),
          mk("core.a.z", "core", "skipped", 0, 0, "no floor declared"),
          mk("tactile.b.w", "tactile", "error", 0, 0, "RuntimeError: boom"),
        ];
        const g = C.groups(checks);
        return {
          verdicts: checks.map(C.verdict),
          packs: g.map((x) => [x.pack, x.checks.length, x.counts]),
          headline: C.headline({ pass: 1, fail: 1, skipped: 1, error: 1 }),
          none: C.headline({ pass: 0, fail: 0, skipped: 0, error: 0 }),
          entries: C.failedEntries(checks[1]),
          href: C.entryHref("test/dots/0001"),
          nums: [C.number(3), C.number(0.123456), C.number(null), C.number([1, 2.5])],
        };
    """)
    assert got["verdicts"] == [
        "all 12 passed",
        "3 of 14 failed",
        "not run: no floor declared",
        "error: RuntimeError: boom",
    ]
    assert got["packs"][0] == [
        "core",
        3,
        {"pass": 1, "fail": 1, "skipped": 1, "error": 0},
    ]
    assert got["packs"][1][0] == "tactile"
    assert got["headline"] == "1 fail · 1 error · 1 pass · 1 skipped"
    assert got["none"] == "no checks"
    assert got["entries"] == ["t/e/1"]
    assert got["href"] == "/#entry=test%2Fdots%2F0001"
    assert got["nums"] == ["3", "0.1235", "none", "[1, 2.5]"]


def test_the_checks_page_names_its_scripts(tactile_dataset, tmp_path):
    api = Api(DatasetService(tactile_dataset, stages=tmp_path))
    html = api.handle(Request.from_target("GET", "/checks")).body.decode()
    for name in ("checks.js", "checks_page.js"):
        assert f"/static/{name}" in html
        assert api.handle(Request.from_target("GET", f"/static/{name}")).status == 200
