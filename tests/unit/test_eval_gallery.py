"""The anomaly gallery (roadmap A9): seeded failures in a data set are all found.

A toy data set with ten or more entries per class is built through the engine; each
fault below is written into one entry of a private copy (a NaN, silence, saturation, a
rate outlier, a long quiet gap, frames from another class, a failed batch row). The
audit and the checks run over the copy, and the gallery must list every seeded entry
with the signal that matches its fault, ahead of the untouched entries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from sensoryforge.evaluate.audit.run import run_audit
from sensoryforge.evaluate.checks import run_checks
from sensoryforge.evaluate.server import Api, DatasetService, Request
from sensoryforge.evaluate.server import api as api_module
from sensoryforge.evaluate.server.gallery import (
    GalleryParams,
    SIGNALS,
    build_gallery,
    robust_z,
)

STAMP = "2026-10-23T14:00:00Z"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def get(api, target):
    response = api.handle(Request.from_target("GET", target))
    return response.status, json.loads(response.body)


def manifest(folder):
    return [
        json.loads(line)
        for line in (Path(folder) / "manifest.jsonl").read_text().splitlines()
    ]


def bundle(folder, entry):
    return Path(folder) / entry


def edit(folder, entry, fn):
    with h5py.File(bundle(folder, entry) / "data.h5", "r+") as f:
        fn(f)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """``(folder, {signal: entry})``: a copy of a 60-draw toy set, one fault each."""
    from tests.fixtures import toy_vision

    root = tmp_path_factory.mktemp("gallery")
    built = toy_vision.build_toy_dataset(root / "built", n_train=60)
    folder = root / "data"
    shutil.copytree(built.data, folder)
    rows = [r for r in manifest(folder) if r["class"]]
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["entry"])
    classes = sorted(c for c in by_class if len(by_class[c]) >= 12 and "blob" in c)
    assert len(classes) >= 2, {c: len(v) for c, v in by_class.items()}
    first, second = classes[0], classes[1]
    pick = {
        "bad_data": by_class[first][0],
        "silent": by_class[first][1],
        "saturated": by_class[first][2],
        "outlier_rate": by_class[first][3],
        "long_quiet": by_class[first][4],
        "class_mismatch": by_class[first][5],
    }

    def pops(f):
        return list(f["populations"])

    def nan(f):
        f[f"populations/{pops(f)[0]}/filtered"][2, 0] = np.nan

    def silent(f):
        for name in pops(f):
            d = f[f"populations/{name}/spikes"]
            d[...] = np.zeros_like(d[...])

    def saturated(f):
        for name in pops(f):
            d = f[f"populations/{name}/spikes"]
            d[...] = np.ones_like(d[...])

    def outlier(f):
        for name in pops(f):
            d = f[f"populations/{name}/spikes"]
            a = d[...]
            a[::5] = 1  # one bin in five: far above the class, below saturation
            d[...] = a

    def quiet(f):
        d = f["stimulus/frames"]
        a = d[...]
        t = a.shape[0]
        a[t // 5 : 4 * t // 5] = 0
        d[...] = a

    other = by_class[second][0]
    with h5py.File(bundle(folder, other) / "data.h5", "r") as f:
        foreign = f["stimulus/frames"][...]

    def mismatch(f):
        f["stimulus/frames"][...] = foreign

    for key, fn in (
        ("bad_data", nan),
        ("silent", silent),
        ("saturated", saturated),
        ("outlier_rate", outlier),
        ("long_quiet", quiet),
        ("class_mismatch", mismatch),
    ):
        edit(folder, pick[key], fn)
    # a failed batch row for one more entry
    ghost = by_class[second][1]
    pick["did_not_run"] = ghost
    for path in sorted((folder / "index").glob("task_*.jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        hit = False
        for line in lines:
            if line["entry"] == ghost:
                line["status"], line["error"], hit = "failed", "seeded", True
        if hit:
            path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return folder, pick


@pytest.fixture(scope="module")
def api(seeded, tmp_path_factory):
    folder, _ = seeded
    out = tmp_path_factory.mktemp("gallery_stages")
    run_audit(folder, out, created_at=STAMP)
    run_checks(folder, out, sample=1, created_at=STAMP)
    return Api(DatasetService(folder, stages=out))


def test_every_seeded_failure_is_found_with_its_own_signal(seeded, api):
    _, pick = seeded
    status, body = get(api, "/api/gallery?limit=500")
    assert status == 200
    flagged = {i["entry"]: {r["signal"] for r in i["reasons"]} for i in body["items"]}
    missing = {
        sig: entry
        for sig, entry in pick.items()
        if sig not in flagged.get(entry, set())
    }
    assert not missing, (missing, flagged)


def test_failures_rank_ahead_of_soft_signals(seeded, api):
    _, pick = seeded
    _, body = get(api, "/api/gallery?limit=500")
    order = [i["entry"] for i in body["items"]]
    assert {order[0], order[1]} == {pick["bad_data"], pick["did_not_run"]}
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    for it in body["items"]:
        sev = [r["severity"] for r in it["reasons"]]
        assert sev == sorted(sev, reverse=True) and it["reasons"]


def test_untouched_entries_are_mostly_not_flagged(seeded, api):
    _, pick = seeded
    _, body = get(api, "/api/gallery?limit=500")
    touched = set(pick.values())
    clean_flagged = [i for i in body["items"] if i["entry"] not in touched]
    total = len(manifest(seeded[0]))
    assert len(clean_flagged) <= 0.25 * total, [
        (i["entry"], [r["signal"] for r in i["reasons"]]) for i in clean_flagged
    ]
    assert body["n_entries"] >= total - 1  # the audit skipped at most the failed one


def test_filters_limit_and_counts(seeded, api):
    _, pick = seeded
    _, body = get(api, "/api/gallery?signal=silent&limit=1")
    assert body["total"] >= 1 and len(body["items"]) == 1
    assert body["items"][0]["entry"] == pick["silent"]
    assert {r["signal"] for r in body["items"][0]["reasons"]} == {"silent"}
    counts = {s["id"]: s["n_entries"] for s in body["signals"]}
    assert counts["silent"] == body["total"] and set(counts) == set(SIGNALS)
    first_class = next(r["class"] for r in manifest(seeded[0]) if r["class"])
    _, one = get(api, f"/api/gallery?class={first_class}&limit=500")
    assert all(i["class"] == first_class for i in one["items"])
    _, none = get(api, "/api/gallery?split=nothing")
    assert none["total"] == 0 and none["items"] == []


def test_bad_requests(api, seeded, tmp_path):
    assert get(api, "/api/gallery?signal=nope")[0] == 400
    assert get(api, "/api/gallery?limit=0")[0] == 400
    bare = Api(DatasetService(seeded[0], stages=tmp_path))
    status, body = get(bare, "/api/gallery")
    assert status == 409 and "audit" in body["error"]


def test_without_checks_the_data_signal_is_unavailable(seeded, tmp_path):
    folder, _ = seeded
    run_audit(folder, tmp_path, created_at=STAMP)
    body = get(Api(DatasetService(folder, stages=tmp_path)), "/api/gallery")[1]
    assert {u["signal"] for u in body["unavailable"]} == {"bad_data"}


def test_the_gallery_follows_its_inputs(seeded, tmp_path):
    folder, pick = seeded
    run_audit(folder, tmp_path, created_at=STAMP)
    api = Api(DatasetService(folder, stages=tmp_path))
    before = get(api, "/api/gallery?limit=500")[1]["n_flagged"]
    run_checks(folder, tmp_path, sample=1, created_at=STAMP)
    after = get(api, "/api/gallery?limit=500")
    assert after[1]["n_flagged"] >= before
    assert pick["bad_data"] in {i["entry"] for i in after[1]["items"]}


def test_the_page_is_served(api):
    html = api.handle(Request.from_target("GET", "/gallery")).body.decode()
    for name in ("gallery.js", "gallery_page.js"):
        assert f"/static/{name}" in html
        assert api.handle(Request.from_target("GET", f"/static/{name}")).status == 200


# --------------------------------------------------------------- the pure functions


def row(entry, cls, split="train", **vals):
    values = []
    for key, v in vals.items():
        mid, _, pop = key.partition("@")
        item = {"id": mid.replace("__", "."), "value": v}
        if pop:
            item["population"] = pop
        values.append(item)
    return {"entry": entry, "split": split, "class": cls, "values": values}


def test_robust_z_has_a_floor_so_a_unanimous_class_still_has_a_scale():
    z = robust_z([0.0] * 9 + [60.0], resolution=1.0)
    assert z[-1] > 3.5 and all(abs(v) < 1 for v in z[:-1])
    assert robust_z([1.0, None, 2.0], None)[1] is None
    assert robust_z([5.0], None) == [None]


def test_params_are_validated():
    for bad in (
        dict(z=0),
        dict(saturation=0),
        dict(mismatch_ratio=0.5),
        dict(min_peers=2),
    ):
        with pytest.raises(ValueError):
            GalleryParams(**bad)


def test_a_small_class_is_not_judged():
    rows = [
        row(f"e{i}", "a", core__response__rate_mean__x=0.01 * (i + 1)) for i in range(4)
    ]
    for r in rows:
        r["values"][0]["id"] = "core.response.rate_mean"
    gallery = build_gallery(rows, status={})
    assert gallery.items == []
    reasons = {u["signal"] for u in gallery.unavailable}
    assert "class_mismatch" in reasons


def test_a_class_that_lacks_a_feature_still_has_a_profile():
    # Class b has no centroid speed at all, so its centre there is NaN; the distance to
    # b uses the features b has instead of going NaN and hiding the mismatch.
    def frames(entry, cls, peak, speed=None):
        values = [{"id": "core.frames.peak", "value": peak}]
        if speed is not None:
            values.append({"id": "core.frames.centroid_speed", "value": speed})
        return {"entry": entry, "split": "s", "class": cls, "values": values}

    rows = [frames(f"a{i}", "a", 1.0 + 0.01 * i, 0.1 + 0.001 * i) for i in range(12)]
    rows += [frames(f"b{i}", "b", 10.0 + 0.01 * i) for i in range(5)]
    rows.append(frames("odd", "a", 10.0, 0.105))
    flagged = {
        i["entry"]: {r["signal"] for r in i["reasons"]}
        for i in build_gallery(rows, status={}).items
    }
    assert flagged == {"odd": {"class_mismatch"}}


def test_an_unreadable_bundle_is_bad_data_whatever_its_batch_status():
    rows = [row("e1", "c"), row("e2", "c")]
    checks = {
        "checks": [
            {
                "id": "core.data.readable",
                "failures": [{"entry": "e1", "detail": "cannot open data.h5"}],
            }
        ]
    }
    status = {"e1": ("ok", None), "e2": ("ok", None)}
    items = build_gallery(rows, checks=checks, status=status).items
    assert [(i["entry"], i["reasons"][0]["signal"]) for i in items] == [
        ("e1", "bad_data")
    ]


def test_unticking_every_signal_shows_none():
    rows = [row("e1", "c")]
    checks = {"checks": [{"id": "core.data.finite", "failures": [{"entry": "e1"}]}]}
    checks["checks"][0]["failures"][0]["detail"] = "NaN"
    gallery = build_gallery(rows, checks=checks, status={})
    assert gallery.payload(None)["total"] == 1
    assert gallery.payload([])["total"] == 0


def test_silence_needs_an_active_stimulus():
    base = [
        {"id": "core.frames.active_fraction", "value": 0.5},
        {"id": "core.frames.dt", "value": 1.0},
        {"id": "core.response.active_unit_fraction", "value": 0.0, "population": "p"},
    ]
    quiet = [dict(v) for v in base]
    quiet[0]["value"] = 0.0
    firing = [dict(v) for v in base]
    firing[2] = dict(firing[2], value=1.0)
    rows = [
        {"entry": "loud", "split": "s", "class": "c", "values": base},
        {"entry": "still", "split": "s", "class": "c", "values": quiet},
    ] + [
        {"entry": f"ok{i}", "split": "s", "class": "c", "values": firing}
        for i in range(4)
    ]
    items = build_gallery(rows, status={}).items
    assert [i["entry"] for i in items] == ["loud"]
    assert items[0]["reasons"][0]["signal"] == "silent"


# --------------------------------------------------------------- the page's own logic


def run_gallery_js(body: str):
    path = Path(api_module.__file__).parent / "static" / "gallery.js"
    script = (
        f"const G = require({json.dumps(str(path))});\n"
        f"const out = (() => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@needs_node
def test_queries_tiers_and_lines_of_the_gallery_page():
    got = run_gallery_js("""
        const all = ["a", "b", "c"];
        return {
          every: G.query(["a", "b", "c"], all, 24),
          some: G.query(["a", "c"], all, 12),
          none: G.query([], all, 12),
          unknown: G.query([], [], 12),
          href: G.entryHref("test/dots/0001"),
          tiers: [1.0, 0.8, 0.3].map(G.tier),
          where: [G.where({ class: "dots", split: "test" }),
                  G.where({ class: null, split: "sessions" })],
          line: [G.reasonLine({ signal: "silent", detail: "none", population: "sa" },
                              { silent: "Silent" }),
                 G.reasonLine({ signal: "x", detail: "d" }, {})],
          shown: [G.shownText({ total: 3, n_entries: 12, items: [1, 2, 3] }),
                  G.shownText({ total: 30, n_entries: 40, items: [1, 2] })],
        };
    """)
    assert got["every"] == "limit=24"
    assert got["some"] == "limit=12&signal=a%2Cc"
    assert got["none"] == "limit=12&signal="
    assert got["unknown"] == "limit=12"
    assert got["href"] == "/#entry=test%2Fdots%2F0001"
    assert got["tiers"] == ["high", "medium", "low"]
    assert got["where"] == ["dots · test", "no class · sessions"]
    assert got["line"] == ["Silent (sa): none", "x: d"]
    assert got["shown"] == [
        "3 flagged of 12 entries",
        "30 flagged of 40 entries · showing the worst 2",
    ]
