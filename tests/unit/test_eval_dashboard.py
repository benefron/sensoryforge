"""Data-set dashboards (roadmap A7): counts, histograms and the coverage map.

The invariant that matters is the last line of the task: every number on a dashboard is
one click from its draws, so each count, bar and cell is checked by asking the draw list
for its ``query`` and comparing the answer with the number.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sensoryforge.evaluate.audit.run import run_audit
from sensoryforge.evaluate.server import Api, DatasetService, Request
from sensoryforge.evaluate.server import api as api_module
from sensoryforge.evaluate.server.dashboard import (
    N_BINS,
    coverage_for_axis,
    histogram,
)

STAMP = "2026-10-23T10:00:00Z"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def get(api, target):
    response = api.handle(Request.from_target("GET", target))
    return response.status, json.loads(response.body)


def total_for(api, query):
    status, body = get(api, f"/api/draws?{query}&limit=1")
    assert status == 200, body
    return body["total"]


def every_number(dash):
    """``(where, n, query)`` for each number a dashboard shows."""
    for key, items in dash["counts"].items():
        for item in items:
            yield f"counts.{key}.{item['label']}", item["n"], item["query"]
    for r, line in enumerate(dash["matrix"]["cells"]):
        for c, cell in enumerate(line):
            yield f"matrix[{r}][{c}]", cell["n"], cell["query"]
    for h in dash["histograms"]:
        for b, (n, q) in enumerate(zip(h["counts"], h["queries"])):
            yield f"{h['id']}[{b}]", n, q
    for cov in dash["coverage"]:
        for b, (n, q) in enumerate(zip(cov["counts"], cov["queries"])):
            yield f"{cov['id']}[{b}]", n, q


@pytest.fixture(scope="module")
def audit_dir(toy_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("toy_audit")
    run_audit(toy_dataset.data, out, created_at=STAMP)
    return out


@pytest.fixture(scope="module")
def toy_api(toy_dataset, audit_dir):
    return Api(DatasetService(toy_dataset.data, stages=audit_dir))


@pytest.fixture(scope="module")
def tactile_api(tactile_dataset, audited):
    return Api(DatasetService(tactile_dataset, stages=audited[1]))


# ------------------------------------------------------------------ the numbers


@pytest.mark.parametrize("which", ["toy_api", "tactile_api"])
def test_every_number_lists_the_draws_behind_it(which, request):
    api = request.getfixturevalue(which)
    status, dash = get(api, "/api/dashboard")
    assert status == 200
    checked = 0
    for where, n, query in every_number(dash):
        assert total_for(api, query) == n, where
        checked += 1
    assert checked > 30
    # the single-filter counts add up to the data set
    for key in ("split", "class", "status", "kind"):
        assert sum(i["n"] for i in dash["counts"][key]) == dash["n_draws"], key


def test_counts_matrix_and_kind_agree_with_the_manifest(toy_api, toy_dataset):
    _, dash = get(toy_api, "/api/dashboard")
    rows = [
        json.loads(x)
        for x in (toy_dataset.data / "manifest.jsonl").read_text().splitlines()
    ]
    by_split = {i["label"]: i["n"] for i in dash["counts"]["split"]}
    assert by_split == {
        s: sum(1 for r in rows if r["split"] == s) for s in {r["split"] for r in rows}
    }
    matrix = dash["matrix"]
    for r, cls in enumerate(matrix["classes"]):
        for c, split in enumerate(matrix["splits"]):
            want = sum(
                1
                for x in rows
                if (x["class"] or "no class") == cls and x["split"] == split
            )
            assert matrix["cells"][r][c]["n"] == want
    assert {i["label"]: i["n"] for i in dash["counts"]["kind"]} == {"draw": len(rows)}
    assert "session_type" not in dash["counts"]


def test_sessions_are_counted_and_stay_out_of_the_coverage_map(
    tactile_api, tactile_dataset
):
    _, dash = get(tactile_api, "/api/dashboard")
    kinds = {i["label"]: i for i in dash["counts"]["kind"]}
    assert kinds["session"]["n"] == dash["n_sessions"] == 1
    status, body = get(tactile_api, f"/api/draws?{kinds['session']['query']}")
    assert status == 200 and [r["split"] for r in body["items"]] == ["sessions"]
    assert all(c["class"] is not None for c in dash["coverage"])


# ------------------------------------------------------------------ histograms


def test_the_length_histogram_is_the_manifest_and_the_rest_come_from_the_audit(toy_api):
    _, dash = get(toy_api, "/api/dashboard")
    by_id = {h["id"]: h for h in dash["histograms"]}
    assert set(by_id) == {"length", "peak", "rate:luma", "rate:chroma"}
    length = by_id["length"]
    assert length["source"] == "manifest" and length["unit"] == "ms"
    assert length["counts"] == [dash["n_draws"]] and length["min"] == length["max"]
    for pop in ("luma", "chroma"):
        h = by_id[f"rate:{pop}"]
        assert h["source"] == "audit" and h["unit"] == "1/ms"
        assert sum(h["counts"]) + h["n_missing"] == dash["n_draws"]
        assert len(h["edges"]) in (2, N_BINS + 1)
    peak = by_id["peak"]
    assert sum(peak["counts"]) + peak["n_missing"] == dash["n_draws"]
    assert dash["audit_present"] is True and dash["unavailable"] == []


def test_a_bar_lists_the_draws_whose_measured_value_falls_in_it(toy_api, audit_dir):
    _, dash = get(toy_api, "/api/dashboard")
    peak = next(h for h in dash["histograms"] if h["id"] == "peak")
    best = {}
    for line in (audit_dir / "audit_entries.jsonl").read_text().splitlines():
        row = json.loads(line)
        vals = [v["value"] for v in row["values"] if v["id"] == "core.frames.peak"]
        best[row["entry"]] = max(vals)
    for b, query in enumerate(peak["queries"]):
        _, page = get(toy_api, f"/api/draws?{query}&limit=1000")
        lo, hi = peak["edges"][b], peak["edges"][b + 1]
        for item in page["items"]:
            v = best[item["entry"]]
            assert lo <= v <= hi and (v < hi or b == len(peak["counts"]) - 1)


def test_histogram_helper_edge_cases():
    entry, per_bin = histogram("h", "t", "audit", None, np.full(5, 2.0))
    assert (
        entry["counts"] == [5]
        and entry["edges"] == [2.0, 2.0]
        and per_bin == [[0, 1, 2, 3, 4]]
    )
    entry, per_bin = histogram(
        "h", "t", "audit", None, np.array([0.0, 1.0, np.nan, 10.0])
    )
    assert entry["n_missing"] == 1 and sum(entry["counts"]) == 3
    # 0 and 1 are bins 0 and 1; the maximum sits in the last bin, not past it
    assert entry["counts"][:2] == [1, 1] and entry["counts"][-1] == 1
    empty, none = histogram("h", "t", "audit", None, np.full(3, np.nan))
    assert empty["counts"] == [] and none == [] and empty["n_missing"] == 3


# -------------------------------------------------------------------- coverage


def test_coverage_counts_the_values_each_axis_reached(toy_api, toy_dataset):
    _, dash = get(toy_api, "/api/dashboard")
    cov = {(c["class"], c["axis"]): c for c in dash["coverage"]}
    sigma = cov[("red_blob", "sigma_mm")]
    assert (sigma["lo"], sigma["hi"], sigma["form"]) == (1.5, 2.5, "numeric")
    assert len(sigma["counts"]) == N_BINS
    rows = [
        json.loads(x)
        for x in (toy_dataset.data / "manifest.jsonl").read_text().splitlines()
    ]
    values = [r["draw"]["values"]["sigma_mm"] for r in rows if r["class"] == "red_blob"]
    assert sigma["n"] == len(values)
    assert sum(sigma["counts"]) + sigma["n_outside"] == len(values)
    want, _ = np.histogram(np.clip(values, 1.5, 2.5), bins=sigma["edges"])
    assert sigma["counts"] == want.tolist()
    assert sigma["n_empty"] == sum(1 for c in sigma["counts"] if c == 0)
    # constants have nothing to cover, and the quiet class has no random axis
    assert ("red_blob", "hold_ms") not in cov and not [k for k in cov if k[0] == "dark"]
    assert ("red_blob", "direction_deg") in cov and cov[("red_blob", "direction_deg")][
        "circular"
    ]


def test_coverage_forms_and_gaps():
    rows = np.arange(6)
    # a numeric range with a gap, a value out of range and a missing one
    entry, hist_id, per_bin = coverage_for_axis(
        "c",
        "a",
        {"range": [0.0, 10.0], "dist": "uniform"},
        rows,
        [0.5, 0.7, 9.9, 12.0, None, 10.0],
    )
    assert hist_id == "cov:c:a" and entry["counts"] == [2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert entry["n"] == 5 and entry["n_outside"] == 1 and entry["n_empty"] == 8
    assert per_bin[0] == [0, 1] and per_bin[9] == [2, 5]
    # log-uniform bins are equal in the logarithm
    entry, _, per_bin = coverage_for_axis(
        "c",
        "f",
        {"range": [1.0, 100.0], "dist": "log_uniform"},
        rows,
        [1.5, 9.0, 11.0, 99.0, 2.0, 5.0],
    )
    assert entry["scale"] == "log" and entry["edges"][5] == pytest.approx(10.0)
    assert entry["counts"] == [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    # a small integer range is shown value by value
    entry, _, per_bin = coverage_for_axis(
        "c", "k", {"range": [1, 4], "int": True}, rows, [1, 4, 4, 2, 9, None]
    )
    assert entry["labels"] == ["1", "2", "3", "4"] and entry["counts"] == [1, 1, 0, 2]
    assert entry["n_outside"] == 1
    # a categorical axis lists its values
    entry, _, per_bin = coverage_for_axis(
        "c",
        "s",
        {"values": ["a", "b", "c"], "weights": [1, 1, 1]},
        rows,
        ["a", "a", "c", "z", None, "b"],
    )
    assert entry["labels"] == ["a", "b", "c"] and entry["counts"] == [2, 1, 1]
    assert entry["n_outside"] == 1
    # a constant, a link and a named distribution have nothing to bin
    for spec in ({"dist": "normal", "mean": 1.0, "std": 2.0}, {"value": 3}):
        assert coverage_for_axis("c", "x", spec, rows, [1.0] * 6) is None


# ------------------------------------------------------------ plumbing and limits


def test_without_the_audit_the_measured_histograms_say_why(toy_dataset, tmp_path):
    api = Api(DatasetService(toy_dataset.data, stages=tmp_path / "none"))
    _, dash = get(api, "/api/dashboard")
    assert [h["id"] for h in dash["histograms"]] == ["length"]
    assert dash["audit_present"] is False
    assert "run the audit" in dash["unavailable"][0]["why"]
    assert dash["coverage"], "the coverage map needs the world, not the audit"


def test_the_dashboard_is_rebuilt_when_the_audit_appears(
    toy_dataset, audit_dir, tmp_path
):
    stages = tmp_path / "evaluate"
    api = Api(DatasetService(toy_dataset.data, stages=stages))
    assert get(api, "/api/dashboard")[1]["audit_present"] is False
    shutil.copytree(audit_dir, stages)
    assert get(api, "/api/dashboard")[1]["audit_present"] is True


def test_a_bad_bin_is_refused(toy_api):
    for bin_ in ("nonsense:0", "length:7", "length:x", "length"):
        status, body = get(toy_api, f"/api/draws?bin={bin_}")
        assert status == 400 and "bin" in body["error"]


def test_a_bin_combines_with_the_other_filters(toy_api):
    _, dash = get(toy_api, "/api/dashboard")
    hist = next(h for h in dash["histograms"] if h["id"] == "peak")
    b = max(range(len(hist["counts"])), key=hist["counts"].__getitem__)
    both = total_for(toy_api, f"{hist['queries'][b]}&split=test")
    only = total_for(toy_api, hist["queries"][b])
    assert 0 <= both <= only == hist["counts"][b]
    random_ = get(toy_api, f"/api/draws/random?{hist['queries'][b]}&pick_seed=3")
    assert random_[0] == 200 and random_[1]["total"] == only


def test_the_page_is_served_and_names_its_scripts(toy_api):
    response = toy_api.handle(Request.from_target("GET", "/dashboard"))
    assert response.status == 200 and response.content_type.startswith("text/html")
    html = response.body.decode()
    for name in ("dashboard.js", "dashboard_page.js"):
        assert f"/static/{name}" in html
        assert (
            toy_api.handle(Request.from_target("GET", f"/static/{name}")).status == 200
        )


# ----------------------------------------------------------- the page's own logic


def run_dashboard_js(body: str):
    path = Path(api_module.__file__).parent / "static" / "dashboard.js"
    script = (
        f"const D = require({json.dumps(str(path))});\n"
        f"const out = (() => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@needs_node
def test_labels_fractions_and_links_of_the_dashboard_page():
    got = run_dashboard_js("""
        const fmt = (v) => String(v);
        const edges = [0, 1, 2, 3];
        return {
          labels: [0, 1, 2].map((i) => D.rangeLabel(edges, i, fmt)),
          flat: D.rangeLabel([5, 5], 0, fmt),
          frac: D.fractions([2, 4, 0]), none: D.fractions([0, 0]),
          href: D.listHref("bin=peak%3A3&split=train", "peak [1, 2)"),
          round: D.listFromHash(
            D.listHref("bin=peak%3A3&split=train", "peak [1, 2)").slice(1)),
          cov: D.coverageText({ counts: [1, 0, 2], n_empty: 1, n: 3, n_outside: 2 }),
          cat: D.coverageLabel({ labels: ["a", "b"] }, 1, fmt),
          bin: D.coverageLabel({ labels: null, edges }, 2, fmt),
          back: D.listFromHash("#list=bin%3Apeak%3A3&label=x%20y"),
          mixed: D.listFromHash("#entry=e1&list=class%3Dr&label=r"),
          none_: D.listFromHash("#entry=e1"),
        };
        """)
    assert got["labels"] == ["[0, 1)", "[1, 2)", "[2, 3]"] and got["flat"] == "5"
    assert got["frac"] == [0.5, 1, 0] and got["none"] == [0, 0]
    assert (
        got["cov"] == "2 of 3 bins reached by 3 draws · 2 outside the declared domain"
    )
    assert got["cat"] == "b" and got["bin"] == "[2, 3]"
    # the link and the hash reading are inverses: the browser gets back the same query
    assert got["back"] == {"query": "bin:peak:3", "label": "x y"}
    assert got["mixed"] == {"query": "class=r", "label": "r"} and got["none_"] is None
    # a link's address is read back to the same query and label
    assert got["round"] == {"query": "bin=peak%3A3&split=train", "label": "peak [1, 2)"}
    assert got["href"].startswith("/#list=")
