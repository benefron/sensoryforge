"""The declared-fields panel and two-draw comparison of the evaluation page (roadmap A6c).

The page's record helpers live in ``static/record.js`` (pure, no DOM) and run here with
node on the declarations of the toy data set (a non-tactile one, A1b), so nothing depends
on a modality. The tests skip when node is not installed.
"""

import json
import shutil
import subprocess

import pytest

from sensoryforge.evaluate.server.api import Api, Request
from sensoryforge.evaluate.server.service import DatasetService

from tests.unit.test_eval_page_geometry import STATIC  # noqa: F401  (the static folder)

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_record(body: str, **inputs):
    """Run ``body`` in node with ``R`` (record.js), ``G`` (geometry.js) and ``inp``."""
    script = (
        f"const R = require({json.dumps(str(STATIC / 'record.js'))});\n"
        f"const G = require({json.dumps(str(STATIC / 'geometry.js'))});\n"
        f"const inp = {json.dumps(inputs)};\n"
        f"const out = (() => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def two_draws(toy_dataset):
    """Two different draws of the toy set as /api/draw returns them."""
    api = Api(DatasetService(toy_dataset.data))
    out = []
    for row in api.service.rows[:2]:
        resp = api.handle(Request.from_target("GET", f"/api/draw?entry={row.entry}"))
        out.append(json.loads(resp.body))
    return out


@needs_node
def test_every_declared_value_of_a_draw_is_in_the_panel(two_draws):
    d = two_draws[0]
    rows = run_record(
        "return R.rows(inp.record, inp.seeds);", record=d["record"], seeds=d["seeds"]
    )
    shown = {r["path"]: r["text"] for r in rows}
    for name, value in d["record"]["draw"]["values"].items():
        assert f"draw.values.{name}" in shown, name
    # the seeds come from the strings the server sent, not from rounded JSON numbers
    for name, text in d["seeds"].items():
        assert shown[f"seeds.{name}"] == text
    groups = run_record(
        "return R.grouped(R.rows(inp.record, inp.seeds)).map((g) => g.group);",
        record=d["record"],
        seeds=d["seeds"],
    )
    assert groups[:3] == ["Entry", "Draw", "Values drawn"]
    assert "Seeds" in groups


@needs_node
def test_a_timeline_reads_as_phases_and_big_integers_say_they_are_rounded():
    got = run_record(
        "return R.rows({draw: {timeline: [['hold', 0, 30], ['slide', 30, 90.5]]}, big: 12345678901234567890, none: [], t: true}, null)"
        ".map((r) => r.text);"
    )
    assert got[0] == "hold 0–30 · slide 30–90.5"
    assert got[1].endswith("(rounded)")
    assert got[2:] == ["(none)", "yes"]


@needs_node
def test_two_draws_line_up_by_path_and_the_differing_fields_are_flagged(two_draws):
    a, b = two_draws
    got = run_record(
        """
        const x = R.rows(inp.a.record, inp.a.seeds), y = R.rows(inp.b.record, inp.b.seeds);
        return R.compare(x, y).map((r) => [r.path, r.same, r.a === null, r.b === null]);
        """,
        a=a,
        b=b,
    )
    by = {path: (same, only_b, only_a) for path, same, only_b, only_a in got}
    assert by["entry"][0] is False  # two different draws
    assert by["world_id"][0] is True  # drawn from one world
    assert by["draw.values.amplitude"][0] is False  # drawn independently
    assert len(got) == len({p for p, *_ in got})  # each path once


@needs_node
def test_a_field_only_one_draw_declares_is_shown_as_absent():
    got = run_record(
        "return R.compare(R.rows({a: 1, only_a: 2}, null), R.rows({a: 1, only_b: 3}, null))"
        ".map((r) => [r.path, r.a, r.b, r.same]);"
    )
    assert got == [
        ["a", "1", "1", True],
        ["only_a", "2", None, False],
        ["only_b", None, "3", False],
    ]


@needs_node
def test_the_second_movie_follows_the_first_one_by_time():
    got = run_record(
        "return [G.matchFrame(10, 1, 1, 100), G.matchFrame(10, 1, 0.5, 100), G.matchFrame(10, 2, 1, 100),"
        " G.matchFrame(90, 1, 1, 50), G.matchFrame(0, 1, 1, 50)];"
    )
    assert got == [10, 20, 20, 49, 0]  # a shorter draw holds its last frame


@needs_node
def test_traces_of_b_are_laid_over_only_when_the_pin_means_the_same_in_b(two_draws):
    a = two_draws[0]["bundle"]
    pops = a["populations"]
    assert pops, "the toy set has populations"
    name, n = pops[0]["name"], pops[0]["n_units"]
    call = "return R.pinProblem(%s, %s, inp.a, %s);"
    same = run_record(call % ('{kind: "sensor", k: 3}', "null", "inp.a"), a=a)
    assert same is None
    unit = run_record(
        call % ('{kind: "unit", index: 0}', "inp.name", "inp.a"), a=a, name=name
    )
    assert unit is None
    too_far = run_record(
        call % ('{kind: "unit", index: inp.n}', "inp.name", "inp.a"),
        a=a,
        name=name,
        n=n,
    )
    assert "only" in too_far
    missing = run_record(call % ('{kind: "unit", index: 0}', '"nope"', "inp.a"), a=a)
    assert "no population nope" in missing
    unreadable = run_record(call % ('{kind: "sensor", k: 0}', "null", "null"), a=a)
    assert "could not be read" in unreadable
    smaller = json.loads(json.dumps(a))
    smaller["layout"]["coords"] = smaller["layout"]["coords"][:-1]
    diff = run_record(
        "return R.pinProblem({kind: 'sensor', k: 0}, null, inp.a, inp.b);",
        a=a,
        b=smaller,
    )
    assert "sensor arrays differ" in diff
    assert run_record("return R.pinProblem(null, null, inp.a, null);", a=a) is None


def test_the_page_serves_the_record_helpers_before_the_app_and_has_the_comparison_controls(
    toy_dataset,
):
    api = Api(DatasetService(toy_dataset.data))
    html = api.handle(Request.from_target("GET", "/")).body.decode()
    assert html.index("/static/record.js") < html.index("/static/app.js")
    r = api.handle(Request.from_target("GET", "/static/record.js"))
    assert r.status == 200 and r.content_type.startswith("text/javascript")
    for needed in (
        "compare-pick",
        "compare-clear",
        "movie-b",
        "overlay-b",
        "fields-body",
        "only-diff",
    ):
        assert f'id="{needed}"' in html
