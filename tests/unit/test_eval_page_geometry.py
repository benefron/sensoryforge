"""The movie's geometry and colour scale (roadmap A6a).

The page's pure functions live in ``static/geometry.js`` and are run here with node, so
the mapping from a stored frame to canvas pixels is pinned: frame axis 0 is x and is
painted to the right, axis 1 is y and is painted upwards. The data is the toy
non-tactile data set (A1b), so nothing depends on a modality. The tests skip when node
is not installed.
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

STATIC = Path(api_module.__file__).parent / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_js(body: str, **inputs):
    """Run ``body`` in node with ``G`` (geometry.js) and ``inp``; its JSON result."""
    script = (
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
def draw(toy_dataset):
    """The toy draw's sensor layout, as /api/draw returns it."""
    api = Api(DatasetService(toy_dataset.data))
    entry = api.service.rows[0].entry
    body = json.loads(
        api.handle(Request.from_target("GET", f"/api/draw?entry={entry}")).body
    )
    return body


@needs_node
def test_x_runs_to_the_right_and_y_upwards(draw):
    layout = draw["bundle"]["layout"]
    coords = np.asarray(layout["coords"])
    nx, ny = layout["regular_grid"]["rows"], layout["regular_grid"]["cols"]
    xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    assert len(xs) == nx and len(ys) == ny
    # sensors at known coordinates: the corners, and one off the diagonal
    probes = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1), (3, 7)]
    for i, j in probes:
        k = i * ny + j  # the contract: sensor k = i * cols + j, x along axis 0
        x, y = coords[k]
        assert x == xs[i] and y == ys[j]
        frame = np.zeros(nx * ny)
        frame[k] = 1.0
        px = run_js(
            "const rgba = G.toRGBA(Float32Array.from(inp.frame), 0, inp.nx, inp.ny,"
            " 0, 1);"
            "let at = null;"
            "for (let cy = 0; cy < inp.ny; cy++) for (let cx = 0; cx < inp.nx; cx++)"
            "  if (rgba[4 * (cy * inp.nx + cx)] > 200) at = [cx, cy];"
            "return at;",
            frame=frame.tolist(),
            nx=nx,
            ny=ny,
        )
        # larger x sits further right, larger y sits higher (a smaller canvas row)
        assert px == [int(np.searchsorted(xs, x)), ny - 1 - int(np.searchsorted(ys, y))]
    # +x moves right and +y moves up, one pixel per sensor step
    steps = run_js(
        "return [G.pixelOf(2, 2, 5, 4), G.pixelOf(3, 2, 5, 4), G.pixelOf(2, 3, 5, 4)];"
    )
    assert steps == [{"cx": 2, "cy": 1}, {"cx": 3, "cy": 1}, {"cx": 2, "cy": 0}]


@needs_node
def test_a_click_maps_back_to_the_sensor_it_painted():
    got = run_js(
        "const bad = [];"
        "for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) {"
        "  const p = G.pixelOf(i, j, 5, 4);"
        "  const s = G.sensorAt(p.cx + 0.5, p.cy + 0.5, 5, 4);"
        "  if (s.i !== i || s.j !== j || s.k !== i * 4 + j) bad.push([i, j]);"
        "}"
        "return [bad, G.sensorAt(-1, 0, 5, 4), G.sensorAt(0, 4, 5, 4),"
        "        G.sensorAt(5, 0, 5, 4)];"
    )
    assert got == [[], None, None, None]


@needs_node
def test_the_colour_scale_prefers_the_audit_peak_and_says_which():
    cases = run_js(
        "return ["
        "  G.colourScale(2.5, 0.1, 1.0),"  # audit peak, running minimum above 0
        "  G.colourScale(2.5, -0.4, 1.0),"  # running minimum below 0 is kept
        "  G.colourScale(null, -0.4, 1.0),"  # no audit: the running range
        "  G.colourScale(0.0, 0.0, 1.0),"  # a peak that is not above the bound
        "  G.colourScale(null, Infinity, -Infinity),"
        "];"
    )
    assert cases == [
        {"low": 0, "high": 2.5, "source": "audit peak"},
        {"low": -0.4, "high": 2.5, "source": "audit peak"},
        {"low": -0.4, "high": 1.0, "source": "running range"},
        {"low": 0.0, "high": 1.0, "source": "running range"},
        {"low": 0, "high": 1, "source": "running range"},
    ]
    peaks = run_js(
        "const fp = {values: [{channel: 1, value: 7}, {channel: null, value: 3}]};"
        "return [G.peakFor(fp, 1), G.peakFor(fp, 0), G.peakFor(null, 0),"
        "        G.peakFor({values: [{channel: 1, value: 7}]}, 0)];"
    )
    assert peaks == [7, 3, None, None]


@needs_node
def test_playback_follows_the_wall_clock_and_skips_frames():
    got = run_js(
        "return ["
        "  G.frameAt(0, 16, 1, 1, 5000),"  # 16 ms of 1 ms frames: 16 frames ahead
        "  G.frameAt(10, 1000, 0.1, 1, 5000),"  # a tenth of real time
        "  G.frameAt(4990, 16, 1, 1, 5000),"  # wraps at the end of the draw
        "  G.frameAt(0, 16, 2, 4, 100),"
        "  G.frameAt(0, -3, 1, 1, 100),"  # a clock that ran backwards never leaves 0
        "];"
    )
    assert got == [16, 110, 6, 8, 0]


@needs_node
def test_axes_are_labelled_from_the_layout_bounds(draw):
    bounds = draw["bundle"]["layout"]["bounds"]
    unit = draw["bundle"]["units"]["space"]
    text = run_js("return G.axisText(inp.b, inp.u);", b=bounds, u=unit)
    assert text["xlabel"] == f"x ({unit})" and text["ylabel"] == f"y ({unit})"
    assert float(text["xmin"]) == pytest.approx(bounds[0][0], rel=1e-3)
    assert float(text["ymax"]) == pytest.approx(bounds[1][1], rel=1e-3)
    assert run_js("return G.axisText([[0, 1], [0, 1]], null).xlabel;") == "x"


# --------------------------------------------------------------- the audit's peak


@pytest.fixture(scope="module")
def audited(toy_dataset, tmp_path_factory):
    stages = tmp_path_factory.mktemp("stages")
    run_audit(toy_dataset.data, stages, packs=["core"])
    return DatasetService(toy_dataset.data, stages), stages


def test_the_draw_carries_the_audit_peak_when_the_audit_ran(audited):
    service, stages = audited
    entry = service.rows[0].entry
    lines = (stages / "audit_entries.jsonl").read_text().splitlines()
    row = next(json.loads(ln) for ln in lines if json.loads(ln)["entry"] == entry)
    expected = [
        v["value"]
        for v in row["values"]
        if v["id"] == "core.frames.peak" and v.get("population") is None
    ]
    peak = service.draw(entry)["frame_peak"]
    assert peak["source"] == "audit_entries.jsonl"
    assert sorted(v["value"] for v in peak["values"]) == sorted(expected)
    assert expected  # the toy audit measured it
    # a second call reads the same line from the indexed offset
    assert service.frame_peak(entry) == peak
    assert service.frame_peak("no-such-draw") is None


def test_the_draw_has_no_audit_peak_without_an_entries_file(toy_dataset, tmp_path):
    service = DatasetService(toy_dataset.data, tmp_path / "nothing")
    assert service.draw(service.rows[0].entry)["frame_peak"] is None


def test_the_entries_index_follows_the_file(audited, tmp_path):
    service, stages = audited
    copy = tmp_path / "stages"
    shutil.copytree(stages, copy)
    mine = DatasetService(service.folder, copy)
    entry = mine.rows[0].entry
    assert mine.frame_peak(entry) is not None
    (copy / "audit_entries.jsonl").write_text("")
    assert mine.frame_peak(entry) is None  # the file changed under the index


# ------------------------------------------------------------------- the page files


def test_the_page_loads_geometry_before_the_app_and_serves_both(audited):
    service, _ = audited
    api = Api(service)
    html = api.handle(Request.from_target("GET", "/")).body.decode()
    assert html.index("/static/geometry.js") < html.index("/static/app.js")
    for name in ("geometry.js", "app.js"):
        r = api.handle(Request.from_target("GET", f"/static/{name}"))
        assert r.status == 200 and r.content_type.startswith("text/javascript")
    for needed in ("speed", "f-seed", "pick-seed", "scale-text", "xlabel", "ylabel"):
        assert f'id="{needed}"' in html


# ------------------------------------------------------------- pinning (A6b)


@needs_node
def test_a_sensor_coordinate_lands_on_its_own_pixel_and_back(draw):
    layout = draw["bundle"]["layout"]
    coords = layout["coords"]
    nx, ny = layout["regular_grid"]["rows"], layout["regular_grid"]["cols"]
    got = run_js(
        """
        const bad = [];
        inp.coords.forEach((c, k) => {
          const i = Math.floor(k / inp.ny), j = k % inp.ny;
          const p = G.mmToPixel(c[0], c[1], inp.bounds, inp.nx, inp.ny);
          const q = G.pixelOf(i, j, inp.nx, inp.ny);
          if (Math.abs(p.cx - (q.cx + 0.5)) > 1e-9 ||
              Math.abs(p.cy - (q.cy + 0.5)) > 1e-9) bad.push(k);
          const back = G.pixelToMm(p.cx, p.cy, inp.bounds, inp.nx, inp.ny);
          if (Math.abs(back.x - c[0]) > 1e-9 ||
              Math.abs(back.y - c[1]) > 1e-9) bad.push(-k);
          const s = G.sensorAt(p.cx, p.cy, inp.nx, inp.ny);
          if (!s || s.k !== k) bad.push(k + 100000);
        });
        return bad;
        """,
        coords=coords,
        bounds=layout["bounds"],
        nx=nx,
        ny=ny,
    )
    assert got == []


@needs_node
def test_a_click_pins_the_nearest_neuron_centre():
    got = run_js("""
        const centers = [[0, 0], [10, 0], [0, 10], [10, 10]];
        return [G.nearestUnit(1, 1, centers), G.nearestUnit(9, 2, centers),
                G.nearestUnit(2, 8, centers), G.nearestUnit(7, 9, centers),
                G.nearestUnit(0, 0, [])];
        """)
    assert got == [0, 1, 2, 3, -1]


def run_traces(body: str, **inputs):
    script = (
        f"const T = require({json.dumps(str(STATIC / 'traces.js'))});\n"
        f"const inp = {json.dumps(inputs)};\n"
        f"const out = (() => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(done.stdout)


@needs_node
def test_the_time_cursor_and_a_click_on_a_trace_agree():
    got = run_traces("""
        const w = 500, left = 60, right = 10, dt = 2, total = 101;
        const out = [];
        for (const f of [0, 1, 37, 100]) {
          const x = T.xOfTime(f * dt, 0, (total - 1) * dt, w, left, right);
          out.push(T.frameOfX(x, w, left, right, dt, total));
        }
        out.push(T.frameOfX(-50, w, left, right, dt, total),
                 T.frameOfX(9999, w, left, right, dt, total));
        return out;
        """)
    assert got == [0, 1, 37, 100, 0, 100]


@needs_node
def test_a_series_projects_with_gaps_and_a_flat_series_gets_a_range():
    got = run_traces("""
        const view = { t0: 0, t1: 10, lo: 0, hi: 2, w: 110, h: 50,
                       left: 10, right: 0, pad: 5 };
        const pts = T.project([0, 5, 10], [0, null, 2], view);
        return { pts, flat: T.range([3, 3, null]), empty: T.range([null]),
                 ticks: T.nonzero([0, 1, 2, 3], [0, 2, 0, -1]) };
        """)
    assert got["pts"] == [[10, 45], None, [110, 5]]
    assert got["flat"]["lo"] < 3 < got["flat"]["hi"]
    assert got["empty"] == {"lo": 0, "hi": 1}
    assert got["ticks"] == {"times": [1, 3], "values": [2, -1]}


@needs_node
def test_the_voltage_probe_note_says_match_or_the_size_of_the_difference():
    got = run_traces("""
        const fmt = (v) => v.toFixed(2);
        const ok = { ids: [4, 12], matches: true, n_differing: 0,
                     max_abs_difference: 0 };
        const bad = { ids: [3], matches: false, n_differing: 1,
                      max_abs_difference: 0.25 };
        const many = { ids: [0, 1, 2, 3, 4, 5, 6, 7], matches: true, n_differing: 0 };
        return [T.probeNote(undefined, fmt), T.probeNote(ok, fmt),
                T.probeNote(bad, fmt), T.probeNote(many, fmt)];
        """)
    assert got[0] == {"text": "", "ok": True}
    assert got[1] == {
        "text": "voltage matches the 2 stored voltage probes (neurons 4, 12)",
        "ok": True,
    }
    assert got[2]["ok"] is False
    assert "differs from the 1 stored voltage probe (neuron 3)" in got[2]["text"]
    assert "largest difference 0.25 mV" in got[2]["text"]
    assert got[3]["text"].endswith("(neurons 0, 1, 2, 3, 4, 5, …)")
