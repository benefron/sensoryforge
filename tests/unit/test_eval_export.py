"""The static export (roadmap A12): a snapshot answers the pages as the server does.

A small toy data set is exported; every recorded answer must equal what the live API
gives, the pages must hold no absolute link, the shim must key requests as the exporter
does, and (with node) the shim must serve a recorded request, load a draw's file on
first use and refuse a request it holds nothing for. Movie files are written only
with ffmpeg.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sensoryforge.evaluate.audit.run import run_audit
from sensoryforge.evaluate.checks import run_checks
from sensoryforge.evaluate.export import ExportOptions, export_snapshot, request_key
from sensoryforge.evaluate.export import video as video_module
from sensoryforge.evaluate.export.arguments import add_arguments
from sensoryforge.evaluate.export.build import rewrite_page
from sensoryforge.evaluate.export.snapshot import walk_queries
from sensoryforge.evaluate.server import Api, DatasetService, Request

STAMP = "2026-10-23T14:00:00Z"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """``(service, exported result)`` for a small toy data set."""
    from tests.fixtures import toy_vision

    root = tmp_path_factory.mktemp("export")
    built = toy_vision.build_toy_dataset(root / "built", n_train=6)
    stages = root / "stages"
    run_audit(built.data, stages, created_at=STAMP)
    run_checks(built.data, stages, sample=1, created_at=STAMP)
    service = DatasetService(built.data, stages=stages)
    result = export_snapshot(
        service, root / "out", ExportOptions(n_draws=2, made_at="test time")
    )
    return service, result


def load_index(out: Path) -> dict:
    text = (out / "snapshot" / "index.js").read_text(encoding="utf-8")
    return json.loads(text[len("window.EvalSnapshot=") : -2])


def load_part(out: Path, name: str) -> dict:
    text = (out / name).read_text(encoding="utf-8")
    return json.loads(text[text.index("]=") + 2 : -2])


def live(service, path, query):
    r = Api(service).handle(
        Request.from_target(
            "GET", path + "?" + "&".join(f"{k}={v}" for k, v in query.items())
        )
    )
    return r.status, r.body


def test_request_key_ignores_the_order_of_the_query():
    assert request_key("/api/x", {"b": "2", "a": "1"}) == request_key(
        "/api/x", {"a": "1", "b": "2"}
    )
    assert request_key("/api/x", {}) == "/api/x?[]"
    assert request_key("/api/x", {"a": "é b"}) == '/api/x?[["a","é b"]]'


def test_walk_queries_finds_every_filter_once():
    payload = {
        "counts": {"split": [{"label": "t", "n": 1, "query": "split=train"}]},
        "histograms": [
            {"queries": ["bin=a%3A0", "bin=a%3A1"], "x": {"query": "split=train"}}
        ],
    }
    assert walk_queries(payload) == ["split=train", "bin=a%3A0", "bin=a%3A1"]


def test_pages_hold_no_absolute_link_and_load_the_snapshot_first():
    html = (
        '<link rel="stylesheet" href="/static/style.css">'
        '<a href="/">a</a><a href="/dashboard">b</a><a href="/gallery">c</a>'
        '<a href="/checks">d</a><script src="/static/geometry.js"></script>'
    )
    out = rewrite_page(html)
    assert not re.search(r'(?:href|src)="/', out)
    assert (
        out.index("snapshot/index.js")
        < out.index("snapshot_shim.js")
        < out.index("static/geometry.js")
    )
    with pytest.raises(ValueError, match="no script"):
        rewrite_page("<html></html>")


def test_the_folder_holds_the_pages_and_their_files(toy):
    _, result = toy
    out = result.out
    for page in ("index.html", "dashboard.html", "gallery.html", "checks.html"):
        text = (out / page).read_text(encoding="utf-8")
        assert not re.search(r'(?:href|src)="/', text), page
        assert 'src="snapshot/index.js"' in text
    assert (out / "static" / "snapshot_shim.js").is_file()
    assert (out / "static" / "app.js").is_file()
    assert not list((out / "static").glob("*.html"))
    assert len(result.draws) == 2
    assert sorted(p.name for p in (out / "snapshot").glob("draw_*.js")) == [
        "draw_0.js",
        "draw_1.js",
    ]


def test_every_recorded_answer_is_the_servers_answer(toy):
    service, result = toy
    index = load_index(result.out)
    table = dict(index["entries"])
    for name in set(index["files"].values()):
        table.update(load_part(result.out, name))
    assert len(table) > 40
    checked = 0
    for key, entry in table.items():
        path, query = key.split("?", 1)
        q = {k: v for k, v in json.loads(query)}
        status, body = live(service, path, q)
        assert entry["s"] == status, key
        if entry.get("x"):
            import base64

            assert base64.b64decode(entry["b"]) == body, key
        else:
            got, want = json.loads(entry["b"]), json.loads(body)
            if (
                "/voltage" in path
            ):  # whether the service had the run cached is not an answer
                got.pop("cached"), want.pop("cached")
            assert got == want, key
        checked += 1
    assert checked == len(table)


def test_a_chosen_draw_has_all_its_frames_and_default_traces(toy):
    service, result = toy
    index = load_index(result.out)
    entry = result.draws[0]
    draw = json.loads(
        index["entries"][request_key("/api/draws", {"limit": "100", "offset": "0"})][
            "b"
        ]
    )
    assert draw["total"] == len(service.rows)
    part = load_part(
        result.out, index["files"][request_key("/api/draw", {"entry": entry})]
    )
    bundle = json.loads(part[request_key("/api/draw", {"entry": entry})]["b"])["bundle"]
    starts = range(0, bundle["n_frames"], 32)
    for s in starts:
        key = request_key(
            "/api/draw/frames",
            {"entry": entry, "start": str(s), "count": "32", "channel": "0"},
        )
        assert key in part
    sources = {
        k.split('"source","')[1].split('"')[0] for k in part if "/api/draw/trace" in k
    }
    assert sources == {"stimulus", "drive", "filtered", "readout"}
    assert any("/api/draw/voltage" in k for k in part)


def test_a_draw_that_is_not_in_the_data_set_is_refused(toy, tmp_path):
    service, _ = toy
    with pytest.raises(ValueError, match="nope"):
        export_snapshot(service, tmp_path / "x", ExportOptions(draws=["nope"]))


def test_a_folder_that_is_not_empty_is_refused(toy, tmp_path):
    service, _ = toy
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        export_snapshot(service, tmp_path)


def test_the_size_limit_stops_the_export(toy, tmp_path):
    service, _ = toy
    with pytest.raises(ValueError, match="size limit"):
        export_snapshot(
            service, tmp_path / "x", ExportOptions(n_draws=1, max_mb=0.0001)
        )


def test_no_pins_records_no_single_point_traces(toy, tmp_path):
    service, _ = toy
    result = export_snapshot(
        service, tmp_path / "x", ExportOptions(n_draws=1, sensors=[], units=[])
    )
    part = load_part(result.out, "snapshot/draw_0.js")
    assert not any("/api/draw/trace" in k or "/api/draw/voltage" in k for k in part)


def test_the_command_line_arguments_parse():
    import argparse

    ap = argparse.ArgumentParser()
    add_arguments(ap)
    a = ap.parse_args(
        ["data", "out", "--unit", "pop:3", "--sensor", "4", "--video", "gif"]
    )
    assert a.unit == ["pop:3"] and a.sensor == [4] and a.video == "gif"


def run_node(script: str, cwd: Path):
    done = subprocess.run(
        [NODE, "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        cwd=cwd,
    )
    return json.loads(done.stdout)


@needs_node
def test_the_shim_serves_the_snapshot_and_refuses_the_rest(toy):
    _, result = toy
    entry = result.draws[0]
    script = f"""
    const fs = require("fs");
    global.window = global;
    global.document = {{ addEventListener() {{}}, createElement() {{
      const s = {{}}; Object.defineProperty(s, "src", {{ set(v) {{
        setTimeout(() => {{
          eval(fs.readFileSync(v, "utf8")); s.onload(); }}, 0); }} }});
      return s; }}, head: {{ append() {{}} }} }};
    eval(fs.readFileSync("snapshot/index.js", "utf8"));
    eval(fs.readFileSync("static/snapshot_shim.js", "utf8"));
    (async () => {{
      const out = {{}};
      let r = await fetch("/api/dataset");
      out.dataset = [r.status, (await r.json()).n_draws];
      const e = encodeURIComponent({json.dumps(entry)});
      const q = "channel=0&count=32&start=0&entry=" + e;
      r = await fetch("/api/draw/frames?" + q);
      const h = (n) => r.headers.get(n);
      out.frames = [r.status, (await r.arrayBuffer()).byteLength,
        h("X-Frame-Start"), h("X-Frame-Total")];
      r = await fetch("/api/draw?entry=nope");
      out.missing = [r.status, (await r.json()).error.includes("not in this snapshot")];
      r = await fetch("/api/stages/checks/run", {{ method: "POST" }});
      out.post = [r.status, (await r.json()).error.includes("read-only")];
      console.log(JSON.stringify(out));
    }})();
    """
    got = run_node(script, result.out)
    assert got["dataset"][0] == 200 and got["dataset"][1] > 0
    assert got["frames"][0] == 200 and got["frames"][1] > 0 and got["frames"][2] == "0"
    assert int(got["frames"][3]) > 0
    assert got["missing"] == [404, True]
    assert got["post"] == [405, True]


@needs_node
def test_the_shim_keys_a_request_as_the_exporter_does(toy):
    _, result = toy
    script = """
    const fs = require("fs");
    global.window = global;
    global.document = { addEventListener() {} };
    global.EvalSnapshot = { entries: {}, files: {}, draws: [], made_at: "t" };
    let seen;
    const src = fs.readFileSync("static/snapshot_shim.js", "utf8");
    eval(src.replace("window.fetch = answer;", "global.__a = answer;"));
    EvalSnapshot.entries['/api/x?[["a","1"],["b","é b"]]'] =
      { s: 200, t: "application/json", b: "{}" };
    (async () => {
      const r = await __a("/api/x?b=%C3%A9+b&a=1");
      console.log(JSON.stringify({ status: r.status }));
    })();
    """
    assert run_node(script, result.out) == {"status": 200}


def test_the_movie_command_scales_and_pads_for_each_format(tmp_path):
    mp4 = video_module.ffmpeg_command("ffmpeg", 40, 30, 30, tmp_path / "a.mp4", "mp4")
    gif = video_module.ffmpeg_command("ffmpeg", 40, 30, 30, tmp_path / "a.gif", "gif")
    assert "scale=iw*12:ih*12:flags=neighbor,pad=ceil(iw/2)*2:ceil(ih/2)*2" in mp4
    assert any("palettegen" in x for x in gif)
    with pytest.raises(ValueError, match="mp4"):
        video_module.ffmpeg_command("ffmpeg", 4, 4, 30, tmp_path / "a", "avi")


def test_the_movie_colours_match_the_browsers():
    lut = video_module._lut()
    assert lut[0].tolist() == [68, 1, 84] and lut[255].tolist() == [253, 231, 37]
    # axis 0 is x and axis 1 is y; the picture has y upwards, so the largest y is row 0
    frame = np.array([[0.0, 1.0], [0.0, 0.0]])  # x=0,y=1 is the only high value
    rgb = video_module.to_rgb(frame, 0.0, 1.0, lut)
    assert rgb.shape == (2, 2, 3) and rgb[0, 0].tolist() == lut[255].tolist()
    assert rgb[1, 0].tolist() == lut[0].tolist()
    assert video_module.scale_for(5.0, -1.0, 2.0) == (-1.0, 5.0)
    assert video_module.scale_for(None, 0.5, 2.0) == (0.5, 2.0)
    assert video_module.scale_for(float("nan"), 0.0, 2.0) == (0.0, 2.0)


def test_a_movie_without_ffmpeg_says_so(toy, tmp_path, monkeypatch):
    service, _ = toy
    monkeypatch.setattr(video_module, "ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
        video_module.write_video(service, service.rows[0].entry, tmp_path / "a.mp4")
    from sensoryforge.evaluate.export import build

    monkeypatch.setattr(build, "ffmpeg_path", lambda: None)
    result = export_snapshot(
        service, tmp_path / "x", ExportOptions(n_draws=1, video="mp4")
    )
    assert result.videos == [] and any("ffmpeg" in n for n in result.notes)


@needs_ffmpeg
def test_a_movie_is_written_when_ffmpeg_exists(toy, tmp_path):
    service, _ = toy
    for fmt in ("mp4", "gif"):
        path = video_module.write_video(
            service, service.rows[0].entry, tmp_path / f"a.{fmt}", fmt, speed=4.0
        )
        assert path.stat().st_size > 100


def test_the_cli_writes_a_snapshot_and_reports_a_bad_folder(toy, tmp_path, capsys):
    from sensoryforge import cli

    service, _ = toy
    args = cli.create_parser().parse_args(
        [
            "evaluate",
            "export",
            str(service.folder),
            str(tmp_path / "o"),
            "--n-draws",
            "1",
            "--no-pins",
        ]
    )
    assert cli.cmd_evaluate(args) == 0
    assert (tmp_path / "o" / "index.html").is_file()
    assert "draws in full" in capsys.readouterr().out
    bad = cli.create_parser().parse_args(
        [
            "evaluate",
            "export",
            str(tmp_path / "missing"),
            str(tmp_path / "p"),
            "--unit",
            "x",
        ]
    )
    assert cli.cmd_evaluate(bad) == 2
    assert "error" in capsys.readouterr().err
