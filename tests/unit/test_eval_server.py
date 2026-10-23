"""The local evaluation server's data layer and endpoints (roadmap A5a).

The data set is the toy non-tactile one (A1b), so nothing here depends on a modality.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import threading
import urllib.error
import urllib.request

import h5py
import numpy as np
import pytest

from sensoryforge.evaluate.server import (
    Api,
    DatasetService,
    Request,
    ServiceError,
    make_server,
)
from sensoryforge.evaluate.server.service import decimate


@pytest.fixture(scope="module")
def folder(toy_dataset):
    return toy_dataset.data


@pytest.fixture(scope="module")
def api(folder):
    return Api(DatasetService(folder))


def get(api, target):
    response = api.handle(Request.from_target("GET", target))
    return response.status, json.loads(response.body)


# --------------------------------------------------------------------- summary


def test_the_summary_counts_the_manifest_and_reads_the_first_bundle(api, toy_dataset):
    status, body = get(api, "/api/dataset")
    assert status == 200
    assert body["n_draws"] == len(toy_dataset.entries)
    assert sum(body["counts"]["split"].values()) == body["n_draws"]
    assert body["counts"]["status"] == {"ok": body["n_draws"]}
    assert body["name"] and body["dataset_id"]
    assert body["sensors"]["n_sensors"] > 0 and body["dt_ms"] > 0
    assert [p["name"] for p in body["populations"]]
    assert body["reference_draw"] in api.service.by_entry
    assert set(body["stages"]) == {"audit", "declared", "compare", "checks"}
    assert not any(s["present"] for s in body["stages"].values())


# ------------------------------------------------------------------- draw list


def test_the_list_pages_and_filters(api):
    _, everything = get(api, "/api/draws?limit=1000")
    total = everything["total"]
    assert len(everything["items"]) == min(total, 1000)
    _, page = get(api, "/api/draws?limit=3&offset=2")
    assert [r["entry"] for r in page["items"]] == [
        r["entry"] for r in everything["items"][2:5]
    ]
    split = everything["items"][0]["split"]
    _, only = get(api, f"/api/draws?split={split}&limit=1000")
    assert only["total"] == sum(r["split"] == split for r in everything["items"])
    assert all(r["split"] == split for r in only["items"])
    first = everything["items"][0]
    _, by_seed = get(api, f"/api/draws?seed={first['seed']}")
    assert first["entry"] in {r["entry"] for r in by_seed["items"]}
    _, text = get(api, f"/api/draws?q={first['entry'].upper()}")
    assert first["entry"] in {r["entry"] for r in text["items"]}
    _, none = get(api, "/api/draws?status=failed")
    assert none["total"] == 0


def test_the_list_sorts_and_refuses_bad_queries(api):
    _, up = get(api, "/api/draws?sort=entry&limit=1000")
    _, down = get(api, "/api/draws?sort=entry&order=desc&limit=1000")
    names = [r["entry"] for r in up["items"]]
    assert names == sorted(names) and [r["entry"] for r in down["items"]] == names[::-1]
    for target in (
        "/api/draws?limit=0",
        "/api/draws?limit=100000",
        "/api/draws?offset=x",
        "/api/draws?sort=nope",
        "/api/draws?order=sideways",
    ):
        status, body = get(api, target)
        assert status == 400 and body["error"], target


def test_a_random_pick_is_repeatable_and_respects_the_filters(api):
    _, a = get(api, "/api/draws/random?pick_seed=7")
    _, b = get(api, "/api/draws/random?pick_seed=7")
    assert a["draw"]["entry"] == b["draw"]["entry"] and a["pick_seed"] == 7
    split = a["draw"]["split"]
    _, c = get(api, f"/api/draws/random?split={split}&pick_seed=3")
    assert c["draw"]["split"] == split
    status, body = get(api, "/api/draws/random?status=failed")
    assert status == 404 and "no draw" in body["error"]


def test_a_class_filter_and_the_no_class_token(api):
    _, everything = get(api, "/api/draws?limit=1000")
    classes = {r["class"] for r in everything["items"]}
    pick = next(c for c in classes if c is not None)
    _, got = get(api, f"/api/draws?class={pick}&limit=1000")
    assert got["total"] == sum(r["class"] == pick for r in everything["items"])
    _, none = get(api, "/api/draws?class=~&limit=1000")
    assert none["total"] == sum(r["class"] is None for r in everything["items"])


# ------------------------------------------------------------------------ draw


def test_one_draw_carries_its_declaration_and_what_its_bundle_holds(api):
    _, listing = get(api, "/api/draws?limit=1")
    entry = listing["items"][0]["entry"]
    status, body = get(api, f"/api/draw?entry={entry}")
    assert status == 200 and body["error"] is None
    assert body["record"]["entry"] == entry and "draw" in body["record"]
    assert all(isinstance(v, str) for v in body["seeds"].values())
    bundle = body["bundle"]
    assert bundle["n_frames"] * bundle["dt_ms"] == pytest.approx(bundle["duration_ms"])
    coords = np.array(bundle["layout"]["coords"])
    assert coords.shape == (bundle["layout"]["n_sensors"], bundle["layout"]["dims"])
    assert [p["name"] for p in bundle["populations"]]


def test_an_unknown_or_missing_entry_is_a_clean_error(api):
    assert get(api, "/api/draw?entry=nope/0")[0] == 404
    assert get(api, "/api/draw")[0] == 400
    assert get(api, "/api/nothing")[0] == 404
    response = api.handle(Request.from_target("POST", "/api/draws"))
    assert response.status == 405 and response.headers["Allow"] == "GET"


# ---------------------------------------------------------------------- traces


def _first(api):
    _, listing = get(api, "/api/draws?limit=1")
    entry = listing["items"][0]["entry"]
    _, body = get(api, f"/api/draw?entry={entry}")
    return entry, body["bundle"]


def test_the_stimulus_trace_equals_the_stored_frames(api, folder):
    entry, bundle = _first(api)
    cols = bundle["layout"]["regular_grid"]["cols"]
    status, body = get(api, f"/api/draw/trace?entry={entry}&source=stimulus&index=0,5")
    assert status == 200 and not body["decimated"]
    with h5py.File(folder / entry / "data.h5", "r") as f:
        frames = f["stimulus/frames"][...]
    frames = frames.reshape(frames.shape[0], -1, frames.shape[-2], frames.shape[-1])
    for series in body["series"]:
        i, j = divmod(series["index"], cols)
        assert np.allclose(series["values"], frames[:, 0, i, j])
        assert len(series["time_ms"]) == bundle["n_frames"]


@pytest.mark.parametrize("source", ["drive", "filtered", "readout"])
def test_population_traces_equal_the_stored_arrays(api, folder, source):
    entry, bundle = _first(api)
    pop = bundle["populations"][0]
    status, body = get(
        api,
        f"/api/draw/trace?entry={entry}&source={source}&population={pop['name']}"
        "&index=1,0",
    )
    assert status == 200 and body["population"] == pop["name"]
    assert [s["index"] for s in body["series"]] == [0, 1]
    key = pop["readout"] if source == "readout" else source
    with h5py.File(folder / entry / "data.h5", "r") as f:
        stored = f[f"populations/{pop['name']}/{key}"][...]
    for series in body["series"]:
        assert np.allclose(series["values"], stored[:, series["index"]])


def test_trace_queries_that_cannot_be_answered_say_why(api):
    entry, bundle = _first(api)
    pop = bundle["populations"][0]["name"]
    base = f"/api/draw/trace?entry={entry}"
    cases = {
        f"{base}": 400,  # no source
        f"{base}&source=voltage": 400,
        f"{base}&source=drive": 404,  # no population
        f"{base}&source=drive&population=nope": 404,
        f"{base}&source=drive&population={pop}&index=999999": 400,
        f"{base}&source=stimulus&index=999999": 400,
        f"{base}&source=stimulus&index=" + ",".join(map(str, range(70))): 400,
        f"{base}&source=stimulus&max_points=1": 400,
        "/api/draw/trace?entry=nope&source=stimulus": 404,
    }
    for target, status in cases.items():
        assert get(api, target)[0] == status, target


def test_decimation_keeps_a_one_frame_spike():
    values = np.zeros(10000)
    values[7345] = 1.0
    kept = decimate(values, 200)
    assert kept.size <= 200 and 7345 in kept
    assert np.all(np.diff(kept) > 0)
    assert decimate(values, 20000).size == 10000


def test_a_long_trace_comes_back_decimated(api):
    entry, bundle = _first(api)
    pop = bundle["populations"][0]["name"]
    status, body = get(
        api, f"/api/draw/trace?entry={entry}&source=drive&population={pop}&max_points=8"
    )
    assert status == 200 and body["decimated"]
    assert all(len(s["values"]) <= 8 for s in body["series"])
    assert all(len(s["values"]) == len(s["time_ms"]) for s in body["series"])


# --------------------------------------------------------- damaged data sets


@pytest.fixture()
def damaged(toy_dataset, tmp_path):
    dest = tmp_path / "data"
    shutil.copytree(toy_dataset.data, dest, ignore=shutil.ignore_patterns(".partial"))
    return dest


def test_a_missing_bundle_is_reported_not_raised(damaged):
    service = DatasetService(damaged)
    api = Api(service)
    victim = service.rows[0].entry
    shutil.rmtree(damaged / victim)
    _, body = get(api, f"/api/draw?entry={victim}")
    assert body["bundle"] is None and "missing_bundle" in body["error"]
    status, trace = get(api, f"/api/draw/trace?entry={victim}&source=stimulus")
    assert status == 422 and "cannot be read" in trace["error"]
    _, summary = get(api, "/api/dataset")  # the summary skips it for the next draw
    assert summary["sensors"] is not None


def test_a_corrupt_index_line_leaves_the_status_unknown(damaged):
    path = next((damaged / "index").glob("task_*.jsonl"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(["{cut off", *lines[1:]]) + "\n")
    service = DatasetService(damaged)
    assert [r.status for r in service.rows].count(None) == 1


def test_a_folder_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetService(tmp_path)
    (tmp_path / "manifest.jsonl").write_text("not json\n")
    with pytest.raises(ValueError, match="not JSON"):
        DatasetService(tmp_path)


# ------------------------------------------------------------------ stage files


def test_stages_are_read_when_present_and_computed_on_request(damaged):
    api = Api(DatasetService(damaged))
    status, body = get(api, "/api/stage/audit")
    assert status == 404 and "computes it" in body["error"]
    assert get(api, "/api/stage/nothing")[0] == 404
    status, body = (lambda r: (r.status, json.loads(r.body)))(
        api.handle(Request.from_target("POST", "/api/stages/compare/run"))
    )
    assert status == 409 and "audit and declared" in body["error"]
    for stage in ("audit", "declared"):
        response = api.handle(Request.from_target("POST", f"/api/stages/{stage}/run"))
        assert response.status == 202
        job = api.jobs.wait(json.loads(response.body)["id"])
        assert job["state"] == "done", job
    response = api.handle(Request.from_target("POST", "/api/stages/compare/run"))
    assert api.jobs.wait(json.loads(response.body)["id"])["state"] == "done"
    status, stages = get(api, "/api/stages")
    assert all(
        s["present"] and s["readable"]
        for name, s in stages.items()
        if name != "checks"  # the checks stage is run on its own (test_eval_checks)
    )
    assert stages["checks"]["present"] is False
    assert stages["audit"]["format"] == "sensoryforge-audit/1"
    status, audit = get(api, "/api/stage/audit")
    assert status == 200 and audit["format"] == "sensoryforge-audit/1"
    assert get(api, "/api/jobs/9999")[0] == 404


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_stage_that_raises_an_unlisted_exception_still_ends_failed(
    damaged, monkeypatch
):
    from sensoryforge.evaluate.server import jobs as module

    def boom(self, stage, packs):
        def run():
            raise TypeError("a metric got a None")

        return run

    monkeypatch.setattr(module.StageJobs, "_runner", boom)
    api = Api(DatasetService(damaged))
    for _ in range(2):  # the stage can start again after it failed
        response = api.handle(Request.from_target("POST", "/api/stages/audit/run"))
        assert response.status == 202
        job = api.jobs.wait(json.loads(response.body)["id"])
        assert job["state"] == "failed"
        assert "TypeError" in job["error"] and "a metric got a None" in job["error"]
        assert job["finished_at"]


def test_a_failing_stage_job_reports_its_error(damaged):
    (damaged / "dataset.json").unlink()
    api = Api(DatasetService(damaged))
    response = api.handle(Request.from_target("POST", "/api/stages/declared/run"))
    job = api.jobs.wait(json.loads(response.body)["id"])
    assert job["state"] == "failed" and job["error"]


# ---------------------------------------------------------------------- frames


def get_frames(api, target):
    response = api.handle(Request.from_target("GET", target))
    return response


def test_a_frame_chunk_is_the_stored_frames_as_raw_float32(api, folder):
    entry = next(iter(api.service.by_entry))
    response = get_frames(api, f"/api/draw/frames?entry={entry}&start=2&count=3")
    assert response.status == 200
    assert response.content_type == "application/octet-stream"
    h = response.headers
    start, count = int(h["X-Frame-Start"]), int(h["X-Frame-Count"])
    height, width = int(h["X-Frame-Height"]), int(h["X-Frame-Width"])
    assert (start, count) == (2, 3) and h["X-Frame-Dtype"] == "float32-le"
    sent = np.frombuffer(response.body, dtype="<f4").reshape(count, height, width)
    path = api.service.reader().bundle_path(entry) / "data.h5"
    with h5py.File(path, "r") as f:
        ds = f["stimulus/frames"]
        stored = ds[2:5] if ds.ndim == 3 else ds[2:5, 0]
        assert int(h["X-Frame-Total"]) == ds.shape[0]
    assert np.array_equal(sent, stored.astype("<f4"))
    assert float(h["X-Frame-Min"]) == float(sent.min())
    assert float(h["X-Frame-Max"]) == float(sent.max())
    assert float(h["X-Frame-Dt-Ms"]) > 0


def test_the_chunks_tile_the_draw_and_the_last_one_is_short(api):
    entry = next(iter(api.service.by_entry))
    first = get_frames(api, f"/api/draw/frames?entry={entry}")
    total = int(first.headers["X-Frame-Total"])
    assert int(first.headers["X-Frame-Count"]) == min(32, total)
    seen = 0
    while seen < total:
        r = get_frames(api, f"/api/draw/frames?entry={entry}&start={seen}&count=50")
        assert r.status == 200 and int(r.headers["X-Frame-Start"]) == seen
        n = int(r.headers["X-Frame-Count"])
        assert 1 <= n <= 50 and len(r.body) == 4 * n * int(
            r.headers["X-Frame-Height"]
        ) * int(r.headers["X-Frame-Width"])
        seen += n
    assert seen == total


def test_a_chunk_stops_at_the_byte_cap(api, monkeypatch):
    from sensoryforge.evaluate.server import service as module

    entry = next(iter(api.service.by_entry))
    one = get_frames(api, f"/api/draw/frames?entry={entry}&count=1")
    frame_bytes = len(one.body)
    monkeypatch.setattr(module, "MAX_CHUNK_BYTES", 2 * frame_bytes)
    r = get_frames(api, f"/api/draw/frames?entry={entry}&count=10")
    assert int(r.headers["X-Frame-Count"]) == 2 and len(r.body) == 2 * frame_bytes


def test_frame_requests_refuse_what_they_cannot_answer(api):
    entry = next(iter(api.service.by_entry))
    for target, status in (
        ("/api/draw/frames", 400),
        ("/api/draw/frames?entry=nope", 404),
        (f"/api/draw/frames?entry={entry}&start=x", 400),
        (f"/api/draw/frames?entry={entry}&start=999999", 400),
        (f"/api/draw/frames?entry={entry}&count=0", 400),
        (f"/api/draw/frames?entry={entry}&channel=99", 400),
    ):
        response = get_frames(api, target)
        assert response.status == status, target
        assert json.loads(response.body)["error"]


# ----------------------------------------------------------------------- the page


def test_the_page_and_its_files_are_served_and_nothing_else(api):
    page = api.handle(Request.from_target("GET", "/"))
    assert page.status == 200 and page.content_type.startswith("text/html")
    assert b"/static/app.js" in page.body
    for name, kind in (("app.js", "javascript"), ("style.css", "text/css")):
        r = api.handle(Request.from_target("GET", f"/static/{name}"))
        assert r.status == 200 and kind in r.content_type and r.body
    assert api.handle(Request.from_target("GET", "/favicon.ico")).status == 204
    for target in (
        "/static/nope.js",
        "/static/../api.py",
        "/static/%2e%2e/api.py",
        "/static/.hidden",
    ):
        assert api.handle(Request.from_target("GET", target)).status == 404, target


# ------------------------------------------------------------------ slab cache


def _stored(service, entry, lo, hi, channel=0):
    with h5py.File(service.reader().bundle_path(entry) / "data.h5", "r") as f:
        ds = f["stimulus/frames"]
        return (ds[lo:hi] if ds.ndim == 3 else ds[lo:hi, channel]).astype("<f4")


def _run(service, entry, start, count, channel=0):
    chunk = service.frames(
        entry, {"start": str(start), "count": str(count), "channel": str(channel)}
    )
    return np.frombuffer(chunk.body, dtype="<f4").reshape(
        chunk.count, chunk.height, chunk.width
    )


def test_a_run_across_a_slab_boundary_equals_the_stored_frames(folder):
    service = DatasetService(folder)
    entry = service.rows[0].entry
    # the toy bundle stores 25-frame chunks, so frames 20-29 cross one boundary
    # and 10-79 cross two
    for channel in (0, 1):
        for start, count in ((20, 10), (10, 70), (0, 5), (95, 5)):
            got = _run(service, entry, start, count, channel)
            assert np.array_equal(
                got, _stored(service, entry, start, start + count, channel)
            )


def test_chunks_in_playback_order_decompress_a_slab_once(folder):
    service = DatasetService(folder)
    entry = service.rows[0].entry
    for start in range(0, 100, 5):
        _run(service, entry, start, 5)
    stats = service.cache_stats()
    assert stats["misses"] == 4  # 100 frames in 25-frame slabs
    assert stats["hits"] == 16 and stats["slabs"] == 4


def test_the_slab_cache_stays_within_its_byte_limit(folder):
    probe = DatasetService(folder)
    entry = probe.rows[0].entry
    one = _run(probe, entry, 0, 25).nbytes
    service = DatasetService(folder, frame_cache_bytes=2 * one + 1)
    for start in (0, 25, 50, 75, 0):
        _run(service, entry, start, 25)
        assert service.cache_stats()["bytes"] <= 2 * one + 1
    stats = service.cache_stats()
    assert stats["slabs"] == 2 and stats["misses"] == 5  # 0 had been evicted
    got = _run(service, entry, 60, 20)
    assert np.array_equal(got, _stored(service, entry, 60, 80))


def test_a_slab_larger_than_the_cache_is_served_and_not_kept(folder):
    service = DatasetService(folder, frame_cache_bytes=1)
    entry = service.rows[0].entry
    got = _run(service, entry, 20, 10)
    assert np.array_equal(got, _stored(service, entry, 20, 30))
    assert service.cache_stats()["slabs"] == 0 and service.cache_stats()["bytes"] == 0
    with pytest.raises(ValueError):
        DatasetService(folder, frame_cache_bytes=-1)


# -------------------------------------------- bundles damaged after they were read


def test_an_h5_error_after_the_peek_is_a_422_naming_the_draw(damaged):
    service = DatasetService(damaged)
    api = Api(service)
    gone, cut = service.rows[0].entry, service.rows[1].entry
    for entry in (gone, cut):
        service.info(entry)  # the peek succeeds and is cached
    path = service.reader().bundle_path(gone) / "data.h5"
    with h5py.File(path, "a") as f:
        del f["stimulus/frames"]
    path = service.reader().bundle_path(cut) / "data.h5"
    os.truncate(path, path.stat().st_size // 2)
    for entry in (gone, cut):
        response = api.handle(
            Request.from_target("GET", f"/api/draw/frames?entry={entry}")
        )
        assert response.status == 422
        assert entry in json.loads(response.body)["error"]
        status, body = get(api, f"/api/draw/trace?entry={entry}&source=stimulus")
        assert status == 422 and "cannot be read" in body["error"]


# ------------------------------------------------- a data set with no bundle yet


def test_the_summary_stops_trying_when_no_bundle_reads(damaged, monkeypatch):
    from sensoryforge.evaluate.server import service as module

    for h5 in list(damaged.rglob("data.h5")):
        shutil.rmtree(h5.parent)
    service = DatasetService(damaged)
    assert len(service.rows) > 3
    monkeypatch.setattr(module, "SUMMARY_MAX_PEEKS", 3)
    peeks = []
    real = service.reader().peek
    monkeypatch.setattr(
        service.reader(), "peek", lambda entry: peeks.append(entry) or real(entry)
    )
    for _ in range(2):
        summary = Api(service).handle(Request.from_target("GET", "/api/dataset"))
        body = json.loads(summary.body)
        assert body["sensors"] is None and body["reference_draw"] is None
        assert "no bundle read in the first 3 draws" in body["reference_note"]
    assert len(peeks) == 6  # three per call, not one per row


def test_the_summary_remembers_the_reference_draw(folder):
    service = DatasetService(folder)
    first = service.summary()["reference_draw"]
    assert first and service._reference == first
    assert service.summary()["reference_draw"] == first


# ----------------------------------------------------------------- HTTP backend


def test_the_standard_library_backend_serves_the_api(folder):
    server = make_server(Api(DatasetService(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/api/health") as r:
            assert r.status == 200 and json.loads(r.read()) == {"ok": True}
            assert r.headers["Content-Type"] == "application/json"
        with urllib.request.urlopen(f"{base}/api/draws?limit=2") as r:
            assert len(json.loads(r.read())["items"]) == 2
        entry = next(iter(Api(DatasetService(folder)).service.by_entry))
        with urllib.request.urlopen(
            f"{base}/api/draw/frames?entry={entry}&count=2"
        ) as r:
            n = int(r.headers["X-Frame-Count"])
            assert len(r.read()) == 4 * n * int(r.headers["X-Frame-Height"]) * int(
                r.headers["X-Frame-Width"]
            )
        with urllib.request.urlopen(f"{base}/") as r:
            assert r.headers["Content-Type"].startswith("text/html")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/draw?entry=nope")
        assert err.value.code == 404 and json.loads(err.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def _raw(server, method, path, headers):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in headers.items():
            conn.putheader(key, value)
        conn.putheader("Content-Length", "0")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_a_request_that_is_not_the_pages_own_is_refused(damaged):
    server = make_server(Api(DatasetService(damaged)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    own = f"127.0.0.1:{port}"
    try:
        # a wrong Host (DNS rebinding) is refused for reads and writes
        status, body = _raw(server, "GET", "/api/health", {"Host": f"evil.test:{port}"})
        assert status == 403 and json.loads(body)["error"]
        status, _ = _raw(server, "GET", "/api/dataset", {"Host": "evil.test"})
        assert status == 403
        # a foreign Origin on a POST is refused and starts no job
        for origin in ("http://evil.test", "null", f"http://localhost:{port + 1}"):
            status, _ = _raw(
                server, "POST", "/api/stages/audit/run", {"Host": own, "Origin": origin}
            )
            assert status == 403, origin
        assert not (damaged / "evaluate" / "audit.json").exists()
        # the page's own requests still work: no Origin on a GET, same origin on a POST
        assert _raw(server, "GET", "/api/health", {"Host": own})[0] == 200
        assert (
            _raw(server, "GET", "/api/health", {"Host": f"localhost:{port}"})[0] == 200
        )
        assert _raw(server, "GET", "/api/health", {"Host": f"[::1]:{port}"})[0] == 200
        status, body = _raw(
            server,
            "POST",
            "/api/stages/declared/run",
            {"Host": own, "Origin": f"http://{own}"},
        )
        assert status == 202 and json.loads(body)["stage"] == "declared"
    finally:
        server.shutdown()
        server.server_close()


# -------------------------------------------------------------------------- CLI


def test_the_cli_has_an_evaluate_serve_command(tmp_path, capsys):
    from sensoryforge import cli

    args = cli.create_parser().parse_args(
        ["evaluate", "serve", str(tmp_path), "--port", "9"]
    )
    assert (args.command, args.evaluate_command, args.port) == ("evaluate", "serve", 9)
    assert args.host == "127.0.0.1" and args.stages is None
    assert args.frame_cache_mb == 256.0
    assert cli.cmd_evaluate(args) == 1  # no manifest in the folder
    assert "Error opening data set" in capsys.readouterr().err
