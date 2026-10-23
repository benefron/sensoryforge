# R-A5 · Review of the evaluation server (streaming design and draw-open time)

Date 2026-10-23 · reviewer Claude Opus 5.5 · branch `phase/a-evaluate` at `ac63641` · scope
roadmap rows A5a and A5b (`sensoryforge/evaluate/server/`, `benchmarks/eval_server_first_frame.py`,
`docs/developer_guide/evaluation_server.md`, the `evaluate serve` command).

**Verdict:** the design holds and the draw-open target is met on real-size bundles. A draw's
first frame arrives in 15–50 ms, against a 1 s target. Playback does not hold. The page asks
for 32 frames at a time, but HDF5 decompresses the stored chunk, which holds hundreds of
frames, and the server keeps nothing between requests. A 120 s session costs about 185 ms
per 32 frames, where the whole 3750-frame slab costs 219 ms. Two smaller defects should also
be fixed before A6 builds on the server: a stage job can stay `running` for good, and the
loopback server accepts requests from other web pages. Row **A5f** (Sonnet, start of S6)
makes the server changes listed under "Required changes". The page-side points go to A6a.

## How this was checked

- The diff `8fdd080..ac63641`: service, API, HTTP adapter, jobs, the page, the benchmark and
  the docs page.
- A real tactile data set at the size the server is meant for. It is built from
  `tests/fixtures/worlds/tactile_small.yml` on an 80×80 grid at 0.15 mm, with gaussian
  receptive fields, Izhikevich neurons and dt 1 ms. It has three 5 s entries and one 30 s
  session (`docs_root/plan/scripts/r_a5_build.py`, 15 s to build). It was timed over real
  HTTP through `make_server` (`docs_root/plan/scripts/r_a5_time.py`).
- The 120 s session layout without the engine: the 30 s session's frames repeated four
  times into an h5 file with the chunk shape h5py picks for `[120000, 80, 80]`, gzip 4.
  Building a real 120 s entry needs the 3.1 GB that F-29e949e describes.
- The job runner, with a stage that raises a `TypeError`.
- The full suite, once, in the background (result in the R-A5 session log).

## What holds

- **Three layers.** `DatasetService` (no HTTP), `Api.handle(Request) -> Response` (no
  sockets) and the standard-library adapter (D-83bf844). A FastAPI backend would only
  replace `http_server.py`. The tests call `Api.handle` directly, plus one test over a real
  socket.
- **Draws on demand.** The index is one manifest scan and one index scan. No bundle is
  opened to list draws, and the record is read from its byte offset. Bundles open only
  through the audit's reader (D-9f6c77b), so a damaged one gives a `422` for that draw.
- **The wire format.** Raw little-endian float32 with the shape in `X-Frame-*` headers needs
  no encoder and no client library (D-7e30f2b, D-11c766e). The body is exactly the stored
  values. A chunk is capped at 8 MiB and is short at the end of a draw.
- **Input checks.** Every query number is range-checked (`_int`). The static route's name
  regex allows no `/` and no leading `.`. Trace ids are sorted for h5py. A non-finite trace
  value becomes `null`, so the JSON stays valid (`allow_nan=False`).
- **The A5 target.** The first frame comes well under 1 s on every size measured (table
  below), and the draw count does not matter: the benchmark's 1.3 ms on 10,000 toy draws
  stands.

## Measured on real-size bundles (through the server)

| Request | 5 s entry `[5000,80,80]`, chunks `(313,5,10)` | 30 s session `[30000,80,80]`, chunks `(938,5,5)` | 120 s layout `[120000,80,80]`, chunks `(3750,3,5)`, h5py only |
|---|---|---|---|
| `/api/draw` (record and 6400 coordinates) | 5 ms, 181 kB | 16 ms, 485 kB | |
| first chunk, 32 frames | 15–17 ms | 50 ms | 185 ms |
| each later chunk in playback order (median, max) | 12–15 ms, ≤ 45 ms | 46 ms, 88 ms | 180–190 ms |
| seek to the middle | 21–29 ms | 88 ms | 180 ms |
| the whole aligned time slab | | | 219 ms (96 MB) |
| stimulus trace, 1 sensor | 3 ms | 19 ms | 21 ms |
| stimulus trace, 64 sensors | 104–123 ms | 643 ms | |
| readout trace, 64 units | 16 ms | 25 ms | |

Apple-silicon laptop, `OMP_NUM_THREADS=1`, one run each.

## Findings

| # | Severity | Where | Finding |
|---|---|---|---|
| 1 | design | `service.py` `frames()` | **Read amplification.** `stimulus/frames` is written with gzip and h5py's automatic chunk shape (`io/bundle.py`). The time extent of that chunk grows with the draw: 313 frames at 5 s, 938 at 30 s, 3750 at 120 s. A 32-frame request decompresses every chunk it touches, which means the whole time slab, and keeps none of it. The 120 s layout costs 185 ms per 32 frames, against 219 ms for the full 3750-frame slab, so about 117 times the necessary work. At 1 ms frames, real-time playback at 60 Hz needs a chunk about every 33 ms. The 5 s entries manage that, but the 30 s session (46 ms) and the 120 s layout (185 ms) cannot. |
| 2 | bug | `jobs.py` `work()` | A stage that raises anything other than `OSError`, `ValueError`, `KeyError` or `RuntimeError` (here a `TypeError` from a metric) leaves its job `running` for good. `start()` then refuses that stage with `409 already running` until the server restarts. Reproduced. |
| 3 | security | `http_server.py` | The server checks neither `Host` nor `Origin`. Any web page open in the same browser can `POST /api/stages/audit/run` (a simple cross-site request needs no preflight) and start CPU-heavy work that writes to the data set's folder. Through DNS rebinding, such a page can also read every `GET` endpoint, which means the data set itself. The loopback bind does not prevent either. |
| 4 | robustness | `service.py` `frames()`, `trace()` | An h5py error after `peek` succeeded (a damaged chunk, a missing dataset) is not a `ServiceError`. It escapes `Api.handle`, the adapter prints a traceback and drops the connection, and the browser sees a network error instead of a `422` naming the draw. |
| 5 | minor | `service.py` `summary()` | When no bundle reads yet (a batch still running, so the status is `None` everywhere), every `/api/dataset` call peeks every row in turn. Failed peeks are not cached. |
| 6 | page (A6a) | `static/app.js` `paint()` | **Orientation.** Frame axis 0 is x (`indexing="ij"`, `layout.centered_lattice`), and the canvas draws axis 0 downwards. The movie therefore shows +x pointing down and +y to the right, a transpose of an x–y plot. A shape sliding in +x moves down the screen. A6b maps canvas clicks to sensors, so this must be settled first. |
| 7 | page (A6a) | `static/app.js` | **Colour scale.** The scale is the running min/max of the chunks fetched so far, so brightness can change during first playback (A5b's note 2). `core.frames.peak` is already in the audit's entries file for every draw. |
| 8 | page (A6a) | `static/app.js` `tick()`, `show()` | **Playback.** The timer waits `min(dt, 50)` ms per frame. Browsers clamp that to at least about 4 ms, so 1 ms frames play at about a quarter of real time, and no frame is ever skipped. A chunk whose prefetch is still in flight is fetched a second time. A failed fetch during play stops playback silently, because the rejection is not handled. |
| 9 | gap (A6b) | bundles | A6's list includes "accumulated current" and "membrane voltage". Bundles store `stimulus/frames`, `drive`, `filtered` and the readout (`spikes`, `events` or `state`), but no voltage and no accumulated current. A6b can show what bundles hold. Voltage needs either a bundle-schema addition (a format bump, and pressure-simulation's contract) or re-simulating one neuron on demand. That choice is Ben's. |
| 10 | cost (A6b) | `trace(source=stimulus)` | A stimulus trace reads one pixel column through every time chunk: about 10 ms per sensor per 30 s of stimulus (64 sensors take 643 ms on the session). This is fine for a few pins. All reads also serialize on h5py's global lock, so a large trace request holds up frame chunks behind it. |
| 11 | note | A5a's log | It says 30 tests. The file has 24 test functions (26 cases) from A5a plus 5 from A5b (A5b's note 5). Nothing to change. |

## Decisions

- **D1 (finding 1).** The server reads stimulus frames in **time slabs aligned to the stored
  chunk**: `ds.chunks[0]` frames, or the whole draw when the dataset is not chunked. It
  keeps the decompressed slabs in one **byte-bounded LRU** keyed `(entry, channel, slab
  start)`, 256 MiB by default (`DatasetService(..., frame_cache_bytes=...)`, and
  `--frame-cache-mb` on `evaluate serve`). A slab larger than the whole cache is read and
  served without being kept. The wire format, the 32-frame default and the 8 MiB cap stay,
  so the page does not change for this. The cache is guarded by the service lock, and two
  requests for the same slab decompress it once. This supersedes the "keeps no cache" half
  of D-11c766e; the rest of that decision stands. **Rejected:** changing the bundle
  writer's chunk shape. Every frame value would stay the same, but the h5 bytes of every
  bundle that pressure-simulation pins would change, and existing data sets would gain
  nothing. If Ben wants it, it is a separate, contract-reviewed change.
- **D2 (finding 2).** A job always leaves `running`. `work()` keeps its list of expected
  exceptions, and a `finally` marks the job `failed` with `"unexpected <Type>: <message>"`
  (read from `sys.exc_info()`) when none of them matched. The exception still reaches
  `threading.excepthook`, so the traceback stays in the server's log. This follows the
  project rule against `except Exception`.
- **D3 (finding 3).** The adapter answers `403` to a request whose `Host` is not one of the
  bound name, `127.0.0.1`, `localhost` or `[::1]` (with the bound port). This blocks DNS
  rebinding. It also answers `403` to a `POST` whose `Origin` header is present and is not
  the server's own origin, which blocks cross-site requests. Both checks happen in the
  adapter, before `Api.handle`, and a later FastAPI backend would need the same checks.
  When someone binds a non-loopback host on purpose, its name joins the allowed `Host`
  values. The docs page says this does not make the server safe to expose.
- **D4 (finding 4).** `frames()` and `trace()` turn `OSError`, `KeyError` and `ValueError`
  from the h5 read into `ServiceError(422, "draw … cannot be read (…)")`, the same message
  shape that `info()` uses.
- **D5 (finding 5).** `summary()` remembers its reference draw once one reads. While none
  reads, it stops after the first 50 failed peeks and says so (`"reference_draw": null`,
  `"sensors": null`).
- **For A6a (findings 6–8), required there, not in A5f.** (a) Draw x to the right and y
  upwards: transpose and flip axis 0 when painting, and label the axes from `/api/draw`'s
  layout. Pin the mapping with a page test that puts a value at known sensor coordinates.
  (b) Colour scale: when the audit's entries file is present, use the draw's
  `core.frames.peak` with a lower bound of `min(0, the running minimum)`, otherwise the
  running range, and show which one is in use. (c) Play by wall clock with
  `requestAnimationFrame`, a speed control, and frame skipping when the speed is above
  what the display shows. Keep one promise per chunk start so a chunk is never fetched
  twice, and stop playback with the error shown when a fetch fails.
- **Open for Ben (finding 9).** Should the inspector show membrane voltage? It could come
  from storing voltage in bundles (a schema change), from re-simulating the pinned neuron
  on demand, or not at all for now. Until then, A6b shows what bundles hold and says that
  voltage is not stored.

## Required changes (row A5f)

1. D1 in `service.py`: a slab cache with a byte limit and hit and miss counters. `frames()`
   copies its run out of the cached slab or slabs; a run can cross a slab boundary. Tests:
   (a) a run that crosses a slab boundary equals `ds[start:start+count]` bit for bit;
   (b) sequential chunks through one slab decompress it once (count the reads with a
   wrapped reader or the counters); (c) eviction keeps the total at or under the limit;
   (d) a slab larger than the limit is served and not kept. The CLI gets `--frame-cache-mb`.
2. D2 in `jobs.py`, with a test where the runner raises `TypeError`: the job ends
   `failed`, the error names `TypeError`, and the stage can start again.
3. D3 in `http_server.py`, with real-socket tests: a wrong `Host` gives 403, a foreign
   `Origin` on a POST gives 403, and the page's own requests (no `Origin` on a GET,
   same-origin `Origin` on a POST) still work.
4. D4, with a test on a bundle whose `stimulus/frames` is truncated or removed after the
   first `peek`.
5. D5, with a test on a data set where no bundle exists yet.
6. Re-run `docs_root/plan/scripts/r_a5_build.py` and `r_a5_time.py`. Expected: later
   chunks in playback order cost about 1–2 ms on the 30 s session (one slab read for each
   938 frames). Put the before and after numbers in `evaluation_server.md`'s "Time to
   first frame" section, which should state the slab cache and the security checks.
   Update the docs page's "Nothing is cached" sentence and the A5 log.
