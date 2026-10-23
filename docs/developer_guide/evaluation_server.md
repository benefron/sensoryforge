# The local evaluation server

The server lets a person look at a data set in a browser instead of through numbers: the
list of draws, one draw's declaration and the traces of its stimulus and populations, and
the audit and comparison files when they exist. It reads one data-set folder from the local
disk. Draws are read **on demand**, so a large data set never has to be rendered in advance.

This page covers the data and JSON endpoints (roadmap A5a), the frame streaming and the page
shell (A5b). The richer draw browser is built on the shell in A6.

## Run it

```bash
sensoryforge evaluate serve DATASET_DIR            # http://127.0.0.1:8765/
python -m sensoryforge.evaluate.server DATASET_DIR --port 9000 --stages out/evaluate
```

`--stages` is the folder of `audit.json`, `declared.json` and `compare.json` (default
`DATASET_DIR/evaluate`, where `python -m sensoryforge.evaluate.audit` writes by default). The
server binds to the loopback address and has no authentication: it reads local files and
offers to run the stages over them, so do not bind it to a public address.

## Three layers

| Layer | File | Knows about |
|---|---|---|
| `DatasetService` | `evaluate/server/service.py` | the folder: manifest, index, bundles. No HTTP. |
| `Api` | `evaluate/server/api.py` | routes. `Api.handle(Request) -> Response`; no sockets. |
| the backend | `evaluate/server/http_server.py` | the standard library's `ThreadingHTTPServer`. |

The endpoints sit behind `Api.handle`, so a FastAPI and Uvicorn backend can replace
`http_server.py` in a later version by calling the same method; nothing else changes.

The server is a viewer, not the blind audit. It reads the whole manifest, because a draw's
declared record is what the viewer shows next to its data. It still opens bundles through the
audit's [reader](audit_reader.md), so a missing or damaged bundle is reported for that draw
(`422` with the reader's code) and never stops the server. The server is not imported by the
audit.

## The index

At start-up the service parses `manifest.jsonl` once and `index/task_*.jsonl` once, and keeps
for each entry its labels (split, class, repeat), its draw seed (as a decimal string, since
seeds exceed what a browser holds exactly), its duration, its batch status and run time, and
the byte offset of its manifest line. No bundle is opened to list draws. A draw's full record
is read from its line when asked for. On a synthetic 10,000-draw manifest of 32 MB the index
takes about 0.3 s, and a filtered page of the list about 1 ms.

A corrupt index line is skipped; the entry it described shows status `unknown` and its bundle
is still read.

## Endpoints

All answers are JSON; an error is `{"error": message}` with the status. Entry ids contain
slashes, so a draw is named in the query, never in the path.

| Request | Answers |
|---|---|
| `GET /api/health` | `{"ok": true}` |
| `GET /api/dataset` | `dataset.json`'s name, ids and splits; counts by split, class and status (`~` stands for no class); the sensor layout, populations, frame step and units of the first readable bundle; the state of the stage files; `index_seconds` |
| `GET /api/draws` | a page of the draw list: `total`, `offset`, `limit`, `items` |
| `GET /api/draws/random` | one draw picked from the filtered ones; `pick_seed` makes the pick repeatable |
| `GET /api/draw?entry=E` | the draw's row, its manifest `record` (the declaration), its `seeds` as strings, and its bundle: `dt_ms`, `n_frames`, the sensor coordinates, the populations and units. A bundle that cannot be read gives `bundle: null` and an `error` |
| `GET /api/draw/trace?entry=E&source=S` | series of one draw (below) |
| `GET /api/draw/frames?entry=E` | a run of stimulus frames as raw bytes (below) |
| `GET /` and `GET /static/NAME` | the page and its files (`index.html`, `app.js`, `style.css`) |
| `GET /api/stages` | per stage: present or not, format, `created_at`, entries, size |
| `GET /api/stage/NAME` | the stage's JSON file as stored (`404` when missing) |
| `POST /api/stages/NAME/run` | start the stage in a background thread (`202` and a job); `compare` needs the other two first (`409`) |
| `GET /api/jobs/ID` | the job's `state` (`running`, `done`, `failed`) and `error` |

**Draw list filters** (all optional): `split`, `class` and `status` take comma-separated
values (`class=dots,~`; status `unknown` is an entry the index has no row for); `q` is a
substring of the entry id (case-insensitive); `seed` is an exact decimal string; `repeat` an
integer. `sort` is `entry` or `seconds` (manifest order by default), `order` is `asc` or
`desc`, `limit` is 1 to 1000 (default 100).

**Traces.** `source` is `stimulus` (the frame value at the sensors in `index`, of `channel`),
or `drive`, `filtered` or `readout` (the population's spikes, events or state) at the units in
`index`, for the `population`. At most 64 indices; `max_points` (default 2000, at most 20000)
caps each series. A longer series is **decimated**: it is cut into buckets and each bucket
keeps the frame of its smallest and of its largest value, so a one-frame spike survives, and
each series carries its own `time_ms`. The answer is `{entry, source, population, readout,
dt_ms, n_frames, decimated, series: [{index, time_ms, values}], units}`.

**Frames.** `start` (default 0), `count` (default 32) and `channel` (default 0). The body is
`count` frames of that channel, `[count, H, W]` row-major, little-endian float32 (`H*W*4`
bytes a frame), exactly the values stored in `data.h5`. A chunk is shorter at the end of the
draw and is capped at 8 MiB, so ask for the next `start` from `X-Frame-Start + X-Frame-Count`.
The shape and range travel in headers: `X-Frame-Start`, `-Count`, `-Height`, `-Width`,
`-Total` (frames in the draw), `-Dt-Ms`, `-Min`, `-Max` (of this chunk, so a player can scale
its colours without a second request) and `X-Frame-Dtype` (`float32-le`). No video encoder is
needed: the browser decodes the bytes into a `Float32Array` and paints them on a canvas
(decision D-7e30f2b). Nothing is cached server-side; a chunk is one HDF5 slice read.

**The page.** `/` is a shell with the draw list (filters, paging, a random draw) and a canvas
that plays the opened draw frame by frame, fetching the next chunk while it shows the current
one. It reports the time from the click to the first painted frame. A6 adds the draw picker
by seed, pinned traces, the declared-fields panel and two-draw comparison to this shell. The
files are plain HTML, CSS and JavaScript with no build step and no external requests.

**Computing a stage.** A stage that is missing can be started from the page. `audit` runs the
`core` pack unless `?packs=core,tactile` says otherwise; `declared` picks the adapter that
accepts the data set; `compare` reads the other two. They write into the stages folder and
run one at a time per stage. A failure is the job's `error`, with the exception's type.

## Time to first frame

`python benchmarks/eval_server_first_frame.py` builds a 10,000-draw data set from the toy
fixture (each bundle's files are hard links to a real one's, so the manifest, index and
folders have a real data set's size), serves it over real HTTP and times the first frame
chunk of 200 random draws, then times the HDF5 read of 32 frames from a stimulus the size of a
tactile run (5000 frames of 80x80, gzip level 4 as the bundle writer stores it).

| What | Median | 95th percentile |
|---|---|---|
| index of 10,000 draws (toy manifest) | 0.11 s | |
| first chunk of a draw not opened before, over HTTP | 1.3 ms | 1.6 ms |
| the same chunk again | 0.7 ms | 0.8 ms |
| read and encode 32 frames of a 5000x80x80 stimulus | 27 ms | 27 ms |

On an Apple-silicon laptop, one run. The first frame of a draw is well under the one-second
target; the cost that grows with the data is the HDF5 read, which depends on the stored
chunk layout, not on the number of draws. The toy bundles are small, so the real-size read is
measured separately and not through the server. Re-measure on the machine and the data set
that matter before relying on it.

## Where the code is

`sensoryforge/evaluate/server/`: `service.py`, `api.py`, `jobs.py`, `http_server.py`,
`__main__.py` and `static/`. The tests are `tests/unit/test_eval_server.py` (on the toy non-tactile data
set, plus one over a real socket). The audit's reader gained `DatasetReader.bundle_path()`
for the server.
