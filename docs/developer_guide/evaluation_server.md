# The local evaluation server

The server lets a person look at a data set in a browser instead of through numbers: the
list of draws, one draw's declaration and the traces of its stimulus and populations, and
the audit and comparison files when they exist. It reads one data-set folder from the local
disk. Draws are read **on demand**, so a large data set never has to be rendered in advance.

This page covers the data and JSON endpoints (roadmap A5a). The frame streaming and the page
itself arrive with A5b.

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

**Computing a stage.** A stage that is missing can be started from the page. `audit` runs the
`core` pack unless `?packs=core,tactile` says otherwise; `declared` picks the adapter that
accepts the data set; `compare` reads the other two. They write into the stages folder and
run one at a time per stage. A failure is the job's `error`, with the exception's type.

## Where the code is

`sensoryforge/evaluate/server/`: `service.py`, `api.py`, `jobs.py`, `http_server.py`,
`__main__.py`. The tests are `tests/unit/test_eval_server.py` (on the toy non-tactile data
set, plus one over a real socket). The audit's reader gained `DatasetReader.bundle_path()`
for the server.
