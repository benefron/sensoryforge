# The local evaluation server

The server lets a person look at a data set in a browser instead of through numbers: the
list of draws, one draw's declaration and the traces of its stimulus and populations, and
the audit and comparison files when they exist. It reads one data-set folder from the local
disk. Draws are read **on demand**, so a large data set never has to be rendered in advance.

This page covers the data and JSON endpoints (roadmap A5a), the frame streaming and the page
shell (A5b), the draw picker and movie (A6a), and pinned points with linked traces and
re-simulated membrane voltage (A6b), and the declared-fields panel and two-draw comparison
(A6c).

## Run it

```bash
sensoryforge evaluate serve DATASET_DIR            # http://127.0.0.1:8765/
python -m sensoryforge.evaluate.server DATASET_DIR --port 9000 --stages out/evaluate
```

`--stages` is the folder of `audit.json`, `declared.json` and `compare.json` (default
`DATASET_DIR/evaluate`, where `python -m sensoryforge.evaluate.audit` writes by default). The
server binds to the loopback address and has no authentication: it reads local files and
offers to run the stages over them, so do not bind it to a public address.

`--frame-cache-mb` (default 256, 0 keeps nothing) sets how much decompressed stimulus the
server keeps between requests; see [Playback and the slab cache](#playback-and-the-slab-cache).
`--voltage-cache-mb` (default 256) does the same for re-simulated membrane voltage.

**Other web pages cannot use it.** A loopback bind does not keep out a page open in the same
browser, which could start a stage with a cross-site `POST`, or read the data set after a DNS
rebinding. The HTTP adapter therefore answers `403` to a request whose `Host` is not
`<bound name>:<port>`, `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`, and to a
`POST` whose `Origin` is present and is not the server's own `http://<Host>`. A `GET` from the
page itself carries no `Origin`, and its `POST` carries the same origin, so the page is not
affected. If you bind another address on purpose, its name joins the allowed `Host` values.
That does not make the server safe to expose: there is still no authentication. A different
HTTP backend would have to repeat both checks.

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
| `GET /api/draw?entry=E` | the draw's row, its manifest `record` (the declaration), its `seeds` as strings, `frame_peak` (the audit's `core.frames.peak` for the draw, read from `audit_entries.jsonl`, or `null`), and its bundle: `dt_ms`, `n_frames`, the sensor coordinates, the populations and units. A bundle that cannot be read gives `bundle: null` and an `error` |
| `GET /api/draw/trace?entry=E&source=S` | series of one draw (below) |
| `GET /api/draw/voltage?entry=E&population=P&index=N` | membrane voltage of neurons, re-simulated, with its comparison against the stored run (below) |
| `GET /api/draw/population?entry=E&population=P[&index=N]` | neuron positions of a population and, with `index`, the receptors one neuron reads (below) |
| `GET /api/draw/frames?entry=E` | a run of stimulus frames as raw bytes (below) |
| `GET /` and `GET /static/NAME` | the page and its files (`index.html`, `app.js`, `geometry.js`, `traces.js`, `style.css`) |
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
(decision D-7e30f2b). A chunk is copied out of a cached time slab, see below.

**The page.** `/` has the draw list and a movie of the opened draw's stimulus. The files are
plain HTML, CSS and JavaScript with no build step and no external requests.

*Picking a draw.* The list filters by split, class, entry id and the draw's seed (exact). The
random button picks among the filtered draws; the status line shows the pick seed it used,
and typing that seed into "Pick seed" repeats the pick. An opened draw is kept in the URL
(`#entry=<id>`), so a link from another page, or a reload, opens the same draw.

*Orientation.* A stored frame is `[x, y]` (axis 0 is x, `indexing="ij"`, sensor
`k = i * cols + j`). The canvas paints x to the right and y upwards, and the axes are
labelled from `/api/draw`'s layout bounds and space unit. `static/geometry.js` holds the
mapping as pure functions (`pixelOf`, `sensorAt` for A6b's clicks, `toRGBA`);
`tests/unit/test_eval_page_geometry.py` runs them with node (skipped when node is not
installed) and puts a value at sensors with known coordinates.

*Colour scale.* When the audit has run, the scale runs from `min(0, running minimum)` to the
draw's `core.frames.peak`, so brightness does not change as chunks arrive. Without the audit's
entries file it is the running range of the chunks fetched so far. A line under the movie shows
the range and which of the two is in use.

*Playback.* The movie plays by wall clock with `requestAnimationFrame` and a speed control
(0.05× to 20×): the cursor advances by the elapsed time at the chosen speed, so frames the
display cannot show are skipped, and 1 ms frames play in real time at 1×. If the cursor
reaches a frame that has not arrived, it waits and "buffering" shows; the waiting is not
counted. There is one promise per chunk start, so a chunk is never fetched twice, at most four
chunks are in flight, and the browser keeps at most 128 MB of decoded chunks (least recently
used out). A failed fetch stops playback and shows the error. Checked in a browser on the
80x80 data set (30 s session, 1 ms frames): 0.1× and 1× land on the expected frame to within
a frame; at 20× the cursor stalls on fetches (it needs about 600 chunks a second), which is
the cost of asking for every chunk and not only the ones painted.

*Pinning and traces (A6b).* "Pin by" chooses what a click on the movie pins: a **sensor**
(the pixel under the pointer, `G.sensorAt`) or, once a population is chosen, the **neuron** whose
centre is nearest (centres come from `/api/draw/population`; the layout's bounds map the click
to a position, `G.pixelToMm`). A red ring marks the pin; a pinned neuron's receptive field is
drawn as translucent squares on the sensors it reads (brighter is a larger weight), and every
neuron's centre is a small dot. Under the movie, one stacked trace per quantity spans the
whole draw with **one shared time cursor** that follows the movie; clicking a trace moves the
movie to that time. A sensor pin shows the stimulus at that sensor (one request); a neuron pin
shows its drive, filtered current, readout (ticks for spikes and events, a line for a state)
and membrane voltage. Each trace is one request and is decimated as above. The pin survives
opening another draw (a neuron pin if the population and its neuron exist there too).
*Accumulated current* is not shown: a bundle does not store it and the neuron stage does not
produce one. `static/traces.js` holds the pure plotting maths (node-tested, like
`geometry.js`).

**Membrane voltage is re-simulated** (decided by Ben 2026-10-23, commit 82034fc). A bundle
stores a population's `filtered` (after gain and sensor noise, before the input floor) and its
readout, not the voltage. `GET /api/draw/voltage` runs the population's neuron again from the
stored `filtered` with `SimulationEngine._run_neuron_stage`, the same code a run uses (the
engine's `_run_pop_from_drive` calls it too, and a test pins that the two give identical spikes
and voltage, noisy neurons included). The neuron is built from `config.json` (its parameters,
`dt_ms`, `integrate_dt_ms`, and the floor the run applied, `encoder.input_floor_ma`); a
population with membrane noise and a `noise_seed` gets that seed's generator. **Verification
happens every time:** the *whole* population is re-run (a population's noise is drawn for all
neurons at once, and the check covers every neuron, not only the pinned one) and its
spikes, events or state are compared with the stored ones. The answer carries `verification`:
`matches`, `n_differing` of `n_units`, the first differing ids, the largest absolute
difference, the bundle's `sensoryforge_sha` and the running code's (the page marks different
shas), the seconds the re-run took, and notes (for example a membrane-noise population without
a `noise_seed` cannot be reproduced). The page shows "matches the stored run" in green or
"differs" in red under the voltage trace. The voltage is bin-end samples, like the engine's
`voltages`. A re-run costs the neuron integration for the whole draw (about 0.1 s for the toy
data set; seconds for 30 s of 1 ms frames and a thousand neurons), runs on the CPU, and is
cached by draw and population (`--voltage-cache-mb`, default 256; a re-run bigger than the
cache is used and not kept), so a second neuron of the same population is instant. A model that
returns no voltage (an analog readout) gets a `422` saying so. A match proves the re-run follows
the run only where the stored readout can tell: a neuron that never crosses threshold has no
spikes to compare. Stored voltage probes (A6p) close that gap: when the bundle holds
`/populations/<name>/voltage_probe` (the opt-in `voltage_probes` key, schema 2.4.0; see
[Data bundles](../user_guide/bundles.md#voltage-probes-schema-240)), the same answer carries
`verification.probes` (`ids`, `matches`, `n_differing`, `max_abs_difference`): the re-run's
voltage at those neurons is compared with the stored voltage sample by sample, float32 exactly,
and the page adds "voltage matches the N stored voltage probes (neurons …)" in green or
"voltage differs from … (largest difference … mV)" in red. A bundle without probes says
nothing about them.

*Declared fields and comparison (A6c).* Under the traces, "Declared fields" lists the open draw's
manifest record (what the world declared for it), flattened to one row per field in sections:
Entry (id, split, class, repeat, world and data-set ids), Draw (the draw record: seed, timeline,
sampling), Values drawn (the fields drawn from the world's axes) and Seeds. Seeds are shown from
the decimal strings of `/api/draw`'s `seeds`, because JSON numbers above 2^53 lose digits in the
browser; any other integer that large is marked "(rounded)". The server sends nothing new for
this: the panel reads `record`. This panel shows the declaration, so it belongs to the
inspector and not to the blind audit, which never reads it.

"Compare with…" then a click in the draw list opens a second draw **B** beside the open one
**A** (`#entry=A&vs=B` in the URL reproduces the pair). B's movie follows A's clock (B's frame is
the one at the same time, `G.matchFrame`; a shorter draw holds its last frame; each movie
waits for the other's chunks) and both are painted with **one colour scale**, the widest of the
two, so equal colours mean equal values; the scale note says so. A pin on A marks the same
sensor or neuron on B and B's traces are laid over A's in orange (a ticks row puts A's ticks in
its upper half and B's in the lower half); B's membrane voltage is re-simulated and verified like
A's, and the note under the trace gives both verdicts. The overlay is shown only when the pin
means the same in B (same number of sensors; the population exists with that many neurons,
`R.pinProblem`); otherwise the page says why and shows A alone. The fields panel becomes three
columns, A and B, with rows that differ shaded and a switch to show only those. Each movie has
its own chunk store (`newStore`) with the same one-promise-per-chunk rule. Not done: the
multi-channel image renderer (the movie shows channel 0 as a heat map) and "two runs" (a second
data-set folder): B is another draw of the same data set. `static/record.js` holds the panel's
pure functions (node-tested like the others).

**Population geometry.** `GET /api/draw/population` reads the population's tensor file in the
bundle (the receptive-field bank: `neuron_centers` and `innervation_weights`) with torch,
imported when first asked. `centers` is `[[x, y]]` in the layout's space unit; with `index`,
`field` is `{index, inputs: [{sensors, weights}]}`, one block per input of the population
(a population reading three channels has three), or `null` when the bank's receptors are not a
whole number of copies of the layout's sensors.

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

**On real-size bundles, through the server** (review R-A5, 2026-10-23, and the re-run in row
A5f: an 80x80 tactile data set at 1 ms, timed over HTTP with
`docs_root/plan/scripts/r_a5_build.py` and `r_a5_time.py`). Opening a draw stays well under a
second; the cost that grew was playback, which the slab cache removes.

| Request (median unless noted) | 5 s entry, before | after | 30 s session, before | after |
|---|---|---|---|---|
| `/api/draw` | 5 ms | 5 ms | 16 ms | 14 ms |
| first chunk, 32 frames | 15–17 ms | 15–17 ms | 50 ms | 53 ms |
| each later chunk in playback order | 12–15 ms (max 45) | **1.1 ms** (max 16) | 46 ms (max 88) | **1.2 ms** (max 48) |
| seek to the middle | 21–29 ms | 23–30 ms | 88 ms | 96 ms |

The maxima after are the chunks that start a new slab (one decompression for each 313 frames
at 5 s and each 938 at 30 s). A 120 s session's layout (3750-frame slabs, 96 MB each) costs
185 ms for a first chunk, read with h5py, and 219 ms for a whole slab, then about a
millisecond for the rest of its 3750 frames; two such slabs fit the default cache.

## Playback and the slab cache

HDF5 stores `stimulus/frames` in gzip chunks whose time extent grows with the draw (313
frames at 5 s, 938 at 30 s, 3750 at 120 s). Reading 32 frames decompresses every chunk it
touches, so before A5f each later chunk of a playback cost as much as the first. Now
`DatasetService.frames()` reads the frames in **time slabs aligned to the stored chunk**
(`ds.chunks[0]` frames) and keeps them in a byte-bounded least-recently-used cache keyed
`(entry, channel, first frame of the slab)`. A run that crosses a slab boundary is copied from
both slabs and equals the stored frames bit for bit. Two requests for the same slab
decompress it once. A slab larger than the whole cache is read and used without being kept,
and a dataset that is not chunked is read for the run alone, because a contiguous read
decompresses nothing. `DatasetService.cache_stats()` returns the hit and miss counters. The
wire format, the 32-frame default and the 8 MiB cap are unchanged, so the page did not change.
The bundle writer's chunk shape was not changed, because that would alter the h5 bytes of every
bundle pressure-simulation pins.

An h5py error while reading a draw (a damaged chunk, a missing dataset) comes back as a `422`
naming the draw, like an unreadable bundle at the first peek. A stage job always leaves
`running`: an unexpected exception ends it `failed` with `unexpected <Type>: <message>`, and
the traceback stays in the server's log. The dataset summary remembers its reference draw once
one reads, and while none reads it stops after 50 failed peeks and says so
(`reference_note`).

## Where the code is

`sensoryforge/evaluate/server/`: `service.py`, `api.py`, `jobs.py`, `http_server.py`,
`resim.py` (the re-run), `__main__.py` and `static/` (`index.html`, `app.js`, `geometry.js`,
`traces.js`, `record.js`, `style.css`). The tests are `tests/unit/test_eval_server.py` (on the toy non-tactile
data set, plus some over a real socket), `tests/unit/test_eval_voltage.py` and
`tests/unit/test_eval_page_geometry.py`, `tests/unit/test_eval_page_compare.py`. The audit's reader gained `DatasetReader.bundle_path()`
for the server.
