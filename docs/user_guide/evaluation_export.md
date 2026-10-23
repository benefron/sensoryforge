# Sharing a snapshot of a data set

`sensoryforge evaluate export` writes a folder you can zip and send to a collaborator. They
open `index.html` in a browser, from disk, with nothing installed and no server running. It
holds the draw list, the dashboard, the anomaly gallery, the checks and a few draws you
choose, shown by the same pages as [the local server](../developer_guide/evaluation_server.md).

## Write one

```bash
sensoryforge evaluate export runs/first runs/first_snapshot \
    --draws train/r0/00000,test/r1/00004 --video gif
```

`runs/first` is the data-set folder (the one `sensoryforge evaluate run` wrote; its stage files
in `runs/first/evaluate/` are used, or name another folder with `--stages`). The second folder
must be new or empty. Open `runs/first_snapshot/index.html`.

| option | meaning |
|---|---|
| `--draws a,b` | entry ids to include in full. Without it, `--n-draws` (default 3) draws spread evenly over the manifest are taken |
| `--sensor K` | a sensor whose stimulus trace is recorded for every chosen draw (repeat for more). Default: the middle sensor |
| `--unit POPULATION:INDEX` | a unit whose drive, filtered current, readout and re-simulated voltage are recorded (repeat for more). Default: each population's middle unit |
| `--no-pins` | record no single-point traces |
| `--max-list-rows N` | rows kept of any draw list (default 1000) |
| `--max-mb N` | refuse to record more than this many megabytes of answers: frames, traces, voltages and lists (default 500) |
| `--video mp4\|gif`, `--video-speed S` | also write a movie of each chosen draw into `videos/`, if ffmpeg is installed (speed 1 is real time) |

The middle sensor and unit are arbitrary, fixed choices so that a pinned trace shows something
when the page opens; name the points you want to show with `--sensor` and `--unit`.

## What is in the folder

```
index.html  dashboard.html  gallery.html  checks.html     the server's pages
static/                                                    their scripts and style
snapshot/index.js                                          answers for every page and draw list
snapshot/draw_0.js ...                                     one chosen draw: frames, record, traces
videos/                                                    only with --video
```

A page asks for data exactly as it does against the server; a small script (`static/snapshot_shim.js`)
answers each request from the recording. The recording is made by asking the server code itself,
so a snapshot answer is the answer the server gave, errors included. A chosen draw's file is read
only when that draw is opened.

## What a snapshot cannot do

It is read-only and finite. Anything it holds nothing for answers `not in this snapshot`, with
the list of draws it does hold:

- **Draws not chosen** show their name in lists but do not open.
- **Pins:** only the recorded points have traces. Clicking another sensor or unit shows the message under the trace.
- **Filters:** the split and class selects, and every bar, count and cell of the dashboard, work.
  Combining two filters, a free-text search, a seed search and *Random draw* do not.
- **Gallery:** it shows all signals; unticking one asks for a list the snapshot does not hold.
- **Runs:** the buttons that compute a stage (the checks page) answer that the snapshot is read-only.

If a collaborator needs more, send the data-set folder and let them run `sensoryforge evaluate serve`.

## Size

Frames are stored as 32-bit floats, base64-coded, so a draw costs about 1.4 times its frames:
`frames * height * width * channels * 4` bytes. The export stops with a message when the
recording (every answer it keeps, not only the frames) passes `--max-mb`. Choose fewer or
shorter draws, or raise the limit.

## Local paths

A snapshot names the data-set folder and the stage folder by their names only. Every recorded
answer has their local paths (as given and resolved) replaced with the folder's name, so
`/Users/you/runs/tiny` is shared as `tiny`, in the page's summary and in the stage files
(`checks.json`'s `inputs.dataset_dir`, for example).

## Videos

A movie uses the draw browser's colours, scale and orientation (x to the right, y upwards) and is
written only when `ffmpeg` is on your path; without it the export finishes and says so. MP4 uses
H.264; GIF is larger but plays anywhere. Frames are enlarged with nearest-neighbour so a small
sensor stays crisp.

## From Python

```python
from sensoryforge.evaluate.export import ExportOptions, export_snapshot
from sensoryforge.evaluate.server import DatasetService

service = DatasetService("runs/first")
result = export_snapshot(service, "runs/first_snapshot", ExportOptions(draws=["train/r0/00000"]))
print(result.pages, result.draws, result.notes)
```
