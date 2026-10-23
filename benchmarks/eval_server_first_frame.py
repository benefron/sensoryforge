"""Time the evaluation server's draw-open path (roadmap A5b).

Builds a many-draw data set from the toy fixture (each draw's bundle files are hard
links to a real one's, so the manifest, the index and the folder are as large as a real
data set's), starts the standard-library server on a free port and times, over real
HTTP, what the page waits for before it paints a draw: the first frame chunk. Then
times the HDF5 read of the first chunk from a stimulus the size of a real tactile run
(``--size`` square, ``--frames`` long, gzip level 4 as ``io/bundle.py`` writes it),
which the toy bundles are too small to show.

    conda run -n sensoryforge python benchmarks/eval_server_first_frame.py --draws 10000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sensoryforge.evaluate.server import Api, DatasetService, make_server  # noqa: E402


def inflate(base: Path, out: Path, draws: int) -> None:
    """A data set of ``draws`` entries; their bundle files are links to ``base``'s."""
    lines = (base / "manifest.jsonl").read_text().splitlines()
    index = {}
    for f in sorted((base / "index").glob("task_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                index[row["entry"]] = row
    (out / "index").mkdir(parents=True)
    shutil.copy(base / "dataset.json", out / "dataset.json")
    manifest, rows = [], []
    for i in range(draws):
        raw = json.loads(lines[i % len(lines)])
        source = raw["entry"]
        entry = f"bulk/{i:05d}"
        raw["entry"] = entry
        manifest.append(json.dumps(raw))
        rows.append(json.dumps({**index[source], "entry": entry, "bundle": entry}))
        folder = out / entry
        folder.mkdir(parents=True)
        for name in ("data.h5", "config.json"):
            os.link(base / index[source]["bundle"] / name, folder / name)
    (out / "manifest.jsonl").write_text("\n".join(manifest) + "\n")
    (out / "index" / "task_0000.jsonl").write_text("\n".join(rows) + "\n")


def percentile(values: list, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarise(name: str, ms: list) -> dict:
    return {
        "what": name,
        "n": len(ms),
        "median_ms": round(statistics.median(ms), 2),
        "p95_ms": round(percentile(ms, 0.95), 2),
        "max_ms": round(max(ms), 2),
    }


def time_server(folder: Path, requests: int) -> list:
    started = time.perf_counter()
    service = DatasetService(folder)
    index_s = time.perf_counter() - started
    server = make_server(Api(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    rng = np.random.default_rng(0)
    entries = [service.rows[i].entry for i in rng.choice(len(service.rows), requests)]
    frames = []
    try:
        t = time.perf_counter()
        service.warm()
        warm_s = time.perf_counter() - t
        for entry in entries:
            t = time.perf_counter()
            with urllib.request.urlopen(
                f"{base}/api/draw/frames?entry={entry}&count=32"
            ) as r:
                r.read()
            frames.append(1000 * (time.perf_counter() - t))
        again = []
        for entry in entries:
            t = time.perf_counter()
            with urllib.request.urlopen(
                f"{base}/api/draw/frames?entry={entry}&count=32"
            ) as r:
                r.read()
            again.append(1000 * (time.perf_counter() - t))
    finally:
        server.shutdown()
        server.server_close()
    print(
        f"index of {len(service.rows)} draws: {index_s:.2f} s; "
        f"reader warm-up {warm_s:.2f} s"
    )
    return [
        summarise("first frame chunk, draw not seen before (HTTP)", frames),
        summarise("same chunk, draw seen before (HTTP)", again),
    ]


def time_big_read(tmp: Path, size: int, n_frames: int, repeats: int) -> dict:
    path = tmp / "big.h5"
    axis = np.arange(size, dtype=np.float32)
    x0 = (size / 2 + (size / 4) * np.sin(np.arange(n_frames) / 400.0)).astype(
        np.float32
    )
    row = np.exp(-((axis[None, :] - x0[:, None]) ** 2) / (2 * 3.0**2))
    col = np.exp(-((axis - size / 2) ** 2) / (2 * 3.0**2)).astype(np.float32)
    frames = (row[:, None, :] * col[None, :, None]).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=frames, compression="gzip", compression_opts=4)
        chunks = f["frames"].chunks
    ms = []
    for _ in range(repeats):
        t = time.perf_counter()
        with h5py.File(path, "r") as f:
            block = f["frames"][0:32]
        block = np.ascontiguousarray(block, dtype="<f4").tobytes()
        ms.append(1000 * (time.perf_counter() - t))
    print(f"[{n_frames},{size},{size}] float32, gzip 4, chunks {chunks}")
    return summarise(
        f"read+encode 32 frames of a {n_frames}x{size}x{size} stimulus", ms
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--frames", type=int, default=5000)
    args = ap.parse_args()
    from tests.fixtures import toy_vision

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        toy = toy_vision.build_toy_dataset(tmp / "toy", n_train=4)
        out = tmp / "bulk"
        inflate(toy.data, out, args.draws)
        rows = time_server(out, args.requests)
        rows.append(time_big_read(tmp, args.size, args.frames, 20))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
