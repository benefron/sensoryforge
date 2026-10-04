"""Record what SensoryForge v1.1.0 produces for old worlds (the IU guard's reference).

The v1.2.0 world additions are *invisible until used*: a world, data set or
session file that uses no v1.2 key must load to the same normalised dict,
``world_id``, draws, records, manifest rows, session records and frames, bit
for bit, as v1.1.0. This script records those numbers for two worlds
(``tactile_small.yml`` and ``every_element_v1_1.yml``) and the data set
``dataset_small.yml``; ``tests/unit/test_world_v1_1_compat.py`` checks the
working tree against them.

Run it on a clean copy of tag v1.1.0 and on the working tree; the two
``reference.json`` files must be identical (``cmp``)::

    git archive v1.1.0 | tar -x -C $SCRATCH/sf_v1_1_0
    SCRIPT=tests/fixtures/make_world_v1_1_reference.py
    PYTHONPATH=$SCRATCH/sf_v1_1_0 python $SCRIPT --out $SCRATCH/ref_v1_1_0
    PYTHONPATH=$PWD python $SCRIPT --out $SCRATCH/ref_tree
    cmp $SCRATCH/ref_v1_1_0/reference.json $SCRATCH/ref_tree/reference.json

The script lives in the working tree and reads its fixtures from there; the
``sensoryforge`` package it imports is whichever ``PYTHONPATH`` names. The
frames are float64 and their sha256 is exact only on the platform recorded in
``reference.json`` (macOS arm64); elsewhere the tests compare to a tolerance.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from sensoryforge.world import (
    Canvas,
    build_dataset,
    load_dataset,
    load_world,
    render,
    sample,
    session,
)

WORLDS = Path(__file__).resolve().parent / "worlds"
WORLD_FILES = {
    "tactile_small": WORLDS / "tactile_small.yml",
    "every_element_v1_1": WORLDS / "every_element_v1_1.yml",
}
DATASET_FILE = WORLDS / "dataset_small.yml"
SAMPLE_SEED = 7
SAMPLE_N = 50
FRAME_DRAWS = 12
SESSION_MS = 300.0


def rounded(value):
    """Floats to 10 significant digits (as the sampling tests' ``DRAWS_DIGEST``)."""
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def digest(obj) -> str:
    """sha256 of the canonical JSON of ``obj`` (floats rounded)."""
    text = json.dumps(rounded(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def frame_times() -> torch.Tensor:
    """24 times, 0 to 115 ms in steps of 5, float64."""
    return torch.arange(0.0, 120.0, 5.0, dtype=torch.float64)


def frames_of(world) -> torch.Tensor:
    """``[12, 24, 12, 12]`` float64 frames of 12 draws on a 12x12 canvas at 0.1 mm."""
    draws = sample(world, n=FRAME_DRAWS, seed=SAMPLE_SEED)
    canvas = Canvas.from_grid(12, 12, 0.1)
    return render(draws, canvas, frame_times(), dtype=torch.float64)


def record() -> tuple:
    """``(reference dict, {world name: frames})``."""
    reference = {
        "platform": {"machine": platform.machine(), "sys_platform": sys.platform},
        "worlds": {},
    }
    frames = {}
    for name, path in WORLD_FILES.items():
        world = load_world(path)
        draws = sample(world, n=SAMPLE_N, seed=SAMPLE_SEED)
        movie = frames_of(world)
        frames[name] = movie
        reference["worlds"][name] = {
            "world_id": world.world_id,
            "normal_form_sha256": hashlib.sha256(
                json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "draws_digest": digest([d.to_dict() for d in draws]),
            "frames_sha256": hashlib.sha256(movie.numpy().tobytes()).hexdigest(),
        }
    small = load_world(WORLD_FILES["tactile_small"])
    reference["session"] = {
        "world": "tactile_small",
        "duration_ms": SESSION_MS,
        "seed": SAMPLE_SEED,
        "index": 0,
        "digest": digest(session(small, SESSION_MS, SAMPLE_SEED, 0).to_dict()),
    }
    spec = load_dataset(DATASET_FILE)
    reference["dataset"] = {
        "dataset_id": spec.dataset_id,
        "manifest_digest": digest([e.to_dict() for e in build_dataset(spec)]),
    }
    return reference, frames


def main() -> None:
    """Write ``reference.json`` and ``frames.pt`` into ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="Directory to write into")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference, frames = record()
    (out / "reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n"
    )
    torch.save(frames, out / "frames.pt")
    import sensoryforge

    print(f"sensoryforge from {sensoryforge.__file__}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
