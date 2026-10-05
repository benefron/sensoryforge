"""Record what SensoryForge v1.2.1 simulates for a small event-converter design.

This is the converter contract's test 1 golden.

v1.3.0 gives the level-crossing unit an optional leaky reference. With the
leak off, the unit, the sigma-delta unit and everything before them must run
as v1.2.1 did, bit for bit. This script runs ``sensoryforge run --design`` on
``converter_v1_2_1/design/`` (one ``level_crossing`` RA and one ``sigma_delta``
SA population over pressure-simulation's 8x8 fixture fields, with receptor
noise and a fixed seed) and records each population's ``drive``, ``filtered``
and ``events``/``spikes`` from the bundle's ``data.h5``;
``tests/contract/test_converter_contract.py::test_1_leak_off_is_v1_2_1_bit_for_bit_engine``
checks the working tree against them.

Run it on an export of tag v1.2.1 (never ``pip install -e .`` from a
worktree, F-053) and, before any change, on the working tree; the two
``golden.json`` files must be identical (``cmp``)::

    git archive v1.2.1 | tar -x -C $SCRATCH/sf_v1_2_1
    SCRIPT=tests/fixtures/make_converter_v1_2_1_golden.py
    PYTHONPATH=$SCRATCH/sf_v1_2_1 python $SCRIPT --out $SCRATCH/golden_v1_2_1
    PYTHONPATH=$PWD python $SCRIPT --out $SCRATCH/golden_tree
    cmp $SCRATCH/golden_v1_2_1/golden.json $SCRATCH/golden_tree/golden.json

The script lives in the working tree and reads its design from there; the
``sensoryforge`` package it imports is whichever ``PYTHONPATH`` names. The
arrays' sha256 are exact only on the platform recorded in ``golden.json``
(macOS arm64 and the recorded torch version); elsewhere the test compares to
SensoryForge's golden tolerance (F-071).
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from sensoryforge.cli import cmd_run, create_parser

DESIGN = Path(__file__).resolve().parent / "converter_v1_2_1" / "design"
STIMULUS = "ramp_gaussian"
DURATION_MS = 80


def record(out: Path) -> None:
    """Run the design through the CLI and write ``golden.npz`` and ``golden.json``.

    Args:
        out: Output directory (created; ``golden.npz`` and ``golden.json``
            are overwritten).
    """
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        args = create_parser().parse_args(
            [
                "run",
                "--design",
                str(DESIGN),
                "--stimulus",
                STIMULUS,
                "--duration",
                str(DURATION_MS),
                "--bundle",
                str(bundle),
            ]
        )
        if cmd_run(args) != 0:
            raise SystemExit("sensoryforge run --design failed")
        arrays = {}
        with h5py.File(bundle / "data.h5", "r") as f:
            for name, group in f["populations"].items():
                for key in group:
                    arrays[f"{name}__{key}"] = group[key][()]
        shutil.rmtree(bundle)
    np.savez(out / "golden.npz", **arrays)
    summary = {
        "what": "sensoryforge run --design converter_v1_2_1/design at v1.2.1",
        "stimulus": STIMULUS,
        "duration_ms": DURATION_MS,
        "platform": {
            "sys_platform": sys.platform,
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "abs_sum": float(np.abs(value.astype(np.float64)).sum()),
            }
            for key, value in sorted(arrays.items())
        },
    }
    (out / "golden.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Parse ``--out`` and record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    record(parser.parse_args().out)


if __name__ == "__main__":
    main()
