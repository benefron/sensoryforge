"""Run a data set through the sensor: one bundle per entry (spec §7).

One process builds the engine once (receptive fields loaded once) and loops
over its entries. Each entry is rendered in float64 on the CPU on the design's
canvas (whatever the engine's device, so a bundle's frames never depend on
it), cast to float32, moved to the engine's device, simulated with its own
noise seeds, and written to ``<out>/.partial/<entry>/`` before an atomic
rename to ``<out>/<entry>/``.
Each task appends to its own ``index/task_<i>.jsonl``, so array tasks never
share a file.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import WORLD_ENTRY_KIND
from sensoryforge.provenance import source_info
from sensoryforge.world import rng
from sensoryforge.world.dataset import DatasetSpec, Entry, build_dataset
from sensoryforge.world.render import Canvas, render_movie
from sensoryforge.world.sampling import Session
from sensoryforge.world.schema import World

#: What one failing entry may raise without stopping the run.
_ENTRY_ERRORS = (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError)


def task_slice(n: int, tasks: int, task_index: int) -> range:
    """The contiguous slice of ``n`` entries that task ``task_index`` runs."""
    if tasks < 1:
        raise ValueError(f"--tasks must be >= 1, got {tasks}")
    if not 0 <= task_index < tasks:
        raise ValueError(f"--task-index must be in [0, {tasks}), got {task_index}")
    return range(task_index * n // tasks, (task_index + 1) * n // tasks)


def select_entries(
    entries: Sequence[Entry], splits: Optional[Sequence[str]] = None
) -> List[Entry]:
    """The entries of the named splits (all when ``splits`` is empty)."""
    if not splits:
        return list(entries)
    known = {e.split for e in entries}
    unknown = sorted(set(splits) - known)
    if unknown:
        raise ValueError(
            f"--splits {unknown} are not in this data set (it has {sorted(known)})"
        )
    keep = set(splits)
    return [e for e in entries if e.split in keep]


def channel_layout(world: World, grid: GridConfig) -> Optional[List[int]]:
    """Where each world channel goes among the grid's channels (``None``: plain)."""
    grid_channels = list(grid.channels or ["value"])
    if len(grid_channels) == 1:
        if len(world.channels) > 1:
            raise ValueError(
                f"the world has channels {world.channels} but grid "
                f"{grid.name!r} has one plane"
            )
        return None
    missing = [c for c in world.channels if c not in grid_channels]
    if missing:
        raise ValueError(
            f"world channels {missing} are not among grid {grid.name!r}'s "
            f"channels {grid_channels}"
        )
    return [grid_channels.index(c) for c in world.channels]


def to_grid_frames(
    movie: torch.Tensor, layout: Optional[List[int]], n_grid_channels: int
) -> torch.Tensor:
    """``[T, H, W]`` as is, or the planes placed in a ``[T, C_grid, H, W]`` stack."""
    if layout is None:
        return movie
    if movie.ndim == 3:
        movie = movie.unsqueeze(1)
    out = torch.zeros(
        (movie.shape[0], n_grid_channels) + tuple(movie.shape[2:]),
        dtype=movie.dtype,
        device=movie.device,
    )
    for c, target in enumerate(layout):
        out[:, target] = movie[:, c]
    return out


def stimulus_payload(entry: Entry) -> Dict[str, Any]:
    """The ``stimulus_config`` that marks a bundle as this world entry (spec §7.3)."""
    item = entry.item
    if isinstance(item, Session):
        layer: Any = [[start, draw.to_layer()] for start, draw in item.items]
    else:
        layer = item.to_layer()
    return {"kind": WORLD_ENTRY_KIND, "entry": entry.to_dict(), "layer": layer}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def run_dataset(
    config: SensoryForgeConfig,
    spec: DatasetSpec,
    out_dir: Union[str, Path],
    *,
    design_manifest: Optional[Dict[str, Any]] = None,
    splits: Optional[Sequence[str]] = None,
    tasks: int = 1,
    task_index: int = 0,
    entry_range: Optional[slice] = None,
    resume: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """Simulate a data set's entries, one bundle each (spec §7.2).

    Args:
        config: The sensor (grids and populations); mutated per entry
            (noise seeds), so pass one that only this run uses.
        spec: The data set.
        out_dir: Output directory (created).
        design_manifest: A design directory's manifest, stamped into every bundle.
        splits: Only these splits.
        tasks: Split the (selected) entries into this many contiguous tasks...
        task_index: ...and run this one.
        entry_range: Run this slice of the selected entries instead of a task.
        resume: Skip entries whose bundle already holds ``data.h5``.
        log: Called with one line per entry.

    Returns:
        ``{"ok": int, "failed": int, "skipped": int}``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = select_entries(build_dataset(spec), splits)
    if entry_range is not None:
        chosen = entries[entry_range]
    else:
        chosen = [entries[i] for i in task_slice(len(entries), tasks, task_index)]
    provenance = source_info()
    _write_json_atomic(
        out / "batch.json",
        {
            "format": "sensoryforge-batch/1",
            "dataset_id": spec.dataset_id,
            "dataset_name": spec.name,
            "world_id": spec.world.world_id,
            "n_entries": len(entries),
            "splits": sorted({e.split for e in entries}),
            "design": design_manifest,
            "config": config.to_dict(),
            "sensoryforge": provenance,
        },
    )
    grid = config.grids[0]
    layout = channel_layout(spec.world, grid)
    n_grid_channels = len(grid.channels or ["value"])
    engine = SimulationEngine(config)
    canvas = Canvas.from_grid_config(grid)
    index_path = out / "index" / f"task_{task_index:04d}.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    design_id = (design_manifest or {}).get("design_id")
    summary = {"ok": 0, "failed": 0, "skipped": 0}

    for entry in chosen:
        final = out / entry.entry
        if resume and (final / "data.h5").exists():
            summary["skipped"] += 1
            continue
        started = time.perf_counter()
        row: Dict[str, Any] = {
            "entry": entry.entry,
            "bundle": entry.entry,
            "task": task_index,
            "design_id": design_id,
            "sensoryforge_sha": provenance["sha"],
        }
        try:
            noise = int(entry.seeds["noise"])
            config.simulation.receptor_noise_seed = noise
            # Every population gets its own 53-bit seed, set or not in the
            # design: the engine uses it only when that population has noise,
            # and no noise is then left to the 32-bit run seed below.
            for i, pop in enumerate(config.populations):
                pop.noise_seed = rng.seed53(noise, "population", i)
            # Always the CPU: CUDA's float64 exp/sin/cos differ from the CPU's
            # in the last bits, and a bundle must equal the CPU render.
            movie = render_movie(
                entry.item,
                canvas,
                config.simulation.dt_ms,
                entry.duration_ms,
                dtype=torch.float64,
                device="cpu",
            )
            frames = to_grid_frames(movie, layout, n_grid_channels)
            frames = frames.to(dtype=torch.float32).to(device=engine.device)
            partial = out / ".partial" / entry.entry
            if partial.exists():
                shutil.rmtree(partial)
            partial.parent.mkdir(parents=True, exist_ok=True)
            engine.run(
                frames.unsqueeze(0),
                bundle_dir=partial,
                stimulus_config=stimulus_payload(entry),
                # numpy seeds only take 32 bits; the 53-bit noise seed itself
                # reaches the receptor and population generators via the config.
                seed=noise & 0xFFFFFFFF,
                bundle_overwrite=True,
                design_manifest=design_manifest,
            )
            if final.exists():
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, final)
            row.update(status="ok", error=None)
            summary["ok"] += 1
        except _ENTRY_ERRORS as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            summary["failed"] += 1
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        with open(index_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if log is not None:
            log(f"{row['status']:>6}  {entry.entry}  ({row['seconds']:.2f} s)")
    return summary


def read_batch_index(out_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every task's index rows merged: one row per entry, the latest run winning.

    Rows keep the order in which entries first appear (task files in task
    order), which is the data set's order for a complete run.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for path in sorted((Path(out_dir) / "index").glob("task_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            old = latest.get(row["entry"])
            if old is None or row["finished_at"] >= old["finished_at"]:
                latest[row["entry"]] = row
    return list(latest.values())
