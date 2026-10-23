"""One command from a world to a report folder (roadmap A11).

``run_pipeline`` chains the stages the report contract defines: it builds the data set's
manifest, simulates one bundle per entry (``sensoryforge batch --dataset``), runs the
audit, the declared adapter, the comparison and (by default) the reference-free checks,
and writes ``pipeline.json`` beside their files in ``OUT/evaluate``. It adds no
measurement of its own. The audit runs as its own ``python -m`` process so that it
cannot see the world package even by accident (the blind rule, contract section 2). This
module is outside the audit's import wall.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
from sensoryforge.evaluate.checks.run import run_checks
from sensoryforge.evaluate.compare.run import run_compare
from sensoryforge.evaluate.declared.run import run_declared
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY
from sensoryforge.evaluate.stage_io import (
    file_sha256,
    producer,
    timestamp,
    write_json,
)
from sensoryforge.world.dataset import (
    DatasetSpec,
    build_dataset,
    write_dataset,
)
from sensoryforge.world.runner import run_dataset

FORMAT = "sensoryforge-pipeline/1"
PIPELINE_FILE = "pipeline.json"
REPORT_DIR = "evaluate"
STAGES = ("build", "batch", "audit", "declared", "compare", "checks")

#: Stage errors that end the pipeline with a message instead of a traceback.
_STAGE_ERRORS = (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError)


class _StageFailed(Exception):
    """A stage did not produce its files; the message is the reason."""


def dataset_mapping(
    world_path: Path, draws: int, seed: int, duration_ms: float
) -> Dict[str, Any]:
    """The data-set mapping that stands for "N draws from this world".

    Args:
        world_path: A world file (``world:`` YAML).
        draws: Number of draws, all in one ``train`` split.
        seed: The data set's seed.
        duration_ms: Duration of every draw in ms.

    Returns:
        A mapping :func:`sensoryforge.world.dataset.load_dataset` accepts.

    Raises:
        ValueError: If ``draws`` is below 1.
    """
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws}")
    return {
        "dataset": {
            "name": f"{Path(world_path).stem}_{draws}",
            "world": str(Path(world_path).resolve()),
            "seed": int(seed),
            "duration_ms": float(duration_ms),
            "splits": {"train": {"n": int(draws)}},
        }
    }


def _run_audit_process(
    folder: Path, out: Path, packs: Sequence[str]
) -> subprocess.CompletedProcess:
    """Run the audit stage in a fresh interpreter."""
    cmd = [
        sys.executable,
        "-m",
        "sensoryforge.evaluate.audit",
        str(folder),
        "--out",
        str(out),
        "--packs",
        ",".join(packs),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _registered(registry: Any, name: str) -> bool:
    try:
        registry.get_class(name)
    except KeyError:
        return False
    return True


def _timed(
    stages: List[Dict[str, Any]],
    name: str,
    fn: Callable[[], Any],
    log: Callable[[str], None],
) -> Any:
    """Run one stage and append its record to ``stages``.

    A stage error (:data:`_STAGE_ERRORS`) is re-raised as ``_StageFailed``; anything
    else marks the stage failed and propagates as it is.
    """
    log(f"[{name}] …")
    started = time.perf_counter()
    record: Dict[str, Any] = {"name": name, "status": "ok", "detail": None}
    stages.append(record)
    try:
        result = fn()
    except _STAGE_ERRORS as exc:
        record.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
        raise _StageFailed(f"{name}: {exc}") from exc
    except BaseException as exc:  # recorded here, re-raised unchanged
        record.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = round(time.perf_counter() - started, 3)
    log(f"[{name}] done in {record['seconds']:.1f} s")
    return result


def run_pipeline(
    config: SensoryForgeConfig,
    spec: DatasetSpec,
    out_dir: Path,
    *,
    design_manifest: Optional[Dict[str, Any]] = None,
    sensor_desc: str = "",
    source: Optional[str] = None,
    packs: Sequence[str] = ("core",),
    adapter: Optional[str] = None,
    run_check_stage: bool = True,
    check_packs: Sequence[str] = ("core",),
    check_sample: int = 8,
    splits: Optional[Sequence[str]] = None,
    resume: bool = False,
    sparse_weights: Union[bool, str] = "auto",
    created_at: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Build, simulate, audit, declare, compare and check a data set.

    Args:
        config: The sensor (grids and populations); mutated per entry by the batch
            runner, so pass one only this call uses.
        spec: The data set (a world and its splits).
        out_dir: The data-set folder (manifest, index, one bundle per entry); the
            stage files go in ``out_dir / "evaluate"``.
        design_manifest: A design directory's manifest, stamped into every bundle.
        sensor_desc: Where the sensor came from (for ``pipeline.json``).
        source: The file the data set was built from (for ``pipeline.json``).
        packs: Metric packs of the audit. Chosen explicitly, never detected from the
            world (contract Q9).
        adapter: A registered declared adapter; default the first that accepts.
        run_check_stage: Run the reference-free checks.
        check_packs: Check packs of that stage.
        check_sample: Entries the engine-level checks re-run.
        splits: Simulate only these splits (the audit then reads those entries).
        resume: Skip entries whose bundle already exists.
        sparse_weights: How the bundles store receptive-field weights (``False`` =
            dense; see :func:`sensoryforge.io.bundle.write_bundle`).
        created_at: UTC timestamp for ``created_at``; default now.
        log: Called with one progress line.

    Returns:
        The ``pipeline.json`` content. ``status`` is ``"failed"`` when a stage did not
        finish (the failing stage's record names the reason; later stages are not run),
        also when some entries failed to simulate.

    Raises:
        ValueError: If a metric pack, check pack or adapter name is not registered
            (checked before anything is simulated).
        Exception: Any error a stage raises outside :data:`_STAGE_ERRORS`, after
            ``pipeline.json`` has been written with that stage marked failed.
    """
    emit = log or (lambda _line: None)
    out_dir = Path(out_dir)
    report = out_dir / REPORT_DIR
    for name in packs:
        if not _registered(METRIC_PACK_REGISTRY, name):
            raise ValueError(f"metric pack {name!r} is not registered")
    if run_check_stage:
        for name in check_packs:
            if not _registered(CHECK_PACK_REGISTRY, name):
                raise ValueError(f"check pack {name!r} is not registered")

    stages: List[Dict[str, Any]] = []
    notes: List[str] = []
    counts: Dict[str, Any] = {}
    batch_summary: Optional[Dict[str, int]] = None
    adapter_name: Optional[str] = None
    error: Optional[str] = None
    unexpected: Optional[BaseException] = None

    try:

        def build() -> List[Any]:
            built = build_dataset(spec)
            write_dataset(spec, built, out_dir)
            return built

        entries = _timed(stages, "build", build, emit)
        counts["entries"] = len(entries)
        emit(f"{len(entries)} entries on world {spec.world.world_id}")

        def simulate() -> Dict[str, int]:
            summary = run_dataset(
                config,
                spec,
                out_dir,
                design_manifest=design_manifest,
                splits=splits,
                resume=resume,
                sparse_weights=sparse_weights,
                log=emit,
            )
            if summary["failed"]:
                raise ValueError(
                    f"{summary['failed']} of {len(entries)} entries failed to "
                    "simulate (see index/ for the reasons)"
                )
            return summary

        try:
            batch_summary = _timed(stages, "batch", simulate, emit)
        except _StageFailed:
            # The summary is lost with the exception; read it back from the index.
            batch_summary = _batch_from_index(out_dir)
            raise

        def audit() -> Dict[str, Any]:
            proc = _run_audit_process(out_dir, report, packs)
            if proc.returncode != 0:
                raise ValueError(proc.stderr.strip() or "the audit exited abnormally")
            emit(proc.stdout.strip())
            return {}

        _timed(stages, "audit", audit, emit)

        declared = _timed(
            stages,
            "declared",
            lambda: run_declared(out_dir, report, adapter=adapter),
            emit,
        )
        adapter_name = declared["adapter"]["name"]
        if (
            _registered(METRIC_PACK_REGISTRY, adapter_name)
            and adapter_name not in packs
        ):
            notes.append(
                f"the declared adapter {adapter_name!r} has a metric pack of the same "
                f"name that was not run; add --packs core,{adapter_name} to compare "
                "its metrics"
            )

        compare = _timed(
            stages,
            "compare",
            lambda: run_compare(report, report, report),
            emit,
        )
        counts["compared_pairs"] = sum(compare["counts"].values())
        counts["audited"] = compare["entries"]["n"]
        if counts["compared_pairs"] == 0:
            notes.append(
                "no metric is on both sides, so the comparison is empty; the declared "
                "adapter's metric ids need the matching pack in --packs"
            )

        if run_check_stage:
            checks = _timed(
                stages,
                "checks",
                lambda: run_checks(
                    out_dir,
                    report,
                    packs=list(check_packs),
                    sample=check_sample,
                ),
                emit,
            )
            counts["checks"] = dict(checks["counts"])
        else:
            stages.append(
                {"name": "checks", "status": "skipped", "detail": None, "seconds": 0.0}
            )
    except _StageFailed as exc:
        error = str(exc)
        emit(f"stopped: {error}")
    except BaseException as exc:  # pipeline.json is written, then it is re-raised
        error = f"{type(exc).__name__}: {exc}"
        unexpected = exc
        emit(f"stopped: {error}")

    present = {s["name"] for s in stages}
    for name in STAGES:
        if name not in present:
            stages.append(
                {"name": name, "status": "skipped", "detail": None, "seconds": 0.0}
            )
    stages.sort(key=lambda s: STAGES.index(s["name"]))

    report.mkdir(parents=True, exist_ok=True)
    files = {
        f.name: file_sha256(f)
        for f in sorted(report.iterdir())
        if f.is_file() and f.name != PIPELINE_FILE
    }
    doc: Dict[str, Any] = {
        "format": FORMAT,
        "created_at": timestamp(created_at),
        "producer": producer(),
        "inputs": {
            "source": source,
            "dataset_id": spec.dataset_id,
            "dataset_name": spec.name,
            "world_id": spec.world.world_id,
            "seed": spec.seed,
            "duration_ms": spec.duration_ms,
            "sensor": sensor_desc,
            "design_id": (design_manifest or {}).get("design_id"),
            "packs": list(packs),
            "adapter": adapter_name,
            "check_packs": list(check_packs) if run_check_stage else None,
            "splits": list(splits) if splits else None,
            "resume": bool(resume),
        },
        "status": "failed" if error else "ok",
        "error": error,
        "stages": stages,
        "batch": batch_summary,
        "counts": counts,
        "files": files,
        "notes": notes,
    }
    write_json(report / PIPELINE_FILE, doc)
    if unexpected is not None:
        raise unexpected
    return doc


def _batch_from_index(out_dir: Path) -> Optional[Dict[str, int]]:
    """``{"ok", "failed", "skipped"}`` from the merged batch index, or ``None``."""
    from sensoryforge.world.runner import read_batch_index

    try:
        rows = read_batch_index(out_dir)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return {
        "ok": sum(r.get("status") == "ok" for r in rows),
        "failed": sum(r.get("status") == "failed" for r in rows),
        "skipped": 0,
    }
