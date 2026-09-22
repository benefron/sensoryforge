"""Parameter sweeps over a whole config: one directory, one run, per point.

:class:`~sensoryforge.core.batch_executor.BatchExecutor` sweeps *stimulus*
parameters only, so a sweep over any other config field (a filter's ``tau_r``,
a grid's ``spacing``, a population's ``input_gain``) needs its own, simpler
model, and this is it:

* :func:`sweep_paths` lists every dotted path worth sweeping in a given
  config -- the numeric schema fields, plus the registries' own
  ``get_param_spec()`` names for each population's filter, neuron and
  receptive-field builder, so a plugin component's parameters are sweepable
  with no code here.
* :func:`write_sweep` writes one ``combo_NNN/config.yml`` per point of the
  cartesian product, plus a ``manifest.json``. Each file is a complete
  canonical config: ``sensoryforge run combo_003/config.yml`` reproduces that
  point exactly, today or on a cluster next year.
* :class:`SweepController` runs those configs as ``QProcess`` subprocesses,
  ``parallel`` at a time. A subprocess (rather than a thread) so that a
  combination big enough to be killed by the OOM killer takes only itself
  down, not the window.
* :func:`write_slurm_script` writes the same sweep as a SLURM array job.

Units follow the rest of SensoryForge: ``duration_ms`` in ms.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore

import sensoryforge
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.gui.session import resolve_parent, set_by_path
from sensoryforge.registry import (
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    NEURON_REGISTRY,
)
from sensoryforge.stimuli.base import ParamSpec

#: How a combination directory is named. ``%03d`` matches the SLURM array
#: script's ``printf`` and is what :class:`SweepManifest` records.
COMBO_DIR_FORMAT = "combo_{:03d}"

#: The manifest filename inside a sweep root.
MANIFEST_FILENAME = "manifest.json"

#: How long ``cancel()`` waits for a terminated process before killing it.
CANCEL_KILL_DELAY_MS = 5000

#: The three ``simulation`` fields worth sweeping. ``device``,
#: ``integrate_dt_ms`` and ``seed`` are deliberately absent: the first two
#: change what a run *means* rather than a parameter of it, and a seed sweep
#: is a different feature (repeats, not a parameter axis).
_SIMULATION_PATHS = ("dt_ms", "duration_ms")

#: Neuron parameters build_neuron always overrides (integration step, noise).
_RUN_OWNED_NEURON = frozenset({"dt", "noise_std"})

#: ``(attribute holding the params dict, registry, attribute naming the
#: component)`` for each of a population's registry-driven parameter groups.
_POPULATION_REGISTRY_GROUPS = (
    ("filter_params", FILTER_REGISTRY, "filter_method"),
    ("model_params", NEURON_REGISTRY, "neuron_model"),
    ("innervation_params", INNERVATION_REGISTRY, "innervation_method"),
)


def _is_numeric(value: Any) -> bool:
    """Whether ``value`` is a number a spinbox could sweep.

    ``bool`` is excluded although it is an ``int`` in Python: a two-point
    sweep over ``True``/``False`` is a different UI, not a numeric range.

    Args:
        value: The current value of a config field.

    Returns:
        ``True`` for ``int``/``float``, ``False`` for everything else.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_field_names(obj: Any) -> List[str]:
    """The dataclass fields of ``obj`` whose current value is numeric.

    A field left at ``None`` (``GridConfig.rows`` on a density-based grid,
    ``SimulationConfig.duration_ms`` before a run) is not offered: there is no
    value to sweep around and no way to know the units.

    Args:
        obj: A dataclass instance (``GridConfig``, ``PopulationConfig``, ...).

    Returns:
        Field names, in declaration order.
    """
    return [
        f.name for f in dataclasses.fields(obj) if _is_numeric(getattr(obj, f.name))
    ]


def _registry_specs(registry: Any, component_name: Optional[str]) -> List[ParamSpec]:
    """``registry.get_param_spec(name)``, empty for a name it does not know.

    Args:
        registry: A :class:`~sensoryforge.registry.ComponentRegistry`.
        component_name: The registered name, in any case (registry lookups are
            case-insensitive, F-046), or ``None``.

    Returns:
        The component's :class:`ParamSpec` list, or ``[]`` when the name is
        empty or not registered -- a config naming a plugin that is not
        installed here still yields a usable path list.
    """
    if not component_name:
        return []
    try:
        return list(registry.get_param_spec(component_name))
    except KeyError:
        return []


def sweep_paths(
    config: SensoryForgeConfig,
) -> List[Tuple[str, Optional[ParamSpec]]]:
    """Every dotted path in ``config`` that can be swept, with its spec.

    Args:
        config: The experiment to enumerate.

    Returns:
        ``(path, spec)`` pairs in screen order -- grids, populations (schema
        fields then filter, neuron and innervation parameters), stimulus,
        simulation. ``spec`` is the component's :class:`ParamSpec` when the
        path comes from a registry (so a UI can show units and ranges) and
        ``None`` for a plain schema field.

    Example:
        >>> [p for p, _ in sweep_paths(config)][:2]   # doctest: +SKIP
        ['grids.0.rows', 'grids.0.cols']
    """
    paths: List[Tuple[str, Optional[ParamSpec]]] = []

    for index, grid in enumerate(config.grids):
        for name in _numeric_field_names(grid):
            paths.append((f"grids.{index}.{name}", None))

    for index, population in enumerate(config.populations):
        # A schema field also set in innervation_params is overridden by it
        # (SimulationEngine.builder_params merges innervation_params last),
        # so sweeping the schema field would change nothing.
        shadowed = set(population.innervation_params or {})
        for name in _numeric_field_names(population):
            if name in shadowed:
                continue
            paths.append((f"populations.{index}.{name}", None))
        for params_attr, registry, component_attr in _POPULATION_REGISTRY_GROUPS:
            specs = _registry_specs(registry, getattr(population, component_attr, None))
            for spec in specs:
                if spec.dtype not in ("float", "int"):
                    continue
                if params_attr == "model_params" and spec.name in _RUN_OWNED_NEURON:
                    continue  # the engine sets these itself (build_neuron)
                paths.append((f"populations.{index}.{params_attr}.{spec.name}", spec))

    for name in _numeric_field_names(config.stimulus):
        paths.append((f"stimulus.{name}", None))

    for name in _SIMULATION_PATHS:
        paths.append((f"simulation.{name}", None))

    return paths


@dataclass
class SweepSpec:
    """Which fields to sweep, and over which values.

    Attributes:
        fields: ``(dotted path, values)`` pairs. The sweep is their cartesian
            product, in the order given (the first field varies slowest).

    Raises:
        ValueError: If there are no fields, or a field has no values.
    """

    fields: List[Tuple[str, List[Any]]]

    def __post_init__(self) -> None:
        """Reject an empty sweep, which would silently write nothing."""
        if not self.fields:
            raise ValueError("a sweep needs at least one field")
        for path, values in self.fields:
            if not values:
                raise ValueError(f"sweep field {path!r} has no values")

    @property
    def n_combinations(self) -> int:
        """How many configs :func:`write_sweep` will write."""
        total = 1
        for _path, values in self.fields:
            total *= len(values)
        return total


@dataclass
class SweepManifest:
    """The written sweep: where it is and what each combination holds.

    Attributes:
        root: The sweep directory, holding one ``combo_NNN/`` per point and
            a ``manifest.json``.
        combos: One entry per combination, each
            ``{"index": int, "dir": "combo_000", "values": {path: value}}``.
    """

    root: Path
    combos: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Accept a ``str`` root."""
        self.root = Path(self.root)

    def combo_dir(self, index: int) -> Path:
        """The directory of combination ``index``.

        Args:
            index: Position in :attr:`combos`.

        Returns:
            ``root/<dir>`` for that entry.

        Raises:
            ValueError: If there is no combination at that index.
        """
        if not 0 <= index < len(self.combos):
            raise ValueError(
                f"no combination at index {index}: the sweep has " f"{len(self.combos)}"
            )
        return self.root / self.combos[index]["dir"]

    def config_path(self, index: int) -> Path:
        """The ``config.yml`` of combination ``index``."""
        return self.combo_dir(index) / "config.yml"

    def bundle_path(self, index: int) -> Path:
        """Where combination ``index`` writes its bundle."""
        return self.combo_dir(index) / "bundle"


def write_sweep(
    config: SensoryForgeConfig,
    spec: SweepSpec,
    *,
    root: Path,
    duration_ms: float,
) -> SweepManifest:
    """Write one complete config per point of ``spec``'s cartesian product.

    Args:
        config: The base experiment. Deep-copied per combination; never
            mutated.
        spec: The fields and values to sweep.
        root: The sweep directory (created, with its parents).
        duration_ms: Written into every combination's
            ``simulation.duration_ms``, so a combination is self-describing
            even if the sweep also sweeps that field (an explicit
            ``simulation.duration_ms`` value in ``spec`` wins -- it is applied
            after).

    Returns:
        The :class:`SweepManifest`, also written to ``root/manifest.json``.

    Raises:
        ValueError: If a swept path does not resolve in ``config`` -- named,
            before anything is written.
    """
    root = Path(root)
    base = copy.deepcopy(config)
    base.simulation.duration_ms = float(duration_ms)
    # Resolve every path once against the base config, so a typo fails here
    # rather than after writing half a sweep. Only the parent is resolved:
    # a params-dict key that does not exist yet (`filter_params.tau_r` on a
    # population that has never overridden it) is a legitimate sweep target,
    # and `set_by_path` creates it.
    for path, _values in spec.fields:
        resolve_parent(base, path)

    root.mkdir(parents=True, exist_ok=True)
    paths = [path for path, _values in spec.fields]
    value_lists: Sequence[List[Any]] = [values for _path, values in spec.fields]

    combos: List[Dict[str, Any]] = []
    for index, combination in enumerate(itertools.product(*value_lists)):
        cfg = copy.deepcopy(base)
        values: Dict[str, Any] = {}
        for path, value in zip(paths, combination):
            set_by_path(cfg, path, value)
            values[path] = value
        combo_dir = root / COMBO_DIR_FORMAT.format(index)
        combo_dir.mkdir(exist_ok=True)
        (combo_dir / "config.yml").write_text(cfg.to_yaml(), encoding="utf-8")
        combos.append({"index": index, "dir": combo_dir.name, "values": values})

    manifest = SweepManifest(root=root, combos=combos)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "root": str(root),
                "duration_ms": float(duration_ms),
                "fields": paths,
                "combos": combos,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return manifest


# --------------------------------------------------------------------- SLURM


def write_slurm_script(
    manifest: SweepManifest,
    *,
    settings: Dict[str, Any],
) -> Path:
    """Write the sweep as a SLURM array job, one array task per combination.

    Args:
        manifest: The written sweep.
        settings: The job settings the old Batch tab's dialog offered --
            ``job_name``, ``partition``, ``time``, ``mem_gb``,
            ``cpus_per_task``, ``gpus``, ``conda_env`` -- and ``script_name``
            (the file written inside the sweep root, default
            ``run_sweep.sh``). Each task runs its combination's own
            ``simulation.duration_ms``; a ``duration_ms`` key is ignored.
            Missing keys take the dialog's own defaults.

    Returns:
        The path of the written script; submit it with ``sbatch``.

    Raises:
        ValueError: If the manifest has no combinations.
    """
    if not manifest.combos:
        raise ValueError("cannot write a SLURM script for a sweep with no combinations")

    job_name = str(settings.get("job_name", "sensoryforge_sweep"))
    partition = str(settings.get("partition", "gpu"))
    walltime = str(settings.get("time", "04:00:00"))
    mem_gb = int(settings.get("mem_gb", 32))
    cpus = int(settings.get("cpus_per_task", 4))
    gpus = int(settings.get("gpus", 1))
    conda_env = str(settings.get("conda_env", "sensoryforge"))
    script_name = str(settings.get("script_name", "run_sweep.sh"))

    root = manifest.root.resolve()
    last = len(manifest.combos) - 1

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={root}/{job_name}_%A_%a.out",
        f"#SBATCH --error={root}/{job_name}_%A_%a.err",
        f"#SBATCH --array=0-{last}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines += [
        "",
        "# Auto-generated by sensoryforge.gui.execution.sweep_controller.",
        f"# Sweep root: {root}",
        f"# Combinations: {len(manifest.combos)}",
        "",
        "set -euo pipefail",
        "source $(conda info --base)/etc/profile.d/conda.sh",
        f"conda activate {conda_env}",
        "",
        f"SWEEP_ROOT={root}",
        'COMBO=$(printf "combo_%03d" "$SLURM_ARRAY_TASK_ID")',
        "",
        "sensoryforge run \\",
        '    "$SWEEP_ROOT/$COMBO/config.yml" \\',
        '    --bundle "$SWEEP_ROOT/$COMBO/bundle"',
        "",
    ]
    script_path = manifest.root / script_name
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


# ---------------------------------------------------------------- subprocess


def sweep_command(manifest: SweepManifest, index: int, duration_ms: float) -> List[str]:
    """The exact argv one combination is run with.

    ``sys.executable -m sensoryforge.cli`` rather than a bare ``sensoryforge``:
    a console script on ``PATH`` may belong to a different checkout or a
    different environment entirely (F-053), and the sweep must run the code
    the window is running.

    Args:
        manifest: The written sweep.
        index: Which combination.
        duration_ms: Unused; kept for callers. The combination's config
            holds its duration.

    Returns:
        The argv list, program first.
    """
    return [
        sys.executable,
        "-m",
        "sensoryforge.cli",
        "run",
        str(manifest.config_path(index)),
        # No --duration: each combination's config carries its own
        # simulation.duration_ms (a duration sweep included), which the CLI
        # runs when no flag overrides it.
        "--bundle",
        str(manifest.bundle_path(index)),
    ]


def sweep_environment() -> QtCore.QProcessEnvironment:
    """The environment a sweep subprocess gets, with ``PYTHONPATH`` pinned.

    The parent's environment, with the directory that holds the *running*
    ``sensoryforge`` package prepended to ``PYTHONPATH``. In a git worktree
    that package is not the one an editable install points at (F-053: never
    ``pip install -e .`` from a worktree), so without this a sweep launched
    from a worktree would silently run a different checkout's code.

    Returns:
        A :class:`QtCore.QProcessEnvironment` for
        :meth:`QProcess.setProcessEnvironment`.
    """
    env = QtCore.QProcessEnvironment.systemEnvironment()
    package_parent = str(Path(sensoryforge.__file__).resolve().parent.parent)
    existing = env.value("PYTHONPATH", "")
    parts = [package_parent] + [p for p in existing.split(os.pathsep) if p]
    env.insert("PYTHONPATH", os.pathsep.join(parts))
    return env


class SweepController(QtCore.QObject):
    """Runs a written sweep as ``parallel`` ``sensoryforge run`` subprocesses.

    Signals:
        log(str): One line of a subprocess's merged stdout/stderr, prefixed
            with its combination directory.
        progress(int, int): ``(combinations finished, total)``.
        finished(int): How many combinations exited non-zero (0 = all good).
        failed(str): A subprocess could not be started at all (a missing
            interpreter, an unreadable working directory). Combination
            failures are counted into ``finished``, not reported here.

    Args:
        parent: Qt parent.

    Example:
        >>> controller = SweepController()                       # doctest: +SKIP
        >>> controller.start(manifest, duration_ms=50.0, parallel=2)  # doctest: +SKIP
    """

    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._manifest: Optional[SweepManifest] = None
        self._duration_ms: float = 0.0
        self._parallel: int = 1
        self._next: int = 0
        self._done: int = 0
        self._n_failed: int = 0
        self._cancelled: bool = False
        self._processes: Dict[int, QtCore.QProcess] = {}

    @property
    def running(self) -> bool:
        """Whether any combination is still executing."""
        return bool(self._processes)

    def start(
        self,
        manifest: SweepManifest,
        *,
        duration_ms: float,
        parallel: int = 1,
    ) -> None:
        """Run every combination of ``manifest``.

        Args:
            manifest: A sweep written by :func:`write_sweep`.
            duration_ms: Unused (each combination's config holds its
                duration); kept for callers.
            parallel: How many subprocesses may run at once (at least 1).

        Raises:
            RuntimeError: If a sweep is already running.
            ValueError: If the manifest has no combinations, or ``parallel``
                is less than 1.
        """
        if self.running:
            raise RuntimeError("a sweep is already running; cancel it first")
        if not manifest.combos:
            raise ValueError("this sweep has no combinations to run")
        if parallel < 1:
            raise ValueError(f"parallel must be at least 1, got {parallel}")

        self._manifest = manifest
        self._duration_ms = float(duration_ms)
        self._parallel = int(parallel)
        self._next = 0
        self._done = 0
        self._n_failed = 0
        self._cancelled = False
        self._processes = {}
        self._fill()

    def cancel(self) -> None:
        """Terminate every running combination, killing after 5 s.

        Combinations not yet started are dropped. ``finished`` is emitted once
        the last process is reaped, counting the terminated ones as failures.
        """
        self._cancelled = True
        for index, process in list(self._processes.items()):
            self.log.emit(f"[{self._dir_name(index)}] cancelling")
            process.terminate()
            QtCore.QTimer.singleShot(CANCEL_KILL_DELAY_MS, process.kill)

    # ----------------------------------------------------------------- inner

    def _dir_name(self, index: int) -> str:
        """The combination's directory name, for log prefixes."""
        if self._manifest is None:
            return COMBO_DIR_FORMAT.format(index)
        return self._manifest.combos[index]["dir"]

    def _fill(self) -> None:
        """Start combinations until ``parallel`` are in flight, or none left."""
        if self._manifest is None:
            return
        while (
            not self._cancelled
            and len(self._processes) < self._parallel
            and self._next < len(self._manifest.combos)
        ):
            self._launch(self._next)
            self._next += 1
        if not self._processes and (
            self._cancelled or self._next >= len(self._manifest.combos)
        ):
            self._emit_finished()

    def _launch(self, index: int) -> None:
        """Start combination ``index`` as a subprocess."""
        if self._manifest is None:
            return
        argv = sweep_command(self._manifest, index, self._duration_ms)
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.setProcessEnvironment(sweep_environment())
        # An explicit working directory: a relative path in a config (an
        # imported-coordinates file, say) must resolve against the
        # combination that owns it, not against wherever the GUI was started.
        process.setWorkingDirectory(str(self._manifest.combo_dir(index)))
        process.readyReadStandardOutput.connect(partial(self._on_output, index))
        process.finished[int, QtCore.QProcess.ExitStatus].connect(
            partial(self._on_process_finished, index)
        )
        process.errorOccurred.connect(partial(self._on_process_error, index))
        self._processes[index] = process
        self.log.emit(f"[{self._dir_name(index)}] $ {' '.join(argv)}")
        process.start(argv[0], argv[1:])

    def _on_output(self, index: int) -> None:
        """Drain one combination's merged output into :attr:`log`."""
        process = self._processes.get(index)
        if process is not None:
            self._on_output_of(index, process)

    def _on_output_of(self, index: int, process: QtCore.QProcess) -> None:
        """Drain ``process``'s buffered output into :attr:`log`."""
        text = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        prefix = self._dir_name(index)
        for line in text.splitlines():
            if line.strip():
                self.log.emit(f"[{prefix}] {line}")

    def _on_process_error(self, index: int, error: int) -> None:
        """Handle a subprocess that could not be started at all.

        ``QProcess`` does not always emit ``finished`` after ``FailedToStart``,
        so this must reap the combination itself -- otherwise one unstartable
        interpreter leaves the sweep waiting for a process that will never
        run, and ``finished`` is never emitted at all.
        """
        if error != QtCore.QProcess.FailedToStart:
            return
        self.failed.emit(
            f"{self._dir_name(index)}: could not start {sys.executable} "
            "-m sensoryforge.cli"
        )
        self._reap(index, ok=False, note="could not start")

    def _on_process_finished(self, index: int, exit_code: int, status: int) -> None:
        """Reap one combination and start the next."""
        ok = exit_code == 0 and status == QtCore.QProcess.NormalExit
        self._reap(index, ok=ok, note=f"exit {exit_code}")

    def _reap(self, index: int, *, ok: bool, note: str) -> None:
        """Count one combination as over, whichever way it ended.

        Idempotent: a process that emits both ``errorOccurred`` and
        ``finished`` is counted once, because the first call removes it from
        the in-flight table.

        Args:
            index: The combination.
            ok: Whether it succeeded.
            note: What to write in the log line.
        """
        process = self._processes.pop(index, None)
        if process is None:
            return
        self._on_output_of(index, process)
        process.deleteLater()
        if not ok:
            self._n_failed += 1
        self._done += 1
        total = len(self._manifest.combos) if self._manifest is not None else 0
        self.log.emit(f"[{self._dir_name(index)}] {note}" + ("" if ok else " (failed)"))
        self.progress.emit(self._done, total)
        self._fill()

    def _emit_finished(self) -> None:
        """Emit ``finished`` once, with the failure count.

        A combination that was cancelled before it ever started counts as a
        failure too: it produced no bundle, and a caller checking
        ``finished(0)`` must not read a cancelled sweep as a complete one.
        """
        if self._manifest is None:
            return
        never_ran = len(self._manifest.combos) - self._done
        self._manifest = None
        self.finished.emit(self._n_failed + never_ran)
