"""The one in-memory experiment the GUI v2 shell is built around.

Every screen binds to a single :class:`Session`: it owns exactly one
:class:`~sensoryforge.config.schema.SensoryForgeConfig`, the device the next
run will use, the open :class:`~sensoryforge.gui.project.ProjectHandle`, and
the last :class:`RunResult`. Screens never hold their own copy of the config
and never reassign ``session.config``; they edit it through
:meth:`Session.set_by_path` (or mutate it and call :meth:`Session.notify`) and
listen to :attr:`Session.configChanged`, filtering by dotted-path prefix::

    session.configChanged.connect(self._on_config_changed)

    def _on_config_changed(self, path: str) -> None:
        if path.startswith("populations.1."):
            self.refresh()

Dotted paths address the config the way the schema nests: ``grids.<i>.<field>``,
``populations.<i>.<field>``, ``populations.<i>.model_params.<key>``,
``populations.<i>.filter_params.<key>``, ``populations.<i>.inputs.<j>.<field>``,
``stimulus.<field>``, ``simulation.<field>``. An integer segment indexes a list,
a segment on a dict is a key, anything else is an attribute. ``""`` (the empty
path) is emitted by nothing here; it is the convention for "the whole config
was replaced", which :attr:`Session.configReplaced` announces separately.

Units follow the rest of SensoryForge: ms at this API, mm for space, mA for
currents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PyQt5 import QtCore

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.project import ProjectHandle

#: Every device name the GUI understands. What this machine actually offers is
#: :func:`available_devices`.
KNOWN_DEVICES = ("cpu", "mps", "cuda")


def available_devices() -> List[str]:
    """The torch devices this machine can run on, ``"cpu"`` first.

    Returns:
        ``["cpu"]`` plus ``"mps"`` if Apple's Metal backend is available and
        ``"cuda"`` if a CUDA device is. Resolved by querying torch, so it is
        cheap to call but is resolved once per :class:`Session`.
    """
    devices = ["cpu"]
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def default_device(devices: List[str]) -> str:
    """Pick the device a new session starts on.

    Args:
        devices: The available device names, as :func:`available_devices`
            returns them.

    Returns:
        The first accelerator in ``devices`` (i.e. the first entry that is not
        ``"cpu"``), or ``"cpu"`` when that is all there is.

    Raises:
        ValueError: If ``devices`` is empty.
    """
    if not devices:
        raise ValueError("no devices available; expected at least 'cpu'")
    for name in devices:
        if name != "cpu":
            return name
    return "cpu"


@dataclass
class RunResult:
    """One finished simulation, with everything a view needs to draw it.

    Attributes:
        config_snapshot: Deep copy of the config taken when the run started,
            so later edits cannot change what the results mean.
        results: Exactly the dict
            :meth:`~sensoryforge.core.simulation_engine.SimulationEngine.run`
            returns with ``return_intermediates=True``.
        stimulus: The rendered stimulus, ``[1, time, H, W]`` (or
            ``[1, time, C, H, W]`` for a multi-channel grid), in mA.
        time_ms: Sample times in ms, one per rendered frame.
        canvas: The ``StimulusCanvas`` the stimulus was rendered on
            (``sensoryforge.stimuli.canvas.StimulusCanvas``); typed ``Any`` so
            this module does not import the stimulus package.
        bundle_dir: Where the run was written as a bundle, or ``None`` when it
            was not bundled (no project open, or a quick run).
        duration_ms: Simulated duration in ms.
        started: Wall-clock time the run started.
        elapsed_s: Wall-clock seconds the run took.
        quick: Whether this was a quick preview of a single population.
    """

    config_snapshot: SensoryForgeConfig
    results: Dict[str, Any]
    stimulus: torch.Tensor
    time_ms: np.ndarray
    canvas: Any
    bundle_dir: Optional[Path] = None
    duration_ms: float = 0.0
    started: datetime = field(default_factory=datetime.now)
    elapsed_s: float = 0.0
    quick: bool = False


def _path_error(path: str, segment: str, detail: str) -> ValueError:
    """Build the one error shape every unresolvable path raises."""
    return ValueError(f"config path {path!r}: segment {segment!r} {detail}")


def _list_index(container: List[Any], segment: str, path: str) -> int:
    """Resolve ``segment`` as an index into ``container``.

    Raises:
        ValueError: If the segment is not an integer, or is out of range --
            naming the segment and the list's length.
    """
    try:
        index = int(segment)
    except ValueError:
        raise _path_error(
            path, segment, f"must be an integer index into a list of {len(container)}"
        ) from None
    if not -len(container) <= index < len(container):
        raise _path_error(
            path, segment, f"is out of range for a list of {len(container)}"
        )
    return index


def _descend(container: Any, segment: str, path: str) -> Any:
    """Read one segment out of ``container`` (list index, dict key, attribute).

    Raises:
        ValueError: If the segment does not resolve, naming it.
    """
    if isinstance(container, list):
        return container[_list_index(container, segment, path)]
    if isinstance(container, dict):
        if segment not in container:
            raise _path_error(
                path, segment, f"is not a key of {sorted(container) or '{}'}"
            )
        return container[segment]
    if not hasattr(container, segment):
        raise _path_error(
            path, segment, f"is not a field of {type(container).__name__}"
        )
    return getattr(container, segment)


class Session(QtCore.QObject):
    """The one in-memory experiment: a SensoryForgeConfig plus transient state.

    Signals:
        configChanged(str): Dotted path of what changed. ``""`` means the whole
            config was replaced (:attr:`configReplaced` says so too).
        configReplaced(): A different config is now in place (load or new);
            every view rebuilds itself from scratch.
        deviceChanged(str): The run device changed.
        resultsChanged(object): The last :class:`RunResult`, or ``None``.
        projectChanged(object): The open :class:`ProjectHandle`, or ``None``.
        staleChanged(bool): ``True`` once the config is edited after a run, so
            views can mark the displayed results as out of date.

    Attributes:
        config: The experiment. Never reassigned except by
            :meth:`replace_config`.
        device: ``"cpu"``, ``"mps"`` or ``"cuda"`` -- the device the next run
            uses. Availability is the caller's business
            (:attr:`available_devices`); this is transient GUI state and is
            deliberately *not* written into ``config.simulation.device``.
        available_devices: What this machine offers, resolved once here.
        project: The open project, or ``None`` for an unsaved experiment.
        last_results: The most recent run, or ``None``.
        stale: Whether the config was edited since :attr:`last_results`.

    Example:
        >>> session = Session(SensoryForgeConfig())          # doctest: +SKIP
        >>> session.set_by_path("simulation.dt_ms", 0.5)     # doctest: +SKIP
    """

    configChanged = QtCore.pyqtSignal(str)
    configReplaced = QtCore.pyqtSignal()
    deviceChanged = QtCore.pyqtSignal(str)
    resultsChanged = QtCore.pyqtSignal(object)
    projectChanged = QtCore.pyqtSignal(object)
    staleChanged = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        config: Optional[SensoryForgeConfig] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Start a session over ``config`` (an empty one when omitted).

        Args:
            config: The experiment to hold. Taken as-is, not copied: the
                session owns it from here on.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.config: SensoryForgeConfig = (
            config if config is not None else SensoryForgeConfig()
        )
        self.available_devices: List[str] = available_devices()
        self.device: str = default_device(self.available_devices)
        self.project: Optional[ProjectHandle] = None
        self.last_results: Optional[RunResult] = None
        self.stale: bool = False

    # ------------------------------------------------------------------ state

    def replace_config(self, config: SensoryForgeConfig) -> None:
        """Put a different config in place and tell every view to rebuild.

        Args:
            config: The new experiment (from a load, a preset, or File > New).

        Raises:
            ValueError: If ``config`` is not a
                :class:`~sensoryforge.config.schema.SensoryForgeConfig`.
        """
        if not isinstance(config, SensoryForgeConfig):
            raise ValueError(
                "replace_config expects a SensoryForgeConfig, got "
                f"{type(config).__name__}"
            )
        self.config = config
        self.configReplaced.emit()

    def notify(self, path: str) -> None:
        """Announce that ``path`` in the config changed.

        Call this after mutating the config directly;
        :meth:`set_by_path` calls it for you.

        Args:
            path: Dotted path of what changed, e.g.
                ``"populations.1.filter_params.tau_r"``.
        """
        self.configChanged.emit(path)
        if self.last_results is not None:
            self._set_stale(True)

    def set_device(self, device: str) -> None:
        """Choose the device the next run uses.

        Args:
            device: One of :data:`KNOWN_DEVICES`. Whether this machine has it
                is not checked here -- see :attr:`available_devices`.

        Raises:
            ValueError: If ``device`` is not a device name SensoryForge knows.
        """
        if device not in KNOWN_DEVICES:
            raise ValueError(
                f"unknown device {device!r}; expected one of {list(KNOWN_DEVICES)}"
            )
        if device == self.device:
            return
        self.device = device
        self.deviceChanged.emit(device)

    def set_results(self, results: Optional[RunResult]) -> None:
        """Publish the results of a finished run (or clear them).

        The session stops being stale: what the views show now matches the
        config as it stands.

        Args:
            results: The finished run, or ``None`` to clear.
        """
        self.last_results = results
        self.resultsChanged.emit(results)
        self._set_stale(False)

    def set_project(self, project: Optional[ProjectHandle]) -> None:
        """Open a project (or close the one that is open).

        The config is not touched: the caller decides whether to
        :meth:`replace_config` with the project's saved one.

        Args:
            project: The project now open, or ``None``.
        """
        self.project = project
        self.projectChanged.emit(project)

    def _set_stale(self, stale: bool) -> None:
        """Set :attr:`stale`, emitting only when it actually changes."""
        if stale == self.stale:
            return
        self.stale = stale
        self.staleChanged.emit(stale)

    # -------------------------------------------------------------- accessors

    def population(self, index: int) -> PopulationConfig:
        """The population at ``index``.

        Args:
            index: Position in ``config.populations`` (negative counts back).

        Returns:
            The population config.

        Raises:
            ValueError: If there is no population at that index.
        """
        populations = self.config.populations
        if not -len(populations) <= index < len(populations):
            raise ValueError(
                f"no population at index {index}: the config has "
                f"{len(populations)} population(s)"
            )
        return populations[index]

    def grid(self, name: str) -> GridConfig:
        """The grid called ``name``.

        Args:
            name: The grid's ``name`` field.

        Returns:
            The grid config.

        Raises:
            ValueError: If no grid has that name -- listing the names there are.
        """
        for grid in self.config.grids:
            if grid.name == name:
                return grid
        known = [g.name for g in self.config.grids]
        raise ValueError(f"no grid named {name!r}; the config has {known}")

    # ------------------------------------------------------------ dotted paths

    def get_by_path(self, path: str) -> Any:
        """Read a value out of the config by dotted path.

        Args:
            path: e.g. ``"populations.0.filter_params.tau_r"``.

        Returns:
            The value at that path (a leaf, or a whole branch such as
            ``"populations.0"``).

        Raises:
            ValueError: If ``path`` is empty or does not resolve, naming the
                segment that failed.
        """
        container, last = self._resolve_parent(path)
        return _descend(container, last, path)

    def set_by_path(self, path: str, value: Any) -> None:
        """Write a value into the config by dotted path, then :meth:`notify`.

        A key that a params dict does not have yet is added (that is how an
        optional model/filter parameter first gets a value); an attribute that
        a dataclass does not have is an error, since it can only be a typo.

        Args:
            path: e.g. ``"populations.0.filter_params.tau_r"``.
            value: The new value.

        Raises:
            ValueError: If ``path`` is empty or does not resolve, naming the
                segment that failed. Nothing is written and nothing is emitted
                in that case.
        """
        container, last = self._resolve_parent(path)
        if isinstance(container, list):
            container[_list_index(container, last, path)] = value
        elif isinstance(container, dict):
            container[last] = value
        elif hasattr(container, last):
            setattr(container, last, value)
        else:
            raise _path_error(
                path, last, f"is not a field of {type(container).__name__}"
            )
        self.notify(path)

    def _resolve_parent(self, path: str) -> tuple:
        """Walk every segment but the last.

        Args:
            path: The dotted path.

        Returns:
            ``(container, last_segment)`` -- the object the last segment
            addresses a member of, and that segment.

        Raises:
            ValueError: If ``path`` is empty, or a segment does not resolve.
        """
        if not path:
            raise ValueError("config path is empty")
        segments = path.split(".")
        container: Any = self.config
        for segment in segments[:-1]:
            container = _descend(container, segment, path)
        return container, segments[-1]
