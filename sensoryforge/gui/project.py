"""A GUI v2 project on disk: one directory, one config, one run per bundle.

The new GUI shell persists exactly two things: the canonical configuration
(``config.yml``, the same YAML the CLI reads) and the runs produced from it
(``runs/<timestamp>_<name>/``, each a bundle directory written by
:mod:`sensoryforge.io.bundle`). ``layout.json`` is an advisory file for the
shell's own window/panel state; nothing else is written.

::

    my-project/
        config.yml                       # SensoryForgeConfig.to_yaml()
        layout.json                      # advisory GUI layout, may be absent
        runs/
            20260917-104233_sa_sweep/    # a bundle: holds config.json
                config.json
                ...

:class:`~sensoryforge.core.experiment_manager.ExperimentManager` is *not* used
by the new shell; it stays for the old tabs until Phase 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Union

from sensoryforge.config.schema import SensoryForgeConfig

CONFIG_FILENAME = "config.yml"
RUNS_DIRNAME = "runs"
LAYOUT_FILENAME = "layout.json"

#: The file that marks a directory under ``runs/`` as a finished bundle.
BUNDLE_MARKER = "config.json"

_RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def _safe_name(name: str) -> str:
    """Filesystem-safe run name: ``"SA #6 sweep"`` -> ``"SA_6_sweep"``.

    Mirrors :func:`sensoryforge.io.bundle._safe_name` so a run directory and
    the bundle written into it read the same way.

    Args:
        name: Free-form name, e.g. the config's ``metadata["name"]``.

    Returns:
        The name with every run of non-alphanumeric characters collapsed to a
        single underscore, or ``"run"`` if nothing usable is left.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "run"


@dataclass
class ProjectHandle:
    """A project directory, and the four paths the GUI needs inside it.

    Attributes:
        root: The project directory. A ``str`` is accepted and converted.

    Example:
        >>> project = ProjectHandle.create(Path("/tmp/demo"), config)
        >>> project.save_config(config)
        >>> project.new_run_dir("sa sweep").name  # doctest: +SKIP
        '20260917-104233_sa_sweep'
    """

    root: Path

    def __post_init__(self) -> None:
        """Accept a ``str`` root, so callers can pass a dialog's return value."""
        self.root = Path(self.root)

    # ------------------------------------------------------------------ paths

    @property
    def config_path(self) -> Path:
        """The canonical config YAML, ``<root>/config.yml``."""
        return self.root / CONFIG_FILENAME

    @property
    def runs_dir(self) -> Path:
        """The directory holding one bundle directory per run."""
        return self.root / RUNS_DIRNAME

    @property
    def layout_path(self) -> Path:
        """Advisory GUI layout state, ``<root>/layout.json`` (may not exist)."""
        return self.root / LAYOUT_FILENAME

    # --------------------------------------------------------------- lifecycle

    @classmethod
    def create(
        cls, root: Union[str, Path], config: SensoryForgeConfig
    ) -> "ProjectHandle":
        """Create a project directory and write ``config`` into it.

        Args:
            root: The directory to create (parents are created as needed). It
                may already exist, as long as it holds no ``config.yml``.
            config: The configuration to write as the project's ``config.yml``.

        Returns:
            A handle on the new project.

        Raises:
            ValueError: If ``root`` already contains a ``config.yml`` -- open
                that project instead of silently overwriting it.
        """
        project = cls(root)
        if project.config_path.exists():
            raise ValueError(
                f"{project.root} already contains a {CONFIG_FILENAME}; "
                "open the existing project instead of creating a new one"
            )
        project.root.mkdir(parents=True, exist_ok=True)
        project.runs_dir.mkdir(exist_ok=True)
        project.save_config(config)
        return project

    @classmethod
    def open(cls, root: Union[str, Path]) -> "ProjectHandle":
        """Open an existing project directory.

        Args:
            root: A directory holding a ``config.yml``.

        Returns:
            A handle on that project.

        Raises:
            ValueError: If ``root`` holds no ``config.yml``.
        """
        project = cls(root)
        if not project.config_path.is_file():
            raise ValueError(
                f"{project.root} is not a SensoryForge project: no "
                f"{CONFIG_FILENAME} in it"
            )
        return project

    # ------------------------------------------------------------------ config

    def save_config(self, config: SensoryForgeConfig) -> None:
        """Write ``config`` to ``config.yml``, replacing what was there.

        Args:
            config: The configuration to persist.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(config.to_yaml(), encoding="utf-8")

    def load_config(self) -> SensoryForgeConfig:
        """Read ``config.yml`` back.

        Returns:
            The project's configuration.

        Raises:
            ValueError: If ``config.yml`` is missing, or does not parse into
                a configuration mapping.
        """
        if not self.config_path.is_file():
            raise ValueError(f"{self.config_path}: no {CONFIG_FILENAME} to load")
        return SensoryForgeConfig.from_yaml_file(self.config_path)

    # -------------------------------------------------------------------- runs

    def new_run_dir(self, name: str) -> Path:
        """Return the path a new run's bundle should be written to.

        The directory is **not** created: ``write_bundle`` creates it, so an
        aborted run leaves nothing behind.

        Args:
            name: Free-form run name, e.g. ``config.metadata["name"]``.

        Returns:
            ``<root>/runs/<YYYYmmdd-HHMMSS>_<safe name>``. A numeric suffix is
            appended if that path is somehow taken already (two runs started
            inside the same second), so a run never overwrites another.
        """
        stamp = datetime.now().strftime(_RUN_TIMESTAMP_FORMAT)
        base = f"{stamp}_{_safe_name(name)}"
        candidate = self.runs_dir / base
        suffix = 2
        while candidate.exists():
            candidate = self.runs_dir / f"{base}-{suffix}"
            suffix += 1
        return candidate

    def list_runs(self) -> List[Path]:
        """The project's finished runs, newest first.

        A run is a directory under ``runs/`` that holds a ``config.json`` --
        the marker :func:`sensoryforge.io.bundle.write_bundle` writes. Loose
        files and half-written directories are ignored.

        Returns:
            Bundle directories sorted by modification time, newest first
            (ties broken by name, descending -- names start with a timestamp).
            Empty if the project has no ``runs/`` directory yet.
        """
        if not self.runs_dir.is_dir():
            return []
        bundles = [
            entry
            for entry in self.runs_dir.iterdir()
            if entry.is_dir() and (entry / BUNDLE_MARKER).is_file()
        ]
        return sorted(
            bundles, key=lambda path: (path.stat().st_mtime, path.name), reverse=True
        )
