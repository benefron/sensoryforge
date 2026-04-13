"""Experiment directory manager.

Owns the concept of a "current project" — a directory on disk with a standard
sub-directory layout.  All GUI tabs share one ExperimentManager instance so
saves always go to the same location without each tab prompting separately.

Directory layout::

    <project>/
        config.yaml          ← canonical SensoryForgeConfig
        stimuli/             ← stimulus .pt or .npy files
        results/             ← HDF5 or .pt spike results
        figures/             ← exported PNG/SVG figures

Usage in a GUI tab::

    if self._em is not None and self._em.is_open:
        save_dir = self._em.results_dir
    else:
        save_dir = Path(QFileDialog.getExistingDirectory(...))
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = ".sensoryforge_project.json"

STIMULI_SUBDIR = "stimuli"
RESULTS_SUBDIR = "results"
FIGURES_SUBDIR = "figures"


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """Manages an on-disk experiment directory shared across GUI tabs.

    Attributes:
        project_dir: Root directory of the currently open project, or None.
    """

    def __init__(self) -> None:
        self._project_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True when a project directory is currently open."""
        return self._project_dir is not None

    @property
    def project_dir(self) -> Optional[Path]:
        """Root directory of the current project, or None."""
        return self._project_dir

    @property
    def config_path(self) -> Optional[Path]:
        """Path to ``config.yaml`` inside the project, or None if no project open."""
        if self._project_dir is None:
            return None
        return self._project_dir / "config.yaml"

    @property
    def stimuli_dir(self) -> Optional[Path]:
        """Path to the ``stimuli/`` sub-directory, or None if no project open."""
        if self._project_dir is None:
            return None
        return self._project_dir / STIMULI_SUBDIR

    @property
    def results_dir(self) -> Optional[Path]:
        """Path to the ``results/`` sub-directory, or None if no project open."""
        if self._project_dir is None:
            return None
        return self._project_dir / RESULTS_SUBDIR

    @property
    def figures_dir(self) -> Optional[Path]:
        """Path to the ``figures/`` sub-directory, or None if no project open."""
        if self._project_dir is None:
            return None
        return self._project_dir / FIGURES_SUBDIR

    # ------------------------------------------------------------------
    # Open / create / close
    # ------------------------------------------------------------------

    def create(self, path: Path, name: Optional[str] = None) -> None:
        """Create a new project at ``path`` and open it.

        Args:
            path: Root directory to create.  Parent must exist.
            name: Human-readable project name.  Defaults to directory name.

        Raises:
            FileExistsError: If ``path`` already exists and is non-empty.
            OSError: If the directory cannot be created.
        """
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Directory '{path}' already exists and is not empty."
            )
        path.mkdir(parents=True, exist_ok=True)
        (path / STIMULI_SUBDIR).mkdir(exist_ok=True)
        (path / RESULTS_SUBDIR).mkdir(exist_ok=True)
        (path / FIGURES_SUBDIR).mkdir(exist_ok=True)

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "sensoryforge_experiment",
            "name": name or path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (path / _MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        self._project_dir = path

    def open(self, path: Path) -> None:
        """Open an existing project directory.

        If ``path`` is missing the manifest file it is adopted anyway —
        sub-directories are created as needed (tolerant open).

        Args:
            path: Root directory of the project.

        Raises:
            NotADirectoryError: If ``path`` is not a directory.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"'{path}' is not a directory.")
        # Ensure sub-directories exist
        (path / STIMULI_SUBDIR).mkdir(exist_ok=True)
        (path / RESULTS_SUBDIR).mkdir(exist_ok=True)
        (path / FIGURES_SUBDIR).mkdir(exist_ok=True)
        self._project_dir = path

    def close(self) -> None:
        """Close the current project without deleting any files."""
        self._project_dir = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def next_results_path(self, stem: str = "run", suffix: str = ".h5") -> Path:
        """Return an auto-numbered results path that does not yet exist.

        Example: ``<project>/results/run_001.h5``

        Args:
            stem: Base name for the file.
            suffix: File extension including the dot.

        Returns:
            A :class:`pathlib.Path` inside ``results_dir`` that is free.

        Raises:
            RuntimeError: If no project is currently open.
        """
        if self._project_dir is None:
            raise RuntimeError("No project open.  Call open() or create() first.")
        rdir = self._project_dir / RESULTS_SUBDIR
        rdir.mkdir(exist_ok=True)
        for i in range(1, 10_000):
            candidate = rdir / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError("Could not find a free results filename after 9999 tries.")

    def next_stimulus_path(self, stem: str = "stimulus", suffix: str = ".pt") -> Path:
        """Return an auto-numbered stimulus path that does not yet exist.

        Args:
            stem: Base name for the file.
            suffix: File extension including the dot.

        Returns:
            A :class:`pathlib.Path` inside ``stimuli_dir`` that is free.

        Raises:
            RuntimeError: If no project is currently open.
        """
        if self._project_dir is None:
            raise RuntimeError("No project open.  Call open() or create() first.")
        sdir = self._project_dir / STIMULI_SUBDIR
        sdir.mkdir(exist_ok=True)
        for i in range(1, 10_000):
            candidate = sdir / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError("Could not find a free stimulus filename after 9999 tries.")

    def list_stimuli(self) -> list[Path]:
        """Return sorted list of stimulus files in the project's stimuli directory.

        Returns:
            Empty list if no project is open or directory is empty.
        """
        if self._project_dir is None:
            return []
        sdir = self._project_dir / STIMULI_SUBDIR
        if not sdir.is_dir():
            return []
        return sorted(sdir.iterdir())

    def list_results(self) -> list[Path]:
        """Return sorted list of result files in the project's results directory.

        Returns:
            Empty list if no project is open or directory is empty.
        """
        if self._project_dir is None:
            return []
        rdir = self._project_dir / RESULTS_SUBDIR
        if not rdir.is_dir():
            return []
        return sorted(rdir.iterdir())

    def __repr__(self) -> str:
        return (
            f"ExperimentManager(project_dir={self._project_dir!r}, "
            f"is_open={self.is_open})"
        )
