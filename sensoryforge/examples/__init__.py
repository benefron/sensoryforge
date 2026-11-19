"""Example configs that ship with the package, for what a preset cannot give.

A preset (:mod:`sensoryforge.presets`) is a whole sensor with no stimulus.
The files here are the small configs a preset cannot stand in for: a sweep
batch file, which holds its own sensor and the sweep, and a stimulus block
that is merged over a preset. ``sensoryforge examples copy <name>`` writes
one into the current folder to edit and run; nothing here is read in place.

Like the presets, the files are read through :mod:`importlib.resources`, so
they are found from an installed wheel with no clone of the repository. The
repository's own ``examples/`` folder (notebook, scripts, reference plugins)
is not part of the package.

Example:
    >>> from sensoryforge.examples import list_examples
    >>> list_examples()
    ['amplitude_sweep', 'moving_edge']

Topic: shipped example configs, `sensoryforge examples list|copy`
Tests: tests/unit/test_examples_package.py, tests/docs/test_docs_snippets.py
Entry points: list_examples, example_description, example_text, copy_example
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Dict, List

EXAMPLE_DIR = importlib.resources.files("sensoryforge.examples")

# One line each, shown by `sensoryforge examples list`.
_DESCRIPTIONS: Dict[str, str] = {
    "amplitude_sweep": (
        "A sweep batch file: a 40x40 sensor with SA and RA populations, run at "
        "three stimulus amplitudes (sensoryforge batch)."
    ),
    "moving_edge": (
        "A stimulus block to merge over a preset: an edge sweeping across the "
        "sensor (sensoryforge run --preset <preset> moving_edge.yml)."
    ),
}


def list_examples() -> List[str]:
    """Names of the shipped example configs, sorted.

    Returns:
        Each ``<name>.yml`` file in this package, without its suffix.
    """
    return sorted(
        entry.name[: -len(".yml")]
        for entry in EXAMPLE_DIR.iterdir()
        if entry.name.endswith(".yml")
    )


def example_description(name: str) -> str:
    """One-line description of an example (empty when it has none)."""
    return _DESCRIPTIONS.get(name, "")


def example_text(name: str) -> str:
    """The text of one shipped example config.

    Args:
        name: An example name from :func:`list_examples`.

    Returns:
        The YAML file's text, comments included.

    Raises:
        ValueError: ``name`` is not a shipped example.
    """
    if name not in list_examples():
        raise ValueError(
            f"no example named {name!r}; the examples are: "
            + ", ".join(list_examples())
        )
    return EXAMPLE_DIR.joinpath(f"{name}.yml").read_text(encoding="utf-8")


def copy_example(name: str, dest: Path | str = ".", force: bool = False) -> Path:
    """Write one example config into a folder, to edit and run.

    Args:
        name: An example name from :func:`list_examples`.
        dest: Folder to write ``<name>.yml`` into; created if missing.
        force: Replace a file of the same name that is already there.

    Returns:
        The path written.

    Raises:
        ValueError: ``name`` is not a shipped example.
        FileExistsError: The file exists and ``force`` is false.
    """
    text = example_text(name)
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.yml"
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to replace it")
    target.write_text(text, encoding="utf-8")
    return target
