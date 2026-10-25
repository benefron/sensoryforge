"""Copy the skills and the glossary into a project where an agent will find them.

``sensoryforge agents install`` writes every skill that ships in
``sensoryforge/agents/skills/<name>/SKILL.md`` to ``<project>/.claude/skills/<name>/``
and the glossary beside them as ``<project>/.claude/skills/sensoryforge-glossary.md``
(Claude Code reads that folder; B8 adds the other environments). The copies are
generated: running the command again overwrites them with the installed release's
files, and nothing else in the folder is touched. :func:`glossary_path` is what
``sensoryforge agents brief`` points at, the project's copy first.

The module needs only the standard library, so it works before torch is imported.

Example:
    >>> "sensoryforge-setup" in [name for name, _ in packaged_skills()]
    True

Topic: agents
Tests: tests/unit/test_agent_skills.py
Entry points: install, packaged_skills, packaged_glossary, glossary_path,
    SKILLS_RELPATH, GLOSSARY_NAME
Pinned by: docs/developer_guide/agent_files.md
Do not change without: the glossary sits beside the skill folders in the package and
    in the project, so a link from a skill to ``../sensoryforge-glossary.md`` works in
    both; install writes only SensoryForge's own files and never edits the user's.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

SKILLS_RELPATH = Path(".claude") / "skills"
"""Where Claude Code looks for a project's skills, relative to the project folder."""

GLOSSARY_NAME = "sensoryforge-glossary.md"
"""The glossary's file name, beside the skill folders (package and project)."""

_PACKAGED = Path(__file__).resolve().parent / "skills"


def packaged_skills() -> List[Tuple[str, Path]]:
    """The skills that ship with this release.

    Returns:
        ``(name, SKILL.md path)`` pairs sorted by name.
    """
    return sorted((p.parent.name, p) for p in _PACKAGED.glob("*/SKILL.md"))


def packaged_glossary() -> Path:
    """The glossary that ships with this release (a copy of the docs page)."""
    return _PACKAGED / GLOSSARY_NAME


def glossary_path(start: Optional[Path] = None) -> Path:
    """The glossary an agent should read: the project's copy, else the packaged one.

    Args:
        start: Where to start looking (default the current folder); the search walks up
            through its parents for ``.claude/skills/sensoryforge-glossary.md``.

    Returns:
        The nearest installed copy, or the copy inside the installed package, which
        always exists.
    """
    here = (Path(start) if start else Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / SKILLS_RELPATH / GLOSSARY_NAME
        if candidate.is_file():
            return candidate
    return packaged_glossary()


def install(project: Path) -> List[Tuple[Path, str]]:
    """Write the skills and the glossary into ``project``.

    Args:
        project: The project folder; ``.claude/skills/`` is created under it.

    Returns:
        ``(path, state)`` for each file, ``state`` being ``"added"``, ``"updated"`` or
        ``"unchanged"``.

    Raises:
        FileNotFoundError: If ``project`` is not an existing folder.
    """
    project = Path(project)
    if not project.is_dir():
        raise FileNotFoundError(f"{project} is not a folder")
    base = project / SKILLS_RELPATH
    sources = [(Path(name) / "SKILL.md", path) for name, path in packaged_skills()]
    sources.append((Path(GLOSSARY_NAME), packaged_glossary()))
    out: List[Tuple[Path, str]] = []
    for relpath, source in sources:
        target = base / relpath
        data = source.read_bytes()
        if not target.is_file():
            state = "added"
        elif target.read_bytes() == data:
            state = "unchanged"
        else:
            state = "updated"
        if state != "unchanged":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        out.append((target, state))
    return out
