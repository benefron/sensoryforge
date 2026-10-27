"""Write the skills and the glossary where each kind of agent will find them.

``sensoryforge agents export --target <env>`` writes every skill that ships in
``sensoryforge/agents/skills/<name>/SKILL.md`` to the folder that environment reads, and
the glossary beside them. The skills are one neutral source: a ``SKILL.md`` is a folder
with ``name`` and ``description`` front matter and a markdown body (the open Agent
Skills layout) that Claude Code, GitHub Copilot, Cursor and Codex each read from a
folder of their own (:data:`EXPORT_TARGETS`, each with the page it was checked against).
``generic`` writes the same files, plus an index, into one folder for an environment
this module does not know; ``sensoryforge agents prompt generic`` is the prompt that
turns them into that environment's format.

Every file written starts (after the front matter) with a "generated, do not edit"
header. The copies are replaced when the command runs again, and nothing else in the
folder is touched. :func:`check` reports files that differ from what this release would
write, without writing. :func:`glossary_path` is what ``sensoryforge agents brief``
points at, the project's copy first.

The module needs only the standard library, so it works before torch is imported.

Example:
    >>> "sensoryforge-setup" in [name for name, _ in packaged_skills()]
    True
    >>> sorted(EXPORT_TARGETS)
    ['claude', 'codex', 'copilot', 'cursor', 'generic']

Topic: agents
Tests: tests/unit/test_agent_skills.py, tests/unit/test_agent_export.py
Entry points: install, check, detect_targets, register_target, packaged_skills,
    packaged_glossary, packaged_prompt, glossary_path, EXPORT_TARGETS, Target,
    SKILLS_RELPATH, GLOSSARY_NAME
Pinned by: docs/developer_guide/agent_files.md, docs/user_guide/agent_export.md
Do not change without: the glossary sits beside the skill folders in the package and
    in every target, so a link from a skill to ``../sensoryforge-glossary.md`` works in
    all of them; export writes only SensoryForge's own files and never edits the user's;
    a target's folder is checked against its tool's own documentation, with the date.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

GLOSSARY_NAME = "sensoryforge-glossary.md"
"""The glossary's file name, beside the skill folders (package and project)."""

INDEX_NAME = "INDEX.md"
"""The generic target's index of the skills it wrote."""

_PACKAGED = Path(__file__).resolve().parent / "skills"
_PROMPTS = Path(__file__).resolve().parent / "prompts"

#: The date every target's folder was last checked against its tool's documentation.
FORMATS_CHECKED = "2026-10-27"


@dataclass(frozen=True)
class Target:
    """One environment's native place for skills.

    Attributes:
        name: The ``--target`` value.
        label: What a person calls the environment.
        skills_dir: Where it reads a project's skills, relative to the project folder;
            each skill is a ``<name>/SKILL.md`` folder inside it.
        markers: Paths (relative to the project) whose presence suggests the project
            already uses this environment; used only to pick a target for ``auto``.
        source: The documentation page ``skills_dir`` was checked against.
        index: Also write :data:`INDEX_NAME` (the generic target).
    """

    name: str
    label: str
    skills_dir: Path
    markers: Tuple[Path, ...] = ()
    source: str = ""
    index: bool = False


EXPORT_TARGETS: Dict[str, Target] = {}
"""Every environment ``agents export`` can write for, by ``--target`` name."""


def register_target(target: Target) -> Target:
    """Add an environment to :data:`EXPORT_TARGETS`.

    Raises:
        ValueError: If the name is taken, or the folder is absolute or leaves the
            project.
    """
    if target.name in EXPORT_TARGETS:
        raise ValueError(f"export target {target.name!r} is already registered")
    parts = Path(target.skills_dir).parts
    if Path(target.skills_dir).is_absolute() or ".." in parts or not parts:
        raise ValueError(f"{target.name}: skills_dir must stay inside the project")
    EXPORT_TARGETS[target.name] = target
    return target


register_target(
    Target(
        "claude",
        "Claude Code",
        Path(".claude") / "skills",
        (Path(".claude"),),
        "https://code.claude.com/docs/en/skills",
    )
)
register_target(
    Target(
        "copilot",
        "GitHub Copilot",
        Path(".github") / "skills",
        (Path(".github") / "skills", Path(".github") / "copilot-instructions.md"),
        "https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/"
        "create-skills",
    )
)
register_target(
    Target(
        "cursor",
        "Cursor",
        Path(".cursor") / "skills",
        (Path(".cursor"),),
        "https://cursor.com/docs/skills",
    )
)
register_target(
    Target(
        "codex",
        "OpenAI Codex",
        Path(".agents") / "skills",
        (Path(".agents"), Path(".codex")),
        "https://learn.chatgpt.com/docs/build-skills",
    )
)
register_target(
    Target(
        "generic",
        "any other environment",
        Path("sensoryforge-agents"),
        (),
        "https://agentskills.io/specification",
        index=True,
    )
)

SKILLS_RELPATH = EXPORT_TARGETS["claude"].skills_dir
"""Where Claude Code looks for a project's skills, relative to the project folder."""


def packaged_skills() -> List[Tuple[str, Path]]:
    """The skills that ship with this release.

    Returns:
        ``(name, SKILL.md path)`` pairs sorted by name.
    """
    return sorted((p.parent.name, p) for p in _PACKAGED.glob("*/SKILL.md"))


def packaged_glossary() -> Path:
    """The glossary that ships with this release (a copy of the docs page)."""
    return _PACKAGED / GLOSSARY_NAME


def packaged_prompt(name: str) -> Path:
    """One of the prompts that ship with this release.

    Args:
        name: ``"install"`` (picks the environment) or ``"generic"`` (for one this
            release does not know).

    Raises:
        ValueError: If there is no such prompt.
    """
    path = _PROMPTS / f"{name}.md"
    if not path.is_file():
        known = sorted(p.stem for p in _PROMPTS.glob("*.md"))
        raise ValueError(f"no prompt named {name!r}; the prompts are {known}")
    return path


def glossary_path(start: Optional[Path] = None) -> Path:
    """The glossary an agent should read: the project's copy, else the packaged one.

    Args:
        start: Where to start looking (default the current folder); the search walks up
            through its parents for ``<skills folder>/sensoryforge-glossary.md`` in
            any target's skills folder.

    Returns:
        The nearest exported copy, or the copy inside the installed package, which
        always exists.
    """
    here = (Path(start) if start else Path.cwd()).resolve()
    for folder in (here, *here.parents):
        for target in EXPORT_TARGETS.values():
            candidate = folder / target.skills_dir / GLOSSARY_NAME
            if candidate.is_file():
                return candidate
    return packaged_glossary()


def _header(target: Target) -> str:
    return (
        f"<!-- Generated by `sensoryforge agents export --target {target.name}` from "
        "SensoryForge's packaged skills. Do not edit: run the command again to "
        "refresh. -->"
    )


def render_skill(text: str, target: Target) -> str:
    """A skill as it is written for ``target``: the header goes after the front matter.

    Raises:
        ValueError: If ``text`` has no front matter (the file would not be a skill).
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("a skill must start with front matter (---)")
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            head, body = "".join(lines[: end + 1]), "".join(lines[end + 1 :])
            return f"{head}\n{_header(target)}\n{body}"
    raise ValueError("a skill's front matter is not closed (---)")


def render_glossary(text: str, target: Target) -> str:
    """The glossary as written for ``target``: the header first, then the page."""
    return f"{_header(target)}\n\n{text}"


def _description(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() == "description":
            return value.strip()
    return ""


def render_index(target: Target) -> str:
    """The generic target's index: each skill, its first sentence and its file."""
    lines = [
        _header(target),
        "",
        "# SensoryForge skills",
        "",
        "Each folder below holds one skill as a `SKILL.md`: front matter with a `name` "
        "and a `description` (what it does and when to use it), then instructions in "
        f"markdown. `{GLOSSARY_NAME}` defines the words the skills use. Every skill "
        "starts by running `sensoryforge agents brief`.",
        "",
        "| Skill | Use it when |",
        "|---|---|",
    ]
    for name, path in packaged_skills():
        first = _description(path).split(". ")[0].rstrip(".")
        lines.append(f"| [{name}]({name}/SKILL.md) | {first}. |")
    return "\n".join(lines) + "\n"


def planned_files(target: str = "claude") -> List[Tuple[Path, str]]:
    """Every file ``target`` gets, relative to the project, with its text.

    Raises:
        ValueError: If ``target`` is not in :data:`EXPORT_TARGETS`.
    """
    spec = _target(target)
    files = [
        (
            spec.skills_dir / name / "SKILL.md",
            render_skill(path.read_text(encoding="utf-8"), spec),
        )
        for name, path in packaged_skills()
    ]
    files.append(
        (
            spec.skills_dir / GLOSSARY_NAME,
            render_glossary(packaged_glossary().read_text(encoding="utf-8"), spec),
        )
    )
    if spec.index:
        files.append((spec.skills_dir / INDEX_NAME, render_index(spec)))
    return files


def _target(name: str) -> Target:
    try:
        return EXPORT_TARGETS[name]
    except KeyError:
        raise ValueError(
            f"unknown target {name!r}; the targets are {sorted(EXPORT_TARGETS)}"
        ) from None


def install(project: Path, target: str = "claude") -> List[Tuple[Path, str]]:
    """Write the skills and the glossary for ``target`` into ``project``.

    Args:
        project: The project folder; the target's skills folder is created under it.
        target: A name in :data:`EXPORT_TARGETS` (default ``"claude"``).

    Returns:
        ``(path, state)`` for each file, ``state`` being ``"added"``, ``"updated"`` or
        ``"unchanged"``.

    Raises:
        FileNotFoundError: If ``project`` is not an existing folder.
        ValueError: If ``target`` is not known.
    """
    project = Path(project)
    if not project.is_dir():
        raise FileNotFoundError(f"{project} is not a folder")
    out: List[Tuple[Path, str]] = []
    for relpath, text in planned_files(target):
        path = project / relpath
        data = text.encode("utf-8")
        if not path.is_file():
            state = "added"
        elif path.read_bytes() == data:
            state = "unchanged"
        else:
            state = "updated"
        if state != "unchanged":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        out.append((path, state))
    return out


def check(project: Path, target: str = "claude") -> List[Tuple[Path, str]]:
    """Compare what is in ``project`` with what this release would write; write nothing.

    Returns:
        ``(path, state)`` for each file, ``state`` being ``"current"``, ``"stale"``
        (the file differs, for instance edited by hand or from an older release) or
        ``"missing"``.
    """
    project = Path(project)
    out: List[Tuple[Path, str]] = []
    for relpath, text in planned_files(target):
        path = project / relpath
        if not path.is_file():
            state = "missing"
        elif path.read_bytes() == text.encode("utf-8"):
            state = "current"
        else:
            state = "stale"
        out.append((path, state))
    return out


def detect_targets(
    project: Path, environ: Optional[Mapping[str, str]] = None
) -> List[str]:
    """The environments ``project`` already seems to use, by the folders they read.

    Only documented folders count, plus Claude Code's own shell marker, so a project
    with none of them gives an empty list rather than a guess.

    Args:
        project: The project folder.
        environ: Environment variables (default the process's).

    Returns:
        Target names (never ``generic``) in registry order.
    """
    env = os.environ if environ is None else environ
    found = []
    for name, spec in EXPORT_TARGETS.items():
        hit = any((Path(project) / m).exists() for m in spec.markers)
        if name == "claude" and env.get("CLAUDECODE"):
            hit = True
        if hit:
            found.append(name)
    return found


def summarize(states: Sequence[Tuple[Path, str]]) -> str:
    """``"3 added, 8 unchanged"``: the states counted, in the order first seen."""
    counts: Dict[str, int] = {}
    for _, state in states:
        counts[state] = counts.get(state, 0) + 1
    return ", ".join(f"{n} {s}" for s, n in counts.items())
