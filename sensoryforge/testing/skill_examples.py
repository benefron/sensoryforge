"""Run the worked example of an agent skill, so its commands cannot rot (roadmap B6).

Every ``SKILL.md`` that ships has a ``## Worked example`` section. This module reads it
and runs it in a scratch folder, exactly as the skill tells an agent to:

* a ``yaml``, ``python`` or ``markdown`` block whose first line is ``# file: NAME`` is
  written to ``NAME``;
* a ``bash`` block is a list of commands, one per line (a trailing ``\\`` continues a
  line). Only ``sensoryforge ...`` and ``python ...`` are allowed, so an example cannot
  depend on the user's machine;
* a comment right after a command states what it must show: ``# prints: TEXT`` (the
  command's output contains TEXT) and ``# exit: N`` (the exit status, default 0);
* any other block (a plan, a sample of output) is illustration and is not run.

The commands run with ``sensoryforge`` resolved to the package that imported this
module, so a test in a worktree runs the worktree's code.

Example:
    >>> text = "## Worked example\\n```bash\\nsensoryforge list-presets\\n```\\n"
    >>> parse_example(text)[0].argv
    ['sensoryforge', 'list-presets']

Topic: testing
Tests: tests/unit/test_agent_skills.py
Entry points: Step, StepResult, parse_example, run_example, example_section
Do not change without: skills ship to users who have only SensoryForge installed, so an
    example runs only ``sensoryforge`` and ``python`` and reads nothing from the
    repository.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import sensoryforge

HEADING = "## Worked example"
FENCE = re.compile(r"^```(\w*)[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
FILE_LINE = re.compile(r"^#\s*file:\s*(\S+)\s*$")


@dataclass
class Step:
    """One thing the example does.

    Attributes:
        kind: ``"file"`` (write ``content`` to ``name``) or ``"command"``.
        name: The file's name relative to the scratch folder (files only).
        content: The file's text (files only).
        argv: The command split like a shell does, as written (commands only).
        prints: Substrings the command's output must contain.
        exit_code: The exit status the command must have.
    """

    kind: str
    name: str = ""
    content: str = ""
    argv: List[str] = field(default_factory=list)
    prints: List[str] = field(default_factory=list)
    exit_code: int = 0


@dataclass
class StepResult:
    """What running a command step gave (files have no result)."""

    step: Step
    returncode: int
    output: str


def example_section(text: str) -> Optional[str]:
    """The text of the ``## Worked example`` section, or ``None`` when there is none."""
    match = re.search(rf"^{re.escape(HEADING)}[ \t]*$", text, re.M)
    if match is None:
        return None
    rest = text[match.end() :]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def parse_example(text: str) -> List[Step]:
    """Read the steps of a skill's worked example.

    Args:
        text: The whole ``SKILL.md`` (or just its example section).

    Returns:
        The steps in order; empty when there is no example section.

    Raises:
        ValueError: On a command that is neither ``sensoryforge`` nor ``python``, or an
            expectation comment with nothing to attach to.
    """
    section = example_section(text)
    if section is None:
        return []
    steps: List[Step] = []
    for fence in FENCE.finditer(section):
        lang, body = fence.group(1), fence.group(2)
        lines = body.splitlines()
        first = FILE_LINE.match(lines[0]) if lines else None
        if lang in ("yaml", "python", "markdown") and first:
            steps.append(
                Step("file", name=first.group(1), content="\n".join(lines[1:]) + "\n")
            )
        elif lang == "bash":
            steps.extend(_commands(lines))
    return steps


def _commands(lines: List[str]) -> List[Step]:
    """The command steps of one ``bash`` block."""
    steps: List[Step] = []
    joined: List[str] = []
    pending = ""
    for raw in lines:
        line = raw.rstrip()
        if pending:
            line = pending + line.lstrip()
            pending = ""
        if line.endswith("\\"):
            pending = line[:-1].rstrip() + " "
            continue
        joined.append(line)
    for line in joined:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            note = stripped[1:].strip()
            if not steps or steps[-1].kind != "command":
                raise ValueError(f"expectation with no command before it: {line!r}")
            if note.startswith("prints:"):
                steps[-1].prints.append(note[len("prints:") :].strip())
            elif note.startswith("exit:"):
                steps[-1].exit_code = int(note[len("exit:") :])
            continue
        argv = shlex.split(stripped)
        if argv[0] not in ("sensoryforge", "python"):
            raise ValueError(
                f"an example runs only sensoryforge and python, not {argv[0]!r}"
            )
        steps.append(Step("command", argv=argv))
    return steps


def _environment() -> Dict[str, str]:
    """The environment of every command: the imported package first on the path."""
    root = str(Path(sensoryforge.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def run_example(
    skill_md: Path, workdir: Path, timeout: float = 300.0
) -> List[StepResult]:
    """Run a skill's worked example in ``workdir``.

    Args:
        skill_md: Path of the ``SKILL.md``.
        workdir: An empty scratch folder; files are written here and commands run here.
        timeout: Seconds allowed for each command.

    Returns:
        One result per command, in order.

    Raises:
        ValueError: If the skill has no worked example, a command does not give the
            exit status or the output the example states, or a file would be written
            outside ``workdir``.
    """
    steps = parse_example(Path(skill_md).read_text(encoding="utf-8"))
    if not any(s.kind == "command" for s in steps):
        raise ValueError(f"{skill_md}: no worked example with commands")
    workdir = Path(workdir).resolve()
    env = _environment()
    results: List[StepResult] = []
    for step in steps:
        if step.kind == "file":
            target = (workdir / step.name).resolve()
            if workdir not in target.parents:
                raise ValueError(f"{skill_md}: file {step.name!r} leaves the folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(step.content, encoding="utf-8")
            continue
        argv = list(step.argv)
        if argv[0] == "sensoryforge":
            argv = [sys.executable, "-m", "sensoryforge.cli", *argv[1:]]
        else:
            argv[0] = sys.executable
        done = subprocess.run(
            argv,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = done.stdout + done.stderr
        results.append(StepResult(step, done.returncode, output))
        shown = " ".join(step.argv)
        if done.returncode != step.exit_code:
            raise ValueError(
                f"{skill_md.parent.name}: `{shown}` exited {done.returncode}, the "
                f"example says {step.exit_code}:\n{output[-1500:]}"
            )
        for want in step.prints:
            if want not in output:
                raise ValueError(
                    f"{skill_md.parent.name}: `{shown}` did not print {want!r}:\n"
                    f"{output[-1500:]}"
                )
    return results
