"""Claude Code prompt hook: point a plain-language request at the right skill.

Installed only when the user says yes to hooks in ``sensoryforge agents setup``; it
runs on every prompt (``UserPromptSubmit``) and prints a short index of the skills that
ship with the package, which Claude Code adds to the prompt's context. It is run by
path, needs only the standard library and never imports SensoryForge, so a prompt does
not wait for torch.

Example:
    $ python sensoryforge/agents/hook.py

Topic: agents
Tests: tests/unit/test_agents_style.py
Entry points: main, render, skill_index
Pinned by: docs/developer_guide/agent_files.md
Do not change without: standard library only: it runs as an agent's prompt hook on a
    machine with nothing else installed.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple


def skill_index(skills_dir: Path) -> List[Tuple[str, str]]:
    """Read ``name`` and ``description`` from each ``<skills_dir>/*/SKILL.md`` header.

    Returns:
        ``(name, description)`` pairs sorted by name; a skill with no name is skipped.
    """
    out = []
    for path in sorted(Path(skills_dir).glob("*/SKILL.md")):
        meta: Dict[str, str] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                key, sep, value = line.partition(":")
                if sep:
                    meta[key.strip()] = value.strip()
        if meta.get("name"):
            out.append((meta["name"], meta.get("description", "")))
    return sorted(out)


def render(index: List[Tuple[str, str]]) -> str:
    """The context text for the prompt; empty when no skill ships."""
    if not index:
        return ""
    lines = [
        "SensoryForge skills are available. If this request is SensoryForge work, use "
        "the skill whose purpose matches what the user means (not only their exact "
        "words), and ask when it is unclear:"
    ]
    for name, description in index:
        first = description.split(". ")[0].rstrip(".")
        lines.append(f"- {name}: {first}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Print the index for the skills next to this file."""
    sys.stdout.write(render(skill_index(Path(__file__).resolve().parent / "skills")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
