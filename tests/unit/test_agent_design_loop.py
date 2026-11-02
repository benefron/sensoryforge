"""Agents do not invent physics (roadmap B7).

The rule is in the docs in plain words, the skills that write values hand an unknown
value to the grounding skill, and the three recorded design sessions in the docs obey
it: the agent asks before it writes a value, and every value it writes carries its
origin.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.testing import skill_examples as ex

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "user_guide" / "agent_design_loop.md"
SKILLS = REPO / "sensoryforge" / "agents" / "skills"
GROUNDING = "sensoryforge-ground-assumptions"

#: A line that sets a value in a world file, and the comment that gives its origin.
VALUE = re.compile(r"\b(range|value|weight):")
ORIGIN = re.compile(r"#\s*(you\b|source\b|reasoning\b|assumption: needs your OK)")
YAML_BLOCK = re.compile(r"^```yaml\n(.*?)^```", re.S | re.M)


def _sessions(text):
    parts = re.split(r"^## (?=Session \d)", text, flags=re.M)[1:]
    return {part.splitlines()[0]: part for part in parts}


def _without_origin(yaml_text):
    return [
        line.strip()
        for line in yaml_text.splitlines()
        if VALUE.search(line) and not ORIGIN.search(line)
    ]


def _first_agent_turn(session):
    turn = session.split("**Agent:**", 1)[1]
    return re.split(r"\*\*You:\*\*|^```", turn, maxsplit=1, flags=re.M)[0]


def _skill(name):
    return (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")


def test_the_rule_is_stated_plainly_in_the_docs():
    text = PAGE.read_text(encoding="utf-8")
    for phrase in (
        "does not invent physics",
        "asks you to describe it",
        "writes you a short report",
        "assumption: needs your OK",
        "exactly one of",
    ):
        assert phrase in text, phrase
    assert "agent_design_loop.md" in (REPO / "mkdocs.yml").read_text()
    assert (
        "agent_design_loop.md" in (REPO / "docs/user_guide/agent_skills.md").read_text()
    )
    assert "Do not invent domain facts" in (REPO / "AGENTS.md").read_text()


def test_three_sessions_are_recorded():
    assert len(_sessions(PAGE.read_text(encoding="utf-8"))) == 3


@pytest.mark.parametrize("title", list(_sessions(PAGE.read_text(encoding="utf-8"))))
def test_the_agent_asks_before_it_writes_a_value(title):
    session = _sessions(PAGE.read_text(encoding="utf-8"))[title]
    assert "?" in _first_agent_turn(session), "the agent's first turn is a question"
    assert "**Assumptions listed:**" in session, "every session ends by listing them"


@pytest.mark.parametrize("title", list(_sessions(PAGE.read_text(encoding="utf-8"))))
def test_every_value_the_agent_writes_carries_its_origin(title):
    session = _sessions(PAGE.read_text(encoding="utf-8"))[title]
    blocks = YAML_BLOCK.findall(session)
    assert blocks, "a session shows the file the agent wrote"
    assert not [line for b in blocks for line in _without_origin(b)]


def test_the_origin_check_catches_a_value_with_none():
    assert _without_origin("a: {range: [1, 2]}  # you") == []
    assert _without_origin("a: {range: [1, 2]}  # assumption: needs your OK") == []
    assert _without_origin("a: {range: [1, 2]}  # plausible") != []
    assert _without_origin("b: {value: 40}") != []


def test_a_skill_that_writes_values_hands_unknowns_to_the_grounding_skill():
    for name in (
        "sensoryforge-design-world",
        "sensoryforge-design-and-plan",
        "sensoryforge-workflow",
    ):
        assert GROUNDING in _skill(name), name


def test_the_grounding_skill_offers_both_ways_and_forbids_memory():
    text = " ".join(_skill(GROUNDING).split())
    for phrase in (
        "They describe it",
        "You look for sources",
        "no value from memory",
        "Open every source you cite",
        "ask the user for sources",
    ):
        assert phrase.lower() in text.lower(), phrase


def _run(folder, *args):
    content = next(
        s.content
        for s in ex.parse_example(_skill(GROUNDING))
        if s.name == "assumptions.py"
    )
    (folder / "assumptions.py").write_text(content, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "assumptions.py", *args],
        cwd=folder,
        env=ex._environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )


TABLE = (
    "| Value | Unit | Used in | Origin | Status |\n|---|---|---|---|---|\n"
    "| a | mm | w.yml | {origin} | {status} |\n"
)


def test_the_audit_flags_open_marks_open_rows_and_rows_with_no_origin(tmp_path):
    (tmp_path / "w.yml").write_text("a: 1  # assumption: needs your OK\n")
    (tmp_path / "clean.yml").write_text("a: 1  # you\n")
    (tmp_path / "ok.md").write_text(
        TABLE.format(origin="you (chat)", status="approved")
    )
    (tmp_path / "open.md").write_text(
        TABLE.format(origin="assumption: needs your OK", status="open")
    )
    (tmp_path / "none.md").write_text(
        TABLE.format(origin="plausible", status="approved")
    )
    assert _run(tmp_path, "w.yml").returncode == 1
    assert _run(tmp_path, "clean.yml", "ok.md").returncode == 0
    assert _run(tmp_path, "clean.yml", "open.md").returncode == 1
    done = _run(tmp_path, "clean.yml", "none.md")
    assert done.returncode == 1 and "no origin for 'a'" in done.stdout
