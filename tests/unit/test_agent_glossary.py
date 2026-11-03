"""The glossary and the task index agents read (roadmap B4).

Every term the skills and the registries use must be defined once, and every file or
test the task index names must exist.
"""

import re
from pathlib import Path

import pytest

from sensoryforge import describe as dsc
from sensoryforge.agents import style
from sensoryforge.cli import create_parser

REPO = Path(__file__).resolve().parents[2]
GLOSSARY = REPO / "docs" / "agents" / "glossary.md"
INDEX = REPO / "docs" / "agents" / "task_index.md"
SKILLS = sorted((REPO / "sensoryforge" / "agents" / "skills").glob("*/SKILL.md"))


def _terms():
    """``{term: section}`` from the ``###`` headings and each ``*Also:*`` list."""
    terms, section = {}, ""
    text = GLOSSARY.read_text(encoding="utf-8")
    for block in re.split(r"(?m)^(?=#{2,3} )", text):
        if block.startswith("## "):
            section = block.splitlines()[0][3:].strip()
        elif block.startswith("### "):
            head, _, body = block.partition("\n")
            names = [n.strip() for n in head[4:].split(",")]
            also = re.search(r"\*Also:\*\s*(.+?)\.?\s*(?:\n\n|\Z)", body, re.S)
            if also:
                names += [" ".join(n.split()) for n in also.group(1).split(",")]
            for name in names:
                terms[name.lower()] = section
    return terms


TERMS = _terms()
TACTILE = "Tactile today, general by design"


def _in_glossary(word):
    word = word.lower().strip()
    return word in TERMS or (word.endswith("s") and word[:-1] in TERMS)


def test_terms_are_defined_once():
    headings = [
        line[4:].strip().lower()
        for line in GLOSSARY.read_text(encoding="utf-8").splitlines()
        if line.startswith("### ")
    ]
    assert len(headings) == len(set(headings))


def test_the_neutral_vocabulary_and_the_named_terms_are_defined():
    for word in (
        "sensor",
        "channel",
        "frame",
        "population",
        "draw",
        "entry",
        "bundle",
        "recipe",
        "world",
        "class",
        "receptor",
        "data set",
    ):
        assert _in_glossary(word), word
    assert TERMS["sensor"] != TACTILE and TERMS["receptor"] == TACTILE


def test_tactile_words_sit_only_in_the_marked_section():
    tactile = {t for t, s in TERMS.items() if s == TACTILE}
    assert {"receptor", "contact", "indentation", "pressure", "SA", "RA"} <= {
        t.upper() if t in ("sa", "ra") else t for t in tactile
    } | {"SA", "RA"}
    for neutral in ("sensor", "stimulus", "population", "draw", "entry", "bundle"):
        assert TERMS[neutral] != TACTILE


# What the code names, and the glossary term that defines it.
KIND_TERMS = {
    "neuron": "neuron model",
    "filter": "filter",
    "stimulus": "stimulus",
    "solver": "solver",
    "grid": "sensor array",
    "innervation": "receptive field",
    "processing": "processing layer",
    "metric_packs": "metric pack",
    "check_packs": "check pack",
    "declared_adapters": "declared adapter",
}
WORLD_TERMS = {
    "shapes": "shape",
    "patterns": "pattern",
    "modulations": "modulation",
    "distributions": "distribution",
    "class_kinds": "class",
}


def test_every_kind_the_registries_name_has_a_glossary_term():
    data = dsc.describe()
    named = set(data["components"]) | set(data["evaluation"])
    assert named == set(KIND_TERMS), "a new registry kind needs a glossary term"
    assert set(WORLD_TERMS) <= set(data["world"])
    for term in [*KIND_TERMS.values(), *WORLD_TERMS.values()]:
        assert _in_glossary(term), term


def test_every_shipped_term_for_a_split_is_defined():
    text = re.sub(r"[-_]", " ", GLOSSARY.read_text(encoding="utf-8"))
    for split in dsc.describe()["world"]["split_kinds"]:
        assert split.replace("_", " ") in text, split


def _plain_words(text):
    """Backticked tokens that are single plain words (not commands, flags or paths)."""
    words = set(re.findall(r"`([a-z][a-z_ -]*)`", text))
    return sorted(w for w in words if not w.startswith("sensoryforge "))


def _known_elsewhere():
    commands = set()
    for action in create_parser()._actions:
        commands |= set(getattr(action, "choices", None) or [])
    registered = {
        n.lower() for kind in dsc.describe()["components"].values() for n in kind
    }
    skills = {p.parent.name for p in SKILLS}
    return commands | registered | skills | {"yes", "no", "defaults"}


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_every_term_a_skill_uses_is_in_the_glossary(path):
    known = _known_elsewhere()
    missing = [
        w
        for w in _plain_words(path.read_text(encoding="utf-8"))
        if not _in_glossary(w) and w.lower() not in known
    ]
    assert (
        not missing
    ), f"{path.parent.name}: define in docs/agents/glossary.md: {missing}"


def test_the_brief_every_skill_reads_carries_the_meaning_rule():
    text = style.render_brief(style.WorkingStyle(), None)
    assert "by their meaning" in text and "never by matching strings" in text
    assert "agents/glossary" in text
    # The task index names tests/, docs/ and .claude/ paths that exist only in a
    # clone; a user's agent is not sent there (R-B6 finding 8).
    assert "task_index" not in text and "task index" not in text


# ------------------------------------------------------------------ task index


def _index_rows():
    rows = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] != "Task":
                rows.append(cells)
    return rows


def test_task_index_names_the_sample_tasks():
    tasks = {r[0].lower() for r in _index_rows()}
    for wanted in ("add a stimulus", "change a recipe or its gains", "read a bundle"):
        assert wanted in tasks, wanted


def test_task_index_rows_have_four_cells_and_files_and_a_read_first():
    for row in _index_rows():
        assert len(row) == 4, row
        assert row[1] and row[3], row


def test_every_path_the_task_index_names_exists():
    missing = []
    for row in _index_rows():
        for cell in row[1:]:
            for token in re.findall(r"`([^`]+)`", cell):
                if token.startswith("sensoryforge ") or " " in token or "<" in token:
                    continue
                if "/" not in token and not token.endswith((".py", ".md", ".yml")):
                    continue
                if not (REPO / token).exists():
                    missing.append(token)
    assert not missing, missing


def test_pages_are_in_the_docs_navigation():
    nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert "agents/glossary.md" in nav and "agents/task_index.md" in nav
