"""Every registered name an agent file mentions still exists (roadmap B10).

``tests/unit/test_agent_skills.py`` checks a skill's commands, flags and links; this
file checks the names inside them (a preset, a metric pack or id, a world shape, a
component) against the registries, so a rename in the code fails CI instead of leaving
a skill that tells an agent to use something that is gone.
"""

from collections import Counter
from pathlib import Path

import pytest

from sensoryforge.testing import agent_names as names

REPO = Path(__file__).resolve().parents[2]
AGENTS = REPO / "sensoryforge" / "agents"
FILES = sorted(
    [*AGENTS.glob("skills/*/SKILL.md"), *AGENTS.glob("prompts/*.md")]
    + [AGENTS / "section.md", AGENTS / "skills" / "sensoryforge-glossary.md"]
    + [REPO / "AGENTS.md", REPO / "docs" / "agents" / "glossary.md"]
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_registered_name_in_an_agent_file_exists(path):
    assert not names.missing_names(path.read_text(encoding="utf-8"))


def test_the_check_reads_names_in_every_category_it_claims():
    """If a pattern stopped matching, the test above would pass by finding nothing."""
    found = Counter(
        category
        for path in FILES
        for category, _ in names.named_things(path.read_text(encoding="utf-8"))
    )
    for category in ("preset", "pack", "metric", "world kind"):
        assert found[category] >= 2, (category, dict(found))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("sensoryforge batch --preset tactile_gone", "preset tactile_gone"),
        ("sensoryforge evaluate run --packs core,nonesuch", "pack nonesuch"),
        ("see `core.frames.nonesuch`", "metric core.frames.nonesuch"),
        ("layer: {shape: {kind: blob}}", "world kind blob"),
        ("axes: {x: {dist: bell}}", "dist bell"),
        ("neuron_model: Nonesuch", "neuron_model Nonesuch"),
    ],
)
def test_the_check_names_what_no_longer_exists(text, expected):
    assert [m for m in names.missing_names(text) if m.startswith(expected)]


@pytest.mark.parametrize(
    "text",
    [
        "sensoryforge batch --preset NAME",
        "sensoryforge evaluate run --packs core,<pack>",
        "sensoryforge evaluate run --packs core,tactile",
        "the family `tactile.episode.rise_time` and `core.frames.peak`",
        "layer: {shape: {kind: gaussian}}",
        "neuron_model: AdEx",
    ],
)
def test_the_check_accepts_placeholders_and_names_that_exist(text):
    assert names.missing_names(text) == []


def test_a_plugin_name_counts_once_it_is_registered():
    from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY

    class Toy(METRIC_PACK_REGISTRY.get_class("core")):
        pass

    names.known_names.cache_clear()
    try:
        METRIC_PACK_REGISTRY.register("toy_pack", Toy)
        assert names.missing_names("--packs toy_pack") == []
    finally:
        METRIC_PACK_REGISTRY._registry.pop("toy_pack", None)  # noqa: SLF001
        names.known_names.cache_clear()
