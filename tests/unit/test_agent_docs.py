"""The agentic-workflow docs section: its pages and the generated skills reference.

The skills reference is generated from the packaged skills
(:mod:`sensoryforge.agents.reference`) and must equal them; the section is in the
site's navigation with every page it promises. The commands, skill names and registered
names on the section's pages are checked by ``test_agent_skills.py`` and
``test_agent_names.py``.
"""

import re
from pathlib import Path

import pytest

from sensoryforge.agents import reference

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "agents" / "skills_reference.md"
REGENERATE = "python -m sensoryforge.agents.reference docs/agents/skills_reference.md"


def test_the_skills_reference_page_is_generated_from_the_skills():
    assert (
        PAGE.read_text(encoding="utf-8") == reference.render_reference()
    ), f"the skills reference is behind the skills; run `{REGENERATE}`"


def test_the_reference_has_every_skill_with_its_worked_example():
    text = reference.render_reference()
    skills = dict(reference.packaged_skills())
    assert set(reference.CHAIN_ORDER) == set(skills), "CHAIN_ORDER names every skill"
    for name in skills:
        assert f"`{name}`" in text, name
    assert text.count("### Worked example") == len(skills)


def test_a_skill_without_a_worked_example_is_refused():
    skill = "---\nname: x\ndescription: d\n---\n\n# X\n\n## 1. Step\n\ntext\n"
    with pytest.raises(ValueError, match="Worked example"):
        reference.render_skill_section("x", skill)


def test_a_heading_inside_a_fence_is_not_a_section():
    skill = (
        "---\nname: x\ndescription: d\n---\n\n# X\n\n## 1. Write\n\n"
        "## Worked example\n\n```yaml\n# file: a.yml\n## not a heading\nk: 1\n```\n"
    )
    out = reference.render_skill_section("x", skill)
    assert "## not a heading\nk: 1" in out
    assert "**Steps.** Write." in out


def test_the_section_is_in_the_navigation_with_its_pages():
    nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    block = nav.split("  - Agentic Workflow:\n", 1)[1].split("\n\n", 1)[0]
    pages = re.findall(r":\s*(\S+\.md)\s*$", block, re.M)
    for page in [
        "agents/index.md",
        "user_guide/agent_setup.md",
        "user_guide/agent_skills.md",
        "agents/skills_reference.md",
        "agents/rules.md",
        "user_guide/agent_export.md",
        "agents/decisions.md",
        "agents/extending.md",
    ]:
        assert page in pages, page
        assert (REPO / "docs" / page).is_file(), page
