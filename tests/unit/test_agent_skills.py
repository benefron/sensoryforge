"""The skills that ship with SensoryForge: their rules, install and examples.

Every skill is a file an agent follows on a user's machine, so each one is held to the
rules of ``docs/developer_guide/agent_files.md``, and each one's worked example is run
for real (roadmap B6).
"""

import re
import sys
from pathlib import Path

import pytest

from sensoryforge.agents import install as ins
from sensoryforge.agents import style
from sensoryforge.testing import skill_examples as ex

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "sensoryforge" / "agents" / "skills"
SKILLS = sorted(SKILL_DIR.glob("*/SKILL.md"))
NAMES = {p.parent.name for p in SKILLS}


def _meta(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", path
    meta = {}
    for line in lines[1 : lines.index("---", 1)]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


# ------------------------------------------------------------------ the rules


def test_the_core_skills_ship():
    assert {
        "sensoryforge-setup",
        "sensoryforge-design-and-plan",
        "sensoryforge-design-world",
    } <= NAMES


def test_the_skills_stay_within_the_file_budget():
    assert len(SKILLS) <= 14  # docs/developer_guide/agent_files.md
    extra = [
        p
        for p in SKILL_DIR.rglob("*")
        if p.is_file() and p.name not in ("SKILL.md", ins.GLOSSARY_NAME)
        if "__pycache__" not in p.parts
    ]
    assert not extra, extra


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_is_well_formed(path):
    text = path.read_text(encoding="utf-8")
    meta = _meta(path)
    assert meta["name"] == path.parent.name
    assert meta["description"].count(". ") >= 1, "the hook lists the first sentence"
    assert 40 <= len(meta["description"].split(". ")[0]) <= 200
    assert (
        len(text.splitlines()) <= 300
    ), "past 300 lines a skill needs a reference file"
    assert "sensoryforge agents brief" in text, "every skill reads the working style"
    assert "## Worked example" in text


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_uses_nothing_that_exists_only_in_a_clone(path):
    text = re.sub(r"https?://\S+", "", path.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w./-])(tests|docs|\.claude)/", text), path.parent.name
    assert not re.search(r"pip install\s+sensoryforge\b", text, re.I)
    assert "pip install -e" not in text


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_names_only_skills_that_exist(path):
    text = path.read_text(encoding="utf-8")
    named = set(re.findall(r"\bsensoryforge-[a-z][a-z-]*[a-z]\b", text))
    named -= {ins.GLOSSARY_NAME[: -len(".md")]}
    assert named <= NAMES, sorted(named - NAMES)


# --------------------------------------------------------------- the harness


def test_the_harness_reads_files_commands_and_expectations():
    text = (
        "# S\n\n## Worked example\n\n```yaml\n# file: a.yml\nx: 1\n```\n\n"
        "```bash\nsensoryforge describe \\\n    --section world\n# prints: shapes\n"
        "# exit: 0\n```\n\n```markdown\nnot run\n```\n\n"
        "## Next\n\n```bash\nrm -rf x\n```\n"
    )
    steps = ex.parse_example(text)
    assert [s.kind for s in steps] == ["file", "command"]
    assert steps[0].name == "a.yml" and steps[0].content == "x: 1\n"
    assert steps[1].argv == ["sensoryforge", "describe", "--section", "world"]
    assert steps[1].prints == ["shapes"] and steps[1].exit_code == 0
    assert ex.parse_example("# no example here\n") == []


def test_the_harness_refuses_a_command_that_is_not_sensoryforge_or_python():
    with pytest.raises(ValueError, match="only sensoryforge and python"):
        ex.parse_example("## Worked example\n```bash\nrm -rf x\n```\n")


def test_the_harness_reports_a_missing_output_and_a_wrong_exit(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "## Worked example\n```bash\nsensoryforge list-presets\n"
        "# prints: no such text anywhere\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not print"):
        ex.run_example(skill, tmp_path)
    skill.write_text(
        "## Worked example\n```bash\nsensoryforge list-presets\n# exit: 1\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exited 0"):
        ex.run_example(skill, tmp_path)


# ------------------------------------------------------------------ the example


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_the_worked_example_of_every_skill_runs(path, tmp_path):
    results = ex.run_example(path, tmp_path)
    assert results, "an example runs at least one command"


# ------------------------------------------------------------------- the install


def test_the_glossary_in_the_package_is_the_docs_page():
    page = (REPO / "docs" / "agents" / "glossary.md").read_bytes()
    assert ins.packaged_glossary().read_bytes() == page, (
        "docs/agents/glossary.md changed: copy it to "
        "sensoryforge/agents/skills/sensoryforge-glossary.md"
    )


def test_the_package_data_includes_the_glossary_and_the_skills():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"agents/skills/*.md"' in text and '"agents/skills/*/*.md"' in text


def test_install_copies_every_skill_and_the_glossary_and_is_repeatable(tmp_path):
    first = ins.install(tmp_path)
    base = tmp_path / ".claude" / "skills"
    assert {s for _, s in first} == {"added"}
    assert (base / ins.GLOSSARY_NAME).read_bytes() == (
        ins.packaged_glossary().read_bytes()
    )
    for name, path in ins.packaged_skills():
        assert (base / name / "SKILL.md").read_bytes() == path.read_bytes()
    assert {s for _, s in ins.install(tmp_path)} == {"unchanged"}
    edited = base / "sensoryforge-setup" / "SKILL.md"
    edited.write_text("changed by hand", encoding="utf-8")
    states = dict(ins.install(tmp_path))
    assert states[edited] == "updated"
    assert (
        edited.read_bytes()
        == dict(ins.packaged_skills())["sensoryforge-setup"].read_bytes()
    )


def test_install_leaves_the_users_own_skills_alone(tmp_path):
    mine = tmp_path / ".claude" / "skills" / "my-skill" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("mine", encoding="utf-8")
    ins.install(tmp_path)
    assert mine.read_text(encoding="utf-8") == "mine"


def test_install_refuses_a_folder_that_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        ins.install(tmp_path / "missing")


def test_the_brief_points_at_the_project_copy_first_then_the_package(tmp_path):
    assert ins.glossary_path(tmp_path) == ins.packaged_glossary()
    ins.install(tmp_path)
    sub = tmp_path / "work" / "deeper"
    sub.mkdir(parents=True)
    installed = tmp_path / ".claude" / "skills" / ins.GLOSSARY_NAME
    assert ins.glossary_path(sub) == installed.resolve()
    text = style.render_brief(style.WorkingStyle(), None, glossary=installed)
    assert str(installed) in text and "agents/glossary" in text
    assert str(ins.packaged_glossary()) in style.render_brief(
        style.WorkingStyle(), None, glossary=ins.packaged_glossary()
    )


def test_the_agents_install_command_writes_into_the_folder_asked_for(
    tmp_path, monkeypatch
):
    from sensoryforge.cli import main

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["sensoryforge", "agents", "install", *args])
        return main()

    assert run("--dest", str(tmp_path)) == 0
    assert (tmp_path / ".claude" / "skills" / ins.GLOSSARY_NAME).is_file()
    assert run("--dest", str(tmp_path / "missing")) == 1
