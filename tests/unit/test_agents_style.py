"""The agent skills' working style: setup, saved answers, brief and hook (B1b)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.agents import style as ws
from sensoryforge.agents.hook import render, skill_index
from sensoryforge.cli import cmd_agents, create_parser

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "sensoryforge" / "agents" / "skills"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project folder as the current folder, with no inherited config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ws.CONFIG_ENV, raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    return tmp_path


def _cli(*argv):
    return cmd_agents(create_parser().parse_args(["agents", *argv]))


def test_no_saved_file_gives_the_self_contained_defaults(project, capsys):
    assert ws.find_config() is None
    assert ws.load_style(None) == ws.WorkingStyle()
    assert _cli("brief") == 0
    out = capsys.readouterr().out
    assert "defaults; nothing saved yet" in out
    assert "Tools: self-contained" in out
    assert "do not start subagents" in out
    assert "one step at a time" in out


def test_setup_with_no_answers_saves_the_defaults(project):
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return ""

    style = ws.ask_style({}, ask, lambda _: None, "unknown", lambda: ["skill:x"])
    assert style == ws.WorkingStyle()
    assert not any("hook" in p.lower() for p in asked)  # not Claude Code: never offered

    def eof(prompt):
        raise EOFError

    assert (
        ws.ask_style({}, eof, lambda _: None, "claude-code", list) == ws.WorkingStyle()
    )


def test_setup_without_input_writes_the_defaults(project):
    assert _cli("setup", "--no-input") == 0
    saved = project / ws.CONFIG_RELPATH
    assert ws.load_style(saved) == ws.WorkingStyle()
    assert ws.find_config() == saved.resolve()


@pytest.mark.parametrize(
    "key,choice",
    [(o.key, c) for o in ws.OPTIONS if o.choices for c in o.choices],
)
def test_every_saved_choice_changes_what_the_skills_are_told(
    project, capsys, key, choice
):
    flag = "--" + key.replace("_", "-")
    assert _cli("setup", "--no-input", flag, choice) == 0
    capsys.readouterr()
    assert _cli("brief") == 0
    out = capsys.readouterr().out
    opt = next(o for o in ws.OPTIONS if o.key == key)
    assert opt.lines[choice] in out
    for other in set(opt.choices) - {choice}:
        assert opt.lines[other] not in out


def test_integration_and_budget_change_the_brief(project, capsys):
    argv = ["--no-input", "--integrate", "yes", "--tool", "mcp:papers"]
    assert _cli("setup", *argv, "--session-tokens", "150000") == 0
    capsys.readouterr()
    _cli("brief")
    out = capsys.readouterr().out
    assert "integration on" in out and "mcp:papers" in out
    assert "self-contained" not in out
    assert "about 150000 tokens" in out


def test_a_rerun_keeps_saved_answers_it_was_not_given(project):
    _cli("setup", "--no-input", "--subagents", "any", "--order", "parallel")
    _cli("setup", "--no-input", "--planning", "full")
    style = ws.load_style(project / ws.CONFIG_RELPATH)
    assert (style.subagents, style.order, style.planning) == ("any", "parallel", "full")
    _cli("setup", "--no-input", "--reset")
    assert ws.load_style(project / ws.CONFIG_RELPATH) == ws.WorkingStyle()


def test_the_search_walks_up_and_the_environment_variable_wins(project, monkeypatch):
    ws.save_style(ws.WorkingStyle(order="parallel"), project / ws.CONFIG_RELPATH)
    deep = project / "a" / "b"
    deep.mkdir(parents=True)
    assert ws.find_config(deep) == (project / ws.CONFIG_RELPATH).resolve()
    other = project / "elsewhere.yml"
    monkeypatch.setenv(ws.CONFIG_ENV, str(other))
    assert ws.find_config(deep) == other


def test_questions_retry_a_bad_answer_and_take_found_tools(project):
    answers = iter(["yes", "", "sideways", "full"])
    lines = []
    style = ws.ask_style(
        {},
        lambda _: next(answers, ""),
        lines.append,
        "unknown",
        lambda: ["mcp:papers", "skill:notes"],
    )
    assert style.integrate is True
    assert style.tools == ("mcp:papers", "skill:notes")
    assert style.planning == "full"
    assert any("planning must be one of" in line for line in lines)


@pytest.mark.parametrize(
    "text,match",
    [
        ("colour: blue\n", "unknown working-style key"),
        ("planning: always\n", "planning must be one of"),
        ("session_tokens: 0\n", "positive integer"),
        ("- a\n- b\n", "mapping"),
    ],
)
def test_a_bad_file_is_refused(project, capsys, text, match):
    path = project / ws.CONFIG_RELPATH
    path.parent.mkdir()
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        ws.load_style(path)
    assert _cli("brief") == 1
    assert match in capsys.readouterr().err


def test_discover_tools_reads_names_only(tmp_path):
    root, home = tmp_path / "proj", tmp_path / "home"
    (root / ".claude" / "skills" / "plots").mkdir(parents=True)
    (root / ".claude" / "skills" / "plots" / "SKILL.md").write_text("x")
    (home / ".claude" / "skills" / "notes").mkdir(parents=True)
    (home / ".claude" / "skills" / "notes" / "SKILL.md").write_text("x")
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"db": {}}}))
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"papers": {}}}))
    assert ws.discover_tools(root, home) == [
        "mcp:db",
        "mcp:papers",
        "skill:notes",
        "skill:plots",
    ]


def test_hooks_are_refused_outside_claude_code(project, capsys):
    assert _cli("setup", "--no-input", "--hooks", "yes") == 1
    assert "only in Claude Code" in capsys.readouterr().err
    assert not (project / ".claude").exists()


def test_the_hook_is_added_beside_other_settings_and_removed_cleanly(project):
    settings = project / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    other = {"type": "command", "command": "echo mine"}
    original = {"model": "x", "hooks": {"UserPromptSubmit": [{"hooks": [other]}]}}
    settings.write_text(json.dumps(original))
    assert _cli("setup", "--no-input", "--hooks", "yes") == 0
    data = json.loads(settings.read_text())
    commands = [
        h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]
    ]
    assert "echo mine" in commands and any("hook.py" in c for c in commands)
    assert data["model"] == "x"
    assert ws.set_hook(settings, True) is False  # already there
    assert _cli("setup", "--no-input", "--hooks", "no") == 0
    assert json.loads(settings.read_text()) == original


def test_the_hook_prints_the_shipped_skills_without_importing_sensoryforge():
    script = REPO / "sensoryforge" / "agents" / "hook.py"
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')\n"
        "assert 'sensoryforge' not in sys.modules and 'torch' not in sys.modules\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code, str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "- sensoryforge-setup:" in out
    assert render([]) == ""


def test_every_shipped_skill_reads_the_brief_first_and_fits_the_budget():
    index = skill_index(SKILLS)
    paths = sorted(SKILLS.glob("*/SKILL.md"))
    assert len(index) == len(paths) >= 1
    assert len({name for name, _ in index}) == len(index)
    # docs/developer_guide/agent_files.md: at most 14 skills
    assert len(index) <= 14
    for path in paths:
        text = path.read_text()
        assert "sensoryforge agents brief" in text, path
        assert all(desc for _, desc in index)


def test_the_user_guide_documents_every_option():
    page = (REPO / "docs" / "user_guide" / "agent_setup.md").read_text()
    for opt in ws.OPTIONS:
        assert f"`{opt.key}`" in page, opt.key
        for choice in opt.choices or ():
            assert f"`{choice}`" in page, (opt.key, choice)


def _settings_after_setup(project, *argv):
    """Run setup with the hook on; return (exit code, the project's settings file)."""
    code = _cli("setup", "--no-input", "--hooks", "yes", *argv)
    return code, project / ".claude" / "settings.local.json"


def _in_project(tmp_path, monkeypatch):
    """A project folder (with .claude/) inside a parent that must stay untouched."""
    parent = tmp_path / "parent"
    proj = parent / "proj"
    (proj / ".claude").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.delenv(ws.CONFIG_ENV, raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    return parent, proj


def test_the_hook_goes_in_the_project_even_when_the_style_file_is_elsewhere(
    tmp_path, monkeypatch
):
    parent, proj = _in_project(tmp_path, monkeypatch)
    # (1) --path to a file that is not <project>/.sensoryforge/agents.yml
    style_file = proj / "style.yml"
    code, settings = _settings_after_setup(proj, "--path", str(style_file))
    assert code == 0 and settings.is_file()
    assert not (parent / ".claude").exists()
    settings.unlink()
    # (2) $SENSORYFORGE_AGENTS_CONFIG elsewhere
    elsewhere = tmp_path / "elsewhere" / "deep" / "agents.yml"
    monkeypatch.setenv(ws.CONFIG_ENV, str(elsewhere))
    code, settings = _settings_after_setup(proj)
    assert code == 0 and settings.is_file() and elsewhere.is_file()
    assert not (tmp_path / "elsewhere" / ".claude").exists()
    assert not (tmp_path / "elsewhere" / "deep" / ".claude").exists()
    settings.unlink()
    monkeypatch.delenv(ws.CONFIG_ENV)
    # (3) a style file found in a parent folder
    found = parent / ".sensoryforge" / "agents.yml"
    found.parent.mkdir()
    ws.save_style(ws.WorkingStyle(), found)
    code, settings = _settings_after_setup(proj)
    assert code == 0 and settings.is_file()
    assert not (parent / ".claude").exists()
    # --hooks no removes it from the project too, not from the parent
    assert _cli("setup", "--no-input", "--hooks", "no") == 0
    assert not settings.exists() or "hook.py" not in settings.read_text()


def test_dest_names_the_project_and_a_new_style_file_goes_there(tmp_path, monkeypatch):
    other = tmp_path / "other"
    (other / ".claude").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv(ws.CONFIG_ENV, raising=False)
    assert _cli("setup", "--no-input", "--hooks", "yes", "--dest", str(other)) == 0
    assert "hook.py" in (other / ".claude" / "settings.local.json").read_text()
    assert (other / ws.CONFIG_RELPATH).is_file()
    assert not (work / ".claude").exists()


BAD_SETTINGS = {
    "hooks is a list": {"hooks": []},
    "hooks is a string": {"hooks": "x"},
    "event is a dict": {"hooks": {"UserPromptSubmit": {"hooks": []}}},
    "group is a string": {"hooks": {"UserPromptSubmit": ["echo hi"]}},
    "group hooks is a dict": {"hooks": {"UserPromptSubmit": [{"hooks": {}}]}},
    "entry is a string": {"hooks": {"UserPromptSubmit": [{"hooks": ["echo hi"]}]}},
}


@pytest.mark.parametrize("name", sorted(BAD_SETTINGS))
def test_set_hook_refuses_a_shape_it_does_not_understand(project, capsys, name):
    settings = project / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    text = json.dumps(BAD_SETTINGS[name])
    settings.write_text(text)
    with pytest.raises(ValueError, match="settings.local.json"):
        ws.set_hook(settings, True)
    # the setup refuses before saving the style: both files stay as they were
    assert _cli("setup", "--no-input", "--hooks", "yes") == 1
    assert "settings.local.json" in capsys.readouterr().err
    assert settings.read_text() == text
    assert not (project / ws.CONFIG_RELPATH).exists()


def test_set_hook_keeps_a_user_group_with_an_empty_hooks_list(project):
    settings = project / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    mine = {"matcher": "x", "hooks": []}
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [mine]}}))
    assert ws.set_hook(settings, True) is True
    assert ws.set_hook(settings, False) is True
    assert json.loads(settings.read_text()) == {"hooks": {"UserPromptSubmit": [mine]}}
