"""The agent-scenario harness and its scenario files, without running an agent (B9).

These run in every test run: the scenario files are well formed, every skill that has
scenarios has every kind, the rubric is scored correctly from a hand-written transcript,
and the agent's command line keeps its tools an allowlist. Building the throwaway
install is marked ``slow``; the live runs are in ``test_scenarios_live.py``.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.agents import harness as h

SCENARIOS = h.load_scenarios()
SHIPPED = sorted(p.parent.name for p in h.SKILLS.glob("*/SKILL.md"))
BY_SKILL = {name: [s for s in SCENARIOS if s.skill == name] for name in SHIPPED}

#: Skills whose scenarios a later row writes (B9c: grounding and setup). The set only
#: shrinks.
PENDING = {
    "sensoryforge-ground-assumptions",
    "sensoryforge-setup",
}


# ------------------------------------------------------------------ the scenario files


def test_pending_names_only_shipped_skills_without_scenarios():
    assert PENDING <= set(SHIPPED)
    assert not [name for name in PENDING if BY_SKILL[name]], "move it out of PENDING"


@pytest.mark.parametrize("name", SHIPPED)
def test_every_skill_has_several_scenarios_of_every_kind(name):
    if name in PENDING:
        pytest.skip("scenarios come in a later row")
    kinds = {s.kind for s in BY_SKILL[name]}
    assert kinds == set(h.KINDS), f"{name} lacks {set(h.KINDS) - kinds}"
    assert len(BY_SKILL[name]) >= len(h.KINDS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_scenario_checks_that_its_own_skill_was_reached(scenario):
    skills = [c["skill"] for c in scenario.rubric() if h.check_kind(c) == "skill"]
    assert scenario.skill in skills
    assert all((h.SKILLS / name / "SKILL.md").is_file() for name in skills)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_prompt_is_in_a_users_words_not_the_skills_name(scenario):
    """Skills are found by meaning, so a prompt never names one."""
    for text in [scenario.prompt, *scenario.replies]:
        assert "sensoryforge-" not in text
        assert "skill" not in text.lower()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_scenarios_patterns_compile_and_its_commands_are_sensoryforge_or_python(
    scenario,
):
    for check in scenario.rubric():
        kind = h.check_kind(check)
        if kind in ("ran", "ran_ok", "not_ran", "answer", "not_answer"):
            re.compile(check[kind])
        if kind == "order":
            assert len(check[kind]) >= 2
            for pattern in check[kind]:
                re.compile(pattern)
        if kind == "file" and "valid" in check:
            assert check["valid"][0] in ("sensoryforge", "python")
        if kind == "judge":
            assert len(check[kind]) > 40, "a judge statement says what must be true"


def test_a_malformed_scenario_is_refused(tmp_path):
    good = {"id": "x.a", "kind": "normal", "prompt": "Hello."}
    cases = [
        ({**good, "kind": "odd"}, "kind"),
        ({**good, "checks": [{"ran": "a", "answer": "b"}]}, "exactly one"),
        ({**good, "style": {"colour": "red"}}, "style keys"),
        ({**good, "setup": [["rm", "-rf", "x"]]}, "only sensoryforge or python"),
        ({**good, "colour": "red"}, "unknown keys"),
    ]
    for raw, message in cases:
        (tmp_path / "s.yml").write_text(
            json.dumps({"skill": "sensoryforge-setup", "scenarios": [raw]})
        )
        with pytest.raises(ValueError, match=message):
            h.load_scenarios(tmp_path)
    (tmp_path / "s.yml").write_text(
        json.dumps({"skill": "no-such-skill", "scenarios": []})
    )
    with pytest.raises(ValueError, match="shipped skill"):
        h.load_scenarios(tmp_path)
    (tmp_path / "s.yml").write_text(
        json.dumps({"skill": "sensoryforge-setup", "scenarios": [good, good]})
    )
    with pytest.raises(ValueError, match="repeat"):
        h.load_scenarios(tmp_path)


def test_shared_files_are_filled_in(tmp_path):
    (tmp_path / "s.yml").write_text(
        json.dumps(
            {
                "skill": "sensoryforge-setup",
                "files": {"w": "world: {}\n"},
                "scenarios": [
                    {
                        "id": "x.a",
                        "kind": "normal",
                        "prompt": "Hi.",
                        "files": {"world.yml": "@w"},
                    }
                ],
            }
        )
    )
    assert h.load_scenarios(tmp_path)[0].files == {"world.yml": "world: {}\n"}


# ------------------------------------------------------------------ transcripts, scores


def _stream(*records):
    return [json.dumps(r) for r in records]


def _turn(text, *calls):
    """One turn's stream: each call is (name, input, result, is_error)."""
    records = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "s1",
            "model": "m",
            "skills": ["sensoryforge-design-world"],
            "plugins": [],
            "mcp_servers": [],
        }
    ]
    for i, (name, given, result, error) in enumerate(calls):
        records.append(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"t{i}",
                            "name": name,
                            "input": given,
                        }
                    ]
                },
            }
        )
        records.append(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i}",
                            "content": result,
                            "is_error": error,
                        }
                    ]
                },
            }
        )
    records.append(
        {
            "type": "result",
            "subtype": "success",
            "result": text,
            "total_cost_usd": 0.25,
            "is_error": False,
            "permission_denials": [{"tool_name": "Bash"}],
        }
    )
    return _stream(*records)


def _transcript():
    transcript = h.Transcript()
    h.parse_stream(
        [
            "not json",
            *_turn(
                "Which lower bound do you want?",
                ("Skill", {"skill": "sensoryforge-design-world"}, "loaded", False),
                ("Bash", {"command": "sensoryforge agents brief"}, "style", False),
                (
                    "Bash",
                    {"command": "sensoryforge validate --json --kind world w.yml"},
                    "Exit code 1\nsigma_mm",
                    True,
                ),
                ("Read", {"file_path": "/x/tests/unit/test_world.py"}, "...", False),
            ),
        ],
        0,
        transcript,
    )
    h.parse_stream(
        _turn(
            "Done: world_v2.yml validates.",
            (
                "Bash",
                {"command": "sensoryforge validate --json --kind world world_v2.yml"},
                '"valid": true',
                False,
            ),
        ),
        1,
        transcript,
    )
    return transcript


def test_a_stream_is_parsed_into_calls_answers_and_costs():
    t = _transcript()
    assert [c.name for c in t.calls] == ["Skill", "Bash", "Bash", "Read", "Bash"]
    assert t.calls[2].is_error and "sigma_mm" in t.calls[2].result
    assert t.answers == [
        "Which lower bound do you want?",
        "Done: world_v2.yml validates.",
    ]
    assert t.cost_usd == pytest.approx(0.5)
    assert len(t.denials) == 2 and t.session_id == "s1" and not t.errors
    missing = h.Transcript()
    h.parse_stream(["{}"], 0, missing)
    assert missing.answers == [""] and "no result record" in missing.errors[0]


def _scenario(*checks, skip=("ran", "no_repo_paths")):
    return h.Scenario(
        id="t.x",
        skill="sensoryforge-design-world",
        kind="normal",
        prompt="Fix my world.",
        replies=["Use 0.1."],
        checks=list(checks),
        skip_defaults=list(skip),
    )


def _scores(scenario, tmp_path, judge=None, before=None):
    workspace = h.Workspace(path=tmp_path, before=before or {})
    return h.score(scenario, _transcript(), workspace, judge=judge)


def test_command_answer_and_order_checks(tmp_path):
    results = _scores(
        _scenario(
            {"skill": "sensoryforge-design-world"},
            {"skill": "sensoryforge-build-dataset"},
            {"ran": "agents brief"},
            {"not_ran": "dataset build"},
            {"not_ran": "validate"},
            {"ran_ok": "validate .*w\\.yml$"},
            {"ran_ok": "validate --json --kind world world_v2"},
            {"order": ["agents brief", "validate"]},
            {"order": ["validate", "agents brief"]},
            {"answer": "world_v2\\.yml"},
            {"answer": "lower bound", "turn": 0},
            {"not_answer": "lower bound"},
            {"asks": True, "turn": 0},
            {"asks": True},
        ),
        tmp_path,
    )
    assert [r.passed for r in results] == [
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert "failed every time" in results[5].detail


def test_default_checks_catch_a_repository_path(tmp_path):
    results = _scores(_scenario(skip=()), tmp_path)
    assert [r.kind for r in results] == ["ran", "no_repo_paths"]
    assert results[0].passed and not results[1].passed
    assert "tests/unit" in results[1].detail
    assert not h.repo_only("cat https://benefron.github.io/sensoryforge/docs/x")
    assert not h.repo_only("sensoryforge describe --section world")
    assert h.repo_only("cat docs/index.md") and h.repo_only("pip install -e .")
    assert h.repo_only("ls /elsewhere/sensoryforge/tests/unit")


def test_file_checks_read_the_workspace(tmp_path):
    (tmp_path / "world.yml").write_text("hold_ms: {range: [30, 50]}\n")
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "runs" / "a" / "manifest.jsonl").write_text("{}\n")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "x.yml").write_text("ignored")
    results = _scores(
        _scenario(
            {"file": "world.yml", "contains": "30\\s*,\\s*50"},
            {"file": "world.yml", "contains": "99"},
            {"file": "runs/*/manifest.jsonl"},
            {"file": "world.yml", "new": True},
            {"no_file": "*.png"},
            {"no_file": "*.yml"},
            {"unchanged": "world.yml"},
            {"unchanged": "gone.yml"},
            {"file": "world.yml", "valid": ["sensoryforge", "validate", "{file}"]},
        ),
        tmp_path,
        before={"world.yml": "hold_ms: {range: [30, 50]}\n", "gone.yml": "x"},
    )
    assert [r.passed for r in results] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert results[5].detail == "world.yml"
    assert results[7].detail == "removed"
    assert "no install" in results[8].detail


def test_judge_lines_are_unscored_without_a_judge_and_read_its_first_word(tmp_path):
    scenario = _scenario(
        {"judge": "The agent asks the user before changing the range."}
    )
    assert _scores(scenario, tmp_path)[0].passed is None
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return "PASS: it asked for the lower bound.\nmore"

    result = _scores(scenario, tmp_path, judge=judge)[0]
    assert result.passed and result.detail == "PASS: it asked for the lower bound."
    assert "Fix my world." in prompts[0] and "Use 0.1." in prompts[0]
    assert "validate --json --kind world w.yml [error]" in prompts[0]
    assert "output: Exit code 1" in prompts[0]
    assert "Which lower bound" in prompts[0]
    assert _scores(scenario, tmp_path, judge=lambda p: "FAIL: no.")[0].passed is False


def test_a_result_scores_the_share_of_checks_and_passes_only_when_all_do(tmp_path):
    checks = [
        h.CheckResult("ran", {"ran": "a"}, True, ""),
        h.CheckResult("judge", {"judge": "b"}, None, ""),
        h.CheckResult("answer", {"answer": "c"}, False, ""),
    ]
    result = h.Result(_scenario(), checks, h.Transcript(), tmp_path, 1.0, "m")
    assert result.score == pytest.approx(0.5) and not result.passed
    result.checks = checks[:2]
    assert result.passed
    result.transcript.errors.append("turn 0: timed out")
    assert not result.passed
    path = h.write_report([result], tmp_path / "report")
    assert "| t.x | normal | m | no |" in path.read_text()
    data = json.loads((tmp_path / "report" / "results.json").read_text())
    assert data[0]["id"] == "t.x" and data[0]["transcript"]["errors"]


# ------------------------------------------------------------------ the agent's argv


def test_the_agent_gets_an_allowlist_and_nothing_of_the_users():
    runner = h.ClaudeRunner(model="sonnet", budget_usd=1.5)
    first = runner.argv("hi", "abc", True, ["WebSearch", "AskUserQuestion"])
    later = runner.argv("again", "abc", False, [])
    for argv in (first, later):
        assert argv[argv.index("--setting-sources") + 1] == "project"
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
        assert argv[argv.index("--append-system-prompt") + 1] == h.HARNESS_NOTE
        assert argv[argv.index("--permission-prompts") + 1] == "none"
        assert not [a for a in argv if "bypass" in a.lower() or "dangerously" in a]
        assert "Bash(sensoryforge *)" in argv and "Bash" not in argv
    assert first[first.index("--session-id") + 1] == "abc" and "--resume" not in first
    assert later[later.index("--resume") + 1] == "abc" and "--session-id" not in later
    disallowed = first[
        first.index("--disallowedTools") + 1 : first.index("--session-id")
    ]
    assert "WebSearch" in first and "AskUserQuestion" not in disallowed
    assert "AskUserQuestion" in later[later.index("--disallowedTools") :]


def test_the_environment_puts_the_install_first_and_drops_the_developers(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
    install = h.Install(
        root=tmp_path,
        venv=tmp_path / "venv",
        wheel=tmp_path / "w.whl",
        package=tmp_path,
    )
    env = install.env()
    assert env["PATH"].split(":")[0] == str(tmp_path / "venv" / "bin")
    assert "PYTHONPATH" not in env and "CONDA_PREFIX" not in env
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")


# ------------------------------------------------------------------ the install (slow)


@pytest.mark.slow
def test_a_workspace_is_a_plain_install_with_the_skills_and_the_style(tmp_path):
    pytest.importorskip("wheel")
    install = h.build_install(tmp_path / "root")
    assert install.venv.resolve() in install.package.parents
    scenario = h.Scenario(
        id="t.ws",
        skill="sensoryforge-setup",
        kind="normal",
        prompt="Hi.",
        style={"pace": "auto", "planning": "none"},
        files={"inputs/world.yml": "world: {}\n"},
        setup=[["python", "-c", "open('made.txt', 'w').write('ok')"]],
    )
    ws = h.prepare_workspace(scenario, install, tmp_path / "ws")
    skills = sorted(
        p.parent.name for p in (ws.path / ".claude" / "skills").glob("*/SKILL.md")
    )
    assert skills == SHIPPED
    assert (ws.path / "inputs" / "world.yml").read_text() == "world: {}\n"
    assert (ws.path / "made.txt").read_text() == "ok"
    brief = h._run_setup(["sensoryforge", "agents", "brief"], ws.path, install)
    assert "carry on" in brief.stdout.lower() or "automatic" in brief.stdout.lower()
    assert str(h.REPO) not in brief.stdout
    assert not (ws.path / "tests").exists() and not (ws.path / "docs").exists()
