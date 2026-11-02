"""Every agent-skill scenario, run by a fresh agent (roadmap B9).

Opt-in: each test is a real agent run that costs usage and takes a minute or more, so
the tests are marked ``agents`` and skipped unless asked for::

    pytest tests/agents --run-agents -k world
    SF_AGENT_MODEL=haiku SF_AGENT_OUT=agent-runs pytest -m agents tests/agents

``SF_AGENT_MODEL`` is the agent's model (default ``sonnet``), ``SF_AGENT_JUDGE`` the
judge's (default ``haiku``; ``none`` leaves judge lines unscored) and ``SF_AGENT_OUT``
the folder that keeps ``results.json``, ``scores.md`` and every transcript (default: the
test's temporary folder). ``python tests/agents/run.py`` does the same with options.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.agents import harness as h

pytestmark = [pytest.mark.agents, pytest.mark.slow]

RESULTS = []


@pytest.fixture(scope="session")
def out(tmp_path_factory):
    folder = Path(
        os.environ.get("SF_AGENT_OUT") or tmp_path_factory.mktemp("agent-report")
    )
    yield folder
    if RESULTS:
        print(f"\nagent scenario scores: {h.write_report(RESULTS, folder)}")


@pytest.fixture(scope="session")
def install(tmp_path_factory):
    if shutil.which("claude") is None:
        pytest.skip("the claude command is not installed")
    return h.build_install(tmp_path_factory.mktemp("agent-install"))


@pytest.mark.parametrize("scenario", h.load_scenarios(), ids=lambda s: s.id)
def test_a_fresh_agent_meets_the_rubric(scenario, install, out):
    runner = h.ClaudeRunner(model=os.environ.get("SF_AGENT_MODEL", "sonnet"))
    judge_model = os.environ.get("SF_AGENT_JUDGE", "haiku")
    judge = None if judge_model == "none" else h.claude_judge(judge_model)
    result = h.run_scenario(scenario, install, runner, judge, out)
    RESULTS.append(result)
    failed = [
        f"{c.kind} {c.spec[c.kind]!r}: {c.detail}"
        for c in result.checks
        if c.passed is False
    ]
    assert result.passed, "\n".join(result.transcript.errors + failed)
