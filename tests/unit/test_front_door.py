"""The agents' front door: AGENTS.md, CLAUDE.md and llms.txt (roadmap B2)."""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGENTS = (REPO / "AGENTS.md").read_text(encoding="utf-8")
AGENT_MAP = "docs/developer_guide/agent_files.md"

# Ten sample tasks in a user's words, and the file AGENTS.md must name for each, so a
# new agent reaches the right file in one hop.
SAMPLE_TASKS = [
    ("add a leaky integrate-and-fire neuron", "docs/developer_guide/extensibility.md"),
    (
        "my YAML's filter time constant is not what runs",
        "sensoryforge/config/defaults.py",
    ),
    ("let the world sampler draw a new shape", "docs/reference/world_contract.md"),
    ("store one more field in every run's output file", "sensoryforge/io/bundle.py"),
    ("the evaluation report misses a statistic", "sensoryforge/evaluate/"),
    ("change the populations screen of the app", "sensoryforge/gui/screens/"),
    (
        "retune the slowly adapting gain of a recipe",
        "scripts/calibrate_recipe_gains.py",
    ),
    ("was this already decided?", "docs_root/LEDGER.md"),
    (
        "change how the level-crossing converter steps",
        "docs/reference/converter_contract.md",
    ),
    ("make the simulation faster", "benchmarks/"),
    ("set how the agent works with me", "docs/user_guide/agent_setup.md"),
]


def _prose(text):
    """``text`` without fenced code blocks."""
    return re.sub(r"```.*?```", "", text, flags=re.S)


@pytest.mark.parametrize("task,path", SAMPLE_TASKS)
def test_each_sample_task_reaches_its_file_in_one_hop(task, path):
    assert f"`{path}`" in _prose(AGENTS), task
    assert (REPO / path).exists(), path


def test_every_path_agents_md_names_exists():
    spans = re.findall(r"`([^`]+)`", _prose(AGENTS))
    paths = [
        s
        for s in spans
        if re.fullmatch(r"[\w.\-/]+", s)
        and ("/" in s or re.search(r"\.(md|py|yml|txt)$", s))
        and not s.startswith(("http", "-"))
    ]
    assert len(paths) > 30
    missing = [p for p in paths if not (REPO / p).exists()]
    assert not missing


def test_agents_md_stays_short_and_claude_md_imports_it():
    assert len(AGENTS.splitlines()) <= 150
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "\n@AGENTS.md\n" in claude
    assert len(claude.splitlines()) <= 60


def test_llms_txt_is_generated_from_the_docs_nav():
    spec = importlib.util.spec_from_file_location(
        "build_llms_txt", REPO / "scripts" / "build_llms_txt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (REPO / "llms.txt").read_text(
        encoding="utf-8"
    ) == module.build(), "llms.txt is out of date: run python scripts/build_llms_txt.py"


EXPORT_FILES = ("docs/user_guide/agent_export.md", "tests/unit/test_agent_export.py")


def test_nothing_points_at_retired_agent_files():
    tracked = subprocess.run(
        ["git", "ls-files", "*.md", "*.py", "*.mdc"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    stale = ("CODE_MAP.md", ".github/copilot-instructions.md", ".cursor/")
    hits = [
        path
        for path in tracked
        if not path.startswith(("docs/development/", "docs_root/"))
        # this test, and the map that records what replaced each retired file
        and path not in ("tests/unit/test_front_door.py", AGENT_MAP)
        # the export of agent files names those paths as where a *user's* tool reads
        # them (this repository's own retired files are the ones guarded here)
        and not path.startswith("sensoryforge/agents/")
        and path not in EXPORT_FILES
        and any(s in (REPO / path).read_text(encoding="utf-8") for s in stale)
    ]
    assert not hits
