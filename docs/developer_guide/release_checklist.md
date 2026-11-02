# Release Checklist

The steps to take before a SensoryForge release is tagged. Each one names the command that
does it, so the list can be followed and checked. Tests, linting and the docs build are
the project's usual checks (see [Contributing](../contributing.md)); this page adds what
only a release needs.

## Before tagging

1. **Tests and lint are green** on the merge commit: `pytest -m "not gui"`, `pytest -m gui`,
   `black --check`, `flake8` (the CI workflow runs them).
2. **Update the agent files.** In Claude Code, run `/update-agent-docs` (it compares with
   the previous tag). An agent reads the skills, prompts and the instruction-file section
   against what changed in `sensoryforge/` and `docs/` and proposes the edits; a maintainer
   approves them. The deterministic half already runs in CI ([how](agent_files.md#keeping-the-agent-files-current)).
3. **Re-run the scenarios of every skill you changed**, and of the export prompts if
   `sensoryforge agents export` changed: `python tests/agents/run.py --skill NAME`. Compare
   with the baseline in [Agent Scenarios](agent_scenarios.md#baseline-scores).
4. **Re-read the dated documentation checks** (`FORMATS_CHECKED` and `INSTRUCTIONS_CHECKED`
   in `sensoryforge/agents/install.py`) against each tool's own pages, and change the date
   only after reading them.
5. **Regenerate what is generated:** `python scripts/build_llms_txt.py`,
   `python -m sensoryforge.agents.reference docs/agents/skills_reference.md`, then
   `sensoryforge agents export --check --target <each>` in a scratch project.
6. **Changelog and version** (`CHANGELOG.md`, `pyproject.toml`), then tag.

The rest of the release documentation (the published docs and the reference) will be
listed on this page as it is written.
