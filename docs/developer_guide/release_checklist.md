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
   `python scripts/build_reference.py` (run it in an environment with no plugin packages),
   `python -m sensoryforge.agents.reference docs/agents/skills_reference.md`, then
   `sensoryforge agents export --check --target <each>` in a scratch project.
6. **Check the docs:** `mkdocs build --strict` (it fails on a broken link or anchor inside the
   site) and `python scripts/check_docs_links.py` (the external links; also run monthly by the
   `docs-links` workflow, or by hand with its "Run workflow" button). A 403 or 429 is reported
   as "unverified", not broken: open the address once by hand.
7. **Changelog and version** (`CHANGELOG.md`, `pyproject.toml` and `sensoryforge/__init__.py`):
   move the `[Unreleased]` entries under a new version heading with its date, then tag.

## After tagging

The `deploy-docs` workflow publishes the site to
[benefron.github.io/sensoryforge](https://benefron.github.io/sensoryforge/) on every push to
`main`, so the docs go live when the release is merged, not at the tag. Open the site once
afterwards and follow a link from the home page to a page added in that release.
