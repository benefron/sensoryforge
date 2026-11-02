Check the agent files (skills, prompts, the instruction-file section, AGENTS.md) against what changed since the last release or merge, and propose the edits that are needed.

$ARGUMENTS is the base to compare with: a tag or a commit (default: the latest tag, `git describe --tags --abbrev=0`). Run this at every release and after every merge to main. It proposes; the maintainer decides.

## What is already checked, so do not redo it

CI fails on its own when any of these breaks: a command or flag a skill shows (`tests/unit/test_agent_skills.py`), a docs link or anchor, a skill that names a skill that is gone, a registered name (`--preset`, `--packs`, a metric id, a world `kind:`/`dist:`, `neuron_model:` and the like; `tests/unit/test_agent_names.py`), the worked example of every skill, a stale glossary copy, `llms.txt` and the export targets, and the file-count budget. If one of those fails, fix the cause first; this command is for what a test cannot see.

## Steps

1. **List what changed.** `git diff --stat BASE..HEAD -- sensoryforge/ docs/` and `git log --oneline BASE..HEAD`. Group the changes by area: commands (`sensoryforge/cli.py`), worlds and data sets (`sensoryforge/world/`), running (`core/`, `io/`), evaluation (`evaluate/`), components and registries, presets, the working style (`agents/style.py`), the agent files themselves.
2. **Find the files each area touches.** For each area read the skills and prompts that cover it (`sensoryforge/agents/skills/*/SKILL.md`, `prompts/`, `section.md`, `AGENTS.md`, `docs/agents/glossary.md`). Skills name a command, a registry or a docs page instead of copying a fact (rule 1 of `docs/developer_guide/agent_files.md`), so most changes need no edit. Look for:
   - a step that is now wrong (the order of commands, a stop that moved, a default or a limit the skill states in words);
   - behaviour the skill does not know about: a new option the step should ask about, a new stop, a new kind of failure;
   - a word that changed meaning, or a new term the glossary lacks;
   - a dated documentation check that is old: `FORMATS_CHECKED` and `INSTRUCTIONS_CHECKED` in `sensoryforge/agents/install.py` (re-read each tool's own page before changing the date);
   - the Worked example, if the output it prints changed in wording only.
3. **Propose, do not rewrite.** For each file list: what changed in the code (with the commit), the line or step that is affected, the smallest edit, and why. A change that needs no edit gets no line. Never copy a parameter list, default, unit or shape into a skill; name where it lives.
4. **Apply only what the maintainer approves**, then run `pytest tests/unit/test_agent_skills.py tests/unit/test_agent_names.py tests/unit/test_agent_export.py -q` and, for each skill you edited, its scenarios: `python tests/agents/run.py --skill NAME` (it costs usage; say so first). Compare the scores with the last baseline in `docs/developer_guide/agent_scenarios.md`.
5. Commit with a ledger trailer: `Finding:` for what had drifted, `Decision:` for a choice about how a skill now behaves.

Never invent a domain fact while editing: a physical value, a model or a unit comes from the maintainer or a source they approve (design principle 3).
