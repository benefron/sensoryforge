# Agent files: sources, targets and budget

This page is the map of every file a coding agent (Claude Code, Cursor, Codex and others)
reads in this repository: which files are sources, which are generated from them, who each
one is for, and how many there may be. Change the map here first, then the files.

## Two audiences

- **Maintainers** work on SensoryForge itself. Their agent files live in the repository only
  and are never shipped: `CLAUDE.md`'s Claude-specific notes, the path-scoped rules in
  `.claude/rules/`, the slash commands in `.claude/commands/`, the living ledger
  (`docs_root/LEDGER.md` and its hooks) and the roadmap in `docs_root/plan/`.
- **Users** use SensoryForge as a tool: design a world, build a data set, run, evaluate,
  extend. Their agent files ship inside the package (`sensoryforge/agents/`), so a
  `pip install` brings them, and work on a machine that has nothing else installed.

`AGENTS.md` at the repository root is the front door for both.

## Sources (written by hand)

| File | For | Read by | Task |
|---|---|---|---|
| `AGENTS.md` | everyone | any agent that reads a root instruction file | B2 |
| `CLAUDE.md` | maintainers | Claude Code; imports `AGENTS.md`, adds Claude-only notes | B2 |
| `sensoryforge/agents/skills/<name>/SKILL.md` | users | the neutral source: front matter `name` and `description`, markdown body (the open Agent Skills layout); every environment's files are generated from it ([export](../user_guide/agent_export.md)) | B1b, B6, B7, C2, C3 |
| `sensoryforge/agents/prompts/install.md`, `generic.md`, `instructions.md` | users | an agent the user gives them to: the first picks the environment's export command, the second adapts the neutral files to an environment with no export, the third builds the project's instruction file in the environment's own way | B8, B8b |
| `sensoryforge/agents/section.md` | users | the one short section SensoryForge adds to a project's instruction file (between marker lines; the rest of that file is the user's) | B8b |
| `sensoryforge/agents/skills/sensoryforge-glossary.md` | users | a copy of `docs/agents/glossary.md` that ships in the package; the docs page is the source and a test keeps the two identical | B6a |
| `sensoryforge/agents/style.py` | users | the working-style options, the setup questions and the brief every skill reads ([setup](../user_guide/agent_setup.md)) | B1b |
| `sensoryforge/agents/hook.py` | users who opt in | Claude Code's prompt hook, standard library only | B1b |
| `.claude/rules/*.md` | maintainers | Claude Code, when a matching file is touched | kept |
| `.claude/commands/*.md` | maintainers | Claude Code slash commands; `update-agent-docs` is the release-time check below | kept, trimmed; B10 |
| `docs/` | everyone | people; agents follow links to it | D |
| `docs/agents/glossary.md`, `docs/agents/task_index.md` | everyone | agents, through the link in `sensoryforge agents brief`; people | B4 |
| `docs_root/LEDGER.md` | maintainers | the session digest hook | kept |
| Tag lines at the end of a module docstring (`sensoryforge/`) | everyone | agents and people, by `grep` ([file headers](file_headers.md)) | B5 |

## Generated targets (never edited by hand)

Each target is written by one command and checked in CI against its source (B10; see [Keeping the agent files current](#keeping-the-agent-files-current)).

| Target | From | Command | Task |
|---|---|---|---|
| `.claude/skills/<name>/SKILL.md` and `.claude/skills/sensoryforge-glossary.md` in the user's project | `sensoryforge/agents/skills/` | `sensoryforge agents install [--dest DIR]` (`sensoryforge/agents/install.py`; overwrites its own files and touches nothing else) | B6a |
| `.github/skills/`, `.cursor/skills/`, `.agents/skills/` (same layout, same files) in the user's project | the same skills | `sensoryforge agents export --target copilot\|cursor\|codex` (`--check` reports files that differ; each target's folder is checked against its tool's documentation, dated in `FORMATS_CHECKED`) | B8 |
| `sensoryforge-agents/` (neutral files and a `README.md`) for an environment with no export | the same skills | `sensoryforge agents export --target generic`, then `sensoryforge agents prompt generic` | B8 |
| `CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md` (the marked section only; the rest is the user's) and `.cursor/rules/sensoryforge.mdc` (whole file) in the user's project | `sensoryforge/agents/section.md` | `sensoryforge agents instructions --target <env> --write` (`--check` reports a missing or stale section; the files each tool reads were checked against its documentation, dated in `INSTRUCTIONS_CHECKED`) | B8b |
| `llms.txt` | the docs navigation in `mkdocs.yml` | `python scripts/build_llms_txt.py` | B2 |
| `.sensoryforge/agents.yml` (the user's answers, not committed) | the setup questions | `sensoryforge agents setup` | B1b |

## What happens to today's files

| Today | State | Fate |
|---|---|---|
| `CLAUDE.md` (343 lines) | current, but held facts that also lived in `CODE_MAP.md` and the docs | done in B2: imports `AGENTS.md` and keeps Claude-only notes; the architecture notes moved to [Architecture Notes](architecture_notes.md) |
| `CODE_MAP.md` | partly repeated `CLAUDE.md`; named deleted GUI tabs | done in B2: folded into `AGENTS.md`'s table and removed |
| `refs/` (module, test and GUI maps) | last changed 2026-09-14; predates GUI v2 | not linked from `AGENTS.md`; replaced by D5's generated reference |
| `.claude/commands/` (5) | `review` and `remediate` name `.github/copilot-instructions.md`, which does not exist; `add-component` overlaps C2 | maintainer-only; the stale references now name `AGENTS.md` (B2); `add-component` retired when C2's skill replaces it |
| `.cursor/` (4 rules, 2 skills, 1 agent) | last changed 2026-09-14; repeats `CLAUDE.md` and predates plugins, worlds and evaluation | retired; B8 generates Cursor's skills from the packaged ones (`export --target cursor`) |
| `.claude/hooks/`, `.claude/settings.json`, `.claude/ledger.conf` | the living ledger's install | maintainer-only, owned by the ledger skill, not edited by hand |
| `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md` | for people | human-first; agents may read them, nothing points agents at them for facts |
| `docs_root/plan/`, `docs/development/reviews/` | roadmap and audits | human and maintainer only, never shipped |

## Rules for every agent file

1. **Name, do not copy.** A skill names the command (`sensoryforge <command> --help`), the
   registry (`sensoryforge list-components`) or the docs page that holds a fact. It never
   copies parameter lists, defaults, units or shapes, which go stale.
2. **Self-contained by default.** A skill uses only SensoryForge's own commands and skills
   and the agent's basic file and shell tools. It uses the user's other skills, MCP servers
   or settings only when the saved working style says so (B1b).
3. **Read the working style first.** Every skill starts by reading
   `sensoryforge agents brief` and follows it: planning depth, subagents, linear or
   parallel work, session and token limits, who reviews.
4. **Meaning, not string match.** A skill maps the user's words to SensoryForge's terms by
   meaning and the session's context, and asks when it is unclear.
5. **No invented domain facts.** Physics, models, parameters and units come from the user,
   from a linked source with a short report the user approves, or from reasoning the user
   approves. Every assumption is labelled with its origin.
6. **Neutral words in the core.** Tactile words appear only in tactile-specific skills or
   sections.
7. **A worked example that a test runs.** Every skill ends with a `## Worked example`
   section: the files it needs (a `yaml` or `python` block whose first line is
   `# file: NAME`) and the commands (a `bash` block, only `sensoryforge ...` and
   `python ...`), each command followed by `# prints: TEXT` lines and, when it should
   fail, `# exit: N`. `sensoryforge/testing/skill_examples.py` writes the files into an
   empty folder and runs the commands against the package that is under test;
   `tests/unit/test_agent_skills.py` does that for every skill. Other blocks (a plan, a
   sample of output) are not run. A command that is slow or machine-specific does not
   belong in an example.
8. **Nothing that exists only in a clone.** A shipped skill and the shipped glossary name no
   path under `tests/`, `docs/` or `.claude/`, no `pip install sensoryforge` (it is not on
   PyPI; say "install from source") and no `pip install -e`. A fact that lives in the docs
   is linked by its published address. The same test file checks this.
9. **Tested by a fresh agent.** Every skill has scenarios in `tests/agents/scenarios/`: a
   request in a user's words, run by a fresh agent from a plain install in an empty
   folder, scored against a rubric. A skill that changes has its scenarios re-run
   ([agent scenarios](agent_scenarios.md)).

## File-count budget

| Group | Limit |
|---|---|
| Hand-written agent files that ship or face users (`AGENTS.md`, `CLAUDE.md`, skills, the three prompts, the section) | **18** (16 used) |
| of which skills (`SKILL.md` files) | **14** |
| Extra reference files inside a skill | 0 by default; one only when the `SKILL.md` passes about 300 lines, and it counts towards the 18 |
| Maintainer-only (`.claude/rules/`, `.claude/commands/`) | 6 rules, 6 commands (`update-agent-docs` is the sixth; `add-component` retires in C2) |
| Generated targets | no limit, but each comes from one command and is checked in CI |

Shipped so far: setup (B1b); design-and-plan and design-world (B6a); build-dataset,
run-simulation and evaluate (B6b); diagnose and interpret-report (B6c); workflow (B6d).
The planned skills fit in 14: setup (B1b); design-and-plan, design-world, build-dataset,
run-simulation, evaluate, diagnose, interpret-report and the master skill, workflow (B6);
ground-assumptions (B7); extend-pipeline (every pipeline component kind), extend-world
(world registries and another modality's class kinds) and extend-evaluation (the
evaluation seams) (C2); and the plugin lifecycle (C3). A task that would pass a limit
merges skills or raises the limit here first, with the reason.

## Keeping the agent files current

Stale instructions mislead more than none, so two checks keep the files true to the code
(B10). The first is deterministic and runs in CI on every push; the second is an agent's
judgement and runs when a release is prepared.

**What CI fails on.** All of this runs in the ordinary test suite (`pytest -m "not gui"`),
so a rename or removal in the code fails the pull request that makes it:

| What a skill, prompt or section names | Test |
|---|---|
| a `sensoryforge` command, subcommand or flag (every command in a body is parsed by the real CLI parser) | `tests/unit/test_agent_skills.py` |
| a docs page or anchor, another skill, a path that exists only in a clone | `tests/unit/test_agent_skills.py` |
| a preset (`--preset`), a metric pack (`--packs`), a metric id (`core.frames.peak`), a world `kind:` or `dist:`, a `neuron_model:`, `filter_method:` or `innervation_method:` | `tests/unit/test_agent_names.py` (the names are read from the live registries by `sensoryforge.testing.agent_names`) |
| the worked example (it is run) | `tests/unit/test_agent_skills.py` |
| the glossary copy, `llms.txt`, the export targets and the instruction-file section | `tests/unit/test_front_door.py`, `test_agent_glossary.py`, `test_agent_export.py` |
| the file-count budget below | `tests/unit/test_agent_skills.py` |

A placeholder (`NAME`, `<pack>`) is not a name. A new kind of name a skill starts to use is
one more category in `agent_names.py`, with a test that a made-up name is caught.

**What an agent checks at a release.** A test cannot see a step that has become wrong, a
new option a step should ask about or a word that changed its meaning. The maintainer
command `/update-agent-docs` ([release checklist](release_checklist.md)) has an agent compare
the skills and prompts with what changed since the previous tag (`git diff` and `git log`),
group the changes by area, and propose the smallest edit for each affected line, citing the
commit. It edits nothing until the maintainer approves, and it re-runs the scenarios
([agent scenarios](agent_scenarios.md)) of every skill it changed. Skills name commands,
registries and pages instead of copying their facts (rule 1), so most releases need no edit.
