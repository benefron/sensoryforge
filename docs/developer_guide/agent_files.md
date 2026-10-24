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
| `sensoryforge/agents/skills/<name>/SKILL.md` | users | Claude Code format; other environments through B8's exports | B1b, B6, B7, C2, C3 |
| `sensoryforge/agents/style.py` | users | the working-style options, the setup questions and the brief every skill reads ([setup](../user_guide/agent_setup.md)) | B1b |
| `sensoryforge/agents/hook.py` | users who opt in | Claude Code's prompt hook, standard library only | B1b |
| `.claude/rules/*.md` | maintainers | Claude Code, when a matching file is touched | kept |
| `.claude/commands/*.md` | maintainers | Claude Code slash commands | kept, trimmed |
| `docs/` | everyone | people; agents follow links to it | D |
| `docs/agents/glossary.md`, `docs/agents/task_index.md` | everyone | agents, through the link in `sensoryforge agents brief`; people | B4 |
| `docs_root/LEDGER.md` | maintainers | the session digest hook | kept |

## Generated targets (never edited by hand)

Each target is written by one command and checked in CI against its source (B10).

| Target | From | Command | Task |
|---|---|---|---|
| `.claude/skills/<name>/` in the user's project | `sensoryforge/agents/skills/` | `sensoryforge agents install` | B6a |
| Files for other environments, and one generic prompt for unknown ones | the same skills | `sensoryforge agents install --target <env>` | B8 |
| `llms.txt` | the docs navigation in `mkdocs.yml` | `python scripts/build_llms_txt.py` | B2 |
| `.sensoryforge/agents.yml` (the user's answers, not committed) | the setup questions | `sensoryforge agents setup` | B1b |

## What happens to today's files

| Today | State | Fate |
|---|---|---|
| `CLAUDE.md` (343 lines) | current, but held facts that also lived in `CODE_MAP.md` and the docs | done in B2: imports `AGENTS.md` and keeps Claude-only notes; the architecture notes moved to [Architecture Notes](architecture_notes.md) |
| `CODE_MAP.md` | partly repeated `CLAUDE.md`; named deleted GUI tabs | done in B2: folded into `AGENTS.md`'s table and removed |
| `refs/` (module, test and GUI maps) | last changed 2026-09-14; predates GUI v2 | not linked from `AGENTS.md`; replaced by D5's generated reference |
| `.claude/commands/` (5) | `review` and `remediate` name `.github/copilot-instructions.md`, which does not exist; `add-component` overlaps C2 | maintainer-only; the stale references now name `AGENTS.md` (B2); `add-component` retired when C2's skill replaces it |
| `.cursor/` (4 rules, 2 skills, 1 agent) | last changed 2026-09-14; repeats `CLAUDE.md` and predates plugins, worlds and evaluation | retired now; B8 generates Cursor's files from the skills |
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

## File-count budget

| Group | Limit |
|---|---|
| Hand-written agent files that ship or face users (`AGENTS.md`, `CLAUDE.md`, skills, B8's install prompt) | **18** |
| of which skills (`SKILL.md` files) | **14** |
| Extra reference files inside a skill | 0 by default; one only when the `SKILL.md` passes about 300 lines, and it counts towards the 18 |
| Maintainer-only (`.claude/rules/`, `.claude/commands/`) | 6 rules, 5 commands |
| Generated targets | no limit, but each comes from one command and is checked in CI |

The planned skills fit in 14: setup (B1b); design-and-plan, design-world, build-dataset,
run-simulation, evaluate, diagnose, interpret-report and the master skill (B6);
ground-assumptions (B7); extend-pipeline (every pipeline component kind), extend-world
(world registries and another modality's class kinds) and extend-evaluation (the
evaluation seams) (C2); and the plugin lifecycle (C3). A task that would pass a limit
merges skills or raises the limit here first, with the reason.
