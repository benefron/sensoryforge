# Using the Skills in Another Agent

SensoryForge's [skills](agent_skills.md) are one set of files in one neutral layout, so they
are not tied to Claude Code. A skill is a folder holding a `SKILL.md`: front matter with a
`name` and a `description` (what it does and when to use it), then instructions in
markdown. This is the open [Agent Skills format](https://agentskills.io/specification);
GitHub Copilot, Cursor, OpenAI Codex and Claude Code each read it from a folder of their
own. `sensoryforge agents export` writes the skills and the [glossary](../agents/glossary.md)
into the folder your environment reads, and nothing else.

SensoryForge is not on PyPI yet, so install it from source first (see
[Installation](../getting_started/installation.md)).

## Export

In your project folder:

```bash
sensoryforge agents export --target cursor     # or claude, copilot, codex
```

| `--target` | Writes to | Format checked against |
|---|---|---|
| `claude` | `.claude/skills/` | [Claude Code: skills](https://code.claude.com/docs/en/skills) |
| `copilot` | `.github/skills/` | [GitHub Copilot: create skills](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-skills) |
| `cursor` | `.cursor/skills/` | [Cursor: agent skills](https://cursor.com/docs/skills) |
| `codex` | `.agents/skills/` | [OpenAI Codex: build skills](https://learn.chatgpt.com/docs/build-skills) |
| `generic` | `sensoryforge-agents/` | [Agent Skills specification](https://agentskills.io/specification) |

The folders were checked against those pages on 2026-10-27. Tools change their formats, so
if an environment does not show the skills after an export, check its page first and report
it; `sensoryforge agents export --check` tells you whether the files on disk are what this
release writes.

Without `--target` the command looks at the project folder (`.claude/`, `.cursor/`,
`.github/skills/`, `.agents/` or `.codex/`) and picks the environment if exactly one
shows; when none or several do, it stops and asks you to name one, because a wrong guess
would put files where nothing reads them. `--dest DIR` exports into another project folder.

Export for the tools you use, not for all of them. Some tools also read another tool's
folder (Copilot's documentation lists `.claude/skills/` and `.agents/skills/`; Cursor's lists
`.agents/skills/`, `.claude/skills/` and `.codex/skills/`), so exporting for two of them
can make one tool find the same skill twice. What it does then is that tool's own
behaviour (Codex, for one, does not merge skills that share a name).

Every file begins (after its front matter) with a "generated, do not edit" header. Run the
command again after upgrading SensoryForge: it replaces its own files and touches nothing
else in the folder, so your own skills, rules and instructions stay as they are.
`--check` writes nothing; it lists each file as `current`, `stale` (edited by hand or from
an older release) or `missing`, and exits 1 if any is not current.

## Let the agent do it

Give your agent this prompt. It picks the right command for the environment it is running
in, and does not guess when it is unsure:

```text
You are an AI coding agent. The user wants SensoryForge's agent skills available in the
environment you run in. SensoryForge ships them as one neutral set of skill files and
writes each environment's native copy with one command. Follow these steps, and ask the
user whenever a step is unclear instead of guessing.

1. **Check SensoryForge is installed.** Run `sensoryforge agents brief`. If the command is
   not found, stop and tell the user: SensoryForge is not installed, and until it is on PyPI
   the only install is from source (https://benefron.github.io/sensoryforge/getting_started/installation/).
2. **Say which environment you are.** Answer from what you are, not from the folders you
   see. Pick the row that is you:

   | You are | Run, in the user's project folder |
   |---|---|
   | Claude Code | `sensoryforge agents export --target claude` |
   | GitHub Copilot (agent mode in VS Code, Copilot CLI, the cloud agent) | `sensoryforge agents export --target copilot` |
   | Cursor | `sensoryforge agents export --target cursor` |
   | OpenAI Codex | `sensoryforge agents export --target codex` |
   | Anything else, or not sure | Do not run a row. Read `sensoryforge agents prompt generic` and follow it. |

   Add `--check` to a row's command to list files that are missing or have changed since
   they were written; it writes nothing.
3. **Report what was written.** The command prints each file as added, updated or unchanged.
   Tell the user the folder and the count. It writes only SensoryForge's own files and
   never edits the user's own instructions, skills or settings. If your environment shows
   its loaded skills, check that the `sensoryforge-` skills appear; if they do not, say what
   you see and ask the user what to do rather than working around it.
4. **Offer the next steps.** Ask whether to set how the skills work with the user, by running
   `sensoryforge agents setup` (it asks a few questions and saves the answers). Ask whether
   to build or update the project's instruction file (`CLAUDE.md`, `AGENTS.md`,
   `.github/copilot-instructions.md` or a Cursor rule), which tells you about SensoryForge at
   the start of every session: if yes, follow `sensoryforge agents prompt instructions`. Then
   say what the user can ask for next: to design a world, build a data set, run a simulation
   or evaluate the result.
```

`sensoryforge agents prompt` prints the same text for pasting.

## An environment that is not listed

`sensoryforge agents export --target generic` writes `sensoryforge-agents/`: an
`README.md` (what to read and in what order, then every skill), each skill as
`<name>/SKILL.md`, and the glossary. Nothing reads that folder by
itself. Give your agent this prompt (`sensoryforge agents prompt generic` prints it); it
looks up where your environment keeps reusable instructions and in what format, asks you
when it cannot find out, and writes the skills there in that format:

```text
You are an AI coding agent in an environment SensoryForge has no ready-made export for.
SensoryForge's agent skills are plain files in one neutral layout; your job is to put them
where this environment reads reusable instructions, in the format it uses. Ask the user
whenever a step is unclear. Never guess a format.

1. **Write the neutral files.** Run `sensoryforge agents export --target generic` in the
   user's project folder. It writes a `sensoryforge-agents/` folder and nothing else:
   - `README.md`: what to read and in what order, and every skill with the situation it is
     for;
   - `<skill name>/SKILL.md`: one skill each, with front matter (`name`, `description`)
     and instructions in markdown;
   - `sensoryforge-glossary.md`: the words the skills use, linked from them as
     `../sensoryforge-glossary.md`.
2. **Find this environment's format in its own documentation.** Look for where it keeps
   reusable instructions (skills, rules, prompts, custom commands or an instruction file),
   which file names and front matter it expects, and whether a project folder or a user
   folder is the right place. Use the environment's documentation or the user's answer,
   never memory alone. If you cannot find it, tell the user what you looked for and ask
   them where the files should go.
3. **Convert, keeping the content.**
   - If the environment reads the same layout (a folder per skill holding `SKILL.md` with
     `name` and `description`), copy the folders as they are, with the glossary beside them
     so the link keeps working.
   - Otherwise write one file per skill in the environment's format, keep the skill's name
     and its `description` as the line that says when to use it, keep the body as written,
     and keep a link to the glossary working. Do not summarise, merge or reword the
     instructions, and do not change a command in them.
   - Keep the "generated, do not edit" header on every file you write.
4. **Write only your own files.** In this step do not edit the user's existing instructions,
   skills or settings (the project's instruction file has its own step, below, and the
   user's agreement). If a name is already taken, say so and ask.
5. **Check and report.** Open one converted file and confirm that the environment can see
   it (many environments list their loaded skills or rules). Tell the user which
   environment you took this to be, where you put the files, what format you used, and
   where in its documentation you found it. Then offer two more steps: to set how the skills
   work with them, by running `sensoryforge agents setup`, and to build or update the
   project's instruction file, by following `sensoryforge agents prompt instructions`.

`sensoryforge agents export --check --target generic` lists neutral files that are missing
or out of date, and writes nothing. When SensoryForge is updated, run the export again and
convert again; the generated files are replaced, not merged.
```

The prompt writes no file of its own in a format it has not found documented: SensoryForge
only claims a format for the four tools above.

## The project's instruction file

Skills are loaded when a request matches them. The file your tool reads at the start of
every session (`CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`, a Cursor rule) is
different: it is always in the agent's context, and it is **yours**. It holds what is true of
your project (how to build and test it, its conventions), and nothing there says that
SensoryForge's skills exist, that every skill starts with `sensoryforge agents brief`, or
that the agent must not invent physics. SensoryForge supplies that as one short section and
leaves the rest of the file to you and your agent.

What each tool's own documentation says, checked on 2026-10-28:

| Tool | Reads | Its own way to draft the file | Page |
|---|---|---|---|
| Claude Code | `CLAUDE.md` (or `.claude/CLAUDE.md`); `AGENTS.md` directly, or by `@AGENTS.md` from a `CLAUDE.md` | the `/init` command | [memory](https://code.claude.com/docs/en/memory) |
| GitHub Copilot | `.github/copilot-instructions.md`; also `.github/instructions/*.instructions.md` (with `applyTo`) and `AGENTS.md` | on github.com, the cloud agent's onboarding prompt; no command in an editor is documented | [repository instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions) |
| Cursor | `.cursor/rules/*.mdc` (front matter `description`, `globs`, `alwaysApply`); also `AGENTS.md` | the `/create-rule` command | [rules](https://cursor.com/docs/context/rules) |
| OpenAI Codex | `AGENTS.md` (or `AGENTS.override.md`), merged from the project root down, 32 KiB by default | none confirmed | [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) |
| any other | what its documentation says | what it offers | [AGENTS.md as an open format](https://agents.md) |

Two ways to use it. Give your agent this prompt, which tells it to build or update the file
the way its environment does, with the freedom to choose the format, to use the tool's own
drafting command where it has one, and to ask you for anything it cannot read from the
project:

```text
You are an AI coding agent. The user wants this project's instruction file (the file your
environment reads at the start of every session) to tell you about SensoryForge, without
losing or overwriting what is already in it. The file is the user's; SensoryForge supplies
one short section, and how the rest of the file is built is up to you and your
environment. Ask the user whenever a step is unclear, and never guess.

1. **Read what SensoryForge gives you.** Run `sensoryforge agents instructions --target generic`.
   It prints the section (between two `sensoryforge:begin` and `sensoryforge:end` markers)
   and writes nothing. Run `sensoryforge agents brief` too, so you know what the skills
   will follow. If the skills are not installed for your environment, give the user
   `sensoryforge agents prompt install` first.
2. **Find your environment's instruction file.** Use your environment's own documentation,
   or the user's answer, never memory alone. What SensoryForge has checked against each
   tool's own pages:

   | Environment | File | Its own way to draft one |
   |---|---|---|
   | Claude Code | `CLAUDE.md` (it also reads `AGENTS.md`) | the `/init` command |
   | GitHub Copilot | `.github/copilot-instructions.md` (it also reads `AGENTS.md`) | on github.com, the cloud agent's onboarding prompt; no command in an editor is documented |
   | Cursor | a rule in `.cursor/rules/` (it also reads `AGENTS.md`) | the `/create-rule` command |
   | OpenAI Codex | `AGENTS.md` | none confirmed |
   | Anything else | whatever its documentation says | whatever it offers |

   If the project already has an `AGENTS.md` that your environment reads, prefer that one
   shared file to a second file. If the environment has its own way to draft the file
   from the project, use it first.
3. **Build or update the file in your environment's way.** If the file does not exist,
   draft it from the project as your environment normally would (how to build and test,
   conventions), taking from the project only what you can read there and asking the user
   for the rest. If it exists, leave every line of it as it is. The format, headings and
   length of the project's own text are yours to choose; keep them short.
4. **Add the SensoryForge section, after the user agrees.** Show the user the section and the
   file it goes in. For `claude`, `copilot` and `codex` run
   `sensoryforge agents instructions --target <claude|copilot|codex> --write`; for `cursor`
   `--target cursor` writes SensoryForge's own rule file. The command adds the section between its
   markers, replaces only that block on a later run, and touches nothing else. For an
   environment with no command of its own, run
   `sensoryforge agents instructions --target generic --file FILE --write`, or, if
   the file needs a different format, copy the section's text into it unchanged and keep
   the two marker lines.
5. **Check and report.** Run the same command with `--check`; it exits 1 unless the section
   is current. Tell the user which file you changed or created, which lines are yours and
   which are SensoryForge's, and what you took from the project, and what you asked
   them for. If your environment can show what it loaded, confirm the section is there.
```

(`sensoryforge agents prompt instructions` prints it.) Or do the SensoryForge part yourself:

```bash
sensoryforge agents instructions --target codex            # shows the section and the file; writes nothing
sensoryforge agents instructions --target codex --write    # adds or refreshes the section
sensoryforge agents instructions --target codex --check    # exit 1 unless the section is there and current
sensoryforge agents instructions --target generic --file AGENTS.md --write   # an environment not in the table
```

Rules the command keeps. It writes nothing without `--write`. In `CLAUDE.md`,
`.github/copilot-instructions.md` and `AGENTS.md` the section sits between two marker lines
(`<!-- sensoryforge:begin ... -->` and `<!-- sensoryforge:end -->`); a new file gets only the
section, an existing file keeps every other byte, and a later run replaces only what is
between the markers. Damaged markers (one without the other, or two blocks) stop the
command and ask you to fix them by hand. For Cursor, SensoryForge owns one whole rule file,
`.cursor/rules/sensoryforge.mdc` (`alwaysApply: true`), and your other rules are not touched.
`--file FILE` puts the section in another file, for example an `AGENTS.md` that several tools
share.

## For maintainers

The skills in `sensoryforge/agents/skills/` are the single source; every export is
generated from them by `sensoryforge/agents/install.py`. The section for instruction files is
`sensoryforge/agents/section.md`. A new environment is one `register_target(Target(...))`
call there (its name, its skills folder, its instruction file, the pages they were checked
against, its own drafting command if the documentation gives one), plus the test that checks
it. A test fails when the generated files
differ from the source, and another when a skill uses a front-matter field or a name the
open format does not allow, so a skill that one tool reads cannot break in another. See
[Agent files: sources, targets and budget](../developer_guide/agent_files.md).
