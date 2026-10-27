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
4. **Offer the next step.** Ask whether to set how the skills work with the user, by running
   `sensoryforge agents setup` (it asks a few questions and saves the answers). Then say what
   the user can ask for next: to design a world, build a data set, run a simulation or
   evaluate the result.
```

`sensoryforge agents prompt` prints the same text for pasting.

## An environment that is not listed

`sensoryforge agents export --target generic` writes `sensoryforge-agents/`: an
`INDEX.md`, each skill as `<name>/SKILL.md`, and the glossary. Nothing reads that folder by
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
   - `INDEX.md`: every skill with the situation it is for;
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
4. **Write only your own files.** Do not edit the user's existing instructions, skills or
   settings. If a name is already taken, say so and ask.
5. **Check and report.** Open one converted file and confirm that the environment can see
   it (many environments list their loaded skills or rules). Tell the user which
   environment you took this to be, where you put the files, what format you used, and
   where in its documentation you found it. Then ask whether to set how the skills work
   with them, by running `sensoryforge agents setup`.

`sensoryforge agents export --check --target generic` lists neutral files that are missing
or out of date, and writes nothing. When SensoryForge is updated, run the export again and
convert again; the generated files are replaced, not merged.
```

The prompt writes no file of its own in a format it has not found documented: SensoryForge
only claims a format for the four tools above.

## For maintainers

The skills in `sensoryforge/agents/skills/` are the single source; every export is
generated from them by `sensoryforge/agents/install.py`. A new environment is one
`register_target(Target(...))` call there (its name, its folder, the page the folder was
checked against), plus the test that checks it. A test fails when the generated files
differ from the source, and another when a skill uses a front-matter field or a name the
open format does not allow, so a skill that one tool reads cannot break in another. See
[Agent files: sources, targets and budget](../developer_guide/agent_files.md).
