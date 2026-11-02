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
