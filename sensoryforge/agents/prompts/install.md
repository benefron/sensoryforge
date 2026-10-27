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
