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
