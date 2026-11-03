# CLAUDE.md

Guidance for Claude Code in this repository. The shared, agent-neutral front door is
`AGENTS.md`, imported here; this file adds only what is specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **The ledger is injected.** At session start a hook (`.claude/settings.json`) prints a
  digest of `docs_root/LEDGER.md` (open findings, recent decisions, retired framings); a
  prompt hook recalls entries a message touches. The hooks in `.claude/hooks/` belong to
  the living-ledger skill; do not edit them by hand.
- **Path-scoped rules.** `.claude/rules/*.md` are delivered when a matching file is
  touched (the world engine, engine parity). Write one when a finding keeps being
  re-discovered; the format is in `.claude/rules/README.md`.
- **Slash commands** in `.claude/commands/` (`/review`, `/remediate`, `/maintain-docs`,
  `/add-component`, `/scientific-edit`) are maintainer tools; they are not shipped.
- **Skills for SensoryForge's users** are written in Claude's format under
  `sensoryforge/agents/skills/` and ship with the package. Where every agent file lives,
  and the file-count budget: `docs/developer_guide/agent_files.md`.
- **Architecture in depth** (data flow, receptive fields, worlds, converters, engines,
  configs, registries, the GUI, known technical debt):
  `docs/developer_guide/architecture_notes.md`. Read the section you are about to change.
- **Roadmap sessions** follow `docs_root/plan/PLAN.md`.
