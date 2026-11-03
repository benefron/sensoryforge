# Design Decisions

This page is for anyone who wants to know why the agentic workflow is built the way it is,
before relying on it or changing it. Each decision below says what was chosen, what else
was on the table, and why. Every one is recorded in the project's
[ledger](https://github.com/benefron/sensoryforge/blob/main/docs_root/LEDGER.md), a single
file of decisions and findings built from commit messages; the id after each decision is
its entry there (search the file for it).

## What an agent may do

**Agents never invent domain facts.** Physics, models, parameters and units come from the
user, from a linked source the user approves, or from reasoning the user approves, and
every assumption is labelled with its origin. The other option was to let an agent fill in
plausible values from what it has learned and let the user correct them afterwards; it was
turned down in favour of the user and the agent defining the physics together, the agent
either grounding a value in sources the user can follow or asking for it. The grounding skill is its own skill, since every skill that writes a value
calls it. Ledger: `D-fac4caa`, `D-28a4edc`.

**"Typical" is never from memory.** When a report is interpreted, what counts as typical
comes from the data set's own spread, the report contract, or a source the user approves.
Ledger: `D-3dfbe6e`.

**Meaning, not strings.** An agent maps the user's words to SensoryForge's terms by meaning
and by the whole session, with the [glossary](glossary.md) as the anchor, instead of
matching the user's words against names in the code. The rule is written once, in the brief
every skill runs first, so it cannot be stated differently in two skills. Ledger:
`D-d97f27f`.

**A missing component is scaffolded, not written.** When a plan needs a neuron, filter or
stimulus that does not exist, the planning skill hands off to `sensoryforge new-component`
and waits for the user's model. Ledger: `D-7904b2b`.

## How it works with you

**Self-contained by default, integration on request.** The skills and files ship inside
the package and depend on nothing on the user's machine. Using the user's own tools, the
routing hook and a working style are offered at setup and stay off until approved. The
other option, reading the user's skills and servers and adapting to them by default, was
turned down: a skill that depends on one machine's setup behaves differently on the next.
Ledger: `D-c889235`.

**The working style is one saved file, read through one command.** The answers to the
setup are saved in `.sensoryforge/agents.yml` and rendered by `sensoryforge agents brief`,
which every skill runs first and which a test requires each skill to name. Ledger:
`D-97c3347`.

**The user sets the pace.** Whether a chain runs step by step or end to end is decided by
the user in the plan or in chat, not by a default. The workflow skill separates an *offer*
("shall I go on?", skipped when the pace is to carry on) from a *decision* (an open value,
a size, a seed, the sensor, the metric packs, the cost of a long run, a failure), and a
decision is asked for at either pace. Ledger: `D-76fdd8a`, `D-6036b92`.

**Hooks are optional and small.** The one hook is Claude Code's prompt hook, a
standard-library script written to the user's local settings only after a yes at setup;
other environments are never offered hooks. The next-step prompts are the skills' own
closing lines plus that hook, rather than new hook machinery. Ledger: `D-c1fc12e`,
`D-dc22c86`.

## Where the files live and how many

**In the package, not a top-level folder.** User-facing agent files live in
`sensoryforge/agents/`, so installing SensoryForge brings them; a top-level `agents/` folder
would exist only in a clone. A shipped skill names no path that exists only in a clone, and
says "install from source (not yet on PyPI)" rather than a `pip install` that does not work
yet. Ledger: `D-280840b`, `D-efa939a`, `D-a93c3e9`.

**Few, general files.** There is a budget: at most 18 hand-written user-facing files, of
which at most 14 are skills; generated files are unlimited but each comes from one command
and is checked in CI. Many narrow files were the alternative; they would go stale and need
updating on every change.
Today there are ten skills. Ledger: `D-8da98c5`, `D-28a4edc`.

**The glossary ships; the task index does not.** The glossary is copied into the package
beside the skill folders (so one relative link works in the package and in every export),
and the brief names the project's copy first. The task index lists paths in a clone, so it
stays on the docs site for people developing SensoryForge. Ledger: `D-c2e83a0`,
`D-32e8ff2`.

## Other environments

**Claude first, then one neutral source.** The skills were written in Claude Code's format
first and then generalised. The options were Claude Code only, a neutral format from the
start, or Claude first and then neutral; the last was chosen, with a specific export only
for environments whose format is documented and a generic one for the rest. The packaged `SKILL.md` files, in the open
Agent Skills layout, are now the one source, and each environment's files are generated
from them with a do-not-edit header and a `--check`. There is no per-tool rewrite of a
skill's body, because Claude Code, Copilot, Cursor and Codex all read the same layout.
Ledger: `D-3a46fe4`, `D-5d48fdc`.

**The project's instruction file stays the user's.** SensoryForge supplies only a marked
section, written only with `--write`; an agent builds the rest of the file in its
environment's own way. Ledger: `D-983e9ab`.

## Keeping it honest

**Every skill ends with a worked example that runs.** Skills are named
`sensoryforge-<verb>`, and the test suite writes each worked example's files and runs its
commands, so a skill cannot drift from the code. The [Skills Reference](skills_reference.md)
is generated from the same files. Ledger: `D-ab61f5c`.

**Descriptions come from the code.** `sensoryforge describe` and `validate --json` are
generated from the registries, never from a typed list, and the glossary test fails when a
registry gains a kind the glossary does not define. Ledger: `D-127f390`, `D-4a6f527`.

**Fresh agents test the skills.** Live scenario tests install a built wheel into an empty
folder and give a fresh agent a request, refusing every permission prompt the scenario does
not allow; they are opt-in, and a release re-runs those of every skill that changed
([Agent Scenarios](../developer_guide/agent_scenarios.md)). The release-time update of the
agent files is a maintainer command, not a shipped skill, because it reads the repository's
history. Ledger: `D-cab78f0`, `D-47468a5`.

## The core stays modality-neutral

New agent-skill work uses the neutral words (sensor, channel, frame, population, draw,
entry), with tactile as the first pack, and is extended by registry and plugin rather than
by editing the core. A skill speaks of a sensor and its frames, and the tactile words
appear only where a tactile pack is in use. Ledger: `D-35016af`.

## The section itself

This section exists because the agentic workflow is a fourth way in, beside the CLI, the
GUI and batch runs, and gets the same documentation: setup, rules, skills, exports and these
decisions. The skills reference is generated so it cannot drift. Ledger: `D-6f0c3cc`.
