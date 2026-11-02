# Working with a Coding Agent

This section is for anyone who would rather describe what they want in words and have a
coding agent (Claude Code, GitHub Copilot, Cursor, Codex or another) do the SensoryForge
work with them. By the end of this page you will have the skills installed, your working
style saved, and a first request made.

The agentic workflow is a fourth way in, beside the [command line](../user_guide/cli.md),
the [GUI](../user_guide/gui_walkthrough.md) and [batch runs](../user_guide/batch_processing.md).
It does not add a new engine: SensoryForge ships **skills**, short instruction files an agent
follows, and every skill does its work with the same `sensoryforge` commands you could type
yourself. What the agent adds is the path from a goal in your own words to those commands,
with you deciding at each point that is yours.

## When to use it

| You want to | Use |
|---|---|
| Go from a question ("can this sensor tell dots from edges?") to a plan, a world, a data set, a simulation and a report, deciding as you go | the agent |
| Describe stimuli in words and change them in words, with a preview each time | the agent ([an example](../user_guide/agent_design_loop.md)) |
| Ask why a number in a report looks wrong, or what a metric means | the agent |
| Run a config you already have, the same way every time, in a script or on a cluster | the [CLI](../user_guide/cli.md) and [batch](../user_guide/batch_processing.md) |
| Look at a sensor, a receptive field or a spike raster and tune a parameter by eye | the [GUI](../user_guide/gui_walkthrough.md) |
| Write new physics (a neuron, a filter, a stimulus) | a [plugin package](../developer_guide/plugins.md); the agent scaffolds it and waits for your model |

The ways in share one config format and one engine, so you can switch at any point: a
world the agent wrote runs from the CLI, and a report the CLI wrote can be read back by the
agent.

## Five-minute start

**1. Install SensoryForge.** It is not on PyPI yet, so install it from source (more options
in [Installation](../getting_started/installation.md)):

```bash
pip install "git+https://github.com/benefron/sensoryforge.git"
```

**2. Put the skills where your agent reads them.** In the folder where you will work:

```bash
sensoryforge agents install        # Claude Code: writes .claude/skills/
```

For another environment, `sensoryforge agents export --target copilot` (or `cursor`,
`codex`, `generic`); see [Using the Skills in Another Agent](../user_guide/agent_export.md).
Run the command again after upgrading SensoryForge. It writes only its own files.

**3. Say how you like to work.** Answer the questions, or press Enter to keep each
self-contained default:

```bash
sensoryforge agents setup
sensoryforge agents brief          # prints what every skill will follow
```

What each question means and changes: [Setup](../user_guide/agent_setup.md).

**4. Make a first request.** Open your agent in the same folder and say what you want, in
your own words. For example:

```text
I want to see whether a sensor array can tell dots from edges. Plan it first, then take it
step by step.
```

The agent reads your working style (`sensoryforge agents brief`), runs a few read-only
commands to see what exists (`sensoryforge list-presets`, `sensoryforge describe`), and
writes a short plan: the stimulus classes, the sensor, the sizes, each marked with who
decides it. Values it does not know, such as the size range of a dot, are marked open and
asked for, never made up. Once you approve the plan, it writes the world and shows you a
preview before going on. The worked plan for this request is the first example in the
[Skills Reference](skills_reference.md#design-and-plan).

**5. Carry on, or stop.** Each step ends by offering the next one; you can stop at any
point and pick up later, because each step reads what the one before wrote into the same
run folder. Say "do the whole thing" to have the workflow skill chain the rest at the pace
you set.

## In this section

- [Setup](../user_guide/agent_setup.md): every option, the saved file, the routing hook.
- [Skills](../user_guide/agent_skills.md): the ten skills, what to say to reach each, and
  how a chained run proceeds.
- [Skills Reference](skills_reference.md): each skill's steps and worked example,
  generated from the skill files.
- [Rules for Agents](rules.md): what an agent will and will not do, in plain words.
- [Designing a World by Describing It](../user_guide/agent_design_loop.md): three recorded
  sessions.
- [Using the Skills in Another Agent](../user_guide/agent_export.md): Copilot, Cursor,
  Codex, an environment with no export, and the project's instruction file.
- [Glossary](glossary.md): the terms the agent maps your words to.
- [Design Decisions](decisions.md): why it is built this way, and what else was considered.
- [Extending It](extending.md): adding a skill, a prompt for a new environment, or a
  modality.
