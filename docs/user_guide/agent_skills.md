# Skills for a Coding Agent

SensoryForge ships **skills**: short, tested instruction files that a coding agent follows
to do a SensoryForge job with you, from a goal in your own words to a report you can read.
They are written in Claude Code's format today; other environments follow. They are
self-contained: they use only SensoryForge's commands and your agent's basic file and shell
tools, and nothing else on your machine unless you say so in the [setup](agent_setup.md).

## Install them

SensoryForge is not on PyPI yet, so install it from source (see
[Installation](../getting_started/installation.md)). Then, in the folder where you work:

```bash
sensoryforge agents install
```

This copies the skills into `.claude/skills/` of that folder, with the
[glossary](../agents/glossary.md) beside them (`sensoryforge-glossary.md`), where Claude
Code finds them. Run it again after upgrading SensoryForge to replace the copies with the
new release's files; it never touches anything else in the folder. Then run the
[setup](agent_setup.md) once so the skills know how you like to work.

## The skills

You do not name a skill. Say what you want; the agent picks the skill whose purpose matches
what you mean, and asks when it is unclear. Each skill reads your saved working style
first, states what it will do, and stops to ask at the points that are yours to decide.

| Skill | Say something like | What you get |
|---|---|---|
| `sensoryforge-setup` | "set up how SensoryForge works with me" | your working style saved |
| `sensoryforge-design-and-plan` | "I want to see whether a sensor can tell dots from edges" | a written plan: the choices and who makes them, the steps with their commands |
| `sensoryforge-design-world` | "I need stimuli of three kinds, varying in size and angle" | a world file that validates, sample draws and a preview picture |
| `sensoryforge-build-dataset` | "make a training and a test set from this world" | a data-set file and its built manifest, with the counts per split |
| `sensoryforge-run-simulation` | "run it on the SA/RA recipe" | one data bundle per entry; one entry timed first; resumable |
| `sensoryforge-evaluate` | "check that the data show what the world says" | a report folder, read back to you with measured, declared and the raw difference side by side |

A usual chain is plan, world, data set, simulation, evaluation, and you can start or stop at
any step: each one reads what the one before wrote.

## What the skills will not do

- **Invent domain facts.** Physics, models, parameters, ranges and units come from you, from
  a source you approve with a short report, or from reasoning you approve. When the agent
  proposes a value to move on, it marks it as an assumption and lists it for you.
- **Judge a difference for you.** The evaluation reports measured, declared and their
  difference. It states no tolerance; you decide what is acceptable.
- **Start long work unasked.** A run that takes time or disk space is measured on one entry
  first, and you are asked before the rest.
- **Write a missing component's physics.** If you need a neuron, filter or stimulus that does
  not exist, the plan scaffolds an installable plugin package
  (`sensoryforge new-component`, see [Plugin Packages](../developer_guide/plugins.md)) and
  waits for your model.

## How they are kept honest

Each skill ends with a worked example: real files and the exact commands, with what each
command must print. A test writes the files into an empty folder and runs the commands, so
a skill whose commands drift from the code fails the test suite. To run one yourself, copy
the files out of the skill's "Worked example" section into a new folder and run its
commands there.
