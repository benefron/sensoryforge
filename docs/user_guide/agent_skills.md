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
| `sensoryforge-ground-assumptions` | "what value should I use for this?" or "find me a source" | the value from you, or a short report of linked sources for you to approve, and a list of every assumption with where it came from |
| `sensoryforge-build-dataset` | "make a training and a test set from this world" | a data-set file and its built manifest, with the counts per split |
| `sensoryforge-run-simulation` | "run it on the SA/RA recipe" | one data bundle per entry; one entry timed first; resumable |
| `sensoryforge-evaluate` | "check that the data show what the world says" | a report folder, read back to you with measured, declared and the raw difference side by side |
| `sensoryforge-diagnose` | "why are five of my six draws silent?" | the entries behind a bad number, their draws, and which of four places holds the cause (the data set, the sensor, the measurement, the code) with the evidence |
| `sensoryforge-interpret-report` | "what does this latency number mean, and is it typical?" | what a metric measures, its unit and resolution from the report contract, and how this data set's own values are spread |
| `sensoryforge-workflow` | "take it from here" or "do the whole thing" | the other skills run in order, each at your pace, with a summary of every output, command and assumption at the end |

A usual chain is plan, world, data set, simulation, evaluation, and you can start or stop at
any step: each one reads what the one before wrote. Diagnose and interpret-report start from
a report you already have.

### Step by step, or end to end

The workflow skill starts from the first piece your folder lacks and runs the chain with the
skill each step names. Your saved working style only suggests the pace; you set it in the
plan or in chat, and it holds until you change it. Step by step, the agent asks before each
next step. End to end, you approve the plan once and the agent carries on, saying what it
does next. Either way it stops for what is yours to decide (any value the plan marks as open
or as an assumption, sizes, the sensor, the cost of a long run) and for any failure. Each of
the other skills ends the same way: it offers the next step at your pace.

One run, one folder. The skills build the data set, simulate it and write the report into the
same folder, `runs/NAME`: `sensoryforge dataset build ... --out runs/NAME`, then
`sensoryforge batch ... --output runs/NAME`, which is also the layout
`sensoryforge evaluate run --out runs/NAME` makes. `--resume` continues only the same data set
on the same sensor (entry ids repeat across data sets, so mixing them would pass unnoticed);
it refuses any other folder and says which values differ, and the skills then start a new
folder. `sensoryforge batch --dataset ... --dry-run` shows what a run would do without doing
it. After a build, the skills also count the entries whose draw ends after the data set's
duration (their end is cut off) and tell you.

Interpreting never judges. The report states no tolerance, so "typical" comes from your data
set's own spread, from the report contract, or from a source you approve, and never from the
agent's memory of what a neuron or a sensor usually does.

## What the skills will not do

The first item is the rule behind [designing a world by describing it](agent_design_loop.md).

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
a skill whose commands drift from the code fails the test suite. The commands in a skill's
own steps are checked too: every `sensoryforge ...` command is parsed with the real command-line
parser (a flag that does not exist fails), every `python -c` line must compile, and every link
to these docs must point at a page and a heading that exist. A built wheel is opened to check
that each skill and the glossary are inside it. To run one yourself, copy the files out of the
skill's "Worked example" section into a new folder and run its commands there.
