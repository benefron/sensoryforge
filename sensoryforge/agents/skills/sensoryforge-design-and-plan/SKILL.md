---
name: sensoryforge-design-and-plan
description: Turn a SensoryForge goal into a written plan of steps and choices before anything runs. Use when someone says what they want to study, build, test or compare with SensoryForge (a set of stimuli, a data set, a sensor, an experiment, a new component) and has not yet said exactly what to run, in whatever words they put it.
---

# Design and plan

Turn a goal into a short plan the user approves: the choices that are theirs, the steps
with exact commands, what each step produces and where it stops. This skill writes the
plan; it runs nothing but read-only commands. The other SensoryForge skills do the work.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. Its planning line sets how long the plan is
(none, a short one, or a full one that waits for approval). If the command is not found,
SensoryForge is not installed here: say so and stop (it installs from source; it is not on
PyPI yet).

## 1. Understand the goal

Ask what the user wants to learn or build, one short question at a time, at most three
before you draft something. You need: the question the result must answer; what the
sensor will be given (the stimuli, and what varies); and what happens to the result
(training, testing a model, comparing two sensors). Map their words to SensoryForge's terms
by meaning (the glossary the brief points to defines them); when you are not sure which
term they mean, ask.

## 2. Find out what exists

Use commands, not memory. Parameter lists and names change between releases.

- `sensoryforge describe` lists every registered component, world kind, evaluation pack and
  shipped recipe; `--section components|world|evaluation|recipes` narrows it and `--json`
  gives the parameters. `sensoryforge list-presets` names the recipes.
- Match each need to something that exists: a stimulus set (a world: skill
  `sensoryforge-design-world`), a data set (`sensoryforge-build-dataset`), a sensor
  (a recipe, a design folder or a config the user has), a simulation run
  (`sensoryforge-run-simulation`), an evaluation (`sensoryforge-evaluate`), reading what a
  report's numbers mean (`sensoryforge-interpret-report`) and tracing a result that is not
  what was expected (`sensoryforge-diagnose`).
- If a need has no match, go to "When a component is missing" below. Plan every step that
  does not depend on it; do not stop the whole plan.

## 3. Mark who decides

Physics, models, parameters, ranges, units, sizes, seeds and what counts as good enough
are the user's. They come from the user, from a source the user approves with a short
report, or from reasoning the user approves. Put an origin on every value in the plan:
you, a source (with its link) or "assumption: needs your OK". Never fill a number in
because it is plausible; leave it as an open question.

## 4. Size it

Say how many entries, how long each is and what the sensor is; then say what that costs in
time and disk only as far as you can measure it. The run-simulation skill times one entry
first, so write "measured before the full run" rather than a guess. If the user set a
session budget, split the plan into sessions that fit it and say where each one stops.

## 5. Write the plan

Show it in this shape, shorter when the brief asks for a short plan:

1. **Goal**, one sentence in the user's words.
2. **Choices**: a table of choice, value and origin (step 3), with the pace the brief
   suggests (the user can change it here or later). Open questions go last.
3. **Steps**, numbered: the skill, the exact command, what it produces, how it is checked
   and where it stops for the user. A usual chain is world, data set, simulation,
   evaluation; use only the steps the goal needs.
4. **Risks**, only the real ones (a long run, a large disk need, a missing component).

## 6. Hand over

Ask the user to approve or change it. Do not start a step before the approval the brief
asks for. Then run the steps with `sensoryforge-workflow`, which uses the skill each step
names and keeps the pace, or one skill at a time if the user prefers. If the user changes
the goal mid-way, change the plan and say what moved.

## When a component is missing

Name what is missing in the glossary's words and check for something close first
(`sensoryforge describe`, by meaning, not by name). If nothing fits:

- A neuron model, filter, stimulus, solver or sensor array: `sensoryforge new-component
  --help`, then scaffold an installable plugin package with `sensoryforge new-component
  <kind> <Name>`. The plugin guide is
  https://benefron.github.io/sensoryforge/developer_guide/plugins/.
- A receptive-field builder or a processing layer: start from the plugin template,
  https://github.com/benefron/sensoryforge/tree/main/examples/plugin_template.
- A world shape, pattern, modulation or distribution: it is registered from Python; see
  https://benefron.github.io/sensoryforge/user_guide/worlds/#extending.

Never write the component's physics yourself. Ask the user for the equations or a source,
and put "component: waiting for your model" in the plan as a step that needs them.

## Worked example

The user says: "I want to see whether a sensor array can tell dots from edges." The
read-only checks the skill runs:

```bash
sensoryforge list-presets
# prints: tactile_sa1_ra1_adex
sensoryforge describe --section world
# prints: world shapes
sensoryforge describe --section evaluation
# prints: metric_packs
```

The plan it writes (shown to the user, not run):

```markdown
Goal: tell dots from edges with one sensor array.

| Choice | Value | Origin |
|---|---|---|
| Stimulus classes | dots, edges | you |
| Sensor | a shipped recipe, to be named | open: which one? |
| Sizes and ranges of each class | not set | open: yours to give |
| Data set size | not set | open: yours to give; time measured on one entry first |

Steps
1. sensoryforge-design-world: write and preview the world. Stops for your OK.
2. sensoryforge-build-dataset: splits and seeds. Stops for your OK.
3. sensoryforge-run-simulation: time one entry, then the rest.
4. sensoryforge-evaluate: report, read back with the raw numbers.
```
