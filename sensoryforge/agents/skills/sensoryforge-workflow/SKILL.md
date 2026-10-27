---
name: sensoryforge-workflow
description: Carry a SensoryForge job from a goal to an explained report by chaining the other skills, step by step or end to end as the user chooses. Use when someone wants the whole job or several steps in a row, says "take it from here", "do all of it" or "what next", or starts SensoryForge work without naming a step, in whatever words they put it.
---

# Workflow

A SensoryForge job usually runs plan, world, data set, simulation, evaluation, then an
explanation. This skill runs that chain with the skill each step names, at the pace the
user chooses, and keeps them informed of where the job stands. It does none of the steps
itself: each step's skill does it, with its own checks and stops.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Find where the job stands

Start from the first missing piece, not from the top. Look, with read-only checks, for
each output in the working folder: a plan the user approved (in this chat), a world file
(`sensoryforge world validate FILE`), a data-set file and a built manifest, bundles from a
run, a report folder. The script in the worked example prints what a folder holds. If a
piece is there but you do not know whether it is current, ask; never rebuild over a
user's file.

One run, one folder: the manifest (`sensoryforge dataset build FILE --out runs/NAME`), the
bundles (`sensoryforge batch --dataset FILE --preset NAME --output runs/NAME`) and the
report (`sensoryforge evaluate run FILE --out runs/NAME`) all go in `runs/NAME`, as
`sensoryforge evaluate run` does by itself. A resume continues
only the same data set on the same sensor; for another, start a new folder.

## 2. Settle the plan and the pace

- **The plan.** If there is none, use `sensoryforge-design-and-plan` first and wait for
  the user's approval of it. If the user named one exact step ("just evaluate this"), run
  that step's skill alone and offer the rest afterwards.
- **The pace.** The brief's pace is only the suggestion. Say it in one sentence ("I will
  ask before each next step" or "I will carry on once you approve the plan") and let the
  user change it, in the plan or in chat. What the user says in this session wins over the
  saved style, and it holds until they change it again.

## 3. Run the chain

| Step | Skill | Produces | Stops for the user |
|---|---|---|---|
| 1 | `sensoryforge-design-and-plan` | the plan | the plan's approval |
| 2 | `sensoryforge-design-world` | a world file and a preview | the classes, ranges and every assumption |
| 3 | `sensoryforge-build-dataset` | a data-set file and its manifest | sizes, splits, seeds |
| 4 | `sensoryforge-run-simulation` | one bundle per entry | the sensor, and the cost of the whole run after one entry is timed |
| 5 | `sensoryforge-evaluate` | a report folder | the metric packs |
| 6 | `sensoryforge-interpret-report`, `sensoryforge-diagnose` | an explanation | only on request, or when something failed or surprised |

Use only the steps the goal needs. Before each step, open that step's skill and follow it;
do not run its commands from memory of this table. After each step say, in one line, what
ran, the folder it wrote and the counts, then go on as step 4 says.

## 4. Two kinds of stops

- **An offer** is "shall I go on to the next step?". When the pace is to ask, ask it. When
  the pace is to carry on, say what comes next and do it.
- **A decision** is something only the user can give: a value the plan marks as open or as
  an assumption, a size, a seed, the sensor, which metric packs, what difference is
  acceptable, the cost of a long run, a model or a source for a missing piece. A decision
  is never skipped, whatever the pace. Ask for it and wait.

Stop and ask, in either pace, when a step fails (name the step and the error; offer
`sensoryforge-diagnose`; do not loop), when a run would cost more than the plan sized,
when a step needs a component that does not exist (see "When a component is missing" in
`sensoryforge-design-and-plan`), and when a step would need a fact about physics, a model
or a unit that nobody has given you.

## 5. Finish

Give the chain's summary: each output in order with its folder; the command that repeats
each step; every assumption with its origin; what failed or was skipped; and what is left
to do. If the brief sets a session budget, stop near it: write what is done, what is next
and the folders, so a new session can continue from step 1 of this skill.

## Hooks

A hook is optional. If the user wants plain-language requests routed to the right skill
without naming one, `sensoryforge-setup` offers it (Claude Code only, off until approved).
With or without a hook, this skill works the same way.

## Worked example

A small job from the first step the folder is missing, checking where it stands after
each step. The plan it would write is shown, not run.

```markdown
Goal: see how a sensor answers small round stimuli.

Pace: I will ask before each next step (your saved style); say "carry on" to change it.

Steps
1. design-world: the file below. Stops for your OK.
2. build-dataset: 4 entries (you), seed 7 (you). Stops for your OK.
3. run-simulation: one entry timed, then the rest.
4. evaluate: the report, read back with the raw numbers.
```

```yaml
# file: world.yml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 40]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
```

```yaml
# file: dataset.yml
dataset:
  name: demo
  world: world.yml
  seed: 7
  duration_ms: 80
  splits:
    train: {n: 4}
```

```python
# file: where.py
"""Where a SensoryForge job stands: which of the chain's outputs a folder holds."""
import json
import sys
from pathlib import Path

from sensoryforge.world import read_batch_index

folder = Path(sys.argv[1])
manifest = folder / "manifest.jsonl"
entries = sum(1 for _ in open(manifest)) if manifest.exists() else 0
rows = read_batch_index(folder) if (folder / "index").exists() else []
done = sum(1 for row in rows if row["status"] == "ok")
print("data set built:", f"{entries} entries" if entries else
      "not in this folder" if done else "not yet")
print("simulated:", ("not yet" if not done else f"{done} of {entries}" if entries
                     else f"{done} ok (no manifest here to count against)"))
pipeline = folder / "evaluate" / "pipeline.json"
if pipeline.exists():
    report = json.load(open(pipeline))
    print("evaluated:", report["status"], json.dumps(report["counts"]))
else:
    print("evaluated: not yet")
```

```bash
python where.py runs/demo
# prints: data set built: not yet
sensoryforge dataset build dataset.yml --out runs/demo
# prints: 4 entries
python where.py runs/demo
# prints: data set built: 4 entries
sensoryforge batch --dataset dataset.yml --preset tactile_sa1_ra1_adex --output runs/demo --entries 0:1
# prints: 1 ok, 0 failed
python where.py runs/demo
# prints: simulated: 1 of 4
sensoryforge evaluate run dataset.yml --out runs/demo --preset tactile_sa1_ra1_adex --resume --no-checks
# prints: report folder
python where.py runs/demo
# prints: evaluated: ok
```
