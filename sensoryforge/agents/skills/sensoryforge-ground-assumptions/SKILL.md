---
name: sensoryforge-ground-assumptions
description: Settle a model, parameter, range or unit that nobody has given, by asking the user or by finding sources they approve, and keep every assumption listed with where it came from. Use before writing any number into a plan, world or data-set file that arrives with no unit or far outside the range the skill or the world states, even for a plain edit; and when a value, an equation, a unit or "what is typical" is missing, when a plan or a file holds a value marked as an assumption, when someone asks for a source, a reference or "the right number", or before any value would be filled in from memory, in whatever words they put it.
---

# Ground assumptions

Physics, models, parameters, ranges and units are not yours to supply, however plausible
a number looks and however sure you are. This skill is how a missing one gets an origin:
the user describes it, or you find sources, link them and write a short report for the
user to approve. Every value that ends up in a file is listed with where it came from.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Say exactly what is missing

Name the value or model, its unit, where it would be used (a file and path, or a plan row)
and what changes if it is different. Check what is already known before you ask: the
user's words so far, the files in the folder, and `sensoryforge describe --json` for a
registered kind's parameters, units and allowed ranges. That tells you what a name or a
unit means; it does not tell you which value to use.

## 2. Offer the two ways and let the user choose

- **They describe it.** The value and its unit, what it stands for, or their own data.
- **You look for sources** and write a report for them to approve (step 4).

Do not pick for them, and do not take a third way: no value from memory, from what is
"usually" so, or from a similar project, even one that looks obvious. If the user cannot
decide, leave the value open (step 5) and carry on with what does not depend on it.

## 3. When they describe it

Write it down in their words with its unit, read it back, and ask for the unit if it is
missing. The origin is **you**. If it looks odd against a registered range, say so and ask;
do not change it.

## 4. When you look for sources

Use a search or fetch tool only if one is listed in this session, and say which. If there
is none, say so and ask the user for sources (links, papers, data files, notes). Do not fill
the gap from memory.

Open every source you cite and read the passage. Write a report of at most a page:

- the need, as in step 1;
- for each source (a few, not a list of everything): the link, the place in it (section,
  table, figure), what it says about this value with the number, unit and conditions as it
  states them, and what is the same and what differs from the case at hand;
- where sources disagree;
- what is not established, and any source you could not open (listed as not read);
- how you read it, labelled as a reading, with no number that is not in a source.

Then wait. The user approves a value or a source, gives their own, or leaves it open.

## 5. Record every assumption

Keep a table, `assumptions.md`, beside the file that holds the value: value, unit, where
it is used, origin, status. The origin is exactly one of:

- **you** (and when or where they said it);
- **source: LINK, place** (the user approved it);
- **reasoning: the argument in one sentence** (the user approved it);
- **assumption: needs your OK** (nobody has approved it yet).

A value not yet approved is written into the file with the comment `# assumption: needs
your OK`, and its status is open. `python assumptions.py FILE assumptions.md` lists the
marked lines in FILE, counts the table's open rows and names any row with no origin; it
exits 1 while anything is open or unsourced.

## 6. Report back

List every assumption with its origin, the open ones first, and where the source reports
are. An open assumption is a decision still the user's, whatever the pace: ask about it. Do
not start a step that depends on an open value without the user's say-so.

## Verify

Every value you wrote has an origin; every source you cite was opened; the script exits 0
only when nothing is open; nothing came from memory.

## Worked example

A world with one size range nobody has approved yet, and the table that records it:

```yaml
# file: world.yml
world:
  name: round
  defaults:
    hold_ms: {range: [20, 40]}  # you
  classes:
    dots:
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}  # assumption: needs your OK
```

```markdown
# file: assumptions.md
| Value | Unit | Used in | Origin | Status |
|---|---|---|---|---|
| sigma_mm range 0.15 to 0.45 | mm | world.yml, classes.dots.axes | assumption: needs your OK | open |
```

After the user says "use 0.2 to 0.5, that is what I measured":

```yaml
# file: world_approved.yml
world:
  name: round
  defaults:
    hold_ms: {range: [20, 40]}  # you
  classes:
    dots:
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.2, 0.5]}}  # you
```

```markdown
# file: assumptions_approved.md
| Value | Unit | Used in | Origin | Status |
|---|---|---|---|---|
| sigma_mm range 0.2 to 0.5 | mm | world_approved.yml, classes.dots.axes | you (chat, 2026-10-27) | approved |
```

```python
# file: assumptions.py
"""Open assumptions in a file, and the rows of a table that have no origin."""
import sys

MARK = "assumption: needs your OK"
ORIGINS = ("you", "source:", "reasoning:", MARK)

path = sys.argv[1]
marked = [(n, ln.strip()) for n, ln in enumerate(open(path), 1) if MARK in ln]
for n, ln in marked:
    print(f"{path}:{n}: {ln}")
rows = opened = unsourced = 0
table = sys.argv[2] if len(sys.argv) > 2 else None
for n, ln in enumerate(open(table), 1) if table else []:
    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
    if len(cells) < 5 or cells[0] == "Value" or set(cells[0]) <= set("-: "):
        continue
    rows += 1
    opened += cells[4] == "open"
    if not cells[3].startswith(ORIGINS):
        unsourced += 1
        print(f"{table}:{n}: no origin for {cells[0]!r}")
print(f"{len(marked)} marked in {path}; {rows} recorded ({opened} open); "
      f"{unsourced} without an origin")
sys.exit(1 if marked or opened or unsourced else 0)
```

```bash
sensoryforge validate --json --kind world world.yml
# prints: "valid": true
python assumptions.py world.yml assumptions.md
# prints: 1 marked in world.yml
# exit: 1
python assumptions.py world_approved.yml assumptions_approved.md
# prints: 0 marked in world_approved.yml
```
