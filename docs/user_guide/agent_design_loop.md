# Designing a World by Describing It

You can build a world by talking to a coding agent that has the
[SensoryForge skills](agent_skills.md): describe what you want, look at what it drafts, and
answer in your own words ("slower", "softer", "more quiet between touches"). The agent
changes the fields those words mean, checks the file, and shows you the result again.

## The rule: the agent does not invent physics

The physics, the models, the parameters, the ranges and the units are yours. The agent
never supplies one from its own memory, however sure it sounds, and it never fills a number
in because it is plausible. For every value it needs and does not have, it does one of two
things, and you choose which:

1. **It asks you to describe it.** You give the value and its unit, or your own data.
2. **It looks for sources, links them and writes you a short report** for you to approve.
   It opens every source it cites and says where in it the number is. If it has no search
   tool in your session it says so and asks you for sources instead.

A value you have not approved is written into the file with the comment
`# assumption: needs your OK` and listed in the agent's output, and it stays your decision
until you settle it. Every value the agent reports comes with its origin, exactly one of:

- **you**: you said it;
- **source**: a link and the place in it, which you approved;
- **reasoning**: an argument you approved;
- **assumption: needs your OK**: nobody has approved it yet.

The skill that does this is `sensoryforge-ground-assumptions`; the skills that write values
(`sensoryforge-design-world`, `sensoryforge-design-and-plan`) call it, and its helper
`python assumptions.py FILE assumptions.md` exits non-zero while any value is open or has no
origin. The rule is tested: the sessions below are checked so that the agent asks before it
writes a value and every value it writes carries its origin.

## Session 1: describe, draft, look

**You:** I want to study small round touches and thin ridges, each touched once.

**Agent:** I can draft that. I have to ask for what I cannot read off your words, because
the ranges are yours. How big is a round touch, how wide is a ridge and at what angles, how
long does a touch last, how strong is it, and how often does each kind occur?

**You:** Rounds 0.15 to 0.45 mm (sigma), ridges 0.05 to 0.15 mm wide at any angle. Touch 5 to
15 ms, hold 20 to 40, release 5 to 15, starting within 10 ms. Strength 0.5 to 1, anywhere
within 0.3 mm of the centre. Rounds 60 percent, ridges 40.

**Agent:** Every number is yours, so each is marked so. Here is the file:

```yaml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}      # you
    touch_ms:   {range: [5, 15]}      # you
    hold_ms:    {range: [20, 40]}     # you
    release_ms: {range: [5, 15]}      # you
    amplitude:  {range: [0.5, 1.0]}   # you
    x_mm: {range: [-0.3, 0.3]}        # you
    y_mm: {range: [-0.3, 0.3]}        # you
  classes:
    dots:
      weight: 0.6                                  # you
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}      # you
    edges:
      weight: 0.4                                  # you
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}            # you
        orientation_deg: {range: [0, 180], circular: true}  # you
```

```text
$ sensoryforge world validate world.yml
w-455d1e8b627a  demo  channels=['value']
  class dots (layered) weight=0.6: amplitude, delay_ms, hold_ms, release_ms, sigma_mm, touch_ms, x_mm, y_mm
  class edges (layered) weight=0.4: amplitude, delay_ms, hold_ms, orientation_deg, release_ms, touch_ms, width_mm, x_mm, y_mm
$ sensoryforge world sample world.yml -n 2 --seed 7
(two JSON records, one per draw: class, every drawn value, a timeline)
```

**Agent:** It validates. The first two draws are both round touches. The first starts
after 2.2 ms, holds from 16.5 to 50.3 ms and ends at 59.9 ms; the second starts after
0.1 ms and ends at 63.1 ms. I have also rendered a preview picture, `preview.png`, one row
per draw. Is this what you meant?

**You:** Yes.

**Assumptions listed:** none open; every range and weight is from you.

## Session 2: change it in words

**You:** Make it slower and softer, and give it more quiet between touches.

**Agent:** I read that as: slower is longer touch and hold times, softer is a lower
amplitude, and quiet between touches is a pause between two touches. This world has one
touch per draw, so there is no quiet between touches yet. Two questions, because the sizes
are yours. By how much, and do you want two touches in a row?

**You:** Twice as long, amplitude 0.2 to 0.6, and yes, two touches. I do not know the pause
yet; put something in for now.

**Agent:** Here is what changes, as a table. I have saved it as `world_v3.yml` and kept
your first file.

| Field | Old | New | Origin |
|---|---|---|---|
| touch_ms | 5 to 15 | 10 to 30 | you ("twice as long") |
| hold_ms | 20 to 40 | 40 to 80 | you ("twice as long") |
| amplitude | 0.5 to 1.0 | 0.2 to 0.6 | you |
| contacts | 1 | 2 | you ("two touches") |
| pause_ms | none | 20 to 40 | assumption: needs your OK |

```yaml
touch_ms:   {range: [10, 30]}     # you: "twice as long"
hold_ms:    {range: [40, 80]}     # you: "twice as long"
amplitude:  {range: [0.2, 0.6]}   # you: "amplitude 0.2 to 0.6"
contacts: {value: 2}              # you: "two touches"
pause_ms: {range: [20, 40]}       # assumption: needs your OK
```

**Agent:** The pause is a placeholder you asked for; the numbers are mine, not from you or
a source, so I have marked it as an open assumption. It validates, and with the same seed
the first draw is now: a touch from 2.2 to 30.9 ms, a hold to 98.4 ms, a release to
108.1 ms, a pause to 128.3 ms, then the second touch, ending at 234.2 ms. The delay is the
same as before (2.2 ms) because the seed is the same, so you can compare the two files draw
by draw. What should the pause be?

**Assumptions listed:** `pause_ms 20 to 40`, **assumption: needs your OK**, open.

## Session 3: a value that needs a source

**You:** Put a vibration on the round touches, at the frequency this kind of receptor is
most sensitive to.

**Agent:** That frequency is a fact about the physics, and I should not supply it from
memory. There are two ways. You can tell me the frequency and what it rests on. Or I can
look for sources and write you a short report to approve, with links and the place in each
source where the number is; I have no search tool in this session, so I would ask you for
the sources. Which would you like?

**You:** I will tell you: 40 Hz, from my own recordings.

**Agent:** Recorded as yours, with your wording. This is the change; the sine modulation
and its frequency are the only new lines.

```yaml
layer:
  shape: {kind: gaussian}
  modulation: {kind: sine}
axes:
  sigma_mm: {range: [0.15, 0.45]}   # you
  frequency_hz: {value: 40}         # you (your recordings, chat, 2026-10-27)
```

| Value | Unit | Used in | Origin | Status |
|---|---|---|---|---|
| frequency_hz 40 | Hz | world_v4.yml, classes.dots.axes | you (your recordings, chat, 2026-10-27) | approved |

**Agent:** It validates, and the draws carry `frequency_hz: 40`. Had you asked me to find
a source and had I a search tool, the report would have had this shape, with nothing in it
that is not in a source:

```text
Need: the frequency of the vibration on the round touches (Hz).
Source 1: <link>, <section or table>. Says: <value and unit as the source states them>,
  under <the conditions it states>. Differs from this case in: <what differs>.
Source 2: <link>, <place>. Says: <...>.
Sources disagree on: <or "no disagreement found">.
Not established: <what no source gave>. Not read: <any source I could not open>.
My reading (a reading, not a source): <one sentence>.
Your decision: approve a value or a source, give your own, or leave it open.
```

**Assumptions listed:** none open; the frequency is from you.

## What to expect from your own sessions

- The agent asks one short question at a time and does not start a long run without you.
- Anything you leave open stays marked until you settle it, and the workflow stops for it
  whatever pace you chose.
- Your saved working style sets how long the plan is and how often the agent checks in; see
  [Working with a Coding Agent](agent_setup.md).
