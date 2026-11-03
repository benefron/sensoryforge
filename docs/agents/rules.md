# Rules for Agents

This page is for anyone deciding whether to let a coding agent work on their SensoryForge
project. It states, in plain words, the rules every SensoryForge skill follows, and what an
agent will and will not do without your approval. The skills carry these rules themselves,
and the brief every skill reads first (`sensoryforge agents brief`) repeats the ones that
depend on your setup, so they hold in any environment.

## The rules

1. **Ground it or ask.** An agent never invents physics, models, parameters, ranges or
   units. It takes them from you, from a source you have linked and approved, or from
   reasoning you have approved. Where it has none of these, it asks. When it proposes a
   value to keep the work moving, it marks the value as an assumption, and every output
   lists its assumptions with where each came from: you, a source you approved, reasoning
   you approved, or "assumption: needs your OK" while nobody has approved it yet. The
   `sensoryforge-ground-assumptions` skill finds sources and writes a short report with
   links for you to approve; see
   [Designing a World by Describing It](../user_guide/agent_design_loop.md) for three
   recorded sessions.
2. **Meaning, not strings.** An agent reads your words by what you mean and by everything
   said in the session, not by matching them against SensoryForge's names. "Bumps",
   "dots" and "point contacts" can all mean one stimulus class; "the response" can mean
   spikes or the filtered current depending on what came before. It maps your words to the
   terms in the [glossary](glossary.md), and asks one short question when the meaning is
   unclear.
3. **Self-contained by default.** The skills ship with SensoryForge and depend on nothing
   else on your machine: not your editor, your shell setup, your hooks or your other tools.
   Out of the box they use only SensoryForge's commands and the agent's basic file and
   shell tools.
4. **Integration is opt-in.** Using your own skills and tools, the routing hook and your
   working style are offered at [setup](../user_guide/agent_setup.md) and stay off until
   you say yes. Nothing is installed into your environment's settings without that yes.
5. **You set the pace.** Whether a chain of skills goes step by step or end to end is your
   choice, in the plan or in chat, and it holds until you change it. The pace changes only
   whether the agent asks "shall I go on?"; the decisions below are asked for either way.
6. **No verdicts that are yours.** An evaluation reports what was measured, what was
   declared and the raw difference. It states no tolerance. Whether a difference is
   acceptable, and what counts as typical, comes from you, from the data set's own
   spread, from the report contract or from a source you approve, never from the agent's
   memory of what a neuron or a sensor usually does.

## What needs your approval

| Without asking | Only after you approve | Never |
|---|---|---|
| Read your working style, the glossary and SensoryForge's own descriptions (`sensoryforge describe`, `sensoryforge list-presets`, `--help`) | A plan, before it is carried out (unless your setup says to work without one) | Invent a physical value, a model, a range or a unit |
| Write and validate files in the run folder it is working in | Any value the plan marks as open or as an assumption | Judge a difference or call a value typical from memory |
| Sample and preview a few draws of a world | Data set sizes, seeds, the sensor, the metric packs | Write a missing component's physics |
| Time one entry of a run to measure its cost | The rest of a run that takes noticeable time or disk space | Change your project's instruction file outside SensoryForge's marked section |
| Offer the next step | Going on after a failure | Install a hook or use your other tools without a yes at setup |

Your own instructions in a session win over your saved working style, and both win over
the skills' defaults. A rule above is never relaxed by the working style: there is no
setting that lets an agent invent a value.

## Where the rules come from

The rules were set when the agentic workflow was designed; [Design Decisions](decisions.md)
lists each with the reasoning and the options that were turned down. The tests that hold
them are described in [Skills for a Coding Agent](../user_guide/agent_skills.md#how-they-are-kept-honest)
and, for agents tested live, in [Agent Scenarios](../developer_guide/agent_scenarios.md).
