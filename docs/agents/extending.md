# Extending the Agentic Workflow

This page is for people developing SensoryForge itself who want to add to the agentic
workflow: a new skill, support for another agent environment, or another sensing modality.
It assumes a clone with the development install (see
[Installation](../getting_started/installation.md)); the paths below are in that clone.
By the end you will know which files to change, which tests hold them, and what must be
regenerated.

Read [Agent Files](../developer_guide/agent_files.md) first: it maps every agent file, says
which are sources and which are generated, and sets the budget on how many there may be.

## Adding a skill

1. **Check that it needs a skill.** Most new work fits an existing skill's steps; the
   [file-count budget](../developer_guide/agent_files.md#file-count-budget) allows at
   most 14 skills, and a new one has to earn its place.
2. **Write `sensoryforge/agents/skills/sensoryforge-<verb>/SKILL.md`.** Front matter with
   `name` (the folder name) and `description` (what it does and when to use it, in the
   user's terms, since agents pick a skill by it). Then the body, following the
   [rules for every agent file](../developer_guide/agent_files.md#rules-for-every-agent-file):
   start with `sensoryforge agents brief`, name commands rather than copying their flags,
   use neutral words, and send any value the user has not given to
   `sensoryforge-ground-assumptions`.
3. **End it with a worked example that runs.** A `## Worked example` section with the files
   (`# file: NAME` as a block's first line) and the commands (`sensoryforge ...` or
   `python ...`), each followed by `# prints: TEXT` and, where it should fail,
   `# exit: N`. The test suite writes the files into an empty folder and runs the commands.
4. **Add it to the reference and the table.** If it belongs in the usual chain, add its
   name to `CHAIN_ORDER` in `sensoryforge/agents/reference.py`; then regenerate the
   [Skills Reference](skills_reference.md):

    ```bash
    python -m sensoryforge.agents.reference docs/agents/skills_reference.md
    ```

    Add a row to the table in [Skills for a Coding Agent](../user_guide/agent_skills.md#the-skills).
5. **Give it scenarios.** A fresh agent must be able to use it from a plain install:
   write scenarios for it as in [Agent Scenarios](../developer_guide/agent_scenarios.md#writing-a-scenario).
6. **Run the checks.**

    ```bash
    pytest tests/unit/test_agent_skills.py tests/unit/test_agent_names.py \
        tests/unit/test_agent_docs.py tests/unit/test_agent_export.py
    ```

    They parse every `sensoryforge` command in the skill with the real command-line
    parser, check every docs link and anchor, check that every registered name it uses
    exists, run its worked example, keep the reference page equal to the skills, and check
    that each environment's export of it is current.

## Supporting another agent environment

There are two routes, and only the second changes SensoryForge.

- **For your own use, with no change to SensoryForge.** Export the neutral files and give
  your agent the generic prompt; it finds where your environment keeps reusable
  instructions and writes the skills there in that format. See
  [An environment that is not listed](../user_guide/agent_export.md#an-environment-that-is-not-listed).
- **As a built-in export.** When an environment's documentation says where it reads skills
  and its instruction file, add one `register_target(Target(...))` call in
  `sensoryforge/agents/install.py`: its `--target` name, its label, its skills folder,
  the files whose presence suggests a project uses it, the documentation page each folder
  was checked against, and its instruction file and kind (a marked section in a file the
  user owns, or a whole file of SensoryForge's own). Add its case to
  `tests/unit/test_agent_export.py`, and record the check's date in `FORMATS_CHECKED` or
  `INSTRUCTIONS_CHECKED`. Do not add a target from a guess: if the documentation does not
  say, the generic route is the right one. The
  [maintainer notes on exports](../user_guide/agent_export.md#for-maintainers) have the
  details.

A prompt for an environment (`sensoryforge/agents/prompts/*.md`) is a hand-written
user-facing file and counts against the same budget as the skills. The docs page that shows a prompt must show it word for word; a test
fails when the two differ.

## Adding a modality

The skills are written in neutral words (sensor, channel, frame, population, draw, entry)
and find what exists by asking SensoryForge (`sensoryforge describe`,
`sensoryforge list-components`), so a new modality needs no change to a skill. It needs the
pieces the skills ask about:

- **Stimuli for it**, as world shapes, patterns or class kinds registered by a plugin
  ([Extending a world](../user_guide/worlds.md#extending)).
- **Metrics for it**, as a metric pack the audit can run
  ([Write a metric pack](../developer_guide/audit_metrics.md#write-a-metric-pack)), and a
  declared adapter so the comparison can read what the world declares
  ([Write an adapter for another modality](../developer_guide/declared_adapter.md#write-an-adapter-for-another-modality)).
- **Its terms in the glossary.** Add the words a user of that modality would say to the
  [glossary](glossary.md), in a section of their own as the tactile words are, so an agent
  can map them by meaning. The glossary test keeps tactile words inside their section, and
  fails when a new kind of registry has no glossary term.

The [toy vision fixture](../developer_guide/toy_vision_fixture.md) is a small non-tactile
example of these pieces, used by the tests to keep the core free of tactile assumptions.
Once a pack is registered, `sensoryforge describe --section evaluation` lists it, which
is where an agent looks.

New components for the pipeline itself (a neuron, a filter, a stimulus, a grid) are
[plugin packages](../developer_guide/plugins.md); the planning skill already hands off to
`sensoryforge new-component` when a plan needs one that does not exist.
