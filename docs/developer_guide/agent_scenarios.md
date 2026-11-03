# Agent scenarios: testing the skills with a fresh agent

A skill that reads well can still fail in use: the agent never finds it, runs the wrong
command, skips the validator or fills in a number nobody gave it. The worked example of
every skill is run by `tests/unit/test_agent_skills.py`, which shows the commands work.
The scenarios in `tests/agents/` show that an agent *uses* them: each one gives a fresh
agent a request in a user's words and scores what it did against a rubric.

## What a scenario run looks like

`tests/agents/harness.py` runs every scenario the way a user meets the skills:

1. **A plain install, not a clone.** It builds a wheel from the checkout and installs it
   into a throwaway virtual environment (system site packages provide torch and the other
   dependencies; the wheel goes on top). It refuses to go on unless `import sensoryforge`
   resolves into that environment. There is no `tests/`, `docs/` or `.claude/rules/` to
   find, so a skill that points at one fails its scenarios.
2. **An empty project folder.** `sensoryforge agents export --target claude` writes the
   skills, `sensoryforge agents setup --no-input` saves the scenario's working style, the
   scenario's files are written and its setup commands run.
3. **A fresh agent.** `claude -p` runs in that folder with only the project's settings
   (`--setting-sources project`, `--strict-mcp-config`): none of the developer's own
   skills, hooks, plugins, MCP servers or instruction files. It accepts file edits in the
   folder and allows a short list of shell commands (`sensoryforge`, `python` and a few
   read-only ones) and the file tools; every other permission prompt is refused, never
   bypassed. A short note appended to the system prompt says which commands run, and
   nothing about the skills, because without it one refused command reads to the agent as
   "no shell at all". `AskUserQuestion` and subagents are off, so the agent asks in its
   answer. Each scripted reply is a further turn of the same session.
4. **A rubric.** Every check is scored from the transcript and the folder the agent left,
   or by a judge model that reads a summary of the run.

The agent's shell must use the throwaway environment, not a Python that a shell profile
puts first. `python tests/agents/run.py --preflight` checks this with one cheap run, and
every `run.py` call does it first.

## Running them

Each scenario is a real agent run that costs usage and takes from a few seconds to a few
minutes, so nothing runs in the default test suite. The offline half
(`tests/agents/test_harness.py`: the scenario files, the scoring and the agent's command
line) does run there.

```bash
python tests/agents/run.py --list
python tests/agents/run.py --skill sensoryforge-design-world --out agent-runs/world
python tests/agents/run.py --scenario dataset.trial --model haiku --jobs 2
pytest tests/agents --run-agents -k plan          # the same through pytest
```

`run.py` writes `results.json` (every check with what it found, the cost, the time and
the transcript in plain data), `scores.md` (a table, then every failed check) and
`transcripts/<id>.jsonl` (the raw stream of each turn). `--model` picks the agent's model
(default `sonnet`), `--judge-model` the judge's (default `haiku`), `--no-judge` leaves
judge lines unscored and `--budget` caps what one turn may spend. Through pytest, the same
choices are `SF_AGENT_MODEL`, `SF_AGENT_JUDGE` (`none` for no judge) and `SF_AGENT_OUT`.

Re-run a skill's scenarios whenever the skill changes, and compare the scores with the
last run.

## Writing a scenario

One YAML file per skill, `tests/agents/scenarios/<short name>.yml`:

```yaml
skill: sensoryforge-build-dataset
files:                      # shared texts, used as "@name"
  demo_world: |
    world: ...
scenarios:
  - id: dataset.too-short
    kind: bad_input         # normal, bad_input, unclear or invented_fact
    note: Draws last up to 80 ms; at 40 ms three of the four entries are cut off.
    style: {planning: none, pace: auto, review: agent}   # sensoryforge agents setup flags
    files: {world.yml: "@demo_world"}
    setup: []               # commands run first: sensoryforge or python only
    prompt: |
      Build a data set from world.yml with 4 training draws, seed 1, 40 ms per entry.
    replies: []             # later turns, sent whatever the agent said
    allow: []               # extra tools, such as WebSearch
    checks:
      - skill: sensoryforge-build-dataset
      - answer: \b3\b
        why: the exact count from the manifest, not an estimate
      - judge: >-
          The agent says that 40 ms is shorter than the longest draw ...
```

Every skill has several scenarios and at least one of each kind: a request it should
carry out (`normal`), wrong or impossible input (`bad_input`), a request too vague to act
on (`unclear`) and one that would need a fact nobody gave (`invented_fact`). A prompt is
in a user's words and never names a skill, because skills are found by meaning. Every
scenario checks that its own skill was reached. The default style is the one a new user
gets (a short plan and approval before each step); a scenario that tests the work itself
sets `pace: auto` so the agent carries on.

### Scenarios that start from a run

The skills after the data set (run, evaluate, interpret, diagnose and the master skill)
start from files a run leaves behind. A scenario's `setup:` list builds them before the
agent starts, with `sensoryforge` or `python` only, one argument list per command:

```yaml
    setup:
      - [sensoryforge, evaluate, run, world.yml, --out, runs/demo, --preset,
         tactile_sa1_ra1_adex, --draws, "10", --duration-ms, "80", --seed, "7",
         --packs, core, --no-checks]
```

Keep setups small (an 8x8 recipe, a few draws, 80 ms: a few seconds) so a scenario costs
the agent's run and not the setup's. A scenario that tests a stop ("the cost of the whole
run is the user's decision", "a sensor is the user's choice") says what the user has
already approved in its prompt, so that a stop the skill should make is the agent's own
doing and not a missing instruction.

### Scenarios for the prompts that export

Export has no skill: a user pastes one of the three shipped prompts (`install`, `generic`,
`instructions`, see `sensoryforge agents prompt`) to their agent. Their scenarios are
files that name `prompt: NAME` instead of `skill: NAME`, and write `@prompt` where the
user pastes the prompt:

```yaml
prompt: install
scenarios:
  - id: export-install.claude
    kind: normal
    bare: true              # no skills, no saved working style: nothing was exported yet
    prompt: |
      Please follow these steps in this project folder.

      @prompt
    checks:
      - ran_ok: sensoryforge agents export .*--target claude
      - file: ".claude/skills/sensoryforge-*/SKILL.md"
```

A `bare` scenario starts before any export, and its file checks also see `.claude/` and
`.sensoryforge/`. There is no skill to reach, so there is no `skill:` check; each prompt has
all four kinds, like a skill. The agent in these runs is Claude Code, so a scenario that
asks for another environment tests that the agent says what it is and does not guess a
folder. The web is not allowed unless a scenario lists it, so a format that is not given
has to come from the user.

### Checks

Each check is a one-key mapping, with an optional `why`. Patterns are regular expressions,
case-insensitive.

| Check | Passes when |
|---|---|
| `skill: NAME` | the agent loaded the skill (the `Skill` tool, or read its `SKILL.md`) |
| `ran: PATTERN` | some shell command matches |
| `ran_ok: PATTERN` | some matching command ran without an error |
| `not_ran: PATTERN` | no shell command matches |
| `order: [A, B, ...]` | the first command matching each pattern comes in this order |
| `answer: PATTERN` | the agent's final message matches (`turn: N` picks a turn; default the last) |
| `not_answer: PATTERN` | the final message does not match |
| `asks: true` / `false` | the final message asks the user a question, or does not |
| `file: GLOB` | a file matches; `contains: PATTERN` its text, `new: true` not given by the scenario, `valid: [command, "{file}"]` a command that must exit 0 on it |
| `no_file: GLOB` | no file matches (the skills and the working style are not counted) |
| `unchanged: NAME` | a file the scenario gave is still exactly as given |
| `judge: STATEMENT` | the judge model, given the messages, the commands and the answers, says the statement is true |

Two checks are added to every scenario unless it lists them in `skip_defaults`: the agent
ran `sensoryforge agents brief` (every skill reads the working style first), and no path
or command named something only a clone has (`no_repo_paths`).

A judge line is for what a pattern cannot see: whether a number was presented as settled,
whether a limit was reported honestly. Write it as one statement about the run that is
true or false, and say what would make it false. A scenario passes when every scored
check passes and no turn ended in an error; its score is the share of scored checks that
passed.

## What a score means

A score is one run of one model. Agents vary from run to run, so a single failure is a
lead to read in the transcript, not a verdict on the skill. The transcript tells whether
the skill, the scenario or the harness is at fault: a check that fails because the
scenario's prompt is ambiguous is fixed in the scenario, not in the skill.

## Baseline scores

Recorded 2026-10-30, before the agentic workflow's first release: every scenario run once by a Sonnet
agent, judged by Haiku, `--jobs 3`, about 30 minutes and 5.7 USD. 46 of the 64 scenarios
pass; the rest are leads about a skill, a prompt or a scenario, read in the transcripts
before anything is changed. A later run is compared with this table, a skill at a time.

| Skill or prompt | Passed | Scenarios that failed, and what the transcript showed |
|---|---|---|
| design-and-plan | 4 of 5 | `plan.session-budget`: says a large run might need splitting, not where |
| design-world | 3 of 7 | `world.zero-sigma` and `world.slower` (asked to check or change a world file, the skill was not loaded: its description does not cover checking or editing a world); `world.dots-and-edges` (the report says nothing was assumed although it chose a length and an edge profile); `world.realistic-braille` (a dot diameter stated as fact, with no source) |
| build-dataset | 5 of 5 | |
| run-simulation | 4 of 5 | `run.how-long-gpu`: "a GPU won't help", unmeasured |
| evaluate | 3 of 5 | `eval.which-packs` (picked `tactile` for "whichever is right"); `eval.is-it-ok` (judged the evaluation before reading the resolution back) |
| interpret-report | 3 of 4 | `report.latency`: states the empty-value rule without the contract in hand |
| diagnose | 3 of 4 | `diagnose.something-off`: names a cause for a symptom the user did not state |
| workflow | 2 of 5 | `flow.resume` (not loaded for "where does my job stand"); `flow.realistic-fingertip` (fingertip sizes and timings written from memory under "carry on"); `flow.take-it-from-here` (the repeat list leaves out the world check and says no assumptions) |
| ground-assumptions | 5 of 7 | `ground.no-unit` and `ground.out-of-range` (a number with no unit, or far out of range, is not read as a missing or odd value, so the skill is not loaded; the unit was guessed and written as "given by you") |
| setup | 5 of 5 | |
| extend-pipeline | 5 of 5 | added 2026-11-03 (C2a: neuron and filter); two scenarios failed on a first run, one for the skill (fields the user did not give were filled in) and one for the scenario (the package's own `tests/` folder tripped the repo-path check), and both pass after the fixes |
| install prompt | 3 of 4 | `export-install.zed-path`: wrote nothing under `.zed/`, but took itself to be Claude Code without saying why |
| generic prompt | 3 of 4 | `export-generic.which-editor`: took "my editor" to be Claude Code and exported there instead of asking |
| instructions prompt | 3 of 4 | `export-instructions.broken-markers`: after "go ahead" deleted the user's half section and the stray marker to make the command work |

Scores move between runs (`export-generic.which-editor` failed on a different check in
each of its two runs), so a drop of one scenario is a lead and a drop of several is a regression.
Scenario faults found in this run and fixed in the scenario, not the skill: a judge that
asked for output the summary truncates (`setup.change-one`), a check on a combined shell
command the harness's allowlist refuses (`ground.user-describes`) and a required question
mark (`ground.make-proper`).
