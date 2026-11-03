# R-B6 · Review of the core skills

Date 2026-10-25 · reviewer Claude Opus 5.5 · branch `phase/b-agentic` at `d4d0888` · scope
roadmap rows B6a–B6d: the nine skills in `sensoryforge/agents/skills/`, the harness
`sensoryforge/testing/skill_examples.py`, `sensoryforge/agents/{install,hook,style}.py` as
far as the skills use them, `tests/unit/test_agent_skills.py`, and the docs pages
`user_guide/agent_skills.md` and `agent_setup.md`.

**Verdict:** the skills are general and self-contained. They mark where every value comes
from, and every worked example runs. They are not yet safe to build B7 on, for one reason in
the code and two in the skills. First, `--resume` skips any entry whose bundle exists. It
does not check which data set or sensor wrote the bundle, and the skills make `--resume` the
normal move. Second, the chain's folders do not line up: the data set is built in one
folder, the run writes to another, and the two skill scripts that read a run folder expect
the manifest inside it. Third, `diagnose` step 3 gives a command that fails, and even with
the flag fixed it prints draws that are not the data set's. The worked examples never caught
these, because only the example section is run. Row **B6f** (Sonnet, start of S18, before
B7) makes the changes listed under "Required changes". Two questions are Ben's (see
"Decisions for Ben").

## How this was checked

- All nine `SKILL.md` files read whole, against the code they name: `cli.py` (`batch`,
  `world sample`, `dataset build`), `world/runner.py::run_dataset`,
  `evaluate/pipeline.py::run_pipeline`, `agents/style.py`'s option texts, `agents/hook.py`
  and `agents/install.py`.
- Every `sensoryforge ...` command in the skills' bodies parsed with `cli.create_parser()`
  (scratch script). Placeholders such as `N` and `S` give false alarms. Once those are
  set aside, one real failure is left: `world sample --n`.
- Reproduced in a scratch folder: a data set's entries against `world sample` with the
  data set's seed; the run-simulation layout (`batch --output runs/demo`) read by the
  workflow skill's `where.py`; and `batch --resume` into that folder with a different data
  set and a different recipe.
- Each URL the skills cite was fetched, and its anchors checked.
- The full suite, once, in the background (result in the R-B6 session log).

## What holds

- **Neutral words.** The skills say sensor, entry, draw, population, channel. Tactile words
  appear only in the world file's own field names (`touch_ms`), in the shipped recipe's name,
  and in one sentence that names the tactile pack as an example.
- **No invented facts.** Every skill sends a value with no origin back to the user: the plan
  table's Origin column, `# assumption: needs your OK` in a world file, "measured on one
  entry" for costs, and no tolerance for any difference. `interpret-report` names the three
  places "typical" may come from and forbids memory. `diagnose` never edits a shipped recipe.
- **Self-contained.** No skill names a path that exists only in a clone, and none says
  `pip install sensoryforge` (tests hold both). Each skill checks that `sensoryforge` is
  installed and stops when it is not. The routing hook uses only the standard library and
  is off until approved.
- **Pace.** The five closing lines ("offer the next step at the pace the brief sets; a
  failure or an open assumption is asked about whatever the pace") agree with the brief's
  `auto` line ("stop only for a question the user must answer"). The workflow skill's split
  into *offers* (skipped at pace `auto`) and *decisions* (never skipped) is the right model.
- **Tested.** Nine skills, nine worked examples, each run by `test_agent_skills.py` in a
  scratch folder through the worktree's own package. The rules (name, description length,
  line budget, glossary words, skill names) are tests, not just guidance.
- **The missing-component hand-off** in `design-and-plan` points at commands and pages that
  exist (all URLs return 200, the `#extending` and `#data-sets` anchors exist). The glossary
  URL is 404 until phase B is on main, as the B6 note expects.

## Findings

| # | Severity | Where | Finding |
|---|---|---|---|
| 1 | high | `world/runner.py:280-312`; run-simulation 3–5, evaluate 1–2, workflow example | `--resume` skips every entry whose `data.h5` exists, with no check of the folder's `batch.json`. Entry ids (`train/r0/00000`) repeat across data sets. Reproduced: a folder simulated from data set `d-a910241f4de9` on `tactile_sa1_ra1_adex`, then resumed with `d-217ed68836ca` on `tactile_sa1_ra1`, prints `0 ok, 0 failed, 3 skipped`. Its `batch.json` now names the second data set and Izhikevich, while every bundle holds the first data set on AdEx. `evaluate run --resume` takes the same path, so its declared side then describes draws the bundles do not hold. |
| 2 | medium | diagnose step 3 | `sensoryforge world sample WORLD --n N --seed S` exits 2 (`unrecognized arguments: --n`; the flag is `-n`). With `-n`, the draws still do not match the data set's: delays 36.5/30.3/39.7 ms from `world sample --seed 7` against 46.7/39.6/42.7 ms for the data set's entries (seed 7). A data set derives each entry's draw seed, so `world sample` with the data set's seed is a different sample. The agent is sent to the wrong draws. |
| 3 | medium | build-dataset step 3 | The duration check samples 200 fresh draws with seed 0, which are not the data set's draws, so it can miss a truncated entry or report one that is not there. The built manifest has an exact `truncated` flag per entry, and diagnose.py already reads it. |
| 4 | medium | build-dataset 4, run-simulation 1, workflow `where.py`, diagnose `entry` | The folders do not line up. build-dataset writes the manifest to `data/NAME`. `batch --dataset` rebuilds the data set from the spec, never reads that manifest and writes none into `--output` (`runner.py:280`). `evaluate run` writes it into `--out`. So run-simulation's "a built manifest" input is never used, and on a `batch` folder `where.py` prints `data set built: not yet` and `simulated: 3 of 0` (reproduced). diagnose.py's `entry` mode opens `DIR/manifest.jsonl` and fails on a folder that only `batch` wrote. The workflow example works only because it builds into the run folder. |
| 5 | medium | harness, `skill_examples.py` | Only the `## Worked example` section runs. The commands in a skill's steps are never checked, and finding 2 slipped through that way. The `python -c` line in run-simulation step 5 is not checked either. |
| 6 | medium | `cli.py:702` (F-c012944, open) | `batch --dataset --dry-run` ignores the flag and simulates every entry. The skill warns about it, but a user or another agent who trusts the flag runs the whole data set. The fix is a few lines. |
| 7 | low | workflow step 2 and table; `style.py` `planning`/`review` | The skills and two working-style options disagree. `planning: none` and `brief` ask for no approval, but the workflow always waits for the plan's approval. `review: user` ("show each step's result before the next") is in every brief, yet no skill says how it combines with `pace`. At the defaults, pace `ask` hides this. At `pace: auto` with `review: user`, the brief says both "carry on" and "show before the next". |
| 8 | low | `style.py` brief, "Words" line | Every user's brief points the agent at the task index URL. Ben decided not to ship the task index because it names `tests/`, `docs/` and `.claude/` paths that exist only in a clone (c041552). From a plain install it is the wrong page to send an agent to. AGENTS.md already points developers at it. |
| 9 | low | workflow example plan | The shown plan fills "4 entries, seed 7" with no origin. design-and-plan step 3 forbids exactly that, and agents copy examples. |
| 10 | low | `test_agent_skills.py` | Package data is checked only as strings in `pyproject.toml`, and every example runs from the source tree. A wheel that lacks the skills or the glossary would pass. The decided rule is to test what a user does from a plain install. |
| 11 | low | `install.py:97-105` | `agents install` overwrites an installed skill that the user edited and reports it as `updated`. That looks the same as a release upgrade. The docs say install replaces the copies, but they do not say that local edits are lost. |
| 12 | low | diagnose 4.3, interpret-report 2 | Metric definitions come from the docs site. Offline the skills have none. Online the site documents main, not the installed release, so a user on an older install can read newer definitions. |

## Answers to the B6d log's questions

- **Pace wording against `style.py`'s `auto` line:** consistent (see "What holds"). The
  conflict lies elsewhere: `planning` and `review` (finding 7).
- **Report contract from the site, and offline:** the fallback is honest. The skill says it
  cannot open the page and gives only what the folder shows. The real cost is version skew,
  not offline use (finding 12; a decision for Ben below).
- **"Typical" only in wording, not in a test:** right as it stands. A unit test cannot check
  whether an agent judges. B9b is the place for it: a fresh agent asked "is this latency
  normal?" must name which of the three sources it used and give no verdict.
- **Move any of C2 earlier (the B6 note):** no. The hand-off to `sensoryforge new-component`
  and the plugin template covers the five kinds it can scaffold, and it points at pages that
  exist. C2c's world-registry skill needs C1 (F-5bd542f) first. Moving C2 into phase B would
  make phase B's merge larger, and the skills would reach main later (the glossary URL 404s
  until then). Instead, B9a should test the hand-off with a "the component I need does not
  exist" scenario.

## Decisions for Ben

1. **`planning` and `review` against the workflow (finding 7).** Recommended: the brief
   wins for the plan, so the workflow waits for approval only at `planning: full`. A plan that
   holds an open decision always waits, because that is a decision. `review` sets who checks
   a step's result and `pace` sets whether to wait. At `review: user` with `pace: auto` the
   agent shows each result and carries on. One sentence in the workflow skill's step 4 and
   one in `agent_setup.md`.
2. **Where metric definitions come from (finding 12).** Recommended: ship
   `docs/reference/report_contract.md` as package data beside the glossary, with a test that
   keeps it identical to the docs page. `interpret-report` and `diagnose` would then read the
   installed copy first and the site second, and definitions would match the installed code.
   This needs the F-dc0c574 pass first (roadmap row names in about 40 places), so it would go
   in D5 (generated reference) rather than B6f. The alternative is to keep the site, which
   is honest but can drift from the installed version.

## Required changes (row B6f)

1. **Resume guard (finding 1),** in `world/runner.py::run_dataset`. With `resume=True` and
   an existing `batch.json`, refuse with a `ValueError` naming both values when its
   `dataset_id`, `world_id` or `config` differs from this run's. Compare the config without
   `simulation.device`, so a resume on another device is allowed. Do not overwrite
   `batch.json` before the check. No draw, frame, seed, entry id or bundle field changes, so
   a correct run is unaffected; say so in the commit and run
   `tests/contract/test_world_contract.py`. Tests: a resume with another data set and one with
   another recipe both refuse and leave the old `batch.json`; a resume with the same inputs on
   another device skips as before; `evaluate run --resume` reports the refusal as a failed
   stage. run-simulation step 3 and evaluate step 1 then say: same data set and sensor only;
   otherwise use a new folder.
2. **`--dry-run` on `--dataset` (finding 6; closes F-c012944).** Build the sensor config and
   the data set, print the entry count, the splits, the sensor and the output folder, simulate
   nothing and exit 0. Test that no bundle and no `index/` row is written. Remove run-simulation
   step 2's warning and offer `--dry-run` there instead.
3. **diagnose step 3 (finding 2).** Remove the `world sample` sentence. The draw is in the
   manifest row (`diagnose.py DIR entry ENTRY`), and the stimulus is in the bundle's frames.
4. **build-dataset step 3 (finding 3).** After the build, count manifest rows with
   `"truncated": true`, with `count.py` printing that count, and report it. Before the build,
   the longest a draw can take, worked out from the world's ranges, is a useful estimate. Label
   it as one (the diagnose example's 60 + 15 + 40 + 15 is that reasoning).
5. **One folder per run (finding 4).** State one convention in build-dataset step 4 and
   run-simulation step 1: build the data set into the folder the run will write
   (`dataset build ... --out runs/NAME`, then `batch ... --output runs/NAME`), which is the
   layout `evaluate run` already makes. run-simulation step 1 then calls the manifest a check
   that the data set builds, not an input. `where.py` reports a `batch` folder with no manifest
   correctly ("simulated: K entries; data set not built here"). diagnose.py's `entry` mode
   says what is missing instead of raising. Update the worked examples to match.
6. **Check the commands in a skill's body (findings 5 and 10).** A test extracts every
   `sensoryforge ...` command from each `SKILL.md`, in inline code and in fenced blocks. It
   replaces each upper-case placeholder from one small table (`N` → `1`, `S` → `0`,
   `MS` → `10`, `FILE`/`WORLD`/`DIR`/`NAME` → a word) and parses the result with
   `cli.create_parser()`. A placeholder missing from the table fails the test, so new ones get
   added. `python -c` lines must compile. Every `https://benefron.github.io/sensoryforge/<path>/`
   URL must map to a page under `docs/`, and its `#anchor` to a heading on it. A second test
   builds a wheel (`pip wheel . --no-deps`) and checks that every `SKILL.md` and the glossary
   are inside. Mark it slow if it takes more than a few seconds.
7. **Origins in the workflow example's plan (finding 9):** "4 entries (you), seed 7 (you)".
8. **The brief's task-index pointer (finding 8):** drop it from `render_brief`. The glossary
   stays. Update `agent_setup.md` if it quotes the line.
9. **Decision 1 above**, once Ben has answered.
10. Docs: `agent_skills.md` (the folder convention and what `--resume` refuses),
    `cli.md` (`--dry-run` with `--dataset`, the resume refusal). Close F-c012944 and record the
    resume guard as a `Fixed:` trailer.

Deferred, with where they go: finding 11 to B10 (keep the agent files current: record the
installed copies' hashes and refuse to overwrite a locally edited skill without `--force`).
Finding 12 to D5, if Ben agrees. The "typical" scenario to B9b. The missing-component
scenario, and one scenario on a non-tactile sensor config written in the scenario itself,
to B9a. Every runnable example uses a tactile recipe today, because no other kind ships.
