# File headers

An agent, or a person, who opens a module should learn what it is for, which tests pin it
and what not to change without, without searching. The module's docstring says what the
file is for; a short block of **tag lines** at its end says the rest. The tag lines are
plain text at the start of a line, so one search lands on the right file.

```python
"""Counter-based randomness for worlds (spec §4.1).

Every random choice in a world is a pure function of integers ... (the prose)

Topic: world
Tests: tests/unit/test_world_rng.py, tests/contract/test_world_contract.py
Entry points: seed53, draw_seeds, uniforms, permutation, mix, hash_parts, key
Pinned by: docs/reference/world_contract.md, .claude/rules/world-engine.md
Do not change without: the constants and the hashing never change: every draw of every
    world changes with them, and pressure-simulation pins the draws.
"""
```

A value is a comma-separated list, or one sentence. A line that starts with spaces
continues the tag above it. The tag block goes after the prose and the docstring's
`Example:` block, and a tag appears once.

## The tags

| Tag | Required | What it holds | What the test checks |
|---|---|---|---|
| `Topic:` | yes | One or more topics from the table below | each is in the list |
| `Tests:` | yes | The tests that pin this module (paths from the repository root) | each path exists |
| `Entry points:` | yes | The names an agent searches for: classes, functions, constants, and `Class.method` | each is defined in the module |
| `Pinned by:` | where one applies | A contract page, a path-scoped rule, a golden fixture, or a ledger id (`F-053`, `D-0437899`) that fixes how this module behaves | each is an existing path or a ledger id |
| `Do not change without:` | where one applies | One sentence: what must still hold, or what must change with it | not checked, so write only what you know is true |
| `Shapes:` | for tensor code | Tensor shapes in and out, batch first | not checked |
| `Units:` | for tensor code | Physical units | not checked |

The prose above the tags keeps the project's docstring style (Google style, tensor shapes
and units where a public API takes them); the tags add to it and repeat none of it. A
parameter list, a default or a unit table never goes in a header: name the command or the
page that holds it (`sensoryforge describe --json`).

## The topics

A topic is a word from one closed list, so `grep` finds every file on it. Add a topic here
and in `TOPICS` (`sensoryforge/testing/file_headers.py`) together.

| Topic | Files that are about |
|---|---|
| `engine` | running a config: the simulation engine and its pipelines |
| `batch` | running many stimuli or a data set, and writing results |
| `config` | the config schema, its defaults and their resolvers |
| `registry` | component registries and registration |
| `plugin` | plugin discovery and scaffolding |
| `cli` | the `sensoryforge` command line |
| `receptive-field` | receptive-field banks and their builders |
| `neuron` | neuron models and their solvers |
| `filter` | temporal filters |
| `stimulus` | stimuli and their rendering |
| `grid` | sensor arrays and their arrangements |
| `event-converter` | level-crossing and sigma-delta converters |
| `world` | worlds, sampling and rendering draws |
| `dataset` | data sets: splits, probes and entries |
| `bundle` | the data bundle a run writes |
| `evaluate` | the evaluation pipeline as a whole |
| `audit` | the blind audit of generated data |
| `compare` | comparing audited data with the declaration |
| `checks` | reference-free checks |
| `server` | the local evaluation server and its exports |
| `gui` | the Qt application |
| `agents` | agent skills, setup and instruction files |
| `recipe` | shipped recipes, presets and their gains |
| `testing` | test helpers shipped in the package |
| `legacy` | kept for old configs; new work uses the canonical path |

## Finding a file

```bash
grep -rn "^Topic:.*receptive-field" sensoryforge/     # every file on a topic
grep -rn "^Entry points:.*SimulationEngine" sensoryforge/   # which file holds a name
grep -rn "^Do not change without:" sensoryforge/      # every hazard, one line each
grep -rn "test_world_rng" sensoryforge/               # which modules a test pins
```

A long value continues on indented lines, which do not start with the tag. To search a
whole value, drop the `^Tag:` and search the bare name, or add `-A3`.

Function names follow a small fixed set of verbs, so the same search works across the
package. New public functions use them:

| Verb | Meaning | Examples |
|---|---|---|
| `build_*` | makes an object from a config or a spec | `build_dataset`, `build_filter` |
| `load_*`, `read_*` | reads a file or a store | `load_world`, `read_batch_index` |
| `write_*` | writes a file | `write_bundle`, `write_dataset` |
| `render_*` | turns a description into frames | `render_frames`, `render_stimulus` |
| `resolve_*` | fills in defaults | `resolve_filter_params` |
| `register_*` | adds a component to a registry | `register_shape`, `register_all` |
| `validate_*`, `check_*` | checks and raises or reports problems | `validate_file`, `check_component` |
| `run_*` | executes one stage | `run_audit`, `run_compare`, `run_pipeline` |
| `cmd_*` | a command-line command | `cmd_run`, `cmd_evaluate_run` |
| `main` | the entry point of a script or a window | `main` |

## Which modules need a header

Every module under `sensoryforge/` of 40 lines or more, except a package `__init__.py` and
a `__main__.py` (glue), needs a header. The modules that agents touch most have one now.
The rest are listed in `tests/unit/file_headers_untagged.txt`, a baseline that only
shrinks:

- a **new module** is not in the baseline, so `tests/unit/test_file_headers.py` fails
  until it has a header;
- a baseline module that gets a header must be **removed from the list** (the test
  tells you), so the list never hides a tagged file;
- a header must be **true**: the checker (`sensoryforge.testing.file_headers.check_module`)
  fails on a topic that is not in the list, a test or contract path that does not exist
  and an entry point the module does not define.

When you change a module, change its header in the same commit: a renamed entry point
fails the test, and a new hazard belongs in `Do not change without:`. When a finding in the
ledger closes, take its id out of `Pinned by:`.

To tag a baseline module, add the block, delete its line from the baseline, and run:

```bash
pytest tests/unit/test_file_headers.py -q
```
