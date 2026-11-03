# Evaluating a world in one command

`sensoryforge evaluate run` takes a world (or a data set) to a report folder in one step. It
builds the data set, simulates one bundle per entry, then runs the evaluation stages the
[report contract](../reference/report_contract.md) defines: the audit, the declared side, the
comparison and the reference-free checks. It measures nothing itself; each stage is the one
you can also run alone.

## Run it

From a world file, with a sensor (a design directory, a preset or a config file):

```bash
sensoryforge evaluate run my_world.yml --out runs/first \
    --design designs/my_sensor --draws 2000 --duration-ms 300 --seed 7 \
    --packs core,tactile
```

From a data-set file (it declares its own splits, held-out classes, sessions and duration, so
`--draws` and `--duration-ms` are refused):

```bash
sensoryforge evaluate run my_dataset.yml --out runs/second --preset tactile_sa1_ra1_adex \
    --packs core,tactile
```

| option | meaning |
|---|---|
| `--out DIR` | the data-set folder (required): manifest, `index/`, one bundle per entry |
| `--design`, `--preset`, `--config` | the sensor, as for `sensoryforge batch --dataset` |
| `--draws N`, `--duration-ms MS`, `--seed S` | for a world file: one `train` split of N draws (seed default 0). The duration has no default |
| `--splits a,b` | simulate only these splits |
| `--packs` | metric packs of the audit (default `core`) |
| `--adapter` | the declared adapter (default: the first that accepts the data set) |
| `--no-checks`, `--check-packs`, `--check-sample` | skip, choose or size the reference-free checks |
| `--resume` | skip entries whose bundle already exists |
| `--device` | `cpu`, `cuda` or `mps` for the simulation |

!!! note "Packs are named, never detected"
    The pipeline does not look at the world to decide which pack to run (contract Q9). For a
    tactile data set, name it: `--packs core,tactile`. If the declared adapter has a pack of
    the same name that you did not run, the pipeline says so in its notes and prints it.

## What you get

Everything lands in `OUT/evaluate/`:

| file | stage |
|---|---|
| `audit.json`, `audit_entries.jsonl` | the audit (measured from the generated data alone) |
| `declared.json`, `declared_entries.jsonl` | what the world file says each entry should show |
| `compare.json`, `compare_entries.jsonl` | measured, declared and their raw difference |
| `checks.json` | the reference-free checks (unless `--no-checks`) |
| `pipeline.json` | this run: inputs, each stage's status and time, counts, the sha256 of the files above |

The command prints a line per stage and the stage times, then the check counts. Look at the
result with `sensoryforge evaluate serve OUT`, or print a comparison table with
`python -m sensoryforge.evaluate.compare OUT/evaluate OUT/evaluate`.

## When something fails

A stage that cannot finish stops the pipeline: the stages after it are recorded as `skipped`,
`pipeline.json` has `"status": "failed"` and an `error` naming the stage and the reason, and
the command exits with status 1. The files of the stages that did finish stay in the folder.
If some entries fail to simulate, the batch stage fails (the reasons are in `OUT/index/`);
fix the cause and run again with `--resume` to simulate only what is missing.

The command also exits with 1 when a check failed, so it can gate a script. A passing run
exits with 0.

## The audit stays blind

The pipeline runs the audit as `python -m sensoryforge.evaluate.audit` in a separate
interpreter. The process that reads the generated data therefore never has the world
package loaded, whatever the rest of the pipeline imports.

## Trying it at size

To see how long a large run takes on your machine, run the repository's scale test, which
uses the tactile test world: `SF_SCALE_DRAWS=5000 pytest -m scale tests/scale -s`. It prints
each stage's time. A recipe bundle takes about 2 MB per entry (its receptive-field weights are stored sparsely; before schema 2.5.0 it was 47 MB), so 3000 draws are about 6 GB. The test still measures one entry first and refuses a run that would write more than 10 GB (`SF_SCALE_MAX_GB`) or fill the disk; point `SF_SCALE_OUT` at a larger volume for more. A 24-draw version of it runs in every CI job. See the [developer
note](../developer_guide/pipeline_stage.md#scale-tests).

## From Python

```python
from sensoryforge.evaluate.pipeline import dataset_mapping, run_pipeline
from sensoryforge.io.design import load_design
from sensoryforge.world.dataset import load_dataset

spec = load_dataset(dataset_mapping("my_world.yml", draws=500, seed=7, duration_ms=300))
doc = run_pipeline(load_design("designs/my_sensor"), spec, "runs/first",
                   packs=["core", "tactile"])
print(doc["status"], doc["counts"])
```

See the [developer note](../developer_guide/pipeline_stage.md) for the format and the code.
