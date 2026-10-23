# The evaluation pipeline

`sensoryforge/evaluate/pipeline.py` chains the evaluation stages for
`sensoryforge evaluate run` (user view: [Evaluating a world in one
command](../user_guide/evaluate_pipeline.md)). The format of `pipeline.json` is in the [report
contract](../reference/report_contract.md), section 15.

## Order and rules

1. **build**: `build_dataset` and `write_dataset` (manifest, `dataset.json`).
2. **batch**: `run_dataset`, the same function as `sensoryforge batch --dataset`. Any entry
   that fails fails the stage.
3. **audit**: a subprocess, `python -m sensoryforge.evaluate.audit`, so the blind rule holds in
   practice, not only by import test.
4. **declared**: `run_declared` (in process; the adapters read the world).
5. **compare**: `run_compare` on the audit and declared files.
6. **checks**: `run_checks`, unless switched off.

Each stage is timed and recorded. A failing stage raises inside the pipeline, which stops, marks
the later stages `skipped`, and still writes `pipeline.json`. `run_pipeline` itself raises
`ValueError` only for names that are not registered (metric pack, check pack), checked before
anything is simulated, so a typo costs nothing.

The pipeline owns no metric. Which packs run is the caller's choice (contract Q9, Q12); when the
declared adapter has a same-named metric pack that was not run, a note says so.

## Scale tests

`tests/scale/` takes the tactile test world through the pipeline at sizes that stress the
stages, not their arithmetic.

| test | runs | what it checks |
|---|---|---|
| `test_pipeline_small.py` | in every CI job (about 25 s) | 24 draws through every stage: each entry is in the manifest, bundle, audit, declared and compare files; a resumed run changes nothing; a batch split into 3 tasks gives the audit rows of one process (wall-clock seconds aside) |
| `test_pipeline_scale.py` | only on request | the same invariants at `SF_SCALE_DRAWS` draws (default 1000) of `SF_SCALE_DURATION_MS` ms (default 100), with the checks |

The marker is `scale`. A test carrying it is skipped unless the run says `-m scale` or
`--run-scale` (the option and the skip are in `tests/conftest.py`, so `pytest -m "not gui"`, the
CI command, never runs one). Run the large test with:

```bash
SF_SCALE_DRAWS=5000 SF_SCALE_OUT=/data/scale_run pytest -m scale tests/scale -s
```

`SF_SCALE_OUT` keeps the data set and the report folder; without it they are removed. The test
prints the time of every stage and the draws per second of the batch stage; it asserts none of
them, and peak memory is not compared across runs at all (F-056). On the 8x8 test design the
batch stage ran at about 7 draws per second on one core, so 5000 draws take about 12 minutes.

## Where the code is

`sensoryforge/evaluate/pipeline.py` (`run_pipeline`, `dataset_mapping`); the command is
`cmd_evaluate_run` in `sensoryforge/cli.py`. The schema is
`schemas/evaluate/pipeline.schema.json`. The tests are `tests/unit/test_eval_pipeline.py` and `tests/scale/`.
