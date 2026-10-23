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
| `test_pipeline_scale.py`, plain run | only on request | the same invariants at `SF_SCALE_DRAWS` draws (default 1000) of `SF_SCALE_DURATION_MS` ms (default 100) on the 8x8 test design, with the checks |
| `test_pipeline_scale.py`, matrix | only on request | both tactile recipes (`tactile_sa1_ra1`, `tactile_sa1_ra1_adex`, 80x80 sensors) on both test worlds (`tactile_small`, `elements_v1_2`), each with a `train` split of about `SF_SCALE_MATRIX_DRAWS` (default 40), a stratified `test` split and two sessions three times the draw length; `tactile_small` adds a held-out class split and probes |

The marker is `scale`. A test carrying it is skipped unless the run says `-m scale` or
`--run-scale` (the option and the skip are in `tests/conftest.py`, so `pytest -m "not gui"`, the
CI command, never runs one). Run the large test with:

```bash
SF_SCALE_DRAWS=5000 SF_SCALE_OUT=/data/scale_run pytest -m scale tests/scale -s
```

**Disk.** A bundle holds each population's receptive-field weights, so an entry on the 80x80
recipes takes about 47 MB (the data itself is about 1 MB); on the 8x8 design it is small. A run
of thousands of entries on a recipe therefore writes over a hundred GB. Before a large case
simulates anything, the test runs one real entry, measures it, and fails with the estimate if
the case would write more than `SF_SCALE_MAX_GB` (default 10) or leave less than
`SF_SCALE_MIN_FREE_GB` (default 20) free on the output volume. To run a bigger case, point
`SF_SCALE_OUT` at a volume with room and raise `SF_SCALE_MAX_GB`. Without `SF_SCALE_OUT` each case
writes to a temporary folder that is removed when it ends, so at most one case is on disk.
With it, the folders are kept.

The test prints the time of every stage and the draws per second of the batch stage; it asserts
none of them, and peak memory is not compared across runs at all (F-056). On the 8x8 test design
the batch stage ran at about 4 to 8 draws per second on one core, and 3 draws per second on the
80x80 recipes.

## Where the code is

`sensoryforge/evaluate/pipeline.py` (`run_pipeline`, `dataset_mapping`); the command is
`cmd_evaluate_run` in `sensoryforge/cli.py`. The schema is
`schemas/evaluate/pipeline.schema.json`. The tests are `tests/unit/test_eval_pipeline.py` and `tests/scale/`.
