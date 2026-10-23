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

## Where the code is

`sensoryforge/evaluate/pipeline.py` (`run_pipeline`, `dataset_mapping`); the command is
`cmd_evaluate_run` in `sensoryforge/cli.py`. The schema is
`schemas/evaluate/pipeline.schema.json`. The tests are `tests/unit/test_eval_pipeline.py`.
