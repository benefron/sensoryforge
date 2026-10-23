# Reference-free checks

The audit says what a data set *shows*, the declared side says what it *should* show, and
the comparison puts the two side by side ([the raw comparison](compare_stage.md)). Some
failures need none of that: they break a promise the simulator makes whatever the world
declares. The same seed must give the same arrays. A mirrored stimulus on a symmetric grid
must give a mirrored response. No array may hold NaN. The checks stage tests those promises
and writes one file, `checks.json`, with a pass or fail for each and the entries or probes
that failed. The format is in the [report contract](../reference/report_contract.md),
section 13.

The checks test the simulator, not the world, and they are not a verdict on the comparison.
They read no stage file, so they change none.

## Run it

```bash
python -m sensoryforge.evaluate.checks DATASET_DIR --out OUT_DIR --packs core,tactile
```

`DATASET_DIR` is the batch output (manifest, index, bundles). `--out` defaults to
`DATASET_DIR/evaluate`. The command prints a table and exits with status 0 when no check
failed (a skipped check is not a failure), 1 when one failed or raised an error, and 2 on a
usage or data error. The options:

| option | meaning |
|---|---|
| `--packs` | check packs, comma separated; default `core`. The tactile pack is named explicitly, like the audit's |
| `--sample N` | entries the re-run checks use (default 8), spread evenly over the entries that ran |
| `--seed N` | the seed of every probe (default 12345) |
| `--limit N` | read at most N entries in the per-entry data checks |
| `--voltage-bounds LOW,HIGH` | membrane-voltage limits in mV for `core.run.voltage`; none are assumed |
| `--param PACK.NAME=VALUE` | a pack parameter, e.g. `tactile.tolerance=1` |
| `--max-failures N` | failures listed per check (the count is always exact) |

From Python:

```python
from sensoryforge.evaluate.checks import run_checks

doc = run_checks("data/", "out/evaluate", packs=["core", "tactile"])
print(doc["counts"])            # {'pass': 14, 'fail': 0, 'skipped': 0, 'error': 0}
```

The evaluation server shows the file on its [checks page](evaluation_server.md) and can run
the stage from the browser.

## What the checks do

The table of the fourteen built-in checks, what each tries and when it passes, is in the
contract (section 13.1). In short:

- **Data checks** read every entry through the audit's blind reader: the entry ran and its
  bundle opens, nothing is NaN or infinite, counts are whole numbers, the time axis is
  regular. One pass over the entries serves all four.
- **Probes** build the engine from the first bundle's own `config.json` and feed it a
  synthetic stimulus: the same stimulus twice (determinism), a stimulus and its mirror
  (symmetry), nothing, one sensor, and a stimulus 1000 times the data's peak. The
  symmetry probe swaps each population's receptive fields for one neuron per sensor, so
  neuron *n* is sensor *n* and a mirror is exact; the neuron, filter and gain stay as they
  were, and every noise source is off.
- **Re-runs** take a few entries and run each population's neuron stage again from its
  stored `filtered` with the same function a run uses (`SimulationEngine._run_neuron_stage`,
  also behind the draw browser's voltage trace): the stored readout must be reproduced, a
  declared input floor must act as a clip of the neuron's input, and the voltage must be
  finite and, if you give limits, inside them.
- **The sampler check** samples the data set's own world at two lengths and by index, and
  needs only `dataset.json`.
- **The tactile pack** has one check: a held indentation at six levels of the data's peak
  must never lower the spike count of a tonic (`sa` filter) neuron by more than `tolerance`.

## Things worth knowing

- **Probe scale is the data's.** A probe's amplitude starts at the largest frame value of
  the sampled entries and doubles until the populations together count at least 20 spikes
  or events, so a symmetry or determinism comparison is never between empty arrays. The
  amplitude is in the check's `numbers`. If the sampled frames are all zero, the probes are
  skipped.
- **A skip says why.** A probe needs one regular grid and a configuration the engine can
  rebuild here: a receptive-field file that has since moved, or a composite grid, skips the
  probes with that reason. `core.run.input_floor` is skipped when no population declares a
  floor.
- **Wiring seeds.** A population with no `seed` for its innervation may draw new wiring
  at each build, and whether it does depends on the global random state's history. So
  `core.run.determinism` compares the freshly built engine on the seeded populations only
  (any difference there fails) and names every unseeded population in its `note`, whether
  or not it differed. Seeding a population brings it into the comparison.
- **Event converters.** `level_crossing` and `sigma_delta` with no refractory period emit
  `floor(|d| / θ)` events in one step by design, so `core.run.saturation` does not bound
  their count and names them in its `note`. With a refractory period of R integration
  steps, the bound is `ceil(steps / R)` per bin; a spiking neuron model's is one per step.
- **A model that fires with no input fails `core.run.empty_stimulus` on purpose**, so you
  see it. Noise is off in the probe.
- **A bundle written on another platform** can differ from a re-run in the last float step;
  `core.run.rerun_matches` then fails and lists the neurons. Use the tolerant comparison of
  `sensoryforge.testing.golden` when comparing across platforms.
- **Whole-batch reproducibility is not re-run here.** The checks show the engine is
  deterministic and that the neuron stage reproduces a stored bundle; rebuilding a bundle
  from the world's draw is the declared side's and the batch's own tests'.
- **Plugins.** A check pack is a `CheckPack` that returns `Check` objects; register it in
  `CHECK_PACK_REGISTRY` (or through the `sensoryforge.components` entry point). A `Check`
  sets `id`, `pack`, `title`, `what`, `scope` and implements `run(ctx)`, returning a
  `CheckResult` or raising `CheckSkipped`. A check may use the engine and the world; the
  audit may not (the import wall tests the audit and the metric packs, not this package).

## Where the code is

`sensoryforge/evaluate/checks/`: `base.py` (the interface and the context), `data_checks.py`,
`engine_checks.py`, `probe.py` (building and running probes), `world_checks.py`,
`core_pack.py`, `registry.py`, `run.py` and `__main__.py`;
`sensoryforge/evaluate/tactile/checks.py` for the tactile pack. The schema is
`schemas/evaluate/checks.schema.json`. The tests are `tests/unit/test_eval_checks.py`: every
check passes on the tactile and the toy data sets and fails on a seeded fault (an array
corrupted in a copy of the data set, or the engine patched to break one promise).
