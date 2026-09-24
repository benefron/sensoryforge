# `tests/fixtures/reference/` — provenance

## What is (and is not) here

This directory holds two reference fixtures, built at different times under
different constraints -- **both still stand**, and this file documents each:

- `sa_ra_qualitative_reference.json` records **qualitative published
  bounds** on SA1/RA1 mechanoreceptor afferent responses to a ramp-and-hold
  skin indentation, hand-derived from cited literature. It does **not**
  contain TouchSim output (see "Why not genuine TouchSim output" below --
  that section describes the situation as of when this fixture was built;
  it has since changed, see the next section).
- `touchsim_ramp_hold.json` (added 2026-09-24, ledger D-4aafcdc) **does**
  contain genuine TouchSim output -- see "`touchsim_ramp_hold.json`" below.

## Why not genuine TouchSim output (historical -- true when this fixture was built)

Wave S (`docs/development/handover/phase4_tasks.md`, Fact P4-a) asks for a
comparison against TouchSim's published response to a probe indentation.
Producing that comparison honestly requires either:

1. Running the `touchsim` Python package (Saal, Delhaye, Rayhaun & Bensmaia,
   *Simulating tactile signals from the whole hand with millisecond
   precision*, PNAS 2017) ourselves and committing its output, or
2. Digitising numeric trace data directly from that paper's published
   figures.

Neither was possible in this environment:

- `touchsim` is confirmed absent from the project environment (Fact P4-a)
  and Wave S is explicitly forbidden from adding it as a dependency
  (F-053: no `pip install`/`conda install`).
- This environment has web search/fetch access but no way to execute
  `touchsim` itself (it is not installed and cannot be installed here), and
  no digitised, machine-readable copy of the PNAS 2017 figures' underlying
  data points was available to fetch — only the paper's prose and abstract,
  which describe the phenomena qualitatively but do not hand over numbers a
  test could assert against without transcription risk.

Per the Wave S brief: *"if you cannot obtain genuine TouchSim output, do
not fabricate a fixture and call it a reference... implement the comparison
against whatever genuinely published numbers you can cite, and report what
you could not get."* This is that fallback, taken deliberately rather than
by omission.

## What is here instead

`sa_ra_qualitative_reference.json` encodes the two defining, uncontroversial
qualitative properties that give the SA ("slowly adapting") and RA
("rapidly adapting") afferent classes their names -- properties TouchSim's
own afferent models are built to reproduce, and which are standard textbook
material independent of any specific figure:

- **SA1 afferents** sustain a firing response roughly proportional to
  indentation depth for as long as the indentation is held (Johnson, K.O.,
  *The roles and functions of cutaneous mechanoreceptors*, Curr. Opin.
  Neurobiol. 11:455-461, 2001; Kandel et al., *Principles of Neural
  Science*, 5th ed., Ch. 21 -- already the project's cited source for SA/RA
  time constants, see `sensoryforge/filters/sa_ra.py`).
- **RA1 afferents** respond only to the onset and offset transients of an
  indentation and are silent (near-zero rate) during the static hold phase
  -- the defining property of "rapidly adapting," restated in Saal et al.
  (2017, PNAS, abstract and Fig. 1) for TouchSim's own RA model.

The JSON's `hold_to_peak_ratio` bounds are a deliberately loose,
hand-derived sanity envelope around these two qualitative facts (SA stays
above half its peak during hold; RA decays to well below its peak) -- not a
transcription of any specific number from a figure. `provenance` in the
JSON records this explicitly.

## What this validates, and what it does not

`tests/validation/test_touchsim_sanity.py` and
`docs/examples/validation_touchsim.ipynb` both read this file and run
SensoryForge's own `SAFilterTorch`/`RAFilterTorch` on a ramp-and-hold
stimulus, then check the filtered responses land inside these bounds.

**This is a sanity check that our SA/RA filters reproduce the correct
qualitative adaptation class, not a quantitative comparison against
TouchSim, and not an equivalence claim.** TouchSim simulates specific,
biophysically detailed afferent populations (SA1, RA1, PC) with their own
receptive-field geometry and dynamics; SensoryForge's SA/RA filters are a
lumped two-state (SA) / one-state (RA) phenomenological model calibrated
against Parvizi-Fard et al. (2021). The two are different models of
different (though related) afferent populations. Passing this check shows
SensoryForge is in the right qualitative regime; it does not show
SensoryForge and TouchSim produce the same numbers, and it would not catch
a quantitative miscalibration that preserved the SA/RA qualitative shape.

## What we could not get (reported per the Wave S brief)

- No genuine TouchSim spike trains or firing-rate traces for any stimulus.
- No digitised numeric points from Saal et al. (2017)'s figures.
- No quantitative agreement claim of any kind against TouchSim.

---

## `touchsim_ramp_hold.json` — genuine TouchSim output (2026-09-24, ledger D-4aafcdc)

Per ledger decision D-4aafcdc: *"the quantitative afferent comparison uses
touchsim output generated once in a throwaway environment and committed as
fixture data; touchsim never becomes a dependency."* This fixture is that
output. The blocker above (no installable `touchsim`) turned out to be
specific to the earlier attempt's environment, not to `touchsim` itself:
`hsaal/touchsim` installs and runs cleanly in a fresh, disposable
Python 3.10 conda environment.

### How it was generated

- **Source**: [`hsaal/touchsim`](https://github.com/hsaal/touchsim)
  (the Python implementation by Hannes Saal, the touchsim/PNAS-2017 paper's
  first author), commit `4ec9f5c382e7d48410566de743d7a4f05de75cec`
  (2025-04-05, "Remove holoviews from requirements.").
- **Licence**: the repository has no `LICENSE` file and GitHub's license
  detector reports none (checked 2026-09-24) -- treated as all rights
  reserved by the author absent an explicit open-source grant. Only this
  fixture's *numeric output* is committed to SensoryForge; no `touchsim`
  source is copied here, and `touchsim` is not, and never becomes, a
  SensoryForge dependency (per D-4aafcdc).
- **Environment**: a throwaway conda env (`python=3.10`, `numpy`, `scipy`,
  `numba`, `matplotlib`, `scikit-image` from `conda-forge`), built entirely
  outside this repository and outside the shared `sensoryforge` conda env;
  `touchsim` was installed into it with `pip install --no-deps .` (never
  `-e`). Versions recorded in the fixture's own `provenance`: Python
  3.10.21, NumPy 2.2.6, SciPy 1.15.2.
- **Generating script**: `scripts/validation/generate_touchsim_reference.py`
  (not imported by SensoryForge or its test suite; its header docstring has
  the exact environment-build and run commands).
- **Protocol**: a punctate probe (0.5 mm radius, touchsim's own default)
  applies a trapezoidal ramp-and-hold indentation -- 50 ms linear ramp up,
  450 ms hold, 50 ms linear ramp down (550 ms total) -- at 7 depths from
  near-threshold to touchsim's typical upper range: 0.025, 0.05, 0.1, 0.2,
  0.4, 0.7, 1.25 mm.
- **Afferents**: SA1, RA (= "RA1" in the literature) and PC (included since
  it was cheap; not required by the brief), each touchsim's `idx=0`
  single-neuron model for its class, placed along a single ray at 8
  distances from the probe centre -- 0, 1, 2, 3, 4, 5, 6, 8 mm (touchsim's
  receptive fields are radially symmetric around the probe, so a ray fully
  captures the distance dependence) -- 24 afferents total.
- **Seed**: `random.seed(42)` and `np.random.seed(42)` fix touchsim's
  membrane-noise RNG (used by every afferent's LIF model) and its
  afferent-model-selection RNG (unused here since `idx` is always given
  explicitly).
- Regenerate with the throwaway env's Python:
  `<throwaway-env>/bin/python scripts/validation/generate_touchsim_reference.py`.

### What's in the JSON

Per afferent: `affclass`, touchsim's `idx`, `(x_mm, y_mm)` position and
`distance_mm` from the probe. Per depth level: each afferent's spike times
(seconds) and summary rates (Hz) for three windows -- `onset` (the 50 ms
ramp-up), `sustained` (the hold, excluding its first 100 ms), and `offset`
(the 50 ms ramp-down). Full `provenance` (repo URL, commit SHA, licence
note, package versions, date, exact generating command, seed). ~23 KB.

### What this validates, and what it does not

`tests/validation/test_touchsim_reference.py` loads this fixture (with only
`json`/`numpy`, no `touchsim` import) and checks properties of
**TouchSim's own output**: SA1 sustains firing through the hold near the
probe; RA is silent during the hold at every depth and fires at both onset
and offset; firing rates increase with indentation depth for afferents near
the probe. The comparison against SensoryForge's own recipes is
`scripts/validation/compare_with_touchsim.py`, which fits one parameter (the
stimulus amplitude per mm of indentation) and compares every other feature at
matched depths; its report is `benchmarks/results/touchsim_comparison/`, and
`tests/validation/test_touchsim_comparison.py` fails when a feature's verdict
changes (F-070).

### Headline rates (Hz), afferents at the probe centre (distance 0 mm)

| depth (mm) | class | onset | sustained | offset |
|---:|:---|---:|---:|---:|
| 0.025 | SA1 | 0 | 0 | 0 |
| 0.025 | RA | 0 | 0 | 0 |
| 0.1 | SA1 | 20 | 0 | 0 |
| 0.1 | RA | 0 | 0 | 0 |
| 0.2 | SA1 | 40 | 8.57 | 0 |
| 0.2 | RA | 20 | 0 | 0 |
| 0.4 | SA1 | 60 | 11.43 | 0 |
| 0.4 | RA | 40 | 0 | 40 |
| 0.7 | SA1 | 100 | 25.71 | 0 |
| 0.7 | RA | 60 | 0 | 40 |
| 1.25 | SA1 | 180 | 42.86 | 0 |
| 1.25 | RA | 100 | 0 | 80 |

SA1's sustained rate is 0 Hz below ~0.2 mm (sub-/near-threshold at the
probe centre) and rises with depth once suprathreshold, staying nonzero
through the whole hold (individual spike times in the JSON show ISIs
sustained out to 496 ms of a 500 ms hold at 1.25 mm). RA's sustained rate is
exactly 0 Hz at every depth tested; its onset and offset rates both rise
with depth. Both are TouchSim's own output, not a SensoryForge claim.
