# `tests/fixtures/reference/` — provenance

## What is (and is not) here

`sa_ra_qualitative_reference.json` records **qualitative published bounds**
on SA1/RA1 mechanoreceptor afferent responses to a ramp-and-hold skin
indentation. It does **not** contain TouchSim output.

## Why not genuine TouchSim output

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
