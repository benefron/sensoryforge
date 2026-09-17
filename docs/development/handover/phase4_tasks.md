# Phase 4 handover — validation, benchmarks and the paper

Prepared 2026-09-16. The approved plan is `docs/developer_guide/roadmap_v1.md` ("Phase 4 —
validation, docs, paper artefacts"). Phase 4 may overlap Phase 3: waves S and T depend only on
Phase 2, and can run in parallel with the GUI work. Waves U and V need Phase 3 finished.

---

## How Phase 4 is run

Same orchestration as Phases 2 and 3. Waves S and T are independent of each other and of Phase 3, so
they run in parallel in their own worktrees. U and V are sequential and last.

| Wave | Depends on | Delivers |
|---|---|---|
| **S** | Phase 2 | Validation against reference data: analytic checks, a published-model comparison, the reproducibility proof (F-022) |
| **T** | Phase 2 | Benchmarks: wall-clock and memory across grid size, neuron count and device |
| **U** | Phase 3 | The complete documentation tree, executed notebooks, and GitHub Pages |
| **V** | U | The JOSS paper draft, the release, and the cross-repository close-out |

---

## 1. Facts the agent must know

**Fact P4-a (verified 2026-09-16).** `touchsim` is **not** installed in the project environment and
is not a dependency. Wave S must not add it as a hard dependency. The comparison in S2 either vendors
a small reference dataset generated once and committed, or is written as an optional test that skips
cleanly when the package is absent. Prefer the committed dataset: a comparison that never runs in CI
is not a validation.

**Fact P4-b (verified 2026-09-16).** `mkdocstrings` 1.0.6 and `mkdocstrings-python` 2.0.8 are
installed and the strict docs build already passes. `nbmake` and `pytest-benchmark` are **not**
installed; Wave T and Wave U add whichever they need to the `docs` and `dev` extras, and to CI.

**Fact P4-c (settled, 2026-09-14).** The open question in pressure-simulation's
`docs_root/architecture/PAPER_A_SIMULATOR.md` — whether the released tool includes the decoder stack
— is decided: **decoding stays out of SensoryForge**, which ships a stable export contract instead
(the Wave J bundle). Wave V updates that note in pressure-simulation so the two repositories agree in
writing.

**Fact P4-d (F-022, open).** There is today exactly one analytic validation test
(`tests/unit/test_filters_vs_theory.py`), no comparison against any published model, an unexecuted
notebook and no benchmark suite. F-022 is Wave S's and Wave T's shared target and closes only when
both land.

---

## 2. Phase 4 guardrails

1. **A validation that cannot fail is not a validation.** Every check in Wave S states the quantity,
   the reference, the tolerance and why that tolerance. A test with a tolerance wide enough to pass a
   broken implementation must be reported, not committed.
2. **Benchmarks are data, not decoration.** Wave T records the machine, the versions and the seed
   with every number, and the table is regenerable by one command.
3. **Every documentation example runs in CI.** A guide whose code is not executed is a guide that
   will be wrong within two releases.
4. **The paper claims only what the repository proves.** Every claim in Wave V's draft points at a
   test, a benchmark or an executed example.
5. **One task, one commit, single-line trailers.**

---

## 3. Wave S — validation (F-022, first half)

### S1. Analytic validation

Extend `tests/unit/test_filters_vs_theory.py` into `tests/validation/` covering the whole spine, each
against a closed-form answer rather than a recorded output:

- **Filters.** SA step response against the analytic two-exponential solution; RA response to a ramp
  against `k3 · dI/dt`; both across three time steps to show the discretisation converges at the
  expected order.
- **Receptive fields.** The `template` builder's weights against the Gaussian evaluated at the same
  distances, and its row norms against unity; the resolvable-distance chain `d → σ = d/π, Δ = d,
  N = A/Δ²` against the arithmetic, at three values of `d`.
- **Sampling.** The Wave L `grid_sample` path recovering an analytic Gaussian at hex receptor
  positions, with the error stated as RMS relative to peak and bounded.
- **Neurons.** Izhikevich regular-spiking and fast-spiking rate-versus-current curves against the
  published figures of Izhikevich (2003), at a tolerance that a wrong preset would violate.

**Done when:** each test names its reference in the docstring, and a deliberate perturbation of the
implementation makes it fail. Record the perturbations.

### S2. Comparison against a published model

A notebook, `docs/examples/validation_touchsim.ipynb`, comparing SensoryForge's SA and RA responses
to a probe indentation against TouchSim's published response for the same stimulus. Given Fact P4-a,
commit the reference traces as a small dataset under `tests/fixtures/reference/` with a README
recording exactly how they were produced, and have the notebook and a test both read that file. State
what agreement is expected and what disagreement would mean — the models differ in their afferent
populations, so this is a sanity comparison, not an equivalence claim, and the text must say so.

**Done when:** the notebook executes in CI and the test asserts the stated agreement.

### S3. The reproducibility proof

One script that, from a clean checkout and a fresh environment, installs the package, runs the
pressure-simulation recipe from Wave K5, and reproduces a committed figure and a committed set of
summary statistics bit-for-bit on the same platform, and within a stated tolerance across platforms.
This is the claim a tool paper lives or dies on.

**Done when:** the script runs green in CI on both operating systems in the matrix, and the commit
carries `Closes: F-022` jointly with Wave T's commit, whichever lands second.

---

## 4. Wave T — benchmarks (F-022, second half)

### T1. The benchmark harness

`benchmarks/run_benchmarks.py` sweeping grid size, neuron count and device, recording for each cell:
wall-clock for the build phase and the run phase separately, peak resident memory, and the machine,
Python, torch and SensoryForge versions plus the seed. Output is a committed JSON file plus a
generated Markdown table.

Use the memory watchdog from the Phase 1 appendix for the memory number rather than inventing a
second mechanism.

### T2. The published table

`docs/reference/benchmarks.md` generated from the JSON by a script, never hand-edited, with a header
stating the machine it was measured on and the command to regenerate it. Include at least one cell
large enough to be interesting (an 80×80 grid with both pressure-simulation populations over one
second) and one small enough to be a smoke test.

### T3. A performance guard in CI

One fast benchmark runs in CI and fails if it regresses by more than a stated factor against the
committed baseline, so a future change that makes the engine ten times slower is caught. Choose the
factor loosely enough that CI noise does not cause false failures, and say what factor was chosen and
why.

**Done when:** the table regenerates from one command, the CI guard passes, and a deliberately
slowed engine makes it fail.

---

## 5. Wave U — the documentation tree

### U1. Complete the structure

Finish every section of the documentation requirement in the roadmap: concepts, getting started, user
guide, extending, reference, developer guide. No stub pages marked "coming in Phase N" may remain.
The API reference is generated by `mkdocstrings` from the docstrings, so the work here is the
narrative pages and the navigation, not restating the API.

### U2. Executed notebooks

The quick-start notebook and the validation notebook execute in CI with `nbmake`, added to the `docs`
extra. A notebook that needs more than two minutes gets a reduced-size parameter set for CI and
states that its published outputs came from the full run.

### U3. Every extending guide executed

`pytest docs/examples` runs the code from every guide in `docs/extending/`. There is one guide per
plugin kind: grid arrangement, stimulus, receptive-field builder, combine, processing layer, filter,
neuron, analog readout, solver, exporter and GUI node. Any guide without executable code is either
given some or merged into a concepts page.

### U4. GitHub Pages

A deploy workflow publishing the strict build on pushes to the default branch. The README's
documentation link points at the published site rather than at the repository.

**Done when:** the strict build is clean, `pytest docs/examples` is green, the notebooks execute, and
the site deploys from a workflow run.

---

## 6. Wave V — the paper and the close-out

### V1. The plugin template repository

`examples/plugin_template/` completed as a standalone installable package adding one receptive-field
builder and one processing layer, with its own tests and README, and a documented path to publishing
it as a GitHub template. A test installs it into a temporary environment and asserts its components
appear in `sensoryforge list-components`.

### V2. The JOSS draft

`paper/paper.md` and `paper/paper.bib`, scoped per pressure-simulation's
`docs_root/architecture/PAPER_A_SIMULATOR.md`: a forward model, not a method paper. JOSS reviewers
check a specific list, so the draft must be able to point at each one: a clean install, documented
API, a reproducible demo that runs headless, tests, community guidelines, a license and a citation
file. Every claim cites a test, a benchmark or an executed example by name.

### V3. Release

`CHANGELOG.md` completed for 1.0.0, `CITATION.cff` checked, version numbers consistent, and a tagged
release with the wheel built and verified from outside the repository the way Phase 1 did.

### V4. Cross-repository close-out

- Update pressure-simulation's `PAPER_A_SIMULATOR.md` with the settled scope decision from Fact P4-c,
  with its own ledger trailer in that repository.
- Bring pressure-simulation's ledger onto the current hook set so `/ledger-status` reports both
  repositories.
- Record in both ledgers that the engines are in golden parity and which test proves it.

**Done when:** both repositories' ledgers agree, the release artefacts exist, and the paper draft is
ready for the user to read.

---

## 7. Phase 4 exit criteria

- Waves S to V complete with their exit checks.
- F-022 closed; no open findings remain except ones the user has explicitly deferred.
- A reader who has never seen the repository can install it, run the demo, reproduce the figure, and
  write a plugin, using only the published documentation.
- All suites, `black`, the CI flake8 subset, `mkdocs build --strict`, `pytest docs/examples` and the
  notebooks pass in CI on both operating systems.
