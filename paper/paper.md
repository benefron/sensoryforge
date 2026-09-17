---
title: 'SensoryForge: A Modular, Extensible Forward-Model Simulator for Sensory Encoding'
tags:
  - Python
  - PyTorch
  - computational neuroscience
  - tactile encoding
  - spiking neural networks
  - sensory encoding
authors:
  - name: Ben Efron
    affiliation: 1
affiliations:
  - name: "AFFILIATION NEEDED"
    index: 1
date: 17 September 2026
bibliography: paper.bib
---

<!--
PLACEHOLDERS LEFT FOR THE USER (also listed in the V2 report):
- `affiliations[0].name` above is a literal placeholder, "AFFILIATION NEEDED"
  -- CITATION.cff carries no affiliation, institution or ORCID for the sole
  author, and none is invented here.
- No ORCID field is set on the author for the same reason.
-->

# Summary

SensoryForge is a PyTorch-based, GPU-capable simulator that turns a spatial
stimulus (a pressure map, an image, or another sampled sensory signal) into
simulated afferent population activity through an explicit, shape-annotated
pipeline: a stimulus is sampled at a population of receptor coordinates,
passed through a receptive-field bank that maps receptors onto neurons,
filtered by a temporal adaptation model (slowly- or rapidly-adapting SA/RA
dynamics, or a user-defined filter), and driven through a spiking or analog
neuron model. Every stage -- grid arrangement, receptive-field builder,
filter, neuron, solver, processing layer, and analog readout -- is a
registered, contract-checked component
(`sensoryforge/registry.py`, `sensoryforge/testing/contracts.py`), so a
third party can add a new component as a standalone, installable Python
package with no changes to the SensoryForge checkout
(`docs/developer_guide/plugins.md`; `examples/plugin_template/` is a
complete, tested, installable worked example, verified in this release
through the real Python entry-point discovery mechanism rather than an
in-process registry call --
`examples/plugin_template/tests/test_entry_point_discovery.py`). A
canonical YAML configuration format (`sensoryforge/config/schema.py`)
drives both a command-line interface and an interactive PyQt5 GUI from the
same execution engine (`sensoryforge/core/simulation_engine.py`), so an
experiment designed interactively reproduces exactly when scaled to a batch
run from the command line.

SensoryForge is a **forward-model tool**, not a decoding or inference
package: it simulates afferent responses to a stimulus, and stops there. A
companion project, pressure-simulation, consumes SensoryForge's stable
export bundle contract for downstream decoding work; that decision -- that
decoding stays out of SensoryForge -- is settled between the two
repositories (see `docs/development/handover/phase4_tasks.md`, Fact P4-c).

# Statement of need

Simulating how a population of sensory afferents encodes a time-varying
stimulus is a common need across computational neuroscience, neuromorphic
engineering, and machine-learning research that uses biologically inspired
encoders. Two purpose-built pieces of software already address parts of
this space for the tactile domain specifically: TouchSim
[@saal2017] simulates biophysically detailed SA1/RA1/PC afferent
populations from a hand model. SensoryForge is not a replacement for such
tools; it targets a different, complementary need: a *modality-agnostic*,
*extensible* simulation core -- the same receptive-field/filter/neuron
pipeline structure is used for a tactile SA/RA pathway
(`sensoryforge/filters/sa_ra.py`, calibrated against
[@parvizifard2021] and [@kandel_principles]) and, with different
components, for a vision on/off-centre pathway
(`sensoryforge/core/processing.py`'s `OnOffLayer`,
`examples/vision_rgb_onoff.py`) -- built so a lab can add its own filter,
neuron model, receptive-field builder, or entire sensor arrangement as an
installable plugin (`docs/developer_guide/plugins.md`) rather than fork the
project. It is designed to be driven identically from an interactive GUI
(for experiment design) and from YAML configuration files (for
reproducible batch runs and parameter sweeps,
`sensoryforge/core/batch_executor.py`), so an experiment built by hand
scales to a cluster without being re-implemented.

# What is validated, and how

Every claim below points at the test, benchmark, or executed example that
proves it. This is deliberate: a forward-model tool paper is only useful if
a reader can independently check each claim against a committed artifact
rather than take it on faith.

- **Analytic validation of the core pipeline.** `tests/validation/`
  (Wave S, S1) checks the SA filter's step response against the closed-form
  two-exponential solution, the RA filter's ramp response against
  `k3 * dI/dt`, the `template` receptive-field builder's weights against
  the Gaussian evaluated at the same distances and its resolvable-distance
  chain (`d -> sigma = d/pi, pitch = d, N = A/pitch^2`) against the
  arithmetic, the `grid_sample`-based stimulus sampling path against an
  analytic Gaussian recovered at hex receptor positions, and the
  Izhikevich regular-spiking and fast-spiking presets' rate-vs-current
  behaviour against the qualitative claims of [@izhikevich2003]
  (`tests/validation/test_filters_analytic.py`,
  `test_receptive_fields_analytic.py`, `test_sampling_analytic.py`,
  `test_neurons_analytic.py`).
- **A published-model sanity comparison, explicitly not an equivalence
  claim.** `tests/validation/test_touchsim_sanity.py` and
  `docs/examples/validation_touchsim.ipynb` check that SensoryForge's SA
  and RA filters reproduce the correct *qualitative* adaptation class
  (sustained SA response during a hold; RA response confined to onset/offset
  transients) against a committed reference derived from published,
  qualitative claims in [@johnson2001], [@kandel_principles], and
  [@saal2017] -- **not** against genuine TouchSim output or digitised
  figure data, because `touchsim` is not installed in this project's
  environment and could not be added as a dependency (see
  `tests/fixtures/reference/README.md` for the full provenance and what
  this comparison does and does not show; this is recorded as an open,
  known limitation in the project's own living ledger,
  `docs_root/LEDGER.md`, finding F-070).
- **Bit-for-bit reproducibility.** `scripts/reproduce_figure.py` and
  `scripts/reproduce_env.sh` install the package into a fresh environment,
  run the pressure-simulation demo recipe
  (`examples/pressure_simulation_recipe.py`), and reproduce a committed set
  of summary statistics exactly (spike counts) and within a stated relative
  tolerance (mean rates) against
  `tests/fixtures/reference/reproducibility/summary_stats.json`;
  `tests/validation/test_reproducibility.py` runs this check in CI and its
  docstring records a deliberate perturbation of the reference (a single
  spike count changed) that makes the check fail, confirming it actually
  discriminates a wrong result from a right one.
- **A reproducible, headless demo.** The same recipe
  (`examples/pressure_simulation_recipe.py`) builds an 80x80-receptor grid,
  designs SA and RA receptive fields from a single resolvable distance, runs
  four stimuli (a ramp-and-hold Gaussian indentation, a moving edge, a
  simulated Braille pattern, and a drifting grating), and writes one export
  bundle per stimulus -- runnable with no display (`--quick` for a fast
  smoke pass) and requiring no GUI dependency.
- **Cross-platform floating-point caveat.** Three zero-tolerance golden
  tests (`tests/integration/test_pressure_sim_parity.py`,
  `tests/integration/test_stimulus_parity.py`, and the receptive-field
  golden weights in `tests/fixtures/rf_engine_golden_weights.pt`) compare
  against fixtures generated on macOS arm64 and have not, as of this
  release, been exercised on another platform; a failure on a different
  architecture may be floating-point rounding rather than a regression, and
  should be diagnosed rather than assumed (ledger finding F-071). No
  quantitative cross-platform equivalence claim is made here beyond that
  caveat.
- **Performance.** `docs/reference/benchmarks.md` (generated from
  `benchmarks/results/latest.json` by `benchmarks/generate_table.py` --
  regenerate with `python -m benchmarks.run_benchmarks`) reports wall-clock
  build and run time and peak resident memory across three grid/population
  sizes on the machine that produced the committed numbers (an Apple M3
  Pro, macOS-15.7.1-arm64), including an 80x80-receptor grid with both SA
  and RA populations run for one simulated second (1945.99 ms median run
  time) and a 10x10 smoke-test cell (37.70 ms median run time). No
  performance number appears in this paper that is not drawn from that
  file. A continuous-integration performance guard
  (`.github/workflows/benchmarks.yml`, `benchmarks/check_regression.py`)
  fails a run more than 3.0x slower than a committed baseline after
  normalising against a fixed reference kernel timed in the same process,
  a factor chosen to tolerate ordinary hardware noise (evidence for the
  choice, including a deliberately slowed engine that does trip the guard,
  is recorded in `benchmarks/check_regression.py`'s docstring and
  `docs/reference/benchmarks.md`); as of this release this guard, like the
  rest of this project's continuous integration, has never executed on
  GitHub's own runners, so its cross-architecture calibration is estimated
  from local experiments, not measured on the runner it will actually run
  on.

# Installation, documentation, and community guidelines

SensoryForge installs from source with `pip install -e .`
(`README.md`; not yet published to PyPI) and declares Python 3.10+,
PyTorch, NumPy, SciPy, scikit-learn, PyYAML, and plotting dependencies in
`pyproject.toml`, with optional extras for the GUI (`PyQt5`, `pyqtgraph`),
HDF5 export, adaptive ODE solvers (`torchdiffeq`), and an equation-based
neuron-model DSL (`sympy`). The documentation tree
(`docs/`, built with `mkdocs` and `mkdocstrings`; `mkdocs build --strict`)
covers concepts, getting-started, user-guide, developer-guide and
API-reference sections, plus executable guides for adding most plugin kinds
(`docs/developer_guide/add_filter.md`, `add_neuron.md`, `add_rf_builder.md`,
`add_stimulus.md`; `docs/extending/add_grid_arrangement.md`,
`add_gui_node.md`, `add_processing_layer.md`, `add_solver.md`), every
`docs/examples/*.py` script executed by `pytest docs/examples`
(`tests/docs/test_docs_examples.py`), and the quick-start and validation
notebooks executed with `nbmake`
(`.github/workflows/tests.yml`); a deploy workflow
(`.github/workflows/deploy-docs.yml`) publishes this build to GitHub Pages
on pushes to the default branch, though as of this release continuous
integration has never executed on GitHub for this repository, so that
workflow has not yet been exercised for real. The project carries an MIT
license (`LICENSE`), a citation file (`CITATION.cff`), and community
guidelines (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`).

# Acknowledgements

None recorded in this repository; none invented for this draft.

# References
