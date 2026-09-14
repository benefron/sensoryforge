# Changelog

All notable user-facing changes to SensoryForge are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates are commit dates from
`docs_root/LEDGER.md`, the project's decision/finding record.

## [Unreleased]

Not yet published to PyPI; install from source (see `CONTRIBUTING.md`).

### Changed (behaviour — re-run any saved results after upgrading)

- **SA filter no longer rectifies by default.** `SAFilterTorch.clip_to_positive` now defaults to
  `False` (sign-preserving SA, matching the sign convention used to recover velocity direction from
  SA activity). Pass `clip_to_positive=True` to restore the old rectified behaviour.
- **RA filter time constant (τ_RA) is 8 ms everywhere**, not 15 or 30 ms. This affects
  `RAFilterTorch`, `CombinedSARAFilter`, the GUI, the CLI/legacy pipeline, and every example config.
- **RA populations default to the fast-spiking Izhikevich preset** (`a=0.1, b=0.2, c=-65, d=2`)
  instead of regular-spiking (`a=0.02, d=8`), in the GUI, `SimulationEngine`, and the legacy
  pipeline alike. Overriding any single one of `a`/`b`/`c`/`d` on a population now keeps the rest of
  the neuron-type preset as the base instead of silently reverting them to regular-spiking.
- **RA filter gain k3 is 2.0**, not 100. On the default ramp/trapezoidal stimulus, fast-spiking RA
  firing drops from roughly 314 Hz (the old GUI default) to roughly 69 Hz. If you have tuned
  `input_gain` presets against the old k3=100 behaviour, they will need retuning.
- **Filter and neuron defaults are now resolved by a single function**
  (`sensoryforge.config.defaults.resolve_filter_params` / `resolve_neuron_params`), used by the GUI,
  `SimulationEngine`, and the legacy pipeline. Previously the GUI and the CLI could silently build
  different models from the same config.
- **Default innervation weights are analytic Gaussian; the stochastic uniform-random builder
  remains available as the control arm.** Pass `use_distance_weights=False` to restore it.
- **`requires-python` is now `>=3.10`** (a neuron module already used 3.10-only syntax). The PyQt5
  GUI is now an optional extra: `pip install -e ".[gui]"`.

### Fixed

- The canonical config adapter no longer squares neuron counts or receptor grid size. A canonical
  population of N neurons per row now builds N² neurons consistently across the legacy pipeline and
  `SimulationEngine` (previously the legacy pipeline could build 16x the intended count); the
  README's 80x80 quick-start no longer risks exhausting memory.
  A legacy hand-written config whose neuron count reads like a total rather than a per-row value
  now raises a clear error instead of silently allocating a huge weight tensor.
- `SensoryForgeConfig.from_yaml` no longer raises `OSError` on a one-line YAML/JSON string longer
  than the filesystem's path-length limit.
- `IzhikevichNeuronTorch`'s `preset` parameter is now keyword-only, fixing a positional-argument
  footgun.
- Package config/GUI parameter files are located via `importlib.resources`, so the package now
  works correctly when installed as a wheel and run outside the source checkout (previously several
  code paths assumed the current working directory was the repository root).

### Added

- `pyproject.toml` (PEP 621) replaces `setup.py`; optional extras `gui`, `hdf5`, `solvers`, `dsl`,
  `dev`, `docs`.
- `pytest -m gui` / `pytest -m "not gui"` markers separate the Qt-backed GUI test suite from the
  rest; both now run reliably in a single process each (previously the Qt suite could crash mid-run
  or at interpreter exit).
- `.github/workflows/tests.yml`: CI across Python 3.10/3.11 on Linux and macOS, plus lint/format/docs
  checks.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.

[Unreleased]: https://github.com/benefron/sensoryforge/commits/main
