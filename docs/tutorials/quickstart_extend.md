# Quick Start: Extend SensoryForge

In five minutes you add a temporal filter of your own, check it against SensoryForge's
contract, see it listed, and run a config that uses it. Nothing is forked or cloned: your
filter is a small package of its own that SensoryForge finds when it is installed. The same
steps work for a neuron model, stimulus or receptive-field builder. A solver or a grid
arrangement goes through steps 1 to 3 the same way, but a config cannot name it yet: see
[Add a Solver](../how-to/extend/add_solver.md) and
[Add a Grid Arrangement](../how-to/extend/add_grid_arrangement.md).

You need SensoryForge installed ([Installation](installation.md); it is not yet on PyPI, so
install from source) and a folder to work in.

## 1. Scaffold the plugin

```bash
sensoryforge new-component filter Smoother
cd sensoryforge-smoother
```

This writes a standalone package:

```text
sensoryforge-smoother/
├── pyproject.toml                 # names the entry point SensoryForge discovers
├── README.md
├── sensoryforge_smoother/
│   ├── __init__.py
│   └── component.py               # SmootherFilterTorch and register()
└── tests/
    └── test_contract.py           # the shared contract check
```

`component.py` holds a working placeholder, a single-pole low-pass with one parameter,
`tau` in ms. `register()` adds it to the filter registry under the name `smoother`. Other
kinds: `neuron`, `stimulus`, `solver`, `grid`, `innervation` (a receptive-field builder).

## 2. Install it and run the contract test

```bash
pip install -e .        # installs your plugin, not SensoryForge
pip install pytest
pytest
```

```text
1 passed
```

The test calls `sensoryforge.testing.contracts.check_component`, the same check every
built-in filter passes: the parameter spec, the output shape of a forward pass, and the
`to_dict` and `from_config` round-trip. Re-run it after every edit to `component.py`. It
runs one forward pass, so it does not catch a filter that keeps state from one call to the
next; start from rest on every call, as the placeholder does.

## 3. See it listed

```bash
sensoryforge list-components
```

```text
📊 Filters:
  ...
  - smoother
```

## 4. Use it in a config

Save this as `smoother.yml`. One population reads a 20 × 20 sensor array through your
filter:

```yaml
grids:
  - name: sensors
    arrangement: grid
    rows: 20
    cols: 20
    spacing: 0.15               # mm
populations:
  - name: smoothed
    target_grid: sensors
    neuron_type: SA
    innervation_method: gaussian
    neurons_per_row: 6          # a 6 x 6 layout, 36 neurons
    connections_per_neuron: 20
    sigma_d_mm: 0.3
    filter_method: smoother     # the name your plugin registered
    filter_params: {tau: 5.0}   # ms
stimulus:
  type: gaussian
  amplitude: 1.0
  sigma: 0.5                    # mm
```

```bash
sensoryforge validate smoother.yml
sensoryforge run smoother.yml --duration 100 --bundle smoothed_run
```

```text
Simulation completed successfully!
smoothed spikes: ...
```

A misspelt `filter_method` stops `validate` with `Unknown filter method: ...`, so a
config never runs silently without your component. The run writes a
[data bundle](../reference/bundles.md) to `smoothed_run/`, and its `config.json` records
`filter_method: smoother`.

## Next

- Replace the placeholder in `component.py` with your filter, then `pytest` again. The
  population's gain still multiplies your filter's output (see
  [Units and Gains](../how-to/units_and_gains.md)).
- [Plugin Packages](../how-to/extend/plugins.md): the entry point, checking the contract
  before you publish, and a config's `plugins:` list.
- One guide per kind: [neuron](../how-to/extend/add_neuron.md),
  [filter](../how-to/extend/add_filter.md), [stimulus](../how-to/extend/add_stimulus.md),
  and the [Extensibility Overview](../how-to/extend/extensibility.md) for the rest.
- A coding agent can do these steps with you: the `sensoryforge-extend-pipeline` and
  `sensoryforge-plugin-lifecycle` skills ([Working with Agents](../agents/index.md)).
