# Plugin Packages

SensoryForge components (neurons, filters, stimuli, solvers, grids) can be
distributed as standalone, installable Python packages — no edits to the
SensoryForge checkout required. This is the recommended route for anyone
outside the core project who wants to add a component: a lab sharing a
custom neuron model, a paper's reproducibility package, an internal tool.

If you are instead contributing a component *to* SensoryForge itself, use
the in-repo route described in `add_neuron.md`, `add_filter.md`, and
`add_stimulus.md` (pass `--in-repo` to the same scaffold command, see
below) — this page is about the plugin-package route specifically.

---

## 1. Generate the package

```bash
sensoryforge new-component filter MyBandpass --dest ~/code
```

- `filter` is the component **kind**: one of `neuron`, `filter`, `stimulus`,
  `solver`, `grid`.
- `MyBandpass` is the component **name** (accepts `CamelCase` or
  `snake_case`; the scaffold derives both a class name and a registry key
  from it).
- `--dest` is the directory the new package is written under (defaults to
  the current directory). `--dest` is ignored if you pass `--in-repo`
  instead (see above) — the two modes are mutually exclusive.

This writes an installable `sensoryforge-my-bandpass/` package containing:

- `pyproject.toml` — declares the package and a
  `[project.entry-points."sensoryforge.components"]` section pointing at
  the package's `register()` function.
- A small importable module with the starter component class (inheriting
  the right base class for `filter`) and a `register()` function that
  registers it with the matching `FILTER_REGISTRY`/`NEURON_REGISTRY`/etc.
- `tests/test_contract.py` — calls
  `sensoryforge.testing.contracts.check_component()` (see below) so the
  generated package has a passing correctness check from the start.
- `README.md` with install/usage instructions.

The scaffold refuses to write anywhere under a `site-packages` or
`dist-packages` directory, so running it against an installed (wheel)
SensoryForge is always safe — it never touches the installed package
location, only `--dest`.

## 2. Fill in the component

Edit the starter class the scaffold generated: implement `forward()` (and
the kind-specific contract method — `compute_weights()` for innervation,
`get_all_coordinates()` for grids, `step()` for solvers), fill in
`get_param_spec()` with your parameters, and make sure `to_dict()` covers
every `__init__` argument so `from_config()` round-trips (see H3 in
`add_neuron.md`/`add_filter.md`/`add_stimulus.md` for the exact contract).

For the `neuron` kind specifically: `SimulationEngine` requires every
neuron model to accept a `noise_std: float` constructor parameter (it is
passed unconditionally by `_build_populations`) — the generated neuron
template already declares it, keep it if you edit the constructor
signature.

## 3. Install and test

```bash
cd sensoryforge-my-bandpass
pip install -e .
pytest
```

`pip install -e .` registers the `sensoryforge.components` entry point in
your environment's package metadata. From that point on, **any**
SensoryForge process that imports `sensoryforge.register_components`
(the CLI, the GUI, `BatchExecutor`, a plain `import sensoryforge`) will
discover and register your component automatically — no config changes
needed to make the name available, only to use it (e.g.
`filter_method: my_bandpass` in a population's config).

A broken or missing plugin never crashes the host process: a failed entry
point produces a `UserWarning` and is skipped
(`sensoryforge/plugins.py:discover_entry_point_plugins`).

## 4. Verify the contract before publishing

`sensoryforge.testing.contracts.check_component(kind, cls)` runs the same
checks used across the project's own contract-test sweep (G4): it asserts
`get_param_spec()` returns a list of `ParamSpec`, runs one `forward()` pass
with the kind's canonical tensor shape, and checks
`from_config(instance.to_dict())` reconstructs correctly. It has no Qt
dependency and needs nothing but `torch` and `sensoryforge` itself, so it
runs anywhere the plugin package runs.

```python
from sensoryforge.testing.contracts import check_component
from my_package.filter import MyBandpassFilter

check_component("filter", MyBandpassFilter)
```

This is exactly what the generated `tests/test_contract.py` calls — running
`pytest` in the scaffolded package is enough to get this check for free.

## 5. The `plugins:` YAML key (an alternative to entry points)

Besides the installed entry point, any SensoryForge config file can name
plugin modules to import directly, without installing anything as a
distribution:

```yaml
plugins:
  - my_package.filter                 # import for side effects (module-level registration)
  - my_package.filter:register        # import, then call register()

grids: [...]
populations: [...]
```

Each entry is a dotted import path, optionally suffixed with `:attr` to
call a specific callable after import (matching the entry-point
convention: the callable is expected to register the component itself). A
plugin that fails to import produces a `UserWarning` and is skipped, same
as a broken entry point.

As of H4, every config loader in the project honours `plugins:` identically
— the CLI (`sensoryforge run`/`validate`/`batch`), `BatchExecutor.from_yaml`,
`SensoryForgeConfig.from_yaml_file`/`from_yaml`, and the GUI's "Load YAML
Configuration" action all route through the single shared loader
(`sensoryforge.config.yaml_utils.load_config_file`), so a config that
depends on a plugin behaves the same way regardless of how it's loaded.
See `docs/user_guide/yaml_configuration.md` for the full `plugins:`
reference.

## Choosing between the entry point and `plugins:`

- Ship a **pip-installable package** with an entry point when the
  component is meant to be reused across configs/projects and installed
  once into an environment.
- Use a config's **`plugins:` list** for a one-off component that lives
  next to a specific experiment's config and doesn't need its own
  package — a plain importable `.py` module is enough.

Both paths end up calling the same registries
(`sensoryforge.registry.NEURON_REGISTRY`, `FILTER_REGISTRY`, etc.), so a
component works identically either way once registered.
