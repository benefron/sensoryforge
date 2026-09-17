# Presets

A **preset** is a complete canonical config, shipped as a YAML file inside
`sensoryforge/presets/`, that you can run directly with no config file of your own.
A preset is data, not code — it is the cheapest way to extend SensoryForge with a
new named, reproducible experiment: add a `.yml` file, no Python required.

## Listing and running presets

```bash
sensoryforge list-presets
```

```
Available SensoryForge Presets:
==================================================
  - tactile_sa1_ra1: The pressure-simulation recipe: 80x80 grid at 0.15 mm, ...
  - tactile_stochastic_control: Named control arm (D-019): identical to ...

💡 Use 'sensoryforge run --preset <name>' to run one, or 'sensoryforge run --preset <name> config.yml' to override it
```

Run a preset with no config file at all:

```bash
sensoryforge run --preset tactile_sa1_ra1 --duration 1000
```

## The two shipped presets

- **`tactile_sa1_ra1`** — the [pressure-simulation recipe](../concepts/pressure_simulation_use_case.md):
  an 80×80 grid at 0.15 mm, one SA and one RA population, both with `template`
  (designed) receptive fields at `resolvable_distance_mm: 0.40`.
- **`tactile_stochastic_control`** — the named control arm (decision D-019 in
  `docs_root/LEDGER.md`): identical grid and populations, but with
  `innervation_method: gaussian` and `use_distance_weights: false` instead of
  `template` — pressure-simulation's uniform-random-weight ("stochastic")
  innervation, for comparing designed against undesigned receptive fields on
  otherwise identical populations. There is no separate `gaussian_stochastic`
  builder name registered anywhere: `GaussianInnervation` already implements
  both the analytic (`use_distance_weights: true`, the default) and the
  stochastic (`false`) weighting, and D-019 settles which one is "the control
  arm."

## Overriding a preset

`sensoryforge run --preset NAME config.yml` uses the preset as the base config and
applies `config.yml`'s values on top: nested dictionaries (`simulation:`, a
population's own fields, ...) are merged key by key; a `grids:` or `populations:`
list in `config.yml` replaces the preset's list outright rather than merging
element by element. For example, to run the pressure-simulation recipe with
Gaussian noise turned off:

```yaml
# quiet.yml
populations:
  - name: SA Population
    noise_std: 0.0
  - name: RA Population
    noise_std: 0.0
```

```bash
sensoryforge run --preset tactile_sa1_ra1 quiet.yml
```

Because a `populations:` list in the override file replaces the preset's list
whole, an override file that touches populations must list **every** population
you want in the run, with every field it needs (`target_grid`, `neuron_model`,
`innervation_method`, ...) — not just the fields you're changing. The example
above works because it restates every field `PopulationConfig` needs by name and
relies on the schema's own defaults for the rest; if your override needs a field
that differs from the default (a different filter, a different innervation
method), include it explicitly.

## Adding a preset

A preset is a normal canonical config — the same format `SensoryForgeConfig.to_yaml()`
produces, and the same format `examples/canonical_config.yml` uses — saved into
`sensoryforge/presets/<name>.yml`. To add one:

1. Build and validate the config however you like (hand-write the YAML, or export
   one from `SensoryForgeConfig(...).to_yaml()` or the GUI).
2. Save it as `sensoryforge/presets/<name>.yml`.
3. Add a one-line entry to `_DESCRIPTIONS` in `sensoryforge/presets/__init__.py` (used
   by `sensoryforge list-presets`; optional, but `sensoryforge list-presets` and
   `preset_description()` fall back to an empty string without one).
4. Add `<name>` to `tests/unit/test_presets.py`'s parametrised
   `test_preset_loads_and_builds_valid_config` so a broken preset fails CI.

No changes to `register_components.py`, the CLI, or `pyproject.toml` are needed —
`sensoryforge/presets/*.yml` is already declared in `[tool.setuptools.package-data]`,
so a new preset ships in the next wheel build automatically as long as it lives in
that directory.

## Loading a preset from Python

```python
from sensoryforge.presets import list_presets, load_preset
from sensoryforge.config.schema import SensoryForgeConfig

print(list_presets())
config_dict = load_preset("tactile_sa1_ra1")
config = SensoryForgeConfig.from_dict(config_dict)
```

`load_preset()` reads through `importlib.resources`, so it works the same way from
a source checkout and from an installed wheel run outside the repository.
