# Config Directory

> **Status:** CORE — canonical config schema and shared defaults consumed by the GUI, CLI, and
> `SimulationEngine`.

## Files

- `schema.py` — `SensoryForgeConfig`/`GridConfig`/`PopulationConfig`/`StimulusConfig`/
  `SimulationConfig` dataclasses (the canonical config format), plus `validate_dt_ms`.
  `from_dict()`/`to_dict()`/`from_yaml()`/`to_yaml()` handle round-trip serialization.
- `defaults.py` — single source of truth for filter/neuron defaults (`resolve_filter_params`,
  `resolve_neuron_params`, `FILTER_DEFAULTS`, `NEURON_PRESET_BY_TYPE`) and `DEFAULT_INTEGRATE_DT_MS`.
  Every caller that needs a filter or neuron default (GUI, `SimulationEngine`, the legacy
  pipeline) goes through this module — see `.claude/rules/engine-parity.md`.
- `default_config.yml` — the default canonical config used when no config file is given.
- `yaml_utils.py` — YAML load/dump helpers shared by the schema and the CLI.

See `CLAUDE.md` in the repository root ("Configuration: Canonical vs Legacy") for how this
schema relates to the legacy dict-based config format.
