# Config Directory

> **Status:** CORE – Single source of truth for pipeline + analytical sweep parameters. Update `pipeline_config.yml` first, then mirror key overrides in demos/tests.

## Files

- `pipeline_config.yml` – Canonical encoding + decoding sweep configuration consumed by pipelines, GUIs, CLIs, and tests.
- `analytic_demo_config.yml` – Minimal subset for CLI/GUI demos; keep schema aligned with `pipeline_config.yml`.
- `viz.yaml` – Rendering + plotting defaults for GUI visualizers and analytical reports.

## Archive

- `archive/full_control_config.yml` – Legacy generalized-pipeline override showcasing per-neuron parameter control. Treat as documentation; start from `pipeline_config.yml` for new work.
