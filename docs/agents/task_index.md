# Task index

For a task in the user's words: the files to change, the tests that pin them, and the rules
and contracts to read first. [`AGENTS.md`](https://github.com/benefron/sensoryforge/blob/main/AGENTS.md)
gives the first hop for each area; this page goes one step further. Terms are defined in the
[glossary](glossary.md). Match the user's words to a row by meaning, not by wording, and ask
when two rows fit.

Parameter lists, defaults and units are never copied here: `sensoryforge describe --json`
prints them from the registries.

| Task | Files | Tests | Read first |
|---|---|---|---|
| Add a stimulus | `sensoryforge/stimuli/`, `sensoryforge/register_components.py` | `tests/unit/test_stimulus_params_and_effective_defaults.py`, `tests/contract/test_component_contracts.py` | `docs/developer_guide/add_stimulus.md`; every stimulus defaults to peak 1.0, non-negative |
| Add a neuron model | `sensoryforge/neurons/` | `tests/validation/test_neurons_analytic.py`, `tests/contract/test_component_contracts.py` | `docs/developer_guide/add_neuron.md`; round-trip every constructor argument through `to_dict()` |
| Add a filter | `sensoryforge/filters/` | `tests/validation/test_filters_analytic.py`, `tests/contract/test_component_contracts.py` | `docs/developer_guide/add_filter.md`; defaults come from `sensoryforge/config/defaults.py` |
| Add a receptive-field builder | `sensoryforge/core/rf_builders/`, `sensoryforge/core/rf_bank.py` | `tests/unit/test_rf_builders_build.py`, `tests/validation/test_receptive_fields_analytic.py` | `docs/developer_guide/add_rf_builder.md`, `docs/user_guide/receptive_fields.md` |
| Add a component as a separate package | `sensoryforge new-component <kind> <Name>` | `tests/unit/test_plugin_discovery.py`, `tests/unit/test_new_component_scaffold.py` | `docs/developer_guide/plugins.md` |
| Change a recipe or its gains | `sensoryforge/presets/`, `scripts/calibrate_recipe_gains.py` | `tests/unit/test_presets.py`, `tests/validation/test_touchsim_comparison.py` | `docs/user_guide/units_and_gains.md`; re-run the calibration script, do not retune a gain by hand |
| Change a config key or its default | `sensoryforge/config/schema.py`, `sensoryforge/config/defaults.py` | `tests/unit/test_config_schema.py`, `tests/unit/test_describe.py` | `docs/user_guide/configuration_schema.md`; then `python scripts/build_schemas.py` |
| Run a config | `sensoryforge/core/simulation_engine.py`, `sensoryforge/cli.py` | `tests/integration/test_gui_engine_equality.py`, `tests/validation/test_reproducibility.py` | `docs/user_guide/cli.md`, `docs/developer_guide/architecture_notes.md` |
| Check a file before running it | `sensoryforge validate --json FILE` | `tests/unit/test_describe.py` | `docs/user_guide/describe.md` |
| See what components and kinds exist | `sensoryforge describe --json` | `tests/unit/test_describe.py` | `docs/user_guide/describe.md` |
| Read a bundle | `sensoryforge/io/bundle.py` | `tests/unit/test_bundle_io.py` | `docs/user_guide/bundles.md`; the bundle schema version changes with any new field |
| Add a field to a bundle | `sensoryforge/io/bundle.py` | `tests/unit/test_bundle_io.py`, `tests/contract/test_world_contract.py` | `docs/reference/world_contract.md`; a field the file does not set must not appear |
| Design a world or a stimulus class | `sensoryforge/world/`, `sensoryforge/stimuli/layered.py` | `tests/unit/test_world_schema.py`, `tests/unit/test_layered_stimuli.py` | `docs/user_guide/worlds.md`, `.claude/rules/world-engine.md` |
| Add a world shape, pattern, modulation or distribution | `sensoryforge/world/kernel.py`, `sensoryforge/world/distributions.py` | `tests/contract/test_world_contract.py` | `docs/reference/world_contract.md`, `.claude/rules/world-engine.md`; register it, do not special-case the renderer |
| Build a data set | `sensoryforge dataset build`, `sensoryforge/world/dataset.py` | `tests/unit/test_world_schema.py` | `docs/user_guide/worlds.md` |
| Run a data set | `sensoryforge batch --dataset`, `sensoryforge/world/runner.py` | `tests/unit/test_batch_executor_bundles.py` | `docs/user_guide/batch_processing.md` |
| Evaluate a world or data set | `sensoryforge evaluate run`, `sensoryforge/evaluate/` | `tests/unit/test_audit_run.py`, `tests/contract/test_audit_import_wall.py` | `docs/user_guide/evaluate_pipeline.md`, `docs/reference/report_contract.md`; the audit never imports `sensoryforge.world` |
| Add a metric pack, check pack or declared adapter | `sensoryforge/evaluate/registry.py`, `sensoryforge/evaluate/checks/registry.py` | `tests/unit/test_audit_metrics.py` | `docs/developer_guide/audit_metrics.md`, `docs/developer_guide/declared_adapter.md` |
| Change an event converter | `sensoryforge/neurons/event_encoders.py` | `tests/contract/test_converter_contract.py` | `docs/reference/converter_contract.md`; never change arithmetic on a path that does not use a new parameter |
| Change the GUI | `sensoryforge/gui/app.py`, `sensoryforge/gui/screens/` | `tests/gui_v2/` | `docs/developer_guide/architecture_notes.md` (GUI Structure) |
| Make a run faster | `benchmarks/` | `benchmarks/` | `docs/reference/benchmarks.md` |
| Change the agent files | `sensoryforge/agents/`, `AGENTS.md`, `docs/agents/` | `tests/unit/test_front_door.py`, `tests/unit/test_agents_style.py`, `tests/unit/test_agent_glossary.py` | `docs/developer_guide/agent_files.md` |
| Find out whether something was already decided | `docs_root/LEDGER.md` | | the entry ids (`F-`, `D-`) |
