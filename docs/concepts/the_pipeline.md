# One experiment, one config

The GUI edits exactly one `SensoryForgeConfig` — the same object `sensoryforge run`,
the batch executor and `SimulationEngine` read. There is no GUI-only format:
**File > Save config** writes the YAML the CLI runs, and **File > Open config** reads
any canonical YAML. See [the GUI walkthrough](../user_guide/gui_walkthrough.md) for
the screens.

## The session

`sensoryforge.gui.session.Session` holds the config. Every screen reads from it and
writes into it by dotted path (`populations.1.filter_params.tau_r`); each write
announces the path, so the other screens, the pipeline strip and the run bar update
without knowing about each other. Replacing the config (opening a file or a preset)
tells every view to rebuild.

The session also holds the validation result, `Session.errors`: what would stop the
engine building this config, keyed by pipeline stage (`grids.0`, `stimulus`,
`populations.1.rf`, `populations.1.combine`, `populations.1.filter`, ...). It is
computed with the engine's own construction code, so the GUI reports a problem while
you edit it, not when a run starts.

## The pipeline strip

The strip draws the config as the pipeline it describes:

```
Sensors      [grid A]  [grid B]
Population 1 [sensor array] → [receptive field] → [filter] → [neuron] → [readout]
Population 2 [sensor array] → [receptive field] ⟶ [combine] → [filter] → ...
                              [receptive field] ⟋
```

One row per population, one chip per stage, a second receptive-field chip and a
*combine* chip when a population has several inputs. A chip's dot is grey at the
default, green when set and red when `Session.errors` names that stage.

## One run path

Every GUI run goes through `sensoryforge.gui.execution.run_controller.RunController`:
it renders the stimulus with `sensoryforge.stimuli.render.render_for_config` (the
renderer the CLI uses) and calls `SimulationEngine.run()` on a background thread. A
sweep runs each job as a separate `sensoryforge` process. There is no second
simulation path in the GUI to drift from the engine, and
`tests/integration/test_gui_engine_equality.py` checks the GUI, the engine and the
CLI give identical results for the same config.

**Rule:** a GUI feature that cannot be expressed as a `SensoryForgeConfig` field is
not a feature. Extend the schema first, then the screen.
