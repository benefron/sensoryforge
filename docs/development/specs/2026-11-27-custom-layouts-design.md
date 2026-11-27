# Custom sensor and neuron layouts: design

2026-11-27. Status: **waiting for approval**. Nothing here is built.

## Intent

A user can write a layout once (a spiral, rings, measured positions) and name it in a
config, for sensors, for neurons or both, without editing SensoryForge. This is the second
step of the decision that an unknown neuron arrangement is an error: first refuse unknown
names, then let users add names.

## How it works today

- **Neurons.** `create_neuron_centers` knows five names and builds `rows x cols` points.
  It does not look at any registry.
- **Sensors.** `build_grid` passes `GridConfig.arrangement` to `ReceptorGrid`, which knows
  the built-in names. `GRID_REGISTRY` holds layout classes, but the engine builds none of
  them. A plugin layout reaches a run only through `coords_file`, and the config has no
  place for its settings.
- **Already suited to this.** `BaseGrid` takes `xlim`, `ylim` and `device`. The template
  `sensoryforge new-component grid` writes also takes `seed` and its own settings. The
  receptive-field bank takes the neuron count from the `[N, 2]` centres.

## Goals

1. One registered class works as a sensor layout and as a neuron layout.
2. The config gives its name and its own settings.
3. The layout decides how many points there are.
4. The same config and seed give the same points. As with the built-in random layouts, a
   random layout given no seed may differ from run to run.
5. The CLI, engine, app, `describe` and `check_component` read the same registry.

## Non-goals

- Changing how any built-in layout places points. Their results stay bit-identical.
- Custom arrangements inside `composite` grid layers.
- 3D positions, or data attached to each point (type, size).
- The older pipeline classes (`generalized_pipeline`, `pipeline`, `notebook_pipeline`) as
  a way to run a custom layout.
- A built-in "neuron positions from a file" layout. A plugin can read a file.

## Approaches

| Approach | For | Against |
|---|---|---|
| **A. Extend the existing grid plugin** (`BaseGrid`, `GRID_REGISTRY`) | One kind of plugin. The generated template already nearly fits. `describe`, `list-components` and the contract check already know it | A grid plugin whose constructor does not take the area cannot be named in a config until changed |
| B. A new layout registry holding plain functions | The smallest thing to write | Two overlapping kinds of layout plugin, and the existing one stays unusable from a config |
| C. Position files for neurons too, as sensors have | Least code | No settings in the config, an extra file to keep with every config |

**Chosen: A.** It was chosen on the author's judgement after the user decided that one
plugin type serves both uses. The user also decided that the layout sets the count.

## Design

### The plugin

A layout that a config can name is a `BaseGrid` subclass registered in `GRID_REGISTRY`
whose constructor accepts, as keywords:

- `xlim`, `ylim`: the area to fill, in mm;
- `device`;
- `seed`, only if the layout is random;
- its own settings, each with a default or required, and listed by `get_param_spec()`.

`get_all_coordinates()` returns `[N, 2]`, `(x, y)` in mm, `N >= 1`.

One shared function builds the points for both uses and checks them. It raises
`ValueError`, naming the layout, when the shape is wrong, a value is not finite, a point
lies outside the area (with a float tolerance), a setting is unknown or missing, or the
constructor does not accept `xlim` and `ylim`. It never moves or drops points.

A grid plugin that does not accept the area still works through `coords_file` as today.
The spiral example (`docs/examples/grid_arrangement_plugin.py`) changes to take the area.

### Config

| Use | Name (existing key) | Settings (new key) |
|---|---|---|
| Sensors | `grids[].arrangement` | `grids[].layout_params` |
| Neurons | `populations[].neuron_arrangement` | `populations[].neuron_layout_params` |

Both new keys default to empty and are **left out of `to_dict()` when empty**, so existing
configs serialize as before. Settings given with a built-in name are an error. A name is
matched exactly as registered. A plugin cannot take a built-in name.

### Sensors

- The area is the rectangle that `rows`, `cols`, `spacing` and the centre describe, as for
  `hex` and `poisson` today. The stimulus canvas stays as it is.
- The engine builds the layout as it builds a position-file grid (the "flat" path), with
  one difference: the grid's bounds are the declared area, not the box around the points.
  This keeps the canvas, the sensor sampling and the neuron area in agreement.
- `rows` and `cols` set the area and canvas only. They do not set the sensor count.
- `density` is an error. The area must have a width and a height above zero.
- `coords_file` together with a custom name is an error.
- `GridConfig.seed` is passed to the layout. A seed given to a layout that takes none is
  an error.

### Neurons

- The area is the target grid's bounds, shrunk by `edge_offset`.
- The count is the number of points returned. `neurons_per_row`, `neuron_rows` and
  `neuron_cols` are ignored, with a warning when the user set one.
- `neuron_jitter_factor` is ignored, with a warning when set above zero.
- `PopulationConfig.seed` is passed to a layout that takes `seed`. A layout without it is
  deterministic, so a population seed (which also seeds the wiring) is not an error here.
- For a receptive-field builder that derives its own centres, a custom neuron layout is
  ignored, with a warning.
- Several inputs share one set of centres, from the first input that needs them, as today.

### Where else it shows up

- **CLI.** `validate` accepts built-in names and registered layouts, and reports bad
  settings with the same message the engine gives.
- **`check_component("grid", ...)`.** When a layout accepts the area: two areas, points
  inside each, and the same seed gives the same points twice.
- **Scaffold and docs.** The template matches the contract. One how-to page covers both
  uses. Schema and reference pages are regenerated.
- **App.** Both arrangement lists show built-ins and registered layouts. Settings use the
  existing parameter form from `get_param_spec()`. Neuron row and column fields are
  disabled for a custom layout. A config naming a layout that is not installed shows the
  error and keeps the name; it is never replaced by another.
- **Saved runs.** Bundles keep storing neuron centres as `[N, 2]`. The config saved in a
  bundle gains a new key only when a custom layout has settings. Rebuilding an engine
  from such a bundle needs the plugin installed, and fails by name without it. The
  `parameters.neurons_per_row` field stays for compatibility and does not describe the
  count of a custom layout; the bundle docs say so.
- **Evaluation.** The audit reads frames on the stimulus canvas, which does not change.
  Engine checks that assume a symmetric or rows x cols arrangement must report
  `skipped`, with the reason, when they cannot apply.

## Constraints

- Built-in sensor and neuron layouts stay bit-identical. Golden tests must pass
  unchanged.
- `pressure-simulation` pins SensoryForge by commit. A search of its code outside the
  archive found that it places neurons with its own `create_neuron_centers`, builds
  SensoryForge's `ReceptorGrid` only with built-in arrangements, and does not use
  `GRID_REGISTRY`. Its test fixtures hold saved configs with the built-in `grid` neuron
  arrangement. No change to a world draw, frame, seed, entry id or
  converter step, so no format tag changes. If building shows otherwise, stop and ask.

## Stages

1. **Core:** shared builder, config keys, engine, CLI, contract check, scaffold, example,
   docs. Usable from YAML and Python.
2. **App:** lists, settings forms, disabled fields, missing-plugin message.

## Done when

- A spiral plugin installed as a package runs from a YAML config as the sensor layout, as
  the neuron layout, and as both, with no `coords_file` and no change to SensoryForge.
- Its saved bundle rebuilds an engine that gives the same spikes.
- Each error above has a test, and the full suite passes with golden files unchanged.

## To verify while planning

- Which stimulus types in `run` and `batch` go through the older pipeline adapter, and
  whether it reads a layout field. A custom layout must render the same frames there, or
  be refused by name.
- Each engine check in `evaluate/checks` against a layout with an arbitrary count.
- Whether batch sweeps and the app's sweep screen can vary a value inside the new
  settings. If they cannot, sweeping layout settings is a later change, not part of this.
- Every app view that works out a neuron count or draws neurons from rows and columns.

## Left to the builder

Names of the shared function and its module, the wording of messages, the float
tolerance, and the form layout in the app.
