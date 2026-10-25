# Glossary

The words SensoryForge uses, defined once, in neutral vocabulary. Every term the agent
skills use is here; a term is defined by what it is in the code, so a definition names the
class, command or page that holds the facts. **Parameter lists, defaults and units are not
copied here**: read them with `sensoryforge describe --json` (see [Describing SensoryForge to
a program](https://benefron.github.io/sensoryforge/user_guide/describe/)).

## Read the user's words by meaning

A user rarely says these words exactly. Map what they say to the term it *means*, using the
whole session (what they have built, run or shown so far), not by matching the word:

- "Trial", "sample" or "example" may be a **draw** (one sampled stimulus) or an **entry**
  (a draw placed in a data set, with its time grid and noise repeat). If the session does not
  decide it, ask which.
- "Dataset", "benchmark" or "the test set" may be a **data set** (the `dataset:` file and its
  manifest), one **split** of it, or the **bundles** a run wrote from it.
- "Model" may be a **neuron model**, a **recipe** (a whole configured pipeline) or a **world**.
- "Input", "pattern" or "texture" may be a **stimulus**, a **class** of stimuli in a world,
  or a **pattern** (the arrangement of elements inside one layer).
- "Sensor" and "channel" are not the same: a **sensor array** has sensors at positions, and
  each sensor can have several **channels** (planes).
- A word from another field (a camera "pixel", an ear's "hair cell", a skin "receptor") is
  a **sensor**, unless the user says it is something else.

When the meaning is still unclear, ask one short question rather than guess. Never rename a
file's keys to the user's words: the file uses the terms below.

## Units and shapes

| Quantity | Unit | Where |
|---|---|---|
| time | ms at user-facing APIs; seconds inside ODE integration | `dt_ms`, `duration_ms`, `*_ms` fields |
| space | mm; coordinates are `(x, y)` | sensor positions, stimulus canvas, shape and pattern fields (`*_mm`) |
| current into a neuron | mA | filter output, `input_gain`, `input_floor` |
| membrane voltage | mV | neuron models, voltage probes |
| stimulus amplitude | the stimulus's own units; peak default 1.0, non-negative unless `signed: true` | `amplitude` |

Tensors are batch-first: `[batch, time, ...]`.

| Stage | Shape |
|---|---|
| stimulus | `[batch, time, H, W]`, or `[batch, time, C, H, W]` with `C` channels |
| sensor responses | `[batch, time, M]` (`M` sensors) |
| receptive-field weights | `[N, M]` (`N` neurons) |
| filter output and neuron drive | `[batch, time, N]`, mA |
| spikes | `[batch, time, N]`, bool |
| analog state readout | `[batch, time, N]`, float |

## The pipeline

### sensor
One sampling point of a sensor array, at an `(x, y)` position in mm. Index `k = i * cols + j`
on a regular grid. The stimulus is sampled at each sensor's own position (bilinear). *Also:*
receptor, pixel, hair cell, sampling point.

### sensor array
All of an experiment's sensors laid out by an arrangement (`grid`, `hex`, `poisson`,
`jittered`, `blue-noise`, `composite`, or one imported from coordinates); `GridConfig`,
registered in `GRID_REGISTRY`. *Also:* grid, sensor grid.

### channel
A named plane of a sensor array (`GridConfig.channels`); the stimulus drives one with
`StimulusConfig.channel`. A single-channel array is the default. *Also:* plane, color band.

### stimulus
What the sensors sample: a time series of frames. Built-in stimuli are registered in
`STIMULUS_REGISTRY`; a **world** generates them at scale.

### frame
One time step of a stimulus: an `[H, W]` image of the stimulus values, in the stimulus's
own units, on a canvas in mm.

### receptive field
How one neuron weights the sensors it reads: a row of the weight matrix `[N, M]`. All of a
population's receptive fields are one `ReceptiveFieldBank` (`core/rf_bank.py`), made by a
registered **receptive-field builder** (`gaussian`, `uniform`, `one_to_one`,
`distance_weighted`, `template`, `imported` or a plugin's). The registry is
`INNERVATION_REGISTRY` and the config key `innervation_method`.

### population
A group of `N` neurons sharing one receptive-field builder, filter and neuron model
(`PopulationConfig`). A config has one or more; inputs from several grids or channels are
**combined** by sum or concatenation.

### filter
The temporal stage between receptive fields and the neuron: it turns the weighted sensor
signal into a current (`[batch, time, N]`, mA). Registered in `FILTER_REGISTRY`.

### neuron model
The dynamics that turn drive into spikes (or, with no threshold, an analog **state**).
Registered in `NEURON_REGISTRY`; the equation DSL compiles new ones. Event converters
(`level_crossing`, `sigma_delta`) are registered here too.

### readout
What a population reports: **spikes** (`[batch, time, N]`, bool) for a model with a threshold,
or an analog **state** trace for one without. `PopulationConfig.readout` can force it.

### solver
The ODE integrator a neuron model uses (`SOLVER_REGISTRY`).

### processing layer
An optional stage between sensors and receptive fields (`PROCESSING_REGISTRY`), such as
`onoff`.

### recipe
A shipped, complete config (`sensoryforge list-presets`; `sensoryforge/presets/`). It fixes
the sensors, receptive fields, filters, neurons and gains together, so results from the same
recipe are comparable.

### config
The YAML that describes a run. The **canonical** config (`SensoryForgeConfig`: `grids`,
`populations`, `stimulus`, `simulation`) is the format to use; a **legacy** config has a
`pipeline` key.

### run
One simulation of a config on a stimulus by `SimulationEngine.run()` (the CLI, the batch
runner and the GUI all use it). It has a `dt_ms` step, an `integrate_dt_ms` integration step,
a duration and a seed.

## Worlds and data sets

### world
A declaration of stimulus **classes** (`world:` YAML, format `sensoryforge-world/1`). The same
world, seed and index always give the same draw. See [Worlds and Data
Sets](https://benefron.github.io/sensoryforge/user_guide/worlds/) and the [world contract](https://benefron.github.io/sensoryforge/reference/world_contract/).

### class
A family of stimuli in a world: a layer plus **axes** that say how its fields vary. A
**held-out class** is drawn only when named. *Not* the Python word.

### layer
What a class draws: a **shape**, a **pattern**, a **motion** and a **modulation**, plus
`background` and `clamp_min`.

### shape
The profile of one element (`gaussian`, `disc`, `bar`, `grating`, `gabor`, ...), registered
by `register_shape`.

### pattern
How elements are placed in a layer (`single`, `grid`, `list`, `random`, `braille`),
registered by `register_pattern`.

### motion
How a layer's elements move during a draw: `none`, `linear`, `circular` or `path`.

### modulation
A time envelope applied to a layer (`none`, `sine`, `pulses`), registered by
`register_modulation`.

### axis
One field a class varies: a constant, a `range`, a list of `values`, or a registered
**distribution**. Axes can be linked (`same_as`).

### distribution
How an axis draws its value (`uniform`, `log_uniform`, or one registered with
`register_distribution`).

### draw
One sampled instance of a class: its field values and its timing. Draw `i` never depends on
how many draws were asked for. Its **record** is the full set of declared values; only the
comparison stage reads it.

### data set
A world plus **splits** and a seed (`dataset:` YAML, format `sensoryforge-dataset/1`); `sensoryforge
dataset build` writes its manifest.

### split
A named part of a data set: `train` and `validation` (declared draws), `test` (stratified),
`probes` (draws just below and above each numeric axis's range), `held_out`, `sessions`,
`fixed`.

### entry
One row of a data set's manifest: a draw (or session) with its id, split, time grid and noise
repeat. One bundle is written per entry.

### session
A long entry made of several draws separated by gaps, up to a duration.

### probe
An entry from the `probes` split, placed outside an axis's range to see how the pipeline
behaves there.

## Outputs and evaluation

### bundle
The folder one entry's run writes: the stimulus, each population's responses and receptive
fields, the config and SensoryForge's sha (`io/bundle.py`; layout in [Data
Bundles](https://benefron.github.io/sensoryforge/user_guide/bundles/)). pressure-simulation reads them.

### report
What `sensoryforge evaluate run` writes for a world or data set: the audit, the declared
side, the comparison, the checks and a browsable view ([report
contract](https://benefron.github.io/sensoryforge/reference/report_contract/)).

### audit
The **blind** measurement of generated data. It reads only the bundles, never the world file
or a draw's declared record.

### declared side
What the world says each draw should contain, extracted by a **declared adapter**
(`DECLARED_ADAPTER_REGISTRY`).
*Also:* declared adapter.

### comparison
The stage that sets measured values (from the audit) beside declared ones and shows the raw
difference before any verdict.

### check
A reference-free test of a bundle's data, run by a **check pack** (`CHECK_PACK_REGISTRY`).
*Also:* check pack.

### metric pack
A set of measurements the audit can run (`METRIC_PACK_REGISTRY`): `core` for any modality,
plus a modality's own.

### snapshot
A folder from `sensoryforge evaluate export` that opens offline for sharing.

## Extending

### component
A class registered under a string name in one of the registries; created from config by name.
Passes `sensoryforge.testing.contracts.check_component`.

### registry
A `ComponentRegistry` that maps names (matched case-insensitively) to classes. Listed by
`sensoryforge list-components` and `sensoryforge describe`.

### ParamSpec
A component's description of one parameter: name, type, default, range, unit, choices and
help. The GUI forms and `sensoryforge describe` are generated from it.

### plugin
An installable package that registers components through the `sensoryforge.components` entry
point (`sensoryforge new-component`).

### contract
An interface another project depends on, pinned by tests and a docs page: the world,
converter and report contracts, published at
https://benefron.github.io/sensoryforge/reference/ (the `*_contract` pages). A change to what
one pins needs a format-tag bump.

### seed
An integer that makes a stochastic step repeatable. Worlds hash it with the draw index;
wiring and noise have their own seeds.

## Tactile today, general by design

SensoryForge's first use is touch. These words belong to the tactile pack, the tactile frame
renderer and the tactile world adapter, and appear nowhere else in the core. Each has a
neutral term above.

### receptor
A tactile **sensor**: a mechanoreceptor under the skin, at a position in mm.

### contact
A touch on the sensor array: onset, hold, optional slide, release. A tactile **draw** has
contacts; `sessions` have several.

### indentation, pressure
The tactile **stimulus** value: how deep or how hard an object presses, in the stimulus's
units. An **indenter** is a shape driven by depth rather than amplitude.

### episode
The timing of one contact: delay, touch, hold, slide, release, pause.

### SA, RA
**Slowly adapting** and **rapidly adapting** afferents: the two shipped tactile populations
(an `SAFilterTorch` or `RAFilterTorch` filter feeding a spiking neuron). SA answers a held
stimulus; RA answers its change.

### afferent
A tactile **neuron**: the nerve fibre that carries a receptor's signal.

### braille
A tactile **pattern** and its helpers: braille cells laid out as dots (`pattern: braille`,
`letter_text`, `braille_cells`).

### pressure-simulation
The sibling project that generates tactile stimuli and pins this code by git sha; it is why
the world, converter and report contracts exist.

### TouchSim
The published tactile simulator the `tactile_sa1_ra1` recipe is fitted against
(its comparison is a test in the repository's validation tests).
