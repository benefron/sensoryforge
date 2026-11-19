---
name: sensoryforge-extend-evaluation
description: Add a metric pack or a declared adapter to SensoryForge's evaluation, as a plugin that passes the contract check. A metric pack is what the audit measures; a declared adapter is what a data set's own declaration says each entry should show; together they let the evaluation measure something new or work on data from another sense. Use when someone asks whether the audit can measure a quantity it does not, wants a new statistic or metric over generated data, wants to compare the data against what their world file declares for a world SensoryForge has no adapter for, wants to evaluate data from vision, hearing or any other modality, or asks how to add a check, a metric or a report column, in whatever words they put it.
---

# Extend the evaluation

The evaluation reads a generated data set twice and sets the two readings side by side.
The **audit** measures it blind, from the frames and responses alone, with **metric
packs**. The **declared stage** reads what the data set's own declaration says each entry
should show, with a **declared adapter**, in the same metric ids and units. Both are
looked up by name in a registry, so you add one by registering it from a small plugin;
nothing inside SensoryForge is edited. You end with a plugin that passes the contract
check on a real data set, and a report the user has read. A new neuron, filter or grid is
`sensoryforge-extend-pipeline`; a new stimulus or a world for another modality is
`sensoryforge-extend-world`.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it, before anything else, even when the answer
turns out to be that it cannot be done. If the command is not found, SensoryForge is not
installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Which seam, and is it already there?

Run `sensoryforge describe --section evaluation`: it lists the names of the packs and
adapters that exist (not their metrics). Choose by meaning:

- **A metric pack** answers "what does the data show". It sees one entry's frames, times,
  layout and populations, and nothing else. A new statistic over frames or spikes, or a
  measure of a quantity that only another sense has (a colour, a pitch), is a pack.
- **A declared adapter** answers "what should it show, according to the data set's own
  file". It reads the declaration, never the data. It is needed once per kind of world: the
  built-in one reads a tactile world. Data from another modality that has a world class
  kind (`sensoryforge-extend-world`) needs an adapter to be compared with its declaration;
  without one the audit still runs and the comparison is empty.
- **A frame renderer** (how the viewer draws a frame) is not a seam you can add to yet: no
  registry for it exists in this release, and the viewer draws channel 0 as a heat map
  itself. If the user needs another picture, say so and tell them it is
  recorded as an open item; do not invent a registry.

An existing pack may already do it. A pack's metric ids, units and definitions are in the
list of metrics in any `audit.json` a run wrote (`sensoryforge-evaluate` makes one; build a
small one if there is none): read it, show the user the metric that matches (name only ids you read there), and stop if
they are content.

## 2. Get the definition from the user

What a metric means is the user's to give: the quantity, its unit, how an entry with
nothing in it is handled (a missing value is `None`, not 0), and one case they can check
by hand. For an adapter: which metric ids the declaration determines, in what unit, and
each case it does not determine (those are left out with a reason, never guessed). Use
`sensoryforge-ground-assumptions` for anything missing and list each value with where it
came from. A frame has no unit of its own: it is the unit the data names, or none.

## 3. A metric pack

Subclass `MetricPack` (`sensoryforge.evaluate.pack`), set its name and version, and
fill in:

- **`metrics(layout, units, dt=None)`** returns one `metric_definition()` per metric, in
  catalogue order. An id is `<pack>.<quantity>.<metric>` (`peaks.frames.n_sensors_over_half`),
  unique in the run. Its scope is `"entry"`, `"channel"` (one value per channel) or
  `"population"`; its value type `"number"`, `"integer"` or `"string"`; its unit is a unit
  object or `None` when nobody states one (use `unit_of()` from
  `sensoryforge.evaluate.audit.units` to build a ratio such as space over time).
- **`measure(view)`** returns `{"id", "value"}` for each metric, in catalogue order, with
  `channel` or `population` added where the scope asks. `view.frames` is `[T, C, M]`,
  `view.time`, `view.dt`, `view.layout`, `view.populations` (each with its data, drive
  and filtered current) and `view.units`. Use numpy; a value is a plain number or `None`; never NaN.
  It must be the same on every call: no clock, no random numbers.
- **Parameters** are constructor arguments; `to_dict()` returns them and `get_param_spec()`
  lists them as `ParamSpec`. `from_config(to_dict())` must give back the same pack.
- **A data-set level value** (how many entries are silent) lists the per-entry ids it needs
  in `dataset_inputs=` and implements `measure_dataset(rows)`.
- **The blind rule.** The pack's module must not import `sensoryforge.world`, directly or
  through another module: a test fails any pack that can reach the declaration. A statistic
  that needs the declaration is not a pack; it belongs to an adapter or to the comparison.

## 4. A declared adapter

Subclass `DeclaredAdapter` (`sensoryforge.evaluate.declared`), set its name and version,
and implement: `accepts(dataset_meta)` (a class method: can it read this data set's
`dataset.json`), `load()`, `describe()` (the data set's identifiers), `units()` (every
quantity the audit names, `None` where the declaration states none), `metrics(units)` (one
`{id, unit, basis, source}` each, in catalogue order), `declare(row, world)` (one
`Declaration(values, omitted)` per manifest row) and `bounds()` (ranges for display, or an
empty list). Rules that keep the two sides comparable:

- Use the audit's metric ids, and state each value in the unit the audit reports for that
  id; the comparison converts nothing.
- Write a value only where the declaration determines it. Give every value you leave out a
  reason in the Declaration's omitted list; the stage counts them, and a missing value is not an error.
- Values come in catalogue order, with `channel` set when the data has more than one.
- The adapter may import the world package, so it lives in a module of its own, never in
  the module of a pack. The stage reads `dataset.json` and the manifest only.

## 5. Check it

Register the plugin, then run `check_evaluation_component(kind, name, example=FOLDER)`
from `sensoryforge.testing.evaluate_contracts`, where the kind is `"metric_pack"` or
`"declared_adapter"` and the example is a data-set folder. Without a data set to point at,
build a small one first (`sensoryforge evaluate run`, as `sensoryforge-evaluate` shows).
The check runs the stage twice on that data and raises `AssertionError` naming the rule:
a metric id outside the pack's name, a value for an id nobody defined, values out of
order, a failed `measure()`, a result that changes between runs, a pack that imports the
world, an adapter that declares an id outside its catalogue or leaves out a value with no
reason. Report that the check passed, by that name; fix the plugin, not the check. Then
read the numbers yourself against the user's hand case from step 2, which the check
cannot know. Pytest is not installed with SensoryForge: call the check from a few lines of
Python, as the example does.

## 6. Use it

Package the module (a folder with `pyproject.toml` and a `sensoryforge.components` entry
point to the register function, as `sensoryforge new-component` writes for the other
kinds; it has no template for these two) and install it into the same environment.
Then `sensoryforge evaluate run FILE --out runs/NAME ... --packs core,<pack-name>
--adapter <adapter-name>` (the default is the core pack alone, and the default adapter is the
first that accepts the data set) and `python -m
sensoryforge.evaluate.audit runs/NAME --packs core,<pack-name>` use them. Or load the plugin in
Python: `load_plugin_import_paths(["module:register"])`. Open the report with the user:
a new metric appears in `audit.json`, a declared one in `declared.json`, and the raw
differences in `compare.json`.

## Verify

The contract check passes for each piece you registered on a data set that was really
built, the user has seen the metric's values and agreed they match their hand case, and no
definition, unit or formula in the plugin came from you unconfirmed.

## Worked example

A pack that counts the sensors that ever pass half of full scale, and an adapter that
declares only each entry's duration (it reads the manifest), checked against a small data
set and run:

```yaml
# file: world.yml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 40]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
```

```python
# file: demo_eval.py
"""An evaluation plugin: a metric pack and a declared adapter."""
from sensoryforge.evaluate.audit.units import QUANTITIES, unit_of
from sensoryforge.evaluate.declared import Declaration, DeclaredAdapter
from sensoryforge.evaluate.pack import MetricPack, metric_definition
from sensoryforge.evaluate.registry import (
    DECLARED_ADAPTER_REGISTRY,
    METRIC_PACK_REGISTRY,
)

PEAKS = "peaks.frames.n_sensors_over_half"


class PeakCount(MetricPack):
    """How many sensors ever pass half of 1 in an entry's frames."""

    name = "peaks"
    version = "1"

    def metrics(self, layout, units, dt=None):
        return [metric_definition(PEAKS, None, scope="entry", value_type="integer")]

    def measure(self, view):
        # view.frames is [T, C, M]; the view holds nothing of the world's declaration.
        return [{"id": PEAKS, "value": int((view.frames.max(axis=(0, 1)) > 0.5).sum())}]


class DurationOnly(DeclaredAdapter):
    """Declares only the entry's duration, which the manifest states."""

    name = "duration_only"
    version = "1"

    @classmethod
    def accepts(cls, dataset_meta):
        return "dataset_id" in dataset_meta

    def load(self, dataset_meta):
        return dict(dataset_meta)

    def describe(self, dataset_meta, world):
        return {
            "dataset_id": str(world["dataset_id"]),
            "world_id": str(world.get("world_id", "")),
            "world_format": "none",
        }

    def units(self, world):
        return {q: None for q in QUANTITIES}

    def metrics(self, units):
        return [
            {
                "id": "core.frames.duration",
                "unit": unit_of("time", units),
                "basis": "manifest",
                "source": "the manifest's duration_ms",
            }
        ]

    def declare(self, row, world):
        value = {"id": "core.frames.duration", "value": float(row["duration_ms"])}
        return Declaration(values=[value], omitted=[])

    def bounds(self, world):
        return []


def register():
    METRIC_PACK_REGISTRY.register("peaks", PeakCount)
    DECLARED_ADAPTER_REGISTRY.register("duration_only", DurationOnly)
```

```python
# file: use_eval.py
"""Register an evaluation plugin, check it against a data set, and run it."""
import json

from sensoryforge.evaluate.audit.run import run_audit
from sensoryforge.evaluate.declared import run_declared
from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.testing.evaluate_contracts import check_evaluation_component

print("loaded", len(load_plugin_import_paths(["demo_eval:register"])))
print("loaded again", len(load_plugin_import_paths(["demo_eval:register"])))

data = "runs/demo"
check_evaluation_component("metric_pack", "peaks", example=data)
check_evaluation_component("declared_adapter", "duration_only", example=data)
print("contract ok")

audit = run_audit(data, "runs/demo/plugin", packs=["core", "peaks"])
row = json.loads(open("runs/demo/plugin/audit_entries.jsonl").readline())
print("peaks", [v["value"] for v in row["values"] if v["id"] == "peaks.frames.n_sensors_over_half"])
declared = run_declared(data, "runs/demo/plugin", adapter="duration_only")
print("declared", [m["id"] for m in declared["metrics"]], "rows", declared["entries"]["n"])
```

```bash
sensoryforge evaluate run world.yml --out runs/demo --preset tactile_sa1_ra1_adex --draws 6 --duration-ms 80 --seed 7 --packs core,tactile
# prints: 0 failed
python use_eval.py
# prints: loaded again 1
# prints: contract ok
# prints: declared ['core.frames.duration'] rows 6
```
