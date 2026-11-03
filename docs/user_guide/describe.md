# Describing SensoryForge to a program

A coding agent, an editor or a script should read what SensoryForge accepts from the
program itself, not from prose that can go stale. Three tools do that. Each is generated
from the live registries and loaders, so a plugin's components appear in them and a
removed component disappears.

## `sensoryforge describe`

```bash
sensoryforge describe                       # a short human listing
sensoryforge describe --json                # everything, as JSON
sensoryforge describe --json --section world
```

`--section` is one of `formats`, `components`, `world`, `evaluation`, `recipes`. The JSON has
these parts:

| Part | What it holds |
|---|---|
| `formats` | the version tags of the world, data-set, bundle and description formats, and the SensoryForge version |
| `components` | for each kind (`neuron`, `filter`, `stimulus`, `solver`, `grid`, `innervation`, `processing`), every registered name with its class, a one-line summary, `params` and `init_params` |
| `world` | the shapes, patterns, modulations, distributions and class kinds a `world:` file may name, with each one's parameters; the keys a world, class, layer, axis and data-set split may carry |
| `evaluation` | the registered metric packs, check packs and declared adapters |
| `recipes` | the shipped presets, each with a digest of its grids and populations |

A parameter in `params` is a `ParamSpec`: `name`, `label`, `dtype`, `default`, `min`, `max`,
`step`, `unit` (empty for a dimensionless one), `choices`, `tooltip`, `help`, `group` and
`advanced`. Units are the component's own; nothing is copied from the docs.

The pages of the API Reference that list components, world kinds, evaluation packs, recipes and
format tags are generated from this output by `python scripts/build_reference.py`:
[Component Reference](../reference/components.md), [World Language
Reference](../reference/world_language.md), [Evaluation Packs
Reference](../reference/evaluation.md), [Recipe Reference](../reference/recipes.md) and [File
Format Tags](../reference/formats.md). A test fails when a page differs from the output, so the
pages and `describe` cannot disagree.

**An empty `params` list means the component declares no `ParamSpec`**, not that it takes no
arguments. `init_params` is then the parameter list: each constructor argument with `required`
and, when it is a plain value, its `default`.

## JSON Schemas

```bash
sensoryforge describe --schema config     # also: world, dataset
python scripts/build_schemas.py           # rewrite schemas/*.schema.json
python scripts/build_schemas.py --check   # CI: fail if a committed copy is stale
```

`schemas/config.schema.json`, `schemas/world.schema.json` and `schemas/dataset.schema.json`
are committed so an editor can use them (for example the YAML language server's
`# yaml-language-server: $schema=...` comment).

- The **config** schema is built from the `SensoryForgeConfig` dataclasses: field names,
  types and defaults. It allows keys it does not know, because the loader ignores them;
  `validate --json` reports those as warnings. Registered component names are listed as
  `examples`, not as an `enum`: a plugin can add names, and the loader matches names
  case-insensitively.
- The **world** and **data-set** schemas are strict about the top-level, class, layer,
  session and split keys, as their loaders are. The kinds a shape, pattern or class may
  take are listed as `examples` for the same reason.

A schema checks the shape of a file. Rules that involve several values together (a probe
outside a field's range, a held-out class that is also a class) are checked only by the
loader: use `validate`.

## `sensoryforge validate --json`

```bash
sensoryforge validate --json my_world.yml
sensoryforge validate --json --kind dataset my_dataset.yml
```

The kind is decided from the keys (a `grids` list is a config, `classes` a world, `splits`
a data set) unless `--kind` says. The report:

```json
{
  "format": "sensoryforge-validate/1",
  "file": "my_world.yml",
  "kind": "world",
  "valid": false,
  "errors": [
    {
      "path": "world.classes.a.layer.shape",
      "message": "unknown shape kind 'gausian'; known: ['bar', ...]",
      "fix": "did you mean 'gaussian'? Valid: bar, ...",
      "source": "world"
    }
  ],
  "warnings": []
}
```

- `path` is the dotted place in the file (`populations.0.neuron_model` is the first
  population's neuron model; the empty string is the file as a whole). For a YAML syntax
  error the record also has `line`.
- `fix` is a suggestion taken from facts the message or a registry states: the nearest valid
  name for a misspelling, or the hint a loader already wrote. It is `null` when there is
  none; nothing is guessed.
- `source` is the stage that found the problem: `yaml`, `schema`, `build` (the engine's own
  receptive-field, filter and neuron builds, the same checks the GUI runs), `world` or
  `dataset`.
- Unknown keys in a config are `warnings`: the loader ignores them, so they do not make the
  file invalid, but a misspelled key silently does nothing.

The exit code is 0 when `valid` is true and 1 otherwise.
