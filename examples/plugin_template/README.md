# sensoryforge-plugin-template

A complete, standalone, installable SensoryForge plugin package -- the
worked example for `docs/developer_guide/plugins.md`. Copy this directory
as the starting point for your own plugin, or use it directly as a GitHub
template repository (see below).

It adds two components, each an independent worked example of one
extension point:

| Component | Kind | Registry | Name | File |
|---|---|---|---|---|
| `RadialFalloffRFBuilder` | receptive-field builder | `INNERVATION_REGISTRY` | `radial_falloff` | `sensoryforge_plugin_template/rf_builder.py` |
| `GainThresholdLayer` | processing layer | `PROCESSING_REGISTRY` | `gain_threshold` | `sensoryforge_plugin_template/processing.py` |

Both are registered through the `sensoryforge.components` entry-point group
declared in `pyproject.toml` -- no changes to a SensoryForge checkout are
needed to use them, and neither is imported by this package's own
`__init__.py` (SensoryForge discovers and imports them lazily).

## Install

```bash
pip install -e .
```

(or, for a real release, `pip install sensoryforge-plugin-template`, once
published). Installing registers both components' entry points in your
environment's package metadata. From that point on, any SensoryForge
process that imports `sensoryforge` -- the CLI, the GUI, `BatchExecutor`, a
plain script -- discovers and registers them automatically.

## Verify

```bash
sensoryforge list-components
```

`radial_falloff` appears under "Innervation Methods" and `gain_threshold`
under "Processing Layers". Any canonical config can now reference them:

```yaml
populations:
  - name: my population
    innervation_method: radial_falloff
    innervation_params:
      radius_mm: 0.5
    processing:
      - method: gain_threshold
        gain: 2.0
        threshold: 0.1
```

## Test

```bash
pip install -e ".[test]"   # pytest only; sensoryforge itself is a runtime dependency
pytest
```

`tests/test_contract.py` runs `sensoryforge.testing.contracts.check_component`
against both components -- the same shape/round-trip/param-spec checks used
across SensoryForge's own contract-test sweep.
`tests/test_entry_point_discovery.py` proves discovery through the real
`importlib.metadata` entry-point mechanism (not an in-process registry
call): it builds a real `*.dist-info` next to a copy of this package's own
source, puts it on `sys.path`, and checks that both the library's discovery
function and the `sensoryforge list-components` CLI (run as a subprocess)
find the two components.

## Publishing this as a GitHub template

To turn a copy of this directory into a reusable starting point for other
plugin authors:

1. Push this directory (as its own repository, e.g.
   `git subtree split --prefix=examples/plugin_template -b plugin-template`
   from a SensoryForge checkout, then push that branch to a new repo) to
   GitHub.
2. In the new repository's **Settings**, tick **Template repository**
   (GitHub's own "repository template" feature -- see GitHub's
   documentation on "Creating a template repository").
3. Anyone can then click **Use this template** on that repository's page to
   get their own copy with a clean git history, rename
   `sensoryforge_plugin_template`/`sensoryforge-plugin-template` throughout
   (`pyproject.toml`'s `name` and `[project.entry-points]` table,
   `tests/`'s imports), replace `RadialFalloffRFBuilder`/`GainThresholdLayer`
   with their own component(s), and `pip install -e .` to start developing.

See `docs/developer_guide/plugins.md` for the full plugin-package reference
(the `sensoryforge new-component` scaffold generator, the `plugins:` YAML
alternative, and the component contract in detail).
