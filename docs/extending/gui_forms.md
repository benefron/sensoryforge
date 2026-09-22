# Extending: making a component editable in the GUI

A component's settings form in the GUI is generated from its `get_param_spec()`
(see [ParamSpec](../developer_guide/extensibility.md)): `float` and `int` become spin
boxes, `bool` a checkbox, a spec with `choices` a drop-down, and a list-valued
default a JSON field. `group` becomes a collapsible section, and `advanced=True`
rows show only with **Advanced** on. A spec whose default is `None` shows *auto*
until set, and is written back as `None`.

**A plugin receptive-field builder, filter, neuron model or stimulus gets its form
with no GUI code**, as long as it is registered in the matching registry:

| Component | Where its form appears |
|---|---|
| Receptive-field builder (`INNERVATION_REGISTRY`) | Populations → Inputs, after choosing the builder |
| Filter (`FILTER_REGISTRY`) | Populations → Filter |
| Neuron model (`NEURON_REGISTRY`) | Populations → Neuron |
| Stimulus (`STIMULUS_REGISTRY`) | Stimulus |

`docs/examples/gui_plugin_param_form.py` registers a demo builder, shows its three
parameters rendering in the Inputs card, and edits one into the config:

```bash
QT_QPA_PLATFORM=offscreen python docs/examples/gui_plugin_param_form.py
```

It runs in CI through `tests/docs/test_docs_examples.py`.

## Show the value that runs

A form shows the spec's `default` for a parameter the config does not set. Make
`get_param_spec()`'s defaults the values your constructor uses, or the form will show
one value while another runs. The built-in builders read theirs from the
constructor signature for this reason (`sensoryforge.core.innervation`).

## What is not a plugin extension point

A new **preview** (the plots beside a form) or a new **screen** is an in-repo change:
screens are listed in `sensoryforge/gui/screens/__init__.py::SCREEN_FACTORIES` and
previews are built in each screen. Draw plots through
`sensoryforge.gui.widgets.plot_factory` and connect pyqtgraph signals only through
`plot_factory.connect` (see the ledger's F-035), and build each plot once for the
lifetime of its screen rather than discarding and rebuilding it.
