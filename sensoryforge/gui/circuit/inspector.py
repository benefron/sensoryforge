"""The Circuit tab's inspector (Phase 3, Wave P): render node parameters from
``get_param_spec()``.

This is the payoff for the Phase 1g plugin contract: a component registered
through :mod:`sensoryforge.registry` -- built-in or a third-party plugin --
gets a working GUI panel with no GUI code of its own, as long as it
implements ``get_param_spec()`` (CLAUDE.md, "ParamSpec and get_param_spec()").

Two layers:

* :func:`build_param_form` -- the generic part. Takes a list of
  :class:`~sensoryforge.stimuli.base.ParamSpec` plus a ``values`` dict and
  returns a ``QWidget`` with one labelled control per spec: a spin box for
  ``"float"``/``"int"``, a checkbox for ``"bool"``, or -- taking priority
  over ``dtype`` -- a combo box whenever ``choices`` is set. ``unit`` is a
  spin box suffix, ``tooltip`` is hover text, ``group`` becomes a section
  header (``QGroupBox``), and ``advanced`` params are hidden unless
  ``expert_mode=True`` (the same ``chk_expert_mode`` convention every other
  tab uses, CLAUDE.md "Expert mode"). Edits call ``on_change(name, value)``
  immediately.
* :func:`build_node_inspector` -- the Circuit-specific part. Given a node
  from :mod:`sensoryforge.gui.circuit.nodes`, works out which registry and
  registered component name the node currently selects (see
  ``NODE_COMPONENT_INFO`` below), pulls that component's
  ``get_param_spec()``, and wires ``build_param_form`` to the node's own
  config object so edits land directly on the node (no separate model).

The empty-spec case (P1's stated risk): a component whose
``get_param_spec()`` returns ``[]`` -- e.g. ``MovingStimulus``
(``sensoryforge/stimuli/builder.py``), whose parameters are nested and not
yet flattened into a spec -- must not render as a blank panel indistinguishable
from "no settings". :func:`build_param_form` always renders a header naming
the component; when the spec list is empty it adds a visible notice instead
of leaving the panel blank (see ``_EMPTY_SPEC_NOTICE``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from sensoryforge.stimuli.base import ParamSpec

_EMPTY_SPEC_NOTICE = (
    "This component does not yet publish a parameter list "
    "(get_param_spec() returned nothing). Its settings cannot be rendered "
    "here; edit its config fields directly."
)


def _make_widget_for_spec(
    spec: ParamSpec,
    value: Any,
    on_change: Callable[[str, Any], None],
) -> QtWidgets.QWidget:
    """Build the one control ``spec`` maps to, wired to call ``on_change``."""
    if spec.choices is not None:
        combo = QtWidgets.QComboBox()
        for choice in spec.choices:
            combo.addItem(str(choice), choice)
        idx = combo.findData(value if value is not None else spec.default)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(
            lambda _i, c=combo, n=spec.name: on_change(n, c.currentData())
        )
        widget: QtWidgets.QWidget = combo
    elif spec.dtype == "bool":
        check = QtWidgets.QCheckBox()
        check.setChecked(bool(value if value is not None else spec.default))
        check.toggled.connect(lambda v, n=spec.name: on_change(n, v))
        widget = check
    elif spec.dtype == "int":
        spin = QtWidgets.QSpinBox()
        spin.setRange(
            int(spec.min_val) if spec.min_val is not None else -(2**31),
            int(spec.max_val) if spec.max_val is not None else 2**31 - 1,
        )
        if spec.step is not None:
            spin.setSingleStep(int(spec.step))
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.setValue(int(value if value is not None else spec.default))
        spin.valueChanged.connect(lambda v, n=spec.name: on_change(n, v))
        widget = spin
    else:  # "float" and anything else numeric-shaped
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(
            float(spec.min_val) if spec.min_val is not None else -1.0e9,
            float(spec.max_val) if spec.max_val is not None else 1.0e9,
        )
        if spec.step is not None:
            spin.setSingleStep(float(spec.step))
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.setValue(float(value if value is not None else spec.default))
        spin.valueChanged.connect(lambda v, n=spec.name: on_change(n, v))
        widget = spin

    if spec.tooltip:
        widget.setToolTip(spec.tooltip)
    elif spec.help:
        widget.setToolTip(spec.help)
    return widget


def build_param_form(
    specs: List[ParamSpec],
    values: Optional[Dict[str, Any]] = None,
    on_change: Optional[Callable[[str, Any], None]] = None,
    *,
    expert_mode: bool = False,
    title: Optional[str] = None,
) -> QtWidgets.QWidget:
    """Render a form widget from a component's ``get_param_spec()`` list.

    Args:
        specs: The component's ``ParamSpec`` list (``get_param_spec()``'s
            return value). May be empty.
        values: Current value for each spec's ``name``, if known; a missing
            or ``None`` entry falls back to the spec's ``default``.
        on_change: Called as ``on_change(name, value)`` whenever a widget's
            value changes. Optional -- omit for a read-only render.
        expert_mode: When ``False`` (the default, matching every other tab's
            ``chk_expert_mode`` unchecked state), widgets for params with
            ``advanced=True`` are built but hidden (``setVisible(False)``)
            rather than omitted, so toggling Expert mode later needs no
            rebuild.
        title: Optional header label (e.g. the registered component name)
            shown above the form.

    Returns:
        A ``QWidget`` with one row per param, grouped by ``ParamSpec.group``,
        or a visible "no parameters published" notice when ``specs`` is
        empty -- never a blank panel.
    """
    values = values or {}
    on_change = on_change or (lambda _name, _value: None)

    root = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(root)
    layout.setContentsMargins(4, 4, 4, 4)

    if title:
        header = QtWidgets.QLabel(f"<b>{title}</b>")
        layout.addWidget(header)

    if not specs:
        notice = QtWidgets.QLabel(_EMPTY_SPEC_NOTICE)
        notice.setWordWrap(True)
        notice.setObjectName("empty_param_spec_notice")
        layout.addWidget(notice)
        layout.addStretch(1)
        return root

    groups: Dict[str, List[ParamSpec]] = {}
    order: List[str] = []
    for spec in specs:
        key = spec.group or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(spec)

    widgets: Dict[str, QtWidgets.QWidget] = {}
    for group_name in order:
        if group_name:
            box = QtWidgets.QGroupBox(group_name)
            form = QtWidgets.QFormLayout(box)
            layout.addWidget(box)
        else:
            box = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(box)
            layout.addWidget(box)
        for spec in groups[group_name]:
            widget = _make_widget_for_spec(spec, values.get(spec.name), on_change)
            widget.setObjectName(f"param_{spec.name}")
            if spec.advanced and not expert_mode:
                widget.setVisible(False)
                label = QtWidgets.QLabel(spec.label)
                label.setVisible(False)
                form.addRow(label, widget)
            else:
                form.addRow(spec.label, widget)
            widgets[spec.name] = widget

    layout.addStretch(1)
    root.setProperty("param_widgets", widgets)
    return root


# ---------------------------------------------------------------------------
# Node -> (registry, selected-name accessor, params accessor) dispatch
# ---------------------------------------------------------------------------


def _grid_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    grid = node.grid
    name = grid.arrangement
    values = {
        "rows": grid.rows,
        "cols": grid.cols,
        "spacing": grid.spacing,
        "density": grid.density,
        "center_x": grid.center_x,
        "center_y": grid.center_y,
        "seed": grid.seed,
    }

    def setter(field: str, value: Any) -> None:
        if hasattr(grid, field):
            setattr(grid, field, value)

    return name, values, setter


def _stimulus_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    stim = node.stimulus
    name = stim.type
    values = dict(stim.to_dict())

    def setter(field: str, value: Any) -> None:
        if hasattr(stim, field):
            setattr(stim, field, value)

    return name, values, setter


def _rf_bank_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    rf = node.pop_input.rf
    name = rf.method
    values = dict(rf.params)

    def setter(field: str, value: Any) -> None:
        rf.params[field] = value

    return name, values, setter


def _processing_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    name = node.spec.get("method", "identity")
    values = dict(node.spec.get("params") or {})

    def setter(field: str, value: Any) -> None:
        node.spec.setdefault("params", {})[field] = value

    return name, values, setter


def _filter_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    name = node.filter_method
    values = dict(node.filter_params)

    def setter(field: str, value: Any) -> None:
        node.filter_params[field] = value

    return name, values, setter


def _readout_params(node) -> Tuple[str, Dict[str, Any], Callable[[str, Any], None]]:
    name = node.fields.get("neuron_model") or "Izhikevich"
    values = dict(node.fields.get("model_params") or {})

    def setter(field: str, value: Any) -> None:
        params = node.fields.get("model_params") or {}
        params[field] = value
        node.fields["model_params"] = params

    return name, values, setter


#: node class name -> (registry-lookup fn, params-accessor fn). The
#: registry-lookup fn is deferred (imports sensoryforge.registry lazily)
#: because registry.py pulls in the whole component tree and the Circuit
#: package should stay importable headless without it at module load time.
def _registry_for(node_type: str):
    from sensoryforge.registry import (
        FILTER_REGISTRY,
        GRID_REGISTRY,
        INNERVATION_REGISTRY,
        NEURON_REGISTRY,
        STIMULUS_REGISTRY,
    )

    return {
        "SensorArray": GRID_REGISTRY,
        "Stimulus": STIMULUS_REGISTRY,
        "RFBank": INNERVATION_REGISTRY,
        "Filter": FILTER_REGISTRY,
        "Readout": NEURON_REGISTRY,
    }.get(node_type)


_PARAM_ACCESSORS = {
    "SensorArray": _grid_params,
    "Stimulus": _stimulus_params,
    "RFBank": _rf_bank_params,
    "Processing": _processing_params,
    "Filter": _filter_params,
    "Readout": _readout_params,
}


def build_node_inspector(node, *, expert_mode: bool = False) -> QtWidgets.QWidget:
    """Build the full inspector panel for one Circuit node.

    Dispatches on the node's class name (``node.__class__.__name__``, which
    matches ``NODE_CLASSES`` keys minus the ``Node`` suffix is not assumed --
    the node's own ``nodeName`` is used) to find the registry the node
    selects a component from (P3: ``INNERVATION_REGISTRY``,
    ``FILTER_REGISTRY``, ``NEURON_REGISTRY``, ``STIMULUS_REGISTRY``,
    ``GRID_REGISTRY`` -- ``ProcessingNode`` uses ``PROCESSING_REGISTRY``
    indirectly via its own spec, and ``CombineNode``/``RecordNode`` have no
    registered component at all), pulls that component's
    ``get_param_spec()``, and renders it via :func:`build_param_form` wired
    directly to the node's own state.

    Args:
        node: A node from :mod:`sensoryforge.gui.circuit.nodes` (e.g. the
            return value of ``Flowchart.createNode`` for one of
            ``NODE_CLASSES``).
        expert_mode: Forwarded to :func:`build_param_form`.

    Returns:
        A ``QWidget``: for a node with a registered component, the rendered
        param form (or the empty-spec notice); for ``CombineNode`` and
        ``RecordNode`` (no registry-backed component), a small dedicated
        panel for their own few fields; visualisation panels are added on
        top for ``SensorArray``, ``RFBank`` and ``Stimulus`` nodes (P2).
    """
    node_type = getattr(node, "nodeName", node.__class__.__name__)

    if node_type == "Combine":
        return _build_combine_inspector(node)
    if node_type == "Record":
        return _build_record_inspector(node)

    accessor = _PARAM_ACCESSORS.get(node_type)
    if accessor is None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel(f"No inspector for node type {node_type!r}"))
        return widget

    component_name, values, setter = accessor(node)

    registry = _registry_for(node_type)
    if node_type == "Processing":
        from sensoryforge.registry import PROCESSING_REGISTRY

        registry = PROCESSING_REGISTRY

    specs: List[ParamSpec] = []
    if registry is not None and registry.is_registered(component_name):
        specs = registry.get_param_spec(component_name)

    form = build_param_form(
        specs,
        values,
        setter,
        expert_mode=expert_mode,
        title=f"{node_type}: {component_name}",
    )

    visual = _build_visualisation(node_type, node)
    if visual is None:
        return form

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(form)
    layout.addWidget(visual)
    return container


def _build_combine_inspector(node) -> QtWidgets.QWidget:
    root = QtWidgets.QWidget()
    layout = QtWidgets.QFormLayout(root)
    combo = QtWidgets.QComboBox()
    combo.addItems(["sum", "concat"])
    combo.setCurrentText(node.combine)
    combo.currentTextChanged.connect(lambda v: setattr(node, "combine", v))
    layout.addRow("Combine mode", combo)
    return root


def _build_record_inspector(node) -> QtWidgets.QWidget:
    root = QtWidgets.QWidget()
    layout = QtWidgets.QFormLayout(root)
    edit = QtWidgets.QLineEdit(node.output_dir or "")
    edit.editingFinished.connect(
        lambda e=edit: setattr(node, "output_dir", e.text() or None)
    )
    layout.addRow("Output dir", edit)
    return root


# ---------------------------------------------------------------------------
# P2 -- lightweight visualisations, reusing the same rendering primitives the
# Mechanoreceptor / Stimulus Designer tabs use rather than their (heavily
# tab-entangled, see gui/circuit/inspector.py module docstring and the Wave
# P report) shared plot widgets directly.
# ---------------------------------------------------------------------------


def _build_visualisation(node_type: str, node) -> Optional[QtWidgets.QWidget]:
    try:
        import pyqtgraph as pg
    except ImportError:  # pragma: no cover - pyqtgraph is a hard dependency
        return None

    if node_type == "SensorArray":
        return _grid_preview(node, pg)
    if node_type == "Stimulus":
        return _stimulus_preview(node, pg)
    if node_type == "RFBank":
        return _rf_bank_preview(node, pg)
    return None


def _grid_preview(node, pg) -> QtWidgets.QWidget:
    """A receptor-position scatter for a ``SensorArrayNode``'s grid.

    Reuses :class:`sensoryforge.core.grid.ReceptorGrid` (the same class the
    Mechanoreceptor tab builds its grid view from) to compute coordinates,
    rather than duplicating grid-arrangement math here.
    """
    plot_widget = pg.PlotWidget()
    plot_widget.setMaximumHeight(220)
    plot_widget.setLabel("bottom", "x", units="mm")
    plot_widget.setLabel("left", "y", units="mm")
    try:
        from sensoryforge.core.grid import ReceptorGrid

        grid = node.grid
        rg = ReceptorGrid(
            grid_size=(grid.rows or 10, grid.cols or 10),
            spacing=grid.spacing,
            arrangement=grid.arrangement,
            center=(grid.center_x, grid.center_y),
            density=grid.density,
            device="cpu",
            seed=grid.seed,
        )
        xx, yy = rg.get_coordinates()
        scatter = pg.ScatterPlotItem(
            x=xx.flatten().cpu().numpy(),
            y=yy.flatten().cpu().numpy(),
            size=4,
            brush=pg.mkBrush(80, 160, 220, 200),
        )
        plot_widget.addItem(scatter)
    except Exception as exc:  # pragma: no cover - defensive: preview is best-effort
        plot_widget.setTitle(f"Preview unavailable: {exc}")
    return plot_widget


def _stimulus_preview(node, pg) -> QtWidgets.QWidget:
    """A single-frame image preview for a ``StimulusNode``.

    Reuses :func:`sensoryforge.stimuli.render.render_stimulus`, the same
    rendering function the Stimulus Designer tab's live preview calls.
    """
    image_view = pg.PlotWidget()
    image_view.setMaximumHeight(220)
    try:
        import torch

        from sensoryforge.stimuli.render import render_stimulus

        stim = node.stimulus
        xx, yy = torch.meshgrid(
            torch.linspace(-5, 5, 40), torch.linspace(-5, 5, 40), indexing="ij"
        )
        params = {k: v for k, v in stim.to_dict().items() if k not in ("name", "type")}
        frames, _ = render_stimulus(
            stim.type, params, xx, yy, dt_ms=1.0, duration_ms=1.0, device="cpu"
        )
        frame = frames[0].detach().cpu().numpy()
        img_item = pg.ImageItem(frame)
        image_view.addItem(img_item)
    except Exception as exc:  # pragma: no cover - defensive: preview is best-effort
        image_view.setTitle(f"Preview unavailable: {exc}")
    return image_view


def _rf_bank_preview(node, pg) -> QtWidgets.QWidget:
    """A neuron-center scatter for an ``RFBankNode``'s builder selection.

    A full receptive-field weight heatmap (as the Mechanoreceptor tab draws)
    needs the upstream ``SensorArrayNode``'s receptor coordinates, which are
    only known once the graph is wired -- here we show what can be known
    from the node alone: the builder's own advertised parameters via the
    param form above this widget, plus a placeholder noting the rest.
    """
    label = QtWidgets.QLabel(
        f"RF builder: {node.pop_input.rf.method!r}. Connect to a SensorArray "
        "and run the graph to see the built receptive-field weights in the "
        "Visualization tab."
    )
    label.setWordWrap(True)
    label.setMaximumHeight(60)
    return label
