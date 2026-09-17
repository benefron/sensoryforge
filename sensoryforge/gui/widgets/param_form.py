"""A form generated from ``ParamSpec`` lists, bound to a Session by path.

:class:`ParamForm` is the one place a component's ``get_param_spec()``
(CLAUDE.md, "ParamSpec and get_param_spec()") turns into editable widgets for
GUI v2. Every screen builds its parameter panels this way instead of writing
per-component widget code: point it at a list of
:class:`~sensoryforge.stimuli.base.ParamSpec`, the dataclass instance or dict
those specs describe, the :class:`~sensoryforge.gui.session.Session` that
owns the config, and the dotted path prefix the target lives at (e.g.
``"populations.0.filter_params"``), and it builds one labelled row per spec,
grouped into :class:`~sensoryforge.gui.widgets.collapsible.CollapsibleGroupBox`
sections by ``spec.group``.

Edits write straight into the target (``setattr`` for a dataclass, item
assignment for a dict) and call :meth:`~sensoryforge.gui.session.Session.notify`
with ``f"{path_prefix}.{spec.name}"`` so every other view bound to the same
config hears about it. A write from *outside* the form -- another screen's
edit, undo, a loaded config -- is picked up through
:attr:`Session.configChanged` and applied to the matching widget under
``blockSignals``, so it never re-triggers a write.

The widget-construction logic here is lifted from (not imported from --
``gui/circuit/inspector.py`` is deleted in Phase 3)
``gui/circuit/inspector.py::_make_widget_for_spec``/``build_param_form``.

F-035 discipline: every Qt signal is connected to a slot built with
``functools.partial(self._write, name)``, never a bound method or a lambda
that closes over a widget -- the widget is looked up from ``self._rows`` by
name inside ``_write``, not captured in a closure.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt5 import QtWidgets

from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.collapsible import CollapsibleGroupBox
from sensoryforge.stimuli.base import ParamSpec

#: What a ``ParamForm`` can be bound to: a dataclass instance (``setattr``)
#: or a dict (item assignment).
Target = Union[Any, Dict[str, Any]]

_MISSING = object()


class _FieldRow:
    """Bookkeeping for one spec's widget and label."""

    __slots__ = ("spec", "widget", "label", "group_box")

    def __init__(
        self,
        spec: ParamSpec,
        widget: QtWidgets.QWidget,
        label: QtWidgets.QLabel,
        group_box: Optional[CollapsibleGroupBox],
    ) -> None:
        self.spec = spec
        self.widget = widget
        self.label = label
        self.group_box = group_box


def _is_json_valued(spec: ParamSpec) -> bool:
    """Whether ``spec`` holds a list/dict value edited as JSON text."""
    return isinstance(spec.default, (list, dict))


def _current_value(target: Target, name: str, default: Any) -> Any:
    """Read ``name`` off ``target`` (dict item or attribute), or ``default``."""
    if isinstance(target, dict):
        return target.get(name, default)
    return getattr(target, name, default)


def _set_value(target: Target, name: str, value: Any) -> None:
    """Write ``value`` for ``name`` into ``target`` (dict item or attribute)."""
    if isinstance(target, dict):
        target[name] = value
    else:
        setattr(target, name, value)


def _make_widget(spec: ParamSpec, value: Any) -> Tuple[QtWidgets.QWidget, str]:
    """Build the one control ``spec`` maps to.

    Args:
        spec: The parameter descriptor.
        value: The target's current value for ``spec.name`` (``spec.default``
            when unset).

    Returns:
        ``(widget, signal_name)`` -- the widget and the name of the signal a
        row should connect to write edits back.
    """
    resolved = value if value is not None else spec.default
    if spec.choices is not None:
        combo = QtWidgets.QComboBox()
        for choice in spec.choices:
            combo.addItem(str(choice), choice)
        idx = combo.findData(resolved)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        return combo, "currentIndexChanged"

    if _is_json_valued(spec):
        line = QtWidgets.QLineEdit()
        line.setText(json.dumps(resolved))
        return line, "editingFinished"

    if spec.dtype == "bool":
        check = QtWidgets.QCheckBox()
        check.setChecked(bool(resolved))
        return check, "toggled"

    if spec.dtype == "int":
        spin = QtWidgets.QSpinBox()
        spin.setRange(
            int(spec.min_val) if spec.min_val is not None else -(2**31),
            int(spec.max_val) if spec.max_val is not None else 2**31 - 1,
        )
        if spec.step is not None:
            spin.setSingleStep(int(spec.step))
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.setValue(int(resolved) if resolved is not None else 0)
        return spin, "valueChanged"

    if spec.dtype == "float":
        dspin = QtWidgets.QDoubleSpinBox()
        dspin.setDecimals(4)
        min_val = float(spec.min_val) if spec.min_val is not None else -1.0e9
        max_val = float(spec.max_val) if spec.max_val is not None else 1.0e9
        dspin.setRange(min_val, max_val)
        if spec.step is not None:
            step = float(spec.step)
        elif spec.min_val is not None and spec.max_val is not None:
            step = (max_val - min_val) / 100.0
        else:
            step = 0.1
        dspin.setSingleStep(step)
        if spec.unit:
            dspin.setSuffix(f" {spec.unit}")
        dspin.setValue(float(resolved) if resolved is not None else 0.0)
        return dspin, "valueChanged"

    # "str" or an unrecognised dtype.
    line = QtWidgets.QLineEdit()
    line.setText("" if resolved is None else str(resolved))
    return line, "editingFinished"


def _read_widget(spec: ParamSpec, widget: QtWidgets.QWidget) -> Any:
    """Read the current value out of ``widget``.

    Raises:
        json.JSONDecodeError: If ``spec`` is JSON-valued and the widget's
            text is not valid JSON. Callers decide what to do with that.
    """
    if spec.choices is not None:
        return widget.currentData()
    if _is_json_valued(spec):
        return json.loads(widget.text())
    if spec.dtype == "bool":
        return widget.isChecked()
    if spec.dtype in ("int", "float"):
        return widget.value()
    return widget.text()


def _write_widget(spec: ParamSpec, widget: QtWidgets.QWidget, value: Any) -> None:
    """Set ``widget`` to ``value`` without re-triggering its edit signal."""
    widget.blockSignals(True)
    try:
        if spec.choices is not None:
            idx = widget.findData(value)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
        elif _is_json_valued(spec):
            widget.setText(json.dumps(value))
            widget.setObjectName(f"param_{spec.name}")
            widget.setToolTip(spec.tooltip or spec.help)
        elif spec.dtype == "bool":
            widget.setChecked(bool(value))
        elif spec.dtype in ("int", "float"):
            widget.setValue(value)
        else:
            widget.setText("" if value is None else str(value))
    finally:
        widget.blockSignals(False)


class ParamForm(QtWidgets.QWidget):
    """A form generated from ``ParamSpec``\\ s, bound to a target through a Session.

    Args:
        specs: The component's ``get_param_spec()`` list. May be empty (an
            empty form is rendered, same as an empty layout).
        target: The dataclass instance or dict the specs describe values on.
        session: The session to write through and read change notifications
            from.
        path_prefix: The dotted path ``target`` lives at, e.g.
            ``"populations.0.filter_params"``; a spec named ``"tau_r"``
            notifies at ``f"{path_prefix}.tau_r"``.
        advanced: Whether rows whose spec is ``advanced=True`` start visible
            (mirrors the tabs' Expert-mode convention, CLAUDE.md "Expert
            mode"). Toggle later with :meth:`set_advanced`.
        parent: Qt parent.
    """

    def __init__(
        self,
        specs: List[ParamSpec],
        target: Target,
        session: Session,
        path_prefix: str,
        *,
        advanced: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._specs = list(specs)
        self._target = target
        self._session = session
        self._path_prefix = path_prefix
        self._rows: Dict[str, _FieldRow] = {}
        self._group_boxes: Dict[str, CollapsibleGroupBox] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        groups: Dict[str, List[ParamSpec]] = {}
        order: List[str] = []
        for spec in self._specs:
            key = spec.group or ""
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(spec)
        if "" in order:
            order.remove("")
            order.insert(0, "")

        for group_name in order:
            if group_name:
                box: Optional[CollapsibleGroupBox] = CollapsibleGroupBox(
                    group_name, start_expanded=True
                )
                form = box.layout()
                layout.addWidget(box)
                self._group_boxes[group_name] = box
            else:
                box = None
                container = QtWidgets.QWidget()
                form = QtWidgets.QFormLayout(container)
                form.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(container)
            for spec in groups[group_name]:
                self._add_row(spec, form, box)

        layout.addStretch(1)

        self._session.configChanged.connect(self._on_config_changed)
        self._session.configReplaced.connect(self._on_config_replaced)

        self.set_advanced(advanced)

    # ------------------------------------------------------------- building

    def _add_row(
        self,
        spec: ParamSpec,
        form: QtWidgets.QFormLayout,
        group_box: Optional[CollapsibleGroupBox],
    ) -> None:
        value = _current_value(self._target, spec.name, spec.default)
        widget, signal_name = _make_widget(spec, value)
        widget.setObjectName(f"param_{spec.name}")

        label_text = spec.label or spec.name.replace("_", " ").capitalize()
        label = QtWidgets.QLabel(label_text)

        tooltip = spec.tooltip or spec.help
        if tooltip:
            widget.setToolTip(tooltip)
            label.setToolTip(tooltip)

        form.addRow(label, widget)

        slot = functools.partial(self._write, spec.name)
        getattr(widget, signal_name).connect(slot)

        self._rows[spec.name] = _FieldRow(spec, widget, label, group_box)

    # -------------------------------------------------------------- writing

    def _write(self, name: str, *_args: Any) -> None:
        """Write the current widget value for ``name`` into the target.

        Called by every field's edit signal via ``functools.partial(self._write,
        name)``; the trailing signal payload (if any) is ignored -- the value
        is always read back from the widget through :func:`_read_widget`, not
        from the signal args, so this is one code path for every widget kind.
        """
        row = self._rows[name]
        spec = row.spec
        widget = row.widget

        if _is_json_valued(spec):
            try:
                new_value = json.loads(widget.text())
            except json.JSONDecodeError as exc:
                widget.setToolTip(str(exc))
                widget.setObjectName("Invalid")
                return
            widget.setObjectName(f"param_{name}")
            widget.setToolTip(spec.tooltip or spec.help)
        else:
            new_value = _read_widget(spec, widget)

        current = _current_value(self._target, name, spec.default)
        if new_value == current:
            return
        _set_value(self._target, name, new_value)
        self._session.notify(f"{self._path_prefix}.{name}")

    # -------------------------------------------------------------- reading

    def _on_config_changed(self, path: str) -> None:
        for name, row in self._rows.items():
            if path != f"{self._path_prefix}.{name}":
                continue
            target_value = _current_value(self._target, name, row.spec.default)
            try:
                widget_value = _read_widget(row.spec, row.widget)
            except json.JSONDecodeError:
                widget_value = _MISSING
            if widget_value != target_value:
                _write_widget(row.spec, row.widget, target_value)

    def _on_config_replaced(self) -> None:
        self.refresh()

    # -------------------------------------------------------------- public

    def set_advanced(self, on: bool) -> None:
        """Show or hide rows (and groups) whose spec is ``advanced=True``.

        A group is hidden entirely when every one of its rows is advanced
        and ``on`` is ``False``; otherwise the group stays visible with only
        its advanced rows hidden.

        Args:
            on: Whether advanced rows should be visible.
        """
        for row in self._rows.values():
            visible = on or not row.spec.advanced
            row.widget.setVisible(visible)
            row.label.setVisible(visible)

        rows_by_group: Dict[CollapsibleGroupBox, List[_FieldRow]] = {}
        for row in self._rows.values():
            if row.group_box is not None:
                rows_by_group.setdefault(row.group_box, []).append(row)
        for box, rows in rows_by_group.items():
            all_advanced = all(r.spec.advanced for r in rows)
            box.setVisible(not (all_advanced and not on))

    def refresh(self) -> None:
        """Re-read every field's widget from the current target state."""
        for name, row in self._rows.items():
            target_value = _current_value(self._target, name, row.spec.default)
            _write_widget(row.spec, row.widget, target_value)

    def widget_for(self, name: str) -> QtWidgets.QWidget:
        """The widget built for the spec named ``name``.

        Raises:
            KeyError: If no spec with that name was in the form.
        """
        return self._rows[name].widget


def specs_for(registry: Any, component_name: str) -> List[ParamSpec]:
    """``registry.get_param_spec(component_name)``, matched case-insensitively.

    Args:
        registry: A :class:`~sensoryforge.registry.ComponentRegistry`
            (``FILTER_REGISTRY``, ``GRID_REGISTRY``, ...).
        component_name: The registered component name, in any case --
            registry lookups are already case-insensitive (CLAUDE.md H1,
            F-046).

    Returns:
        The component's ``ParamSpec`` list (possibly empty).
    """
    return registry.get_param_spec(component_name)
