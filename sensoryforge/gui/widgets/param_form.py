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

from PyQt5 import QtGui, QtWidgets

from sensoryforge.gui.session import Session, get_by_path
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
        low = int(spec.min_val) if spec.min_val is not None else -(2**31) + 1
        high = int(spec.max_val) if spec.max_val is not None else 2**31 - 1
        if _is_optional_number(spec):
            low -= 1  # the "auto" position, one below the smallest real value
            _mark_auto(spin)
        spin.setRange(low, high)
        if spec.step is not None:
            spin.setSingleStep(int(spec.step))
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.setValue(int(resolved) if resolved is not None else spin.minimum())
        return spin, "valueChanged"

    if spec.dtype == "float":
        if _needs_scientific(spec, resolved):
            dspin = ScientificSpinBox()
        else:
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
        if isinstance(dspin, ScientificSpinBox) and spec.step is None:
            # A range of many decades has no useful fixed step.
            dspin.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        else:
            dspin.setSingleStep(step)
        if _is_optional_number(spec):
            # The "auto" position, one step below the smallest real value.
            dspin.setMinimum(min_val - (step if step > 0 else 1.0))
            _mark_auto(dspin)
        if spec.unit:
            dspin.setSuffix(f" {spec.unit}")
        dspin.setValue(float(resolved) if resolved is not None else dspin.minimum())
        return dspin, "valueChanged"

    # "str" or an unrecognised dtype.
    line = QtWidgets.QLineEdit()
    line.setText("" if resolved is None else str(resolved))
    return line, "editingFinished"


def _is_optional_number(spec: ParamSpec) -> bool:
    """A number whose unset value (``None``) means "let the component decide"."""
    return spec.dtype in ("int", "float") and spec.default is None


def _mark_auto(spin: QtWidgets.QAbstractSpinBox) -> None:
    """Show the box's minimum as "auto", read back as ``None``.

    An unset optional number (a template lattice's edge offset, which is
    pitch/2 when unset) must not be shown as 0: writing the 0 back changes
    what is built.
    """
    spin.setSpecialValueText("auto")
    spin.setProperty("auto_is_none", True)


def _is_auto(widget: QtWidgets.QWidget) -> bool:
    return bool(widget.property("auto_is_none")) and widget.value() == widget.minimum()


class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    """A float box that shows and keeps very small values (``2.5e-11``).

    ``QDoubleSpinBox`` rounds its value to its decimals, so with four decimals
    a membrane capacitance of ``1e-13`` F is displayed as ``0.0000`` and the
    first edit writes zero into the config. This box keeps 30 decimals
    internally and formats with ``%g``.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setDecimals(30)

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt name)
        return f"{value:.6g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802 (Qt name)
        body = text
        if self.suffix() and body.endswith(self.suffix()):
            body = body[: -len(self.suffix())]
        try:
            return float(body.strip())
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):  # noqa: D102
        body = text
        if self.suffix() and body.endswith(self.suffix()):
            body = body[: -len(self.suffix())]
        body = body.strip()
        try:
            float(body)
        except ValueError:
            partial = body == "" or body[-1] in "eE+-." or body in "+-"
            state = (
                QtGui.QValidator.Intermediate if partial else QtGui.QValidator.Invalid
            )
            return state, text, pos
        return QtGui.QValidator.Acceptable, text, pos


def _needs_scientific(spec: ParamSpec, value: Any) -> bool:
    """Whether four decimals would show ``spec``'s values as zero."""
    for candidate in (value, spec.default, spec.min_val, spec.max_val):
        try:
            number = abs(float(candidate))
        except (TypeError, ValueError):
            continue
        if 0.0 < number < 1.0e-3:
            return True
    return False


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
        return None if _is_auto(widget) else widget.value()
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
            if value is None:
                # "auto" for an optional number; otherwise the lowest value.
                widget.setValue(widget.minimum())
            else:
                widget.setValue(int(value) if spec.dtype == "int" else float(value))
        else:
            widget.setText("" if value is None else str(value))
    finally:
        widget.blockSignals(False)


class ParamForm(QtWidgets.QWidget):
    """A form generated from ``ParamSpec``\\ s, bound to a target through a Session.

    Args:
        specs: The component's ``get_param_spec()`` list. May be empty, in
            which case a visible notice is shown instead of a blank form.
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

        # A component that declares no ParamSpecs must not render as a blank
        # panel: blank reads as "this component has no settings", when the
        # truth is "its settings are not editable here". Say so.
        self.empty_notice: Optional[QtWidgets.QLabel] = None
        if not self._specs:
            self.empty_notice = QtWidgets.QLabel(
                "This component declares no editable parameters "
                "(its get_param_spec() is empty). Edit them in the YAML config."
            )
            self.empty_notice.setWordWrap(True)
            self.empty_notice.setObjectName("EmptyFormNotice")
            layout.addWidget(self.empty_notice)

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
        # The old config's objects are gone from the session. Rebind to the
        # object now at this form's path, or every later edit would be written
        # into the discarded config (and notified under a path that, in the
        # new one, names something else).
        try:
            target = get_by_path(self._session.config, self._path_prefix)
        except ValueError:
            self.setEnabled(False)
            return
        self._target = target
        self.setEnabled(True)
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
