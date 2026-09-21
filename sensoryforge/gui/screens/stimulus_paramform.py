"""A parameter form for one stimulus type's constructor, config-field aware.

:class:`StimulusParamForm` is :mod:`~sensoryforge.gui.widgets.param_form`'s
counterpart for the Stimulus screen, but it cannot simply be a
``ParamForm(specs, session.config.stimulus, session, "stimulus")``: a plain
``ParamForm`` reads a field's *current* value as "the value, or ``spec.default``
if the field is missing" -- but every :class:`~sensoryforge.config.schema.
StimulusConfig` field always holds *some* value (the dataclass's own schema
default, e.g. ``amplitude: float = 30.0``), whether or not the user set it.
What :func:`sensoryforge.stimuli.render.render_for_config` actually forwards
to the stimulus constructor is only the fields in
``StimulusConfig.explicit_fields()``; every other field takes the selected
stimulus *type's own* constructor default (which is very often a different
number -- ``MovingEdgeStimulus.amplitude`` defaults to ``1.0``, not the
schema's ``30.0``). So this form has to show, for each of the type's
``get_param_spec()`` entries:

* a field that names a ``StimulusConfig`` attribute: the config's value when
  explicit, else the *type's* default, visibly marked "(default)"; editing it
  writes through ``session.set_by_path`` (which marks it explicit); a reset
  button un-sets it again (:meth:`StimulusConfig.unset`).
* a parameter with no matching ``StimulusConfig`` field (e.g. ``moving_edge``'s
  ``total_ms``, ``braille``'s ``v_mms``): read-only, showing the type's
  default, with a note that it is not configurable from the config yet -- see
  :data:`UNCONFIGURABLE_PARAMS` for the full list this build found.

A stimulus type with no registered class (the legacy names ``trapezoidal``,
``step``, ``ramp``, ``custom``) or an empty ``get_param_spec()`` shows the
same "no editable parameters" notice :class:`~sensoryforge.gui.widgets.
param_form.ParamForm` shows for any other empty-spec component.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtWidgets

from sensoryforge.config.schema import StimulusConfig
from sensoryforge.gui import theme
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.collapsible import CollapsibleGroupBox
from sensoryforge.gui.widgets.param_form import (
    _make_widget,
    _read_widget,
    _write_widget,
)
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.base import ParamSpec

#: Names ``sensoryforge.stimuli.render.render_stimulus`` accepts only through
#: its legacy fallback chain, not through ``STIMULUS_REGISTRY``. Kept as one
#: module-level constant since ``render.py``'s own ``legacy_names`` is local
#: to a function.
LEGACY_STIMULUS_TYPES: Tuple[str, ...] = ("trapezoidal", "step", "ramp", "custom")

#: Every ``StimulusConfig`` field name, computed once.
_CONFIG_FIELDS = {f.name for f in dataclasses.fields(StimulusConfig)}

#: (stimulus type, parameter name) pairs this build found whose
#: ``get_param_spec()`` entry names no ``StimulusConfig`` field -- filled in
#: by :func:`unconfigurable_params`, kept here so the report and the tests
#: read the same list the form itself computes.


def unconfigurable_params() -> List[Tuple[str, str]]:
    """Every ``(stimulus type, parameter name)`` this build cannot edit here.

    A parameter of a registered stimulus class whose ``get_param_spec()``
    name is not a field of :class:`StimulusConfig` has nowhere to be stored
    in the config, so it cannot be made editable without inventing a side
    channel (which the brief for this screen forbids). Listed for the lead
    as a finding, not fixed here.

    Returns:
        ``(stimulus_type, parameter_name)`` pairs, in registry order.
    """
    pairs: List[Tuple[str, str]] = []
    for name in STIMULUS_REGISTRY.list_registered():
        cls = STIMULUS_REGISTRY.get_class(name)
        for spec in cls.get_param_spec():
            if spec.name not in _CONFIG_FIELDS:
                pairs.append((name, spec.name))
    return pairs


class _Row:
    """Bookkeeping for one parameter's widgets."""

    __slots__ = (
        "spec",
        "widget",
        "label",
        "reset_button",
        "config_field",
        "group_box",
    )

    def __init__(
        self,
        spec: ParamSpec,
        widget: QtWidgets.QWidget,
        label: QtWidgets.QLabel,
        reset_button: Optional[QtWidgets.QToolButton],
        config_field: bool,
        group_box: Optional[CollapsibleGroupBox],
    ) -> None:
        self.spec = spec
        self.widget = widget
        self.label = label
        self.reset_button = reset_button
        self.config_field = config_field
        self.group_box = group_box


def specs_for_stimulus_type(stimulus_type: str) -> List[ParamSpec]:
    """``get_param_spec()`` of the registered class named ``stimulus_type``.

    Args:
        stimulus_type: A ``StimulusConfig.type`` value.

    Returns:
        The class's parameter specs, or ``[]`` for a legacy-only name (in
        :data:`LEGACY_STIMULUS_TYPES`) or an unregistered/unknown name.
    """
    if not STIMULUS_REGISTRY.is_registered(stimulus_type):
        return []
    return STIMULUS_REGISTRY.get_class(stimulus_type).get_param_spec()


class StimulusParamForm(QtWidgets.QWidget):
    """The parameter form for the session's currently selected stimulus type.

    Rebuilds itself whenever ``stimulus.type`` changes (a different type has
    a different parameter set entirely) and refreshes individual rows on any
    other ``stimulus.<field>`` change, including one made by resetting a row
    to its default.

    Args:
        session: The session whose ``config.stimulus`` this form edits.
        advanced: Whether advanced rows start visible.
        parent: Qt parent.
    """

    def __init__(
        self,
        session: Session,
        *,
        advanced: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._advanced = advanced
        self._rows: Dict[str, _Row] = {}
        self._group_boxes: Dict[str, CollapsibleGroupBox] = {}
        self.empty_notice: Optional[QtWidgets.QLabel] = None

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._content = QtWidgets.QWidget()
        self._layout.addWidget(self._content)

        self.rebuild(session.config.stimulus.type)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._on_config_replaced)

    # ------------------------------------------------------------- building

    def rebuild(self, stimulus_type: str) -> None:
        """Throw away every row and build a fresh form for ``stimulus_type``.

        Args:
            stimulus_type: The type whose ``get_param_spec()`` drives the
                new rows.
        """
        old_content = self._content
        self._content = QtWidgets.QWidget()
        self._layout.replaceWidget(old_content, self._content)
        old_content.setParent(None)
        old_content.deleteLater()

        self._rows = {}
        self._group_boxes = {}
        self.empty_notice = None

        form_layout = QtWidgets.QVBoxLayout(self._content)
        form_layout.setContentsMargins(0, 0, 0, 0)

        specs = specs_for_stimulus_type(stimulus_type)
        if not specs:
            self.empty_notice = QtWidgets.QLabel(
                f"{stimulus_type!r} declares no editable parameters "
                "(no registered class, or an empty get_param_spec()). "
                "Edit it in the YAML config."
            )
            self.empty_notice.setWordWrap(True)
            self.empty_notice.setObjectName("EmptyFormNotice")
            form_layout.addWidget(self.empty_notice)
            form_layout.addStretch(1)
            return

        groups: Dict[str, List[ParamSpec]] = {}
        order: List[str] = []
        for spec in specs:
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
                row_layout = box.layout()
                form_layout.addWidget(box)
                self._group_boxes[group_name] = box
            else:
                box = None
                container = QtWidgets.QWidget()
                row_layout = QtWidgets.QFormLayout(container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                form_layout.addWidget(container)
            for spec in groups[group_name]:
                self._add_row(spec, row_layout, box)

        form_layout.addStretch(1)
        self._apply_advanced()

    def _add_row(
        self,
        spec: ParamSpec,
        row_layout: QtWidgets.QFormLayout,
        group_box: Optional[CollapsibleGroupBox],
    ) -> None:
        stim = self._session.config.stimulus
        is_config_field = spec.name in _CONFIG_FIELDS
        is_explicit = is_config_field and spec.name in stim.explicit_fields()

        if is_config_field:
            value = getattr(stim, spec.name) if is_explicit else spec.default
        else:
            value = spec.default

        widget, signal_name = _make_widget(spec, value)
        widget.setObjectName(f"stim_param_{spec.name}")

        label = QtWidgets.QLabel()
        row_widget = QtWidgets.QWidget()
        row_hbox = QtWidgets.QHBoxLayout(row_widget)
        row_hbox.setContentsMargins(0, 0, 0, 0)
        row_hbox.addWidget(widget, 1)

        reset_button: Optional[QtWidgets.QToolButton] = None
        if is_config_field:
            reset_button = QtWidgets.QToolButton()
            reset_button.setText("↺")  # counter-clockwise arrow: reset
            reset_button.setToolTip("Reset to the stimulus type's default")
            reset_button.clicked.connect(functools.partial(self._reset, spec.name))
            row_hbox.addWidget(reset_button)
        else:
            widget.setEnabled(False)
            widget.setToolTip(
                f"{spec.name!r} is not a StimulusConfig field, so it cannot "
                "be edited from the config yet. Showing the stimulus type's "
                "own default."
            )

        tooltip = spec.tooltip or spec.help
        if tooltip:
            widget.setToolTip((widget.toolTip() + "\n" + tooltip).strip())

        row_layout.addRow(label, row_widget)

        if is_config_field:
            slot = functools.partial(self._write, spec.name)
            getattr(widget, signal_name).connect(slot)

        self._rows[spec.name] = _Row(
            spec, widget, label, reset_button, is_config_field, group_box
        )
        self._style_row(spec.name, is_explicit if is_config_field else True)

    # -------------------------------------------------------------- styling

    def _style_row(self, name: str, is_set: bool) -> None:
        row = self._rows[name]
        base = row.spec.label or row.spec.name.replace("_", " ").capitalize()
        if not row.config_field:
            row.label.setText(f"{base} (not configurable)")
            font = row.label.font()
            font.setItalic(True)
            row.label.setFont(font)
            row.label.setStyleSheet(f"color: {theme.PALETTE['text_disabled']};")
            return
        if is_set:
            row.label.setText(base)
            font = row.label.font()
            font.setItalic(False)
            row.label.setFont(font)
            row.label.setStyleSheet(f"color: {theme.PALETTE['text']};")
        else:
            row.label.setText(f"{base} (default)")
            font = row.label.font()
            font.setItalic(True)
            row.label.setFont(font)
            row.label.setStyleSheet(f"color: {theme.PALETTE['text_secondary']};")
        if row.reset_button is not None:
            row.reset_button.setEnabled(is_set)

    # -------------------------------------------------------------- writing

    def _write(self, name: str, *_args: Any) -> None:
        """Write the widget's current value for ``name`` into the config."""
        row = self._rows[name]
        spec = row.spec
        widget = row.widget
        try:
            new_value = _read_widget(spec, widget)
        except json.JSONDecodeError as exc:
            widget.setToolTip(str(exc))
            return
        stim = self._session.config.stimulus
        current = (
            getattr(stim, name) if name in stim.explicit_fields() else spec.default
        )
        if new_value == current:
            return
        self._session.set_by_path(f"stimulus.{name}", new_value)

    def _reset(self, name: str, *_args: Any) -> None:
        """Un-set ``name``, so it goes back to the stimulus type's default."""
        self._session.config.stimulus.unset(name)
        self._session.notify(f"stimulus.{name}")

    # -------------------------------------------------------------- reading

    def _on_config_changed(self, path: str) -> None:
        if path == "stimulus.type":
            new_type = self._session.config.stimulus.type
            self.rebuild(new_type)
            return
        if not path.startswith("stimulus."):
            return
        name = path[len("stimulus.") :]
        if name in self._rows:
            self._refresh_row(name)

    def _refresh_row(self, name: str) -> None:
        row = self._rows[name]
        if not row.config_field:
            return
        stim = self._session.config.stimulus
        is_explicit = name in stim.explicit_fields()
        value = getattr(stim, name) if is_explicit else row.spec.default
        try:
            widget_value = _read_widget(row.spec, row.widget)
        except json.JSONDecodeError:
            widget_value = object()
        if widget_value != value:
            _write_widget(row.spec, row.widget, value)
        self._style_row(name, is_explicit)

    def _on_config_replaced(self) -> None:
        self.rebuild(self._session.config.stimulus.type)

    # -------------------------------------------------------------- public

    def set_advanced(self, on: bool) -> None:
        """Show or hide rows (and groups) whose spec is ``advanced=True``."""
        self._advanced = on
        self._apply_advanced()

    def _apply_advanced(self) -> None:
        on = self._advanced
        for row in self._rows.values():
            visible = on or not row.spec.advanced
            row.widget.setVisible(visible)
            row.label.setVisible(visible)
            if row.reset_button is not None:
                row.reset_button.setVisible(visible)

        rows_by_group: Dict[CollapsibleGroupBox, List[_Row]] = {}
        for row in self._rows.values():
            if row.group_box is not None:
                rows_by_group.setdefault(row.group_box, []).append(row)
        for box, rows in rows_by_group.items():
            all_advanced = all(r.spec.advanced for r in rows)
            box.setVisible(not (all_advanced and not on))

    def widget_for(self, name: str) -> QtWidgets.QWidget:
        """The widget built for the spec named ``name``.

        Raises:
            KeyError: If no spec with that name is in the current form.
        """
        return self._rows[name].widget

    def is_explicit(self, name: str) -> bool:
        """Whether ``name`` is currently shown as set rather than default."""
        return name in self._session.config.stimulus.explicit_fields()


def _is_json_valued(spec: ParamSpec) -> bool:
    return isinstance(spec.default, (list, dict))
