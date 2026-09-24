"""A parameter form for one stimulus type's constructor, config-field aware.

:class:`StimulusParamForm` is :mod:`~sensoryforge.gui.widgets.param_form`'s
counterpart for the Stimulus screen, but it cannot simply be a
``ParamForm(specs, session.config.stimulus, session, "stimulus")``: a plain
``ParamForm`` reads a field's *current* value as "the value, or ``spec.default``
if the field is missing" -- but every :class:`~sensoryforge.config.schema.
StimulusConfig` field always holds *some* value (the dataclass's own schema
default, e.g. ``orientation_deg: float = 0.0``), whether or not the user set it.
What :func:`sensoryforge.stimuli.render.render_for_config` actually forwards
to the stimulus constructor is only the fields in
``StimulusConfig.explicit_fields()``; every other field takes the selected
stimulus *type's own* constructor default (which is very often a different
number -- ``MovingEdgeStimulus.orientation_deg`` defaults to ``50.0``, not the
schema's ``0.0``). So this form has to show, for each of the type's
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

Every parameter in ``get_param_spec()`` is editable here, whether or not it
names a ``StimulusConfig`` field:

* a field that names a ``StimulusConfig`` attribute: the config's value when
  explicit, else :func:`sensoryforge.stimuli.render.effective_defaults`'s
  value for that name (what actually runs, not ``spec.default`` -- see the
  module-level note below), visibly marked "(default)"; editing it writes
  through ``session.set_by_path`` (which marks it explicit); a reset button
  un-sets it again (:meth:`StimulusConfig.unset`).
* a parameter with no matching ``StimulusConfig`` field (e.g. ``moving_edge``'s
  ``total_ms``, ``braille``'s ``v_mms``): stored under ``stimulus.params.<name>``
  instead -- explicit iff the key is present in ``stim.params``, else shown at
  its effective default; editing it writes through ``session.set_by_path``
  (which creates the key); a reset button pops the key again.

A parameter's displayed value is always ``effective_defaults(type)[name]``
when unset, not ``spec.default``: the two disagree for several types (a
Gaussian's sigma is 0.2 mm by ``GaussianStimulus.__init__`` but 1.0 mm is what
actually renders, via :mod:`sensoryforge.stimuli.render`'s legacy-default
compatibility layer) and showing the wrong one is exactly the "shown != used"
bug this form exists to prevent. When that effective default falls outside
``spec.min_val``/``spec.max_val``, the widget's range is widened to include it
rather than silently clamping the displayed value (a clamped display is the
same bug in a different shape).
"""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtWidgets

from sensoryforge.gui.screens.stimulus_substimuli import is_composed
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
from sensoryforge.stimuli.render import effective_defaults, takes_default_ramps

#: Names ``sensoryforge.stimuli.render.render_stimulus`` accepts only through
#: its legacy fallback chain, not through ``STIMULUS_REGISTRY``. Kept as one
#: module-level constant since ``render.py``'s own ``legacy_names`` is local
#: to a function.
LEGACY_STIMULUS_TYPES: Tuple[str, ...] = ("trapezoidal", "step", "ramp", "custom")

#: The ramp-hold-ramp envelope of a still stimulus. ``None`` shows as auto:
#: each ramp an eighth of the run, the hold the rest.
ENVELOPE_SPECS = [
    ParamSpec(
        "ramp_up_ms",
        dtype="float",
        default=None,
        min_val=0.0,
        max_val=1.0e6,
        unit="ms",
        group="Timing",
        tooltip="Rise time. Auto: an eighth of the run. 0 switches it on as a step.",
    ),
    ParamSpec(
        "plateau_ms",
        dtype="float",
        default=None,
        min_val=0.0,
        max_val=1.0e6,
        unit="ms",
        group="Timing",
        tooltip="Hold time at full amplitude. Auto: whatever the ramps leave.",
    ),
    ParamSpec(
        "ramp_down_ms",
        dtype="float",
        default=None,
        min_val=0.0,
        max_val=1.0e6,
        unit="ms",
        group="Timing",
        tooltip="Fall time. Auto: an eighth of the run.",
    ),
]

#: Stimulus parameters the run owns, never shown in the stimulus form.
RUN_OWNED_PARAMS = frozenset({"dt_ms"})

#: Stimulus type -> parameter names that a dedicated editor owns instead of
#: this form (``layered``'s ``combine`` is the :class:`~sensoryforge.gui.
#: screens.stimulus_layers.LayerEditor`'s combo, not a row here).
_TYPE_OWNED_PARAMS: Dict[str, frozenset] = {"layered": frozenset({"combine"})}

#: Every ``StimulusConfig`` field name, computed once.
_CONFIG_FIELDS = {f.name for f in dataclasses.fields(StimulusConfig)}

#: Prefix used for a parameter with no ``StimulusConfig`` field of its own;
#: it lives at ``stimulus.params.<name>`` instead (``StimulusConfig.params``).
_PARAMS_PATH_PREFIX = "stimulus.params."


def unconfigurable_params() -> List[Tuple[str, str]]:
    """Every ``(stimulus type, parameter name)`` this form cannot edit.

    Historically, a parameter of a registered stimulus class whose
    ``get_param_spec()`` name was not a field of :class:`StimulusConfig` had
    nowhere to be stored and was shown read-only. Since ``StimulusConfig.params``
    (a catch-all dict for exactly these parameters) was added, every declared
    parameter of every registered stimulus type is editable here -- this
    always returns ``[]`` now. Kept (rather than deleted outright) as the one
    place a future audit can re-check that claim by calling it.

    Returns:
        ``[]``.
    """
    return []


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


def _effective_default(stimulus_type: str, spec: ParamSpec) -> Any:
    """The value ``spec.name`` takes when unset, for the currently selected type.

    This is :func:`sensoryforge.stimuli.render.effective_defaults`'s answer,
    not ``spec.default`` -- see the module docstring's "shown != used" note.

    Args:
        stimulus_type: A ``StimulusConfig.type`` value.
        spec: The parameter descriptor.

    Returns:
        ``effective_defaults(stimulus_type)[spec.name]``, or ``spec.default``
        when ``stimulus_type`` is not registered or does not declare that key.
    """
    if not STIMULUS_REGISTRY.is_registered(stimulus_type):
        return spec.default
    defaults = effective_defaults(stimulus_type)
    return defaults[spec.name] if spec.name in defaults else spec.default


def _widen_to_include(spec: ParamSpec, *values: Any) -> ParamSpec:
    """A copy of *spec* whose numeric range is widened to include *values*.

    ``_make_widget`` clamps its initial value to ``spec.min_val``/``max_val``
    on construction, so a spec whose effective default (or current explicit
    value) falls outside its own declared range would otherwise display a
    silently clamped number -- the same "shown != used" bug this form exists
    to prevent, just moved into the widget instead of the label. Only int/float,
    non-enum specs have a numeric range to widen.

    Args:
        spec: The parameter descriptor.
        *values: Every value this row might need to display.

    Returns:
        *spec* unchanged if no value falls outside its range, else a new
        :class:`ParamSpec` with ``min_val``/``max_val`` widened just enough.
    """
    if spec.dtype not in ("int", "float") or spec.choices is not None:
        return spec
    min_val = spec.min_val
    max_val = spec.max_val
    changed = False
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if min_val is not None and numeric < min_val:
            min_val = numeric
            changed = True
        if max_val is not None and numeric > max_val:
            max_val = numeric
            changed = True
    if not changed:
        return spec
    data = spec.to_dict()
    data["min_val"] = min_val
    data["max_val"] = max_val
    return ParamSpec(**data)


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
    # A stimulus's own dt_ms always follows the run's step (the renderer sets
    # it; any other value would play the stimulus at the wrong speed), so it
    # is set on the run bar, not here.
    owned = _TYPE_OWNED_PARAMS.get(stimulus_type, frozenset())
    specs = [
        spec
        for spec in STIMULUS_REGISTRY.get_class(stimulus_type).get_param_spec()
        if spec.name not in RUN_OWNED_PARAMS and spec.name not in owned
    ]
    if takes_default_ramps(stimulus_type):
        # A still image is ramped in and out (render.default_envelope); its
        # envelope is the StimulusConfig ramp/plateau fields, auto until set.
        specs += ENVELOPE_SPECS
    return specs


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
        old_content.hide()  # hide before detaching (see populations_cards)
        old_content.setParent(None)
        old_content.deleteLater()

        self._rows = {}
        self._group_boxes = {}
        self.empty_notice = None

        form_layout = QtWidgets.QVBoxLayout(self._content)
        form_layout.setContentsMargins(0, 0, 0, 0)

        specs = specs_for_stimulus_type(stimulus_type)
        if not specs and not is_composed(stimulus_type) and stimulus_type != "layered":
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
        if is_config_field:
            is_explicit = spec.name in stim.explicit_fields()
        else:
            is_explicit = spec.name in stim.params

        default_value = _effective_default(stim.type, spec)
        if is_config_field:
            value = getattr(stim, spec.name) if is_explicit else default_value
        else:
            value = stim.params[spec.name] if is_explicit else default_value

        widget_spec = _widen_to_include(spec, default_value, value)
        if default_value is None and spec.dtype in ("int", "float"):
            # Follows the run when unset (effective_defaults gives None): the
            # widget must offer "auto", not the class's fixed default.
            widget_spec = ParamSpec(**{**widget_spec.to_dict(), "default": None})

        widget, signal_name = _make_widget(widget_spec, value)
        widget.setObjectName(f"stim_param_{spec.name}")

        label = QtWidgets.QLabel()
        row_widget = QtWidgets.QWidget()
        row_hbox = QtWidgets.QHBoxLayout(row_widget)
        row_hbox.setContentsMargins(0, 0, 0, 0)
        row_hbox.addWidget(widget, 1)

        reset_button = QtWidgets.QToolButton()
        reset_button.setText("↺")  # counter-clockwise arrow: reset
        reset_button.setToolTip("Reset to the stimulus type's default")
        reset_button.clicked.connect(functools.partial(self._reset, spec.name))
        row_hbox.addWidget(reset_button)

        tooltip = spec.tooltip or spec.help
        if tooltip:
            widget.setToolTip((widget.toolTip() + "\n" + tooltip).strip())

        row_layout.addRow(label, row_widget)

        if is_config_field:
            slot = functools.partial(self._write, spec.name)
        else:
            slot = functools.partial(self._write_param, spec.name)
        getattr(widget, signal_name).connect(slot)

        self._rows[spec.name] = _Row(
            widget_spec, widget, label, reset_button, is_config_field, group_box
        )
        self._style_row(spec.name, is_explicit)

    # -------------------------------------------------------------- styling

    def _style_row(self, name: str, is_set: bool) -> None:
        row = self._rows[name]
        base = row.spec.label or row.spec.name.replace("_", " ").capitalize()
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
        """Write the widget's current value for a ``StimulusConfig`` field."""
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
            getattr(stim, name)
            if name in stim.explicit_fields()
            else _effective_default(stim.type, spec)
        )
        if new_value == current:
            return
        self._session.set_by_path(f"stimulus.{name}", new_value)

    def _write_param(self, name: str, *_args: Any) -> None:
        """Write the widget's current value into ``stimulus.params[name]``."""
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
            stim.params[name]
            if name in stim.params
            else _effective_default(stim.type, spec)
        )
        if new_value == current:
            return
        self._session.set_by_path(f"{_PARAMS_PATH_PREFIX}{name}", new_value)

    def _reset(self, name: str, *_args: Any) -> None:
        """Un-set ``name``, so it goes back to the stimulus type's default."""
        row = self._rows[name]
        if row.config_field:
            self._session.config.stimulus.unset(name)
            self._session.notify(f"stimulus.{name}")
        else:
            self._session.config.stimulus.params.pop(name, None)
            self._session.notify(f"{_PARAMS_PATH_PREFIX}{name}")

    # -------------------------------------------------------------- reading

    def _on_config_changed(self, path: str) -> None:
        if path == "stimulus.type":
            new_type = self._session.config.stimulus.type
            self.rebuild(new_type)
            return
        if not path.startswith("stimulus."):
            return
        name = path[len("stimulus.") :]
        if name.startswith("params."):
            name = name[len("params.") :]
        if name in self._rows:
            self._refresh_row(name)

    def _refresh_row(self, name: str) -> None:
        row = self._rows[name]
        stim = self._session.config.stimulus
        default_value = _effective_default(stim.type, row.spec)
        if row.config_field:
            is_explicit = name in stim.explicit_fields()
            value = getattr(stim, name) if is_explicit else default_value
        else:
            is_explicit = name in stim.params
            value = stim.params[name] if is_explicit else default_value
        row.spec = _widen_to_include(row.spec, value)
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
        row = self._rows.get(name)
        stim = self._session.config.stimulus
        if row is not None and not row.config_field:
            return name in stim.params
        return name in stim.explicit_fields()


def _is_json_valued(spec: ParamSpec) -> bool:
    return isinstance(spec.default, (list, dict))
