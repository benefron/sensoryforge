"""The five cards of the Populations screen's centre column (Task 2.3).

Each card is a small :class:`QtWidgets.QGroupBox` bound to the *currently
selected* population; :class:`PopulationsScreen` calls :meth:`_Card.set_population`
whenever the selection changes. A card rebuilds its contents (not just its
values) when a *structural* field of its own population changes -- which
filter method is picked, how many inputs there are, which neuron model --
since that changes which widgets exist at all; non-structural fields (a
filter's ``tau_r``, a gain) are kept live by the :class:`ParamForm`\\ s a
card builds, which already listen to
:attr:`~sensoryforge.gui.session.Session.configChanged` on their own.

Multi-input population editing (I1): a population with no ``inputs`` list is
in *sugar* mode -- ``target_grid``/``innervation_method``/``sigma_d_mm``/...
are edited directly, ``channel`` is always ``"value"`` and ``gain`` is
always ``1.0`` (the schema's sugar fields do not cover either). ``+ input``
converts it to an *explicit* two-entry ``inputs`` list, resetting the sugar
fields to their defaults first (:class:`~sensoryforge.config.schema.PopulationConfig`
raises if both are set away from default at construction time -- resetting
them keeps a later YAML round trip clean, see ``config/schema.py``). Removing
inputs back down to one leaves ``inputs`` as a single explicit entry rather
than clearing it back to ``[]``; ``PopulationConfig.to_dict()`` already folds
a single sugar-shaped entry back into the short form, so the written YAML is
identical either way.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.config.defaults import resolve_neuron_params
from sensoryforge.config.schema import (
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
)
from sensoryforge.gui.bench import find_population, population_index
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.param_form import ParamForm, specs_for
from sensoryforge.registry import FILTER_REGISTRY, INNERVATION_REGISTRY, NEURON_REGISTRY
from sensoryforge.stimuli.base import ParamSpec

#: Curated filter choices (registry aliases "safilter"/"rafilter"/"identity"
#: collapsed to their one canonical spelling each, plus the no-op).
FILTER_CHOICES = ["none", "SA", "RA"]

#: Curated neuron model choices (registry alias "dsl" collapsed to its GUI
#: spelling "DSL (Custom)").
NEURON_MODEL_CHOICES = ["Izhikevich", "AdEx", "MQIF", "FA", "SA", "DSL (Custom)"]

#: Neuron-layout arrangement choices, same registry grid/hex/... names used
#: for receptor grids (create_neuron_centers accepts the same vocabulary).
ARRANGEMENT_CHOICES = ["grid", "poisson", "hex", "jittered_grid", "blue_noise"]


def _is_dsl_model(name: str) -> bool:
    return (name or "").strip().casefold() in ("dsl", "dsl (custom)")


def _specs_with_defaults(
    specs: List[ParamSpec], overrides: Dict[str, Any]
) -> List[ParamSpec]:
    """Copy ``specs``, replacing each named spec's ``default`` from ``overrides``.

    Used so a form shows the value the engine will actually resolve for an
    unset key (CLAUDE.md's resolver rule) instead of the component's own
    ``ParamSpec.default``, without writing anything into the target dict
    just by displaying it.
    """
    result = []
    for spec in specs:
        if spec.name in overrides:
            kwargs = spec.to_dict()
            kwargs["default"] = overrides[spec.name]
            result.append(ParamSpec(**kwargs))
        else:
            result.append(spec)
    return result


def _set_combo(
    combo: QtWidgets.QComboBox, items: List[str], current: Optional[str]
) -> None:
    """Populate ``combo`` and select ``current`` -- before any signal is connected."""
    combo.addItems(items)
    index = combo.findText(current or "", QtCore.Qt.MatchFixedString)
    combo.setCurrentIndex(index if index >= 0 else -1)


#: Neuron parameters the engine always sets itself (``build_neuron``): the
#: integration step is ``simulation.integrate_dt_ms`` and the noise is the
#: population's ``noise_std``. A form row for them would be ignored.
RUN_OWNED_NEURON_PARAMS = frozenset({"dt", "noise_std"})


class _Card(QtWidgets.QGroupBox):
    """Shared plumbing: track the selected population, rebuild on structural change."""

    def __init__(
        self, title: str, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(title, parent)
        self._session = session
        self._population_name: Optional[str] = None
        self._outer = QtWidgets.QVBoxLayout(self)
        self._content: Optional[QtWidgets.QWidget] = None
        self._advanced = False
        # True while this card writes one of its own fields; the card already
        # shows the value, and rebuilding would delete the widget being typed
        # into (typing "125" kept only the "1").
        self._writing = False
        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._on_config_replaced)

    def set_population(self, name: Optional[str]) -> None:
        """Point this card at a different population and rebuild."""
        self._population_name = name
        self._rebuild()

    def _pop_cfg(self) -> Optional[PopulationConfig]:
        return find_population(self._session.config, self._population_name)

    def _pop_index(self) -> Optional[int]:
        if self._population_name is None:
            return None
        try:
            return population_index(self._session.config, self._population_name)
        except ValueError:
            return None

    def _set_content(self, widget: QtWidgets.QWidget) -> None:
        if self._content is not None:
            self._outer.removeWidget(self._content)
            # Hidden first, then detached: detaching a visible widget makes it
            # a top-level window until Qt deletes it (seen as leftover
            # windows in tests); detached, it leaves this card's children.
            self._content.hide()
            self._content.setParent(None)
            self._content.deleteLater()
        self._content = widget
        self._outer.addWidget(widget)
        for form in widget.findChildren(ParamForm):
            form.set_advanced(self._advanced)

    def set_advanced(self, on: bool) -> None:
        """Show or hide advanced parameter rows (the toolbar's Advanced)."""
        self._advanced = bool(on)
        if self._content is not None:
            for form in self._content.findChildren(ParamForm):
                form.set_advanced(self._advanced)

    def _write(self, path: str, value) -> None:
        """Write one of this card's own fields without rebuilding the card."""
        self._writing = True
        try:
            self._session.set_by_path(path, value)
        finally:
            self._writing = False

    def _on_config_replaced(self) -> None:
        self._rebuild()

    #: Whether this card shows the grids (its choices go stale otherwise).
    _SHOWS_GRIDS = False

    def _on_config_changed(self, path: str) -> None:
        if self._writing:
            return
        if self._SHOWS_GRIDS and (path == "grids" or path.startswith("grids.")):
            self._rebuild()
            return
        index = self._pop_index()
        if index is None:
            return
        prefix = f"populations.{index}."
        if not path.startswith(prefix):
            return
        rel = path[len(prefix) :]
        if self._is_structural(rel):
            self._rebuild()

    def _is_structural(self, rel_path: str) -> bool:  # pragma: no cover - overridden
        return False

    def _rebuild(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class FilterCard(_Card):
    """Filter method + its resolved parameter form."""

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Filter", session, parent)

    def _is_structural(self, rel_path: str) -> bool:
        return rel_path == "filter_method"

    def _rebuild(self) -> None:
        pop_cfg = self._pop_cfg()
        index = self._pop_index()
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        if pop_cfg is None or index is None:
            self._set_content(content)
            return

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Method:"))
        combo = QtWidgets.QComboBox()
        current = pop_cfg.filter_method or "none"
        _set_combo(combo, FILTER_CHOICES, current)
        combo.currentTextChanged.connect(functools.partial(self._on_method, index))
        row.addWidget(combo)
        row.addStretch(1)
        layout.addLayout(row)

        if current.strip().casefold() not in ("none", "identity", ""):
            specs = specs_for(FILTER_REGISTRY, current)
            form = ParamForm(
                specs,
                pop_cfg.filter_params,
                self._session,
                f"populations.{index}.filter_params",
            )
            layout.addWidget(form)

        self._set_content(content)

    def _on_method(self, index: int, text: str) -> None:
        if not text:
            return
        pop_cfg = self._session.config.populations[index]
        if text == pop_cfg.filter_method:
            return
        # The old filter's parameters mean nothing to the new one (SA tau_r
        # given to RA is a build error, and the RA form cannot show it).
        if pop_cfg.filter_params:
            self._session.set_by_path(f"populations.{index}.filter_params", {})
        self._session.set_by_path(f"populations.{index}.filter_method", text)


class NeuronCard(_Card):
    """Neuron model + its resolved parameter form, or the DSL editor entry point."""

    #: Emitted with the population name when "Edit equations..." is clicked.
    dslEditRequested = QtCore.pyqtSignal(str)

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Neuron", session, parent)

    def _is_structural(self, rel_path: str) -> bool:
        return rel_path in ("neuron_model", "neuron_type", "dsl_config")

    def _rebuild(self) -> None:
        pop_cfg = self._pop_cfg()
        index = self._pop_index()
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        if pop_cfg is None or index is None:
            self._set_content(content)
            return

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Model:"))
        combo = QtWidgets.QComboBox()
        current = pop_cfg.neuron_model or "Izhikevich"
        _set_combo(combo, NEURON_MODEL_CHOICES, current)
        combo.currentTextChanged.connect(functools.partial(self._on_model, index))
        row.addWidget(combo)
        row.addStretch(1)
        layout.addLayout(row)

        if _is_dsl_model(current):
            if not pop_cfg.dsl_config:
                badge_text = "no equations yet"
            else:
                has_threshold = bool(pop_cfg.dsl_config.get("threshold"))
                badge_text = "spiking" if has_threshold else "analog"
            badge = QtWidgets.QLabel(f"Readout: {badge_text}")
            badge.setObjectName("SectionTitle")
            layout.addWidget(badge)
            edit_btn = QtWidgets.QPushButton("Edit equations…")
            edit_btn.clicked.connect(
                functools.partial(self.dslEditRequested.emit, self._population_name)
            )
            layout.addWidget(edit_btn)
        else:
            specs = [
                spec
                for spec in specs_for(NEURON_REGISTRY, current)
                if spec.name not in RUN_OWNED_NEURON_PARAMS
            ]
            resolved = resolve_neuron_params(current, pop_cfg.neuron_type, {})
            specs = _specs_with_defaults(specs, resolved)
            form = ParamForm(
                specs,
                pop_cfg.model_params,
                self._session,
                f"populations.{index}.model_params",
            )
            layout.addWidget(form)

        self._set_content(content)

    def _on_model(self, index: int, text: str) -> None:
        if not text:
            return
        pop_cfg = self._session.config.populations[index]
        if text == pop_cfg.neuron_model:
            return
        # Same-named parameters differ between models (Izhikevich a = 0.02 is
        # not AdEx a = 2.0 nS), so the old model's values are not carried over.
        if pop_cfg.model_params:
            self._session.set_by_path(f"populations.{index}.model_params", {})
        self._session.set_by_path(f"populations.{index}.neuron_model", text)


class ReadoutCard(_Card):
    """Input gain, noise, and DSL readout mode."""

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Readout & noise", session, parent)

    def _is_structural(self, rel_path: str) -> bool:
        # Its own fields, changed from elsewhere (a sweep preview, a reload);
        # its own edits go through _write and do not rebuild.
        return rel_path.split(".")[0] in (
            "input_gain",
            "noise_std",
            "noise_seed",
            "readout",
            "neuron_model",
            "dsl_config",
        )

    def _rebuild(self) -> None:
        pop_cfg = self._pop_cfg()
        index = self._pop_index()
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        if pop_cfg is None or index is None:
            self._set_content(content)
            return

        gain_spin = QtWidgets.QDoubleSpinBox()
        gain_spin.setRange(0.0, 10000.0)
        gain_spin.setDecimals(2)
        gain_spin.setValue(pop_cfg.input_gain)
        gain_spin.valueChanged.connect(
            functools.partial(self._on_float, f"populations.{index}.input_gain")
        )
        form.addRow("Input gain:", gain_spin)

        noise_spin = QtWidgets.QDoubleSpinBox()
        noise_spin.setRange(0.0, 1000.0)
        noise_spin.setDecimals(3)
        noise_spin.setValue(pop_cfg.noise_std)
        noise_spin.valueChanged.connect(
            functools.partial(self._on_float, f"populations.{index}.noise_std")
        )
        form.addRow("Noise std:", noise_spin)

        seed_spin = QtWidgets.QSpinBox()
        seed_spin.setRange(-1, 2**31 - 1)
        seed_spin.setSpecialValueText("auto")
        seed_spin.setValue(pop_cfg.noise_seed if pop_cfg.noise_seed is not None else -1)
        seed_spin.valueChanged.connect(
            functools.partial(self._on_optional_int, f"populations.{index}.noise_seed")
        )
        form.addRow("Noise seed:", seed_spin)

        readout_combo = QtWidgets.QComboBox()
        _set_combo(
            readout_combo, ["auto", "spiking", "analog"], pop_cfg.readout or "auto"
        )
        readout_combo.currentTextChanged.connect(
            functools.partial(self._on_text, f"populations.{index}.readout")
        )
        form.addRow("Readout:", readout_combo)

        self._set_content(content)

    def _on_float(self, path: str, value: float) -> None:
        self._write(path, value)

    def _on_optional_int(self, path: str, value: int) -> None:
        self._write(path, None if value < 0 else value)

    def _on_text(self, path: str, text: str) -> None:
        if not text:
            return
        self._write(path, text)


class NeuronLayoutCard(_Card):
    """Neuron lattice layout, disabled when the RF builder derives its own."""

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Neuron layout", session, parent)

    def _is_structural(self, rel_path: str) -> bool:
        return rel_path.split(".")[0] in (
            "neurons_per_row",
            "neuron_rows",
            "neuron_cols",
            "neuron_arrangement",
            "seed",
            "innervation_method",
            "inputs",
        )

    def _rebuild(self) -> None:
        pop_cfg = self._pop_cfg()
        index = self._pop_index()
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        if pop_cfg is None or index is None:
            self._set_content(content)
            return

        derives = False
        for pop_input in pop_cfg.effective_inputs():
            try:
                builder_cls = INNERVATION_REGISTRY.get_class(pop_input.rf.method)
            except KeyError:
                continue
            if builder_cls.DERIVES_NEURON_CENTERS:
                derives = True
                break

        if derives:
            note = QtWidgets.QLabel(
                "This population's receptive-field builder derives its own "
                "neuron lattice; these fields are ignored. See the RF "
                "footprint bench for the derived neuron count."
            )
            note.setWordWrap(True)
            note.setToolTip(
                "innervation_method derives its own neuron centres "
                "(DERIVES_NEURON_CENTERS)"
            )
            form.addRow(note)

        per_row = QtWidgets.QSpinBox()
        per_row.setRange(1, 2000)
        per_row.setValue(pop_cfg.neurons_per_row)
        per_row.setEnabled(not derives)
        per_row.valueChanged.connect(
            functools.partial(self._on_int, f"populations.{index}.neurons_per_row")
        )
        form.addRow("Neurons per row:", per_row)

        rows_spin = QtWidgets.QSpinBox()
        rows_spin.setRange(-1, 2000)
        rows_spin.setSpecialValueText("auto")
        rows_spin.setValue(
            pop_cfg.neuron_rows if pop_cfg.neuron_rows is not None else -1
        )
        rows_spin.setEnabled(not derives)
        rows_spin.valueChanged.connect(
            functools.partial(self._on_optional_int, f"populations.{index}.neuron_rows")
        )
        form.addRow("Rows:", rows_spin)

        cols_spin = QtWidgets.QSpinBox()
        cols_spin.setRange(-1, 2000)
        cols_spin.setSpecialValueText("auto")
        cols_spin.setValue(
            pop_cfg.neuron_cols if pop_cfg.neuron_cols is not None else -1
        )
        cols_spin.setEnabled(not derives)
        cols_spin.valueChanged.connect(
            functools.partial(self._on_optional_int, f"populations.{index}.neuron_cols")
        )
        form.addRow("Cols:", cols_spin)

        arrangement_combo = QtWidgets.QComboBox()
        _set_combo(
            arrangement_combo, ARRANGEMENT_CHOICES, pop_cfg.neuron_arrangement or "grid"
        )
        arrangement_combo.setEnabled(not derives)
        arrangement_combo.currentTextChanged.connect(
            functools.partial(self._on_text, f"populations.{index}.neuron_arrangement")
        )
        form.addRow("Arrangement:", arrangement_combo)

        seed_spin = QtWidgets.QSpinBox()
        seed_spin.setRange(-1, 2**31 - 1)
        seed_spin.setSpecialValueText("auto")
        seed_spin.setValue(pop_cfg.seed if pop_cfg.seed is not None else -1)
        seed_spin.setEnabled(not derives)
        seed_spin.valueChanged.connect(
            functools.partial(self._on_optional_int, f"populations.{index}.seed")
        )
        form.addRow("Seed:", seed_spin)

        self._set_content(content)

    def _on_int(self, path: str, value: int) -> None:
        self._write(path, value)

    def _on_optional_int(self, path: str, value: int) -> None:
        self._write(path, None if value < 0 else value)

    def _on_text(self, path: str, text: str) -> None:
        if not text:
            return
        self._write(path, text)


class InputsCard(_Card):
    """One row per input; "+ input" / "Remove" convert between sugar and explicit."""

    _SHOWS_GRIDS = True

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Inputs", session, parent)

    def _is_structural(self, rel_path: str) -> bool:
        if rel_path in (
            "inputs",
            "combine",
            "target_grid",
            "innervation_method",
            "target_layers",
        ):
            return True
        parts = rel_path.split(".")
        if parts[0] == "inputs" and len(parts) >= 3 and parts[2] in ("grid", "channel"):
            return True
        if parts[0] == "inputs" and len(parts) >= 4 and parts[2:4] == ["rf", "method"]:
            return True
        return False

    def _rebuild(self) -> None:
        pop_cfg = self._pop_cfg()
        index = self._pop_index()
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        if pop_cfg is None or index is None:
            self._set_content(content)
            return

        explicit = bool(pop_cfg.inputs)
        inputs = pop_cfg.effective_inputs()
        for j, pop_input in enumerate(inputs):
            layout.addWidget(
                self._build_row(pop_cfg, index, j, pop_input, explicit, len(inputs))
            )

        add_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ input")
        add_btn.clicked.connect(functools.partial(self._on_add_input, index))
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        if explicit and len(inputs) > 1:
            combine_row = QtWidgets.QHBoxLayout()
            combine_row.addWidget(QtWidgets.QLabel("Combine:"))
            combine_combo = QtWidgets.QComboBox()
            _set_combo(combine_combo, ["sum", "concat"], pop_cfg.combine)
            combine_combo.currentTextChanged.connect(
                functools.partial(self._on_text, f"populations.{index}.combine")
            )
            combine_row.addWidget(combine_combo)
            combine_row.addStretch(1)
            layout.addLayout(combine_row)

        self._set_content(content)

    def _build_row(
        self,
        pop_cfg: PopulationConfig,
        index: int,
        j: int,
        pop_input: PopulationInput,
        explicit: bool,
        n_inputs: int,
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(f"Input {j + 1}")
        form = QtWidgets.QFormLayout(box)

        grid_names = [g.name for g in self._session.config.grids]
        grid_combo = QtWidgets.QComboBox()
        _set_combo(grid_combo, grid_names, pop_input.grid)
        grid_path = (
            f"populations.{index}.inputs.{j}.grid"
            if explicit
            else f"populations.{index}.target_grid"
        )
        grid_combo.currentTextChanged.connect(
            functools.partial(self._on_text, grid_path)
        )
        form.addRow("Grid:", grid_combo)

        if explicit:
            channels = ["value"]
            for grid_cfg in self._session.config.grids:
                if grid_cfg.name == pop_input.grid:
                    channels = list(grid_cfg.channels)
                    break
            channel_combo = QtWidgets.QComboBox()
            _set_combo(channel_combo, channels, pop_input.channel)
            channel_combo.currentTextChanged.connect(
                functools.partial(
                    self._on_text, f"populations.{index}.inputs.{j}.channel"
                )
            )
            form.addRow("Channel:", channel_combo)

        method_choices = sorted(INNERVATION_REGISTRY.list_registered())
        method_combo = QtWidgets.QComboBox()
        _set_combo(method_combo, method_choices, pop_input.rf.method)
        method_path = (
            f"populations.{index}.inputs.{j}.rf.method"
            if explicit
            else f"populations.{index}.innervation_method"
        )
        method_combo.currentTextChanged.connect(
            functools.partial(self._on_text, method_path)
        )
        form.addRow("RF builder:", method_combo)

        if explicit:
            gain_spin = QtWidgets.QDoubleSpinBox()
            gain_spin.setRange(-1000.0, 1000.0)
            gain_spin.setDecimals(3)
            gain_spin.setValue(pop_input.gain)
            gain_spin.valueChanged.connect(
                functools.partial(
                    self._on_float, f"populations.{index}.inputs.{j}.gain"
                )
            )
            form.addRow("Gain:", gain_spin)

        try:
            builder_cls = INNERVATION_REGISTRY.get_class(pop_input.rf.method)
        except KeyError:
            builder_cls = None
        if builder_cls is not None and builder_cls.DERIVES_NEURON_CENTERS:
            note = QtWidgets.QLabel("This builder derives its own neuron lattice.")
            note.setWordWrap(True)
            form.addRow(note)

        if builder_cls is not None:
            specs = specs_for(INNERVATION_REGISTRY, pop_input.rf.method)
            # Show, for every key not set here, the value the engine will
            # build with: the population-wide fields and innervation_params
            # (SimulationEngine.builder_params), then this input's own params.
            effective = SimulationEngine.builder_params(pop_cfg)
            if explicit:
                effective.update(pop_input.rf.params)
            # A tuple (weight_range) is edited as a JSON list.
            effective = {
                k: list(v) if isinstance(v, tuple) else v for k, v in effective.items()
            }
            specs = _specs_with_defaults(specs, effective)
            target = pop_input.rf.params if explicit else pop_cfg.innervation_params
            param_prefix = (
                f"populations.{index}.inputs.{j}.rf.params"
                if explicit
                else f"populations.{index}.innervation_params"
            )
            param_form = ParamForm(specs, target, self._session, param_prefix)
            form.addRow(param_form)

        if explicit and n_inputs > 1:
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.clicked.connect(
                functools.partial(self._on_remove_input, index, j)
            )
            form.addRow(remove_btn)

        return box

    def _on_text(self, path: str, text: str) -> None:
        if not text:
            return
        self._session.set_by_path(path, text)

    def _on_float(self, path: str, value: float) -> None:
        # An input's gain: no rebuild, so typing is not interrupted.
        self._write(path, value)

    def _on_add_input(self, index: int) -> None:
        pop_cfg = self._session.config.populations[index]
        prefix = f"populations.{index}"
        grid_names = [g.name for g in self._session.config.grids]

        if not pop_cfg.inputs:
            first = pop_cfg.effective_inputs()[0]
            second_grid = first.grid or (grid_names[0] if grid_names else None)
            # sigma_d_mm/connections_per_neuron/use_distance_weights/
            # resolvable_distance_mm and innervation_params are population-
            # wide sugar fields today (SimulationEngine.builder_params()
            # reads them straight off pop_cfg for every input); switching to
            # explicit mode resets them to their schema defaults below, so
            # any value the population actually had must move into the
            # first input's own rf.params first, or it is silently lost
            # (e.g. a "template" builder's resolvable_distance_mm).
            extra_params: Dict[str, Any] = dict(pop_cfg.innervation_params)
            for key, default in (
                ("sigma_d_mm", 0.3),
                ("connections_per_neuron", 28),
                ("use_distance_weights", True),
                ("resolvable_distance_mm", None),
            ):
                value = getattr(pop_cfg, key)
                if value != default and key not in extra_params:
                    extra_params[key] = value
            new_inputs = [
                PopulationInput(
                    grid=first.grid,
                    channel=first.channel,
                    rf=RFBuilderConfig(method=first.rf.method, params=extra_params),
                    gain=1.0,
                    layers=first.layers,
                    processing=[],
                ),
                PopulationInput(grid=second_grid, rf=RFBuilderConfig()),
            ]
            # Reset every sugar field to its dataclass default: PopulationConfig
            # raises at construction time if both `inputs` and a non-default
            # sugar field are set (see module docstring).
            self._session.set_by_path(f"{prefix}.target_grid", None)
            self._session.set_by_path(f"{prefix}.target_layers", None)
            self._session.set_by_path(f"{prefix}.innervation_method", "gaussian")
            self._session.set_by_path(f"{prefix}.sigma_d_mm", 0.3)
            self._session.set_by_path(f"{prefix}.connections_per_neuron", 28)
            self._session.set_by_path(f"{prefix}.use_distance_weights", True)
            self._session.set_by_path(f"{prefix}.resolvable_distance_mm", None)
            self._session.set_by_path(f"{prefix}.innervation_params", {})
            self._session.set_by_path(f"{prefix}.inputs", new_inputs)
            # "concat" always runs regardless of how many neurons each
            # input's builder lays out; "sum" additionally requires every
            # input to produce the same neuron count (SimulationEngine
            # raises otherwise), which a freshly added, still-default
            # second input cannot promise -- especially when the first
            # input's builder derives its own lattice (e.g. "template").
            # The user can switch to "sum" once the layouts actually match.
            self._session.set_by_path(f"{prefix}.combine", "concat")
        else:
            new_input = PopulationInput(
                grid=grid_names[0] if grid_names else None, rf=RFBuilderConfig()
            )
            updated = list(pop_cfg.inputs) + [new_input]
            self._session.set_by_path(f"{prefix}.inputs", updated)

    def _on_remove_input(self, index: int, j: int) -> None:
        pop_cfg = self._session.config.populations[index]
        updated = list(pop_cfg.inputs)
        del updated[j]
        self._session.set_by_path(f"populations.{index}.inputs", updated)
        if len(updated) <= 1:
            # Back to (at most) one input: "combine" is meaningless again --
            # reset it to the schema default so a YAML round trip matches
            # what it looked like before "+ input" was ever pressed.
            self._session.set_by_path(f"populations.{index}.combine", "sum")
