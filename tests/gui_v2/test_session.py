"""The GUI v2 Session is the one in-memory experiment every screen binds to.

Covers :mod:`sensoryforge.gui.session`: the signal emitted by each mutator,
dotted-path reads and writes into the config, the errors a bad path raises,
stale bookkeeping against the last results, and the config snapshot a
:class:`RunResult` carries.
"""

import copy
import datetime

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.project import ProjectHandle  # noqa: E402
from sensoryforge.gui.session import (  # noqa: E402
    KNOWN_DEVICES,
    RunResult,
    Session,
    available_devices,
    default_device,
)


def _config() -> SensoryForgeConfig:
    """Two populations: one sugar-shaped, one with explicit inputs."""
    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="skin", rows=8, cols=8, spacing=0.4),
            GridConfig(name="deep", rows=4, cols=4, spacing=0.8),
        ],
        populations=[
            PopulationConfig(
                name="SA",
                neuron_type="SA",
                target_grid="skin",
                model_params={"a": 0.02},
                filter_params={"tau_r": 5.0},
            ),
            PopulationConfig(
                name="RA",
                neuron_type="RA",
                inputs=[
                    PopulationInput(grid="skin", rf=RFBuilderConfig(), gain=1.0),
                    PopulationInput(grid="deep", rf=RFBuilderConfig(), gain=2.0),
                ],
                filter_params={"tau_r": 8.0},
            ),
        ],
        metadata={"name": "demo"},
    )
    config.stimulus.amplitude = 30.0
    config.simulation.dt_ms = 1.0
    return config


@pytest.fixture
def session(qtbot) -> Session:
    """A Session over :func:`_config` (``qtbot`` guarantees a QApplication)."""
    return Session(_config())


# ------------------------------------------------------------------ construction


def test_a_fresh_session_starts_with_no_project_results_or_staleness(session):
    assert session.project is None
    assert session.last_results is None
    assert session.stale is False
    assert isinstance(session.config, SensoryForgeConfig)


def test_a_session_built_without_a_config_gets_an_empty_one(qtbot):
    assert Session().config == SensoryForgeConfig()


def test_available_devices_always_offers_cpu_first(session):
    assert session.available_devices[0] == "cpu"
    assert set(session.available_devices) <= set(KNOWN_DEVICES)
    assert available_devices()[0] == "cpu"


def test_the_default_device_is_the_first_accelerator_if_there_is_one():
    assert default_device(["cpu"]) == "cpu"
    assert default_device(["cpu", "mps"]) == "mps"
    assert default_device(["cpu", "cuda"]) == "cuda"
    assert default_device(["cpu", "mps", "cuda"]) == "mps"


def test_a_session_starts_on_the_default_device(session):
    assert session.device == default_device(session.available_devices)


# ---------------------------------------------------------------------- signals


def test_replace_config_swaps_the_config_and_announces_a_rebuild(session, qtbot):
    replacement = SensoryForgeConfig(grids=[GridConfig(name="other")])

    with qtbot.waitSignal(session.configReplaced, timeout=1000):
        session.replace_config(replacement)

    assert session.config is replacement


def test_replace_config_rejects_something_that_is_not_a_config(session):
    with pytest.raises(ValueError, match="SensoryForgeConfig"):
        session.replace_config({"grids": []})


def test_notify_emits_the_path_that_changed(session, qtbot):
    with qtbot.waitSignal(session.configChanged, timeout=1000) as blocker:
        session.notify("populations.0.filter_params.tau_r")

    assert blocker.args == ["populations.0.filter_params.tau_r"]


def test_set_device_announces_only_a_real_change(session, qtbot):
    session.set_device("cpu")

    with qtbot.waitSignal(session.deviceChanged, timeout=1000) as blocker:
        session.set_device("cuda")
    assert blocker.args == ["cuda"]
    assert session.device == "cuda"

    with qtbot.assertNotEmitted(session.deviceChanged):
        session.set_device("cuda")


def test_set_device_rejects_an_unknown_device(session):
    with pytest.raises(ValueError, match="gpu"):
        session.set_device("gpu")


def test_set_project_announces_the_new_project(session, qtbot, tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())

    with qtbot.waitSignal(session.projectChanged, timeout=1000) as blocker:
        session.set_project(project)

    assert blocker.args == [project]
    assert session.project is project

    with qtbot.waitSignal(session.projectChanged, timeout=1000) as blocker:
        session.set_project(None)
    assert blocker.args == [None]


def test_set_results_publishes_the_results_object(session, qtbot):
    result = _run_result(session)

    with qtbot.waitSignal(session.resultsChanged, timeout=1000) as blocker:
        session.set_results(result)

    assert blocker.args == [result]
    assert session.last_results is result

    with qtbot.waitSignal(session.resultsChanged, timeout=1000) as blocker:
        session.set_results(None)
    assert blocker.args == [None]
    assert session.last_results is None


# ------------------------------------------------------------------- staleness


def test_editing_without_results_is_not_stale(session, qtbot):
    with qtbot.assertNotEmitted(session.staleChanged):
        session.notify("stimulus.amplitude")

    assert session.stale is False


def test_the_first_edit_after_a_run_makes_the_session_stale(session, qtbot):
    session.set_results(_run_result(session))

    with qtbot.waitSignal(session.staleChanged, timeout=1000) as blocker:
        session.notify("stimulus.amplitude")
    assert blocker.args == [True]
    assert session.stale is True

    # Already stale: the second edit changes nothing, so nothing is emitted.
    with qtbot.assertNotEmitted(session.staleChanged):
        session.notify("stimulus.spread")


def test_new_results_clear_staleness(session, qtbot):
    session.set_results(_run_result(session))
    session.notify("stimulus.amplitude")

    with qtbot.waitSignal(session.staleChanged, timeout=1000) as blocker:
        session.set_results(_run_result(session))

    assert blocker.args == [False]
    assert session.stale is False


# ------------------------------------------------------------------- accessors


def test_population_and_grid_are_looked_up_by_index_and_name(session):
    assert session.population(1).name == "RA"
    assert session.population(-1).name == "RA"
    assert session.grid("deep").spacing == pytest.approx(0.8)


def test_an_unknown_population_index_names_the_range(session):
    with pytest.raises(ValueError, match="2 population"):
        session.population(5)


def test_an_unknown_grid_name_lists_the_known_ones(session):
    with pytest.raises(ValueError, match="skin"):
        session.grid("elbow")


# ----------------------------------------------------------------- paths: read


@pytest.mark.parametrize(
    "path, expected",
    [
        ("grids.0.spacing", 0.4),
        ("grids.1.name", "deep"),
        ("populations.0.name", "SA"),
        ("populations.0.model_params.a", 0.02),
        ("populations.1.filter_params.tau_r", 8.0),
        ("populations.1.inputs.1.gain", 2.0),
        ("populations.1.inputs.0.grid", "skin"),
        ("populations.1.inputs.0.rf.method", "gaussian"),
        ("stimulus.amplitude", 30.0),
        ("simulation.dt_ms", 1.0),
        ("metadata.name", "demo"),
    ],
)
def test_get_by_path_reaches_every_kind_of_segment(session, path, expected):
    assert session.get_by_path(path) == expected


def test_get_by_path_can_return_a_whole_branch(session):
    assert session.get_by_path("populations.0") is session.config.populations[0]


# ---------------------------------------------------------------- paths: write


def test_set_by_path_writes_a_dataclass_field_and_announces_it(session, qtbot):
    with qtbot.waitSignal(session.configChanged, timeout=1000) as blocker:
        session.set_by_path("grids.0.spacing", 0.25)

    assert session.config.grids[0].spacing == pytest.approx(0.25)
    assert blocker.args == ["grids.0.spacing"]


def test_set_by_path_writes_into_a_params_dict(session):
    session.set_by_path("populations.1.filter_params.tau_r", 12.0)

    assert session.config.populations[1].filter_params["tau_r"] == pytest.approx(12.0)


def test_set_by_path_may_add_a_key_a_params_dict_does_not_have_yet(session):
    session.set_by_path("populations.0.model_params.b", 0.2)

    assert session.config.populations[0].model_params["b"] == pytest.approx(0.2)


def test_set_by_path_writes_a_field_of_a_nested_input(session):
    session.set_by_path("populations.1.inputs.0.gain", 3.5)
    session.set_by_path("populations.1.inputs.0.rf.method", "one_to_one")

    pop_input = session.config.populations[1].inputs[0]
    assert pop_input.gain == pytest.approx(3.5)
    assert pop_input.rf.method == "one_to_one"


def test_set_by_path_writes_a_list_element(session):
    session.set_by_path("grids.0.color.0", 10)

    assert session.config.grids[0].color[0] == 10


def test_set_by_path_marks_the_session_stale_after_a_run(session):
    session.set_results(_run_result(session))

    session.set_by_path("stimulus.amplitude", 40.0)

    assert session.stale is True


# ---------------------------------------------------------------- paths: errors


@pytest.mark.parametrize(
    "path, message",
    [
        ("", "empty"),
        ("grids.0.nonesuch", "nonesuch"),
        ("nonesuch.0", "nonesuch"),
        ("grids.9.name", "9"),
        ("grids.first.name", "first"),
        ("populations.0.model_params.missing", "missing"),
    ],
)
def test_a_path_that_does_not_resolve_names_the_segment(session, path, message):
    with pytest.raises(ValueError, match=message):
        session.get_by_path(path)


@pytest.mark.parametrize(
    "path, message",
    [
        ("", "empty"),
        ("grids.0.nonesuch", "nonesuch"),
        ("grids.9.name", "9"),
        ("grids.first.name", "first"),
        ("grids.0.color.9", "9"),
    ],
)
def test_a_bad_write_path_names_the_segment_and_changes_nothing(
    session, path, message, qtbot
):
    before = copy.deepcopy(session.config)

    with qtbot.assertNotEmitted(session.configChanged):
        with pytest.raises(ValueError, match=message):
            session.set_by_path(path, 1.0)

    assert session.config == before


# ------------------------------------------------------------------- RunResult


def _run_result(session: Session) -> RunResult:
    """A minimal RunResult; the fields a run fills are exercised elsewhere."""
    return RunResult(
        config_snapshot=copy.deepcopy(session.config),
        results={"spikes": {}},
        stimulus=torch.zeros(1, 4, 8, 8),
        time_ms=np.arange(4, dtype=float),
        canvas=None,
        duration_ms=4.0,
    )


def test_a_run_result_snapshot_is_an_independent_deep_copy(session):
    result = _run_result(session)

    assert result.config_snapshot == session.config
    assert result.config_snapshot is not session.config

    session.set_by_path("stimulus.amplitude", 999.0)

    assert result.config_snapshot.stimulus.amplitude == pytest.approx(30.0)


def test_a_run_result_defaults_to_an_unbundled_full_run(session):
    result = _run_result(session)

    assert result.bundle_dir is None
    assert result.quick is False
    assert result.elapsed_s == pytest.approx(0.0)
    assert isinstance(result.started, datetime.datetime)
