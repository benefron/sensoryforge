"""Regression for F-042 (task E9): dt_ms must be a whole multiple of
integrate_dt_ms.

Before this fix, a record step that wasn't a whole multiple of the
integration step silently rescaled neuron time in
SimulationEngine._run_pop_from_drive: n = round(dt_ms / integrate_dt_ms)
just rounds, so e.g. 0.12 ms record bins would integrate at 0.10 ms
instead of the requested 0.12 ms, with no error.
"""

import pytest

from sensoryforge.config.schema import SimulationConfig, validate_dt_ms
from sensoryforge.core.simulation_engine import SimulationEngine

ACCEPTED = [1.0, 0.5, 0.1, 0.05]
REJECTED = [0.12, 0.07, 0.03]


@pytest.mark.parametrize("dt_ms", ACCEPTED)
def test_validate_dt_ms_accepts_whole_multiples(dt_ms):
    validate_dt_ms(dt_ms, integrate_dt_ms=0.05)  # must not raise


@pytest.mark.parametrize("dt_ms", REJECTED)
def test_validate_dt_ms_rejects_non_whole_multiples(dt_ms):
    # 0.03 < integrate_dt_ms is rejected by the >= check instead (still a
    # ValueError naming both values); 0.12/0.07 hit the whole-multiple check.
    with pytest.raises(ValueError, match=r"whole multiple|>= integrate_dt_ms"):
        validate_dt_ms(dt_ms, integrate_dt_ms=0.05)


def test_validate_dt_ms_rejects_dt_ms_below_integrate_dt_ms():
    with pytest.raises(ValueError, match="integrate_dt_ms"):
        validate_dt_ms(0.02, integrate_dt_ms=0.05)


@pytest.mark.parametrize("dt_ms", ACCEPTED)
def test_simulation_config_accepts_valid_dt_ms(dt_ms):
    config = SimulationConfig(dt_ms=dt_ms)  # must not raise
    assert config.dt_ms == dt_ms


@pytest.mark.parametrize("dt_ms", REJECTED)
def test_simulation_config_rejects_invalid_dt_ms(dt_ms):
    with pytest.raises(ValueError, match=r"0\.05"):
        SimulationConfig(dt_ms=dt_ms)


def test_run_pop_from_drive_rejects_invalid_dt_ms_for_direct_callers():
    import torch

    with pytest.raises(ValueError, match="whole multiple"):
        SimulationEngine._run_pop_from_drive(
            drive=torch.zeros(1, 5, 2),
            filter_module=None,
            neuron_model=lambda x: (x, x > 1e9),
            dt_ms=0.12,
            integrate_dt_ms=0.05,
        )
