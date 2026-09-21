"""StimulusConfig records which fields were set, not just their values.

The schema holds one default for every field of every stimulus type, so a
value equal to the default says nothing about intent. The renderer forwards
only set fields (the rest take the stimulus type's own defaults); inferring
"set" from "differs from the default" silently discarded a deliberate
``sigma: 2.0``, and forwarding everything froze a moving edge at
``start == end == [0, 0]``.
"""

import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.stimuli.render import render_for_config

SCHEMA_SIGMA = StimulusConfig().sigma


def test_a_value_written_in_yaml_counts_even_at_the_default():
    stim = StimulusConfig.from_dict({"type": "gaussian", "sigma": SCHEMA_SIGMA})
    assert "sigma" in stim.explicit_fields()
    assert "start" not in stim.explicit_fields()


def test_assignment_after_construction_counts():
    stim = StimulusConfig(type="gaussian")
    assert "sigma" not in stim.explicit_fields()
    stim.sigma = SCHEMA_SIGMA
    assert "sigma" in stim.explicit_fields()


def test_explicitness_survives_a_dict_round_trip():
    stim = StimulusConfig.from_dict({"type": "gaussian", "sigma": SCHEMA_SIGMA})
    again = StimulusConfig.from_dict(stim.to_dict())
    assert again.explicit_fields() == stim.explicit_fields()
    assert "start" not in stim.to_dict(), "unset fields must not be written"


def test_a_whole_config_yaml_round_trip_keeps_a_moving_edge_moving():
    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="G", arrangement="grid", rows=40, cols=40, spacing=0.15)
        ],
        populations=[],
        stimulus=StimulusConfig(type="moving_edge"),
    )
    reloaded = SensoryForgeConfig.from_yaml(config.to_yaml())
    frames = render_for_config(reloaded, duration_ms=330.0, dt_ms=1.0)[0][0]
    assert not torch.allclose(frames[60], frames[280])


def test_a_deliberate_sigma_equal_to_the_schema_default_is_honoured():
    def peak_width(stim):
        config = SensoryForgeConfig(
            grids=[
                GridConfig(name="G", arrangement="grid", rows=60, cols=60, spacing=0.15)
            ],
            populations=[],
            stimulus=stim,
        )
        frame = render_for_config(config, duration_ms=5.0, dt_ms=1.0)[0][0, 2]
        return int((frame > 0.5 * frame.max()).sum())

    unset = peak_width(StimulusConfig(type="gaussian"))
    chosen = peak_width(
        StimulusConfig.from_dict({"type": "gaussian", "sigma": SCHEMA_SIGMA})
    )
    assert chosen != unset, "sigma written in the block was ignored"
