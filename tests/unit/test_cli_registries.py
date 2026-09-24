"""Regression for F-018 (task F2): the CLI reads the live component
registries instead of a hardcoded, driftable print block.

Before this fix, `cmd_list_components` printed a fixed string that didn't
know about newly-registered components (e.g. it already omitted "fa"/"sa"
and listed a nonexistent "center_surround" filter), and `cmd_validate`
always instantiated a `GeneralizedTactileEncodingPipeline`, even for
canonical configs, so canonical-only errors slipped through.
"""

import argparse
import io
from contextlib import redirect_stdout


from sensoryforge.cli import cmd_list_components, cmd_validate
from sensoryforge.registry import NEURON_REGISTRY


class _DummyNeuron:
    """Minimal stand-in registered directly with NEURON_REGISTRY."""

    @classmethod
    def from_config(cls, config):
        return cls()


def test_list_components_reflects_a_newly_registered_neuron():
    NEURON_REGISTRY.register("dummy_test_neuron", _DummyNeuron)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = cmd_list_components(argparse.Namespace())
        assert exit_code == 0
        assert "dummy_test_neuron" in buf.getvalue()
    finally:
        del NEURON_REGISTRY._registry["dummy_test_neuron"]


def test_list_components_lists_registered_innervation_methods():
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_list_components(argparse.Namespace())
    output = buf.getvalue()
    for name in ["gaussian", "one_to_one", "uniform", "distance_weighted"]:
        assert name in output


def _canonical_config() -> dict:
    return {
        "grids": [
            {"name": "g", "rows": 4, "cols": 4, "spacing": 1.0, "arrangement": "grid"}
        ],
        "populations": [
            {
                "name": "SA Pop",
                "target_grid": "g",
                "neuron_type": "SA",
                "neurons_per_row": 2,
                "innervation_method": "gaussian",
                "connections_per_neuron": 4,
                "sigma_d_mm": 2.0,
                "filter_method": "none",
                "neuron_model": "Izhikevich",
                "input_gain": 1.0,
                "noise_std": 0.0,
                "seed": 42,
            }
        ],
        "simulation": {"dt_ms": 1.0, "device": "cpu"},
    }


def test_validate_canonical_config_uses_simulation_engine(tmp_path, capsys):
    import yaml

    config_path = tmp_path / "canonical.yml"
    config_path.write_text(yaml.safe_dump(_canonical_config()))

    exit_code = cmd_validate(argparse.Namespace(config=str(config_path)))
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SA Pop" in captured.out


def test_validate_canonical_config_with_bad_filter_method_fails(tmp_path):
    import yaml

    config = _canonical_config()
    config["populations"][0]["filter_method"] = "does_not_exist"
    config_path = tmp_path / "bad_canonical.yml"
    config_path.write_text(yaml.safe_dump(config))

    exit_code = cmd_validate(argparse.Namespace(config=str(config_path)))
    assert exit_code == 1
