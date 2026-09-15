"""F-045: neuron models must round-trip ALL constructor parameters.

Before this fix, every concrete neuron's ``to_dict()``/``from_config()`` only
round-tripped ``dt`` (inherited from ``BaseNeuron``'s default implementation),
silently dropping every other constructor argument -- e.g. an Izhikevich
neuron's ``a``/``b``/``c``/``d``/``v_init``/``threshold`` etc. This test
constructs each of the five neuron models with non-default parameters and
checks that ``to_dict()`` captures every constructor argument and that
``from_config()`` reconstructs an equivalent instance.

See ``.superpowers/sdd/phase1_tasks/task-H3-brief.md`` and ledger F-045.
"""

from __future__ import annotations

import inspect

import pytest

from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch, IZHIKEVICH_PRESETS
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch
from sensoryforge.neurons.fa import FANeuronTorch
from sensoryforge.neurons.sa import SANeuronTorch


def _init_param_names(cls):
    sig = inspect.signature(cls.__init__)
    names = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(name)
    return names


def _assert_roundtrip(instance, *, excluded=frozenset()):
    cls = type(instance)
    d1 = instance.to_dict()

    missing = [
        name
        for name in _init_param_names(cls)
        if name not in d1 and name not in excluded
    ]
    assert (
        not missing
    ), f"{cls.__name__}.to_dict() is missing constructor parameters: {missing}"

    reconstructed = cls.from_config(d1)
    d2 = reconstructed.to_dict()
    assert d1 == d2, f"{cls.__name__}: from_config(to_dict()) is not a fixed point"
    return d1


class TestIzhikevichRoundtrip:
    def test_explicit_params_round_trip(self):
        neuron = IzhikevichNeuronTorch(
            a=0.03,
            b=0.21,
            c=-60.0,
            d=6.0,
            v_init=-70.0,
            u_init=-15.0,
            dt=0.1,
            threshold=25.0,
            a_std=0.001,
            b_std=0.002,
            c_std=0.5,
            d_std=0.3,
            threshold_std=1.0,
            seed=42,
            noise_std=0.5,
            v_floor=-110.0,
        )
        d = _assert_roundtrip(neuron, excluded=frozenset({"preset"}))
        assert d["a"] == 0.03
        assert d["b"] == 0.21
        assert d["c"] == -60.0
        assert d["d"] == 6.0
        assert d["v_init"] == -70.0
        assert d["threshold"] == 25.0
        assert d["seed"] == 42

    def test_preset_resolves_to_numeric_values_not_preset_name(self):
        neuron = IzhikevichNeuronTorch(preset="FS")
        d = neuron.to_dict()

        # Resolved numeric a/b/c/d must be present and correct.
        expected = IZHIKEVICH_PRESETS["FS"]
        assert d["a"] == expected["a"]
        assert d["b"] == expected["b"]
        assert d["c"] == expected["c"]
        assert d["d"] == expected["d"]

        # The preset name itself must not leak into the dict.
        assert "FS" not in d.values()
        assert d.get("preset") != "FS"

        # from_config() must reconstruct identical behavior without the
        # preset name -- passing only the resolved dict is sufficient.
        reconstructed = IzhikevichNeuronTorch.from_config(d)
        assert reconstructed.a == neuron.a
        assert reconstructed.b == neuron.b
        assert reconstructed.c == neuron.c
        assert reconstructed.d == neuron.d
        assert reconstructed.to_dict() == d


class TestAdExRoundtrip:
    def test_explicit_params_round_trip(self):
        neuron = AdExNeuronTorch(
            EL=-72.0,
            VT=-52.0,
            DeltaT=2.5,
            tau_m=22.0,
            tau_w=120.0,
            a=3.0,
            b=0.1,
            v_reset=-59.0,
            v_spike=22.0,
            R=1.2,
            v_init=-68.0,
            w_init=1.0,
            dt=0.1,
            noise_std=0.4,
            v_floor=-125.0,
        )
        d = _assert_roundtrip(neuron)
        assert d["EL"] == -72.0
        assert d["a"] == 3.0
        assert d["v_init"] == -68.0
        assert d["w_init"] == 1.0


class TestMQIFRoundtrip:
    def test_explicit_params_round_trip(self):
        neuron = MQIFNeuronTorch(
            a=0.05,
            b=0.3,
            vr=-58.0,
            vt=-38.0,
            v_reset=-59.0,
            v_peak=32.0,
            d=3.0,
            tau_m=12.0,
            tau_u=110.0,
            v_init=-55.0,
            u_init=2.0,
            dt=0.1,
            noise_std=0.2,
            v_floor=-118.0,
        )
        d = _assert_roundtrip(neuron)
        assert d["a"] == 0.05
        assert d["v_init"] == -55.0
        assert d["u_init"] == 2.0


class TestFARoundtrip:
    def test_explicit_params_round_trip(self):
        neuron = FANeuronTorch(
            vb=0.5,
            A=2.0,
            theta=1.5,
            tau_ref=3.0,
            dt=0.1,
            input_gain=2.5,
            baseline_mode="sequence",
            tau_dc=60.0,
            noise_std=0.1,
        )
        d = _assert_roundtrip(neuron)
        assert d["vb"] == 0.5
        assert d["baseline_mode"] == "sequence"
        assert d["input_gain"] == 2.5


class TestSARoundtrip:
    def test_explicit_params_round_trip(self):
        neuron = SANeuronTorch(
            I_tau=30e-12,
            I_th=9.0e-9,
            I_tau_ahp=22e-12,
            I_th_ahp=17e-12,
            I_tau_refractory=1.7e-9,
            C_mem=110e-15,
            C_adap=260e-15,
            C_refractory=210e-15,
            U_T=27e-3,
            kappa=0.71,
            Ia_frac=0.75,
            dt=0.2,
            z_reset=1e-12,
            I_in_op=550e-12,
            current_scale=2e-12,
            noise_std=0.05,
        )
        d = _assert_roundtrip(neuron)
        assert d["I_tau"] == 30e-12
        assert d["Ia_frac"] == 0.75
        assert d["dt"] == 0.2

    def test_default_auto_dt_round_trips(self):
        # dt=None triggers auto-derivation from tau_s; the resolved value
        # (not None) must be what round-trips.
        neuron = SANeuronTorch()
        d = _assert_roundtrip(neuron)
        assert d["dt"] == pytest.approx(neuron.dt_ms)


@pytest.mark.parametrize(
    "cls",
    [
        IzhikevichNeuronTorch,
        AdExNeuronTorch,
        MQIFNeuronTorch,
        FANeuronTorch,
        SANeuronTorch,
    ],
)
def test_default_instance_to_dict_has_every_constructor_param(cls):
    excluded = frozenset({"preset"}) if cls is IzhikevichNeuronTorch else frozenset()
    _assert_roundtrip(cls(), excluded=excluded)
