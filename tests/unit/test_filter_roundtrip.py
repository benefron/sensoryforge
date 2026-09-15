"""F-049: filters must round-trip ALL constructor parameters.

Before this fix, ``SAFilterTorch``/``RAFilterTorch`` only overrode ``forward()``
and inherited ``BaseFilter``'s default ``to_dict()``/``from_config()``, which
round-trips only ``dt``. A filter built with non-default ``tau_r``/``tau_d``/
``k1``/``k2``/``clip_to_positive`` (SA) or ``tau_RA``/``k3`` (RA) silently lost
every other constructor argument on ``from_config(to_dict())``, reconstructing
with the class defaults instead.

See ``docs_root/LEDGER.md`` F-049 and the analogous neuron fix at F-045
(``tests/unit/test_neuron_roundtrip.py``).
"""

from __future__ import annotations

import inspect

from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch


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


def _assert_roundtrip(instance):
    cls = type(instance)
    d1 = instance.to_dict()

    missing = [name for name in _init_param_names(cls) if name not in d1]
    assert (
        not missing
    ), f"{cls.__name__}.to_dict() is missing constructor parameters: {missing}"

    reconstructed = cls.from_config(d1)
    d2 = reconstructed.to_dict()
    assert d1 == d2, f"{cls.__name__}: from_config(to_dict()) is not a fixed point"
    return d1


class TestSAFilterRoundtrip:
    def test_explicit_params_round_trip(self):
        filt = SAFilterTorch(
            tau_r=7.0,
            tau_d=40.0,
            k1=0.1,
            k2=2.0,
            clip_to_positive=True,
        )
        d = _assert_roundtrip(filt)
        assert d["tau_r"] == 7.0
        assert d["tau_d"] == 40.0
        assert d["k1"] == 0.1
        assert d["k2"] == 2.0
        assert d["clip_to_positive"] is True

        reconstructed = SAFilterTorch.from_config(d)
        assert reconstructed.tau_r == filt.tau_r
        assert reconstructed.tau_d == filt.tau_d
        assert reconstructed.k1 == filt.k1
        assert reconstructed.k2 == filt.k2
        assert reconstructed.clip_to_positive == filt.clip_to_positive
        assert reconstructed.dt == filt.dt

    def test_default_instance_round_trips(self):
        _assert_roundtrip(SAFilterTorch())


class TestRAFilterRoundtrip:
    def test_explicit_params_round_trip(self):
        filt = RAFilterTorch(tau_RA=12.0, k3=5.0)
        d = _assert_roundtrip(filt)
        assert d["tau_RA"] == 12.0
        assert d["k3"] == 5.0

        reconstructed = RAFilterTorch.from_config(d)
        assert reconstructed.tau_RA == filt.tau_RA
        assert reconstructed.k3 == filt.k3
        assert reconstructed.dt == filt.dt

    def test_default_instance_round_trips(self):
        _assert_roundtrip(RAFilterTorch())
