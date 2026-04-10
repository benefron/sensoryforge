"""Tests for dt consistency between GUI pipeline and neuron models (item 8).

The GUI protocol_backend defaults to DEFAULT_DT_MS=1.0 while neuron models
default to dt=0.05. This 20x mismatch causes Forward Euler instability in
the subthreshold regime, producing spurious oscillations and bursting.
"""

import pytest
import torch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch


# ---------------------------------------------------------------------------
# Test 1: Document the dt defaults — changes here are visible in git history
# ---------------------------------------------------------------------------

def test_neuron_model_dt_defaults():
    """All neuron model dt defaults must be < 0.5 ms for Euler stability."""
    models = [
        IzhikevichNeuronTorch(),
        AdExNeuronTorch(),
        MQIFNeuronTorch(),
    ]
    for model in models:
        assert hasattr(model, "dt"), f"{type(model).__name__} must expose .dt"
        assert model.dt < 0.5, (
            f"{type(model).__name__}.dt={model.dt} ms — default should be "
            "< 0.5 ms for Forward Euler stability with typical parameters"
        )


def test_gui_default_dt_documented():
    """Document that GUI protocol_backend uses DEFAULT_DT_MS=1.0.

    This test imports the constant so that any change to the GUI default
    appears clearly in test failures and git history.
    """
    from sensoryforge.gui import protocol_backend
    assert hasattr(protocol_backend, "DEFAULT_DT_MS"), (
        "protocol_backend must export DEFAULT_DT_MS so the GUI dt default "
        "is explicit and trackable"
    )
    # Fixed from 1.0 → 0.1 ms (item 8): 1.0 ms caused Forward Euler instability.
    assert protocol_backend.DEFAULT_DT_MS == 0.1, (
        f"GUI DEFAULT_DT_MS={protocol_backend.DEFAULT_DT_MS}; "
        "expected 0.1 ms (safe Euler bound). Update this test if changed intentionally."
    )


# ---------------------------------------------------------------------------
# Test 2: Euler stability — subthreshold input must NOT produce spikes
# ---------------------------------------------------------------------------

def _count_spikes(model_cls, dt_ms: float, amplitude: float,
                  steps: int = 500) -> int:
    """Run model with constant sub/suprathreshold drive, return spike count."""
    model = model_cls(dt=dt_ms)
    drive = torch.full((1, steps, 1), float(amplitude))
    _, spikes = model(drive)
    return int(spikes.sum().item())


@pytest.mark.parametrize("model_cls,amplitude", [
    (IzhikevichNeuronTorch, 3.0),   # Below Izhikevich rheobase (~10 mA)
    (MQIFNeuronTorch, 3.0),          # Below MQIF rheobase
])
def test_no_spurious_spikes_small_dt(model_cls, amplitude):
    """Subthreshold drive at dt=0.05 ms must produce zero spikes."""
    n = _count_spikes(model_cls, dt_ms=0.05, amplitude=amplitude)
    assert n == 0, (
        f"{model_cls.__name__} fired {n} spurious spikes at dt=0.05 ms "
        f"with subthreshold amplitude={amplitude} — possible reset bug"
    )


@pytest.mark.parametrize("model_cls,amplitude", [
    (IzhikevichNeuronTorch, 3.0),
    (MQIFNeuronTorch, 3.0),
])
def test_large_dt_instability_documented(model_cls, amplitude):
    """At dt=1.0 ms, Euler diverges: subthreshold input produces spurious spikes.

    This test DOCUMENTS the known instability caused by the GUI DEFAULT_DT_MS=1.0
    mismatch. Once the GUI is fixed to use dt <= 0.1 ms (or the neuron models are
    validated at dt=1.0 ms), update or remove this test.
    """
    n_small = _count_spikes(model_cls, dt_ms=0.05, amplitude=amplitude, steps=500)
    n_large = _count_spikes(model_cls, dt_ms=1.0, amplitude=amplitude, steps=500)
    # Document: large dt produces more spurious spikes than small dt
    assert n_large >= n_small, (
        f"{model_cls.__name__}: dt=1.0ms produced {n_large} spikes, "
        f"dt=0.05ms produced {n_small}. "
        "If large-dt is now more stable, the instability is fixed — remove this test."
    )


# ---------------------------------------------------------------------------
# Test 3: No double-fire — exactly one spike per threshold crossing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [
    IzhikevichNeuronTorch,
    AdExNeuronTorch,
    MQIFNeuronTorch,
])
def test_no_consecutive_spikes(model_cls):
    """No two consecutive time steps may both fire for the same neuron.

    The spike+reset logic should prevent v from remaining at threshold
    for two consecutive steps.
    """
    model = model_cls(dt=0.05)
    drive = torch.full((1, 300, 1), 20.0)
    _, spikes = model(drive)
    s = spikes[0, :, 0].float()
    consecutive = int((s[:-1] * s[1:]).sum().item())
    assert consecutive == 0, (
        f"{model_cls.__name__} has {consecutive} consecutive double-fire "
        "events — spike reset may apply in the wrong time slot"
    )


# ---------------------------------------------------------------------------
# Test 4: Voltage reset value correctness after spike
# ---------------------------------------------------------------------------

def test_izhikevich_resets_to_c_after_spike():
    """After a spike, membrane voltage must reset to c (not remain at threshold)."""
    c_val = -65.0
    model = IzhikevichNeuronTorch(dt=0.05, c=c_val, d=8.0)
    drive = torch.full((1, 300, 1), 20.0)
    v_trace, spikes = model(drive)

    spike_found = False
    for t in range(spikes.shape[1] - 1):
        if spikes[0, t, 0]:
            v_after = v_trace[0, t + 1, 0].item()
            assert abs(v_after - c_val) < 5.0, (
                f"After spike at t={t}, v={v_after:.1f} mV; "
                f"expected reset to c={c_val} mV (within 5 mV tolerance)"
            )
            spike_found = True
            break
    assert spike_found, "No spike was produced — increase drive amplitude"


def test_adex_resets_to_v_reset_after_spike():
    """AdEx voltage must reset to v_reset after a spike."""
    v_reset = -58.0
    model = AdExNeuronTorch(dt=0.05, v_reset=v_reset)
    drive = torch.full((1, 300, 1), 500.0)  # AdEx needs larger drive
    v_trace, spikes = model(drive)

    spike_found = False
    for t in range(spikes.shape[1] - 1):
        if spikes[0, t, 0]:
            v_after = v_trace[0, t + 1, 0].item()
            assert abs(v_after - v_reset) < 5.0, (
                f"AdEx after spike at t={t}, v={v_after:.1f} mV; "
                f"expected reset to v_reset={v_reset} mV"
            )
            spike_found = True
            break
    assert spike_found, "No AdEx spike — increase drive amplitude"


# ---------------------------------------------------------------------------
# Test 5: Neuron models accept explicit dt and honour it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [
    IzhikevichNeuronTorch,
    AdExNeuronTorch,
    MQIFNeuronTorch,
])
@pytest.mark.parametrize("dt_ms", [0.05, 0.1, 0.5])
def test_neuron_accepts_explicit_dt(model_cls, dt_ms):
    """Neuron models must accept dt as a constructor parameter and store it."""
    model = model_cls(dt=dt_ms)
    assert abs(model.dt - dt_ms) < 1e-9, (
        f"{model_cls.__name__}(dt={dt_ms}).dt={model.dt}; "
        "constructor must store dt exactly"
    )
