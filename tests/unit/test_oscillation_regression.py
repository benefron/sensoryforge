"""Regression tests for oscillation / instability with noise + SA filter + gain (item 8).

Root causes identified:
  RC-1: SA filter k2·dI/dt term amplifies noise 849x vs k1·I term.
         Output swings to -428 mA (56.5% negative) — drives neurons to
         non-physiological hyperpolarization.
  RC-2: Izhikevich v can drop to -166 mV (far below reset c=-65) when
         drive is deeply negative, because the model was designed for
         positive mechanoreceptor input currents only.
  RC-3: SA mechanoreceptors have zero minimum firing rate — negative output
         is never physiologically correct. Clamping to >=0 is the fix.
"""

import math
import pytest
import torch
from sensoryforge.filters.sa_ra import SAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch

DT = 0.1   # ms


# ---------------------------------------------------------------------------
# SA filter output must be non-negative (physiological correctness)
# ---------------------------------------------------------------------------

def test_sa_filter_output_non_negative_for_noise():
    """SA filter output must never be negative for noise input.

    SA mechanoreceptors have zero minimum firing rate; negative output is
    non-physiological and causes extreme hyperpolarization in neuron models.
    """
    torch.manual_seed(0)
    noise = torch.randn(1, 2000, 4) * 30.0  # amp=30, 4 neurons

    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    out = sa(noise)

    assert (out >= 0).all(), (
        f"SA filter output has {(out < 0).sum().item()} negative values "
        f"(min={out.min():.3f}). SA mechanoreceptors cannot have negative "
        "output — filter must clamp to >=0."
    )


def test_sa_filter_output_non_negative_for_step_decrement():
    """SA output must remain >=0 even during a step-down stimulus."""
    steps = 500
    stim = torch.zeros(1, steps, 1)
    stim[:, :200, :] = 30.0   # ON for first 200 steps
    # OFF after step 200 — large dI/dt at the edge can make x negative

    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    out = sa(stim)

    assert (out >= 0).all(), (
        f"SA filter went negative during step-down: min={out.min():.4f}"
    )


def test_sa_filter_derivative_amplification_bounded():
    """SA filter output RMS should be bounded relative to input for noise.

    The k2·dI/dt term amplifies noise. After fix, the filter must clamp
    output to >=0, so the noise-amplified negative swings are removed.
    The output RMS must be < 5× the input RMS for unit-amplitude noise.
    """
    torch.manual_seed(1)
    noise = torch.randn(1, 5000, 1) * 1.0  # unit amplitude

    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    out = sa(noise)

    input_rms = noise.std().item()
    output_rms = out.std().item()
    assert output_rms < 5.0 * input_rms, (
        f"SA filter amplified noise by {output_rms/input_rms:.1f}x "
        f"(input_rms={input_rms:.3f}, output_rms={output_rms:.3f}). "
        "Noise amplification should be bounded after output clamping."
    )


# ---------------------------------------------------------------------------
# Neuron model v must stay within physiological bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,v_min_expected", [
    (IzhikevichNeuronTorch, -120.0),   # c=-65, so v shouldn't go below c - 55
    (AdExNeuronTorch, -130.0),         # EL=-70, v shouldn't go below -130
    (MQIFNeuronTorch, -120.0),         # vr=-60, shouldn't go below -120
])
def test_neuron_v_bounded_below_with_large_negative_input(model_cls, v_min_expected):
    """Neuron voltage must not drop below physiological minimum.

    With large negative input current, the Euler step can push v far below
    the reset potential (e.g., -166 mV for Izhikevich). This is non-physical
    and causes pathological dynamics via the v^2 / exp terms.
    """
    model = model_cls(dt=DT)
    # Large sustained negative drive — non-physical but should be handled safely
    large_neg = torch.full((1, 500, 1), -200.0)
    v_trace, spikes = model(large_neg)

    v_min = v_trace.min().item()
    assert v_min >= v_min_expected, (
        f"{model_cls.__name__}: v_min={v_min:.1f} mV < {v_min_expected} mV. "
        "Neuron v must be clamped to prevent non-physiological hyperpolarization."
    )


# ---------------------------------------------------------------------------
# Full pipeline: noise + SA filter + gain=100 — no extreme voltage excursions
# ---------------------------------------------------------------------------

def test_noise_sa_gain100_no_extreme_voltages():
    """Noise → SA filter → gain=100 → neuron: v must stay in [-120, 50].

    This is the scenario the user reported as 'very weird activity'.
    After fixing the SA filter's non-negative output, the voltage trace
    must remain within a physiological range.
    """
    torch.manual_seed(42)
    noise = torch.randn(1, 3000, 1) * 30.0

    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    sa_out = sa(noise)
    drive = sa_out * 100.0

    neuron = IzhikevichNeuronTorch(dt=DT)
    v_trace, spikes = neuron(drive)

    v_min = v_trace.min().item()
    v_max = v_trace.max().item()

    assert v_min >= -120.0, (
        f"v_min={v_min:.1f} mV — Izhikevich v went far below physiological "
        "range. SA filter output clamping should prevent this."
    )
    assert v_max < 50.0, (
        f"v_max={v_max:.1f} mV — v should not greatly exceed threshold (30 mV) "
        "in the stored trace (threshold clipping should apply)."
    )


def test_noise_sa_gain100_no_nan_inf():
    """Noise + SA filter + gain=100 must produce finite outputs."""
    torch.manual_seed(42)
    noise = torch.randn(1, 3000, 1) * 30.0

    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    drive = sa(noise) * 100.0

    neuron = IzhikevichNeuronTorch(dt=DT)
    v_trace, spikes = neuron(drive)

    assert not torch.isnan(v_trace).any(), "NaN in v_trace (noise + SA + gain=100)"
    assert not torch.isinf(v_trace).any(), "Inf in v_trace (noise + SA + gain=100)"


# ---------------------------------------------------------------------------
# SA filter step response must remain correct after clamping fix
# ---------------------------------------------------------------------------

def test_sa_filter_step_response_unchanged():
    """Clamping output to >=0 must not affect the step response (all-positive).

    The step response is always positive — clamping should be a no-op here.
    """
    sa = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=DT)
    step = torch.ones(1, 2000, 1)  # constant positive step
    out = sa(step)

    assert (out >= 0).all(), "Step response should be non-negative"
    # Final value should approach k1 * I = 0.05 * 1.0 = 0.05
    assert abs(out[0, -1, 0].item() - 0.05) < 0.005, (
        f"Steady-state value {out[0,-1,0].item():.4f} != expected k1=0.05"
    )
