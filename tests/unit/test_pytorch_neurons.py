import math

import torch

from sensoryforge.neurons import (
    AdExNeuronTorch,
    FANeuronTorch,
    IzhikevichNeuronTorch,
    MQIFNeuronTorch,
    SANeuronTorch,
)


def _constant_drive(steps: int, features: int, current: float) -> torch.Tensor:
    return torch.full((1, steps, features), current, dtype=torch.float32)


def test_izhikevich_neuron_emits_spikes_under_constant_drive():
    neuron = IzhikevichNeuronTorch(dt=0.1, threshold=30.0)
    drive = _constant_drive(steps=50, features=4, current=12.0)

    _, spikes = neuron(drive)
    assert spikes.shape == (1, 51, 4)
    assert spikes[:, 1:].any()


def test_adex_neuron_runs_without_nan():
    neuron = AdExNeuronTorch(dt=0.1)
    drive = _constant_drive(steps=40, features=2, current=1.5)

    voltages, spikes = neuron(drive)
    assert torch.isfinite(voltages).all()
    assert torch.isfinite(spikes.float()).all()


def _spike_times_ms(spikes: torch.Tensor, dt: float) -> torch.Tensor:
    """Extract spike times (ms) from a [1, steps+1, 1] boolean spike trace.

    Drops the initial sample (index 0, before any input is applied) so a
    spike at time index ``t`` reads as ``t * dt`` ms.
    """
    sp = spikes[0, 1:, 0]
    return torch.nonzero(sp).flatten().float() * dt


def test_adex_ra1_phasic_silences_within_30ms_under_constant_drive():
    """RA1_phasic: strong spike-triggered adaptation must silence firing
    well within the tactile RA population's fast (tens-of-ms) time scale,
    even though the drive current never changes.

    Bench current is 5.0 mA, not the original 40.0 mA: Phase 2b T2b
    re-tuned RA1_phasic's ``R`` from 1.0 to 8.0 against the
    ``tactile_sa1_ra1`` recipe's measured RA onset-transient drive (peak
    ~5.67-13.87 mA across the four benchmark stimuli), which moved the
    rheobase (the saddle-node current above which no stable subthreshold
    fixed point exists -- see the ``ADEX_PRESETS`` docstring) from ~60.6
    mA down to ~7.57 mA. 40 mA is now far past that threshold (the
    neuron fires continuously instead of settling), while 5.0 mA sits
    just below it and reproduces the same phasic, single-early-spike
    character the model was designed for.
    """
    neuron = AdExNeuronTorch(preset="RA1_phasic", dt=0.05)
    steps = int(500.0 / 0.05)
    drive = _constant_drive(steps=steps, features=1, current=5.0)

    _, spikes = neuron(drive)
    times = _spike_times_ms(spikes, dt=0.05)

    assert ((times >= 0.0) & (times < 30.0)).any(), "expected at least one early spike"
    assert not (
        times >= 30.0
    ).any(), f"unexpected spike(s) after 30 ms: {times.tolist()}"


def test_adex_sa1_tonic_keeps_spiking_through_full_drive():
    """SA1_tonic: weak adaptation must not silence firing -- the neuron
    keeps spiking regularly for the whole 500 ms constant-drive window."""
    neuron = AdExNeuronTorch(preset="SA1_tonic", dt=0.05)
    steps = int(500.0 / 0.05)
    drive = _constant_drive(steps=steps, features=1, current=40.0)

    _, spikes = neuron(drive)
    times = _spike_times_ms(spikes, dt=0.05)

    assert (times >= 400.0).any(), "expected spikes in the final 100 ms"
    isi = times[1:] - times[:-1]
    cv = (isi.std() / isi.mean()).item()
    assert cv < 0.5, f"ISI coefficient of variation too high: {cv}"


def test_mqif_neuron_supports_zero_drive():
    neuron = MQIFNeuronTorch(dt=0.1)
    drive = _constant_drive(steps=30, features=1, current=0.0)

    voltages, spikes = neuron(drive)
    assert voltages.shape[1] == drive.shape[1] + 1
    assert spikes.sum() == 0


def test_fa_neuron_threshold_behavior():
    neuron = FANeuronTorch(dt=0.1, theta=0.2)
    drive = torch.zeros(1, 20, 3)
    drive[:, :10, :] = 0.0
    drive[:, 10:, :] = 0.5

    _, spikes = neuron(drive)
    assert spikes.any()


def test_sa_neuron_supports_parameter_sampling():
    neuron = SANeuronTorch(dt=0.2)
    drive = _constant_drive(steps=25, features=2, current=5e-10)

    traces, spikes = neuron(drive)
    assert traces.shape[1] == drive.shape[1] + 1
    assert spikes.dtype == torch.bool
    assert math.isclose(float(neuron.dt_ms), 0.2, rel_tol=1e-6)
