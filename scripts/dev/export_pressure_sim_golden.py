"""Export a small golden fixture from pressure-simulation (task E5).

Writes ``tests/fixtures/pressure_sim_golden/case_small.npz`` (resolved
relative to this file, so it lands in the SensoryForge repo regardless of
the current working directory) containing everything
``tests/integration/test_pressure_sim_parity.py`` needs to reproduce this
run bit-for-bit through SimulationEngine: the stimulus, per-population
innervation weights, filter/model parameters, drive, filtered response,
and spikes.

Run from the pressure-simulation repo root, with the sensoryforge conda
Python (that repo's own tests pass in that environment)::

    cd "~/Documents/pressure simulation"
    PYTHONPATH=. /opt/miniconda3/envs/sensoryforge/bin/python \\
        /path/to/sensoryforge/scripts/dev/export_pressure_sim_golden.py

A small case: 8x8 grid, T=200 record bins, dt_ms=1.0, one SA and one RA
population with fixed (seeded, saved) random innervation weights, and a
positive trapezoid stimulus (kept positive throughout so neither side's
Izhikevich neuron reaches SensoryForge's -120 mV v_floor clamp, which
pressure-simulation's neuron does not have -- F-037).

``run_encoding`` does not return the filtered response, so this script
recomputes drive and filtered response itself, exactly as
``encode_runner.run_encoding`` does internally (same filter classes, same
call order: innervation -> filter(dt=dt_ms) -> * input_gain), and takes
spikes from ``run_encoding`` itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from encoding.encode_runner import PopConfig, run_encoding
from encoding.filters_torch import SAFilterTorch, RAFilterTorch

OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "pressure_sim_golden"
    / "case_small.npz"
)

H, W = 8, 8
T = 200
DT_MS = 1.0
N_SA = 4
N_RA = 4
SEED = 20260914
INPUT_GAIN = 50.0

SA_MODEL_PARAMS = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}  # regular-spiking
RA_MODEL_PARAMS = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}  # fast-spiking
SA_FILTER_PARAMS = {"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0}
RA_FILTER_PARAMS = {"tau_RA": 8.0, "k3": 2.0}


def _trapezoid_stimulus(
    T: int, H: int, W: int, amplitude: float = 5.0, ramp: int = 40, plateau: int = 120
) -> torch.Tensor:
    """A spatially fixed Gaussian bump, temporally trapezoidal, always >= 0."""
    t = torch.arange(T, dtype=torch.float32)
    amp = torch.zeros(T)
    down_start = ramp + plateau
    total = ramp + plateau + ramp

    up = t < ramp
    amp[up] = t[up] / ramp
    amp[(t >= ramp) & (t < down_start)] = 1.0
    down = (t >= down_start) & (t < total)
    amp[down] = 1.0 - (t[down] - down_start) / ramp
    amp = amp.clamp(0.0, 1.0) * amplitude

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    spatial = torch.exp(-(xx**2 + yy**2) / (2 * 0.5**2))
    return amp[:, None, None] * spatial[None, :, :]


def main() -> None:
    torch.manual_seed(SEED)

    stimulus = _trapezoid_stimulus(T, H, W, amplitude=5.0)
    assert (stimulus >= 0).all()

    sa_weights = torch.rand(N_SA, H, W) * 0.5 + 0.1
    ra_weights = torch.rand(N_RA, H, W) * 0.5 + 0.1

    pop_configs = {
        "SA Pop": PopConfig(
            name="SA Pop",
            neuron_type="SA",
            filter_method="sa",
            input_gain=INPUT_GAIN,
            noise_std=0.0,
            model_params=dict(SA_MODEL_PARAMS),
            filter_params=dict(SA_FILTER_PARAMS),
        ),
        "RA Pop": PopConfig(
            name="RA Pop",
            neuron_type="RA",
            filter_method="ra",
            input_gain=INPUT_GAIN,
            noise_std=0.0,
            model_params=dict(RA_MODEL_PARAMS),
            filter_params=dict(RA_FILTER_PARAMS),
        ),
    }
    innervation_weights = {"SA Pop": sa_weights, "RA Pop": ra_weights}

    _, per_pop_spikes = run_encoding(
        stimulus, innervation_weights, pop_configs, dt_ms=DT_MS
    )

    # Recompute drive + filtered response exactly as run_encoding does
    # internally (it doesn't return them).
    stim_flat = stimulus.view(T, H * W)
    recomputed = {}
    for name, pcfg in pop_configs.items():
        W_mat = innervation_weights[name].view(innervation_weights[name].shape[0], -1)
        drive = (stim_flat @ W_mat.T).unsqueeze(0)  # [1, T, N_pop]
        if pcfg.filter_method == "sa":
            filt = SAFilterTorch(dt=DT_MS, **pcfg.filter_params)
        else:
            filt = RAFilterTorch(dt=DT_MS, **pcfg.filter_params)
        filtered = filt.forward(drive, reset_states=True)
        filtered = filtered * pcfg.input_gain
        recomputed[name] = {"drive": drive, "filtered": filtered}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        H=H,
        W=W,
        T=T,
        dt_ms=DT_MS,
        input_gain=INPUT_GAIN,
        stimulus=stimulus.numpy().astype(np.float32),
        sa_weights=sa_weights.numpy().astype(np.float32),
        ra_weights=ra_weights.numpy().astype(np.float32),
        sa_model_params=np.array(json.dumps(SA_MODEL_PARAMS)),
        ra_model_params=np.array(json.dumps(RA_MODEL_PARAMS)),
        sa_filter_params=np.array(json.dumps(SA_FILTER_PARAMS)),
        ra_filter_params=np.array(json.dumps(RA_FILTER_PARAMS)),
        sa_drive=recomputed["SA Pop"]["drive"].numpy().astype(np.float32),
        sa_filtered=recomputed["SA Pop"]["filtered"].numpy().astype(np.float32),
        ra_drive=recomputed["RA Pop"]["drive"].numpy().astype(np.float32),
        ra_filtered=recomputed["RA Pop"]["filtered"].numpy().astype(np.float32),
        sa_spikes=per_pop_spikes["SA Pop"].numpy().astype(np.float32),
        ra_spikes=per_pop_spikes["RA Pop"].numpy().astype(np.float32),
    )
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")
    print(
        f"SA spikes: {per_pop_spikes['SA Pop'].sum().item():.0f}, "
        f"RA spikes: {per_pop_spikes['RA Pop'].sum().item():.0f}"
    )


if __name__ == "__main__":
    main()
