"""Your first simulation: a shipped recipe, a Gaussian stimulus, a raster and rates.

This is the complete script of ``docs/tutorials/first_simulation.md``. Run it with
``python first_simulation.py``; it is also run by ``tests/docs/test_docs_examples.py``
(with a shorter duration and the figure in a temporary folder, set in
``SF_FIRST_SIM_MS`` and ``SF_FIRST_SIM_OUT``).
"""

import os

import matplotlib.pyplot as plt

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset
from sensoryforge.stimuli.render import render_for_config

plt.switch_backend("Agg")  # no window: the script saves a figure

DURATION_MS = float(os.environ.get("SF_FIRST_SIM_MS", "300"))
FIGURE = os.environ.get("SF_FIRST_SIM_OUT", "first_simulation.png")

# 1. A recipe, with the stimulus replaced by a Gaussian (peak 1, sigma 0.5 mm).
config_dict = load_preset("tactile_sa1_ra1")
config_dict["stimulus"] = {"type": "gaussian", "amplitude": 1.0, "sigma": 0.5}
config = SensoryForgeConfig.from_dict(config_dict)
dt_ms = config.simulation.dt_ms

# 2. Render the stimulus: [batch, time, height, width].
stimulus, _time_ms, _canvas, _dropped = render_for_config(
    config, duration_ms=DURATION_MS, dt_ms=dt_ms
)
print("stimulus", tuple(stimulus.shape))

# 3. Run. Results are {population name: {"spikes": [batch, time, neurons], ...}}.
engine = SimulationEngine(config)
results = engine.run(stimulus)

# 4. Look at the spikes: batch first, so take batch 0.
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, (name, pop) in zip(axes, results.items()):
    spikes = pop["spikes"][0]  # [time, neurons]
    n_neurons = spikes.shape[1]
    t_idx, n_idx = (spikes > 0).nonzero(as_tuple=True)
    ax.scatter(t_idx * dt_ms, n_idx, s=2, marker="|")
    ax.set_ylabel(f"{name}\nneuron")
    total = int(spikes.sum())
    print(
        f"{name}: {total} spikes from {n_neurons} neurons "
        f"({total / n_neurons:.2f} per neuron, {DURATION_MS:.0f} ms)"
    )
axes[-1].set_xlabel("time (ms)")
fig.tight_layout()
fig.savefig(FIGURE, dpi=120)
print("saved", FIGURE)
