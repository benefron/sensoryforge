# sensoryforge-leaky-threshold

A reference SensoryForge `neuron` plugin: a small, tested, installable package to copy
from. It registers `LeakyThresholdNeuronTorch` as `leaky_threshold` in `NEURON_REGISTRY`.

A leaky integrate-and-fire membrane: `dv/dt = (-(v - v_reset) + I) / tau_m`, a spike when `v` reaches `v_thresh`, then `v` returns to `v_reset`. A textbook model written to show the contract (`forward`, `to_dict`, `get_param_spec`, `reset_state`, the `noise_std` argument every neuron takes), not a model of any real cell. Its test checks one hand-worked step.

It was started with `sensoryforge new-component neuron leaky_threshold --dest .` and then filled in.

## Install and check

From this folder, in the environment that has SensoryForge:

```bash
pip install -e .
sensoryforge list-components      # leaky_threshold appears
pip install pytest && pytest      # the contract check and the hand-worked case
```

Use it by name in a sensor config (`neuron_model: leaky_threshold` or `filter_method: leaky_threshold`);
`../sensor.yml` does, and `../README.md` runs it in a data set.

Code: `sensoryforge_leaky_threshold/component.py`. Contract: `sensoryforge.testing.contracts.check_component`.
