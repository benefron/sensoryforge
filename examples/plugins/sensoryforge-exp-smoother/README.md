# sensoryforge-exp-smoother

A reference SensoryForge `filter` plugin: a small, tested, installable package to copy
from. It registers `ExpSmootherFilterTorch` as `exp_smoother` in `FILTER_REGISTRY`.

A single-pole low-pass filter over time: `y[t] = y[t-1] + (x[t] - y[t-1]) * dt / tau`. A textbook filter written to show the contract (`forward` with its state, `reset_state`, `to_dict`, `from_config`, `get_param_spec`), not a model of any real receptor. Its test checks a step response by hand.

It was started with `sensoryforge new-component filter exp_smoother --dest .` and then filled in.

## Install and check

From this folder, in the environment that has SensoryForge:

```bash
pip install -e .
sensoryforge list-components      # exp_smoother appears
pip install pytest && pytest      # the contract check and the hand-worked case
```

Use it by name in a sensor config (`neuron_model: exp_smoother` or `filter_method: exp_smoother`);
`../sensor.yml` does, and `../README.md` runs it in a data set.

Code: `sensoryforge_exp_smoother/component.py`. Contract: `sensoryforge.testing.contracts.check_component`.
