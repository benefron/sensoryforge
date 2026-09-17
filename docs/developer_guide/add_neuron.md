# Adding a New Neuron Model

This guide covers adding a new spiking neuron model to SensoryForge.
After following it, your model will be available in the registry, the CLI,
and the Spiking Neurons GUI tab.

There are two ways to ship a new neuron model — pick one before you start:

- **Plugin package** (recommended for most cases, including anyone outside
  the core project): `sensoryforge new-component neuron MyNeuron --dest DIR`
  scaffolds an installable package that registers itself via an entry
  point — no edits to this checkout. See `plugins.md` for the full guide;
  this page's steps 2 and 4 (the class body and its test) still apply
  verbatim inside the scaffolded package.
- **In-repo** (for contributing to SensoryForge itself):
  `sensoryforge new-component neuron MyNeuron --in-repo` writes directly
  into `sensoryforge/neurons/`, `tests/`, `docs/`, and step 3 below (manual
  registration in `register_components.py`) applies.

Component names are matched **case-insensitively** (H1, F-046): `"LIF"`
and `"lif"` are the same registration, so you don't need to worry about
exact-case collisions with a built-in neuron's name — though a genuine
collision (same name, different class) still raises.

The complete runnable example is `docs/examples/plugin_neuron.py`: the `LIFNeuronTorch` class
below is excerpted from it, so what you can copy from this page is what actually runs. It defines
the class, registers it, runs the shared contract check, and drives it with a synthetic step
current. It is executed by `tests/docs/test_docs_examples.py`.

A neuron model with no `threshold`/`reset` is read out as a continuous state trace instead of
spikes (an **analog readout** -- not a separate plugin kind, since it is the DSL neuron model
taking that branch rather than a new registry): see `docs/user_guide/analog_readouts.md` and its
executed example, `docs/examples/analog_dsl.py`.

---

## 1. Understand the API contract

All neuron models must inherit from `BaseNeuron`
(`sensoryforge/neurons/base.py`) and implement:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward(I)` | `[B, T, N] → tuple[Tensor, Tensor] or Tensor` | Integrate one batch |
| `reset_state()` | `→ None` | Reset membrane state between runs |
| `from_config(config)` | `dict → cls` | Construct from YAML dict |
| `to_dict()` | `→ dict` | Serialise for YAML round-trip |
| `get_param_spec()` | `→ list[ParamSpec]` | Required on every component (G1); GUI auto-discovery |

`to_dict()` must include **every** `__init__` parameter, and
`from_config(instance.to_dict())` must round-trip to an equal `to_dict()`
(H3's completeness requirement — before H3 several built-in neurons only
round-tripped `dt`, not `a`/`b`/`c`/`d`/`tau_m`/etc., F-045). Checked by
`sensoryforge.testing.contracts.check_component("neuron", cls)`.

**Tensor conventions:**

- Input `I`: `[batch, time, N_neurons]` in **mA**  
- Returns `(v_trace, spikes)` or just `spikes`:
  - `v_trace`: `[batch, time, N_neurons]` in **mV**
  - `spikes`: `[batch, time, N_neurons]` bool or `{0, 1}` float

**Time units:** `dt` is stored in **ms** in the model; the ODE integrator
converts to seconds internally.

**`noise_std` is required:** `SimulationEngine._build_populations` (in
`sensoryforge/core/simulation_engine.py`) unconditionally passes
`noise_std=pop_cfg.noise_std` to every neuron's constructor. Every neuron
model — built-in or plugin — must therefore accept a `noise_std: float`
constructor parameter, or it raises `TypeError` the moment it is used in
an actual `SimulationEngine` simulation (F-046-adjacent gap fixed in the
`sensoryforge new-component neuron` scaffold template; see
`plugins.md`).

---

## 2. Create the module file

Add `sensoryforge/neurons/my_neuron.py`.  Example — a simple leaky
integrate-and-fire (LIF) model:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from sensoryforge.neurons.base import BaseNeuron


class LIFNeuronTorch(BaseNeuron):
    """Leaky Integrate-and-Fire neuron (population vectorised).

    Args:
        tau_m: Membrane time constant in ms.
        v_thresh: Spike threshold in mV.
        v_reset: Reset potential in mV.
        v_rest: Resting potential in mV.
        r_m: Membrane resistance in MΩ.
        dt: Integration time step in ms.
        v_floor: Hard lower bound on membrane potential (anti-runaway guard).
        noise_std: Additive noise intensity on the membrane voltage
            (mV/sqrt(ms)). Required: `SimulationEngine` unconditionally
            passes `noise_std` to every neuron's constructor
            (`_build_populations` in `core/simulation_engine.py`), so every
            neuron model must accept it or a real simulation raises
            `TypeError`, even if a model chooses not to use it internally.
    """

    DEFAULT_CONFIG = {
        "tau_m": 20.0,
        "v_thresh": -50.0,
        "v_reset": -65.0,
        "v_rest": -70.0,
        "r_m": 10.0,
        "dt": 0.1,
        "v_floor": -100.0,
        "noise_std": 0.0,
    }

    def __init__(
        self,
        tau_m: float = 20.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        v_rest: float = -70.0,
        r_m: float = 10.0,
        dt: float = 0.1,
        v_floor: float = -100.0,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.r_m = r_m
        self.dt = dt          # ms
        self.v_floor = v_floor
        self.noise_std = noise_std
        self._v: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._v = None

    def forward(
        self, I: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate LIF dynamics over a batch of stimulus traces.

        Args:
            I: Input current [batch, time, N_neurons] in mA.

        Returns:
            (v_trace, spikes): Both [batch, time, N_neurons].
              v_trace in mV; spikes as float {0., 1.}.
        """
        batch, T, N = I.shape
        device = I.device
        dt_s = self.dt / 1000.0   # ms → s

        v = (
            self._v
            if self._v is not None
            else torch.full((batch, N), self.v_rest, device=device)
        )

        v_traces = []
        spike_traces = []

        for t in range(T):
            I_t = I[:, t, :]              # [B, N]
            dv = (-(v - self.v_rest) + self.r_m * I_t) / self.tau_m * (dt_s * 1e3)
            v_next = v + dv * (dt_s * 1e3)

            # Clamp to floor
            if self.v_floor is not None:
                v_next = v_next.clamp(min=self.v_floor)

            if self.noise_std > 0.0:
                v_next = v_next + torch.randn_like(v_next) * (self.noise_std * (dt_s * 1e3) ** 0.5)

            spike = (v_next >= self.v_thresh).float()
            v_next = torch.where(spike.bool(), torch.full_like(v_next, self.v_reset), v_next)

            v_traces.append(v_next.unsqueeze(1))
            spike_traces.append(spike.unsqueeze(1))
            v = v_next

        self._v = v.detach()
        v_trace = torch.cat(v_traces, dim=1)
        spikes = torch.cat(spike_traces, dim=1)
        return v_trace, spikes

    @classmethod
    def from_config(cls, config: dict) -> "LIFNeuronTorch":
        merged = {**cls.DEFAULT_CONFIG, **config}
        return cls(**merged)

    def to_dict(self) -> dict:
        return {
            "tau_m": self.tau_m,
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "r_m": self.r_m,
            "dt": self.dt,
            "v_floor": self.v_floor,
            "noise_std": self.noise_std,
        }

    @classmethod
    def get_param_spec(cls) -> list:
        from sensoryforge.stimuli.base import ParamSpec
        return [
            ParamSpec("tau_m", dtype="float", default=20.0,
                      min_val=0.1, max_val=200.0, unit="ms"),
            ParamSpec("v_thresh", dtype="float", default=-50.0,
                      min_val=-90.0, max_val=0.0, unit="mV"),
            ParamSpec("v_reset", dtype="float", default=-65.0,
                      min_val=-90.0, max_val=0.0, unit="mV"),
            ParamSpec("v_floor", dtype="float", default=-100.0,
                      min_val=-200.0, max_val=-50.0, unit="mV",
                      advanced=True),
            ParamSpec("noise_std", dtype="float", default=0.0,
                      min_val=0.0, max_val=20.0, unit="mV/sqrt(ms)",
                      tooltip="Additive Langevin noise intensity on v"),
        ]
```

> **Performance note:** Do not loop over the batch or neuron dimensions —
> vectorise using tensor operations.  The `for t in range(T)` loop over
> time steps is unavoidable for recurrent state, but never loop over `B`
> or `N`.

---

## 3. Register the neuron (in-repo route only)

If you scaffolded with `--in-repo` (or are hand-writing a contribution to
this checkout), add the registration to
`sensoryforge/register_components.py`. If you are shipping a plugin
package instead, skip this step — the scaffold's `register()` function
plus your package's entry point handles it; see `plugins.md`.

```python
from sensoryforge.neurons.my_neuron import LIFNeuronTorch  # add import

def register_all():
    ...
    NEURON_REGISTRY.register("LIF", LIFNeuronTorch)         # add registration
    ...
```

---

## 4. Write unit tests

Add `tests/unit/test_lif_neuron.py`:

```python
import torch
import pytest
from sensoryforge.neurons.my_neuron import LIFNeuronTorch


@pytest.fixture
def neuron():
    return LIFNeuronTorch(dt=0.1)


def test_output_shapes(neuron):
    I = torch.ones(1, 100, 16) * 3.0
    v, spikes = neuron(I)
    assert v.shape == I.shape
    assert spikes.shape == I.shape


def test_spikes_are_binary(neuron):
    I = torch.ones(1, 200, 8) * 5.0
    _, spikes = neuron(I)
    unique = spikes.unique().tolist()
    assert all(v in [0.0, 1.0] for v in unique)


def test_zero_drive_no_spikes(neuron):
    I = torch.zeros(1, 100, 8)
    _, spikes = neuron(I)
    assert spikes.sum().item() == 0


def test_strong_drive_produces_spikes(neuron):
    I = torch.ones(1, 500, 4) * 20.0
    _, spikes = neuron(I)
    assert spikes.sum().item() > 0


def test_state_reset(neuron):
    I = torch.ones(1, 50, 4) * 5.0
    v1, _ = neuron(I)
    neuron.reset_state()
    v2, _ = neuron(I)
    assert torch.allclose(v1, v2, atol=1e-5), "reset_state must clear internal state"


def test_from_config_roundtrip():
    n = LIFNeuronTorch(tau_m=15.0)
    n2 = LIFNeuronTorch.from_config(n.to_dict())
    assert n2.tau_m == n.tau_m


def test_v_floor_prevents_runaway():
    n = LIFNeuronTorch(v_floor=-80.0, dt=0.1)
    I = torch.ones(1, 100, 4) * -100.0   # negative drive
    v, _ = n(I)
    assert v.min().item() >= -80.0 - 1e-4
```

---

## 5. Use from YAML

```yaml
populations:
  - name: My LIF Pop
    target_grid: main_grid
    neuron_type: SA
    neuron_model: LIF          # ← registry key
    model_params:
      tau_m: 20.0
      v_thresh: -50.0
    ...
```

---

## Checklist

- [ ] Inherits from `BaseNeuron`
- [ ] `forward()` accepts `[B, T, N]` and returns `(v_trace, spikes)` or `spikes`
- [ ] All loops over time steps only; no loops over batch or neuron dims
- [ ] `v_floor` parameter guards against membrane runaway
- [ ] `noise_std: float` constructor parameter accepted (required by `SimulationEngine`, see above)
- [ ] `reset_state()` zeros internal hidden state
- [ ] `to_dict()` includes every `__init__` parameter and is a round-trip fixed point with `from_config()` (H3); default dict in `DEFAULT_CONFIG`
- [ ] `get_param_spec()` implemented (required on every component, G1)
- [ ] Registered — either via a plugin package's entry point, or in `register_all()` with a PascalCase key (in-repo route)
- [ ] `dt` stored in ms; convert to seconds for ODE integration
- [ ] Unit tests cover: shapes, binary spikes, zero drive, state reset, roundtrip, `check_component("neuron", cls)`
