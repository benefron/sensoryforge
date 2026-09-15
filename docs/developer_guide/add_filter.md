# Adding a New Temporal Filter

This guide covers adding a new mechanoreceptor temporal filter to SensoryForge.
Filters transform the raw spatial drive into a population-specific current
waveform (e.g. SA = slow-adapting, RA = rapidly-adapting).

There are two ways to ship a new filter — pick one before you start:

- **Plugin package** (recommended for most cases, including anyone outside
  the core project): `sensoryforge new-component filter MyFilter --dest DIR`
  scaffolds an installable package that registers itself via an entry
  point — no edits to this checkout. See `plugins.md` for the full guide;
  this page's steps 2 and 4 (the class body and its test) still apply
  verbatim inside the scaffolded package.
- **In-repo** (for contributing to SensoryForge itself):
  `sensoryforge new-component filter MyFilter --in-repo` writes directly
  into `sensoryforge/filters/`, `tests/`, `docs/`, and step 3 below (manual
  registration in `register_components.py`) applies.

Component names are matched **case-insensitively** (F-046): `"Bandpass"`
and `"bandpass"` are the same registration, so you don't need to worry
about exact-case collisions with a built-in filter's name — though a
genuine collision (same name, different class) still raises.

---

## 1. Understand the API contract

All filters must inherit from `BaseFilter` (`sensoryforge/filters/base.py`):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward(x)` | `[B, T, N] → [B, T, N]` | Apply temporal filter |
| `reset_state()` | `→ None` | Clear filter state between runs |
| `from_config(config)` | `dict → cls` | Construct from YAML dict |
| `to_dict()` | `→ dict` | Serialise for YAML round-trip |
| `get_param_spec()` | `→ list[ParamSpec]` | Required on every component (G1); GUI auto-discovery |

`to_dict()` should include **every** `__init__` parameter, and
`from_config(instance.to_dict())` should round-trip to an equal `to_dict()`.
`sensoryforge.testing.contracts.check_component("filter", cls)` checks the
basic `from_config`/`to_dict` round-trip, `get_param_spec()` presence, and
a forward-pass shape check — but **not** full parameter-completeness: the
H3 completeness check (every `__init__` argument present in `to_dict()`)
is currently wired up for neurons only (`_check_neuron` in
`sensoryforge/testing/contracts.py`), not `_check_filter`. `SAFilterTorch`
and `RAFilterTorch` would fail it today (missing `tau_r`/`tau_d`/`k1`/`k2`/
`clip_to_positive` and `tau_RA`/`k3` respectively). Not yet enforced for
other kinds (see ledger F-049).

> **Polymorphism warning:** The interface uses `reset_state()` (singular).
> The existing `SAFilterTorch` has `reset_states()` (plural) — this is a
> known inconsistency tracked as tech debt.  New filters must use
> `reset_state()`.

**Tensor conventions:**

- Input `x`: `[batch, time, N_neurons]` in **mA**  
- Output: same shape, same units
- `dt` is in **ms** at the user-facing API

---

## 2. Create the module file

Example — a bandpass (RA-like) filter using a difference of exponentials:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from sensoryforge.filters.base import BaseFilter


class BandpassFilterTorch(BaseFilter):
    """Difference-of-exponentials bandpass filter (RA-type).

    Computes:  I_out = tau_on * dI/dt − I_out / tau_off

    Where ``tau_on`` is the onset time constant and ``tau_off`` is the
    decay time constant in ms.

    Args:
        tau_on: Onset integration time constant in ms.
        tau_off: Decay time constant in ms.
        dt: Simulation time step in ms.
        clip_to_positive: If True, clamp output to ≥ 0.
    """

    DEFAULT_CONFIG = {
        "tau_on": 5.0,
        "tau_off": 30.0,
        "dt": 0.1,
        "clip_to_positive": True,
    }

    def __init__(
        self,
        tau_on: float = 5.0,
        tau_off: float = 30.0,
        dt: float = 0.1,
        clip_to_positive: bool = True,
    ) -> None:
        super().__init__()
        if tau_on <= 0:
            raise ValueError(f"tau_on must be positive, got {tau_on}")
        if tau_off <= 0:
            raise ValueError(f"tau_off must be positive, got {tau_off}")
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.dt = dt
        self.clip_to_positive = clip_to_positive
        self._I_on: torch.Tensor | None = None
        self._I_out: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._I_on = None
        self._I_out = None

    def forward(self, x: torch.Tensor, reset_states: bool = True) -> torch.Tensor:
        """Apply bandpass filter over the time dimension.

        Args:
            x: Input drive [batch, time, N_neurons] in mA.
            reset_states: If True, reset hidden state before processing.
                Must be True for stateless repeated calls.

        Returns:
            Filtered current [batch, time, N_neurons] in mA.
        """
        if reset_states:
            self.reset_state()

        batch, T, N = x.shape
        device = x.device

        I_on = (
            self._I_on
            if self._I_on is not None
            else torch.zeros(batch, N, device=device)
        )
        I_out = (
            self._I_out
            if self._I_out is not None
            else torch.zeros(batch, N, device=device)
        )

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            dI_on = (x_t - I_on) / self.tau_on * self.dt
            I_on = I_on + dI_on
            dI_out = (I_on - I_out) / self.tau_off * self.dt
            I_out = I_out + dI_out
            out_t = I_on - I_out
            if self.clip_to_positive:
                out_t = out_t.clamp(min=0.0)
            outputs.append(out_t.unsqueeze(1))

        self._I_on = I_on.detach()
        self._I_out = I_out.detach()
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_config(cls, config: dict) -> "BandpassFilterTorch":
        merged = {**cls.DEFAULT_CONFIG, **config}
        return cls(**merged)

    def to_dict(self) -> dict:
        return {
            "tau_on": self.tau_on,
            "tau_off": self.tau_off,
            "dt": self.dt,
            "clip_to_positive": self.clip_to_positive,
        }

    @classmethod
    def get_param_spec(cls) -> list:
        from sensoryforge.stimuli.base import ParamSpec
        return [
            ParamSpec("tau_on", dtype="float", default=5.0,
                      min_val=0.1, max_val=200.0, unit="ms"),
            ParamSpec("tau_off", dtype="float", default=30.0,
                      min_val=0.1, max_val=500.0, unit="ms"),
            ParamSpec("clip_to_positive", dtype="bool", default=True,
                      advanced=True),
        ]
```

---

## 3. Register the filter (in-repo route only)

If you scaffolded with `--in-repo` (or are hand-writing a contribution to
this checkout), add the registration to
`sensoryforge/register_components.py`. If you are shipping a plugin
package instead, skip this step — the scaffold's `register()` function
plus your package's entry point handles it; see `plugins.md`.

```python
from sensoryforge.filters.my_filter import BandpassFilterTorch  # add import

def register_all():
    ...
    FILTER_REGISTRY.register("bandpass", BandpassFilterTorch)    # add line
    ...
```

---

## 4. Write unit tests

Add `tests/unit/test_bandpass_filter.py`:

```python
import torch
import pytest
from sensoryforge.filters.my_filter import BandpassFilterTorch


@pytest.fixture
def filt():
    return BandpassFilterTorch(dt=0.1)


def test_output_shape(filt):
    x = torch.ones(1, 100, 8) * 2.0
    y = filt(x)
    assert y.shape == x.shape


def test_output_non_negative_with_clip(filt):
    x = torch.randn(1, 100, 8)   # may go negative
    y = filt(x)
    assert y.min().item() >= 0.0


def test_output_can_be_negative_without_clip():
    filt = BandpassFilterTorch(clip_to_positive=False)
    x = torch.randn(1, 100, 8) * 5.0
    y = filt(x)
    # Just check shapes; values may be negative
    assert y.shape == x.shape


def test_state_reset_reproducibility(filt):
    x = torch.ones(1, 60, 4) * 3.0
    y1 = filt(x, reset_states=True)
    y2 = filt(x, reset_states=True)
    assert torch.allclose(y1, y2, atol=1e-5), "Repeated calls with reset must be identical"


def test_zero_input_zero_output(filt):
    x = torch.zeros(1, 100, 8)
    y = filt(x)
    assert y.abs().max().item() < 1e-6


def test_from_config_roundtrip():
    f = BandpassFilterTorch(tau_on=8.0, tau_off=25.0)
    f2 = BandpassFilterTorch.from_config(f.to_dict())
    assert f2.tau_on == f.tau_on and f2.tau_off == f2.tau_off
```

---

## 5. Use from YAML

```yaml
populations:
  - name: RA Pop
    filter_method: bandpass     # ← registry key
    filter_params:
      tau_on: 5.0
      tau_off: 30.0
    ...
```

---

## Checklist

- [ ] Inherits from `BaseFilter`
- [ ] `forward(x, reset_states=True)` accepts `[B, T, N]` and returns same shape
- [ ] `reset_state()` (singular) clears all hidden state tensors
- [ ] `clip_to_positive=True` default for SA/RA-type physiology (prevents negative drive)
- [ ] `from_config()` merges with `DEFAULT_CONFIG` before passing to `__init__`
- [ ] `to_dict()` includes every `__init__` parameter and is a round-trip fixed point with `from_config()` (H3)
- [ ] `get_param_spec()` implemented (required on every component, G1)
- [ ] Registered — either via a plugin package's entry point, or in `register_all()` with a lowercase snake_case key (in-repo route)
- [ ] Unit tests cover: shape, non-negativity, state reset reproducibility, zero input, `check_component("filter", cls)`
