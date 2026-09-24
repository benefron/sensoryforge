import math
from typing import Any, Dict, List

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec

#: Named (a, b, c, d) parameter sets from Izhikevich (2003), "Simple Model of
#: Spiking Neurons", IEEE Trans. Neural Networks 14(6):1569-1572, Fig. 2. Pass
#: ``preset=...`` to :class:`IzhikevichNeuronTorch` instead of spelling out
#: a/b/c/d by hand. ``RS`` reproduces the class's historical defaults exactly.
IZHIKEVICH_PRESETS: dict = {
    "RS": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0},  # regular spiking
    "FS": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0},  # fast spiking
    "IB": {"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.0},  # intrinsically bursting
    "CH": {"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0},  # chattering
    "LTS": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},  # low-threshold spiking
}


class IzhikevichNeuronTorch(BaseNeuron):
    r"""Project-compatible Izhikevich neuron with optional parameter noise.

    Continuous dynamics per feature ``f`` and time ``t``:

    - Membrane: ``dv/dt = 0.04 v^2 + 5 v + 140 - u + I_t``
    - Recovery: ``du/dt = a (b v - u)``

    Forward Euler integrates both states via
    ``state_{t+1} = state_t + dt * dstate/dt``.
    Additive Langevin noise ``η_t ~ N(0, noise_std · sqrt(dt))`` perturbs
    ``v`` before threshold detection.

    When ``v >= threshold`` the neuron emits a spike, resets ``v`` to ``c`` and
    increments ``u`` by ``d`` (with broadcast over batch/features). Parameters
    ``a, b, c, d, threshold`` may be floats or ``(mean, std)`` tuples that are
    sampled per feature each forward pass. Pass ``preset=`` (one of
    ``IZHIKEVICH_PRESETS``, e.g. ``"FS"`` for a fast-spiking population) to
    set a/b/c/d from a named regime instead of spelling them out; an
    explicit ``a=``/``b=``/``c=``/``d=`` still overrides the preset value for
    that one parameter. ledger F-004: RA/RA-I (Meissner) populations should
    use ``preset="FS"`` for parity with pressure-simulation; SA/SA-I
    (Merkel) populations use the default ``"RS"``.

    Inputs ``I`` use shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(v_trace, spikes)`` where ``v_trace`` tracks
    voltages for ``steps+1`` samples and ``spikes`` is a boolean tensor of the
    same shape showing threshold crossings.
    """

    #: F-045: ``preset`` is a constructor convenience that expands to
    #: concrete ``a``/``b``/``c``/``d`` values -- it is intentionally
    #: excluded from ``to_dict()`` (which stores the *resolved* numbers
    #: instead, matching ``resolve_neuron_params`` in
    #: ``sensoryforge/config/defaults.py``) and so from the round-trip
    #: completeness check in ``sensoryforge.testing.contracts``.
    _TO_DICT_EXCLUDE_PARAMS = frozenset({"preset"})

    def __init__(
        self,
        a=None,
        b=None,
        c=None,
        d=None,
        v_init=-65.0,
        u_init=None,
        dt=0.05,
        threshold=30.0,
        a_std=0.0,
        b_std=0.0,
        c_std=0.0,
        d_std=0.0,
        threshold_std=0.0,
        seed=None,
        noise_std: float = 0.0,
        v_floor: float = -120.0,
        *,
        preset: str = "RS",
    ):
        super().__init__()
        if preset not in IZHIKEVICH_PRESETS:
            raise ValueError(
                f"Unknown Izhikevich preset {preset!r}; choose one of "
                f"{sorted(IZHIKEVICH_PRESETS)}"
            )
        defaults = IZHIKEVICH_PRESETS[preset]
        self.preset = preset
        self.a = defaults["a"] if a is None else a
        self.b = defaults["b"] if b is None else b
        self.c = defaults["c"] if c is None else c
        self.d = defaults["d"] if d is None else d
        self.v_init = v_init
        # Handle tuple params: use mean for u_init calculation. Use the
        # resolved self.b (preset default or explicit override), not the
        # raw `b` argument, which is None whenever the caller relies on the
        # preset.
        b_val = self.b[0] if isinstance(self.b, tuple) else self.b
        self.u_init = b_val * v_init if u_init is None else u_init
        self.dt = dt
        self.threshold = threshold
        self.a_std = a_std
        self.b_std = b_std
        self.c_std = c_std
        self.d_std = d_std
        self.threshold_std = threshold_std
        self.seed = seed
        # Langevin noise intensity (additive, mV/sqrt(ms))
        self.noise_std = noise_std
        # Physiological voltage floor (mV). Izhikevich dynamics can drive v
        # far below the reset value c when input current is very negative.
        # Clamping prevents non-physical hyperpolarization and avoids runaway
        # in the v^2 quadratic term. Default: -120 mV (generous K+ reversal).
        self.v_floor = v_floor

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless Izhikevich).

        The Izhikevich model re-initialises v and u each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter's *resolved* value (F-045).

        ``a``/``b``/``c``/``d`` are the concrete numeric values actually in
        effect on this instance -- whether they came from an explicit
        override or from expanding ``preset`` at construction time. The
        ``preset`` name itself is deliberately not stored: it cannot alone
        reconstruct an instance that had one parameter overridden on top of
        a preset (task A8 semantics), so the resolved numbers are the
        ground truth. ``from_config()`` (inherited ``cls(**config)``)
        reconstructs an equivalent instance without needing ``preset``.

        Returns:
            Dictionary with every ``__init__`` parameter except ``preset``.
        """
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "v_init": self.v_init,
            "u_init": self.u_init,
            "dt": self.dt,
            "threshold": self.threshold,
            "a_std": self.a_std,
            "b_std": self.b_std,
            "c_std": self.c_std,
            "d_std": self.d_std,
            "threshold_std": self.threshold_std,
            "seed": self.seed,
            "noise_std": self.noise_std,
            "v_floor": self.v_floor,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return parameter specifications for UI auto-generation (F-045)."""
        return [
            ParamSpec(
                "a",
                dtype="float",
                default=0.02,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="1/ms",
                tooltip="Recovery variable time scale",
            ),
            ParamSpec(
                "b",
                dtype="float",
                default=0.20,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="",
                tooltip="Recovery variable sensitivity to v",
            ),
            ParamSpec(
                "c",
                dtype="float",
                default=-65.0,
                min_val=-100.0,
                max_val=0.0,
                step=1.0,
                unit="mV",
                tooltip="Post-spike reset voltage",
            ),
            ParamSpec(
                "d",
                dtype="float",
                default=8.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Post-spike recovery variable increment",
            ),
            ParamSpec(
                "v_init",
                dtype="float",
                default=-65.0,
                min_val=-100.0,
                max_val=50.0,
                step=1.0,
                unit="mV",
                tooltip="Initial membrane voltage",
            ),
            ParamSpec(
                "u_init",
                dtype="float",
                default=-13.0,
                min_val=-50.0,
                max_val=50.0,
                step=1.0,
                unit="",
                tooltip="Initial recovery variable (default: b * v_init)",
            ),
            ParamSpec(
                "dt",
                dtype="float",
                default=0.05,
                min_val=0.001,
                max_val=5.0,
                step=0.01,
                unit="ms",
                tooltip="Integration time step",
            ),
            ParamSpec(
                "threshold",
                dtype="float",
                default=30.0,
                min_val=-20.0,
                max_val=60.0,
                step=1.0,
                unit="mV",
                tooltip="Spike detection voltage",
            ),
            ParamSpec(
                "a_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="1/ms",
                tooltip="Per-feature std for a (0 = homogeneous)",
            ),
            ParamSpec(
                "b_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="",
                tooltip="Per-feature std for b (0 = homogeneous)",
            ),
            ParamSpec(
                "c_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="mV",
                tooltip="Per-feature std for c (0 = homogeneous)",
            ),
            ParamSpec(
                "d_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Per-feature std for d (0 = homogeneous)",
            ),
            ParamSpec(
                "threshold_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="mV",
                tooltip="Per-feature std for threshold (0 = homogeneous)",
            ),
            ParamSpec(
                "seed",
                dtype="int",
                default=0,
                min_val=0,
                max_val=2**31 - 1,
                step=1,
                unit="",
                tooltip="Random seed for parameter sampling (None = unseeded)",
            ),
            ParamSpec(
                "noise_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.1,
                unit="mV/sqrt(ms)",
                tooltip="Additive Langevin noise intensity on v",
            ),
            ParamSpec(
                "v_floor",
                dtype="float",
                default=-120.0,
                min_val=-200.0,
                max_val=-50.0,
                step=1.0,
                unit="mV",
                tooltip="Physiological voltage floor (clamp)",
            ),
        ]

    def _dynamics(self, v, u, input_t):
        """Compute Izhikevich state derivatives (subthreshold dynamics only).

        Args:
            v: Membrane voltage tensor [batch, features] in mV.
            u: Recovery variable tensor [batch, features].
            input_t: External current at this time step [batch, features] in mA.

        Returns:
            Tuple (dv, du) of derivative tensors, each [batch, features].
            Uses the current model parameters (a, b) stored on self.
        """
        dv = 0.04 * v**2 + 5 * v + 140 - u + input_t
        du = self.a * (self.b * v - u)
        return dv, du

    def forward(
        self,
        input_current,
        a=None,
        b=None,
        c=None,
        d=None,
        threshold=None,
        solver=None,
    ):
        """
        input_current: torch.Tensor, shape [batch, steps, features]
        Optional per-neuron parameters a, b, c, d: each can be a float or
        (mean, std) tuple.
        Returns:
            v_trace: [batch, steps+1, features]
            spikes: [batch, steps+1, features] (bool)
        """
        batch, steps, features = input_current.shape
        device = input_current.device
        dtype = input_current.dtype

        # Helper to sample per-neuron parameters if (mean, std) tuple is given
        def get_param(val, shape):
            if isinstance(val, tuple) and len(val) == 2:
                mean, std = val
                return torch.normal(
                    mean=torch.full(shape, mean, dtype=dtype, device=device),
                    std=torch.full(shape, std, dtype=dtype, device=device),
                )
            else:
                # Handle tensor/list inputs that might need broadcasting or are
                # already correct
                if val is not None:
                    if not torch.is_tensor(val):
                        try:
                            t_val = torch.tensor(val, dtype=dtype, device=device)
                        except Exception:
                            # Fallback for scalar float/int
                            t_val = torch.tensor(float(val), dtype=dtype, device=device)
                    else:
                        t_val = val.to(dtype=dtype, device=device)

                    if t_val.ndim == 0:
                        return t_val.expand(shape)
                    if t_val.shape == shape:
                        return t_val
                    # Try broadcasting
                    try:
                        return t_val.expand(shape)
                    except RuntimeError:
                        # If shapes don't match and can't broadcast, we have a problem.
                        # But let's return t_val and let the caller crash or handle it,
                        # or try to force it if it's a size mismatch issue.
                        pass

                return torch.full(
                    shape,
                    val if val is not None else 0.0,
                    dtype=dtype,
                    device=device,
                )

        # Use provided or default parameters
        a_val = a if a is not None else self.a
        b_val = b if b is not None else self.b
        c_val = c if c is not None else self.c
        d_val = d if d is not None else self.d
        threshold_val = threshold if threshold is not None else self.threshold
        a_tensor = get_param(a_val, (features,))
        b_tensor = get_param(b_val, (features,))
        c_tensor = get_param(c_val, (features,))
        d_tensor = get_param(d_val, (features,))
        threshold_tensor = get_param(threshold_val, (features,))

        v = torch.full((batch, features), self.v_init, dtype=dtype, device=device)
        # Fix tuple-b u_init: torch.full requires scalar, use expand for tensor
        # (resolves ReviewFinding#M1)
        if isinstance(self.b, tuple):
            u = (b_tensor * self.v_init).unsqueeze(0).expand(batch, features).clone()
        else:
            u = torch.full((batch, features), self.u_init, dtype=dtype, device=device)
        v_trace = torch.zeros((batch, steps + 1, features), dtype=dtype, device=device)
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        v_trace[:, 0, :] = v

        for t in range(steps):
            fired = v >= threshold_tensor.unsqueeze(0).expand_as(v)
            v_vis = v.clone()
            v_vis[fired] = threshold_tensor.unsqueeze(0).expand_as(v)[fired]
            v_trace[:, t, :] = v_vis
            spikes[:, t, :] = fired
            v_next = torch.where(fired, c_tensor.unsqueeze(0).expand_as(v), v)
            u_next = torch.where(fired, u + d_tensor.unsqueeze(0).expand_as(u), u)
            not_fired = ~fired
            dv = 0.04 * v**2 + 5 * v + 140 - u + input_current[:, t, :]
            du = a_tensor.unsqueeze(0) * (b_tensor.unsqueeze(0) * v - u)
            # NOTE: _dynamics() uses self.a/self.b; forward() uses per-neuron
            # tensors for heterogeneous populations (a_tensor, b_tensor).
            # The two are equivalent when a=self.a, b=self.b (no std).
            # Add Langevin noise when integrating v (not during reset)
            if self.noise_std != 0.0:
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            u_next = torch.where(not_fired, u + self.dt * du, u_next)
            # Clamp v to physiological floor to prevent non-physical
            # hyperpolarization when drive is very negative (e.g., noise
            # through SA filter). Does not affect spiking neurons since
            # they are reset to c before this clamp is applied.
            if self.v_floor is not None:
                v_next = v_next.clamp(min=self.v_floor)
            v = v_next
            u = u_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= threshold_tensor.unsqueeze(0).expand_as(v)
        return v_trace, spikes
