import math
from typing import Any, Dict, List

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec


class FANeuronTorch(BaseNeuron):
    r"""Fast-adapting tactile neuron with explicit amplifier/threshold stages.

    For each feature/time step ``t`` the model evaluates:

    1. **Baseline tracking**:
       Sequence mode uses ``m_t = mean(x)``.
        EMA mode uses ``m_t = m_{t-1} + α (x_t - m_{t-1})`` with
       ``α = dt / tau_dc``.
       Inputs are scaled by ``input_gain`` before baseline removal.
    2. **Amplifier**:
       ``v_det(t) = v_b - A · (g · x_t - m_t)`` with ``g = input_gain``.
       Optional Langevin noise adds ``η_t ~ N(0, noise_std · sqrt(dt))``.
    3. **Threshold/refractory**:
       Spikes fire when ``|v_det - v_b| >= θ`` and the neuron is not
       refractory.
       Each spike resets ``v`` to ``v_b`` for ``ceil(tau_ref / dt)`` steps.

    Inputs ``x`` have shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(va_trace, spikes)`` where ``va_trace`` stores
    amplifier voltages ``v_a`` with shape ``[batch, steps+1, features]`` and
    ``spikes`` is a boolean tensor of the same shape. Parameters may be floats
    or ``(mean, std)`` tuples; tuples trigger per-feature sampling each forward
    pass to capture receptor variability.
    """

    def __init__(
        self,
        vb=0.0,
        A=1.0,
        theta=1.0,
        tau_ref=2.0,
        dt=0.05,
        input_gain=1.0,
        baseline_mode: str = "ema",  # 'sequence' | 'ema'
        tau_dc=50.0,  # ms, only used for 'ema'
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.vb = vb
        self.A = A
        self.theta = theta
        self.tau_ref = tau_ref
        self.dt = dt
        self.input_gain = input_gain
        self.baseline_mode = baseline_mode
        self.tau_dc = tau_dc
        self.noise_std = noise_std

        # Stateful buffers (created lazily on first forward)
        self._ref_count = None  # [batch, features] int steps
        self._ema_mean = None  # [batch, features]

    @staticmethod
    def _param_tensor(val, shape, device, dtype):
        """Float or (mean, std) -> tensor of given shape on device/dtype."""
        if isinstance(val, tuple) and len(val) == 2:
            mean, std = val
            mean_t = torch.full(shape, float(mean), dtype=dtype, device=device)
            std_t = torch.full(shape, float(std), dtype=dtype, device=device)
            return torch.normal(mean=mean_t, std=std_t)
        return torch.full(shape, float(val), dtype=dtype, device=device)

    def _reset_states(self, batch, features, device, dtype):
        self._ref_count = torch.zeros(
            (batch, features), dtype=torch.long, device=device
        )
        self._ema_mean = torch.zeros((batch, features), dtype=dtype, device=device)

    def reset_state(self) -> None:
        """Reset internal refractory and EMA states.

        Call between independent sequences to avoid temporal carryover
        (resolves ReviewFinding#H6).
        """
        self._ref_count = None
        self._ema_mean = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter's current value (F-045).

        Returns:
            Dictionary with every ``__init__`` parameter. Excludes internal
            stateful buffers (``_ref_count``/``_ema_mean``), which are not
            constructor arguments and are re-initialised lazily.
        """
        return {
            "vb": self.vb,
            "A": self.A,
            "theta": self.theta,
            "tau_ref": self.tau_ref,
            "dt": self.dt,
            "input_gain": self.input_gain,
            "baseline_mode": self.baseline_mode,
            "tau_dc": self.tau_dc,
            "noise_std": self.noise_std,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return parameter specifications for UI auto-generation (F-045)."""
        return [
            ParamSpec(
                "vb",
                dtype="float",
                default=0.0,
                min_val=-20.0,
                max_val=20.0,
                step=0.1,
                unit="mV",
                tooltip="Baseline amplifier voltage",
            ),
            ParamSpec(
                "A",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=50.0,
                step=0.5,
                unit="",
                tooltip="Amplifier gain",
            ),
            ParamSpec(
                "theta",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=50.0,
                step=0.1,
                unit="mV",
                tooltip="Spike threshold on |v_det - vb|",
            ),
            ParamSpec(
                "tau_ref",
                dtype="float",
                default=2.0,
                min_val=0.0,
                max_val=100.0,
                step=0.5,
                unit="ms",
                tooltip="Refractory period",
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
                "input_gain",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=500.0,
                step=0.5,
                unit="",
                tooltip="Gain applied to input before baseline removal",
            ),
            ParamSpec(
                "baseline_mode",
                dtype="str",
                default="ema",
                unit="",
                tooltip="Baseline tracking mode: 'sequence' or 'ema'",
                choices=["sequence", "ema"],
            ),
            ParamSpec(
                "tau_dc",
                dtype="float",
                default=50.0,
                min_val=0.1,
                max_val=1000.0,
                step=1.0,
                unit="ms",
                tooltip="EMA baseline time constant (only used for 'ema')",
            ),
            ParamSpec(
                "noise_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.1,
                unit="mV/sqrt(ms)",
                tooltip="Additive Langevin noise intensity on the amplifier",
            ),
        ]

    def forward(
        self,
        input_current: torch.Tensor,
        *,
        vb=None,
        A=None,
        theta=None,
        tau_ref=None,
        input_gain=None,
        baseline_mode=None,
        tau_dc=None,
        reset_states: bool = True,
    ):
        """
        Args:
            input_current: Input signal [batch, steps, features].
            vb, A, theta, tau_ref, input_gain: Optional overrides (float or
                (mean, std)) sampled per-feature.
            baseline_mode: 'sequence' | 'ema'. If None, uses constructor value.
            tau_dc: EMA time constant in ms (for 'ema' mode).
            reset_states: Reset refractory and EMA states at sequence start.
        Returns:
            va_trace, spikes: both [batch, steps+1, features]
        """
        # Renamed x → input_current for API consistency (resolves ReviewFinding#H6)
        assert (
            input_current.dim() == 3
        ), "input_current must be [batch, steps, features]"
        B, T, F = input_current.shape
        device, dtype = input_current.device, input_current.dtype

        # Resolve per-feature parameters
        vb_t = self._param_tensor(self.vb if vb is None else vb, (F,), device, dtype)
        A_t = self._param_tensor(self.A if A is None else A, (F,), device, dtype)
        th_t = self._param_tensor(
            self.theta if theta is None else theta, (F,), device, dtype
        )
        gain_t = self._param_tensor(
            self.input_gain if input_gain is None else input_gain, (F,), device, dtype
        )
        tref_t = self._param_tensor(
            self.tau_ref if tau_ref is None else tau_ref,
            (F,),
            device,
            dtype,
        )
        # Convert tau_ref (ms) to integer steps per feature
        ref_steps_f = torch.clamp(torch.ceil(tref_t / max(self.dt, 1e-6)), min=1).to(
            torch.long
        )

        # Baseline mode
        bmode = self.baseline_mode if baseline_mode is None else baseline_mode
        tau_dc_val = self.tau_dc if tau_dc is None else tau_dc
        if isinstance(tau_dc_val, tuple):
            tau_dc_val = float(tau_dc_val[0])

        # Initialize state
        if self._ref_count is None or reset_states:
            self._reset_states(B, F, device, dtype)

        # Prepare outputs
        va_trace = torch.zeros((B, T + 1, F), dtype=dtype, device=device)
        spikes = torch.zeros((B, T + 1, F), dtype=torch.bool, device=device)
        va_trace[:, 0, :] = vb_t.view(1, 1, -1)

        # Precompute sequence mean if needed
        if bmode == "sequence":
            mean_seq = (input_current * gain_t.view(1, 1, -1)).mean(dim=1, keepdim=True)

        # EMA coefficient
        if bmode == "ema":
            alpha = self.dt / max(tau_dc_val, 1e-6)

        # Time loop
        refc = self._ref_count
        ema_m = self._ema_mean
        vb_b = vb_t.view(1, -1)
        A_b = A_t.view(1, -1)
        th_b = th_t.view(1, -1)
        gain_b = gain_t.view(1, -1)
        ref_steps_b = ref_steps_f.view(1, -1)

        for t in range(T):
            xt = input_current[:, t, :] * gain_b

            # Baseline <x>
            if bmode == "sequence":
                m_t = mean_seq[:, 0, :]
            elif bmode == "ema":
                if reset_states and t == 0:
                    ema_m = xt.clone()
                else:
                    ema_m = ema_m + alpha * (xt - ema_m)
                m_t = ema_m
            else:
                raise ValueError("baseline_mode must be 'sequence' or 'ema'")

            # Stage 1: amplifier with optional Langevin noise
            va_det = vb_b - A_b * (xt - m_t)
            if self.noise_std != 0.0:
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(va_det) * (self.noise_std * sqrt_dt)
                va_cand = va_det + eta
            else:
                va_cand = va_det

            # Refractory handling
            in_ref = refc > 0
            va_next = torch.where(in_ref, vb_b, va_cand)

            # Stage 2: threshold (only if not refractory)
            fired = (~in_ref) & (torch.abs(va_cand - vb_b) >= th_b)

            # Stage 3: reset/clamp and set refractory for fired units
            va_next = torch.where(fired, vb_b, va_next)
            refc = torch.where(in_ref, refc - 1, refc)
            refc = torch.where(fired, ref_steps_b, refc)

            # Record
            va_trace[:, t + 1, :] = va_next
            spikes[:, t, :] = fired
            spikes[:, t + 1, :] = fired

        # Final state update
        self._ref_count = refc
        if bmode == "ema":
            self._ema_mean = ema_m

        return va_trace, spikes
