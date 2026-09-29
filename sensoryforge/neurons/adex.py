import math
from typing import Any, Dict, List, Optional

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec

#: Historical positional defaults of :class:`AdExNeuronTorch` (pre-preset).
#: Kept as a single source of truth so ``AdExNeuronTorch()`` with no
#: ``preset`` stays bit-identical to the class's original behaviour.
_ADEX_CLASS_DEFAULTS: dict = {
    "EL": -70.0,
    "VT": -50.0,
    "DeltaT": 2.0,
    "tau_m": 20.0,
    "tau_w": 100.0,
    "a": 2.0,
    "b": 0.0,
    "v_reset": -58.0,
    "v_spike": 20.0,
    "R": 1.0,
    # Absolute refractory period (ms): after a spike, v is held at v_reset
    # for t_ref while w keeps evolving. 0 keeps the historical behaviour.
    "t_ref": 0.0,
}

#: Named AdEx regimes. Pass ``preset=...`` to :class:`AdExNeuronTorch`
#: instead of spelling out every parameter by hand -- mirrors
#: :data:`sensoryforge.neurons.izhikevich.IZHIKEVICH_PRESETS` exactly (an
#: explicit keyword argument still overrides that one preset value;
#: ``preset`` is excluded from ``to_dict()``).
#:
#: The model is Brette & Gerstner (2005), J. Neurophysiol. 94:3637-3642; the
#: numeric values are this project's, in its mA/mV/ms convention (``R * I``
#: lands in mV, and ``w`` enters dv/dt in mV).
#:
#: ``SA1_tonic`` / ``RA1_phasic`` (the tactile defaults, resolved by
#: neuron_type) have no adaptation, a 2 ms refractory period and no voltage
#: clamp (D-ce22df3, D-f5853a4). Their rheobase is
#: ``((VT - EL) - DeltaT) / R``: 3.0 mA for SA (``R`` = 6) and 2.25 mA for RA
#: (``R`` = 8). The recipe's per-population ``input_gain`` places the drive
#: above it (``scripts/calibrate_recipe_gains.py``). "Phasic" RA behaviour
#: comes from the RA filter, which differentiates a static drive to zero, not
#: from the neuron.
#:
#: ``SA1_adapting`` / ``RA1_adapting`` add the spike-frequency adaptation
#: fitted to TouchSim's SA1/RA afferents (D-f4d0967, D-d9bd411), as an opt-in:
#: ``model_params: {preset: SA1_adapting}``.
_ADEX_AFFERENT_BASE: dict = {
    "EL": -70.0,
    "VT": -50.0,
    "DeltaT": 2.0,
    "tau_m": 20.0,
    "v_reset": -70.0,
    "v_spike": 20.0,
    # 2 ms absolute refractory period (D-f5853a4): caps the rate near
    # 500 Hz, a physiological ceiling for an afferent.
    "t_ref": 2.0,
}

ADEX_PRESETS: dict = {
    # The tactile defaults carry no adaptation (a = b = 0, D-ce22df3): the
    # neuron is an exponential integrate-and-fire with a refractory period,
    # whose steady rate follows its drive. That keeps it simple enough for
    # hardware and keeps rate a function of the present drive only, which
    # pressure-simulation's Kalman-filter inference relies on. SA's ramp
    # response comes from its filter (k2) and RA's transience from the RA
    # filter, not from the neuron.
    "SA1_tonic": {**_ADEX_AFFERENT_BASE, "tau_w": 110.0, "a": 0.0, "b": 0.0, "R": 6.0},
    "RA1_phasic": {**_ADEX_AFFERENT_BASE, "tau_w": 50.0, "a": 0.0, "b": 0.0, "R": 8.0},
    # Opt-in adaptation: the values fitted to TouchSim's SA1/RA afferents
    # (D-f4d0967, D-d9bd411; scripts/validation/fit_afferents.py). SA adapts
    # through a spike-triggered b with a 110 ms tau_w, so its rate falls from
    # the ramp to the hold and rises gradually with drive; RA through a large
    # a and b, so few spikes follow each transient. w enters dv/dt in mV, so
    # after strong stimulation it can drive the voltage far below rest (ledger
    # F-f59aa11); set v_floor to clamp it if needed.
    "SA1_adapting": {
        **_ADEX_AFFERENT_BASE,
        "tau_w": 110.0,
        "a": 0.02,
        "b": 28.0,
        "R": 6.0,
    },
    "RA1_adapting": {
        **_ADEX_AFFERENT_BASE,
        "tau_w": 50.0,
        "a": 2.0,
        "b": 40.0,
        "R": 8.0,
    },
}


class AdExNeuronTorch(BaseNeuron):
    r"""Adaptive exponential integrate-and-fire neuron (batched PyTorch).

    Per feature and time step the model evaluates:

    1. **Membrane voltage**
        \( \frac{dv}{dt} = \frac{-(v-EL) + \Delta_T e^{(v-VT)/\Delta_T}
        - w + R I}{\tau_m} \).
        Euler update: ``v_{t+1} = v_t + dt * dv/dt`` plus optional Langevin
        noise ``η_t ~ N(0, noise_std · sqrt(dt))``.
    2. **Adaptation current**
        \( \frac{dw}{dt} = \frac{a (v-EL) - w}{\tau_w} \) with Euler update
        ``w_{t+1} = w_t + dt * dw/dt``.
    3. **Spike/reset**
        When ``v >= v_spike`` a spike is emitted, ``v`` is set to
        ``v_reset`` and ``w`` increments by ``b``. Spikes persist for one
        step in ``spikes``.

    Inputs ``I`` have shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(v_trace, spikes)`` where ``v_trace`` stores the
    membrane trajectory ``[batch, steps+1, features]`` before resets and
    ``spikes`` provides boolean events of the same shape.

    Pass ``preset=`` (one of :data:`ADEX_PRESETS`, e.g. ``"RA1_phasic"``)
    to set ``EL``/``VT``/``DeltaT``/``tau_m``/``tau_w``/``a``/``b``/
    ``v_reset``/``v_spike``/``R`` from a named firing-pattern regime instead
    of spelling them out; an explicit keyword argument for any one of those
    ten parameters still overrides the preset value for that parameter
    only (the other nine keep coming from the preset).
    """

    #: ``preset`` is a constructor convenience that expands to concrete
    #: numeric values -- excluded from ``to_dict()`` (which stores the
    #: *resolved* numbers), matching
    #: :attr:`sensoryforge.neurons.izhikevich.IzhikevichNeuronTorch._TO_DICT_EXCLUDE_PARAMS`.  # noqa: E501
    _TO_DICT_EXCLUDE_PARAMS = frozenset({"preset"})

    #: The ten parameters a preset expands into (see :data:`ADEX_PRESETS`).
    _PRESET_PARAM_NAMES = (
        "EL",
        "VT",
        "DeltaT",
        "tau_m",
        "tau_w",
        "a",
        "b",
        "v_reset",
        "v_spike",
        "R",
        "t_ref",
    )

    def __init__(
        self,
        EL=None,
        VT=None,
        DeltaT=None,
        tau_m=None,
        tau_w=None,
        a=None,
        b=None,
        v_reset=None,
        v_spike=None,
        R=None,
        t_ref=None,
        v_init=None,
        w_init=None,
        dt=0.05,
        noise_std: float = 0.0,
        v_floor: Optional[float] = None,
        *,
        preset: Optional[str] = None,
    ):
        super().__init__()
        if preset is not None and preset not in ADEX_PRESETS:
            raise ValueError(
                f"Unknown AdEx preset {preset!r}; choose one of "
                f"{sorted(ADEX_PRESETS)}"
            )
        self.preset = preset
        preset_values = ADEX_PRESETS[preset] if preset is not None else {}
        explicit = {
            "EL": EL,
            "VT": VT,
            "DeltaT": DeltaT,
            "tau_m": tau_m,
            "tau_w": tau_w,
            "a": a,
            "b": b,
            "v_reset": v_reset,
            "v_spike": v_spike,
            "R": R,
            "t_ref": t_ref,
        }
        for name in self._PRESET_PARAM_NAMES:
            value = explicit[name]
            if value is None:
                value = preset_values.get(name, _ADEX_CLASS_DEFAULTS[name])
            setattr(self, name, value)
        self.dt = dt
        self.v_init = self.EL if v_init is None else v_init
        self.w_init = 0.0 if w_init is None else w_init
        # Langevin noise intensity (additive, mV/sqrt(ms))
        self.noise_std = noise_std
        # Optional voltage floor (mV). None (default, D-ce22df3) integrates
        # freely; a value clamps v from below, a guard against Euler blow-up
        # when integrating strongly negative drive at a coarse step (D-007).
        self.v_floor = v_floor

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless AdEx).

        The AdEx model re-initialises v and w each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter's current value (F-045).

        Returns:
            Dictionary with every ``__init__`` parameter.
        """
        return {
            "EL": self.EL,
            "VT": self.VT,
            "DeltaT": self.DeltaT,
            "tau_m": self.tau_m,
            "tau_w": self.tau_w,
            "a": self.a,
            "b": self.b,
            "v_reset": self.v_reset,
            "v_spike": self.v_spike,
            "R": self.R,
            "t_ref": self.t_ref,
            "v_init": self.v_init,
            "w_init": self.w_init,
            "dt": self.dt,
            "noise_std": self.noise_std,
            "v_floor": self.v_floor,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return parameter specifications for UI auto-generation (F-045)."""
        return [
            ParamSpec(
                "EL",
                dtype="float",
                default=-70.0,
                min_val=-100.0,
                max_val=-40.0,
                step=1.0,
                unit="mV",
                tooltip="Leak reversal potential",
            ),
            ParamSpec(
                "VT",
                dtype="float",
                default=-50.0,
                min_val=-80.0,
                max_val=-20.0,
                step=1.0,
                unit="mV",
                tooltip="Spike threshold (exponential term)",
            ),
            ParamSpec(
                "DeltaT",
                dtype="float",
                default=2.0,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                unit="mV",
                tooltip="Slope factor of the exponential term",
            ),
            ParamSpec(
                "tau_m",
                dtype="float",
                default=20.0,
                min_val=1.0,
                max_val=200.0,
                step=1.0,
                unit="ms",
                tooltip="Membrane time constant",
            ),
            ParamSpec(
                "tau_w",
                dtype="float",
                default=100.0,
                min_val=1.0,
                max_val=500.0,
                step=5.0,
                unit="ms",
                tooltip="Adaptation time constant",
            ),
            ParamSpec(
                "a",
                dtype="float",
                default=2.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Subthreshold adaptation coupling",
            ),
            ParamSpec(
                "b",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Spike-triggered adaptation increment",
            ),
            ParamSpec(
                "v_reset",
                dtype="float",
                default=-58.0,
                min_val=-100.0,
                max_val=0.0,
                step=1.0,
                unit="mV",
                tooltip="Post-spike reset voltage",
            ),
            ParamSpec(
                "v_spike",
                dtype="float",
                default=20.0,
                min_val=-20.0,
                max_val=60.0,
                step=1.0,
                unit="mV",
                tooltip="Spike detection voltage",
            ),
            ParamSpec(
                "R",
                dtype="float",
                default=1.0,
                min_val=0.01,
                max_val=20.0,
                step=0.1,
                unit="",
                tooltip="Membrane resistance",
            ),
            ParamSpec(
                "t_ref",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="ms",
                tooltip="Absolute refractory period",
                help="After a spike the voltage is held at v_reset for this "
                "long while the adaptation current keeps evolving. The SA1/RA1 "
                "presets use 2 ms, which caps the rate near 500 Hz.",
                advanced=True,
            ),
            ParamSpec(
                "v_init",
                dtype="float",
                default=-70.0,
                min_val=-100.0,
                max_val=50.0,
                step=1.0,
                unit="mV",
                tooltip="Initial membrane voltage (default: EL)",
            ),
            ParamSpec(
                "w_init",
                dtype="float",
                default=0.0,
                min_val=-20.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Initial adaptation current",
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
                default=None,
                min_val=-200.0,
                max_val=-50.0,
                step=1.0,
                unit="mV",
                tooltip="Voltage floor (clamp); auto means no clamp",
                help="Unset by default (D-ce22df3). Set it (e.g. -130.0 mV) to "
                "clamp v from below when integrating strongly negative input at "
                "a coarse step.",
                advanced=True,
            ),
        ]

    def _dynamics(self, v, w, input_t):
        """Compute AdEx state derivatives (subthreshold dynamics only).

        Args:
            v: Membrane voltage tensor [batch, features] in mV.
            w: Adaptation current tensor [batch, features].
            input_t: External current at this time step [batch, features] in mA.

        Returns:
            Tuple (dv, dw) of derivative tensors, each [batch, features].
        """
        exp_term = self.DeltaT * torch.exp((v - self.VT) / self.DeltaT)
        dv = (-(v - self.EL) + exp_term - w + self.R * input_t) / self.tau_m
        dw = (self.a * (v - self.EL) - w) / self.tau_w
        return dv, dw

    def forward(self, input_current, solver=None):
        batch, steps, features = input_current.shape
        device, dtype = input_current.device, input_current.dtype
        v = torch.full((batch, features), self.v_init, dtype=dtype, device=device)
        w = torch.full((batch, features), self.w_init, dtype=dtype, device=device)
        v_trace = torch.zeros((batch, steps + 1, features), dtype=dtype, device=device)
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        v_trace[:, 0, :] = v
        # Absolute refractory period (D-f5853a4): steps left during which v is
        # held at v_reset. With t_ref = 0 this stays unused and the update
        # is the historical one, bit for bit.
        ref_steps = int(round(self.t_ref / self.dt)) if self.t_ref else 0
        refractory = (
            torch.zeros((batch, features), dtype=torch.int64, device=device)
            if ref_steps > 0
            else None
        )
        for t in range(steps):
            fired = v >= self.v_spike
            v_vis = v.clone()
            v_vis[fired] = self.v_spike
            v_trace[:, t, :] = v_vis
            spikes[:, t, :] = fired
            v_next = torch.where(fired, torch.full_like(v, self.v_reset), v)
            w_next = torch.where(fired, w + self.b, w)
            not_fired = ~fired
            dv, dw = self._dynamics(v, w, input_current[:, t, :])
            # Add Langevin noise to v integration (not during reset)
            if self.noise_std != 0.0:
                # scale by sqrt(dt) for discrete-time white noise
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            w_next = torch.where(not_fired, w + self.dt * dw, w_next)
            if refractory is not None:
                held = refractory > 0
                v_next = torch.where(held, torch.full_like(v, self.v_reset), v_next)
                refractory = torch.where(
                    fired,
                    torch.full_like(refractory, ref_steps),
                    (refractory - 1).clamp(min=0),
                )
            if self.v_floor is not None:
                v_next = v_next.clamp(min=self.v_floor)
            v = v_next
            w = w_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= self.v_spike
        return v_trace, spikes
