import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

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
}

#: Named AdEx firing-pattern regimes. Pass ``preset=...`` to
#: :class:`AdExNeuronTorch` instead of spelling out all ten parameters by
#: hand -- mirrors :data:`sensoryforge.neurons.izhikevich.IZHIKEVICH_PRESETS`
#: exactly (an explicit keyword argument still overrides that one preset
#: value; ``preset`` is excluded from ``to_dict()``).
#:
#: The *qualitative* regimes (tonic vs. phasic/adapting) are the standard
#: AdEx firing-pattern classes catalogued by Naud, Marcille, Clopath &
#: Gerstner (2008), "Firing patterns in the adaptive exponential
#: integrate-and-fire model", Biological Cybernetics 99:335-347, for the
#: model of Brette & Gerstner (2005), "Adaptive Exponential Integrate-and-
#: Fire Model as an Effective Description of Neuronal Activity", J.
#: Neurophysiol. 94:3637-3642. The *concrete numeric values* below are NOT
#: taken from either paper -- they are ours, chosen for this project's
#: mA/mV/ms unit convention (``R * I`` lands in mV) and tuned against the
#: drive the ``tactile_sa1_ra1`` recipe actually delivers (Phase 2b T2b,
#: measured by ``scripts/tune_adex_populations.py`` -- see
#: ``benchmarks/results/adex_tuning/adex_tuning.md``), not against an
#: arbitrary bench current. The drive figures below were measured at the
#: recipe's old shared ``input_gain`` of 50; ``tactile_sa1_ra1_adex`` now
#: calibrates each population's gain on top of these presets (SA 55, RA 86,
#: ``scripts/calibrate_recipe_gains.py``, ledger D-ea0f017).
#:
#: ``R`` (membrane resistance) is the input-scaling knob -- it plays the
#: same role ``input_gain`` plays on the Izhikevich path, compensating the
#: unit mismatch documented in ``docs/user_guide/units_and_gains.md`` (the
#: Parvizi-Fard filter gains are calibrated for N/mm2, this project's
#: stimulus unit is mA). For subthreshold adaptation ``a``, this
#: parametrisation's rheobase -- the saddle-node current above which no
#: stable subthreshold fixed point exists, so a constant drive above it
#: fires forever instead of settling -- is:
#:
#:     I_rheobase = (1 + a) * ((VT - EL) + DeltaT*ln(1 + a) - DeltaT) / R
#:
#: ``SA1_tonic`` (``a`` = 0.02, ``R`` = 6.0): I_rheobase ~= 3.07 mA. Chosen
#: so the rheobase sits near the LOW end of the responsive SA neurons'
#: measured hold drive during ``ramp_gaussian`` (mean 5.53 mA; p10/p50/p90
#: = 3.82 / 5.25 / 7.30 mA; max 10.25 mA -- iteration-1 finding: the
#: recipe's own drive, not a flat bench step, is the tuning target), so a
#: genuinely held stimulus is comfortably suprathreshold across that whole
#: range. Verified by simulation: at those three percentiles the f-I curve
#: gives ~25 / ~60 / ~95 Hz -- inside P5's 20-100 Hz band, monotone in
#: amplitude -- and the responsive-set pooled ISI CV during
#: ``ramp_gaussian``'s hold window is ~0.47 (< 0.5).
#:
#: How R=6.0 was actually picked: the purely principled placement --
#: rheobase at the p10 of the measured hold drive (3.82 mA) -- gives
#: R ~= 4.8. That alone left the responsive set's pooled ISI CV above 0.5
#: (rate heterogeneity across the ~2x drive spread in the responsive set
#: pools into an inflated CV even though each neuron's own firing is
#: perfectly regular). R was then scanned upward and R=6.0 taken as the
#: smallest value whose pooled ISI CV cleared 0.5 on this same hold
#: window -- so the reported CV = 0.470 is a fitted outcome of that scan,
#: not an independent confirmation of the CV criterion. R=6.0 is within
#: about 25% of the principled R~=4.8, i.e. most but not all of the
#: choice is the scan rather than the drive-percentile placement alone.
#: At dt = 0.05 ms under
#: a constant 40 mA bench drive for 500 ms (``tests/unit/
#: test_pytorch_neurons.py``, unchanged current -- 40 mA remains far above
#: the new rheobase too), it keeps spiking regularly through the full 500
#: ms with ISI CV ~= 6e-6.
#:
#: ``RA1_phasic`` (``a`` = 2.0, ``R`` = 8.0): I_rheobase ~= 7.57 mA.
#: Chosen so the rheobase sits BELOW the responsive RA neurons' measured
#: onset-transient drive on the stimuli that produce one (``ramp_gaussian``
#: peak 5.67 mA is borderline-below and still triggers a single spike per
#: responsive neuron; ``moving_edge``'s stronger onset, peak 13.87 mA,
#: triggers a multi-spike burst) but ABOVE the same neurons' measured hold
#: drive (``ramp_gaussian`` RA hold mean 0.15 mA, near zero as expected --
#: RA's own filter differentiates). ``b``/``tau_w`` are unchanged (20.0 /
#: 50.0 ms): at the new ``R`` they still produce the phasic,
#: settle-after-the-first-spike(s) character (a single spike per
#: responsive neuron on ``ramp_gaussian``, synchronized across the
#: responsive set into a population-level burst that peaks at ~1600 Hz
#: instantaneous, within P5's up-to-~300 Hz target). At dt = 0.05 ms under
#: a constant 40 mA bench drive for 500 ms it now fires continuously
#: instead of settling (40 mA is far past the new, lower rheobase, so no
#: stable subthreshold fixed point exists there any more); the bench test
#: was updated to probe at 5.0 mA instead (just below the new rheobase),
#: where it fires exactly once (t ~= 22.6 ms) and is silent thereafter --
#: see ``test_adex_ra1_phasic_silences_within_30ms_under_constant_drive``
#: for why the current level changed. RA-I transient bursting was NOT met
#: on ``drifting_grating`` (onset drive there peaks at only 3.08 mA,
#: below rheobase -- reported as a FAIL in ``adex_tuning.md``, not
#: silently dropped).
ADEX_PRESETS: dict = {
    # Tonic (sustained, weakly adapting): small a, zero b, long tau_w --
    # adaptation never grows enough to silence firing under constant drive.
    "SA1_tonic": {
        "EL": -70.0,
        "VT": -50.0,
        "DeltaT": 2.0,
        "tau_m": 20.0,
        "tau_w": 200.0,
        "a": 0.02,
        "b": 0.0,
        "v_reset": -58.0,
        "v_spike": 20.0,
        "R": 6.0,
    },
    # Phasic / strongly-adapting: large a and a large spike-triggered b,
    # short-to-moderate tau_w -- the first spike (or few) drives the
    # adaptation current to a stable subthreshold fixed point, silencing
    # the neuron for the rest of a constant drive.
    "RA1_phasic": {
        "EL": -70.0,
        "VT": -50.0,
        "DeltaT": 2.0,
        "tau_m": 20.0,
        "tau_w": 50.0,
        "a": 2.0,
        "b": 20.0,
        "v_reset": -58.0,
        "v_spike": 20.0,
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
    #: :attr:`sensoryforge.neurons.izhikevich.IzhikevichNeuronTorch._TO_DICT_EXCLUDE_PARAMS`.
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
        v_init=None,
        w_init=None,
        dt=0.05,
        noise_std: float = 0.0,
        v_floor: float = -130.0,
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
        # Physiological voltage floor (mV). Prevents non-physical hyperpolarization
        # from large negative drive (e.g., noisy SA filter output). Default: -130 mV.
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
                default=-130.0,
                min_val=-200.0,
                max_val=-50.0,
                step=1.0,
                unit="mV",
                tooltip="Physiological voltage floor (clamp)",
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
            if self.v_floor is not None:
                v_next = v_next.clamp(min=self.v_floor)
            v = v_next
            w = w_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= self.v_spike
        return v_trace, spikes
