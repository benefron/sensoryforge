"""Event encoders: a signed level-crossing unit (RA) and a sigma-delta unit (SA).

Two population models that are not membrane models at all. They turn a
population's drive into events by the two simplest analog-to-event circuits,
and exist so the RA/SA split can be studied as a split of *time scales*
(fast signed change vs. slow absolute level) instead of a split of
biological mechanisms (D-2026-10-01, ``docs_root/DECISIONS.md``).

* :class:`LevelCrossingNeuron` (registered as ``"level_crossing"``): an
  event-camera-style unit. Each neuron keeps a reference level. When its
  input has risen by ``theta`` since the last event it emits an ON event
  (``+1``); when it has fallen by ``theta`` it emits an OFF event (``-1``).
  Every event moves the reference by ``theta`` in the event's direction. The
  sign is carried by the event; nothing downstream has to recover it. Its
  events are **signed counts**, so the engine returns them under the key
  ``"events"`` (never ``"spikes"``) and a bundle stores them in an ``events``
  dataset (see ``docs/user_guide/bundles.md``).
* :class:`SigmaDeltaNeuron` (registered as ``"sigma_delta"``): a non-leaky
  integrate-and-fire unit with subtractive reset (first-order sigma-delta
  modulator). ``u += drive * dt``; while ``u >= theta`` a spike is emitted and
  ``u -= theta``. For a constant drive the rate is exactly
  ``drive / theta`` spikes per ms, with no rheobase, and the quantisation
  error is first-order noise-shaped (pushed to high frequencies).

Hardware (FPGA/ASIC) view: level-crossing = a reference register, a
comparator and a sign bit; sigma-delta = an accumulator, a comparator and a
subtractor. Both are one-cycle-per-sample logic, and the AER address-event
format carries the polarity bit natively.

Both follow the :class:`~sensoryforge.neurons.base.BaseNeuron` contract:
``forward(input_current [batch, steps, N])`` returns
``(state_trace, events)``, both ``[batch, steps+1, N]``, index 0 being the
initial state (no event). The state trace is the reference level
(level-crossing) or the accumulator (sigma-delta), in the drive's units /
drive units x ms respectively. The loop runs over time only; every step is
vectorised over batch and neurons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec

#: Relative tolerance (in units of ``theta``) for a crossing that lands on an
#: exact multiple of ``theta``: without it a float32 ramp whose increment is
#: exactly ``theta`` would sometimes read 0.99999994 thresholds and lag one
#: step.
_CROSSING_EPS = 1e-5

#: Accepted ``initial_reference`` values for :class:`LevelCrossingNeuron`.
INITIAL_REFERENCE_CHOICES = ("zero", "first")


def _refractory_steps(refractory_ms: float, dt: float) -> int:
    """Refractory period in integration steps (0 = none, else >= 1)."""
    if refractory_ms <= 0.0:
        return 0
    return max(1, int(round(refractory_ms / dt)))


class LevelCrossingNeuron(BaseNeuron):
    """Signed level-crossing (send-on-delta) event encoder, for RA.

    Each neuron holds a reference level ``ref``. At every integration step,
    with input ``x``::

        d = x - ref
        k = sign(d) * floor(|d| / theta)       # signed event count
        ref <- ref + k * theta

    so an ON event (``+1``) marks a rise of ``theta`` since the last event and
    an OFF event (``-1``) a fall of ``theta``. A step that moves the input by
    several ``theta`` emits several events at once, as one signed count. The
    running sum of events times ``theta`` therefore tracks the input to
    within ``theta`` (``|x - ref| < theta`` after every step without a
    refractory period): the encoding is invertible up to one quantum.

    With ``refractory_ms > 0`` a neuron emits at most one event per
    ``refractory_ms`` (a comparator with dead time). The reference then moves
    by exactly one ``theta`` per event, so a change faster than
    ``theta / refractory_ms`` is paid out late rather than lost -- the unit is
    slew-rate limited and catches up once the input slows.

    The input is the population's pooled drive after the input gain. For an
    RA population use ``filter_method: none``: the unit differentiates by
    construction, so the drive must **not** be passed through the rectified
    RA derivative filter first (that would differentiate twice and discard
    the OFF half). Its input is never floored at 0 mA
    (``config.defaults.resolve_input_floor``), since the OFF events live in
    the falling half.

    Attributes:
        SIGNED_EVENTS: ``True`` -- the engine labels the output ``"events"``.
    """

    SIGNED_EVENTS = True

    def __init__(
        self,
        dt: float = 0.05,
        theta: float = 1.0,
        refractory_ms: float = 0.0,
        initial_reference: str = "zero",
        noise_std: float = 0.0,
    ) -> None:
        """Build a level-crossing population model.

        Args:
            dt: Integration step, ms.
            theta: Level-crossing threshold (the quantum), in the drive's
                units (mA after gain). Must be > 0.
            refractory_ms: Minimum interval between two events of one neuron,
                ms. ``0`` (default) = none, and several events per step are
                allowed.
            initial_reference: ``"zero"`` (reference starts at 0, so a drive
                already above ``theta`` at t = 0 emits its ON events on the
                first step) or ``"first"`` (reference starts at the first
                input sample, so the unit reports changes only).
            noise_std: Standard deviation of Gaussian noise added to the input
                at every integration step (comparator noise), in the drive's
                units. ``0`` = none. Drawn from the global RNG.

        Raises:
            ValueError: For a non-positive ``theta`` or ``dt``, a negative
                ``refractory_ms`` or ``noise_std``, or an unknown
                ``initial_reference``.
        """
        super().__init__(dt=dt)
        if dt <= 0:
            raise ValueError(f"dt must be > 0 ms, got {dt!r}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta!r}")
        if refractory_ms < 0:
            raise ValueError(f"refractory_ms must be >= 0, got {refractory_ms!r}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std!r}")
        if initial_reference not in INITIAL_REFERENCE_CHOICES:
            raise ValueError(
                f"initial_reference must be one of {INITIAL_REFERENCE_CHOICES}, "
                f"got {initial_reference!r}"
            )
        self.theta = float(theta)
        self.refractory_ms = float(refractory_ms)
        self.initial_reference = initial_reference
        self.noise_std = float(noise_std)

    def forward(
        self, input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a drive as signed level-crossing events.

        Args:
            input_current: Drive ``[batch, steps, N]`` in mA (after gain).

        Returns:
            Tuple of:
                - ``ref_trace``: Reference level ``[batch, steps+1, N]`` in mA;
                  index 0 is the initial reference.
                - ``events``: Signed event counts ``[batch, steps+1, N]``,
                  ``torch.int16``: ``+k`` = k ON events, ``-k`` = k OFF events
                  in that step; index 0 is always 0.

        Raises:
            ValueError: If ``input_current`` is not 3-D.
        """
        if input_current.dim() != 3:
            raise ValueError(
                "Expected 3-D input [batch, steps, N], got shape "
                f"{list(input_current.shape)}"
            )
        batch, steps, n = input_current.shape
        device = input_current.device
        dtype = (
            input_current.dtype
            if input_current.is_floating_point()
            else torch.float32
        )
        x_all = input_current.to(dtype)
        if self.noise_std > 0.0:
            x_all = x_all + torch.randn_like(x_all) * self.noise_std

        if self.initial_reference == "first" and steps > 0:
            ref = x_all[:, 0, :].clone()
        else:
            ref = torch.zeros((batch, n), dtype=dtype, device=device)

        ref_trace = torch.empty((batch, steps + 1, n), dtype=dtype, device=device)
        events = torch.zeros((batch, steps + 1, n), dtype=torch.int16, device=device)
        ref_trace[:, 0, :] = ref

        theta = self.theta
        ref_steps = _refractory_steps(self.refractory_ms, self.dt)
        blocked = (
            torch.zeros((batch, n), dtype=torch.int64, device=device)
            if ref_steps > 0
            else None
        )
        for t in range(steps):
            d = x_all[:, t, :] - ref
            magnitude = torch.floor(d.abs() / theta + _CROSSING_EPS)
            if blocked is None:
                k = torch.sign(d) * magnitude
            else:
                allowed = blocked == 0
                k = torch.sign(d) * ((magnitude >= 1) & allowed).to(dtype)
                fired = k != 0
                blocked = torch.where(
                    fired,
                    torch.full_like(blocked, ref_steps - 1),
                    (blocked - 1).clamp(min=0),
                )
            ref = ref + k * theta
            ref_trace[:, t + 1, :] = ref
            events[:, t + 1, :] = k.to(torch.int16)
        return ref_trace, events

    def reset_state(self) -> None:
        """No state persists between calls; each ``forward`` starts fresh."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter (F-045)."""
        return {
            "dt": self.dt,
            "theta": self.theta,
            "refractory_ms": self.refractory_ms,
            "initial_reference": self.initial_reference,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LevelCrossingNeuron":
        """Build from a :meth:`to_dict`-style dict."""
        return cls(**config)

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Parameter descriptors for the GUI forms (G1)."""
        return [
            ParamSpec(
                "theta",
                dtype="float",
                default=1.0,
                min_val=1e-6,
                max_val=1e6,
                step=0.1,
                unit="mA",
                tooltip="Level-crossing threshold: the input change per event",
                help=(
                    "An ON event is emitted each time the drive has risen by "
                    "theta since the last event, an OFF event each time it "
                    "has fallen by theta; the reference moves by theta per "
                    "event."
                ),
            ),
            ParamSpec(
                "refractory_ms",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1000.0,
                step=0.5,
                unit="ms",
                tooltip="Minimum interval between two events (0 = none)",
            ),
            ParamSpec(
                "initial_reference",
                dtype="str",
                default="zero",
                choices=list(INITIAL_REFERENCE_CHOICES),
                tooltip="Reference at t = 0: 0, or the first input sample",
                advanced=True,
            ),
            ParamSpec(
                "noise_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1e6,
                unit="mA",
                tooltip="Comparator noise added to the input each step",
                advanced=True,
            ),
        ]


class SigmaDeltaNeuron(BaseNeuron):
    """Sigma-delta (non-leaky integrate-and-fire, subtractive reset), for SA.

    Each neuron integrates its drive and fires when the integral reaches
    ``theta``, subtracting ``theta`` rather than resetting to zero::

        u <- u + dt * x               (optionally - dt * u / leak_tau_ms)
        n = floor(u / theta)          (spikes this step, >= 0)
        u <- u - n * theta

    Because no charge is ever discarded, the spike count up to time ``t`` is
    ``floor(integral of x / theta)``: for a constant drive the rate is exactly
    ``x / theta`` spikes per ms (``1000 * x / theta`` Hz) with no rheobase, and
    the count tracks the integral to within one spike. The quantisation error
    is the first difference of a bounded sequence, so its spectrum rises with
    frequency (first-order noise shaping) and a low-pass of the spike train
    recovers a slow drive.

    ``leak_tau_ms`` (default none) makes the accumulator leaky, which adds a
    rheobase ``theta / leak_tau_ms`` and breaks the exact linearity.
    ``refractory_ms`` caps the rate at ``1 / refractory_ms``: at most one spike
    per ``refractory_ms``; while the neuron is refractory its accumulator is
    clipped at ``theta`` (anti-windup), so it saturates cleanly instead of
    bursting when the drive drops.

    Negative drive discharges the accumulator (it can go below zero); a
    tactile SA population's input is floored at 0 mA by default anyway.

    Attributes:
        SIGNED_EVENTS: ``False`` -- ordinary, non-negative spike counts.
    """

    SIGNED_EVENTS = False

    def __init__(
        self,
        dt: float = 0.05,
        theta: float = 100.0,
        leak_tau_ms: Optional[float] = None,
        refractory_ms: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        """Build a sigma-delta population model.

        Args:
            dt: Integration step, ms.
            theta: Charge per spike, in drive units x ms (mA*ms). The rate is
                ``drive / theta`` spikes/ms. Must be > 0. Default 100 gives
                10 Hz per mA.
            leak_tau_ms: Accumulator leak time constant, ms; ``None``
                (default) = no leak, exact linearity.
            refractory_ms: Minimum inter-spike interval, ms; ``0`` = none, and
                several spikes per step are allowed.
            noise_std: Standard deviation of Gaussian noise added to the input
                at every integration step, in the drive's units. ``0`` = none.
                Drawn from the global RNG.

        Raises:
            ValueError: For a non-positive ``theta``, ``dt`` or
                ``leak_tau_ms``, or a negative ``refractory_ms``/``noise_std``.
        """
        super().__init__(dt=dt)
        if dt <= 0:
            raise ValueError(f"dt must be > 0 ms, got {dt!r}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta!r}")
        if leak_tau_ms is not None and leak_tau_ms <= 0:
            raise ValueError(f"leak_tau_ms must be > 0 or None, got {leak_tau_ms!r}")
        if refractory_ms < 0:
            raise ValueError(f"refractory_ms must be >= 0, got {refractory_ms!r}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std!r}")
        self.theta = float(theta)
        self.leak_tau_ms = None if leak_tau_ms is None else float(leak_tau_ms)
        self.refractory_ms = float(refractory_ms)
        self.noise_std = float(noise_std)

    def forward(
        self, input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a drive as sigma-delta spike counts.

        Args:
            input_current: Drive ``[batch, steps, N]`` in mA (after gain).

        Returns:
            Tuple of:
                - ``u_trace``: Accumulator ``[batch, steps+1, N]`` in mA*ms;
                  index 0 is the initial value (0).
                - ``spikes``: Spike counts per step ``[batch, steps+1, N]``,
                  ``torch.int16`` (>= 0); index 0 is always 0.

        Raises:
            ValueError: If ``input_current`` is not 3-D.
        """
        if input_current.dim() != 3:
            raise ValueError(
                "Expected 3-D input [batch, steps, N], got shape "
                f"{list(input_current.shape)}"
            )
        batch, steps, n = input_current.shape
        device = input_current.device
        dtype = (
            input_current.dtype
            if input_current.is_floating_point()
            else torch.float32
        )
        x_all = input_current.to(dtype)
        if self.noise_std > 0.0:
            x_all = x_all + torch.randn_like(x_all) * self.noise_std

        u = torch.zeros((batch, n), dtype=dtype, device=device)
        u_trace = torch.empty((batch, steps + 1, n), dtype=dtype, device=device)
        spikes = torch.zeros((batch, steps + 1, n), dtype=torch.int16, device=device)
        u_trace[:, 0, :] = u

        dt = self.dt
        theta = self.theta
        decay = None if self.leak_tau_ms is None else dt / self.leak_tau_ms
        ref_steps = _refractory_steps(self.refractory_ms, dt)
        blocked = (
            torch.zeros((batch, n), dtype=torch.int64, device=device)
            if ref_steps > 0
            else None
        )
        for t in range(steps):
            if decay is None:
                u = u + dt * x_all[:, t, :]
            else:
                u = u + dt * x_all[:, t, :] - decay * u
            count = torch.floor(u / theta + _CROSSING_EPS).clamp(min=0)
            if blocked is not None:
                allowed = blocked == 0
                count = ((count >= 1) & allowed).to(dtype)
                blocked = torch.where(
                    count > 0,
                    torch.full_like(blocked, ref_steps - 1),
                    (blocked - 1).clamp(min=0),
                )
            u = u - count * theta
            if blocked is not None:
                u = u.clamp(max=theta)
            u_trace[:, t + 1, :] = u
            spikes[:, t + 1, :] = count.to(torch.int16)
        return u_trace, spikes

    def reset_state(self) -> None:
        """No state persists between calls; each ``forward`` starts fresh."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter (F-045)."""
        return {
            "dt": self.dt,
            "theta": self.theta,
            "leak_tau_ms": self.leak_tau_ms,
            "refractory_ms": self.refractory_ms,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SigmaDeltaNeuron":
        """Build from a :meth:`to_dict`-style dict."""
        return cls(**config)

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Parameter descriptors for the GUI forms (G1)."""
        return [
            ParamSpec(
                "theta",
                dtype="float",
                default=100.0,
                min_val=1e-6,
                max_val=1e9,
                step=1.0,
                unit="mA*ms",
                tooltip="Charge per spike; rate = drive / theta spikes per ms",
            ),
            ParamSpec(
                "leak_tau_ms",
                dtype="float",
                default=None,
                min_val=1e-3,
                max_val=1e6,
                unit="ms",
                tooltip="Accumulator leak (unset = none, exact linearity)",
                advanced=True,
            ),
            ParamSpec(
                "refractory_ms",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1000.0,
                step=0.5,
                unit="ms",
                tooltip="Minimum inter-spike interval (0 = none)",
            ),
            ParamSpec(
                "noise_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=1e6,
                unit="mA",
                tooltip="Noise added to the input each step",
                advanced=True,
            ),
        ]


__all__ = [
    "LevelCrossingNeuron",
    "SigmaDeltaNeuron",
    "INITIAL_REFERENCE_CHOICES",
]
