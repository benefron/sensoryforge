"""Contact-episode timing and modulation, shared by layered stimuli and worlds.

One *contact* is a ramp up (``up``), a still ``hold``, a moving ``slide`` and
a ramp down (``down``). An episode is ``contacts`` of them after ``onset``,
``pause`` apart. Every function is pure torch arithmetic on tensors that
broadcast against the time tensor ``t`` (ms), so the same code serves one
layer (0-d parameters) and a batch of world draws (``[g, 1]`` parameters).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _safe(x: torch.Tensor) -> torch.Tensor:
    """``x`` where positive, else 1: a divisor that never divides by zero."""
    return torch.where(x > 0, x, torch.ones_like(x))


def contact_terms(
    t: torch.Tensor,
    onset: torch.Tensor,
    up: torch.Tensor,
    hold: torch.Tensor,
    slide: torch.Tensor,
    down: torch.Tensor,
    contacts: torch.Tensor,
    pause: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The contact envelope and clock at times ``t``.

    Args:
        t: Times in ms, any shape.
        onset: Quiet lead-in before the first touch, ms.
        up: Ramp up (touch), ms.
        hold: Still contact at full amplitude, ms.
        slide: Moving contact at full amplitude, ms.
        down: Ramp down (release), ms.
        contacts: Number of contacts (a float tensor holding an integer).
        pause: Lift between contacts, ms.

    Returns:
        ``(envelope, tau, k, local)``, each broadcast to ``t``: the envelope
        in ``[0, 1]`` (exactly 0 before ``onset``, in pauses and after the
        last contact); ``tau``, ms since the current contact's touch began;
        ``k``, the contact's index; ``local = t - onset``.
    """
    local = t - onset
    cycle = up + hold + slide + down
    period = cycle + pause
    k = torch.floor(local / _safe(period))
    tau = local - k * period
    plateau_end = up + hold + slide
    env = torch.zeros_like(tau)
    env = torch.where((tau >= 0) & (tau < up), tau / _safe(up), env)
    env = torch.where((tau >= up) & (tau < plateau_end), torch.ones_like(env), env)
    env = torch.where(
        (tau >= plateau_end) & (tau < cycle),
        1.0 - (tau - plateau_end) / _safe(down),
        env,
    )
    active = (local >= 0) & (k < contacts)
    env = torch.where(active, env, torch.zeros_like(env))
    return env.clamp(0.0, 1.0), tau, k, local


def span_progress(
    tau: torch.Tensor,
    k: torch.Tensor,
    local: torch.Tensor,
    contacts: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    """Motion progress in ``[0, 1]``, spread over all contacts.

    Within each contact the motion runs over ``[start, start + length)`` of
    the contact's clock; contact ``k`` covers ``[k, k + 1] / contacts`` of the
    path, so a re-touch lands where the previous contact ended.

    Args:
        tau: Ms since the current contact's touch (from :func:`contact_terms`).
        k: Contact index (from :func:`contact_terms`).
        local: ``t - onset`` (from :func:`contact_terms`).
        contacts: Number of contacts.
        start: Start of the moving span within a contact, ms.
        length: Length of the moving span, ms; 0 means no motion.

    Returns:
        Progress, broadcast to ``tau``: 0 before the first contact, 1 after the
        last, 0 everywhere when ``length`` is 0.
    """
    within = ((tau - start) / _safe(length)).clamp(0.0, 1.0)
    progress = ((k + within) / contacts).clamp(0.0, 1.0)
    progress = torch.where(local < 0, torch.zeros_like(progress), progress)
    return torch.where(length > 0, progress, torch.zeros_like(progress))


def sine_modulation(
    tc: torch.Tensor,
    frequency_hz: torch.Tensor,
    depth: torch.Tensor,
    phase_deg: torch.Tensor,
) -> torch.Tensor:
    """Vibration: ``1 - depth * (1 - cos(2 pi f tc + phase)) / 2``, in ``[0, 1]``.

    Args:
        tc: Ms since the contact's touch.
        frequency_hz: Vibration frequency, Hz.
        depth: 0 (none) to 1 (from zero to peak).
        phase_deg: Phase at the touch, degrees (0: at peak).

    Returns:
        The modulation factor, broadcast to ``tc``.
    """
    angle = 2.0 * math.pi * frequency_hz * tc / 1000.0 + phase_deg * (math.pi / 180.0)
    return 1.0 - depth * (1.0 - torch.cos(angle)) / 2.0


def pulse_modulation(
    tc: torch.Tensor,
    rate_hz: torch.Tensor,
    duty: torch.Tensor,
    edge_ms: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Repeated indentation: a train of taps, ``1 - depth * (1 - pulse)``.

    Each period ``P = 1000 / rate_hz`` ms the pulse rises linearly over
    ``edge`` from 0, holds 1 until ``duty * P``, falls linearly over ``edge``,
    then stays 0. ``edge`` is clamped to ``min(edge_ms, duty P, (1 - duty) P)``.

    Args:
        tc: Ms since the contact's touch.
        rate_hz: Taps per second.
        duty: Fraction of each period pressed, in ``(0, 1)``.
        edge_ms: Rise and fall time of each tap, ms (0: a step).
        depth: 0 (no taps) to 1 (lift fully between taps).

    Returns:
        The modulation factor in ``[0, 1]``, broadcast to ``tc``.
    """
    period = 1000.0 / rate_hz
    on = duty * period
    edge = torch.minimum(torch.minimum(edge_ms, on), period - on)
    phase = torch.remainder(tc, period)
    rise = torch.where(edge > 0, phase / _safe(edge), torch.ones_like(phase))
    fall = torch.where(
        edge > 0, 1.0 - (phase - on) / _safe(edge), torch.zeros_like(phase)
    )
    pulse = torch.where(
        phase < on,
        rise.clamp(max=1.0),
        torch.where(phase < on + edge, fall, torch.zeros_like(phase)),
    )
    return 1.0 - depth * (1.0 - pulse.clamp(0.0, 1.0))
