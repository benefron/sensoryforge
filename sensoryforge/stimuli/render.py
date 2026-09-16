"""One stimulus renderer, dispatching through ``STIMULUS_REGISTRY`` (Phase 2, Wave K, F-052).

Before this module, ``GeneralizedTactileEncodingPipeline.generate_stimulus`` (the
function every runner called, including the CLI for canonical configs) dispatched
stimulus names through a hard-coded ``if``/``elif`` chain that knew nine names.
``STIMULUS_REGISTRY`` held (and holds) more -- ``composite``, ``edge_grating``,
``gabor`` were registered but unreachable from a config file, and a third-party
stimulus plugin could be registered but never executed (F-052).

:func:`render_stimulus` is the single entry point that fixes this: it looks a
stimulus type up in ``STIMULUS_REGISTRY`` first (so any registered or plugin
stimulus is runnable from a config file) and only falls back to the legacy
pipeline's chain for the handful of names that were never registered as
components (``trapezoidal``, ``step``, ``ramp``, ``custom``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from sensoryforge.registry import STIMULUS_REGISTRY


def _temporal_envelope(
    time_ms: torch.Tensor,
    *,
    ramp_up_ms: float,
    plateau_ms: float,
    ramp_down_ms: float,
    amplitude: float,
) -> torch.Tensor:
    """Pressure-simulation's ramp/plateau/ramp envelope (``encode_runner.py:45-64``).

    Args:
        time_ms: Time axis ``[T]`` in ms.
        ramp_up_ms: Linear rise duration in ms.
        plateau_ms: Hold-at-1.0 duration in ms.
        ramp_down_ms: Linear fall duration in ms.
        amplitude: Peak scale applied to the [0, 1] envelope.

    Returns:
        Envelope values ``[T]`` in ``[0, amplitude]``.
    """
    ramp_up_ms = max(float(ramp_up_ms), 0.0)
    plateau_ms = max(float(plateau_ms), 0.0)
    ramp_down_ms = max(float(ramp_down_ms), 0.0)
    down_start = ramp_up_ms + plateau_ms
    total_dur = ramp_up_ms + plateau_ms + ramp_down_ms

    amp = torch.zeros_like(time_ms)
    if ramp_up_ms > 0:
        up_mask = time_ms < ramp_up_ms
        amp[up_mask] = time_ms[up_mask] / ramp_up_ms
    else:
        amp[time_ms < ramp_up_ms + 1e-6] = 1.0
    amp[(time_ms >= ramp_up_ms) & (time_ms < down_start)] = 1.0
    if ramp_down_ms > 0:
        down_mask = (time_ms >= down_start) & (time_ms <= down_start + ramp_down_ms)
        amp[down_mask] = torch.clamp(
            1.0 - (time_ms[down_mask] - down_start) / ramp_down_ms, 0.0, 1.0
        )
    amp = torch.where(time_ms > total_dur, torch.zeros_like(amp), amp)
    return amp.clamp(0.0, 1.0) * amplitude


def render_stimulus(
    stimulus_type: str,
    params: Dict[str, Any],
    xx: torch.Tensor,
    yy: torch.Tensor,
    dt_ms: float,
    duration_ms: Optional[float] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a named stimulus to ``[T, H, W]`` frames on a fixed time axis.

    Args:
        stimulus_type: A name in ``STIMULUS_REGISTRY`` (any built-in or
            plugin-registered stimulus class), or one of the legacy names
            ``GeneralizedTactileEncodingPipeline.generate_stimulus`` still
            knows (``trapezoidal``, ``step``, ``ramp``, ``custom``).
        params: Stimulus-specific parameters, passed to
            ``cls.from_config(params)`` for a registered stimulus, or as
            ``**stimulus_params`` to the legacy pipeline's generator.
        xx: X-coordinate meshgrid ``[H, W]`` in mm.
        yy: Y-coordinate meshgrid ``[H, W]`` in mm.
        dt_ms: Time step in ms, used to build the time axis for a registered
            stimulus (the legacy path builds its own).
        duration_ms: Total duration in ms. For a registered stimulus this
            sets ``T = len(time_ms)`` via the half-step-guarded ``arange``
            below; a stimulus that itself returns ``[T, H, W]`` is truncated
            or zero-padded to that length. ``None`` lets the stimulus (or,
            for a single-frame stimulus, the temporal-envelope parameters)
            determine the length.
        device: Torch device string for the returned tensors.

    Returns:
        ``(frames, time_ms)``: frames ``[T, H, W]`` float32, time_ms ``[T]``.

    Raises:
        ValueError: If ``stimulus_type`` is not registered and not one of
            the legacy pipeline's names, listing the registered names.
    """
    if STIMULUS_REGISTRY.is_registered(stimulus_type):
        return _render_registered(
            stimulus_type, params, xx, yy, dt_ms, duration_ms, device
        )

    # Fall back to the legacy pipeline's chain for names that are not (and
    # may never be) registered components: trapezoidal, step, ramp, custom.
    legacy_names = {"trapezoidal", "step", "ramp", "custom"}
    if stimulus_type not in legacy_names:
        raise ValueError(
            f"Unknown stimulus type {stimulus_type!r}. Registered stimuli: "
            f"{STIMULUS_REGISTRY.list_registered()}. Legacy pipeline names: "
            f"{sorted(legacy_names)}."
        )
    return _render_legacy(stimulus_type, params, xx, yy, dt_ms, duration_ms, device)


def _time_axis(dt_ms: float, duration_ms: float, device: str) -> torch.Tensor:
    """Pressure-simulation's half-step-guarded time axis (Fact K-a/K1)."""
    return torch.arange(
        0.0, float(duration_ms) + 0.5 * float(dt_ms), float(dt_ms), device=device
    )


def _render_registered(
    stimulus_type: str,
    params: Dict[str, Any],
    xx: torch.Tensor,
    yy: torch.Tensor,
    dt_ms: float,
    duration_ms: Optional[float],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cls = STIMULUS_REGISTRY.get_class(stimulus_type)

    # The temporal-envelope keys (ramp_up_ms/plateau_ms/ramp_down_ms/
    # total_ms) are render_stimulus's own vocabulary for expanding a
    # single-frame ([H, W]) stimulus over time -- not every BaseStimulus
    # subclass accepts them as constructor parameters, but a [T, H, W]
    # stimulus (like the K2 ported ones) legitimately does. Try the full
    # params first (so a [T, H, W] stimulus's own total_ms/ramp_up_ms/etc.
    # reach its constructor); only strip the envelope keys and retry if
    # construction rejects one of them.
    _ENVELOPE_KEYS = {"ramp_up_ms", "plateau_ms", "ramp_down_ms", "total_ms"}
    try:
        instance = cls.from_config(dict(params))
    except TypeError as exc:
        if not _ENVELOPE_KEYS & set(params):
            raise
        stripped = {k: v for k, v in params.items() if k not in _ENVELOPE_KEYS}
        try:
            instance = cls.from_config(stripped)
        except TypeError:
            raise exc
    xx = xx.to(device)
    yy = yy.to(device)
    frame = instance(xx, yy)

    if frame.dim() == 2:
        # Single-frame stimulus: expand with the temporal envelope.
        envelope_ms = (
            duration_ms
            if duration_ms is not None
            else float(
                params.get(
                    "total_ms",
                    params.get("ramp_up_ms", 0.0)
                    + params.get("plateau_ms", 0.0)
                    + params.get("ramp_down_ms", 0.0),
                )
                or 1.0
            )
        )
        time_ms = _time_axis(dt_ms, envelope_ms, device)
        amp = _temporal_envelope(
            time_ms,
            ramp_up_ms=params.get("ramp_up_ms", 0.0),
            plateau_ms=params.get("plateau_ms", envelope_ms),
            ramp_down_ms=params.get("ramp_down_ms", 0.0),
            amplitude=1.0,
        )
        frames = frame.unsqueeze(0) * amp.view(-1, 1, 1)
        return frames, time_ms

    if frame.dim() == 3:
        t = frame.shape[0]
        if duration_ms is not None:
            time_ms = _time_axis(dt_ms, duration_ms, device)
            target_t = time_ms.numel()
            if t > target_t:
                frame = frame[:target_t]
            elif t < target_t:
                pad = torch.zeros(
                    target_t - t, *frame.shape[1:], dtype=frame.dtype, device=device
                )
                frame = torch.cat([frame, pad], dim=0)
        else:
            time_ms = torch.arange(t, dtype=torch.float32, device=device) * float(dt_ms)
        return frame, time_ms

    raise ValueError(
        f"{cls.__name__}.forward() must return a [H, W] or [T, H, W] tensor, "
        f"got shape {list(frame.shape)}"
    )


def _render_legacy(
    stimulus_type: str,
    params: Dict[str, Any],
    xx: torch.Tensor,
    yy: torch.Tensor,
    dt_ms: float,
    duration_ms: Optional[float],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Imported lazily to avoid a hard import cycle (generalized_pipeline
    # imports register_components, which this module is part of).
    from sensoryforge.core.generalized_pipeline import (
        GeneralizedTactileEncodingPipeline,
    )

    rows, cols = xx.shape
    spacing = float(xx[1, 0] - xx[0, 0]) if rows > 1 else 0.15
    center_x = float(xx[:, 0].mean() + xx[:, -1].mean()) / 2.0 if rows > 0 else 0.0
    center_y = float(yy[0, :].mean() + yy[-1, :].mean()) / 2.0 if cols > 0 else 0.0
    legacy_config: Dict[str, Any] = {
        "pipeline": {
            "device": device,
            "grid_size": (rows, cols),
            "spacing": spacing,
            "center": [center_x, center_y],
        },
        "neurons": {"dt": dt_ms},
        "temporal": {"dt": dt_ms},
        "simulation": {"dt_ms": dt_ms},
    }
    pipeline = GeneralizedTactileEncodingPipeline.from_config(legacy_config)
    stim_params = dict(params)
    if duration_ms is not None:
        stim_params.setdefault("duration", duration_ms)
    frames, time_ms, _ = pipeline.generate_stimulus(
        stimulus_type=stimulus_type, **stim_params
    )
    frames = frames.squeeze(0).to(device)
    time_ms = time_ms.to(device)
    return frames, time_ms
