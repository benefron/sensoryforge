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

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from sensoryforge.registry import STIMULUS_REGISTRY

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.stimuli.canvas import StimulusCanvas

# ---------------------------------------------------------------------------
# Legacy-default compatibility (K8).
#
# Five registered names also exist in GeneralizedTactileEncodingPipeline's
# own hard-coded chain: gaussian, moving, repeated_pattern, texture,
# timeline. Before K1, every one of these went through that legacy chain
# only, whose generators (_generate_gaussian_stimulus etc.,
# core/generalized_pipeline.py) have their OWN default parameter values,
# independent of the registered class's own constructor defaults (which
# exist for direct, non-CLI use and are not changed here -- other callers
# rely on them, e.g. GaussianStimulus's amplitude=1.0/sigma=0.2 default is
# correct for a hand-built config that means to use a narrow, unit-peak
# blob). Since K1 routes these five through STIMULUS_REGISTRY first, a
# config that omits a parameter the legacy generator defaulted now silently
# got the *registered class's* default instead -- for "gaussian" specifically,
# amplitude 1.0 vs. the legacy 30.0 and sigma 0.2 mm vs. 1.0 mm, a stimulus
# 20x narrower and 25x weaker for the same config text (measured on a
# 40x40/0.15mm grid, amplitude=10, no sigma: 32 receptors above 10% of peak
# and frame energy 111.7 with the registered default, vs. 648 receptors and
# 2777.6 energy with the legacy one).
#
# _LEGACY_DEFAULTS applies each name's own legacy default values into
# params before construction, but only for keys the caller did not already
# supply -- so a config with an explicit amplitude/sigma is unaffected, and
# the *registered class's own* default (used when a name is NOT in this
# map, or accessed directly rather than through render_stimulus) is never
# touched.
#
# Three of the five (moving, repeated_pattern, timeline) wrap an arbitrary
# constituent BaseStimulus (base_stimulus/sub_stimuli) that has no sensible
# bare default -- the legacy generator does not wrap a component at all, it
# calls a formula function (gaussian_pressure_torch) directly, so there is
# no single value to default a missing key to for those three. The default
# values here reconstruct the *equivalent* nested config that produces the
# legacy generator's own default probe (a StaticStimulus wrapping the same
# gaussian_pressure_torch formula with the legacy's default amplitude/
# sigma/center), which is sufficient for repeated_pattern (a static,
# state-free tiling of that probe). It is not sufficient for moving or
# timeline: MovingStimulus.forward()/TimelineStimulus.forward() each
# return exactly ONE frame per call, advanced by a separate .step() the
# generic single-frame envelope-expansion path below never calls -- so
# render_stimulus("moving", ...) currently returns the SAME static frame
# repeated across every time sample, not a moving one, regardless of any
# default map. This is a distinct, more serious bug than default drift
# (Finding, see the K8 commit trailer) and is not fixed by this map; the
# "moving" and "timeline" entries below make construction succeed with the
# legacy-equivalent starting frame, but the comparison test for both
# documents (with numbers) that later frames diverge, rather than
# asserting a false bit-identical claim.
_LEGACY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gaussian": {
        "amplitude": 30.0,
        "sigma": 1.0,
        "center_x": 0.0,
        "center_y": 0.0,
    },
    "texture": {
        "amplitude": 30.0,
        "wavelength": 2.0,
        "orientation": 0.0,
        "phase": 0.0,
        "sigma": 2.0,
        "center_x": 0.0,
        "center_y": 0.0,
    },
    "repeated_pattern": {
        "base_stimulus": {
            "class": "StaticStimulus",
            "stim_type": "gaussian",
            "params": {
                "amplitude": 30.0,
                "sigma": 0.5,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        },
        "copies_x": 3,
        "copies_y": 2,
        "spacing_x": 0.5,
        "spacing_y": 0.5,
    },
    "moving": {
        "base_stimulus": {
            "class": "StaticStimulus",
            "stim_type": "gaussian",
            "params": {
                "amplitude": 30.0,
                "sigma": 1.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        },
        "motion_type": "linear",
        "motion_params": {"start": (-2.0, 0.0), "end": (2.0, 0.0)},
    },
}


def _apply_legacy_defaults(
    stimulus_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill in legacy-generator defaults for keys *params* does not set.

    Args:
        stimulus_type: The registered name.
        params: Caller-supplied parameters (not mutated).

    Returns:
        A new dict: *params* with any missing legacy-default key added.
    """
    defaults = _LEGACY_DEFAULTS.get(stimulus_type)
    if not defaults:
        return params
    merged = dict(params)
    for key, value in defaults.items():
        merged.setdefault(key, value)
    return merged


def effective_defaults(stimulus_type: str) -> Dict[str, Any]:
    """The value each parameter of ``stimulus_type`` takes when it is not set.

    This is the one answer to "what will run if I leave this alone?", and what
    a form should show for an unset parameter. It is the type's declared
    defaults (``get_param_spec()``) overlaid with the legacy-generator
    defaults :func:`render_stimulus` fills in for the names both paths know --
    so a Gaussian reports amplitude 30, which is what is rendered, not the
    constructor's 1.0.

    Args:
        stimulus_type: A registered stimulus name (case-insensitive).

    Returns:
        ``{parameter name: default}``; nested legacy entries (``moving``'s
        ``base_stimulus``) are included as they are.

    Raises:
        KeyError: If ``stimulus_type`` is not registered.
    """
    defaults: Dict[str, Any] = {
        spec.name: spec.default
        for spec in STIMULUS_REGISTRY.get_param_spec(stimulus_type)
    }
    defaults.update(_LEGACY_DEFAULTS.get(stimulus_type.lower(), {}))
    return defaults


# Names whose registered component takes a different parameter vocabulary
# from the legacy generator it replaced, mapped to the keys that say the
# caller is speaking the component's vocabulary rather than the legacy one.
#
# `moving` is the case (F-057). The legacy generator took a flat
# amplitude/sigma/start/end/motion_type; the registered MovingStimulus takes
# a nested base_stimulus{...} plus motion_params{...}. A defaults map cannot
# bridge that, because the values have to be *translated*, not filled in, so
# a caller using the legacy vocabulary is routed to the legacy generator and
# gets exactly the frames they got before. A caller who passes the
# component's own keys gets the component, driven properly by
# _render_stepped. Writing that translation is the real fix and is tracked,
# not attempted here.
_PREFER_LEGACY_WITHOUT: Dict[str, frozenset] = {
    "moving": frozenset({"base_stimulus", "motion_params"}),
}


def _prefers_legacy(stimulus_type: str, params: Dict[str, Any]) -> bool:
    """Whether to route a registered name to the legacy generator instead.

    Args:
        stimulus_type: The registered name.
        params: Caller-supplied parameters, before any defaults are applied.

    Returns:
        ``True`` when this name needs translation the registry path cannot
        do and the caller supplied none of the component's own keys.
    """
    signals = _PREFER_LEGACY_WITHOUT.get(stimulus_type)
    if signals is None:
        return False
    return not (signals & set(params))


def _scale_trajectory_to_duration(
    params: Dict[str, Any],
    dt_ms: float,
    duration_ms: Optional[float],
) -> Dict[str, Any]:
    """Make a motion trajectory span the requested duration (F-057).

    The legacy generator built its trajectory with exactly
    ``duration / dt`` steps, so the blob traversed the whole path in the
    time asked for. The registered ``MovingStimulus`` instead defaults to a
    fixed 100-step trajectory, so at any other duration it covers the wrong
    fraction of the path -- half of it at 50 ms and dt 1 ms, for instance.

    Injecting ``num_steps`` keeps the two paths agreeing. A caller who sets
    ``num_steps`` explicitly means it, and is left alone.

    Args:
        params: Parameters after legacy defaults (not mutated).
        dt_ms: Record step, ms.
        duration_ms: Requested duration, or ``None`` to leave the
            trajectory at whatever length it declares.

    Returns:
        A new dict, with ``motion_params["num_steps"]`` set when it applies.
    """
    if duration_ms is None:
        return params
    motion_params = params.get("motion_params")
    if not isinstance(motion_params, dict) or "num_steps" in motion_params:
        return params
    n_steps = max(int(round(float(duration_ms) / float(dt_ms))), 1)
    merged = dict(params)
    merged["motion_params"] = {**motion_params, "num_steps": n_steps}
    return merged


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
    channels: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a named stimulus to frames on a fixed time axis.

    Args:
        stimulus_type: A name in ``STIMULUS_REGISTRY`` (any built-in or
            plugin-registered stimulus class), or one of the legacy names
            ``GeneralizedTactileEncodingPipeline.generate_stimulus`` still
            knows (``trapezoidal``, ``step``, ``ramp``, ``custom``).
        params: Stimulus-specific parameters, passed to
            ``cls.from_config(params)`` for a registered stimulus, or as
            ``**stimulus_params`` to the legacy pipeline's generator. May
            include ``"channel"`` (from ``StimulusConfig.channel``, Phase 2
            Wave L), naming which of *channels* this stimulus fills; that
            key is stripped before construction, never passed to the
            stimulus class itself.
        xx: X-coordinate meshgrid ``[H, W]`` in mm.
        yy: Y-coordinate meshgrid ``[H, W]`` in mm.
        dt_ms: Time step in ms, used to build the time axis for a registered
            stimulus (the legacy path builds its own).
        duration_ms: Total duration in ms, meaning a *duration*: for a
            registered stimulus this sets ``T = round(duration_ms / dt_ms)``
            frames on ``time_ms = arange(T) * dt_ms`` (K9) -- the same
            convention the legacy pipeline and ``--duration`` use, so a
            registered and a legacy stimulus given the same duration/dt_ms
            produce the same frame count. A stimulus that itself returns
            ``[T, H, W]`` is truncated or zero-padded to that length. When
            ``duration_ms`` is ``None``, a single-frame stimulus instead
            falls back to its own ``total_ms``/``ramp_up_ms``/``plateau_ms``/
            ``ramp_down_ms`` envelope parameters -- there, ``total_ms`` is a
            *last-sample* field (pressure-simulation's own convention,
            ``encode_runner.py``, Fact K-a) and keeps the half-step-guarded
            ``arange(0, total_ms + 0.5*dt_ms, dt_ms)`` axis. These are two
            different fields with two different conventions; do not conflate
            them.
        device: Torch device string for the returned tensors.
        channels: The target grid's channel names (``GridConfig.channels``,
            Phase 2 Wave L), or ``None``/a single name for the ordinary
            single-channel case. With more than one channel, the rendered
            stimulus fills the plane named by ``params["channel"]``
            (default: ``channels[0]``) of a ``[T, C, H, W]`` tensor whose
            other planes are zero -- composing several stimulus configs
            with different ``channel`` values (each its own
            :func:`render_stimulus` call) into one multi-channel tensor is
            the caller's job (they are summed elementwise: each call's
              non-target planes are already zero). See
            ``docs/concepts/units_and_shapes.md``.

    Returns:
        ``(frames, time_ms)``: frames ``[T, H, W]`` float32 when *channels*
        has at most one entry, else ``[T, C, H, W]``; ``time_ms`` is
        ``[T]`` either way.

    Raises:
        ValueError: If ``stimulus_type`` is not registered and not one of
            the legacy pipeline's names, listing the registered names; or
            if ``params["channel"]`` names a channel not in *channels*.
    """
    params = dict(params)
    channel_name = params.pop("channel", None)

    if STIMULUS_REGISTRY.is_registered(stimulus_type) and not _prefers_legacy(
        stimulus_type, params
    ):
        frames, time_ms = _render_registered(
            stimulus_type, params, xx, yy, dt_ms, duration_ms, device
        )
    else:
        # Fall back to the legacy pipeline's chain for names that are not
        # (and may never be) registered components: trapezoidal, step,
        # ramp, custom.
        legacy_names = {"trapezoidal", "step", "ramp", "custom"}
        # Names routed here deliberately by _prefers_legacy are valid too.
        if stimulus_type not in legacy_names and stimulus_type not in (
            _PREFER_LEGACY_WITHOUT
        ):
            raise ValueError(
                f"Unknown stimulus type {stimulus_type!r}. Registered "
                f"stimuli: {STIMULUS_REGISTRY.list_registered()}. Legacy "
                f"pipeline names: {sorted(legacy_names)}."
            )
        frames, time_ms = _render_legacy(
            stimulus_type, params, xx, yy, dt_ms, duration_ms, device
        )

    if channels is None or len(channels) <= 1:
        return frames, time_ms

    target = channel_name if channel_name is not None else channels[0]
    if target not in channels:
        raise ValueError(
            f"render_stimulus: channel {target!r} is not one of {channels}"
        )
    index = channels.index(target)
    multi = torch.zeros(
        frames.shape[0],
        len(channels),
        *frames.shape[1:],
        dtype=frames.dtype,
        device=device,
    )
    multi[:, index] = frames
    return multi, time_ms


def _time_axis(dt_ms: float, total_ms: float, device: str) -> torch.Tensor:
    """Pressure-simulation's half-step-guarded, *last-sample* time axis.

    Use only when the field being consumed is a last-sample field like
    pressure-simulation's own ``total_ms`` (``encode_runner.py:45-64``,
    Fact K-a) -- **not** for ``render_stimulus``'s own ``duration_ms``
    argument, which means a duration (K9); see :func:`_duration_axis`.
    """
    return torch.arange(
        0.0, float(total_ms) + 0.5 * float(dt_ms), float(dt_ms), device=device
    )


def _duration_axis(dt_ms: float, duration_ms: float, device: str) -> torch.Tensor:
    """``round(duration_ms / dt_ms)`` frames on ``arange(n) * dt_ms`` (K9).

    This is ``render_stimulus``'s own ``duration_ms`` convention -- a
    duration, not a last-sample field -- matching the legacy pipeline's
    ``n_timesteps = int(duration / dt)`` and the CLI's ``--duration``. It is
    deliberately *not* the half-step-guarded axis :func:`_time_axis` uses
    for a ``total_ms`` last-sample field: the two fields mean different
    things and must not share a convention (see the K9 finding in the Wave
    K report -- an earlier version of this module conflated them, giving a
    registered stimulus one more frame than a legacy one for the same
    duration/dt_ms).
    """
    n = max(int(round(float(duration_ms) / float(dt_ms))), 0)
    return torch.arange(n, dtype=torch.float32, device=device) * float(dt_ms)


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
    params = _apply_legacy_defaults(stimulus_type, params)
    params = _scale_trajectory_to_duration(params, dt_ms, duration_ms)

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

    if frame.dim() == 2 and _is_stepped(instance):
        # Stepped stimulus (F-057): forward() returns the frame at the
        # instance's *current* step and step() advances it, so calling
        # forward() once and broadcasting it over time yields a stimulus
        # that never moves. Iterate instead.
        #
        # This is why `moving` has to be handled here rather than by a
        # defaults map: the registered MovingStimulus
        # (stimuli/builder.py, not the same-named class in stimuli/moving.py,
        # which returns a whole [T, H, W] sequence) is stateful by design.
        return _render_stepped(instance, xx, yy, dt_ms, duration_ms, device, params)

    if frame.dim() == 2:
        # Single-frame stimulus: expand with the temporal envelope.
        if duration_ms is not None:
            # Caller-specified duration (K9): a duration, not a last-sample
            # field -- round(duration_ms / dt_ms) frames, no half-step guard.
            time_ms = _duration_axis(dt_ms, duration_ms, device)
            plateau_default = float(duration_ms)
        else:
            # No duration_ms given: fall back to the stimulus's own envelope
            # parameters. total_ms there is pressure-simulation's own
            # last-sample field (Fact K-a) -- keep the half-step guard.
            envelope_ms = float(
                params.get(
                    "total_ms",
                    params.get("ramp_up_ms", 0.0)
                    + params.get("plateau_ms", 0.0)
                    + params.get("ramp_down_ms", 0.0),
                )
                or 1.0
            )
            time_ms = _time_axis(dt_ms, envelope_ms, device)
            plateau_default = envelope_ms
        amp = _temporal_envelope(
            time_ms,
            ramp_up_ms=params.get("ramp_up_ms", 0.0),
            plateau_ms=params.get("plateau_ms", plateau_default),
            ramp_down_ms=params.get("ramp_down_ms", 0.0),
            amplitude=1.0,
        )
        frames = frame.unsqueeze(0) * amp.view(-1, 1, 1)
        return frames, time_ms

    if frame.dim() == 3:
        t = frame.shape[0]
        if duration_ms is not None:
            # K9: duration_ms means a duration here too, not a last-sample
            # field -- the stimulus's own internal total_ms (if any) already
            # used the half-step guard correctly inside its own forward().
            time_ms = _duration_axis(dt_ms, duration_ms, device)
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


def _is_stepped(instance: Any) -> bool:
    """Whether *instance* advances through time via ``step()``.

    A stimulus may express time in one of two ways: return the whole
    ``[T, H, W]`` sequence from ``forward()``, or return the current frame
    and advance on ``step()``. Both are legitimate (Fact K-c), but the
    second needs driving, and a stimulus that is genuinely static also
    inherits a no-op ``step()`` from :class:`BaseStimulus`. Treat an
    instance as stepped only when it has both a ``step`` and a trajectory
    to step along, so a static stimulus is not needlessly re-rendered.
    """
    return (
        callable(getattr(instance, "step", None))
        and getattr(instance, "trajectory", None) is not None
        and len(getattr(instance, "trajectory")) > 1
    )


def _render_stepped(
    instance: Any,
    xx: torch.Tensor,
    yy: torch.Tensor,
    dt_ms: float,
    duration_ms: Optional[float],
    device: str,
    params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drive a stepped stimulus to build its ``[T, H, W]`` sequence.

    Frames come from ``duration_ms`` when the caller gave one, otherwise
    from the trajectory's own length. ``step()`` saturates at the last
    trajectory entry rather than raising, so a duration longer than the
    trajectory holds the final position -- the same thing the instance
    would do if driven by hand.

    The instance is reset before and after, so rendering twice gives the
    same answer and leaves no state behind for the next caller.
    """
    n_traj = len(instance.trajectory)
    if duration_ms is not None:
        time_ms = _duration_axis(dt_ms, duration_ms, device)
        n_frames = time_ms.numel()
    else:
        n_frames = n_traj
        time_ms = torch.arange(n_frames, dtype=torch.float32, device=device) * float(
            dt_ms
        )

    instance.reset_state()
    frames = []
    for _ in range(n_frames):
        frames.append(instance(xx, yy))
        instance.step()
    instance.reset_state()
    stacked = torch.stack(frames, dim=0)

    amp = _temporal_envelope(
        time_ms,
        ramp_up_ms=params.get("ramp_up_ms", 0.0),
        plateau_ms=params.get("plateau_ms", float(time_ms[-1]) + float(dt_ms)),
        ramp_down_ms=params.get("ramp_down_ms", 0.0),
        amplitude=1.0,
    )
    return stacked * amp.view(-1, 1, 1), time_ms


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


def render_for_config(
    config: "SensoryForgeConfig",
    *,
    duration_ms: float,
    dt_ms: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, Optional["StimulusCanvas"], List[Tuple[str, Any]]
]:
    """Render ``config.stimulus`` on ``config``'s first grid's canvas (Task 0.6, F-061).

    This is the one place that turns a :class:`~sensoryforge.config.schema.
    SensoryForgeConfig`'s canonical ``stimulus:`` block into frames --
    :mod:`sensoryforge.cli` (``sensoryforge run``) and
    :func:`sensoryforge.gui.circuit.run.render_graph_stimulus` both call it,
    so a config run from the CLI and the same config run from the Circuit
    tab render byte-identical stimuli.

    ``StimulusConfig.to_dict()`` carries every field the schema has
    (administrative ones like ``motion``/``composition_mode``/``channel``
    included); a given registered stimulus class's constructor only accepts
    its own subset. A ``TypeError`` naming an unexpected keyword is retried
    with that keyword dropped, the same way :func:`render_stimulus`'s own
    envelope-key handling works, rather than hard-coding a per-type field
    list here -- every dropped ``(key, value)`` pair is returned so the
    caller can decide whether it is worth warning about (only a value the
    user actually changed from the schema default is, see
    :func:`sensoryforge.gui.circuit.run._dropped_params_warning`).

    Args:
        config: The reconstructed :class:`SensoryForgeConfig`.
        duration_ms: Stimulus duration in ms.
        dt_ms: Record step in ms (``config.simulation.dt_ms``, passed
            explicitly rather than read off *config* so a caller overriding
            ``--duration``/``dt_ms`` independently of the loaded config can
            do so without mutating it).

    Returns:
        ``(stimulus, time_ms, canvas, dropped)``: ``stimulus`` is
        ``frames.unsqueeze(0)`` (batch dimension added), matching what
        :meth:`~sensoryforge.core.simulation_engine.SimulationEngine.run`
        expects; ``time_ms`` is the ``[T]`` time axis; ``canvas`` is the
        :class:`~sensoryforge.stimuli.canvas.StimulusCanvas` the frames were
        rendered on, or ``None`` when *config* has no grids (the synthetic
        40x40 fallback canvas is used in that case, matching
        :func:`render_stimulus`'s own default); ``dropped`` is the list of
        ``(field_name, value)`` pairs the stimulus class's constructor
        rejected.
    """
    from sensoryforge.stimuli.canvas import stimulus_canvas

    canvas: Optional["StimulusCanvas"] = None
    if config.grids:
        # The stimulus names its grid in `target_layer`; without one, or with a
        # name that matches no grid, it is rendered on the first grid.
        target = getattr(config.stimulus, "target_layer", None)
        grid_cfg = next(
            (g for g in config.grids if target and g.name == target),
            config.grids[0],
        )
        canvas = stimulus_canvas(grid_cfg, device=config.simulation.device)
        xx, yy = canvas.xx, canvas.yy
    else:
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 40),
            torch.linspace(-1, 1, 40),
            indexing="ij",
        )

    stim = config.stimulus
    # Forward only the fields the user set (StimulusConfig.explicit_fields()).
    # StimulusConfig carries a default for every field of every stimulus type
    # (start and end both [0, 0], an 800 ms plateau, ...); forwarding those
    # made them override the chosen type's own defaults, so
    # `stimulus: {type: moving_edge}` rendered an edge with start == end that
    # never moved, with no error. Explicitness is recorded, not inferred from
    # the value, so a field deliberately set to the schema default still counts.
    explicit = stim.explicit_fields() - {"name", "type", "params"}
    full = stim.to_full_dict()
    # `stimulus.params` carries the type's parameters that have no named
    # field; a named field, when set, is the authority for its own name.
    stimulus_params = dict(getattr(stim, "params", None) or {})
    stimulus_params.update({k: full[k] for k in explicit if k in full})
    dropped: List[Tuple[str, Any]] = []
    while True:
        try:
            frames, time_ms = render_stimulus(
                stim.type,
                stimulus_params,
                xx,
                yy,
                dt_ms=dt_ms,
                duration_ms=duration_ms,
                device=config.simulation.device,
            )
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if match is None or match.group(1) not in stimulus_params:
                raise
            key = match.group(1)
            dropped.append((key, stimulus_params.pop(key)))

    stimulus_tensor = frames.unsqueeze(0)
    return stimulus_tensor, time_ms, canvas, dropped


def dropped_params_warning(
    stimulus_type: str, dropped: List[Tuple[str, Any]]
) -> Optional[str]:
    """The warning text for :func:`render_for_config`'s discarded settings, or ``None``.

    Shared by :mod:`sensoryforge.cli` and
    :mod:`sensoryforge.gui.circuit.run` so both callers of
    :func:`render_for_config` describe a dropped keyword the same way.

    Args:
        stimulus_type: The stimulus's registered name, for the message.
        dropped: ``(field_name, value)`` pairs the constructor rejected.

    Returns:
        A message naming only the fields whose value the caller had changed
        from the schema default, or ``None`` when every discarded field was
        untouched and there is nothing worth saying.
    """
    deliberate = [
        f"{key}={value!r}"
        for key, value in dropped
        if not _is_stimulus_schema_default(key, value)
    ]
    if not deliberate:
        return None
    return (
        f"Stimulus {stimulus_type!r} does not accept {', '.join(deliberate)}; "
        "the value(s) you set were ignored and the stimulus ran without them."
    )


def _is_stimulus_schema_default(field_name: str, value: Any) -> bool:
    """Whether *value* is what ``StimulusConfig`` would hold untouched.

    ``StimulusConfig.to_dict()`` carries every field the schema defines,
    most of which a given stimulus class knows nothing about. Discarding
    those is housekeeping. Discarding one the caller actually set is a
    changed stimulus, so the two cases are told apart here rather than
    warning about all of them and training the reader to ignore it.

    Args:
        field_name: The dropped keyword.
        value: The value it held.

    Returns:
        ``True`` when the field is unknown to the schema or still at its
        declared default.
    """
    import dataclasses

    from sensoryforge.config.schema import StimulusConfig

    for field in dataclasses.fields(StimulusConfig):
        if field.name != field_name:
            continue
        if field.default is not dataclasses.MISSING:
            return value == field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value == field.default_factory()  # type: ignore[misc]
        return False
    return True


def is_default_stimulus_config(config: "SensoryForgeConfig") -> bool:
    """Whether ``config.stimulus`` is an untouched ``StimulusConfig()`` default.

    Used by ``sensoryforge run``/``sensoryforge validate`` (Task 0.6) to
    decide whether a canonical config's ``stimulus:`` block was ever set by
    its author, or is just the schema default that ``SensoryForgeConfig()``
    fills in on its own -- only in the latter case is it reasonable to fall
    back to the legacy trapezoid default instead of rendering the block.

    Args:
        config: The loaded :class:`SensoryForgeConfig`.

    Returns:
        ``True`` when ``config.stimulus.to_dict()`` equals a fresh
        ``StimulusConfig().to_dict()``.
    """
    from sensoryforge.config.schema import StimulusConfig

    return config.stimulus.to_dict() == StimulusConfig().to_dict()
