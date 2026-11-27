"""Build the points of a registered layout, for sensors and for neurons.

A layout a config can name is a :class:`~sensoryforge.core.grid_base.BaseGrid`
subclass registered in ``GRID_REGISTRY`` whose constructor accepts ``xlim``,
``ylim`` and ``device`` as keywords, ``seed`` only if it is random, and its own
settings. ``get_all_coordinates()`` returns ``[N, 2]`` ``(x, y)`` positions in
mm, and the layout decides ``N``. The built-in names are not built here: they
keep their own code in ``core/grid.py`` and ``core/innervation.py``.

Topic: grid, registry
Tests: tests/unit/test_custom_layouts.py
Entry points: build_layout_points, layout_problem, custom_layout_names,
    layout_setting_specs, layout_takes_seed, BUILTIN_LAYOUTS
Pinned by: docs/how-to/extend/add_grid_arrangement.md
Do not change without: points are checked, never moved or dropped, and a name is
    matched exactly as registered, so a config never gets a layout it did not ask for.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Mapping, Optional, Tuple

import torch

from sensoryforge.core.innervation import NEURON_ARRANGEMENTS
from sensoryforge.registry import GRID_REGISTRY

#: Layout names with their own code; a plugin cannot take one.
BUILTIN_LAYOUTS: Tuple[str, ...] = NEURON_ARRANGEMENTS

#: Constructor arguments the caller supplies, so a config cannot set them.
_SUPPLIED = ("xlim", "ylim", "device", "seed")

#: How far (mm) a point may sit outside the area through float rounding.
_TOLERANCE_MM = 1e-5


def _registered_names() -> List[str]:
    from sensoryforge.register_components import register_all

    register_all()
    return GRID_REGISTRY.list_registered()


def _parameters(name: str) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(GRID_REGISTRY.get_class(name).__init__).parameters


def _takes_area(name: str) -> bool:
    return {"xlim", "ylim", "device"} <= set(_parameters(name))


def custom_layout_names() -> List[str]:
    """Names of the registered plugin layouts a config can use.

    Returns:
        Sorted names, without the built-ins and without grid plugins whose
        constructor does not accept the area.
    """
    return [
        name
        for name in _registered_names()
        if name not in BUILTIN_LAYOUTS and _takes_area(name)
    ]


def layout_takes_seed(name: str) -> bool:
    """Whether the installed layout ``name`` has a ``seed`` constructor argument."""
    return name in custom_layout_names() and "seed" in _parameters(name)


def layout_setting_specs(name: str) -> list:
    """The ``ParamSpec`` of each setting a config may give layout ``name``.

    Args:
        name: An installed plugin layout.

    Returns:
        Its ``get_param_spec()`` without what SensoryForge supplies
        (``xlim``, ``ylim``, ``device``, ``seed``).
    """
    return [
        spec
        for spec in GRID_REGISTRY.get_param_spec(name)
        if spec.name not in _SUPPLIED
    ]


def layout_problem(name: str, params: Mapping[str, Any]) -> Optional[str]:
    """Why a config cannot use layout ``name`` with ``params``, if it cannot.

    The layout is not built, so a bad setting *value* is only found by
    :func:`build_layout_points`.

    Args:
        name: Layout name, exactly as registered.
        params: The layout's own settings.

    Returns:
        The reason, or ``None`` when the name and the setting names are usable.
    """
    if name in BUILTIN_LAYOUTS:
        return f"{name!r} is a built-in layout and is not built as a plugin"
    if name not in _registered_names():
        return (
            f"Unknown layout: {name!r}. Built-in: {', '.join(BUILTIN_LAYOUTS)}. "
            f"Registered plugins: {', '.join(custom_layout_names()) or 'none'}"
        )
    parameters = _parameters(name)
    missing_area = [k for k in ("xlim", "ylim", "device") if k not in parameters]
    if missing_area:
        return (
            f"layout {name!r} cannot be named in a config: its constructor does "
            f"not accept {', '.join(missing_area)}. Add them, or save its points "
            "and use a grid's coords_file"
        )
    # A constructor with **kwargs takes setting names its signature cannot list;
    # the layout itself then rejects a bad one when it is built.
    takes_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    for key in params:
        if key in _SUPPLIED:
            return (
                f"layout {name!r}: {key!r} is supplied by SensoryForge and cannot "
                "be a layout setting"
            )
        if key not in parameters and not takes_any:
            known = [k for k in parameters if k not in ("self",) + _SUPPLIED]
            return (
                f"layout {name!r} has no setting {key!r}; its settings are "
                f"{', '.join(known) or 'none'}"
            )
    for key, parameter in parameters.items():
        required = parameter.default is inspect.Parameter.empty and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        if required and key not in ("self",) + _SUPPLIED and key not in params:
            return f"layout {name!r} needs the setting {key!r}"
    return None


def build_layout_points(
    name: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    params: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    require_seed_support: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build and check the points of the plugin layout ``name``.

    Args:
        name: Layout name, exactly as registered.
        xlim: ``(min, max)`` of the area along x, in mm.
        ylim: ``(min, max)`` of the area along y, in mm.
        params: The layout's own settings.
        seed: Passed to a layout whose constructor takes ``seed``; ``None``
            leaves the layout's own default.
        require_seed_support: Refuse a ``seed`` the layout cannot take, for a
            seed that was set for the layout alone.
        device: Device of the returned tensor.

    Returns:
        Positions ``[N, 2]``, float32, ``(x, y)`` in mm; ``N`` is the layout's.

    Raises:
        ValueError: If the layout cannot be used (see :func:`layout_problem`),
            rejects a setting, or returns points that are not ``[N, 2]`` with
            ``N >= 1``, not finite, or outside the area.
    """
    params = dict(params or {})
    problem = layout_problem(name, params)
    if problem:
        raise ValueError(problem)
    takes_seed = "seed" in _parameters(name)
    if seed is not None and not takes_seed and require_seed_support:
        raise ValueError(f"layout {name!r} takes no seed, but seed={seed} is set")
    kwargs = {**params, "xlim": tuple(xlim), "ylim": tuple(ylim), "device": device}
    if takes_seed and seed is not None:
        kwargs["seed"] = seed
    try:
        layout = GRID_REGISTRY.get_class(name)(**kwargs)
        points = layout.get_all_coordinates()
    # The one broad catch: this runs third-party plugin code, which may raise
    # anything (ZeroDivisionError for a 0 setting, say). Callers, the GUI's
    # validation on every edit among them, handle ValueError; anything else
    # would escape a Qt slot and abort the app.
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"layout {name!r} failed: {type(exc).__name__}: {exc}"
        ) from exc

    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 2
        or points.shape[1] != 2
        or points.shape[0] < 1
    ):
        shape = list(getattr(points, "shape", [])) or type(points).__name__
        raise ValueError(
            f"layout {name!r} must return positions [N, 2] with N >= 1, got {shape}"
        )
    points = points.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(points).all()):
        raise ValueError(f"layout {name!r} returned positions that are not finite")
    outside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for axis, (low, high) in enumerate((xlim, ylim)):
        outside |= points[:, axis] < low - _TOLERANCE_MM
        outside |= points[:, axis] > high + _TOLERANCE_MM
    if bool(outside.any()):
        x, y = points[outside][0].tolist()
        raise ValueError(
            f"layout {name!r} put {int(outside.sum())} of {points.shape[0]} points "
            f"outside the area x {tuple(xlim)}, y {tuple(ylim)} mm; the first is "
            f"({x:.4f}, {y:.4f})"
        )
    return points
