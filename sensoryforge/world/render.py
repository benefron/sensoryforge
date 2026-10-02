"""Rendering draws at arbitrary times on arbitrary coordinates (spec §5).

``render`` groups draws by class (and by any non-numeric values, such as a
profile axis), evaluates each group as one broadcast computation, chunked to
a memory budget, and adds each group's frames into its rows. Every operation
is elementwise or a fixed-order sum, so a draw renders to the same bits alone,
in a batch, or in any chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.world.sampling import Draw, Session
from sensoryforge.world.schema import World

Item = Union[Draw, Session]

#: Elements per chunk of one group's ``[g, K, *S]`` output (float64: 64 MB).
DEFAULT_MAX_ELEMENTS = 2**23


@dataclass(frozen=True)
class Canvas:
    """Where to render: x, y in mm (float64, CPU), any common shape ``S``."""

    xx: torch.Tensor
    yy: torch.Tensor

    def __post_init__(self) -> None:
        xx = torch.as_tensor(self.xx, dtype=torch.float64).detach().cpu()
        yy = torch.as_tensor(self.yy, dtype=torch.float64).detach().cpu()
        if xx.shape != yy.shape:
            raise ValueError(
                f"xx {list(xx.shape)} and yy {list(yy.shape)} differ in shape"
            )
        object.__setattr__(self, "xx", xx)
        object.__setattr__(self, "yy", yy)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The coordinate shape ``S`` frames take."""
        return tuple(self.xx.shape)

    @classmethod
    def from_grid(
        cls,
        rows: int,
        cols: int,
        spacing_mm: float,
        center_mm: Tuple[float, float] = (0.0, 0.0),
    ) -> "Canvas":
        """A centred ``rows x cols`` lattice (``ij``: dim 0 is x), float64."""
        if rows < 1 or cols < 1 or spacing_mm <= 0:
            raise ValueError(
                f"need rows, cols >= 1 and spacing > 0; "
                f"got {rows}, {cols}, {spacing_mm}"
            )
        cx, cy = float(center_mm[0]), float(center_mm[1])
        half_x = (rows - 1) * spacing_mm / 2.0
        half_y = (cols - 1) * spacing_mm / 2.0
        x = torch.linspace(cx - half_x, cx + half_x, rows, dtype=torch.float64)
        y = torch.linspace(cy - half_y, cy + half_y, cols, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return cls(xx, yy)

    @classmethod
    def from_grid_config(cls, grid_cfg: GridConfig) -> "Canvas":
        """The canvas ``stimulus_canvas`` builds, in float64."""
        if grid_cfg.coords_file:
            old = stimulus_canvas(grid_cfg)
            return cls(old.xx, old.yy)
        return cls.from_grid(
            grid_cfg.rows or 40,
            grid_cfg.cols or 40,
            grid_cfg.spacing,
            (grid_cfg.center_x, grid_cfg.center_y),
        )

    @classmethod
    def from_points(cls, xy: Any) -> "Canvas":
        """Scattered points ``[M, 2]`` (x, y) in mm; frames are ``[..., M]``."""
        xy = torch.as_tensor(xy, dtype=torch.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"points must be [M, 2], got {list(xy.shape)}")
        return cls(xy[:, 0], xy[:, 1])


def movie_times(dt_ms: float, duration_ms: float) -> torch.Tensor:
    """``t_k = k * dt_ms`` for ``k < round(duration_ms / dt_ms)``, float64.

    At least one step.
    """
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")
    steps = max(int(round(float(duration_ms) / float(dt_ms))), 1)
    return torch.arange(steps, dtype=torch.float64) * float(dt_ms)


def _as_item(item: Any, world: Optional[World]) -> Item:
    if isinstance(item, (Draw, Session)):
        return item
    if isinstance(item, dict):
        if world is None:
            raise ValueError("rendering a draw record needs world=")
        if "items" in item:
            return Session.from_dict(item, world)
        return Draw.from_dict(item, world)
    raise ValueError(
        f"cannot render {type(item).__name__}; give Draw, Session or their records"
    )


def _jobs(
    items: Sequence[Item], times: torch.Tensor
) -> Iterator[Tuple[int, Draw, torch.Tensor]]:
    """``(row, draw, local times)``: a session becomes one job per draw, windowed."""
    for row, item in enumerate(items):
        if isinstance(item, Session):
            starts = [start for start, _ in item.items] + [item.duration_ms]
            for j, (start, draw) in enumerate(item.items):
                stop = min(starts[j + 1], item.duration_ms)
                window = (times[row] >= start) & (times[row] < stop)
                local = torch.where(
                    window, times[row] - start, torch.full_like(times[row], -1.0)
                )
                yield row, draw, local
        else:
            yield row, item, times[row]


def _group_key(draw: Draw) -> Tuple[Any, ...]:
    spec = draw.spec
    fixed = tuple(
        sorted(
            (name, value)
            for name, value in draw.values.items()
            if isinstance(value, (str, bool))
            and spec.bindings[name][0] in ("shape", "modulation")
        )
    )
    return (id(draw.world), draw.class_name, fixed)


def render(
    items: Sequence[Any],
    canvas: Canvas,
    times_ms: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    world: Optional[World] = None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> torch.Tensor:
    """Render draws (or sessions) at arbitrary times (spec §5.1).

    Args:
        items: Draws, sessions, or their records (records need ``world``).
        canvas: Where to render.
        times_ms: ``[K]`` times shared by every item, or ``[n, K]`` per item;
            ms from each item's start. Times before 0 or after the item's end
            render exactly zero.
        dtype: ``torch.float32`` or ``torch.float64``.
        device: ``cpu``, ``cuda`` or ``mps`` (float32 only on MPS).
        world: The world records belong to.
        max_elements: Chunk budget per group, in output elements.

    Returns:
        ``[n, K, *S]``, or ``[n, K, C, *S]`` when the world has ``C > 1`` channels.

    Raises:
        ValueError: For a bad dtype, float64 on MPS, mismatched ``times_ms``,
            or items from worlds with different channels.
    """
    device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    if dtype == torch.float64 and device.type == "mps":
        raise ValueError("MPS has no float64: render on cpu or cuda, or in float32")
    resolved = [_as_item(item, world) for item in items]
    n = len(resolved)
    times = torch.as_tensor(times_ms, dtype=torch.float64).detach().cpu()
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(n, -1)
    if times.ndim != 2 or times.shape[0] != n:
        raise ValueError(
            f"times_ms must be [K] or [n, K] with n = {n}, got {list(times.shape)}"
        )
    k_count = times.shape[1]
    channels = resolved[0].world.channels if resolved else ["value"]
    if any(item.world.channels != channels for item in resolved):
        raise ValueError("render: items come from worlds with different channels")
    multi = len(channels) > 1
    shape = (
        (n, k_count, len(channels)) + canvas.shape
        if multi
        else (n, k_count) + canvas.shape
    )
    out = torch.zeros(shape, dtype=dtype, device=device)
    if n == 0 or k_count == 0:
        return out

    groups: Dict[Tuple[Any, ...], List[Tuple[int, Draw, torch.Tensor]]] = {}
    for job in _jobs(resolved, times):
        groups.setdefault(_group_key(job[1]), []).append(job)
    X = canvas.xx.to(device=device, dtype=dtype)
    Y = canvas.yy.to(device=device, dtype=dtype)
    per_chunk = max(1, int(max_elements) // max(1, k_count * X.numel()))
    for members in groups.values():
        spec = members[0][1].spec
        target = out[:, :, channels.index(spec.channel)] if multi else out
        for start in range(0, len(members), per_chunk):
            chunk = members[start : start + per_chunk]
            local = torch.stack([m[2] for m in chunk]).to(device=device, dtype=dtype)
            frames = spec.kind_obj.render_group(
                spec, [m[1] for m in chunk], X, Y, local
            )
            rows = torch.tensor([m[0] for m in chunk], device=device)
            target.index_add_(0, rows, frames)
    return out


def render_movie(
    item: Any,
    canvas: Canvas,
    dt_ms: float,
    duration_ms: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    world: Optional[World] = None,
) -> torch.Tensor:
    """One item's frames at ``t_k = k * dt_ms``: ``[T, *S]`` (or ``[T, C, *S]``)."""
    times = movie_times(dt_ms, duration_ms)
    return render([item], canvas, times, dtype=dtype, device=device, world=world)[0]
