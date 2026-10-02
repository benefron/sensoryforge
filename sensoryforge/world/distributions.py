"""Axes of a world: how one parameter is sampled, stratified and probed.

See spec §3.2 (forms), §4.2 (from ``u`` to values) and §6.2 (strata, probes).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

FORMS = ("constant", "numeric", "int", "categorical", "registered")
NUMERIC_DISTS = ("uniform", "log_uniform")
_AXIS_KEYS = {
    "value",
    "range",
    "dist",
    "int",
    "values",
    "weights",
    "circular",
    "probes",
}


def plain(value: Any) -> Any:
    """A JSON-plain scalar (str, int, float, bool or None), else ``ValueError``."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"axis values must be numbers or strings, got {value!r}")


def is_number(value: Any) -> bool:
    """True for an int or a float (not a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: A number with an exponent but no decimal point, which PyYAML (YAML 1.1)
#: reads as text: ``3e-1`` is the string ``'3e-1'``, ``3.0e-1`` the float.
_EXPONENT_TEXT = re.compile(r"^[-+]?[0-9]+[eE][-+]?[0-9]+$")


def text_number_hint(value: Any) -> str:
    """A hint for a number YAML read as text (``'3e-1'``), else ``""``."""
    if isinstance(value, str) and _EXPONENT_TEXT.match(value.strip()):
        return (
            " (YAML reads a number with an exponent but no decimal point as "
            "text: write 3.0e-1, not 3e-1)"
        )
    return ""


def _finite_number(value: Any, where: str, what: str) -> float:
    """``value`` as a float, or ``ValueError`` if it is not a finite number."""
    if not is_number(value):
        hint = text_number_hint(value)
        raise ValueError(f"{where}: {what} must be numbers, got {value!r}{hint}")
    if not math.isfinite(value):
        raise ValueError(f"{where}: {what} must be finite, got {value!r}")
    return float(value)


def _braille_cells() -> List[str]:
    return [
        "".join(combo) for size in range(1, 7) for combo in combinations("123456", size)
    ]


#: The 63 non-empty six-dot braille cells as dot-number strings ("1" ... "123456").
BRAILLE_CELLS: List[str] = _braille_cells()


@dataclass(frozen=True)
class Distribution:
    """A registered distribution: values from ``u``, and its finite support if any."""

    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]]
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None


DISTRIBUTIONS: Dict[str, Distribution] = {}


def register_distribution(
    name: str,
    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]],
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None,
    *,
    replace: bool = False,
) -> None:
    """Register a distribution usable as ``{dist: <name>}`` on an axis.

    Args:
        name: The name axes use.
        sample: ``sample(u, axis) -> values``, ``u`` a float64 array in ``[0, 1)``.
        support: ``support(axis) -> values`` for a finite distribution (needed
            to stratify it), else ``None``.
        replace: Replace a distribution already registered under ``name``.

    Raises:
        ValueError: For ``uniform``/``log_uniform`` (built in, never
            replaced), or a taken name when ``replace`` is false.
    """
    if name in NUMERIC_DISTS:
        raise ValueError(f"{name!r} is a built-in numeric distribution")
    if name in DISTRIBUTIONS and not replace:
        raise ValueError(
            f"distribution {name!r} is already registered; "
            "pass replace=True to replace it"
        )
    DISTRIBUTIONS[name] = Distribution(sample=sample, support=support)


def _sample_braille_cells(u: np.ndarray, axis: "AxisSpec") -> List[str]:
    idx = np.minimum((u * len(BRAILLE_CELLS)).astype(np.int64), len(BRAILLE_CELLS) - 1)
    return [BRAILLE_CELLS[i] for i in idx]


register_distribution(
    "braille_cells", _sample_braille_cells, lambda axis: list(BRAILLE_CELLS)
)


@dataclass(frozen=True)
class AxisSpec:
    """One axis of a world class (spec §3.2).

    Attributes:
        name: The axis name as declared (bare, or dotted like ``shape.width_mm``).
        form: ``constant``, ``numeric``, ``int``, ``categorical`` or ``registered``.
        value: The constant (``constant`` only).
        lo, hi: The range (``numeric``, ``int``).
        dist: ``uniform``/``log_uniform`` (numeric) or a registered name.
        values, weights: The categories and their weights (``categorical``).
        circular: An angle: no out-of-range probes.
        probes: False to switch probes off for this axis.
        options: Extra keys passed to a registered distribution.
        domain: ``(lo, hi)`` valid values of the bound field; probes stay inside.
    """

    name: str
    form: str
    value: Any = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    dist: str = "uniform"
    values: Tuple[Any, ...] = ()
    weights: Tuple[float, ...] = ()
    circular: bool = False
    probes: bool = True
    options: Tuple[Tuple[str, Any], ...] = ()
    domain: Tuple[Optional[float], Optional[float]] = (None, None)

    # ------------------------------------------------------------ parsing

    @classmethod
    def from_dict(cls, name: str, spec: Any, where: Optional[str] = None) -> "AxisSpec":
        """Parse spec dict (value, range, values, or dist).

        Range bounds and weights must be finite numbers; constants and
        categorical values may be numbers or strings, and numbers must be
        finite. Whether a value suits the field it binds is checked when the
        world loads (:mod:`sensoryforge.world.schema`).

        Args:
            name: The axis name.
            spec: Its declaration.
            where: The path errors name (default ``axis '<name>'``).

        Raises:
            ValueError: Naming the axis and what is wrong.
        """
        where = where or f"axis {name!r}"
        if not isinstance(spec, dict):
            raise ValueError(
                f"{where}: expected a mapping such as {{range: [lo, hi]}}, got {spec!r}"
            )
        flags = {
            "circular": bool(spec.get("circular", False)),
            "probes": bool(spec.get("probes", True)),
        }
        if "value" in spec:
            if set(spec) != {"value"}:
                raise ValueError(
                    f"{where}: a constant takes only 'value', got {sorted(spec)}"
                )
            value = plain(spec["value"])
            if is_number(value):
                _finite_number(value, where, "constants")
            return cls(name=name, form="constant", value=value)
        if "dist" in spec and spec["dist"] not in NUMERIC_DISTS and "range" not in spec:
            dist = spec["dist"]
            if dist not in DISTRIBUTIONS:
                known = sorted(DISTRIBUTIONS)
                raise ValueError(
                    f"{where}: unknown distribution {dist!r}; known: {known}"
                )
            options = tuple(
                sorted(
                    (k, plain(v))
                    for k, v in spec.items()
                    if k not in {"dist", "circular", "probes"}
                )
            )
            return cls(
                name=name, form="registered", dist=dist, options=options, **flags
            )
        unknown = set(spec) - _AXIS_KEYS
        if unknown:
            allowed = sorted(_AXIS_KEYS)
            raise ValueError(
                f"{where}: unknown keys {sorted(unknown)}; allowed: {allowed}"
            )
        if "values" in spec:
            if not isinstance(spec["values"] or [], (list, tuple)):
                raise ValueError(f"{where}: 'values' must be a list")
            values = tuple(plain(v) for v in spec["values"] or [])
            if not values:
                raise ValueError(f"{where}: 'values' is empty")
            for v in values:
                if is_number(v):
                    _finite_number(v, where, "values")
            raw_weights = spec.get("weights", [1.0] * len(values))
            if not isinstance(raw_weights, (list, tuple)):
                raise ValueError(f"{where}: 'weights' must be a list")
            weights = tuple(_finite_number(w, where, "weights") for w in raw_weights)
            if len(weights) != len(values):
                raise ValueError(
                    f"{where}: {len(weights)} weights for {len(values)} values"
                )
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError(f"{where}: weights must be >= 0 with a positive sum")
            return cls(
                name=name, form="categorical", values=values, weights=weights, **flags
            )
        if "range" in spec:
            bounds = spec["range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{where}: range must be [lo, hi], got {bounds!r}")
            lo = _finite_number(bounds[0], where, "range bounds")
            hi = _finite_number(bounds[1], where, "range bounds")
            if lo > hi:
                raise ValueError(f"{where}: lo > hi in range {bounds!r}")
            if spec.get("int"):
                if lo != math.floor(lo) or hi != math.floor(hi):
                    raise ValueError(
                        f"{where}: an int axis needs integer bounds, got {bounds!r}"
                    )
                return cls(name=name, form="int", lo=lo, hi=hi, **flags)
            dist = spec.get("dist", "uniform")
            if dist not in NUMERIC_DISTS:
                raise ValueError(
                    f"{where}: dist {dist!r} for a range must be one of {NUMERIC_DISTS}"
                )
            if dist == "log_uniform" and lo <= 0:
                raise ValueError(f"{where}: log_uniform needs lo > 0, got {lo}")
            return cls(name=name, form="numeric", lo=lo, hi=hi, dist=dist, **flags)
        raise ValueError(
            f"{where}: needs one of value, range, values or dist; got {sorted(spec)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """The normalised declaration (what the world id hashes)."""
        if self.form == "constant":
            return {"value": self.value}
        if self.form == "numeric":
            out: Dict[str, Any] = {"range": [self.lo, self.hi], "dist": self.dist}
        elif self.form == "int":
            out = {"range": [int(self.lo), int(self.hi)], "int": True}
        elif self.form == "categorical":
            out = {"values": list(self.values), "weights": list(self.weights)}
        else:
            out = {"dist": self.dist, **dict(self.options)}
        if self.circular:
            out["circular"] = True
        if not self.probes:
            out["probes"] = False
        return out

    def with_domain(self, lo: Optional[float], hi: Optional[float]) -> "AxisSpec":
        """This axis with the bound field's valid range attached."""
        return replace(self, domain=(lo, hi))

    # ----------------------------------------------------------- sampling

    @property
    def is_random(self) -> bool:
        """False for a constant."""
        return self.form != "constant"

    @property
    def _log(self) -> bool:
        return self.form == "numeric" and self.dist == "log_uniform"

    def _to_scale(self, v: float) -> float:
        return math.log(v) if self._log else float(v)

    def _from_scale(self, t: np.ndarray) -> np.ndarray:
        return np.exp(t) if self._log else np.asarray(t, dtype=np.float64)

    def sample(self, u: np.ndarray) -> List[Any]:
        """One value per ``u`` (declared sampling, spec §4.2)."""
        u = np.asarray(u, dtype=np.float64)
        if self.form == "constant":
            return [self.value] * u.size
        if self.form == "numeric":
            if self._log:
                t_lo, t_hi = math.log(self.lo), math.log(self.hi)
                return np.exp(t_lo + u * (t_hi - t_lo)).tolist()
            return (self.lo + u * (self.hi - self.lo)).tolist()
        if self.form == "int":
            span = int(self.hi) - int(self.lo) + 1
            steps = np.minimum(np.floor(u * span).astype(np.int64), span - 1)
            return [int(self.lo) + int(s) for s in steps]
        if self.form == "categorical":
            w = np.asarray(self.weights, dtype=np.float64)
            cum = np.cumsum(w) / w.sum()
            idx = np.minimum(
                np.searchsorted(cum, u, side="right"), len(self.values) - 1
            )
            return [self.values[i] for i in idx]
        return DISTRIBUTIONS[self.dist].sample(u, self)

    def support(self) -> Optional[List[Any]]:
        """The finite set of values, or ``None`` for a continuous axis."""
        if self.form == "categorical":
            return list(self.values)
        if self.form == "int":
            return list(range(int(self.lo), int(self.hi) + 1))
        if self.form == "registered":
            dist = DISTRIBUTIONS[self.dist]
            return None if dist.support is None else list(dist.support(self))
        return None

    # --------------------------------------------------- strata and probes

    def _require_numeric(self) -> None:
        if self.form != "numeric":
            raise ValueError(f"axis {self.name!r} is {self.form}, not numeric")

    def bin_edges(self, bins: int) -> List[float]:
        """``bins + 1`` edges, equal on the sampling scale (log for log_uniform)."""
        self._require_numeric()
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        inner = [
            float(self._from_scale(t_lo + k / bins * (t_hi - t_lo)))
            for k in range(1, bins)
        ]
        return [self.lo, *inner, self.hi]

    def stratum_values(
        self, labels: np.ndarray, bins: int, u: np.ndarray
    ) -> List[float]:
        """Sample values in strata by inverse CDF."""
        self._require_numeric()
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        t = t_lo + (np.asarray(labels, dtype=np.float64) + np.asarray(u)) / bins * (
            t_hi - t_lo
        )
        return np.clip(self._from_scale(t), self.lo, self.hi).tolist()

    def bin_label(self, b: int, bins: int) -> str:
        """``"[a, b)"`` (``"[a, b]"`` for the last bin), 6 significant digits."""
        edges = self.bin_edges(bins)
        close = "]" if b == bins - 1 else ")"
        return f"[{edges[b]:.6g}, {edges[b + 1]:.6g}{close}"

    def probe_values(
        self, side: str, bins: int, u: np.ndarray
    ) -> Optional[List[float]]:
        """Values one bin-width below or above, or ``None`` if no room.

        Probes stay inside :attr:`domain`; circular and ``probes: false`` axes,
        and non-numeric ones, have none.
        """
        if self.form != "numeric" or self.circular or not self.probes:
            return None
        if side not in ("below", "above"):
            raise ValueError(f"probe side must be 'below' or 'above', got {side!r}")
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        width = (t_hi - t_lo) / bins
        if width <= 0:
            return None
        d_lo, d_hi = self.domain
        u = np.asarray(u, dtype=np.float64)
        if side == "below":
            a, b = t_lo - width, t_lo
            if d_lo is not None and not (self._log and d_lo <= 0):
                a = max(a, self._to_scale(d_lo))
            if a >= b:
                return None
            v = self._from_scale(a + u * (b - a))
            v = np.minimum(v, np.nextafter(self.lo, -np.inf))
            if d_lo is not None:
                v = np.maximum(v, d_lo)
        else:
            a, b = t_hi, t_hi + width
            if d_hi is not None:
                b = min(b, self._to_scale(d_hi))
            if b <= a:
                return None
            v = self._from_scale(b - u * (b - a))
            v = np.maximum(v, np.nextafter(self.hi, np.inf))
            if d_hi is not None:
                v = np.minimum(v, d_hi)
        return v.tolist()

    # -------------------------------------------------------- fixed draws

    def midpoint(self) -> Any:
        """The value a fixed draw takes when it does not set this axis."""
        if self.form == "constant":
            return self.value
        if self.form == "numeric":
            t = (self._to_scale(self.lo) + self._to_scale(self.hi)) / 2.0
            return float(self._from_scale(t))
        if self.form == "int":
            return int(math.floor((self.lo + self.hi) / 2.0))
        if self.form == "categorical":
            return self.values[0]
        support = self.support()
        if support:
            return support[0]
        return DISTRIBUTIONS[self.dist].sample(np.zeros(1), self)[0]

    def contains(self, value: Any) -> bool:
        """Whether ``value`` lies in the declared range or set."""
        if self.form == "constant":
            return value == self.value
        if self.form in ("numeric", "int"):
            return isinstance(value, (int, float)) and self.lo <= value <= self.hi
        support = self.support()
        return True if support is None else value in support
