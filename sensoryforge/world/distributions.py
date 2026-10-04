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

FORMS = ("constant", "numeric", "int", "categorical", "registered", "link")
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
    "stratify",
}


def plain(value: Any) -> Any:
    """A JSON-plain scalar (str, int, float, bool or None), else ``ValueError``."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"axis values must be numbers or strings, got {value!r}")


def _option(value: Any) -> Any:
    """A registered distribution's option: a scalar, or a tuple of scalars."""
    if isinstance(value, (list, tuple)):
        return tuple(plain(v) for v in value)
    return plain(value)


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
    """A registered distribution: values from ``u``, and its finite support if any.

    A distribution without a finite support may declare ``bounds``: the
    closed interval holding every number it draws, which the loader checks
    against the domain of the field the axis binds.
    """

    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]]
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None
    quantile: bool = False
    check: Optional[Callable[["AxisSpec"], None]] = None
    bounds: Optional[Callable[["AxisSpec"], Tuple[float, float]]] = None


DISTRIBUTIONS: Dict[str, Distribution] = {}


def register_distribution(
    name: str,
    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]],
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None,
    *,
    replace: bool = False,
    quantile: bool = False,
    check: Optional[Callable[["AxisSpec"], None]] = None,
    bounds: Optional[Callable[["AxisSpec"], Tuple[float, float]]] = None,
) -> None:
    """Register a distribution usable as ``{dist: <name>}`` on an axis.

    Args:
        name: The name axes use.
        sample: ``sample(u, axis) -> values``, ``u`` a float64 array in ``[0, 1)``.
        support: ``support(axis) -> values`` for a finite distribution (needed
            to stratify it), else ``None``.
        replace: Replace a distribution already registered under ``name``.
        quantile: True if ``sample(u, axis)`` is a quantile function of ``u``
            (continuous, monotone in ``u``): the axis can then be stratified
            into ``bins`` equal-probability strata on the ``u`` scale.
        check: ``check(axis)`` validates the axis' options when it loads;
            raise ``ValueError`` (the loader prefixes the path).
        bounds: ``bounds(axis) -> (lo, hi)`` for a distribution of numbers
            without a finite support: every value it draws lies in
            ``[lo, hi]``. The loader checks the interval against the bound
            field's domain; such a distribution binds a number field only
            when it declares its bounds.

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
    DISTRIBUTIONS[name] = Distribution(
        sample=sample,
        support=support,
        quantile=quantile,
        check=check,
        bounds=bounds,
    )


def _sample_braille_cells(u: np.ndarray, axis: "AxisSpec") -> List[str]:
    idx = np.minimum((u * len(BRAILLE_CELLS)).astype(np.int64), len(BRAILLE_CELLS) - 1)
    return [BRAILLE_CELLS[i] for i in idx]


register_distribution(
    "braille_cells", _sample_braille_cells, lambda axis: list(BRAILLE_CELLS)
)


def _travel_stretch(ratio: float) -> float:
    """The stretch ``s > 1`` whose angular central Gaussian has travel ratio ``ratio``.

    The expected travel along the long axis over the expected travel across it
    is ``R(s) = s * atan(k) / artanh(k / s)`` with ``k = sqrt(s**2 - 1)``;
    ``R`` rises from 1 at ``s = 1``. Solved by 200 bisection steps on
    ``[1, 1e6]``.

    Args:
        ratio: The travel ratio, ``> 1``.

    Returns:
        ``s`` (dimensionless).
    """

    def travel(s: float) -> float:
        k = math.sqrt(s * s - 1.0)
        return s * math.atan(k) / math.atanh(k / s)

    lo, hi = 1.0, 1.0e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if mid * mid - 1.0 <= 0.0 or travel(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_STRETCH_CACHE: Dict[float, float] = {}


def _stretch(ratio: float) -> float:
    if ratio not in _STRETCH_CACHE:
        _STRETCH_CACHE[ratio] = _travel_stretch(ratio)
    return _STRETCH_CACHE[ratio]


def _biased_options(axis: "AxisSpec") -> Tuple[float, float]:
    opts = dict(axis.options)
    unknown = set(opts) - {"travel_ratio", "axis_deg"}
    if unknown:
        raise ValueError(
            f"biased_direction: unknown options {sorted(unknown)}; "
            "allowed: ['axis_deg', 'travel_ratio']"
        )
    if "travel_ratio" not in opts:
        raise ValueError("biased_direction: needs travel_ratio")
    ratio = _finite_number(opts["travel_ratio"], "biased_direction", "travel_ratio")
    if ratio <= 0:
        raise ValueError(f"biased_direction: travel_ratio must be > 0, got {ratio}")
    axis_deg = _finite_number(opts.get("axis_deg", 0.0), "biased_direction", "axis_deg")
    return ratio, axis_deg


def _sample_biased_direction(u: np.ndarray, axis: "AxisSpec") -> List[float]:
    """Angles (degrees, in [0, 360)) of an anisotropic Gaussian velocity.

    ``theta = axis + atan2(sin 2 pi u, s cos 2 pi u)``: the angular central
    Gaussian stretched ``s`` times along ``axis_deg`` (0 deg = +x), with ``s``
    solved from the declared ``travel_ratio``. A ratio below 1 stretches the
    perpendicular axis; 1 is uniform. One uniform per draw, monotone in ``u``.
    """
    ratio, axis_deg = _biased_options(axis)
    u = np.asarray(u, dtype=np.float64)
    if ratio == 1.0:
        return ((axis_deg + 360.0 * u) % 360.0).tolist()
    if ratio > 1.0:
        s, base = _stretch(ratio), axis_deg
    else:
        s, base = _stretch(1.0 / ratio), axis_deg + 90.0
    phi = 2.0 * np.pi * u
    theta = base + np.degrees(np.arctan2(np.sin(phi), s * np.cos(phi)))
    return (theta % 360.0).tolist()


register_distribution(
    "biased_direction",
    _sample_biased_direction,
    quantile=True,
    check=_biased_options,
    bounds=lambda axis: (0.0, 360.0),
)


_LETTER_KEYS = {"letters", "weights", "cells", "lines"}
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _letter_options(axis: "AxisSpec") -> Tuple[str, np.ndarray, int, int]:
    """``(letters, cumulative weights, cells, lines)`` of a ``letter_text`` axis."""
    # Imported here: layered imports the world kernel lazily, never the reverse.
    from sensoryforge.stimuli.layered import _BRAILLE

    opts = dict(axis.options)
    unknown = set(opts) - _LETTER_KEYS
    if unknown:
        raise ValueError(
            f"letter_text: unknown options {sorted(unknown)}; "
            f"allowed: {sorted(_LETTER_KEYS)}"
        )
    letters = opts.get("letters", _ALPHABET)
    if not isinstance(letters, str) or not letters:
        raise ValueError(
            f"letter_text: letters must be non-empty text, got {letters!r}"
        )
    bad = sorted({ch for ch in letters if ch != " " and ch not in _BRAILLE})
    if bad:
        raise ValueError(
            f"letter_text: letters {bad} have no braille cell (a-z and space only)"
        )
    raw = opts.get("weights", (1.0,) * len(letters))
    if not isinstance(raw, tuple):
        raise ValueError(f"letter_text: weights must be a list, got {raw!r}")
    if len(raw) != len(letters):
        raise ValueError(f"letter_text: {len(raw)} weights for {len(letters)} letters")
    w = np.array(
        [_finite_number(v, "letter_text", "weights") for v in raw], dtype=np.float64
    )
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("letter_text: weights must be >= 0 with a positive sum")
    counts = []
    for name in ("cells", "lines"):
        n = opts.get(name, 1)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"letter_text: {name} must be an int >= 1, got {n!r}")
        counts.append(n)
    cells, lines = counts
    if cells * lines > 1 and axis.stratify:
        raise ValueError(
            "letter_text: several letters per draw have no finite support, so "
            "the axis cannot be stratified; declare it with stratify: false"
        )
    return letters, np.cumsum(w) / w.sum(), cells, lines


def _sample_letter_text(u: np.ndarray, axis: "AxisSpec") -> List[str]:
    """Text drawn letter by letter: ``cells`` letters on each of ``lines`` lines.

    The draw's 53 bits are recovered exactly from ``u`` and expanded to one
    sub-uniform per letter, each mapped through the cumulative weights; lines
    are joined with ``/``.
    """
    from sensoryforge.world import rng

    letters, cum, cells, lines = _letter_options(axis)
    n = cells * lines
    bits = np.floor(np.asarray(u, dtype=np.float64) * float(rng.SEED_LIMIT))
    out = []
    for b in bits.astype(np.int64).tolist():
        sub = rng.uniforms(rng.draw_seeds(b, range(n)), "letter")
        idx = np.minimum(np.searchsorted(cum, sub, side="right"), len(letters) - 1)
        chars = [letters[i] for i in idx]
        out.append(
            "/".join("".join(chars[r * cells : (r + 1) * cells]) for r in range(lines))
        )
    return out


def _letter_support(axis: "AxisSpec") -> Optional[List[str]]:
    letters, _, cells, lines = _letter_options(axis)
    return list(letters) if cells * lines == 1 else None


register_distribution(
    "letter_text",
    _sample_letter_text,
    _letter_support,
    check=_letter_options,
)


@dataclass(frozen=True)
class FieldValues:
    """The values one field can take in a class: a finite set, or an interval.

    Load-time checks that involve several fields together (a registered
    shape's ``check``) read the extremes from it.

    Attributes:
        values: Every value, when there are finitely many; else ``None``.
        lo, hi: Otherwise the closed interval ``[lo, hi]`` of numbers; both
            ``None`` when nothing is known about the values.
        integer: The interval holds only its integers.
    """

    values: Optional[Tuple[Any, ...]] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    integer: bool = False

    @classmethod
    def exactly(cls, value: Any) -> "FieldValues":
        """The one value ``value``."""
        return cls(values=(value,))

    @property
    def known(self) -> bool:
        """False when nothing is known about the values."""
        return self.values is not None or self.lo is not None

    def _numbers(self) -> List[float]:
        return [float(v) for v in self.values or () if is_number(v)]

    def largest(self) -> Optional[float]:
        """The largest number, or ``None`` if none is known."""
        if self.values is not None:
            numbers = self._numbers()
            return max(numbers) if numbers else None
        return self.hi

    def smallest(self) -> Optional[float]:
        """The smallest number, or ``None`` if none is known."""
        if self.values is not None:
            numbers = self._numbers()
            return min(numbers) if numbers else None
        return self.lo

    def smallest_positive(self) -> Optional[float]:
        """The infimum of the positive values (``0.0`` for an interval of
        reals starting at 0); ``None`` if no positive value is possible."""
        if self.values is not None:
            positive = [v for v in self._numbers() if v > 0]
            return min(positive) if positive else None
        if self.hi is None or self.hi <= 0:
            return None
        if self.lo > 0:
            return self.lo
        return 1.0 if self.integer else 0.0

    def may_be(self, value: Any) -> bool:
        """Whether ``value`` is possible (True when nothing is known)."""
        if self.values is not None:
            return value in self.values
        if not self.known:
            return True
        return is_number(value) and self.lo <= value <= self.hi


def class_field_values(axes: Dict[str, "AxisSpec"]) -> Dict[str, FieldValues]:
    """``{axis name: FieldValues}`` for a class's axes, a link as its target."""
    return {
        name: (axes[axis.link] if axis.form == "link" else axis).field_values()
        for name, axis in axes.items()
    }


def fill_links(axes: Dict[str, "AxisSpec"], values: Dict[str, Any]) -> None:
    """Set every link axis of ``axes`` in ``values`` to its target's value (in place).

    A link's target is never itself a link (checked when the world loads), so
    one pass after the other axes are filled is enough.
    """
    for name, axis in axes.items():
        if axis.form == "link":
            values[name] = values[axis.link]


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
        stratify: False to draw the axis i.i.d. in stratified splits (no bin
            label, no limit on an int axis's size).
        link: For form ``link``: the name of the axis whose value this axis
            copies within the same draw (``{same_as: <axis>}``).
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
    stratify: bool = True
    link: str = ""
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
        if "same_as" in spec:
            target = spec["same_as"]
            if set(spec) != {"same_as"} or not isinstance(target, str) or not target:
                raise ValueError(
                    f"{where}: a link is declared as exactly "
                    f"{{same_as: <axis name>}}, got {spec!r}"
                )
            return cls(name=name, form="link", link=target)
        flags = {
            "circular": bool(spec.get("circular", False)),
            "probes": bool(spec.get("probes", True)),
        }
        if "stratify" in spec:
            if not isinstance(spec["stratify"], bool):
                raise ValueError(
                    f"{where}: stratify must be true or false, got {spec['stratify']!r}"
                )
            if not spec["stratify"]:
                flags["stratify"] = False
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
                    (k, _option(v))
                    for k, v in spec.items()
                    if k not in {"dist", "circular", "probes", "stratify"}
                )
            )
            axis = cls(
                name=name, form="registered", dist=dist, options=options, **flags
            )
            check = DISTRIBUTIONS[dist].check
            if check is not None:
                try:
                    check(axis)
                except ValueError as exc:
                    raise ValueError(f"{where}: {exc}") from None
            return axis
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
        if self.form == "link":
            return {"same_as": self.link}
        if self.form == "numeric":
            out: Dict[str, Any] = {"range": [self.lo, self.hi], "dist": self.dist}
        elif self.form == "int":
            out = {"range": [int(self.lo), int(self.hi)], "int": True}
        elif self.form == "categorical":
            out = {"values": list(self.values), "weights": list(self.weights)}
        else:
            out = {
                "dist": self.dist,
                **{k: list(v) if isinstance(v, tuple) else v for k, v in self.options},
            }
        if self.circular:
            out["circular"] = True
        if not self.probes:
            out["probes"] = False
        if not self.stratify:
            out["stratify"] = False
        return out

    def with_domain(self, lo: Optional[float], hi: Optional[float]) -> "AxisSpec":
        """This axis with the bound field's valid range attached."""
        return replace(self, domain=(lo, hi))

    # ----------------------------------------------------------- sampling

    @property
    def is_random(self) -> bool:
        """False for a constant and for a link (which copies another axis)."""
        return self.form not in ("constant", "link")

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
        if self.form == "link":
            raise ValueError(f"axis {self.name!r} is a link: it copies {self.link!r}")
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
            found = None if dist.support is None else dist.support(self)
            return None if found is None else list(found)
        return None

    def field_values(self) -> FieldValues:
        """The values this axis can take (a link's are its target's).

        Raises:
            ValueError: For a link (resolve it to its target first).
        """
        if self.form == "link":
            raise ValueError(f"axis {self.name!r} is a link: it copies {self.link!r}")
        if self.form == "constant":
            return FieldValues.exactly(self.value)
        if self.form == "numeric":
            return FieldValues(lo=self.lo, hi=self.hi)
        if self.form == "int":
            return FieldValues(lo=self.lo, hi=self.hi, integer=True)
        support = self.support()
        if support is not None:
            return FieldValues(values=tuple(support))
        bounds = DISTRIBUTIONS[self.dist].bounds
        if bounds is not None:
            lo, hi = bounds(self)
            return FieldValues(lo=float(lo), hi=float(hi))
        return FieldValues()

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

    def quantile_values(
        self, labels: np.ndarray, bins: int, u: np.ndarray
    ) -> List[Any]:
        """``sample((b + u') / bins)``: values in equal-probability strata."""
        v = (np.asarray(labels, dtype=np.float64) + np.asarray(u)) / bins
        return DISTRIBUTIONS[self.dist].sample(v, self)

    def quantile_label(self, b: int, bins: int) -> str:
        """``"[sample(b/bins), sample((b+1)/bins))"``, 6 significant digits."""
        edges = DISTRIBUTIONS[self.dist].sample(
            np.array([b / bins, (b + 1) / bins]), self
        )
        return f"[{edges[0]:.6g}, {edges[1]:.6g})"

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
        if self.form == "link":
            raise ValueError(f"axis {self.name!r} is a link: it copies {self.link!r}")
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
