"""Counter-based randomness for worlds (spec §4.1).

Every random choice in a world is a pure function of integers: a 64-bit
splitmix64 mix of a seed, a draw's index and a *slot* naming the choice.
Nothing is drawn from a stream, so draw ``i`` depends only on ``(seed, i)``
-- never on how many draws were asked for -- and the bits are the same on
every machine (integer arithmetic plus one exact scaling to ``[0, 1)``).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Union

import numpy as np

#: Seeds are kept below 2**53 so they survive JSON in any language.
SEED_LIMIT = 2**53

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SECOND = np.uint64(0xD1B54A32D192ED03)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MASK64 = (1 << 64) - 1

Part = Union[int, str, np.integer]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """One splitmix64 output per element of ``x`` (uint64, wrapping)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def key(part: Part) -> int:
    """A 64-bit integer for one part of a key: ints mod 2**64, strings by SHA-256."""
    if isinstance(part, str):
        return int.from_bytes(
            hashlib.sha256(part.encode("utf-8")).digest()[:8], "little"
        )
    if isinstance(part, bool):
        raise TypeError("a hash key part must be an int or a str, not a bool")
    return int(part) & _MASK64


def mix(a, b) -> np.ndarray:
    """Hash two uint64 values (or arrays of them, broadcasting) into one."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return _splitmix64(_splitmix64(a) ^ _splitmix64(b + _SECOND))


def hash_parts(*parts: Part) -> int:
    """A 64-bit hash of a sequence of ints and strings."""
    if not parts:
        raise ValueError("hash_parts needs at least one part")
    h = np.asarray(key(parts[0]), dtype=np.uint64)
    for part in parts[1:]:
        h = mix(h, key(part))
    return int(h)


def seed53(*parts: Part) -> int:
    """A seed below 2**53 derived from ``parts`` (the hash's top 53 bits)."""
    return hash_parts(*parts) >> 11


def draw_seeds(seed: int, indices: Iterable[int]) -> np.ndarray:
    """Per-draw seeds ``H(seed, i)`` (top 53 bits) for each index ``i``."""
    idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
    idx = idx.astype(np.int64, copy=False)
    if idx.size and int(idx.min()) < 0:
        raise ValueError("draw indices must be >= 0")
    return mix(np.uint64(key(seed)), idx.astype(np.uint64)) >> np.uint64(11)


def uniforms(seeds, slot: Part) -> np.ndarray:
    """``u = H(seed, slot)`` mapped exactly to ``[0, 1)``, one per seed (float64)."""
    bits = mix(np.asarray(seeds, dtype=np.uint64), key(slot)) >> np.uint64(11)
    return bits.astype(np.float64) * (1.0 / SEED_LIMIT)


def permutation(n: int, *parts: Part) -> np.ndarray:
    """A permutation of ``range(n)`` determined by ``parts`` (stable argsort)."""
    u = uniforms(draw_seeds(seed53(*parts), np.arange(n)), "permutation")
    return np.argsort(u, kind="stable")
