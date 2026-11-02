"""Find the registered names an agent file mentions and say which no longer exist (B10).

A skill names a preset, a metric pack, a metric id, a world shape or a component the way
a user would type it, so a rename in the code leaves the skill pointing at nothing. This
module reads the text of a skill, a prompt or the section and returns every name it can
check against a registry, and the ones the registries no longer hold. It checks only
places where a word is a registered name for certain (``--preset X``, ``--packs a,b``, a
metric id such as ``core.frames.peak``, ``kind: X`` and ``dist: X`` in a world snippet,
``neuron_model: X`` and the like in a config snippet). A placeholder (``NAME``,
``<pack>``) is not a name. Commands and flags are checked by the CLI parser in
``tests/unit/test_agent_skills.py``, links by the docs-page test.

Example:
    >>> missing_names("sensoryforge evaluate run --packs core,nonesuch")
    ['pack nonesuch (--packs)']
    >>> missing_names("sensoryforge evaluate run --packs core")
    []

Topic: testing
Tests: tests/unit/test_agent_names.py
Entry points: named_things, missing_names, known_names
Do not change without: a category is added only where the word is a registered name
    for certain; a guess turns a skill's prose into a failing test.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np

_PLACEHOLDER = re.compile(r"^(?:[A-Z][A-Z_]*(?::[A-Z][A-Z_]*)?|<[\w|-]+>|\.\.\.|…)$")
_FLAG = re.compile(r"--(preset|packs)[ =]([^\s`'\"\\)]+)")
_METRIC = re.compile(
    r"(?<![\w./-])([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})(?![\w/-])"
)
_KIND = re.compile(r"(?<![\w-])(kind|dist): *([A-Za-z][\w-]*)")
_CONFIG_KEYS = {
    "neuron_model": "NEURON_REGISTRY",
    "filter_method": "FILTER_REGISTRY",
    "innervation_method": "INNERVATION_REGISTRY",
}
_CONFIG = re.compile(r"(?<![\w-])(%s): *([A-Za-z][\w-]*)" % "|".join(_CONFIG_KEYS))


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_")


@lru_cache(maxsize=1)
def known_names() -> Dict[str, Set[str]]:
    """Every name the registries hold now, by category.

    Returns:
        ``{category: names}`` with ``preset``, ``pack``, ``metric``, ``world kind``,
        ``dist`` and one entry per component registry key of :data:`_CONFIG_KEYS`.
        Names are lower case with ``-`` read as ``_``.
    """
    # Imports are here so that a plugin loaded by an earlier import is counted.
    from sensoryforge import registry as reg
    from sensoryforge.evaluate.audit.layout import Layout
    from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY
    from sensoryforge.presets import list_presets
    from sensoryforge.world import distributions, kernel

    out: Dict[str, Set[str]] = {
        "preset": {_norm(n) for n in list_presets()},
        "pack": {_norm(n) for n in METRIC_PACK_REGISTRY.list_registered()},
        "dist": {_norm(n) for n in distributions.DISTRIBUTIONS},
        "world kind": {
            _norm(n)
            for table in (
                kernel.SHAPE_KINDS,
                kernel.PATTERN_KINDS,
                kernel.MODULATION_KINDS,
            )
            for n in table
        },
        "metric": set(),
    }
    layout = Layout(coords=np.zeros((4, 3)), channel_names=None, n_channels=1)
    for pack in out["pack"]:
        instance = METRIC_PACK_REGISTRY.get_class(pack)()
        out["metric"] |= {m["id"] for m in instance.metrics(layout, {}, 1.0)}
    for key, registry_name in _CONFIG_KEYS.items():
        registry = getattr(reg, registry_name)
        out[key] = {_norm(n) for n in registry.list_registered()}
    return out


def named_things(text: str) -> List[Tuple[str, str]]:
    """The checkable names in ``text`` as ``(category, name)``, in order of appearance.

    Args:
        text: The text of an agent file.

    Returns:
        Pairs where the category is one of :func:`known_names`'s keys (a ``--preset``
        value is ``preset``, a ``--packs`` list gives one ``pack`` each).
    """
    found: List[Tuple[int, str, str]] = []
    for match in _FLAG.finditer(text):
        flag, value = match.group(1), match.group(2).rstrip(".,;")
        for part in value.split(","):
            if part and not _PLACEHOLDER.match(part) and "<" not in part:
                found.append(
                    (match.start(), "preset" if flag == "preset" else "pack", part)
                )
    for match in _KIND.finditer(text):
        word = match.group(2)
        if not _PLACEHOLDER.match(word):
            category = "world kind" if match.group(1) == "kind" else "dist"
            found.append((match.start(), category, word))
    for match in _CONFIG.finditer(text):
        found.append((match.start(), match.group(1), match.group(2)))
    packs = known_names()["pack"]
    for match in _METRIC.finditer(text):
        if match.group(1).split(".")[0] in packs:
            found.append((match.start(), "metric", match.group(1)))
    return [(category, name) for _, category, name in sorted(found)]


def missing_names(text: str) -> List[str]:
    """Names in ``text`` that no registry holds, each as ``"category name (where)"``.

    A metric id also counts when it is the start of a registered one
    (``tactile.episode`` for ``tactile.episode.rise_time``), because a skill may name a
    family.

    Args:
        text: The text of an agent file.

    Returns:
        Empty when every checkable name exists.
    """
    known = known_names()
    out = []
    for category, name in named_things(text):
        if category == "metric":
            ok = any(m == name or m.startswith(name + ".") for m in known["metric"])
        else:
            ok = _norm(name) in known[category]
        if not ok:
            where = {"preset": "--preset", "pack": "--packs"}.get(category, category)
            out.append(f"{category} {name} ({where})")
    return out
