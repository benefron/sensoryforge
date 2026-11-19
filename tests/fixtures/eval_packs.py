"""Metric packs that pass, or break one rule each, for the contract-check tests.

They live in a module of their own because the check reads the imports of the module
that defines a pack (a pack may not import the world), and the test module does.
"""

from __future__ import annotations

import random

from sensoryforge.evaluate.pack import MetricPack, metric_definition


class Counts(MetricPack):
    """Passes: counts the positive frame values of an entry."""

    name = "counts"
    version = "1"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("counts.frames.n", None, scope="entry")]

    def measure(self, view):
        return [{"id": "counts.frames.n", "value": int((view.frames > 0).sum())}]


class Stray(Counts):
    """Fails: an id outside the pack's own name."""

    name = "stray"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("core.frames.n", None, scope="entry")]


class Undefined(Counts):
    """Fails: a value for an id the catalogue does not define."""

    name = "undefined"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("undefined.frames.n", None, scope="entry")]

    def measure(self, view):
        return [{"id": "undefined.frames.m", "value": 1}]


class Raises(Counts):
    """Fails: measure() raises."""

    name = "raises"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("raises.frames.n", None, scope="entry")]

    def measure(self, view):
        raise ValueError("no")


class Random(Counts):
    """Fails: two runs differ."""

    name = "random"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("random.frames.n", None, scope="entry")]

    def measure(self, view):
        return [{"id": "random.frames.n", "value": random.random()}]


class Drift(Counts):
    """Fails: to_dict() does not survive from_config()."""

    name = "drift"

    def __init__(self, scale=1.0):
        self.scale = scale

    def to_dict(self):
        return {"scale": self.scale * 2}
