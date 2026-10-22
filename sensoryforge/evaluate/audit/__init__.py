"""The blind audit: measurements from a data set's generated data alone.

Reads bundles through :class:`DatasetReader` (the allow-list of the report contract,
section 2) and never imports ``sensoryforge.world`` or opens a world file. A2a is the
reader layer and the counts and metadata; the metrics follow in A2b and A2c.
"""

from sensoryforge.evaluate.audit.inventory import Inventory, inventory
from sensoryforge.evaluate.audit.layout import Layout
from sensoryforge.evaluate.audit.reader import (
    BundleInfo,
    DatasetReader,
    EntryRow,
    EntryView,
    PopulationView,
    ReadError,
)
from sensoryforge.evaluate.audit.units import UnitConflict, resolve_units

__all__ = [
    "BundleInfo",
    "DatasetReader",
    "EntryRow",
    "EntryView",
    "Inventory",
    "Layout",
    "PopulationView",
    "ReadError",
    "UnitConflict",
    "inventory",
    "resolve_units",
]
