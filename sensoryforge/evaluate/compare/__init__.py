"""The compare stage (roadmap A4, report contract section 9).

Pairs the audit's measured values with the declared side's values and reports the raw
difference ``measured - declared``. It gives no verdict, converts no unit and reads
nothing but ``audit.json`` and ``declared.json`` with their entry files.
"""

from sensoryforge.evaluate.compare.distribution import (
    ks_distance,
    pair_histogram,
    wasserstein_distance,
)
from sensoryforge.evaluate.compare.pairing import STATS, pair_values
from sensoryforge.evaluate.compare.run import (
    COMPARE_FILE,
    ENTRIES_FILE,
    pair_summaries,
    run_compare,
)
from sensoryforge.evaluate.compare.table import (
    format_distribution_table,
    format_entry_table,
    format_histograms,
    format_stat_table,
)

__all__ = [
    "COMPARE_FILE",
    "ENTRIES_FILE",
    "STATS",
    "format_distribution_table",
    "format_entry_table",
    "format_histograms",
    "format_stat_table",
    "ks_distance",
    "pair_histogram",
    "pair_summaries",
    "pair_values",
    "run_compare",
    "wasserstein_distance",
]
