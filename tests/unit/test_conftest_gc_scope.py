"""The cyclic garbage collector stays on in every session (F-035 closed).

The GUI test harness used to call ``gc.disable()`` for any session that
collected ``gui``-marked tests, because a pyqtgraph ScatterPlotItem
segfaulted when the collector swept a stale ViewBox lambda left by the old
Grid tab. GUI v2 connects pyqtgraph signals only through
``plot_factory.connect`` and the old tab is deleted, so the workaround is
gone; the full suite runs with the collector enabled.
"""

import gc


def test_gc_is_enabled():
    assert gc.isenabled()
