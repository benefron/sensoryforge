"""Regression for task C4: the GC workaround must not leak into non-GUI sessions.

conftest.py's pytest_collection_modifyitems only calls gc.disable() when the
session collects at least one `gui`-marked test. This file carries no `gui`
marker itself, so under `pytest -m "not gui"` (where no gui-marked item is
collected at all) the garbage collector must stay enabled. The check is
written against the session's actual collected items rather than a bare
`assert gc.isenabled()` so it also holds -- correctly, not contradictorily --
under an unfiltered `pytest` run that collects this file alongside gui-marked
ones (where gc.disable() is expected to have fired for the whole session).
"""

import gc


def test_gc_enabled_iff_no_gui_tests_collected(request):
    collected_gui = any(
        item.get_closest_marker("gui") is not None for item in request.session.items
    )
    if collected_gui:
        assert not gc.isenabled(), (
            "gc should be disabled once any gui-marked test is collected in "
            "this session (F-035 workaround, scoped in conftest.py's "
            "pytest_collection_modifyitems)"
        )
    else:
        assert gc.isenabled(), (
            "gc must stay enabled when the session collects no gui-marked "
            "tests (F-035: gc.disable() is a workaround for a GUI-only "
            "pyqtgraph crash and should not cost non-GUI sessions anything)"
        )
