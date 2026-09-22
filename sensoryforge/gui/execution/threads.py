"""Keep every running worker thread alive until it has finished.

A ``QThread`` destroyed while its thread still runs aborts the process
("QThread: Destroyed while thread is still running"). A widget that owns a
worker thread can be destroyed at any time (its window closes, a test ends),
taking its only reference to the thread with it. Every GUI worker thread is
therefore registered here from start to finish, independently of the widget
that started it, and :func:`wait_all` joins whatever is still running at
shutdown.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtCore, sip

#: Running thread -> the worker object running in it (also kept alive).
_RUNNING: Dict[QtCore.QThread, Optional[QtCore.QObject]] = {}


def keep_alive(thread: QtCore.QThread, worker: Optional[QtCore.QObject] = None) -> None:
    """Hold ``thread`` (and the ``worker`` moved into it) until it finishes.

    Call before ``thread.start()``.

    Args:
        thread: A parentless worker thread.
        worker: The object whose slot runs in ``thread``.
    """
    # Threads that have fully stopped are released here, on the GUI thread --
    # never from their own "finished" signal, which fires inside the thread
    # before it has stopped (dropping the last reference there is the very
    # abort this module prevents).
    for done in [t for t in _RUNNING if _gone(t) or t.isFinished()]:
        del _RUNNING[done]
    _RUNNING[thread] = worker


def _gone(thread: QtCore.QThread) -> bool:
    """Whether Qt has already deleted ``thread`` (an owner's deleteLater)."""
    return sip.isdeleted(thread)


def running() -> int:
    """How many registered threads have not finished."""
    return sum(1 for t in _RUNNING if not _gone(t) and t.isRunning())


def wait_all(timeout_ms: int = 30_000) -> None:
    """Wait for every registered thread to finish (shutdown and tests).

    Args:
        timeout_ms: How long to wait for each thread, in ms.
    """
    for thread in list(_RUNNING):
        if _gone(thread):
            del _RUNNING[thread]
            continue
        thread.quit()
        thread.wait(timeout_ms)
