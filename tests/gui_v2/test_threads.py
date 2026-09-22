"""Worker threads stay referenced until they have finished (execution/threads.py)."""

import gc
import time

import pytest

pytestmark = pytest.mark.gui

from PyQt5 import QtCore  # noqa: E402

from sensoryforge.gui.execution import threads  # noqa: E402


class _Sleeper(QtCore.QObject):
    done = QtCore.pyqtSignal()

    def work(self):
        time.sleep(0.2)
        self.done.emit()


def test_a_thread_whose_owner_forgot_it_keeps_running_to_the_end(qtbot):
    thread = QtCore.QThread()
    worker = _Sleeper()
    worker.moveToThread(thread)
    thread.started.connect(worker.work)
    worker.done.connect(thread.quit)
    threads.keep_alive(thread, worker)
    thread.start()
    del thread, worker  # the owner is gone; a running QThread must not be freed
    gc.collect()
    assert threads.running() >= 1
    threads.wait_all()
    assert threads.running() == 0
