"""The one place the GUI reads and writes persisted preferences.

Every expert-mode toggle, collapsible-section state and last-used workspace
goes through :func:`gui_settings`. Two reasons it is a single function
rather than ``QtCore.QSettings(...)`` scattered across the tabs:

* **Tests must not see or change a developer's real preferences.** Qt keeps
  these per user (a plist on macOS, a registry key on Windows, an ini file on
  Linux). The GUI tests used to read them, so a test passed on a machine where
  someone had once expanded a section and failed on a fresh CI runner, and
  running the suite could overwrite the preferences of whoever ran it.
  Setting ``SENSORYFORGE_SETTINGS_DIR`` redirects everything to an ini file in
  that directory; ``tests/conftest.py`` does so for every test session.
* **One name.** The tabs had used two different organisation/application
  pairs, one of them Qt's implicit default, so some preferences landed in an
  "unknown-organization" store.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore

ORGANIZATION = "SensoryForge"
APPLICATION = "GUI"
SETTINGS_DIR_ENV = "SENSORYFORGE_SETTINGS_DIR"


def gui_settings() -> QtCore.QSettings:
    """Return the settings store for the SensoryForge GUI.

    Returns:
        A ``QSettings`` backed by the platform's native per-user store, or by
        ``<SENSORYFORGE_SETTINGS_DIR>/SensoryForge-GUI.ini`` when that
        environment variable is set.
    """
    override = os.environ.get(SETTINGS_DIR_ENV)
    if override:
        path = os.path.join(override, f"{ORGANIZATION}-{APPLICATION}.ini")
        return QtCore.QSettings(path, QtCore.QSettings.IniFormat)
    return QtCore.QSettings(ORGANIZATION, APPLICATION)
