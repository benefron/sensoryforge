"""Regenerate the GUI screenshots used by ``docs/user_guide/gui_walkthrough.md``.

Runs the GUI fully offscreen (``QT_QPA_PLATFORM=offscreen``, set automatically if
not already set), loads the pressure-simulation recipe
(``sensoryforge/presets/tactile_sa1_ra1.yml``), performs a short real run, and
grabs the window on each of the five screens. These are grabs of the real
widget tree, not illustrations, so a layout change shows up the next time this
script runs instead of the walkthrough silently going stale.

Usage:
    QT_QPA_PLATFORM=offscreen python docs/scripts/generate_gui_screenshots.py
    python docs/scripts/generate_gui_screenshots.py --output-dir /tmp/shots

Writes ``gui_sensors.png``, ``gui_stimulus.png``, ``gui_populations.png``,
``gui_results.png`` and ``gui_batch.png`` to ``docs/assets/gui/`` by default.
``--output-dir`` sends them elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "gui"
PRESET = REPO_ROOT / "sensoryforge" / "presets" / "tactile_sa1_ra1.yml"

# F-053: running this file directly puts docs/scripts on sys.path[0], not the
# repository root, so `import sensoryforge` could resolve to another checkout
# through an editable install. Put this checkout first, explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Window size of every screenshot, in pixels.
WINDOW_SIZE = (1400, 900)
#: Length of the run shown on the Run & Results screenshot, in ms.
RUN_MS = 300.0


def main(output_dir: Path = ASSETS_DIR) -> list:
    """Write one PNG per screen; return their paths."""
    # Preferences go to a throwaway folder so a screenshot never depends on,
    # or changes, the user's own settings (F-072).
    os.environ.setdefault("SENSORYFORGE_SETTINGS_DIR", tempfile.mkdtemp())

    from PyQt5 import QtCore, QtTest, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui import theme
    from sensoryforge.gui.app import STAGE_ORDER, SensoryForgeApp
    from sensoryforge.gui.session import Session

    theme.apply(app)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = Session(SensoryForgeConfig.from_yaml_file(PRESET))
    window = SensoryForgeApp(session)
    window.resize(*WINDOW_SIZE)
    window.show()

    finished = []
    window.run_controller.finished.connect(lambda *_: finished.append(True))
    window.run_bar.duration_spin.setValue(RUN_MS)
    window.run_bar.run_button.click()
    deadline = QtCore.QDeadlineTimer(120_000)
    while not finished and not deadline.hasExpired():
        QtTest.QTest.qWait(100)
    if not finished:
        raise RuntimeError("the run did not finish within 120 s")

    written = []
    for row, stage in enumerate(STAGE_ORDER):
        window.stage_list.setCurrentRow(row)
        QtTest.QTest.qWait(800)  # debounced previews render
        path = output_dir / f"gui_{stage}.png"
        window.grab().save(str(path))
        written.append(path)
    window.close()
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Where to write the PNGs (default: docs/assets/gui/).",
    )
    for image in main(parser.parse_args().output_dir):
        print(image, flush=True)
    os._exit(0)  # skip Qt's interpreter-teardown crash (F-016)
