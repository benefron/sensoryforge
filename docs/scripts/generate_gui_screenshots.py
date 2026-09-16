"""Regenerate the Circuit tab screenshots used by ``docs/user_guide/gui_walkthrough.md``.

Runs the GUI fully offscreen (``QT_QPA_PLATFORM=offscreen``, set automatically if not
already set) and grabs real pixmaps of the Circuit tab as it walks through the
pressure-simulation recipe (``sensoryforge/presets/tactile_sa1_ra1.yml``): the tab
right after loading the preset into the graph, the inspector showing a selected
node's parameters, and the tab after a run has produced a bundle. These are not
illustrations -- they are grabs of the real widget tree at each step, so a change to
the Circuit tab's layout shows up here the next time this script runs, instead of
the walkthrough silently going stale.

Usage:
    QT_QPA_PLATFORM=offscreen python docs/scripts/generate_gui_screenshots.py

Writes PNGs to ``docs/assets/gui/``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "gui"

# F-053: running this file directly puts its own directory (docs/scripts) on
# sys.path[0], not the repository root, so a plain `import sensoryforge`
# would fall through to whatever the environment's editable install points
# at -- which, with git worktrees in play, can be a different checkout
# entirely (see tests/docs/test_docs_examples.py for the same fix applied
# to docs/examples/*.py). Put this checkout's root first, explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.circuit.serialise import config_to_graph
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    config = SensoryForgeConfig.from_yaml(
        str(REPO_ROOT / "sensoryforge" / "presets" / "tactile_sa1_ra1.yml")
    )

    tab = CircuitTab()
    tab.resize(1400, 900)
    tab.show()
    app.processEvents()

    # 1. The graph right after loading the pressure-simulation preset.
    config_to_graph(config, tab.flowchart)
    app.processEvents()
    _grab(tab, "circuit_loaded_preset.png")

    # 2. A node selected, showing the inspector (P1/P2: params from
    #    get_param_spec() plus the reused Mechanoreceptor-tab-style preview).
    grid_node = tab.nodes().get("Main Grid")
    if grid_node is not None:
        tab.select_node(grid_node)
        app.processEvents()
        _grab(tab, "circuit_inspector_sensor_array.png")

    filter_node = tab.nodes().get("SA Population__filter")
    if filter_node is not None:
        tab.select_node(filter_node)
        app.processEvents()
        _grab(tab, "circuit_inspector_filter.png")

    # 3. Exercise a run (point the Record node at a temp bundle dir) as part
    #    of what this script proves still works -- but do not grab a fourth
    #    screenshot for it: the Circuit canvas itself does not change when a
    #    run finishes (results land on the Visualization tab, unchanged by
    #    Wave R), so a "circuit_after_run.png" would be pixel-identical to
    #    whichever node was last selected and would only mislead a reader
    #    into looking for a difference that is not there.
    record_node = tab.nodes().get("Record")
    with tempfile.TemporaryDirectory() as tmp_dir:
        if record_node is not None:
            record_node.from_config(
                {
                    "output_dir": tmp_dir,
                    "simulation": config.simulation.to_dict(),
                    "metadata": {},
                }
            )
        tab.run_graph(duration_ms=20.0)

    print(f"Wrote screenshots to {ASSETS_DIR}")


def _grab(widget, filename: str) -> None:
    pixmap = widget.grab()
    out_path = ASSETS_DIR / filename
    ok = pixmap.save(str(out_path))
    if not ok or pixmap.width() == 0 or pixmap.height() == 0:
        raise RuntimeError(
            f"Offscreen grab of the Circuit tab produced an unusable image for "
            f"{filename} ({pixmap.width()}x{pixmap.height()}, saved={ok}). "
            "Ship the walkthrough without screenshots rather than a placeholder."
        )
    print(f"  {out_path.relative_to(REPO_ROOT)}  ({pixmap.width()}x{pixmap.height()})")


if __name__ == "__main__":
    main()
