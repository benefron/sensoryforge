"""Smoke test for scripts/tune_adex_populations.py (Phase 2b, T2).

Runs the script's ``main()`` entry point in ``--quick`` mode against a
``tmp_path`` output directory and asserts the markdown report and at least
one PNG are written. Kept fast (quick mode; no --grid override needed --
the script's own ``--quick`` run over the 80x80 preset already completes
in well under 20s on CPU).
"""

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "tune_adex_populations.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("tune_adex_populations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tune_adex_populations_writes_report_and_png(tmp_path):
    """The quick-mode run writes the markdown report and at least one PNG."""
    module = _load_script_module()
    result = module.main(
        out_dir=tmp_path,
        quick=True,
        seed=0,
        input_gain_override=None,
    )

    md_path = result["md_path"]
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8").strip() != ""

    png_paths = result["png_paths"]
    assert len(png_paths) >= 1
    assert any(p.exists() for p in png_paths)
