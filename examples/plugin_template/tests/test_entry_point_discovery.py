"""V1: prove discovery through the real entry-point mechanism, not an
in-process registry call.

Mirrors the technique from ``tests/unit/test_circuit_plugin_palette.py``
(Wave P, P3): this checkout's *real* ``sensoryforge_plugin_template`` source
is copied into a scratch directory next to a hand-written
``sensoryforge-plugin-template-0.1.0.dist-info/entry_points.txt`` (parsed
from this package's own ``pyproject.toml``, so the test fails if the two
drift apart), that directory is put on ``sys.path``, and the *unmodified*
``importlib.metadata.entry_points`` call --
:func:`sensoryforge.plugins.discover_entry_point_plugins` makes exactly this
call -- is used to find it. No ``FILTER_REGISTRY.register()``/
``INNERVATION_REGISTRY.register()``/``PROCESSING_REGISTRY.register()`` call
appears anywhere in this file: every registration below happens because the
real component-discovery path found and ran ``register()``.

This is genuine entry-point discovery without ``pip install`` (F-053 forbids
installing into the shared conda environment from a worktree); the
accompanying report also records one real ``pip install`` of the built wheel
into a disposable virtualenv, which this in-process test cannot substitute
for.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_NAME = "sensoryforge_plugin_template"
_DIST_NAME = "sensoryforge-plugin-template"
_DIST_VERSION = "0.1.0"


def _entry_points_from_pyproject() -> dict:
    """Parse the ``[project.entry-points."sensoryforge.components"]`` table
    out of this package's own ``pyproject.toml`` (no ``tomllib`` dependency
    assumed for Python 3.10), so the dist-info this test builds cannot
    silently drift from what actually ships."""
    text = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split('[project.entry-points."sensoryforge.components"]')[1]
    section = section.split("\n[")[0]
    entries = {}
    for line in section.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, _, target = line.partition("=")
        entries[name.strip()] = target.strip().strip('"')
    return entries


@pytest.fixture
def installed_layout(tmp_path, monkeypatch):
    """A real copy of the plugin's source plus a real ``*.dist-info``,
    exactly the on-disk shape ``pip install`` (editable or not) produces,
    with nothing about SensoryForge's registries faked."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    shutil.copytree(_PACKAGE_ROOT / _PACKAGE_NAME, site_dir / _PACKAGE_NAME)

    entry_points = _entry_points_from_pyproject()
    assert entry_points, "pyproject.toml declared no sensoryforge.components entries"

    dist_info = site_dir / f"{_DIST_NAME}-{_DIST_VERSION}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n" f"Name: {_DIST_NAME}\n" f"Version: {_DIST_VERSION}\n"
    )
    entry_points_txt = "[sensoryforge.components]\n" + "\n".join(
        f"{name} = {target}" for name, target in entry_points.items()
    )
    (dist_info / "entry_points.txt").write_text(entry_points_txt + "\n")

    monkeypatch.syspath_prepend(str(site_dir))
    importlib.invalidate_caches()
    yield entry_points
    for name in list(sys.modules):
        if name == _PACKAGE_NAME or name.startswith(_PACKAGE_NAME + "."):
            del sys.modules[name]


def test_real_entry_points_scan_finds_both_components(installed_layout):
    """Step 1: the unmodified ``importlib.metadata`` scan sees both entries
    -- not a monkeypatched fake, not an in-process registration."""
    eps = importlib.metadata.entry_points(group="sensoryforge.components")
    names = {ep.name for ep in eps}
    for expected_name in installed_layout:
        assert expected_name in names


def test_discovery_registers_both_components_for_real(installed_layout):
    """Step 2: SensoryForge's own discovery function (the one
    ``register_components.register_all()`` calls at import time) finds and
    runs both ``register()`` functions."""
    from sensoryforge.registry import INNERVATION_REGISTRY, PROCESSING_REGISTRY
    from sensoryforge.register_components import register_all

    register_all()
    assert INNERVATION_REGISTRY.is_registered("radial_falloff")
    assert PROCESSING_REGISTRY.is_registered("gain_threshold")


def test_components_appear_in_list_components_cli(installed_layout):
    """Step 3 (V1's stated proof): the component names show up in
    ``sensoryforge list-components``, run as the real subprocess entry
    point would run it, with the scratch site directory on ``PYTHONPATH``
    so the subprocess's own ``importlib.metadata`` scan sees the same
    on-disk dist-info this test built (a subprocess does not inherit the
    in-process ``sys.path`` a fixture-only ``monkeypatch`` would give it)."""
    import os

    site_dir = None
    for entry in sys.path:
        if entry.endswith("/site") or entry.endswith("\\site"):
            site_dir = entry
            break
    assert site_dir is not None, "installed_layout fixture did not prepend a site dir"

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = site_dir + (os.pathsep + existing if existing else "")

    result = subprocess.run(
        [sys.executable, "-m", "sensoryforge.cli", "list-components"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "radial_falloff" in result.stdout
    assert "gain_threshold" in result.stdout
