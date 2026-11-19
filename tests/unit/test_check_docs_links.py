"""Tests for scripts/check_docs_links.py (no network)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_links.py"
spec = importlib.util.spec_from_file_location("check_docs_links", SCRIPT)
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


def _site(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "index.html").write_text(
        '<a href="https://docs.python.org/x#frag">x</a>'
        '<a href="https://benefron.github.io/sensoryforge/a/">own</a>'
        '<a href="https://benefron.github.io/sensoryforge/missing/">gone</a>'
        '<link href="https://fonts.gstatic.com">',
        encoding="utf-8",
    )
    return tmp_path


def test_collect_skips_bare_origins_and_strips_anchors(tmp_path):
    found = links.collect(_site(tmp_path))
    assert "https://docs.python.org/x" in found
    assert "https://fonts.gstatic.com" not in found


def test_own_pages_are_judged_by_the_local_build(tmp_path):
    site = _site(tmp_path)
    assert links.own_page_exists(links.SITE_URL + "a/", site)
    assert not links.own_page_exists(links.SITE_URL + "missing/", site)
