"""The agent header on a module's docstring (roadmap B5).

The checker is tested on small synthetic modules, then run on the real tree: every
module that needs a header has a true one, except those in the shrinking baseline.
"""

import textwrap
from pathlib import Path

import pytest

from sensoryforge.testing import file_headers as fh

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("file_headers_untagged.txt")
PAGE = REPO / "docs" / "developer_guide" / "file_headers.md"


def _module(root, body, rel="pkg/mod.py"):
    """Write ``body`` (dedented) as ``root/rel`` and return the path."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


GOOD = '''\
"""Do one thing.

Longer prose.

Topic: engine, config
Tests: tests/test_mod.py
Entry points: run, Thing, Thing.go
Pinned by: docs/contract.md, F-053
Do not change without: the contract page, because a consumer pins this.
    A second line of the same note.
Units: ms
"""


def run():
    pass


class Thing:
    def go(self):
        pass
'''


@pytest.fixture
def tree(tmp_path):
    """A tiny repo: one test file, one docs file, and ``pkg/mod.py``."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "contract.md").write_text("", encoding="utf-8")
    _module(tmp_path, GOOD)
    return tmp_path


def test_parse_header_reads_tags_and_continuation_lines():
    header = fh.parse_header(GOOD.split('"""')[1])
    assert header["Topic"] == "engine, config"
    assert header["Entry points"] == "run, Thing, Thing.go"
    assert header["Do not change without"].endswith("A second line of the same note.")
    assert "Shapes" not in header


def test_a_true_header_has_no_problems(tree):
    assert fh.check_module(tree / "pkg" / "mod.py", tree) == []


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("Topic: engine, config", "Topic: engine, wibble", "wibble"),
        ("Tests: tests/test_mod.py", "Tests: tests/nope.py", "tests/nope.py"),
        ("Entry points: run,", "Entry points: run, ghost,", "ghost"),
        ("Thing.go", "Thing.stop", "Thing.stop"),
        ("docs/contract.md", "docs/missing.md", "docs/missing.md"),
        ("F-053", "ledger entry", "ledger entry"),
        ("Tests: tests/test_mod.py\n", "", "Tests"),
        ("Units: ms\n", "Units: ms\nUnits: mm\n", "twice"),
    ],
)
def test_check_module_names_each_kind_of_untruth(tree, old, new, fragment):
    path = tree / "pkg" / "mod.py"
    path.write_text(GOOD.replace(old, new), encoding="utf-8")
    problems = fh.check_module(path, tree)
    assert problems and any(fragment in p for p in problems), problems


def test_a_module_without_a_docstring_is_reported(tree):
    path = _module(tree, "x = 1\n", "pkg/bare.py")
    assert any("docstring" in p for p in fh.check_module(path, tree))


def test_needs_header_skips_glue_and_short_files(tree):
    assert fh.needs_header(tree / "pkg" / "mod.py") is False  # 28 lines
    long_file = _module(tree, "x = 1\n" * fh.MIN_LINES, "pkg/long.py")
    assert fh.needs_header(long_file) is True
    for glue in ("__init__.py", "__main__.py"):
        glue_file = _module(tree, "x = 1\n" * fh.MIN_LINES, "pkg/" + glue)
        assert fh.needs_header(glue_file) is False


def test_every_module_that_needs_a_header_has_a_true_one():
    """The ratchet: tagged modules stay true, untagged ones are only the baseline."""
    baseline = set(BASELINE.read_text(encoding="utf-8").split())
    failures, missing = [], []
    for path in fh.source_modules(REPO):
        rel = path.relative_to(REPO).as_posix()
        if rel in baseline:
            continue
        problems = fh.check_module(path, REPO)
        if problems:
            failures.append(f"{rel}: {'; '.join(problems)}")
        elif not fh.parse_header(fh.docstring_of(path) or ""):
            missing.append(rel)
    assert not failures, "\n".join(failures)
    assert not missing, missing


def test_baseline_only_shrinks():
    """A baseline entry that now has a header, or no longer exists, must go."""
    stale = []
    for rel in BASELINE.read_text(encoding="utf-8").split():
        path = REPO / rel
        if not path.exists() or not fh.needs_header(path):
            stale.append(f"{rel}: gone or no longer needs a header")
        elif fh.parse_header(fh.docstring_of(path) or ""):
            stale.append(f"{rel}: has a header, remove it from the baseline")
    assert not stale, "\n".join(stale)


def test_the_page_documents_every_tag_and_topic():
    text = PAGE.read_text(encoding="utf-8")
    for tag in fh.TAGS:
        assert f"`{tag}:`" in text, f"tag {tag!r} is not on the page"
    for topic in fh.TOPICS:
        assert f"`{topic}`" in text, f"topic {topic!r} is not on the page"
