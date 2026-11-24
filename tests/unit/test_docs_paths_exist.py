"""Every docs page path named in code, tests, README, AGENTS.md, llms.txt exists.

A page that moves must take its references with it (Df1); this test fails when a
reference names ``docs/<...>.md`` and no such file exists.

Topic: docs-site paths
Tests: itself
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"docs/([A-Za-z0-9_./-]+?\.md)\b")
SCAN_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".cfg"}
# Records of what was, not live references.
SKIP_PREFIXES = (
    "docs/development/",
    "docs_root/",
    ".claude/rules/ledger/",
    "tests/agents/runs/",
    "CHANGELOG.md",
    ".claude/",
    "tests/unit/test_docs_paths_exist.py",
    "tests/unit/test_file_headers.py",  # fixtures that name pages on purpose
)
# Paths a template or a message names on purpose, never as a page of this repo.
ALLOWED = {"docs/foo.md", "docs/page.md"}


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [
        f
        for f in out
        if f
        and Path(f).suffix in SCAN_SUFFIXES
        and not f.startswith(SKIP_PREFIXES)
        and (ROOT / f).is_file()
    ]


def test_every_docs_path_named_in_the_repo_exists():
    missing = []
    for rel in _tracked_files():
        # pages name siblings by relative link; the strict build checks those
        if rel.startswith("docs/") and rel.endswith(".md"):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        for m in PATTERN.finditer(text):
            path = "docs/" + m.group(1)
            if path.startswith(
                "docs/development/"
            ):  # engineering records, not the site
                continue
            if path in ALLOWED or "*" in path or "<" in path:
                continue
            if not (ROOT / path).exists():
                missing.append(f"{rel}: {path}")
    assert not missing, "docs paths that do not exist:\n" + "\n".join(
        sorted(set(missing))
    )


# A public page carries no ledger id, roadmap label or internal path (Df2). The one
# exception is the page that lists the design decisions with their ids.
LABEL = re.compile(
    r"\b[FDCRA]-(?:[0-9a-f]{7}|\d{2,3}|Q\d)\b|\bWave [A-Z]\d?\b|\bPhase \d|docs_root"
)
PUBLIC_EXCEPTIONS = {"docs/agents/decisions.md"}


def test_no_public_page_carries_a_ledger_id_or_roadmap_label():
    found = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("docs/development/") or rel in PUBLIC_EXCEPTIONS:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            m = LABEL.search(line)
            if m:
                found.append(f"{rel}:{n}: {m.group(0)}")
    assert not found, "\n".join(found)


def test_the_preset_descriptions_carry_no_ledger_id_or_roadmap_label():
    from sensoryforge.presets import preset_description, list_presets

    for name in list_presets():
        assert not LABEL.search(preset_description(name)), name
