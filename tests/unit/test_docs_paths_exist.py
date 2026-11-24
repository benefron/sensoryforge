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
