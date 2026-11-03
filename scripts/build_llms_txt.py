"""Write ``llms.txt``, an index of the published docs for tools that read one.

The index follows the llms.txt layout (a title, a one-line summary, then one section of
links per top-level docs group) and is built from ``mkdocs.yml`` alone: its
``site_name``, ``site_description``, ``site_url`` and ``nav``. Nothing is written by
hand, so the index cannot drift from the site; ``tests/unit/test_front_door.py`` fails
when the committed file differs from what this script writes.

Usage:
    python scripts/build_llms_txt.py            # rewrite llms.txt at the repo root
    python scripts/build_llms_txt.py --check    # exit 1 if llms.txt is out of date
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _NavLoader(yaml.SafeLoader):
    """A safe loader that reads mkdocs' ``!!python/name:`` tags as plain strings."""


_NavLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def _page_url(site_url: str, page: str) -> str:
    """The published URL of a docs page (mkdocs' directory URLs)."""
    path = page[: -len(".md")] if page.endswith(".md") else page
    if path == "index":
        path = ""
    elif path.endswith("/index"):
        path = path[: -len("index")]
    else:
        path += "/"
    return f"{site_url.rstrip('/')}/{path}"


def _links(items: List[Any], site_url: str, depth: int = 0) -> List[str]:
    """Flatten a nav subtree into ``- [Title](url)`` lines, nesting by indent."""
    lines = []
    for item in items:
        for title, value in item.items():
            pad = "  " * depth
            if isinstance(value, list):
                lines.append(f"{pad}- {title}")
                lines.extend(_links(value, site_url, depth + 1))
            else:
                lines.append(f"{pad}- [{title}]({_page_url(site_url, value)})")
    return lines


def build(mkdocs_yml: Path = ROOT / "mkdocs.yml") -> str:
    """Return the ``llms.txt`` text for the docs described by ``mkdocs_yml``."""
    config = yaml.load(mkdocs_yml.read_text(encoding="utf-8"), Loader=_NavLoader)
    site_url = config["site_url"]
    repo_url = config.get("repo_url", "").rstrip("/")
    name = config["site_name"].replace(" Documentation", "")
    out = [
        f"# {name}",
        "",
        f"> {config['site_description'].strip()}",
        "",
        f"Coding agents working in the repository start at "
        f"[AGENTS.md]({repo_url}/blob/main/AGENTS.md).",
        "",
    ]
    singles = []
    for item in config["nav"]:
        for title, value in item.items():
            if isinstance(value, list):
                out += [f"## {title}", ""] + _links(value, site_url) + [""]
            else:
                singles.append({title: value})
    if singles:
        out += ["## Other", ""] + _links(singles, site_url) + [""]
    return "\n".join(out)


def main() -> int:
    """Write or check ``llms.txt`` at the repository root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if llms.txt is out of date"
    )
    args = parser.parse_args()
    target = ROOT / "llms.txt"
    text = build()
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current != text:
            print("llms.txt is out of date: run python scripts/build_llms_txt.py")
            return 1
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
