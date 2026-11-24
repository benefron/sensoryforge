"""Check the external links of the built docs site.

Run ``mkdocs build --strict`` first (it fails on a broken link or anchor inside
the site, see ``validation`` in ``mkdocs.yml``), then run this script.
It reads every ``href`` that starts with ``http`` in ``site/**/*.html`` and asks
each distinct address once; an address under the published site is checked against the
local build. It needs the network, so CI runs it on demand and monthly (the
``docs-links`` workflow), not on every push.

Exit status is 1 when an address answers 404 or 410 or cannot be reached; a 403, 429 or
5xx is reported as "unverified", since sites often refuse automated clients.

Topic: docs link check
Tests: tests/unit/test_check_docs_links.py
Entry points: python scripts/check_docs_links.py [--site site] [--timeout 15]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HREF = re.compile(r'href="(https?://[^"#]+)')
SKIP_PREFIXES = ("http://localhost", "http://127.0.0.1", "https://example.")
AGENT = "sensoryforge-docs-link-check"
SITE_URL = "https://benefron.github.io/sensoryforge/"


def is_bare_origin(url: str) -> bool:
    """True for ``https://host`` with no path (a preconnect hint, not a page)."""
    return url.count("/") <= 2


def own_page_exists(url: str, site: Path) -> bool:
    """Whether an address under the published site is a page in the local build."""
    rel = url[len(SITE_URL) :].strip("/")
    target = site / rel
    return target.is_file() or (target / "index.html").is_file()


def collect(site: Path) -> dict[str, list[str]]:
    """Map each external address to the pages (relative to `site`) that link to it."""
    found: dict[str, list[str]] = {}
    for page in sorted(site.rglob("*.html")):
        for url in HREF.findall(page.read_text(encoding="utf-8", errors="replace")):
            if url.startswith(SKIP_PREFIXES) or is_bare_origin(url):
                continue
            pages = found.setdefault(url, [])
            rel = page.relative_to(site).as_posix()
            if rel not in pages:
                pages.append(rel)
    return found


def check(url: str, timeout: float) -> tuple[str, str]:
    """Return ``(verdict, detail)``; verdict is ``ok``, ``broken`` or ``unverified``."""
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return "ok", str(reply.status)
        except urllib.error.HTTPError as err:
            if err.code in (404, 410):
                return "broken", str(err.code)
            if method == "HEAD":
                continue  # some servers refuse HEAD; ask again with GET
            return "unverified", str(err.code)
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            if method == "HEAD":
                continue
            return "broken", str(getattr(err, "reason", err))
    return "unverified", "no answer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default="site", help="the built site (default: site)")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    site = Path(args.site)
    if not site.is_dir():
        print(
            f"{site} is not a folder; run `mkdocs build --strict` first",
            file=sys.stderr,
        )
        return 2
    found = collect(site)
    own = [u for u in found if u.startswith(SITE_URL)]
    remote = [u for u in found if u not in own]
    # An address under the site itself is judged by the local build (not live until
    # the site is deployed).
    results = [
        (u, "ok" if own_page_exists(u, site) else "broken", "not in the build")
        for u in own
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results += list(
            pool.map(lambda u: (u, *check(u, args.timeout)), sorted(remote))
        )
    broken = [r for r in results if r[1] == "broken"]
    unverified = [r for r in results if r[1] == "unverified"]
    for url, _, detail in broken:
        print(f"BROKEN {detail}: {url}  (on {', '.join(found[url][:3])})")
    for url, _, detail in unverified:
        print(f"unverified {detail}: {url}")
    print(
        f"{len(results)} addresses: {len(broken)} broken, {len(unverified)} unverified"
    )
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
