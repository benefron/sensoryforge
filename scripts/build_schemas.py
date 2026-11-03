#!/usr/bin/env python
"""Write (or check) the JSON Schemas in ``schemas/`` from the live registries (B3).

    python scripts/build_schemas.py          # rewrite schemas/*.schema.json
    python scripts/build_schemas.py --check  # exit 1 if a committed file is stale

The same text is printed by ``sensoryforge describe --schema <name>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sensoryforge.describe import SCHEMAS, schema_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "schemas"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    args = parser.parse_args()
    stale = []
    for name in SCHEMAS:
        path = OUT / f"{name}.schema.json"
        text = schema_text(name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"stale: {', '.join(stale)}; run python scripts/build_schemas.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
