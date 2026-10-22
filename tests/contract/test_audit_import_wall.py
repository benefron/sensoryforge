"""The blind audit never imports the world package, and its core says nothing tactile.

Roadmap task A2 (D-c52e668, report contract section 2 and design principle 1). Three
checks: a static import graph over every module reachable from the audit and from a
metric pack, a clean subprocess that really reads a data set and then looks at
``sys.modules``, and a vocabulary lint over the core audit source.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = "sensoryforge"
WALLED = "sensoryforge.world"
EVAL = REPO / PACKAGE / "evaluate"


def _module_file(name: str, src_root: Path) -> Optional[Path]:
    """The file a dotted module name resolves to under ``src_root``, if any."""
    base = src_root.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _import_nodes(tree: ast.AST, import_time_only: bool) -> Iterable[ast.AST]:
    """The import statements of a module; with ``import_time_only``, those that run when
    it is imported (module and class level, any branch), not inside a function body."""
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if import_time_only and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child
            stack.append(child)


def _imported_names(
    path: Path, module: str, is_pkg: bool, import_time_only: bool
) -> Set[str]:
    """Every module name an ``import`` in ``path`` could load."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = module if is_pkg else module.rpartition(".")[0]
    names: Set[str] = set()
    for node in _import_nodes(tree, import_time_only):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        else:
            if node.level:
                parts = pkg.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def reachable(
    roots: Iterable[Path], src_root: Path, package: str, strict: str = ""
) -> Dict[str, List[str]]:
    """Every ``package`` module reachable from ``roots``, with the chain reaching it.

    Importing ``a.b.c`` also runs ``a/__init__`` and ``a/b/__init__``, so the parent
    packages are followed too. Modules under ``strict`` (the evaluation code, which the
    wall is about) are read in full, imports inside functions included. Every other
    module is followed only through the imports that run when it is imported: its lazy
    imports are not the audit's, and the clean-subprocess test covers what really loads.

    Returns:
        ``{module: chain}`` where ``chain`` lists the modules from a root to it.
    """
    seen: Dict[str, List[str]] = {}
    todo: List[List[str]] = []
    for r in roots:
        rel = r.relative_to(src_root).with_suffix("")
        name = ".".join(rel.parts)
        name = name[: -len(".__init__")] if name.endswith(".__init__") else name
        todo.append([name])
    while todo:
        chain = todo.pop()
        name = chain[-1]
        if name in seen or not (name == package or name.startswith(package + ".")):
            continue
        path = _module_file(name, src_root)
        if path is None:
            continue
        seen[name] = chain
        is_pkg = path.name == "__init__.py"
        lazy_too = bool(strict) and (name == strict or name.startswith(strict + "."))
        wanted = set(_imported_names(path, name, is_pkg, not lazy_too))
        for dotted in list(wanted):
            parts = dotted.split(".")
            wanted.update(".".join(parts[:i]) for i in range(1, len(parts)))
        parent = name.rpartition(".")[0]
        if parent:
            wanted.add(parent)
        for nxt in sorted(wanted):
            if nxt not in seen:
                todo.append(chain + [nxt])
    return seen


def _wall_roots() -> List[Path]:
    """The audit package, and every metric pack and registry module that exists."""
    roots = sorted((EVAL / "audit").rglob("*.py"))
    roots += [EVAL / "__init__.py"]
    roots += sorted(EVAL.glob("*/metrics.py")) + sorted(EVAL.glob("*/metrics/*.py"))
    roots += [p for p in (EVAL / "registry.py",) if p.is_file()]
    return roots


def _breaches(seen: Dict[str, List[str]], walled: str) -> Dict[str, List[str]]:
    return {
        m: chain
        for m, chain in seen.items()
        if m == walled or m.startswith(walled + ".")
    }


def test_nothing_reachable_from_the_audit_imports_the_world():
    seen = reachable(_wall_roots(), REPO, PACKAGE, "sensoryforge.evaluate")
    assert "sensoryforge.evaluate.audit.reader" in seen  # the walk found the audit
    assert "sensoryforge.evaluate.audit" in seen
    breaches = _breaches(seen, WALLED)
    assert not breaches, "the audit reaches the world package: " + "; ".join(
        " -> ".join(c) for c in breaches.values()
    )


def test_the_walker_finds_a_transitive_import(tmp_path):
    """A planted a -> b -> world chain (lazy, relative and from-style) is found."""
    pkg = tmp_path / "toypkg"
    (pkg / "audit").mkdir(parents=True)
    (pkg / "world").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "world" / "__init__.py").write_text("")
    (pkg / "audit" / "__init__.py").write_text("")
    (pkg / "audit" / "a.py").write_text("from . import b\n")
    (pkg / "audit" / "b.py").write_text("def f():\n    from toypkg.helpers import h\n")
    (pkg / "helpers.py").write_text(
        "try:\n    from toypkg import world\nexcept ImportError:\n    pass\n"
    )
    seen = reachable([pkg / "audit" / "a.py"], tmp_path, "toypkg", "toypkg.audit")
    assert _breaches(seen, "toypkg.world")
    assert seen["toypkg.world"][0] == "toypkg.audit.a"
    clean = reachable(
        [pkg / "audit" / "__init__.py"], tmp_path, "toypkg", "toypkg.audit"
    )
    assert not _breaches(clean, "toypkg.world")


def test_a_clean_process_reads_a_data_set_without_loading_the_world(toy_dataset):
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        from pathlib import Path
        from sensoryforge.evaluate.audit import DatasetReader, inventory
        reader = DatasetReader(Path(sys.argv[2]))
        inv = inventory(reader)
        view = reader.read_entry(reader.rows[0].entry)
        assert inv.n_ok == 12 and view.frames.ndim == 3
        loaded = sorted(
            m for m in sys.modules if m.split(".")[:2] == ["sensoryforge", "world"]
        )
        print(loaded)
        """)
    out = subprocess.run(
        [sys.executable, "-I", "-c", code, str(REPO), str(toy_dataset.data)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"the world package was imported: {out.stdout}"


# The words of one modality. The contract keeps them out of the core: "contact,
# pressure, indentation, receptor" appear only in the tactile pack, its renderer and
# adapter.
TACTILE_WORDS = re.compile(
    r"(tactile|contact|pressure|indentation|indenter|receptor|touch|skin|afferent)",
    re.IGNORECASE,
)


#: The one core module that names a modality: it registers the built-in metric packs.
NAMES_PACKS = "registry.py"


def _core_sources() -> List[Path]:
    files = sorted((EVAL / "audit").rglob("*.py"))
    files += sorted(p for p in EVAL.glob("*.py") if p.name != NAMES_PACKS)
    return files


def test_the_core_audit_uses_no_tactile_word():
    files = _core_sources()
    assert any(p.name == "reader.py" for p in files)
    assert not any(p.name == NAMES_PACKS for p in files)  # the one exemption
    hits = [
        f"{p.relative_to(REPO)}:{n}: {m.group(0)}"
        for p in files
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        for m in [TACTILE_WORDS.search(line)]
        if m
    ]
    assert not hits, "tactile words in the core audit:\n" + "\n".join(hits)


@pytest.mark.parametrize(
    "line", ["# a contact interval", "pressure_peak = 1", "Receptor grid", "touch_ms"]
)
def test_the_lint_fires_on_tactile_words(line):
    assert TACTILE_WORDS.search(line)


def test_the_tactile_metric_pack_is_inside_the_wall():
    """The pack may use tactile words but never reaches the world package."""
    roots = _wall_roots()
    assert EVAL / "tactile" / "metrics.py" in roots
    seen = reachable(roots, REPO, PACKAGE, "sensoryforge.evaluate")
    assert "sensoryforge.evaluate.tactile.metrics" in seen
    assert not _breaches(seen, WALLED)
    # the package itself imports neither the pack nor the (future) declared adapter
    tree = ast.parse((EVAL / "tactile" / "__init__.py").read_text())
    assert not list(_import_nodes(tree, import_time_only=False))
