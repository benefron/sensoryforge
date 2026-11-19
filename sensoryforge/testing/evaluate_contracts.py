"""Contract checks for the evaluation seams (C2d, report contract section 10).

:func:`sensoryforge.testing.contracts.check_component` checks the sensing pipeline's
components. The evaluation stages have their own plug-in points, a metric pack (what
the audit measures) and a declared adapter (what a data set's declaration says it
should show). They are checked here, by registered name, after the plugin has
registered them::

    from sensoryforge.testing.evaluate_contracts import check_evaluation_component

    check_evaluation_component("metric_pack", "peaks", example="runs/demo")
    check_evaluation_component("declared_adapter", "my_modality", example="runs/demo")

Without an ``example`` only what needs no data is checked. With ``example``, the path of
a data-set folder, the check runs the stage on it twice and holds the result to the
rules below. Nothing here reads the contract's JSON schemas (a plain install has none),
so the rules are the ones the code can state.

Checks, by kind:

* both kinds: the registered class subclasses the base class, its ``name`` is the
  registered name and its ``version`` a non-empty string; ``get_param_spec()`` is a list
  of ``ParamSpec``; ``from_config(to_dict())`` round-trips;
* ``metric_pack``: its module does not import ``sensoryforge.world`` (a pack sits inside
  the audit's blind wall; only that module's own imports are read, the import-wall test
  covers the rest); every definition from ``metrics()`` is complete, its id is
  ``<name>.<quantity>.<metric>``, ids are unique; with an example, the audit runs with
  the pack without a measure failure, each value names a defined id, values come in
  catalogue order, and two runs give the same bytes;
* ``declared_adapter``: with an example, the adapter accepts the data set's
  ``dataset.json``, the stage runs, the catalogue is complete with unique ids, every
  value and every omitted metric names a catalogue id, every omission has a reason,
  there is one row per manifest entry, and two runs give the same bytes.

Topic: testing, evaluate
Tests: tests/unit/test_evaluate_contracts.py
Entry points: check_evaluation_component, available_evaluation_kinds
Pinned by: docs/developer_guide/audit_metrics.md,
    docs/developer_guide/declared_adapter.md, docs/reference/report_contract.md,
    sensoryforge/agents/skills/sensoryforge-extend-evaluation
Do not change without: a built-in pack or adapter that cannot pass is recorded as an
    open ledger item, never by weakening a check here; the frame renderer is not
    checked because no registry for it exists (see the ledger).
"""

from __future__ import annotations

import ast
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from sensoryforge.evaluate.audit.units import QUANTITIES
from sensoryforge.stimuli.base import ParamSpec

_DEFINITION_KEYS = (
    "id",
    "pack",
    "level",
    "value_type",
    "unit",
    "resolution",
    "scope",
    "summarised",
)
_SCOPES = ("entry", "channel", "population")
_VALUE_TYPES = ("number", "integer", "string")
_LEVELS = ("entry", "dataset")
_PINNED = "2000-01-01T00:00:00Z"


def available_evaluation_kinds() -> List[str]:
    """The kinds :func:`check_evaluation_component` supports."""
    return sorted(_CHECKS)


def _fail(label: str, why: str) -> None:
    raise AssertionError(f"{label}: {why}")


def _class_of(registry: Any, name: str, label: str) -> type:
    try:
        return registry.get_class(name)
    except KeyError:
        raise ValueError(
            f"{label} {name!r} is not registered "
            f"(registered: {registry.list_registered()})"
        )


def _check_common(cls: type, base: type, name: str, label: str) -> None:
    """Rules both kinds share: base class, name, version, ParamSpec, round trip."""
    if not (isinstance(cls, type) and issubclass(cls, base)):
        _fail(label, f"{cls!r} is not a subclass of {base.__name__}")
    if str(getattr(cls, "name", "")).lower() != name.lower() or not cls.name:
        _fail(label, f"its name attribute is {cls.name!r}, not the registered {name!r}")
    if not isinstance(cls.version, str) or not cls.version:
        _fail(label, f"version must be a non-empty string, got {cls.version!r}")
    specs = cls.get_param_spec()
    if not isinstance(specs, list) or not all(isinstance(s, ParamSpec) for s in specs):
        _fail(label, "get_param_spec() must return a list of ParamSpec")
    first = cls.from_config({})
    if not isinstance(first, cls):
        _fail(label, "from_config({}) did not return an instance of the class")
    params = first.to_dict()
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _fail(label, f"to_dict() must be JSON-plain: {exc}")
    again = cls.from_config(params).to_dict()
    if again != params:
        _fail(
            label, f"from_config(to_dict()) changed the parameters: {params} -> {again}"
        )


def _world_imports(cls: type) -> List[str]:
    """Imports of ``sensoryforge.world`` in the module that defines ``cls``."""
    try:
        path = inspect.getsourcefile(cls)
    except (TypeError, OSError):  # defined in a console or a notebook: no file to read
        return []
    if path is None:
        return []
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                a.name for a in node.names if a.name.startswith("sensoryforge.world")
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("sensoryforge.world"):
                found.append(node.module)
            elif node.module == "sensoryforge" and any(
                a.name == "world" for a in node.names
            ):
                found.append("sensoryforge.world")
    return found


def _synthetic_layout() -> Any:
    from sensoryforge.evaluate.audit.layout import Layout

    xs = np.linspace(0.0, 1.0, 16)
    coords = np.stack([xs, np.zeros_like(xs)], axis=1)
    return Layout(coords=coords, channel_names=None, n_channels=1, regular_grid=None)


def _folder(example: Any, label: str) -> Path:
    if example is None:
        raise ValueError(f"{label}: pass example= the path of a data-set folder")
    path = Path(example)
    if not (path / "manifest.jsonl").is_file():
        raise ValueError(f"{label}: {path} has no manifest.jsonl (a data-set folder)")
    return path


def _check_metric_pack(name: str, example: Any) -> None:
    from sensoryforge.evaluate.audit.run import run_audit
    from sensoryforge.evaluate.pack import MetricPack
    from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY

    label = f"metric pack {name!r}"
    cls = _class_of(METRIC_PACK_REGISTRY, name, "metric pack")
    _check_common(cls, MetricPack, name, label)
    walls = _world_imports(cls)
    if walls:
        _fail(
            label,
            f"its module imports {walls}: a pack is inside the audit's blind wall and "
            "may not read the world",
        )
    pack = cls.from_config({})
    units = {q: None for q in QUANTITIES}
    definitions = pack.metrics(_synthetic_layout(), units, 1.0)
    ids = [d.get("id") for d in definitions]
    if len(set(ids)) != len(ids):
        _fail(label, f"metric ids repeat: {ids}")
    for d in definitions:
        missing = [k for k in _DEFINITION_KEYS if k not in d]
        if missing:
            _fail(
                label,
                f"definition {d.get('id')!r} lacks {missing}; use metric_definition()",
            )
        parts = str(d["id"]).split(".")
        if len(parts) < 3 or parts[0].lower() != name.lower() or d["pack"] != parts[0]:
            _fail(
                label,
                f"id {d['id']!r} must be <pack>.<quantity>.<metric>, pack {name!r}",
            )
        if d["scope"] not in _SCOPES:
            _fail(label, f"{d['id']}: scope {d['scope']!r} is not one of {_SCOPES}")
        if d["value_type"] not in _VALUE_TYPES:
            _fail(
                label,
                f"{d['id']}: value_type {d['value_type']!r} is not one of "
                f"{_VALUE_TYPES}",
            )
        if d["level"] not in _LEVELS:
            _fail(label, f"{d['id']}: level {d['level']!r} is not one of {_LEVELS}")
    if example is None:
        return

    folder = _folder(example, label)
    runs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            audit = run_audit(folder, Path(tmp), packs=[name], created_at=_PINNED)
            rows = (Path(tmp) / "audit_entries.jsonl").read_bytes()
            runs.append((json.dumps(audit, sort_keys=True), rows))
    audit = json.loads(runs[0][0])
    bad = [f for f in audit["failures"] if f.get("stage") == "measure"]
    if bad:
        _fail(label, f"measure() failed on {len(bad)} entries, first: {bad[0]}")
    defined = [m["id"] for m in audit["metrics"]]
    order = {i: k for k, i in enumerate(defined)}
    for line in runs[0][1].decode("utf-8").splitlines():
        row = json.loads(line)
        seen = [v["id"] for v in row["values"]]
        unknown = sorted(set(seen) - set(order))
        if unknown:
            _fail(label, f"entry {row['entry']} has values for undefined ids {unknown}")
        if [order[i] for i in seen] != sorted(order[i] for i in seen):
            _fail(label, f"entry {row['entry']}: values are not in catalogue order")
    if runs[0] != runs[1]:
        _fail(
            label,
            "two audits of the same data gave different results (not deterministic)",
        )


def _check_declared_adapter(name: str, example: Any) -> None:
    # Importing the package registers the built-in adapter.
    import sensoryforge.evaluate.declared  # noqa: F401
    from sensoryforge.evaluate.declared import DeclaredAdapter, run_declared
    from sensoryforge.evaluate.registry import DECLARED_ADAPTER_REGISTRY

    label = f"declared adapter {name!r}"
    cls = _class_of(DECLARED_ADAPTER_REGISTRY, name, "declared adapter")
    _check_common(cls, DeclaredAdapter, name, label)
    if example is None:
        return

    folder = _folder(example, label)
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if not cls.accepts(meta):
        _fail(label, "accepts(dataset_meta) is False for this data set's dataset.json")
    runs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            declared = run_declared(folder, Path(tmp), adapter=name, created_at=_PINNED)
            rows = (Path(tmp) / "declared_entries.jsonl").read_bytes()
            runs.append((json.dumps(declared, sort_keys=True), rows))
    declared = json.loads(runs[0][0])
    catalogue = declared["metrics"]
    ids = [m.get("id") for m in catalogue]
    if len(set(ids)) != len(ids):
        _fail(label, f"catalogue ids repeat: {ids}")
    for m in catalogue:
        missing = [k for k in ("id", "unit", "basis", "source") if k not in m]
        if missing:
            _fail(label, f"catalogue entry {m.get('id')!r} lacks {missing}")
    order = {i: k for k, i in enumerate(ids)}
    stray = sorted(set(declared["omitted"]) - set(order))
    if stray:
        _fail(label, f"omitted names ids that are not in the catalogue: {stray}")
    for metric, reasons in declared["omitted"].items():
        if any(not r for r in reasons):
            _fail(label, f"{metric}: an omission has no reason")
    manifest = [
        json.loads(line)["entry"]
        for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [json.loads(x) for x in runs[0][1].decode("utf-8").splitlines()]
    if sorted(manifest) != [r["entry"] for r in lines]:
        _fail(label, "it must write one row per manifest entry, sorted by entry id")
    for row in lines:
        seen = [v["id"] for v in row["values"]]
        unknown = sorted(set(seen) - set(order))
        if unknown:
            _fail(
                label,
                f"entry {row['entry']} declares ids not in the catalogue: {unknown}",
            )
        if [order[i] for i in seen] != sorted(order[i] for i in seen):
            _fail(label, f"entry {row['entry']}: values are not in catalogue order")
    if runs[0] != runs[1]:
        _fail(label, "two runs over the same data set gave different results")


_CHECKS: Dict[str, Callable[[str, Any], None]] = {
    "metric_pack": _check_metric_pack,
    "declared_adapter": _check_declared_adapter,
}


def check_evaluation_component(kind: str, name: str, example: Any = None) -> None:
    """Run the contract checks on one registered evaluation component.

    Register the component first (its plugin's ``register`` function), then check it by
    name. See the module docstring for the checks of each kind.

    Args:
        kind: One of :func:`available_evaluation_kinds`: ``"metric_pack"`` or
            ``"declared_adapter"``.
        name: The registered name.
        example: The path of a data-set folder (the batch output: ``manifest.jsonl``,
            ``dataset.json``, the bundles). Without it only the checks that need no
            data run. A declared adapter needs only ``dataset.json`` and
            ``manifest.jsonl``; a metric pack needs the bundles too.

    Raises:
        ValueError: If ``kind`` is unknown, ``name`` is not registered, or ``example``
            is not a data-set folder.
        AssertionError: If a check fails, naming the component and the rule.

    Example:
        >>> from sensoryforge.testing import evaluate_contracts as ec
        >>> ec.check_evaluation_component("metric_pack", "core")
    """
    if kind not in _CHECKS:
        raise ValueError(
            f"Unknown evaluation kind {kind!r}; "
            f"choose one of {available_evaluation_kinds()}"
        )
    _CHECKS[kind](name, example)
