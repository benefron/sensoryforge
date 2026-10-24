"""Machine-readable description of SensoryForge: what exists, and what a file may say.

Three things, all generated from the live registries and loaders so they cannot drift
(B3, ``docs/agents/``):

- :func:`describe` -- every registered component with its ``ParamSpec`` list, the world
  language (shapes, patterns, modulations, distributions, class kinds), the evaluation
  seams and the shipped recipes, as one JSON-plain dict
  (``sensoryforge describe --json``).
- :func:`config_schema`, :func:`world_schema`, :func:`dataset_schema` -- JSON Schemas
  for the three YAML formats (``sensoryforge describe --schema``; committed copies in
  ``schemas/``, rebuilt by ``scripts/build_schemas.py``).
- :func:`validate_file` -- a file's problems as ``{path, message, fix}`` records
  (``sensoryforge validate --json``).

Units and shapes are never copied here: a ``ParamSpec`` carries its own unit, and tensor
shapes live in the component docstrings. Importing this module registers every built-in
component and runs plugin discovery, so a plugin's components appear too.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
import re
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

FORMAT = "sensoryforge-describe/1"
REPORT_FORMAT = "sensoryforge-validate/1"

#: The JSON Schema draft the generated schemas declare.
SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _plain(value: Any) -> Any:
    """``value`` as JSON-plain data (tuples become lists, unknown objects text)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _plain(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _summary(obj: Any) -> str:
    """The first paragraph of ``obj``'s docstring on one line ('' if it has none)."""
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n")[0].split())


def spec_to_dict(spec: Any) -> Dict[str, Any]:
    """A ``ParamSpec`` as a JSON-plain dict of every field it carries.

    Args:
        spec: A :class:`~sensoryforge.stimuli.base.ParamSpec`.

    Returns:
        ``{name, label, dtype, default, min, max, step, unit, choices, tooltip,
        help, group, advanced}``; ``unit`` is ``""`` for a dimensionless parameter.
    """
    return {
        "name": spec.name,
        "label": spec.label,
        "dtype": spec.dtype,
        "default": _plain(spec.default),
        "min": _plain(spec.min_val),
        "max": _plain(spec.max_val),
        "step": _plain(spec.step),
        "unit": spec.unit,
        "choices": _plain(spec.choices),
        "tooltip": spec.tooltip,
        "help": spec.help,
        "group": spec.group,
        "advanced": bool(spec.advanced),
    }


def _registries() -> Dict[str, Any]:
    """``{kind: registry}`` for the pipeline component kinds, built-ins registered."""
    from sensoryforge.register_components import register_all
    from sensoryforge.registry import (
        FILTER_REGISTRY,
        GRID_REGISTRY,
        INNERVATION_REGISTRY,
        NEURON_REGISTRY,
        PROCESSING_REGISTRY,
        SOLVER_REGISTRY,
        STIMULUS_REGISTRY,
    )

    register_all()
    return {
        "neuron": NEURON_REGISTRY,
        "filter": FILTER_REGISTRY,
        "stimulus": STIMULUS_REGISTRY,
        "solver": SOLVER_REGISTRY,
        "grid": GRID_REGISTRY,
        "innervation": INNERVATION_REGISTRY,
        "processing": PROCESSING_REGISTRY,
    }


def _init_params(cls: type) -> List[Dict[str, Any]]:
    """The constructor's parameters: ``[{name, required, default}]``.

    The fallback for a component whose ``params`` list is empty (it declares no
    ``ParamSpec``): the real argument names, with defaults that are JSON-plain
    (``default`` is omitted when the constructor's default is not).
    """
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    out: List[Dict[str, Any]] = []
    for p in sig.parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        item: Dict[str, Any] = {"name": p.name, "required": p.default is p.empty}
        if p.default is not p.empty and (
            p.default is None or isinstance(p.default, (str, bool, int, float))
        ):
            item["default"] = p.default
        out.append(item)
    return out


def _component(registry: Any, name: str) -> Dict[str, Any]:
    """One registered component: class, summary, ParamSpec list and constructor."""
    cls = registry.get_class(name)
    entry: Dict[str, Any] = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "summary": _summary(cls),
        "init_params": _init_params(cls),
    }
    try:
        entry["params"] = [spec_to_dict(s) for s in registry.get_param_spec(name)]
    except (TypeError, ValueError, NotImplementedError) as exc:
        # A class whose get_param_spec needs arguments: say so, do not hide it.
        entry["params"] = []
        entry["params_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _components() -> Dict[str, Dict[str, Any]]:
    return {
        kind: {name: _component(reg, name) for name in reg.list_registered()}
        for kind, reg in _registries().items()
    }


def _world() -> Dict[str, Any]:
    """The world language: what a ``world:`` file may name, from its registries."""
    from sensoryforge.world import dataset, distributions, kernel, kinds, schema

    shapes = {
        n: {
            "params": [spec_to_dict(s) for s in k.specs],
            "unbounded": k.unbounded,
            "indenter": k.indenter,
        }
        for n, k in sorted(kernel.SHAPE_KINDS.items())
    }
    patterns = {
        n: {"params": [spec_to_dict(s) for s in k.specs], "placement": k.placement}
        for n, k in sorted(kernel.PATTERN_KINDS.items())
    }
    modulations = {
        n: {"params": [spec_to_dict(s) for s in k.specs]}
        for n, k in sorted(kernel.MODULATION_KINDS.items())
    }
    dists = {
        n: {
            "summary": _summary(d.sample),
            "quantile": d.quantile,
            "finite_support": d.support is not None,
            "declares_bounds": d.bounds is not None,
        }
        for n, d in sorted(distributions.DISTRIBUTIONS.items())
    }
    class_kinds = {
        n: {"class": f"{k.__class__.__module__}.{k.__class__.__qualname__}"}
        for n, k in sorted(kinds.CLASS_KINDS.items())
    }
    return {
        "format": schema.FORMAT,
        "dataset_format": dataset.FORMAT,
        "top_keys": sorted(schema._TOP_KEYS),
        "class_keys": sorted(schema._CLASS_KEYS),
        "session_keys": sorted(schema._SESSION_KEYS),
        "layer_keys": ["shape", "pattern", "motion", "modulation", *kinds._LAYER_KEYS],
        "axis_keys": sorted(distributions._AXIS_KEYS | {"same_as"}),
        "numeric_distributions": list(distributions.NUMERIC_DISTS),
        "motion_kinds": list(kinds._MOTION_KINDS),
        "shapes": shapes,
        "patterns": patterns,
        "modulations": modulations,
        "distributions": dists,
        "class_kinds": class_kinds,
        "split_kinds": dict(dataset.SPLIT_KINDS),
        "split_keys": {k: sorted(v) for k, v in sorted(dataset._KIND_KEYS.items())},
    }


def _evaluation() -> Dict[str, Any]:
    """The evaluation seams: metric packs, check packs, declared adapters."""
    from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
    from sensoryforge.evaluate.registry import (
        DECLARED_ADAPTER_REGISTRY,
        METRIC_PACK_REGISTRY,
    )

    # The adapters register when their module loads (the audit's import wall keeps
    # that module out of evaluate.registry).
    import sensoryforge.evaluate.declared  # noqa: F401

    def entries(registry: Any) -> Dict[str, Any]:
        return {
            n: {
                "class": f"{registry.get_class(n).__module__}."
                f"{registry.get_class(n).__qualname__}",
                "summary": _summary(registry.get_class(n)),
            }
            for n in registry.list_registered()
        }

    return {
        "metric_packs": entries(METRIC_PACK_REGISTRY),
        "check_packs": entries(CHECK_PACK_REGISTRY),
        "declared_adapters": entries(DECLARED_ADAPTER_REGISTRY),
    }


def _recipes() -> Dict[str, Any]:
    """The shipped canonical-config presets, each with a short digest of its config."""
    from sensoryforge.presets import list_presets, load_preset, preset_description

    out: Dict[str, Any] = {}
    for name in list_presets():
        cfg = load_preset(name)
        out[name] = {
            "description": preset_description(name),
            "grids": [
                {
                    k: g.get(k)
                    for k in (
                        "name",
                        "arrangement",
                        "rows",
                        "cols",
                        "spacing",
                        "channels",
                    )
                    if k in g
                }
                for g in cfg.get("grids", [])
            ],
            "populations": [
                {
                    k: p.get(k)
                    for k in ("name", "neuron_model", "neuron_type", "filter_method")
                    if k in p
                }
                for p in cfg.get("populations", [])
            ],
        }
    return out


def _formats() -> Dict[str, Any]:
    """The version tags of every file format SensoryForge reads or writes."""
    import sensoryforge
    from sensoryforge.io import bundle
    from sensoryforge.world import dataset, schema

    return {
        "sensoryforge": sensoryforge.__version__,
        "world": schema.FORMAT,
        "dataset": dataset.FORMAT,
        "bundle": bundle.SCHEMA_VERSION,
        "bundle_with_voltage_probes": bundle.PROBE_SCHEMA_VERSION,
        "bundle_with_sparse_weights": bundle.SPARSE_SCHEMA_VERSION,
        "describe": FORMAT,
    }


def describe() -> Dict[str, Any]:
    """Everything SensoryForge can name, as one JSON-plain dict.

    Built from the live registries, so a plugin's components appear and a removed
    one disappears. Nothing here is copied from prose.

    Returns:
        ``{format, formats, components, world, evaluation, recipes}``:
        ``components[kind][name]`` is ``{class, summary, init_params, params}`` with
        ``params`` the component's :func:`spec_to_dict` list (each carries its own
        unit and default); an empty ``params`` means the component declares no
        ``ParamSpec``, and ``init_params`` (its constructor) is then the parameter list.
    """
    return {
        "format": FORMAT,
        "formats": _formats(),
        "components": _components(),
        "world": _world(),
        "evaluation": _evaluation(),
        "recipes": _recipes(),
    }


# ----------------------------------------------------------------- JSON Schemas

_SCALAR = {"type": ["string", "number", "boolean", "null"]}

_PY_TO_JSON = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _type_schema(tp: Any) -> Dict[str, Any]:
    """A JSON Schema for a dataclass field's annotation."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return {}
    if origin is typing.Union:
        parts = [a for a in args if a is not type(None)]
        inner = _type_schema(parts[0]) if len(parts) == 1 else {}
        if len(parts) == 1 and "type" in inner and isinstance(inner["type"], str):
            return {**inner, "type": [inner["type"], "null"]}
        return {}
    if origin in (list, typing.List):
        return {"type": "array", "items": _type_schema(args[0]) if args else {}}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if dataclasses.is_dataclass(tp):
        return _dataclass_schema(tp)
    if tp is float:
        # YAML 1.1 reads 3e-1 as text; the loader rejects it later with a hint.
        return {"type": "number"}
    return dict(_PY_TO_JSON.get(tp, {}))


def _dataclass_schema(cls: Any) -> Dict[str, Any]:
    """An object schema for a config dataclass; unknown keys are allowed.

    The loaders ignore keys they do not know, so the schema does not forbid them either;
    ``validate --json`` reports unknown keys as warnings instead.
    """
    hints = typing.get_type_hints(cls)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for f in dataclasses.fields(cls):
        schema = _type_schema(hints[f.name])
        if f.default is not dataclasses.MISSING and not isinstance(
            f.default, (list, dict)
        ):
            schema = {**schema, "default": _plain(f.default)}
        properties[f.name] = schema
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            required.append(f.name)
    out: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    doc = _summary(cls)
    if doc:
        out["description"] = doc
    return out


def config_schema() -> Dict[str, Any]:
    """JSON Schema of a canonical config (``SensoryForgeConfig``), from its dataclasses.

    Registered component names become ``examples`` (not ``enum``): a plugin may add
    names, and the loader matches names case-insensitively.
    """
    from sensoryforge.config.schema import SensoryForgeConfig

    regs = _registries()
    schema = _dataclass_schema(SensoryForgeConfig)
    props = schema["properties"]
    pop = props["populations"]["items"]["properties"]
    pop["neuron_model"]["examples"] = regs["neuron"].list_registered()
    pop["filter_method"]["examples"] = regs["filter"].list_registered()
    pop["innervation_method"]["examples"] = regs["innervation"].list_registered()
    props["stimulus"]["properties"]["type"]["examples"] = regs[
        "stimulus"
    ].list_registered()
    props["grids"]["items"]["properties"]["arrangement"]["examples"] = regs[
        "grid"
    ].list_registered()
    schema.update(
        {
            "$schema": SCHEMA_DRAFT,
            "$id": "https://benefron.github.io/sensoryforge/schemas/config.schema.json",
            "title": "SensoryForge canonical config",
        }
    )
    schema["properties"]["plugins"] = {
        "type": "array",
        "description": "Plugin packages or paths imported before the config is built.",
    }
    return schema


def _axis_schema() -> Dict[str, Any]:
    return {
        "description": "One axis: a constant, or a mapping naming a range, a list "
        "of values or a distribution.",
        "oneOf": [
            _SCALAR,
            {
                "type": "object",
                "properties": {
                    "value": _SCALAR,
                    "range": {"type": "array", "items": {"type": "number"}},
                    "dist": {"type": "string"},
                    "int": {"type": "boolean"},
                    "values": {"type": "array"},
                    "weights": {"type": "array", "items": {"type": "number"}},
                    "circular": {"type": "boolean"},
                    "probes": {"type": "boolean"},
                    "stratify": {"type": "boolean"},
                    "same_as": {"type": "string"},
                },
            },
        ],
    }


def world_schema() -> Dict[str, Any]:
    """JSON Schema of a ``world:`` file (top-level, class and session keys strict)."""
    from sensoryforge.world import kernel, kinds, schema as world

    axes = {"type": "object", "additionalProperties": _axis_schema()}

    def part(table: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"kind": {"type": "string", "examples": sorted(table)}},
        }

    layer = {
        "type": "object",
        "properties": {
            "shape": part(kernel.SHAPE_KINDS),
            "pattern": part(kernel.PATTERN_KINDS),
            "modulation": part(kernel.MODULATION_KINDS),
            "motion": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(kinds._MOTION_KINDS)}
                },
            },
            "background": {"type": "number", "minimum": 0},
            "clamp_min": {"type": "number", "maximum": 0},
        },
        "additionalProperties": False,
    }
    cls = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "examples": sorted(kinds.CLASS_KINDS)},
            "weight": {"type": "number", "exclusiveMinimum": 0},
            "layer": layer,
            "axes": axes,
            "channel": {"type": "string"},
            "use": {"oneOf": [{"type": "string"}, {"type": "array"}]},
        },
        "additionalProperties": False,
    }
    classes = {"type": "object", "minProperties": 1, "additionalProperties": cls}
    body = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "modality": {"type": "string"},
            "units": {"type": "object", "additionalProperties": {"type": "string"}},
            "channels": {"type": "array", "items": {"type": "string"}},
            "defaults": axes,
            "groups": {"type": "object", "additionalProperties": axes},
            "classes": classes,
            "held_out": {"type": "object", "additionalProperties": cls},
            "fixed_draws": {"type": "object"},
            "sessions": {
                "type": "object",
                "properties": {k: {} for k in sorted(world._SESSION_KEYS)},
                "additionalProperties": False,
            },
        },
        "required": ["classes"],
        "additionalProperties": False,
    }
    return {
        "$schema": SCHEMA_DRAFT,
        "$id": "https://benefron.github.io/sensoryforge/schemas/world.schema.json",
        "title": f"SensoryForge world ({world.FORMAT})",
        "description": "A world, optionally under a top-level 'world:' key.",
        "oneOf": [
            body,
            {
                "type": "object",
                "properties": {"world": body},
                "required": ["world"],
                "additionalProperties": False,
            },
        ],
    }


def dataset_schema() -> Dict[str, Any]:
    """JSON Schema of a ``dataset:`` file. Dataset and split keys are strict."""
    from sensoryforge.world import dataset

    split = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(dataset._KINDS)},
            "n": {"type": "integer", "minimum": 1},
            "repeats": {"type": "integer", "minimum": 1},
            "noise_repeats": {"type": "integer", "minimum": 1},
            "stratified": {},
            "per_bin": {"type": "integer", "minimum": 1},
            "bins": {"type": "integer", "minimum": 1},
            "duration_ms": {"type": "number", "exclusiveMinimum": 0},
            "draws": {},
        },
        "additionalProperties": False,
    }
    body = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "world": {"type": "string", "description": "Path of the world file."},
            "world_id": {"type": "string"},
            "seed": {"type": "integer"},
            "duration_ms": {"type": "number", "exclusiveMinimum": 0},
            "splits": {"type": "object", "additionalProperties": split},
        },
        "required": ["splits"],
        "additionalProperties": False,
    }
    return {
        "$schema": SCHEMA_DRAFT,
        "$id": "https://benefron.github.io/sensoryforge/schemas/dataset.schema.json",
        "title": f"SensoryForge data set ({dataset.FORMAT})",
        "description": "A data set, optionally under a top-level 'dataset:' key.",
        "oneOf": [
            body,
            {
                "type": "object",
                "properties": {"dataset": body},
                "required": ["dataset"],
                "additionalProperties": False,
            },
        ],
    }


SCHEMAS = {
    "config": config_schema,
    "world": world_schema,
    "dataset": dataset_schema,
}


# -------------------------------------------------------------------- validate

#: A leading dotted path in an error message: ``world.classes.a.axes.b: ...``.
_PATH_RE = re.compile(
    r"^([A-Za-z_][\w\-]*(?:(?:\.|\[)[\w\-\.\[\]]*)?)\s*:\s+(.*)$", re.S
)
#: A list a message offers as the valid choices (``allowed: [...]``, ``Available: a``).
_CHOICES_RE = re.compile(
    r"(?:allowed|known|available(?: presets)?|valid)\s*:\s*(\[[^\]]*\]|[^.\n]+)", re.I
)
_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _choices_in(message: str) -> List[str]:
    """The names a message lists as valid, if it lists any."""
    found = _CHOICES_RE.search(message)
    if not found:
        return []
    return [t for t in re.findall(r"[A-Za-z_][\w\-\.]*", found.group(1)) if t]


#: Words of an "Unknown <thing>: <name>" message -> the registry kind that holds <name>.
_UNKNOWN_RE = re.compile(r"[Uu]nknown ([a-z ]+?)\s*[:'\"]+\s*['\"]?([\w\-\.]+)")
_KIND_WORDS = (
    ("neuron", "neuron"),
    ("filter", "filter"),
    ("stimulus", "stimulus"),
    ("solver", "solver"),
    ("grid", "grid"),
    ("arrangement", "grid"),
    ("innervation", "innervation"),
    ("builder", "innervation"),
    ("processing", "processing"),
)


def _registry_fix(message: str) -> Optional[str]:
    """Close registered names for an ``Unknown <kind>: <name>`` message, or ``None``."""
    found = _UNKNOWN_RE.search(message)
    if not found:
        return None
    phrase, name = found.group(1).lower(), found.group(2)
    kind = next((k for word, k in _KIND_WORDS if word in phrase), None)
    if kind is None:
        return None
    names = _registries()[kind].list_registered()
    folded = {n.casefold(): n for n in names}
    close = difflib.get_close_matches(name.casefold(), list(folded), n=1, cutoff=0.6)
    if close:
        return (
            f"did you mean {folded[close[0]]!r}? Registered {kind}s: {', '.join(names)}"
        )
    return f"registered {kind}s: {', '.join(names)}"


def suggest_fix(message: str) -> Optional[str]:
    """A suggested fix drawn from the message itself, or ``None``.

    Only derivations from facts the message or the registries state: a close match for
    a misspelled name among the valid names the message lists, a close match among the
    registered components for an ``Unknown <kind>: <name>`` message, and the hint text a
    loader already wrote (``write 3.0e-1, not 3e-1``). Nothing is invented.
    """
    choices = _choices_in(message)
    if choices:
        for wrong in _QUOTED_RE.findall(message) + re.findall(
            r"\[([^\]]+)\]", message.split(";")[0]
        ):
            for token in re.findall(r"[A-Za-z_][\w\-\.]*", wrong):
                if token in choices:
                    continue
                close = difflib.get_close_matches(token, choices, n=1, cutoff=0.6)
                if close:
                    return f"did you mean {close[0]!r}? Valid: {', '.join(choices)}"
        return f"valid: {', '.join(choices)}"
    hint = re.search(r"\((YAML reads[^)]*)\)", message)
    if hint:
        return hint.group(1)
    return _registry_fix(message)


def _problem(path: str, message: str, source: str, fix: Optional[str] = None) -> Dict:
    return {
        "path": path,
        "message": message,
        "fix": fix if fix is not None else suggest_fix(message),
        "source": source,
    }


def _from_exception(exc: BaseException, source: str, root: str = "") -> Dict[str, Any]:
    text = str(exc).strip("'\"") if isinstance(exc, KeyError) else str(exc)
    found = _PATH_RE.match(text)
    if found:
        return _problem(found.group(1), found.group(2), source)
    return _problem(root, text, source)


def _kind_of(data: Dict[str, Any]) -> str:
    """The format of a parsed file: config, legacy-config, world, dataset, unknown."""
    if "pipeline" in data:
        return "legacy-config"
    if isinstance(data.get("dataset"), dict) or (
        "splits" in data and not isinstance(data.get("grids"), list)
    ):
        return "dataset"
    if isinstance(data.get("world"), dict) or "classes" in data:
        return "world"
    if isinstance(data.get("grids"), list) or isinstance(data.get("populations"), list):
        return "config"
    return "unknown"


def _unknown_keys(
    data: Dict[str, Any], cls: Any, path: str, out: List[Dict[str, Any]]
) -> None:
    """Warn about keys a config dataclass does not have, recursing into its children."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    for key, value in data.items():
        where = f"{path}.{key}" if path else str(key)
        if key not in names:
            close = difflib.get_close_matches(str(key), names, n=1, cutoff=0.6)
            msg = f"unknown key {key!r}; it is ignored"
            fix = f"did you mean {close[0]!r}?" if close else None
            out.append({**_problem(where, msg, "schema", fix), "severity": "warning"})
            continue
        tp = hints[key]
        args = typing.get_args(tp)
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            _unknown_keys(value, tp, where, out)
        elif (
            typing.get_origin(tp) in (list, typing.List)
            and args
            and dataclasses.is_dataclass(args[0])
            and isinstance(value, list)
        ):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _unknown_keys(item, args[0], f"{where}.{i}", out)


def _validate_config(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    from sensoryforge.config.schema import SensoryForgeConfig

    out: List[Dict[str, Any]] = []
    _registries()
    _unknown_keys(
        {k: v for k, v in data.items() if k not in ("plugins",)},
        SensoryForgeConfig,
        "",
        out,
    )
    try:
        config = SensoryForgeConfig.from_dict(data)
    except (ValueError, KeyError, TypeError) as exc:
        out.append(_from_exception(exc, "schema"))
        return out
    # The engine's own build checks, keyed by pipeline stage (no Qt, no I/O).
    from sensoryforge.gui.validation import validate

    for key, message in sorted(validate(config).items()):
        out.append(_problem(key, message, "build"))
    return out


def _validate_legacy(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    from sensoryforge.core.generalized_pipeline import (
        GeneralizedTactileEncodingPipeline,
    )

    try:
        GeneralizedTactileEncodingPipeline.from_config(data)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        return [_from_exception(exc, "build")]
    return []


def _validate_world(path: Path) -> List[Dict[str, Any]]:
    from sensoryforge.world import load_world

    try:
        load_world(path)
    except (ValueError, KeyError, TypeError) as exc:
        return [_from_exception(exc, "world")]
    return []


def _validate_dataset(path: Path) -> List[Dict[str, Any]]:
    from sensoryforge.world.dataset import load_dataset

    try:
        load_dataset(path)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        return [_from_exception(exc, "dataset")]
    return []


def validate_file(path: str, kind: str = "auto") -> Dict[str, Any]:
    """Check a config, world or data set file and report its problems as data.

    Args:
        path: The YAML file.
        kind: ``"auto"`` (decide from the keys), ``"config"``, ``"world"`` or
            ``"dataset"``.

    Returns:
        ``{format, file, kind, valid, errors, warnings}``. Each problem is
        ``{path, message, fix, source}``: ``path`` is the dotted location in the file
        (``populations.0.neuron_model``; ``""`` for the file as a whole), ``fix`` a
        suggestion taken from the message (``None`` when there is none) and ``source``
        the stage that found it (``yaml``, ``schema``, ``build``, ``world``,
        ``dataset``). ``valid`` is true when ``errors`` is empty; unknown keys are
        warnings.
    """
    report: Dict[str, Any] = {
        "format": REPORT_FORMAT,
        "file": str(path),
        "kind": kind,
        "valid": False,
        "errors": [],
        "warnings": [],
    }
    errors: List[Dict[str, Any]] = report["errors"]
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(_problem("", f"cannot read the file: {exc}", "yaml"))
        return report
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        errors.append(
            {
                **_problem("", f"{problem}{where}", "yaml"),
                "line": mark.line + 1 if mark else None,
            }
        )
        return report
    if not isinstance(data, dict):
        errors.append(_problem("", "the file must be a YAML mapping", "yaml"))
        return report
    if kind == "auto":
        kind = _kind_of(data)
    report["kind"] = kind
    if kind == "config":
        problems = _validate_config(data)
    elif kind == "world":
        problems = _validate_world(Path(path))
    elif kind == "dataset":
        problems = _validate_dataset(Path(path))
    elif kind == "legacy-config":
        problems = _validate_legacy(data)
    else:
        problems = [
            _problem(
                "",
                "cannot tell what this file is: a config has grids and populations, "
                "a world has classes, a data set has splits",
                "schema",
                "pass --kind config, world or dataset",
            )
        ]
    for p in problems:
        (report["warnings"] if p.get("severity") == "warning" else errors).append(p)
    report["valid"] = not errors
    return report


def schema_text(name: str) -> str:
    """A generated schema as the text committed in ``schemas/`` (sorted-stable)."""
    import json

    if name not in SCHEMAS:
        raise ValueError(f"unknown schema {name!r}; known: {sorted(SCHEMAS)}")
    return json.dumps(SCHEMAS[name](), indent=2, sort_keys=False) + "\n"
