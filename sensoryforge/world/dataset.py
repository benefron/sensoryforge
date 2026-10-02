"""Data sets on a world: splits, strata, probes, seeds and the manifest (spec §6)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml

from sensoryforge.provenance import source_info
from sensoryforge.world import rng
from sensoryforge.world.distributions import AxisSpec
from sensoryforge.world.sampling import Draw, Session, sample, session
from sensoryforge.world.schema import ClassSpec, World, load_world

FORMAT = "sensoryforge-dataset/1"
#: Split name -> its kind when the split gives no ``kind:``.
SPLIT_KINDS = {
    "train": "declared",
    "validation": "declared",
    "test": "stratified",
    "probes": "probes",
    "held_out": "held_out",
    "sessions": "sessions",
    "fixed": "fixed",
}
_KINDS = set(SPLIT_KINDS.values())
_SPLIT_KEYS = {
    "kind",
    "n",
    "repeats",
    "noise_repeats",
    "stratified",
    "per_bin",
    "bins",
    "duration_ms",
    "draws",
}
_DATASET_KEYS = {"name", "world", "world_id", "seed", "duration_ms", "splits"}
#: An int axis with more values than this cannot be stratified one bin per value.
MAX_INT_STRATA = 64

Item = Union[Draw, Session]


@dataclass(frozen=True)
class SplitSpec:
    """One split of a data set (spec §6.1)."""

    name: str
    kind: str
    n: int = 0
    repeats: int = 1
    noise_repeats: int = 1
    bins: int = 0
    per_bin: int = 0
    duration_ms: Optional[float] = None
    draws: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """The normalised split (what the data-set id hashes)."""
        out: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "repeats": self.repeats,
            "noise_repeats": self.noise_repeats,
        }
        if self.n:
            out["n"] = self.n
        if self.bins:
            out["bins"] = self.bins
        if self.per_bin:
            out["per_bin"] = self.per_bin
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.draws:
            out["draws"] = list(self.draws)
        return out


@dataclass
class DatasetSpec:
    """A parsed data-set spec. Build it with :func:`load_dataset`."""

    name: str
    world: World
    seed: int
    duration_ms: float
    splits: List[SplitSpec]
    source: Dict[str, Any]
    dataset_id: str = ""


def _positive_int(
    raw: Dict[str, Any], key: str, where: str, default: Optional[int] = None
) -> int:
    value = raw.get(key, default)
    if value is None or int(value) < 1:
        raise ValueError(f"{where}: {key} >= 1 required, got {value!r}")
    return int(value)


def _strata(raw: Dict[str, Any], where: str) -> Tuple[int, int]:
    strat = raw.get("stratified")
    if not isinstance(strat, dict):
        raise ValueError(f"{where}: give stratified: {{bins: B, per_bin: m}}")
    return (
        _positive_int(strat, "bins", f"{where}.stratified"),
        _positive_int(strat, "per_bin", f"{where}.stratified"),
    )


def _parse_split(
    name: str, raw: Any, world: World, test_bins: Optional[int]
) -> SplitSpec:
    where = f"dataset.splits.{name}"
    raw = dict(raw or {})
    unknown = set(raw) - _SPLIT_KEYS
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_SPLIT_KEYS)}"
        )
    kind = raw.get("kind", SPLIT_KINDS.get(name))
    if kind not in _KINDS:
        raise ValueError(
            f"{where}: unknown kind {kind!r}; name the split one of "
            f"{sorted(SPLIT_KINDS)} or give kind: one of {sorted(_KINDS)}"
        )
    common = {
        "name": name,
        "kind": kind,
        "repeats": _positive_int(raw, "repeats", where, 1),
        "noise_repeats": _positive_int(raw, "noise_repeats", where, 1),
    }
    if kind == "declared":
        return SplitSpec(**common, n=_positive_int(raw, "n", where))
    if kind == "stratified":
        bins, per_bin = _strata(raw, where)
        return SplitSpec(**common, bins=bins, per_bin=per_bin)
    if kind == "probes":
        return SplitSpec(
            **common,
            per_bin=_positive_int(raw, "per_bin", where),
            bins=_positive_int(raw, "bins", where, test_bins or 5),
        )
    if kind == "held_out":
        if not world.held_out:
            raise ValueError(f"{where}: the world declares no held_out classes")
        if "stratified" in raw:
            bins, per_bin = _strata(raw, where)
            return SplitSpec(**common, bins=bins, per_bin=per_bin)
        return SplitSpec(**common, n=_positive_int(raw, "n", where))
    if kind == "sessions":
        duration = float(raw.get("duration_ms", 0))
        if duration <= 0:
            raise ValueError(f"{where}: duration_ms must be > 0")
        return SplitSpec(
            **common, n=_positive_int(raw, "n", where), duration_ms=duration
        )
    draws = tuple(str(d) for d in raw.get("draws") or [])
    if not draws:
        raise ValueError(f"{where}: list the fixed draws, e.g. draws: [braille_H]")
    missing = [d for d in draws if d not in world.fixed]
    if missing:
        raise ValueError(
            f"{where}: no fixed draw {missing}; the world has {sorted(world.fixed)}"
        )
    return SplitSpec(**common, draws=draws)


def load_dataset(
    source: Union[str, Path, Dict[str, Any]],
    base_dir: Optional[Union[str, Path]] = None,
) -> DatasetSpec:
    """Load a data-set spec (``dataset:`` YAML) and the world it names.

    Args:
        source: A YAML path, or the mapping itself.
        base_dir: Where a relative ``world:`` path is resolved for a mapping
            (default: the current directory); a file resolves against its own
            directory.

    Raises:
        ValueError: For an invalid spec, or a ``world_id`` pin that does not
            match the world's content.
    """
    if isinstance(source, dict):
        data = source
        base = Path(base_dir) if base_dir is not None else Path.cwd()
    else:
        path = Path(source)
        data = yaml.safe_load(path.read_text())
        base = path.resolve().parent
    raw = data.get("dataset", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(
            "a data set is a mapping (optionally under a top-level 'dataset:' key)"
        )
    unknown = set(raw) - _DATASET_KEYS
    if unknown:
        raise ValueError(
            f"dataset: unknown keys {sorted(unknown)}; allowed: {sorted(_DATASET_KEYS)}"
        )
    world_ref = raw.get("world")
    if isinstance(world_ref, dict):
        world = load_world(world_ref)
    elif isinstance(world_ref, (str, Path)):
        world_path = Path(world_ref)
        world = load_world(
            world_path if world_path.is_absolute() else base / world_path
        )
    else:
        raise ValueError(
            "dataset.world: give a path to a world file, or a world mapping"
        )
    pin = raw.get("world_id")
    if pin is not None and pin != world.world_id:
        raise ValueError(
            f"dataset.world_id pins {pin} but the world's content gives "
            f"{world.world_id}: "
            "the world changed since this data set was declared"
        )
    if "seed" not in raw:
        raise ValueError("dataset.seed: required (an integer)")
    seed = int(raw["seed"])
    duration_ms = float(raw.get("duration_ms", 0))
    if duration_ms <= 0:
        raise ValueError("dataset.duration_ms: required, > 0")
    splits_raw = raw.get("splits") or {}
    if not splits_raw:
        raise ValueError("dataset.splits: declare at least one split")
    test_raw = splits_raw.get("test") or {}
    test_bins = (
        (test_raw.get("stratified") or {}).get("bins")
        if isinstance(test_raw, dict)
        else None
    )
    splits = [_parse_split(str(n), s, world, test_bins) for n, s in splits_raw.items()]
    name = str(raw.get("name", "dataset"))
    normal = {
        "format": FORMAT,
        "name": name,
        "seed": seed,
        "duration_ms": duration_ms,
        "world_id": world.world_id,
        "splits": [s.to_dict() for s in splits],
    }
    canonical = json.dumps(normal, sort_keys=True, separators=(",", ":"))
    return DatasetSpec(
        name=name,
        world=world,
        seed=seed,
        duration_ms=duration_ms,
        splits=splits,
        source=json.loads(json.dumps(raw, default=str)),
        dataset_id="d-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )


@dataclass
class Entry:
    """One row of a data set's manifest (spec §6.4-§6.5)."""

    entry: str
    split: str
    repeat: int
    noise_repeat: int
    class_name: Optional[str]
    item: Item
    bins: Dict[str, Any]
    probe: Optional[Dict[str, str]]
    seeds: Dict[str, Optional[int]]
    duration_ms: float
    world_id: str
    dataset_id: str
    base: str = field(default="", repr=False)

    @property
    def truncated(self) -> bool:
        """True when the draw runs past ``duration_ms`` (a session: its last draw)."""
        if isinstance(self.item, Session):
            return self.item.truncated
        return self.item.end_ms > self.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """The manifest row."""
        return {
            "entry": self.entry,
            "split": self.split,
            "repeat": self.repeat,
            "noise_repeat": self.noise_repeat,
            "class": self.class_name,
            "draw": self.item.to_dict(),
            "bins": dict(self.bins),
            "probe": self.probe,
            "seeds": dict(self.seeds),
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "world_id": self.world_id,
            "dataset_id": self.dataset_id,
        }


def stratify_class(
    world: World, cls: ClassSpec, bins: int, per_bin: int, class_seed: int
) -> Tuple[List[Draw], List[Dict[str, Any]]]:
    """A Latin hypercube of ``bins * per_bin`` draws of one class (spec §6.2).

    Returns:
        ``(draws, labels)``: each draw and its bin label per random axis.
    """
    n = bins * per_bin
    seeds = rng.draw_seeds(class_seed, np.arange(n))
    columns: Dict[str, List[Any]] = {}
    labels: List[Dict[str, Any]] = [{} for _ in range(n)]
    for name, axis in cls.axes.items():
        if not axis.is_random:
            columns[name] = [axis.value] * n
            continue
        order = rng.permutation(n, class_seed, name)
        support = axis.support()
        if support is not None:
            if axis.form == "int" and len(support) > MAX_INT_STRATA:
                raise ValueError(
                    f"class {cls.name!r}: int axis {name!r} has {len(support)} values; "
                    f"stratifying allows at most {MAX_INT_STRATA} "
                    "(declare it as a float range)"
                )
            picks = (np.arange(n) % len(support))[order]
            columns[name] = [support[i] for i in picks]
            for j, i in enumerate(picks):
                labels[j][name] = support[i]
        elif axis.form == "numeric":
            picks = np.repeat(np.arange(bins), per_bin)[order]
            columns[name] = axis.stratum_values(picks, bins, rng.uniforms(seeds, name))
            for j, b in enumerate(picks):
                labels[j][name] = axis.bin_label(int(b), bins)
        else:
            raise ValueError(
                f"class {cls.name!r}: axis {name!r} uses distribution {axis.dist!r}, "
                "which declares no finite support, so it cannot be stratified"
            )
    draws = [
        Draw(
            world=world,
            class_name=cls.name,
            values={k: columns[k][j] for k in cls.axes},
            seed=int(class_seed),
            index=j,
            draw_seed=int(seeds[j]),
            sampling="stratified",
        )
        for j in range(n)
    ]
    return draws, labels


def _by_name(classes: Dict[str, ClassSpec]) -> List[ClassSpec]:
    """Classes in name order: a data set never depends on how a world lists them."""
    return [classes[name] for name in sorted(classes)]


def _probe_axes(cls: ClassSpec) -> List[AxisSpec]:
    return [
        a
        for a in cls.random_axes
        if a.form == "numeric" and not a.circular and a.probes
    ]


def _probe_draws(
    world: World, cls: ClassSpec, axis: AxisSpec, side: str, split: SplitSpec, seed: int
) -> Optional[List[Draw]]:
    n = split.per_bin
    seeds = rng.draw_seeds(seed, np.arange(n))
    values = axis.probe_values(side, split.bins, rng.uniforms(seeds, axis.name))
    if values is None:
        return None
    columns: Dict[str, List[Any]] = {}
    for name, other in cls.axes.items():
        if name == axis.name:
            columns[name] = values
        elif other.is_random:
            columns[name] = other.sample(rng.uniforms(seeds, name))
        else:
            columns[name] = [other.value] * n
    return [
        Draw(
            world=world,
            class_name=cls.name,
            values={k: columns[k][j] for k in cls.axes},
            seed=int(seed),
            index=j,
            draw_seed=int(seeds[j]),
            sampling="probe",
            out_of_range=(axis.name,),
        )
        for j in range(n)
    ]


def _split_items(
    spec: DatasetSpec, split: SplitSpec, split_seed: int
) -> Iterator[Tuple[str, Item, Dict[str, Any], Optional[Dict[str, str]]]]:
    """``(stem, item, bins, probe)`` for every entry of one (split, repeat)."""
    world = spec.world
    if split.kind == "declared":
        for i, draw in enumerate(sample(world, n=split.n, seed=split_seed)):
            yield f"{i:05d}", draw, {}, None
    elif split.kind in ("stratified", "held_out") and split.bins:
        classes = world.classes if split.kind == "stratified" else world.held_out
        for cls in _by_name(classes):
            draws, labels = stratify_class(
                world, cls, split.bins, split.per_bin, rng.seed53(split_seed, cls.name)
            )
            for i, (draw, label) in enumerate(zip(draws, labels)):
                yield f"{cls.name}/{i:04d}", draw, label, None
    elif split.kind == "held_out":
        held = sorted(world.held_out)
        for i, draw in enumerate(
            sample(world, n=split.n, seed=split_seed, classes=held)
        ):
            yield f"{draw.class_name}/{i:04d}", draw, {}, None
    elif split.kind == "probes":
        for cls in _by_name(world.classes):
            for axis in _probe_axes(cls):
                for side in ("below", "above"):
                    side_seed = rng.seed53(split_seed, cls.name, axis.name, side)
                    draws = _probe_draws(world, cls, axis, side, split, side_seed)
                    for i, draw in enumerate(draws or []):
                        yield (
                            f"{cls.name}/{axis.name}-{side}/{i:03d}",
                            draw,
                            {axis.name: side},
                            {"axis": axis.name, "side": side},
                        )
    elif split.kind == "sessions":
        for i in range(split.n):
            yield f"{i:03d}", session(
                world, split.duration_ms, seed=split_seed, index=i
            ), {}, None
    else:
        for name in split.draws:
            yield name, world.fixed_draw(name), {}, None


def build_dataset(spec: DatasetSpec) -> List[Entry]:
    """Every entry of the data set, in split order, then id order (spec §6.4).

    Raises:
        ValueError: If any draw seed or noise seed appears twice.
    """
    entries: List[Entry] = []
    for split in spec.splits:
        for repeat in range(split.repeats):
            split_seed = rng.seed53(spec.seed, split.name, repeat)
            if split.kind == "declared" or split.repeats > 1:
                prefix = f"{split.name}/r{repeat}"
            else:
                prefix = split.name
            duration = (
                split.duration_ms if split.kind == "sessions" else spec.duration_ms
            )
            for stem, item, bins, probe in _split_items(spec, split, split_seed):
                base = f"{prefix}/{stem}"
                is_session = isinstance(item, Session)
                draw_seed = item.session_seed if is_session else item.draw_seed
                for noise_repeat in range(split.noise_repeats):
                    entry_id = (
                        base if split.noise_repeats == 1 else f"{base}.n{noise_repeat}"
                    )
                    entries.append(
                        Entry(
                            entry=entry_id,
                            split=split.name,
                            repeat=repeat,
                            noise_repeat=noise_repeat,
                            class_name=None if is_session else item.class_name,
                            item=item,
                            bins=bins,
                            probe=probe,
                            seeds={
                                "draw": draw_seed,
                                "noise": rng.seed53(spec.seed, "noise", entry_id),
                            },
                            duration_ms=float(duration),
                            world_id=spec.world.world_id,
                            dataset_id=spec.dataset_id,
                            base=base,
                        )
                    )
    _check_unique(entries)
    return entries


def _check_unique(entries: List[Entry]) -> None:
    noise_owner: Dict[int, str] = {}
    draw_owner: Dict[int, str] = {}
    for e in entries:
        noise = e.seeds["noise"]
        if noise in noise_owner:
            raise ValueError(
                f"noise seed {noise} of {e.entry} repeats {noise_owner[noise]}; "
                "choose another data-set seed"
            )
        noise_owner[noise] = e.entry
        draw = e.seeds["draw"]
        if draw is None:
            continue
        owner = draw_owner.setdefault(draw, e.base)
        if owner != e.base:
            raise ValueError(
                f"draw seed {draw} of {e.entry} repeats {owner}; "
                "choose another data-set seed"
            )


def skipped_probes(spec: DatasetSpec) -> List[str]:
    """``"<class>/<axis>-<side>"`` for probe sides with no room in the domain."""
    out: List[str] = []
    for split in spec.splits:
        if split.kind != "probes":
            continue
        for cls in _by_name(spec.world.classes):
            for axis in _probe_axes(cls):
                for side in ("below", "above"):
                    if axis.probe_values(side, split.bins, np.zeros(1)) is None:
                        out.append(f"{cls.name}/{axis.name}-{side}")
    return sorted(set(out))


def write_dataset(
    spec: DatasetSpec, entries: List[Entry], out_dir: Union[str, Path]
) -> Path:
    """Write ``dataset.json`` and ``manifest.jsonl`` (a row per entry)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    for e in entries:
        per_split = counts.setdefault(e.split, {})
        key = e.class_name or "session"
        per_split[key] = per_split.get(key, 0) + 1
    info = {
        "format": FORMAT,
        "name": spec.name,
        "dataset_id": spec.dataset_id,
        "world_id": spec.world.world_id,
        "seed": spec.seed,
        "duration_ms": spec.duration_ms,
        "splits": [s.to_dict() for s in spec.splits],
        "spec": spec.source,
        "world": spec.world.to_dict(),
        "world_description": spec.world.description,
        "sensoryforge": source_info(),
        "n_entries": len(entries),
        "counts": counts,
        "skipped_probes": skipped_probes(spec),
    }
    (out / "dataset.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    with open(out / "manifest.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")
    return out


def load_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """The rows of ``manifest.jsonl`` (``path`` is the file or its directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
