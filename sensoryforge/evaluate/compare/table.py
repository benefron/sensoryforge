"""Printable tables of ``compare.json`` (report contract, section 9).

Numbers are written with ``repr``: the shortest text that reads back as the same number,
so a printed difference is the stored one, never a rounded copy. A missing value is
``-``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from sensoryforge.evaluate.compare.pairing import STATS

_BAR = 24


def number(value: Any) -> str:
    """A value as the table prints it (``repr`` for numbers, ``-`` for ``None``)."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _label(row: Mapping[str, Any]) -> str:
    text = row["id"]
    if row.get("channel") is not None:
        text += f"[ch{row['channel']}]"
    if row.get("population") is not None:
        text += f"[{row['population']}]"
    return text


def _render(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    return "\n".join(
        [line(header), line(["-" * w for w in widths])] + [line(r) for r in rows]
    )


def _units(compare: Mapping[str, Any]) -> Dict[str, str]:
    return {
        m["id"]: m["unit"]["symbol"]
        for m in compare["metrics"]
        if m["unit"] is not None
    }


def _group(compare: Mapping[str, Any], group: str) -> Any:
    if group == "dataset":
        return compare["dataset"]
    kind, sep, label = group.partition(":")
    if not sep or kind not in ("class", "split"):
        raise ValueError(
            f"group must be 'dataset', 'class:LABEL' or 'split:LABEL', got {group!r}"
        )
    labels = compare["groupings"][kind]
    if label not in labels:
        raise ValueError(f"no {kind} {label!r}; there are {sorted(labels)}")
    return labels[label]


def format_stat_table(
    compare: Mapping[str, Any],
    group: str = "dataset",
    stats: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
) -> str:
    """The measured/declared/difference table of one group's summaries.

    Args:
        compare: The ``compare.json`` content.
        group: ``"dataset"``, ``"class:LABEL"`` or ``"split:LABEL"``.
        stats: Statistics to show (default all of them: n, mean, std, min, q05 ... max).
        metrics: Metric ids to show (default all).

    Returns:
        A text table, one row per metric and statistic, with the unit and the pairing.

    Raises:
        ValueError: If the group or a statistic is unknown.
    """
    wanted = tuple(STATS if stats is None else stats)
    for stat in wanted:
        if stat not in STATS:
            raise ValueError(f"unknown statistic {stat!r}; there are {list(STATS)}")
    units = _units(compare)
    rows = [
        [
            _label(p),
            p["stat"],
            number(p["measured"]),
            number(p["declared"]),
            number(p["diff"]),
            number(p["rel_diff"]),
            units.get(p["id"], "-"),
            p["pairing"],
        ]
        for p in _group(compare, group)
        if p["stat"] in wanted and (metrics is None or p["id"] in metrics)
    ]
    return _render(
        [
            "metric",
            "stat",
            "measured",
            "declared",
            "diff",
            "rel_diff",
            "unit",
            "pairing",
        ],
        rows,
    )


def format_entry_table(
    compare_path: Union[str, Path], entry: str, metrics: Optional[Sequence[str]] = None
) -> str:
    """The pairs of one entry.

    Args:
        compare_path: The ``compare.json`` file.
        entry: The entry id.
        metrics: Metric ids to show (default all).

    Returns:
        A text table, one row per declared value of the entry.

    Raises:
        ValueError: If the entry is not in the comparison.
    """
    path = Path(compare_path)
    compare = json.loads(path.read_text(encoding="utf-8"))
    units = _units(compare)
    with open(path.parent / compare["entries"]["file"], encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["entry"] != entry:
                continue
            rows = [
                [
                    _label(p),
                    number(p["measured"]),
                    number(p["declared"]),
                    number(p["diff"]),
                    number(p["rel_diff"]),
                    number(p["resolution"]),
                    units.get(p["id"], "-"),
                    p["pairing"],
                ]
                for p in row["pairs"]
                if metrics is None or p["id"] in metrics
            ]
            return _render(
                [
                    "metric",
                    "measured",
                    "declared",
                    "diff",
                    "rel_diff",
                    "resolution",
                    "unit",
                    "pairing",
                ],
                rows,
            )
    raise ValueError(
        f"no entry {entry!r} in {path.parent / compare['entries']['file']}"
    )


def _distributions(compare: Mapping[str, Any], group: str) -> List[Dict[str, Any]]:
    block = compare["distributions"]
    if group == "dataset":
        return block["dataset"]
    kind, _, label = group.partition(":")
    _group(compare, group)  # validates the group
    return block[kind][label]


def format_distribution_table(
    compare: Mapping[str, Any], group: str = "dataset"
) -> str:
    """The mean difference, KS and Wasserstein distances of one group.

    Args:
        compare: The ``compare.json`` content.
        group: ``"dataset"``, ``"class:LABEL"`` or ``"split:LABEL"``.

    Returns:
        A text table, one row per metric.

    Raises:
        ValueError: If the group is unknown or the file has no ``distributions``.
    """
    if "distributions" not in compare:
        raise ValueError("this compare.json carries no distributions")
    units = _units(compare)
    rows = [
        [
            _label(d),
            number(d["n_pairs"]),
            number(d["diff_mean"]),
            number(d["diff_abs_mean"]),
            number(d["ks"]),
            number(d["wasserstein"]),
            units.get(d["id"], "-"),
        ]
        for d in _distributions(compare, group)
    ]
    return _render(
        [
            "metric",
            "n_pairs",
            "diff_mean",
            "diff_abs_mean",
            "ks",
            "wasserstein",
            "unit",
        ],
        rows,
    )


def format_histograms(
    compare: Mapping[str, Any], metric: str, group: str = "dataset"
) -> str:
    """The measured and declared histograms of a metric side by side.

    Args:
        compare: The ``compare.json`` content.
        metric: The metric id.
        group: ``"dataset"``, ``"class:LABEL"`` or ``"split:LABEL"``.

    Returns:
        One block per channel or population of the metric: a row per bin with its
        edges, the measured and declared counts and a bar for each.

    Raises:
        ValueError: If the group or metric is unknown.
    """
    if "distributions" not in compare:
        raise ValueError("this compare.json carries no distributions")
    found = [d for d in _distributions(compare, group) if d["id"] == metric]
    if not found:
        raise ValueError(f"no distribution of {metric!r} in {group!r}")
    blocks = []
    for d in found:
        title = f"{_label(d)}  n_pairs={d['n_pairs']}  ks={number(d['ks'])}  "
        title += f"wasserstein={number(d['wasserstein'])}"
        hist = d["histogram"]
        if hist is None:
            blocks.append(title + "\n(no pairs)")
            continue
        top = max(max(hist["measured"]), max(hist["declared"]), 1)

        def bar(count: int) -> str:
            return "#" * ((count * _BAR + top - 1) // top)

        edges = hist["edges"]
        last = len(hist["measured"]) - 1
        rows = [
            [
                f"[{number(edges[i])}, {number(edges[i + 1])}"
                + ("]" if i == last else ")"),
                number(hist["measured"][i]),
                bar(hist["measured"][i]),
                number(hist["declared"][i]),
                bar(hist["declared"][i]),
            ]
            for i in range(len(hist["measured"]))
        ]
        blocks.append(
            title + "\n" + _render(["bin", "measured", "", "declared", ""], rows)
        )
    return "\n\n".join(blocks)
