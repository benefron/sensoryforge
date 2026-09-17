"""Guard against configuration_schema.md drifting from the config dataclasses (F-059).

Wave M added fields to ``GridConfig``/``PopulationConfig``/``StimulusConfig`` that
``docs/user_guide/configuration_schema.md`` never documented, and separately the
stimulus/simulation tables named fields (``center_x``, ``duration``, ``dt``-as-primary)
that never existed on those dataclasses. Rather than re-generating the page (it carries
hand-written grouping, examples and cross-references that are worth keeping), this test
checks every public dataclass field name appears literally somewhere in the page, so a
future field addition that isn't documented fails CI instead of silently drifting again.
"""

import dataclasses
from pathlib import Path

import pytest

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SimulationConfig,
    StimulusConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DOC = REPO_ROOT / "docs" / "user_guide" / "configuration_schema.md"

# A handful of fields are internal/deprecated bookkeeping that the page deliberately
# doesn't give its own row (the deprecated `dt` alias IS documented; nothing is
# excluded from GridConfig/PopulationConfig/StimulusConfig -- every field must appear).
DATACLASSES = [
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    StimulusConfig,
    SimulationConfig,
]


@pytest.fixture(scope="module")
def doc_text():
    return SCHEMA_DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("cls", DATACLASSES, ids=lambda c: c.__name__)
def test_every_dataclass_field_is_documented(cls, doc_text):
    missing = [f.name for f in dataclasses.fields(cls) if f"`{f.name}`" not in doc_text]
    assert not missing, (
        f"{cls.__name__} field(s) {missing} not documented (as `field_name`) in "
        f"{SCHEMA_DOC.relative_to(REPO_ROOT)} -- add a row for each (F-059)"
    )


def _documented_fields_by_class(doc_text):
    """Field names in each class's tables, keyed by class name.

    The page gives each dataclass a ``## ClassName`` section whose tables
    have one ``| `field` | ...`` row per field; the nested ``PopulationInput``
    and ``RFBuilderConfig`` tables sit inside ``PopulationConfig`` under a
    bold ``**`Name` fields:**`` line, and the next ``####`` heading returns
    to ``PopulationConfig``. Example blocks and non-class sections are
    ignored, so a field name mentioned in prose or in example YAML does not
    count as a documented row.
    """
    import re

    names = {cls.__name__ for cls in DATACLASSES}
    documented = {name: set() for name in names}
    owner = None
    for line in doc_text.splitlines():
        heading = re.match(r"^## (\w+)", line)
        if heading:
            owner = heading.group(1) if heading.group(1) in names else None
            continue
        nested = re.match(r"^\*\*`(\w+)` fields:\*\*", line)
        if nested and nested.group(1) in names:
            owner = nested.group(1)
            continue
        if line.startswith("#### ") and owner in ("PopulationInput", "RFBuilderConfig"):
            owner = "PopulationConfig"
        if line.startswith("### Example"):
            owner = None
        row = re.match(r"^\| `(\w+)` \|", line)
        if row and owner:
            documented[owner].add(row.group(1))
    return documented


@pytest.mark.parametrize("cls", DATACLASSES, ids=lambda c: c.__name__)
def test_every_documented_field_exists(cls, doc_text):
    """The reverse direction, which is the drift this page actually had.

    The check above only proves each real field is mentioned. The drift
    Wave U found went the other way: the stimulus and simulation tables
    had rows for ``center_x``, ``duration`` and a primary ``dt`` that no
    dataclass defines, and a reader configuring from the page would have
    written keys that are silently ignored. A table row for a field that
    does not exist fails here.
    """
    documented = _documented_fields_by_class(doc_text)[cls.__name__]
    assert documented, (
        f"no field rows found for {cls.__name__}; the page structure this test "
        "parses may have changed, which would make the check vacuous"
    )
    real = {f.name for f in dataclasses.fields(cls)}
    phantom = sorted(documented - real)
    assert not phantom, (
        f"{SCHEMA_DOC.relative_to(REPO_ROOT)} documents {cls.__name__} field(s) "
        f"{phantom} that the dataclass does not have"
    )
