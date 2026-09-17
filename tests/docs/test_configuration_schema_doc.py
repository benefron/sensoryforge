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
