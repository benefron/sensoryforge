"""Tests for YAML loading utilities with duplicate key validation."""

from __future__ import annotations

import io
import pytest

from sensoryforge.config.yaml_utils import load_yaml


def test_load_yaml_accepts_valid_mapping() -> None:
    yaml_text = """
    pipeline:
      device: cpu
      seed: 7
    """
    result = load_yaml(io.StringIO(yaml_text))
    assert result["pipeline"]["device"] == "cpu"
    assert result["pipeline"]["seed"] == 7


def test_load_yaml_rejects_duplicate_keys() -> None:
    yaml_text = """
    bayesian_estimator:
      rho: 0.9
    bayesian_estimator:
      rho: 0.95
    """
    with pytest.raises(ValueError, match="Duplicate key"):
        load_yaml(io.StringIO(yaml_text))
