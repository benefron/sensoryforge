"""YAML utilities with duplicate-key validation."""

from __future__ import annotations

from typing import Any, TextIO

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key '{key}' detected in YAML.")
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml(stream: TextIO) -> Any:
    """Load YAML from a file-like object with duplicate-key validation.

    Args:
        stream: File-like object containing YAML content.

    Returns:
        Parsed YAML data structure.

    Raises:
        ValueError: If duplicate keys are detected in the YAML mapping.
    """
    return yaml.load(stream, Loader=UniqueKeyLoader)
