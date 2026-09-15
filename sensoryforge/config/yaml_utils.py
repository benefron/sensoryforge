"""YAML utilities with duplicate-key validation and shared config loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, TextIO, Union

import yaml

logger = logging.getLogger(__name__)


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys."""


def _construct_mapping(
    loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False
) -> dict:
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


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and honour its ``plugins:`` list.

    This is the single shared entry point every config loader in
    SensoryForge (CLI, :class:`~sensoryforge.core.batch_executor.BatchExecutor`,
    :meth:`~sensoryforge.config.schema.SensoryForgeConfig.from_yaml_file`, and
    the GUI's "Load YAML Configuration" action) should call, so that a
    config's ``plugins:`` list is honoured identically regardless of which
    loader reads it (F-048).

    A ``plugins:`` entry is a dotted import path to a module, optionally
    suffixed with ``:attr`` to call a specific callable after import (e.g.
    ``my_package.plugin:register``). See :mod:`sensoryforge.plugins` for the
    full convention. Each successfully loaded plugin is logged at INFO
    level.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Parsed configuration dictionary (the ``plugins:`` key, if present,
        is left in place for downstream consumers).

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is empty or does not parse to a dict.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = load_yaml(f)

    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"YAML file {config_path} did not produce a dict "
            f"(got {type(config).__name__})"
        )

    plugins = config.get("plugins")
    if plugins:
        from sensoryforge.plugins import load_plugin_import_paths

        loaded = load_plugin_import_paths(plugins)
        for name in loaded:
            logger.info("Loaded plugin from config '%s': %s", config_path, name)

    return config
