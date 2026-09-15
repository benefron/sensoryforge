"""Plugin discovery for third-party SensoryForge components (G2).

Two discovery paths, both best-effort: a broken or missing plugin produces a
``UserWarning`` and is skipped, never a crash for the whole process.

1. **Entry points.** A distribution can advertise components under the
   ``sensoryforge.components`` group in its packaging metadata, e.g. (in
   ``pyproject.toml``)::

       [project.entry-points."sensoryforge.components"]
       my_filter = "my_package.plugin:register"

   The referenced object is loaded and, if callable, called with no
   arguments -- it is expected to import the relevant registries
   (:mod:`sensoryforge.registry`) and call ``.register(name, cls)`` itself,
   exactly like a hand-written entry in ``register_components.py``.
   :func:`discover_entry_point_plugins` is called automatically from
   :func:`sensoryforge.register_components.register_all`.

2. **YAML ``plugins:`` list.** A config file may list dotted import paths::

       plugins:
         - my_package.plugin

   Each path is imported for its side effects (the module registers its own
   components at import time, as ``register_components`` does), optionally
   followed by ``:attr`` to call a specific callable after import, e.g.
   ``my_package.plugin:register``. :func:`load_plugin_import_paths` is
   called from :func:`sensoryforge.cli.load_config_file` when a config
   declares ``plugins:``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Iterable, List

ENTRY_POINT_GROUP = "sensoryforge.components"


def discover_entry_point_plugins(group: str = ENTRY_POINT_GROUP) -> List[str]:
    """Load and invoke every entry point registered under ``group``.

    Args:
        group: Entry-point group name to scan.

    Returns:
        Names of the entry points that loaded and ran successfully.
    """
    import importlib.metadata as importlib_metadata

    loaded: List[str] = []
    try:
        entry_points = importlib_metadata.entry_points(group=group)
    except Exception as exc:  # pragma: no cover - defensive, metadata backend issues
        warnings.warn(
            f"sensoryforge plugin discovery: could not enumerate entry points "
            f"for group '{group}': {exc}",
            UserWarning,
        )
        return loaded

    for ep in entry_points:
        try:
            obj = ep.load()
            if callable(obj):
                obj()
            loaded.append(ep.name)
        except Exception as exc:
            warnings.warn(
                f"sensoryforge plugin '{ep.name}' ({ep.value}) failed to load: {exc}",
                UserWarning,
            )
    return loaded


def load_plugin_import_paths(import_paths: Iterable[str]) -> List[str]:
    """Import each ``module`` or ``module:attr`` path in ``import_paths``.

    Args:
        import_paths: Dotted import paths, optionally suffixed with
            ``:attr`` to call a callable after import (e.g. a ``register()``
            function). Without ``:attr``, the module is imported for its
            side effects only.

    Returns:
        The import paths that loaded successfully.
    """
    loaded: List[str] = []
    for path in import_paths:
        module_path, _, attr_name = path.partition(":")
        try:
            module = importlib.import_module(module_path)
            if attr_name:
                obj = getattr(module, attr_name)
                if callable(obj):
                    obj()
            loaded.append(path)
        except Exception as exc:
            warnings.warn(
                f"sensoryforge plugin import '{path}' failed: {exc}",
                UserWarning,
            )
    return loaded
