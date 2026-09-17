"""sensoryforge-plugin-template: a worked, installable SensoryForge plugin.

Two components, each registered under the ``sensoryforge.components``
entry-point group declared in ``pyproject.toml``:

- :class:`~sensoryforge_plugin_template.rf_builder.RadialFalloffRFBuilder`
  (``"radial_falloff"`` in ``INNERVATION_REGISTRY``): a receptive-field
  builder with a linear radial falloff.
- :class:`~sensoryforge_plugin_template.processing.GainThresholdLayer`
  (``"gain_threshold"`` in ``PROCESSING_REGISTRY``): a processing layer
  applying a gain and a rectifying threshold.

Neither is imported here -- SensoryForge discovers and imports them lazily,
through the entry points, the first time
:func:`sensoryforge.register_components.register_all` runs (which happens
at import time of ``sensoryforge`` itself). See ``README.md`` for the
install and verification steps.
"""
