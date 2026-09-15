"""Receptive-field builders beyond the biological innervation methods.

Every builder is a :class:`~sensoryforge.core.innervation.BaseInnervation`
subclass registered in ``INNERVATION_REGISTRY`` whose :meth:`build` returns a
:class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank`:

- :class:`~sensoryforge.core.rf_builders.template.TemplateRFBuilder`
  (``"template"``): designed receptive fields from one resolvable distance.
- :class:`~sensoryforge.core.rf_builders.imported.ImportedRFBuilder`
  (``"imported"``): receptive fields read from files.

See ``docs/user_guide/receptive_fields.md`` and
``docs/developer_guide/add_rf_builder.md``.
"""

from sensoryforge.core.rf_builders.template import TemplateRFBuilder

__all__ = ["TemplateRFBuilder"]
