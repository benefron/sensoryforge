"""Execution controllers for the GUI v2 shell.

Two ways to run a :class:`~sensoryforge.config.schema.SensoryForgeConfig` from
the window, and one renderer they share:

* :mod:`~sensoryforge.gui.execution.render` -- turn a config into the stimulus
  tensor the engine takes, on the config's own grid canvas.
* :mod:`~sensoryforge.gui.execution.run_controller` -- one interactive run on a
  ``QThread`` worker, with progress, cooperative cancel and a bundle.
* :mod:`~sensoryforge.gui.execution.sweep_controller` -- a parameter sweep as
  one directory (and one ``sensoryforge run`` subprocess, or one SLURM array
  task) per combination, so an out-of-memory combination cannot take the
  window down with it.
"""
