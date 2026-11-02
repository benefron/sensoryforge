## SensoryForge

This project uses SensoryForge, a toolkit for designing sets of sensor stimuli, simulating
how a sensor responds and evaluating the result. It is installed from source (not yet on
PyPI).

- For SensoryForge work (a world, a data set, a simulation, an evaluation, a new
  component) use the `sensoryforge-*` skills. Pick the one whose purpose matches what the
  user means, not only their words, and ask when it is unclear. `sensoryforge-workflow`
  chains the others.
- Every skill starts with `sensoryforge agents brief`, which prints how this user wants
  to work and where the glossary is. If nothing is saved yet, offer
  `sensoryforge agents setup`.
- Do not invent physics, models, parameters or units. They come from the user or from
  sources the user approves (`sensoryforge-ground-assumptions`), and every assumption is
  labelled with where it came from.
- Name a command with `sensoryforge <command> --help` instead of copying flags from
  memory; `sensoryforge list-components` and `sensoryforge describe` list what exists.
- If the skills are not installed for this tool, `sensoryforge agents export` writes them.
