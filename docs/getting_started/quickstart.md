# Quick Starts

Five ways in, five minutes each. Every one starts from a plain install
([Installation](installation.md): from the git URL, no clone; not yet on PyPI) and needs no
example file.

| You want to | Quick start | You end with |
|---|---|---|
| Add your own neuron, filter, stimulus, solver, grid or receptive-field builder | [Extend SensoryForge](quickstart_extend.md) | a plugin package that passes the contract test and runs in a config |
| Run a simulation from the terminal | [Command line](quickstart_cli.md) | a data bundle from a shipped recipe, with your own stimulus, read in Python |
| Design an experiment by eye | [The GUI](quickstart_gui.md) | a run's raster, rates and traces, and its config saved for the command line |
| Generate stimuli at scale and check them | [World to data set to report](quickstart_world.md) | a simulated data set and a report of measured against declared values |
| Say what you want in your own words | [Ask an agent](../agents/index.md#five-minute-start) | the skills installed in your agent and your working style saved |

The commands in each were run from a wheel installed into a fresh environment; on a
laptop CPU each run takes seconds, and the world quick start's 40 draws about half a
minute.

## From Python

[First Simulation](first_simulation.md) builds a pipeline in a script, runs it and plots
the spikes.

## Next

- [Core Concepts](concepts.md): sensor arrays, receptive fields, filters, neurons and
  populations, and how they connect.
- [User Guide](../user_guide/overview.md): every feature in depth.
- [Tutorials](../tutorials/quickstart_tutorial.md): longer worked examples.
