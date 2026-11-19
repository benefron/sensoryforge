# Quick Start: The GUI

In five minutes you open the GUI on a working experiment, change its stimulus, run it,
look at the spikes, and save it as a config the command line runs unchanged.

## 1. Install with the GUI extra and launch

The GUI needs PyQt5, which the `gui` extra adds:

```bash
pip install "sensoryforge[gui] @ git+https://github.com/benefron/sensoryforge.git"
python -m sensoryforge.gui.main
```

The window opens on the last project you used or, the first time, on the
`tactile_sa1_ra1` preset: an 80 × 80 sensor array read by an SA and an RA population.
**File > Open preset** loads any of the four shipped presets.

## 2. Change the stimulus

Click **Stimulus** in the stage list on the left. Choose `gaussian` as the type and set
its `sigma` and `amplitude`. The preview on the right renders with the same function a
run uses. A value shown in italics is the default that will run.

## 3. Run and look

Click **Run & Results**, then **Run** in the bar at the bottom (set the duration there,
for example 300 ms). The run happens in the background; the screen then shows the
stimulus, the neuron map, the raster, the population rates and one neuron's input and
membrane potential, all on one time cursor. **Play** moves the cursor; click a neuron on
the map to trace it. **Export figures…** saves every visible panel as PNG and SVG.

If a population fired no spikes, a banner names it with its peak drive: raise the
stimulus amplitude, or the population's input gain on the **Populations** screen (see
[Units and Gains](../user_guide/units_and_gains.md)).

## 4. Save it for the command line

**File > Export > YAML…** writes the experiment as a canonical config. The CLI runs it
unchanged and gives the same result:

```bash
sensoryforge run my_experiment.yml --bundle cli_run
```

**File > New project…** instead makes a folder with `config.yml` and a `runs/` folder,
and every later run is saved there as a [bundle](../user_guide/bundles.md); **Open
bundle...** on the Run & Results screen opens any bundle, from the GUI, the CLI or a batch.

## Next

- Every screen, card and bench test: the [GUI Walkthrough](../user_guide/gui_walkthrough.md).
- Sweep a parameter from the **Batch** screen, locally or as a SLURM array.
- The same run from the terminal: [Quick Start: Command Line](quickstart_cli.md).
