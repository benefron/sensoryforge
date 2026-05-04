# SensoryForge GUI Walkthrough

This guide walks through a complete simulation from start to finish using
the SensoryForge graphical interface.

## Launch

```bash
conda activate sensoryforge
python sensoryforge/gui/main.py
```

The window opens with five tabs:

| Tab | Purpose |
|-----|---------|
| **Grid & Innervation** | Configure receptor grid and neuron populations |
| **Stimulus Designer** | Design the tactile stimulus |
| **Spiking Neurons** | Choose neuron model and run simulation |
| **Visualization** | View spike rasters and drive signals |
| **Batch** | Run parameter sweeps and export SLURM scripts |

### Expert mode

Each tab has an **Expert mode** checkbox at the top of its control panel
(unchecked by default).

| Mode | What you see |
|------|-------------|
| **Basic** (default) | Essential controls only — grid size, population settings, neuron model, run button |
| **Expert** | All advanced controls — separate seeds, position offsets, weight ranges, far-connection tuning, filter parameters, DSL editor, CSV import/export |

State is saved in `QSettings` and persists across sessions.

---

## Step 1 — Grid & Innervation

### 1a. Create a grid

1. In the **Grid & Innervation** tab, click **Add Grid**.
2. Set **Rows** and **Cols** (e.g. 20 × 20).
3. Set **Spacing (mm)** — typical skin receptor density: 0.15–0.5 mm.
4. Arrangement: **grid** (regular) or **hex** (hexagonal packing).
5. Click **Generate Grids** to create the receptor array.

The scatter plot on the right shows receptor positions.

### 1b. Add a neuron population

1. Click **Add Population**.
2. Set **Neuron type** (SA, RA, or SA2).
3. Set **Neurons per row** — more neurons = finer spatial resolution.
4. Set **Innervation method**: `gaussian` (smooth receptive fields) or
   `point` (nearest-neighbour).
5. Set **σ_d (mm)**: Gaussian receptive field width.
6. Click the 🔵 circle in the colour column to pick a display colour.

The innervation weights are built automatically.

### 1c. Import a custom population from CSV

If you have pre-computed neuron positions and innervation weights, you can
bypass the built-in innervation module entirely:

1. Add a population as normal (step 1b).
2. Enable **Expert mode** (see below) to reveal the **Custom CSV** section
   inside Population Settings.
3. Click **Export CSV Folder…** on any existing instantiated population to
   create a valid folder with the right file format.
4. Edit the exported CSVs or replace them with your own data.
5. Click **Import CSV Folder…** and select the folder.

The folder must contain three files:

| File | Format | Description |
|------|--------|-------------|
| `neuron_positions.csv` | header row `x_mm,y_mm`, then N rows | Neuron center coordinates in mm |
| `innervation_weights.csv` | N rows × M columns, no header | Weight matrix (rows = neurons, cols = receptors) |
| `manifest.json` | JSON | Metadata: `num_neurons`, `num_receptors`, file names |

**Important:** `M` (number of receptors) must match the current grid's
receptor count.  If it doesn't, SensoryForge will warn and load the population
with zero-filled receptor coordinates instead of crashing.

Once imported, the population uses the CSV data for all subsequent
visualizations and simulations.  The CSV stub is preserved through
**Generate Population(s)** — regeneration only affects non-CSV populations.

### 1d. Save the configuration

Click **Save As…** in the bottom toolbar to save the grid + population
configuration as a `bundle.json` inside a project folder.

---

## Step 2 — Stimulus Designer

Switch to the **Stimulus Designer** tab.

### 2a. Choose a stimulus type

Click one of the type buttons:

| Button | Shape |
|--------|-------|
| **Gaussian** | Smooth bump — most common |
| **Point** | Ideal point contact |
| **Edge** | Sharp edge or grating |
| **Texture** | Gabor patch or edge grating |
| **Moving** | Any shape with linear / circular / slide motion |

### 2b. Set parameters

Adjust the sliders and spinboxes that appear.  The **live preview** on the
right updates in real time.

- **Amplitude**: peak pressure in mA
- **Sigma**: width of the stimulus in mm (for Gaussian/Gabor)
- **Center X / Y**: position on the skin in mm

### 2c. Build a sequence

1. Set **Duration (ms)** and **dt (ms)**.
2. Click **Add to Stack** to append this stimulus to the timeline.
3. Repeat for each segment.
4. Click **Preview** to animate the stack.

### 2d. Save

Click **Save As…** to save the stimulus stack as `stimulus.json`.

---

## Step 3 — Spiking Neurons

Switch to the **Spiking Neurons** tab.

### 3a. Configure the neuron model

Expand **Model Parameters**.

| Field | Typical value | Notes |
|-------|--------------|-------|
| Neuron model | Izhikevich | Best general-purpose model |
| dt (ms) | 0.1 | Lower = more accurate, slower |
| Device | cpu | Use `cuda` if GPU is available |

### 3b. Choose a filter

Expand **Filter Parameters**.

- **Filter method**: `SA` for slow-adapting, `RA` for rapidly-adapting, or
  `none` to pass drive directly.
- **Input gain**: multiply the drive before the neuron. Useful for scaling.
- **Noise σ**: add Gaussian noise to the drive.

### 3c. Run

Click **▶ Run Simulation**.  The progress bar shows each population.
When finished, a spike raster appears in the panel.

---

## Step 4 — Visualization

Switch to the **Visualization** tab.

### 4a. Choose a layout

Select a preset from the **Layout** dropdown and click **Apply**:

| Preset | Shows |
|--------|-------|
| Default | Stimulus + 2 population heatmaps + raster |
| 2×2 Grid | Stimulus, heatmap, activity, raster |
| Focus + Details | Large raster + stimulus + firing rate |

### 4b. Navigate time

Use the **playback bar** at the bottom:
- ▶ **Play** to animate at the simulation time step
- ◀◀ / ▶▶ to jump to start/end
- Drag the scrubber to any time step

### 4c. Customise panels

Each panel has:
- **⚙** — open the settings sidebar (colormap, axis limits)
- **▾** — change panel type (e.g. swap raster for firing rate)
- **✕** — remove the panel

Panels can be **dragged and dropped** to rearrange, or **floated** into
a separate window by dragging the panel title bar.

---

## Step 5 — Export and batch scaling

Once the interactive simulation looks right:

1. Go to **File → Save Config (YAML)** to write a `config.yml` file.
2. Run the same config from the CLI:

   ```bash
   sensoryforge run config.yml --duration 1000 --output results.h5
   ```

3. For parameter sweeps, use the **Batch** tab:
   - Load a batch YAML (see `examples/batch_config.yml`)
   - Click **▶ Run Batch** for local execution
   - Click **Export SLURM Script…** for cluster submission

---

## Projects

Use **File → New Project…** to create a project directory with the
standard layout:

```
my_experiment/
    config.yaml       ← saved with File → Save Config
    stimuli/          ← stimulus stacks
    results/          ← HDF5 / .pt output files
    figures/          ← exported PNG/SVG figures
```

When a project is open, all save dialogs default to the project directory.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load YAML config |
| `Ctrl+S` | Save YAML config |
| `Ctrl+N` | New project |
| `Ctrl+P` | Open project |
| `Ctrl+Q` | Quit |
| Space | Play / pause (Visualization tab) |
