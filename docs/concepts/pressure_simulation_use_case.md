# The pressure-simulation use case

SensoryForge's purpose is to be a clean-slate simulator of sensor arrays, receptive
fields, sensory neurons and their readouts — general enough to generate data for any
sensor design. **pressure-simulation** (a sibling repository,
`~/Documents/pressure simulation` in a typical checkout) is the first consumer of
that generality: it studies tactile encoding and decoding with a Kalman-filter
readout, and it needs SensoryForge to reproduce its exact stimulus ensemble and
receptive-field design so the two repositories' results are comparable.

This page is the spine end to end: what pressure-simulation supplies, what
SensoryForge supplies, and how the two meet at the bundle. It ties together three
things documented elsewhere in more depth:
[Receptive Fields](../user_guide/receptive_fields.md) (Phase 2, Wave I),
[Data Bundles](../user_guide/bundles.md) (Phase 2, Wave J), and
[Presets](../user_guide/presets.md) (Phase 2, Wave K).

## What pressure-simulation supplies

pressure-simulation's contribution to this recipe is a *design*, not a simulator:

- **A resolvable distance `d`** (millimetres) — the spatial resolution the
  receptive-field design should achieve. `d = 0.40 mm` is the value used throughout
  this recipe and in the `tactile_sa1_ra1` preset.
- **A stimulus ensemble** — four canonical tactile stimuli it uses to probe the
  encoder: a ramping Gaussian blob, a moving oriented edge, a sliding braille
  letter, and a drifting grating. See [Extended Stimuli](../user_guide/extended_stimuli.md)
  for the general stimulus system and `sensoryforge/stimuli/tactile.py` for these
  four specifically.
- **Mutual-information scoring** — pressure-simulation's own decoding pipeline
  scores how much information a population's spikes carry about the stimulus. That
  scoring is out of scope for SensoryForge; SensoryForge's job ends at producing the
  spikes (or the bundle that carries them) for pressure-simulation to score.

## What SensoryForge supplies

Given `d` and the stimulus ensemble, SensoryForge builds the rest of the forward
model:

1. **The grid** — an 80×80 receptor lattice at 0.15 mm spacing (`ReceptorGrid`,
   `sensoryforge/core/grid.py`), matching pressure-simulation's own
   `GridManager(grid_size=80, spacing=0.15)`. Fact K-a (verified 2026-09-16): both
   repositories build coordinate meshgrids with
   `torch.meshgrid(x, y, indexing="ij")`, so frame element `[i, j]` is at
   `(x[i], y[j])` in both — there is no axis transpose to undo when comparing them.
2. **Receptive fields** — the `template` builder (`sensoryforge/core/rf_builders/
   template.py`, D-020) derives `sigma = d / pi`, `pitch = d`, and a neuron count
   from `d` alone: at `d = 0.40 mm` on this 80×80/0.15 mm grid, that is
   `sigma ≈ 0.1273 mm` and **900 neurons** per population (a 30×30 lattice at 0.40 mm
   pitch). See [Receptive Fields](../user_guide/receptive_fields.md) for the full
   `d → sigma, pitch, N` chain and the designed-versus-stochastic distinction (D-019).
3. **Filters and neurons** — an SA population (`sa` filter, regular-spiking
   Izhikevich preset) and an RA population (`ra` filter, fast-spiking preset,
   `k3 = 2.0`, D-Q1), both resolver-owned defaults (`sensoryforge/config/
   defaults.py`) so the GUI, CLI and this recipe agree.
4. **The stimulus ensemble** — `sensoryforge/stimuli/tactile.py`'s four classes are
   transcriptions (not reimplementations) of pressure-simulation's own stimulus
   generators, checked bit-for-bit against a fixture exported directly from that
   repository (`tests/integration/test_stimulus_parity.py`,
   `scripts/regenerate_stimulus_golden.py`).
5. **The bundle** — `SimulationEngine.run(..., bundle_dir=...)` writes a Wave J
   bundle pressure-simulation's own viewer logic can open (a superset of its
   `1.0.0` "mechanoreceptor bundle" format). See [Data Bundles](../user_guide/bundles.md).

## Running the recipe

The whole chain — preset, grid, receptive fields, all four stimuli, four bundles —
is one script:

```bash
python examples/pressure_simulation_recipe.py           # the real recipe (~10s on CPU)
python examples/pressure_simulation_recipe.py --quick    # shortened durations (~2s)
```

or from the CLI directly, one stimulus at a time:

```bash
sensoryforge list-presets
sensoryforge run --preset tactile_sa1_ra1 --duration 1100 --bundle out/ramp_gaussian
```

Both paths use the `tactile_sa1_ra1` preset (`sensoryforge/presets/
tactile_sa1_ra1.yml`) as the single source of truth for the grid and population
configuration, so a config loaded from the preset and a config loaded from the
written bundle's `config.json` describe the same run.

## What is deliberately out of scope here

- **Real receptor sampling for non-grid arrangements** (hex, Poisson, jittered,
  blue-noise) and **composite grids in the engine** are Wave L, not this recipe —
  the pressure-simulation recipe uses a plain regular grid throughout.
- **Mutual-information scoring, decoding, and the Kalman-filter readout** live in
  pressure-simulation itself; SensoryForge's contribution ends at the bundle.
