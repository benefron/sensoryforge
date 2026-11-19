# Quick Start: Run from the Command Line

In five minutes you run a shipped recipe, change its stimulus with a small file of your
own, and read the result in Python. You need SensoryForge installed
([Installation](installation.md)) and nothing else: no example file, no clone.

## 1. Run a preset

Four complete configs ship in the package:

```bash
sensoryforge list-presets
sensoryforge run --preset tactile_sa1_ra1 --duration 200 --bundle first_run
```

```text
Note: this config has no 'stimulus:' block; running the default trapezoidal stimulus (type='trapezoidal').
Running simulation (duration: 200.0ms)...
Bundle written to first_run

Simulation completed successfully!
SA Population spikes: ...
RA Population spikes: ...
```

`tactile_sa1_ra1` is an 80 × 80 sensor array at 0.15 mm read by an SA and an RA
population ([the recipe](../concepts/pressure_simulation_use_case.md)). The run takes a few
seconds on a laptop CPU; add `--device cuda` or `--device mps` for a GPU.

## 2. Change the stimulus

A config file given after `--preset` is merged over it, so it only has to say what
differs. Save this as `gaussian.yml`:

```yaml
stimulus:
  type: gaussian
  amplitude: 1.0      # a unit peak, the recipe's convention
  sigma: 0.5          # mm
```

```bash
sensoryforge run --preset tactile_sa1_ra1 gaussian.yml --duration 200 --bundle gaussian_run
```

Every stimulus type and its parameters: `sensoryforge describe --section components`. Nested
blocks merge key by key; a list (such as `populations:`) in your file replaces the
preset's whole list.

## 3. Read the result

The bundle folder holds the config, the stimulus and every population's spikes:

```python
from sensoryforge.io.bundle import load_bundle

b = load_bundle("gaussian_run")
spikes = b.populations["SA Population"]["spikes"]   # [time, neurons] spike counts
print(spikes.shape, int(spikes.sum()), b.meta["dt_ms"])
print(b.stimulus.shape)                              # [time, 80, 80]
```

```text
torch.Size([200, 900]) ... 1.0
torch.Size([200, 80, 80])
```

`data.h5` in the same folder opens with `h5py` alone, without SensoryForge
([Data Bundles](../user_guide/bundles.md)).

## Next

- Write a whole config of your own: the
  [Configuration Schema](../user_guide/configuration_schema.md), or print it with
  `sensoryforge describe --schema config`.
- Sweep a parameter: `sensoryforge batch` ([Batch Processing](../tutorials/batch_processing_tutorial.md)).
- Every command and flag: [CLI](../user_guide/cli.md), or `sensoryforge <command> --help`.
