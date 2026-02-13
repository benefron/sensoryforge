# Extended Stimuli

## Overview

Extended stimuli include textures (Gabor, gratings, noise), moving contacts, and temporal sequences. They provide richer tactile patterns for encoding experiments.

## Texture Stimuli

### Gabor Texture

Localized sinusoidal pattern — useful for orientation and spatial frequency sensitivity.

```python
import torch
from sensoryforge.stimuli.texture import gabor_texture

xx, yy = torch.meshgrid(
    torch.linspace(-1, 1, 64),
    torch.linspace(-1, 1, 64),
    indexing="ij",
)

gabor = gabor_texture(
    xx, yy,
    wavelength=0.5,
    orientation=0.0,
    sigma=0.3,
    amplitude=1.0,
)
# gabor.shape: [64, 64]
```

### Edge Grating

Periodic edge pattern for texture discrimination.

```python
from sensoryforge.stimuli.texture import edge_grating

grating = edge_grating(
    xx, yy,
    orientation=0.0,
    spacing=0.6,
    count=5,
)
```

### Noise Texture

Spatially correlated random pattern (height × width in pixels).

```python
from sensoryforge.stimuli.texture import noise_texture

noise = noise_texture(64, 64, scale=0.3, kernel_size=5, seed=42)
# noise.shape: [64, 64]
```

## Moving Stimuli

### Linear Motion

A stimulus that moves along a straight path.

```python
import torch
from sensoryforge.stimuli.moving import linear_motion, MovingStimulus
from sensoryforge.stimuli.gaussian import gaussian_stimulus

# Create trajectory: 50 steps from (0,0) to (1,0)
trajectory = linear_motion((0.0, 0.0), (1.0, 0.0), 50)

# Generator: Gaussian bump at (cx, cy)
def gauss_gen(xx, yy, cx, cy):
    return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)

moving = MovingStimulus(trajectory, gauss_gen)

# Generate on grid
xx, yy = torch.meshgrid(
    torch.linspace(-2, 2, 32),
    torch.linspace(-2, 2, 32),
    indexing="ij",
)
temporal_stimulus = moving(xx, yy)
# temporal_stimulus.shape: [50, 32, 32]
```

### Circular Motion

Stimulus moving along a circular path.

```python
import math
from sensoryforge.stimuli.moving import circular_motion

# Quarter circle, 50 steps
trajectory = circular_motion(
    center=(0.0, 0.0),
    radius=0.5,
    num_steps=50,
    start_angle=0.0,
    end_angle=math.pi / 2,
)
# trajectory.shape: [50, 2]
```

### Tap Sequence

Repeated taps at a fixed position. Requires a stimulus generator and coordinate grid.

```python
from sensoryforge.stimuli.moving import tap_sequence
from sensoryforge.stimuli.gaussian import gaussian_stimulus

def gauss_gen(xx, yy, cx, cy):
    return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)

xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
taps = tap_sequence(
    position=(0.5, 0.5),
    num_taps=5,
    tap_duration=20,
    interval_duration=10,
    stimulus_generator=gauss_gen,
    xx=xx,
    yy=yy,
)
# taps.shape: [total_steps, 32, 32]
```

## Pipeline Integration

Use stimulus types in YAML configuration:

```yaml
stimuli:
  - type: gaussian
    center: [0.0, 0.0]
    sigma: 0.3
    amplitude: 1.0
  - type: texture
    kernel: gabor
    wavelength: 0.5
    orientation: 0.0
  - type: moving
    motion:
      kind: linear
      start: [0.0, 0.0]
      end: [1.0, 0.0]
      steps: 100
```

## See Also

- [Composite Grid](composite_grid.md)
- [Equation DSL](equation_dsl.md)
- [Batch Processing](batch_processing.md) — parameter sweeps over stimuli
