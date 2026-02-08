# Extended Stimuli

## Overview
Extended stimuli include textures, gratings, and moving contacts for richer tactile patterns.

## Usage

### Texture Example
```python
from sensoryforge.stimuli.texture import gabor_texture
import torch

xx, yy = torch.meshgrid(
    torch.linspace(-1, 1, 64),
    torch.linspace(-1, 1, 64),
    indexing="ij",
)

gabor = gabor_texture(xx, yy, wavelength=0.5, orientation=0.0)
```

### Moving Stimulus Example
```python
from sensoryforge.stimuli.moving import linear_motion, MovingStimulus
from sensoryforge.stimuli.gaussian import gaussian_stimulus
import torch

trajectory = linear_motion((0.0, 0.0), (1.0, 0.0), 50)

def stim_gen(xx, yy, cx, cy):
    return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)

moving = MovingStimulus(trajectory, stim_gen)
```

## Configuration
```yaml
stimuli:
  - type: texture
    kernel_size: 5
    scale: 0.3
  - type: moving
    motion:
      kind: horizontal
      span: 2.0
```

## See Also
- [Composite Grid](composite_grid.md)
- [Equation DSL](equation_dsl.md)
