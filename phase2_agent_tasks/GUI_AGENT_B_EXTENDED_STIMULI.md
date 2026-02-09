# Agent B: Extended Stimuli in StimulusDesignerTab

**Priority:** Phase A (parallel)  
**Files to modify:** `sensoryforge/gui/tabs/stimulus_tab.py`  
**Estimated complexity:** Large  
**Dependencies:** None — fully independent of Agents A, C, D

---

## Pre-requisites

Read these documents first:
1. `phase2_agent_tasks/SHARED_CONTEXT.md` — coding standards, project overview
2. `.github/copilot-instructions.md` — full coding conventions, docstring style, commit format
3. `reviews/GUI_PHASE2_INTEGRATION_PLAN.md` — overall plan (your section is "Agent B")

---

## Context

The `StimulusDesignerTab` currently supports only 3 stimulus types: **Gaussian**, 
**Point**, and **Edge** — all using legacy functions from `sensoryforge.stimuli.stimulus`.

Phase 2 added three new stimulus modules:
- `sensoryforge.stimuli.texture` — Gabor textures, edge gratings, noise patterns
- `sensoryforge.stimuli.moving` — Linear/circular motion, tap sequences, slide trajectories
- `sensoryforge.stimuli.gaussian` — New modular Gaussian API (compatible with legacy)

These modules are fully implemented and tested but **not exposed in the GUI**.

---

## Phase 2 Stimulus APIs (read-only reference)

### Texture Module (`sensoryforge.stimuli.texture`)

```python
# Functional API
from sensoryforge.stimuli.texture import gabor_texture, edge_grating, noise_texture

# gabor_texture(xx, yy, wavelength, orientation, sigma, phase, amplitude) → Tensor
stimulus = gabor_texture(xx, yy, wavelength=0.5, orientation=0.0, sigma=0.3)

# edge_grating(xx, yy, orientation, spacing, count, edge_width, amplitude) → Tensor
stimulus = edge_grating(xx, yy, orientation=0.0, spacing=0.6, count=5)

# noise_texture(shape, noise_type, scale, seed) → Tensor
#   noise_type: "white", "pink", "perlin"
stimulus = noise_texture((64, 64), noise_type="perlin", scale=10)

# nn.Module API
from sensoryforge.stimuli.texture import GaborTexture, EdgeGrating

gabor = GaborTexture(wavelength=0.5, orientation=0.0, sigma=0.3)
result = gabor(xx, yy)  # → Tensor [grid_h, grid_w]
```

### Moving Module (`sensoryforge.stimuli.moving`)

```python
from sensoryforge.stimuli.moving import (
    linear_motion, circular_motion, custom_path_motion,
    MovingStimulus, tap_sequence, slide_trajectory,
)

# Path generators → Tensor [num_steps, 2]
path = linear_motion(start=(0, 0), end=(2, 0), num_steps=100)
path = circular_motion(center=(0, 0), radius=1.0, num_steps=100)

# MovingStimulus wraps a spatial generator + path → [num_steps, grid_h, grid_w]
def stim_gen(xx, yy, cx, cy):
    return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 0.3**2))

moving = MovingStimulus(stimulus_fn=stim_gen, path=path)
frames = moving(xx, yy)  # → [num_steps, grid_h, grid_w]

# Convenience: tap_sequence, slide_trajectory
frames = tap_sequence(xx, yy, stim_gen, positions=[(0,0), (1,1)], duration=50)
frames = slide_trajectory(xx, yy, stim_gen, start=(-2,0), end=(2,0), num_steps=100)
```

### Gaussian Module (`sensoryforge.stimuli.gaussian`)

```python
from sensoryforge.stimuli.gaussian import gaussian_stimulus, GaussianStimulus

# Functional
stim = gaussian_stimulus(xx, yy, center=(0, 0), sigma=0.5, amplitude=1.0)

# nn.Module
gs = GaussianStimulus(center=(0, 0), sigma=0.5, amplitude=1.0)
result = gs(xx, yy)  # → Tensor [grid_h, grid_w]
```

---

## default_params.json Phase 2 Stimulus Defaults (already exists, unused)

```json
{
  "phase2": {
    "stimulus_types": ["gaussian", "trapezoidal", "texture", "moving", "step", "ramp"],
    "extended_stimuli": {
      "texture": {
        "gabor": {"wavelength": 0.5, "orientation": 0.0, "sigma": 0.3, "amplitude": 1.0},
        "edge_grating": {"orientation": 0.0, "spacing": 0.6, "count": 5, "edge_width": 0.05, "amplitude": 1.0}
      },
      "moving": {
        "linear": {"start": [0.0, 0.0], "end": [2.0, 0.0], "num_steps": 100},
        "circular": {"center": [0.0, 0.0], "radius": 1.0, "num_steps": 100}
      }
    }
  }
}
```

---

## Current StimulusDesignerTab Architecture

```
StimulusDesignerTab(QWidget)
├── Constructor: __init__(self, mechanoreceptor_tab)
├── Uses grid from mechanoreceptor_tab for coordinate meshes (xx, yy)
├── Stimulus type buttons: Gaussian | Point | Edge (QButtonGroup)
├── StimulusConfig dataclass: shape, type, amplitude, position, spread, etc.
├── Motion mode: Static | Moving (with velocity, angle params)
├── Temporal controls: ramp_up, plateau, ramp_down (ms)
├── Animation: frame_slider, play/pause, timer-based frame stepping
├── Preview canvas: matplotlib FigureCanvas showing 2D heatmap
├── Library: save/load JSON stimulus configs to/from project directory
└── Key methods:
    ├── _generate_static_frame() → single-frame stimulus Tensor
    ├── _generate_all_frames() → [timesteps, grid_h, grid_w]
    ├── _update_preview() → renders current frame on canvas
    ├── _on_stimulus_type_changed() → swaps parameter controls
    ├── _play_animation() / _stop_animation()
    └── _save_stimulus() / _load_stimulus()
```

---

## What To Do

### B.1: Add Texture Stimulus Type

1. **Add "Texture" to stimulus type button group** alongside Gaussian/Point/Edge.
   - Use the same `QButtonGroup` pattern as existing types.

2. **Create texture sub-type selector** (visible when Texture is selected):
   - `QComboBox` with: Gabor, Edge Grating, Noise
   - Connect to `_on_texture_subtype_changed()`

3. **Add parameter controls per texture sub-type**:
   - **Gabor**: wavelength (0.1–10.0), orientation (0–π), sigma (0.05–5.0), 
     phase (0–2π), amplitude (0.1–100.0)
   - **Edge Grating**: orientation (0–π), spacing (0.1–5.0), count (1–20), 
     edge_width (0.01–1.0), amplitude (0.1–100.0)
   - **Noise**: noise_type (white/pink/perlin), scale (1–100)
   - Load defaults from `default_params.json > phase2 > extended_stimuli > texture`

4. **Implement texture frame generation** in `_generate_static_frame()`:
   ```python
   if self.stimulus_type == "texture":
       if self.texture_subtype == "gabor":
           from sensoryforge.stimuli.texture import gabor_texture
           return gabor_texture(xx, yy, wavelength=..., orientation=..., ...)
       elif self.texture_subtype == "edge_grating":
           from sensoryforge.stimuli.texture import edge_grating
           return edge_grating(xx, yy, orientation=..., spacing=..., ...)
       elif self.texture_subtype == "noise":
           from sensoryforge.stimuli.texture import noise_texture
           return noise_texture(xx.shape, noise_type=..., scale=...)
   ```

5. **Update preview** to work with texture outputs.

### B.2: Add Moving Stimulus Type

1. **Add "Moving" to stimulus type button group**.

2. **Create motion sub-type selector** (visible when Moving is selected):
   - `QComboBox` with: Linear, Circular, Tap Sequence, Slide

3. **Add parameter controls per motion sub-type**:
   - **Linear**: start_x, start_y, end_x, end_y, num_steps
   - **Circular**: center_x, center_y, radius, num_steps
   - **Tap Sequence**: positions (editable table), duration per tap
   - **Slide**: start_x, start_y, end_x, end_y, num_steps
   - Load defaults from `default_params.json > phase2 > extended_stimuli > moving`

4. **Add base stimulus selector for moving type**: Since `MovingStimulus` wraps
   a spatial generator, let the user choose what shape moves: Gaussian blob or 
   custom (use `gaussian_stimulus` as default spatial function).

5. **Implement moving frame generation** in `_generate_all_frames()`:
   ```python
   if self.stimulus_type == "moving":
       from sensoryforge.stimuli.moving import (
           linear_motion, circular_motion, MovingStimulus,
           tap_sequence, slide_trajectory,
       )
       from sensoryforge.stimuli.gaussian import gaussian_stimulus
       
       def spatial_fn(xx, yy, cx, cy):
           return gaussian_stimulus(xx, yy, center=(cx, cy), sigma=self.sigma)
       
       if self.motion_subtype == "linear":
           path = linear_motion(start=..., end=..., num_steps=...)
           ms = MovingStimulus(stimulus_fn=spatial_fn, path=path)
           return ms(xx, yy)
       # ... etc
   ```

6. **Update animation system** to handle variable-length frame sequences from
   moving stimuli. The slider range should adjust to `num_steps`.

### B.3: Integrate New Gaussian Module (optional, low priority)

- Optionally update the existing Gaussian type to use `sensoryforge.stimuli.gaussian`
  instead of legacy functions, ensuring backward compatibility.
- This is nice-to-have, not required.

### B.4: Update Save/Load

- Extend `StimulusConfig` dataclass (or create new fields) to store texture/moving
  params.
- Update `_save_stimulus()` and `_load_stimulus()` to persist new stimulus types.

---

## Key Constraints

- **DO NOT change existing Gaussian/Point/Edge behavior** — they must work as before.
- **Keep the same UI pattern**: button group for type, dynamic parameter panel swap.
- **Animation system must work** for both static (texture) and dynamic (moving) stimuli.
- **Use imports from Phase 2 modules** — do not re-implement stimulus generation.
- **Read defaults from `default_params.json`** `phase2.extended_stimuli` section.

---

## Verification

- Launch GUI → Stimulus tab shows 5 buttons: Gaussian, Point, Edge, Texture, Moving
- Select Texture > Gabor → param controls appear, preview shows Gabor pattern
- Select Texture > Edge Grating → param controls swap, preview shows gratings
- Select Moving > Linear → preview animates a moving Gaussian blob
- Select Moving > Circular → preview animates circular path
- Play/pause animation works for moving stimuli
- Save/load preserves texture and moving configs
- Existing Gaussian/Point/Edge still work unchanged

---

## Commit Format

```
feat(gui): add texture and moving stimulus types to StimulusDesignerTab
```

## Tests

Add a minimal test in `tests/unit/test_gui_agent_b.py` that:
- Imports stimulus modules without error
- Verifies `gabor_texture`, `edge_grating`, `noise_texture` produce expected shapes
- Verifies `linear_motion`, `circular_motion` produce expected path shapes
- Verifies `MovingStimulus` produces expected frame tensor shapes

(Full GUI integration tests will be added in Phase B.)
