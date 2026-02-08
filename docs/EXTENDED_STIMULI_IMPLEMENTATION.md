# Extended Stimuli Implementation Documentation

**Feature:** Gaussian, Texture, and Moving Stimulus Modules  
**Branch Merged:** `copilot/update-task-docs-completion`  
**Commit:** feat: Add extended stimulus modules (Gaussian, texture, moving)  
**Date:** February 8, 2026

## Overview

The Extended Stimuli package provides three specialized stimulus generation modules for tactile and visual sensory encoding experiments. These modules go beyond basic stimuli to enable realistic, scientifically grounded sensory experiments including:

- **Gaussian stimuli:** Smooth, localized pressure bumps (touch/vision)
- **Texture stimuli:** Periodic gratings, checkerboards, and noise patterns
- **Moving stimuli:** Temporal sequences including taps, slides, and strokes

All modules are PyTorch-native, GPU-compatible, and fully differentiable for end-to-end learning.

## Location in Codebase

### Main Implementation
- **Package:** `sensoryforge/stimuli/`
- **Files:**
  - `gaussian.py` (316 lines) - Gaussian bump generation
  - `texture.py` (419 lines) - Texture pattern generation
  - `moving.py` (519 lines) - Temporal stimulus sequences
  - `__init__.py` (updated) - Package exports

### Test Suite
- **Files:**
  - `tests/unit/test_gaussian_stimulus.py` (356 lines) - 24 tests
  - `tests/unit/test_texture_stimulus.py` (438 lines) - 31 tests
  - `tests/unit/test_moving_stimulus.py` (441 lines) - 31 tests
- **Total Coverage:** 86 tests, 100% pass rate
- **Runtime:** ~0.15 seconds

## Architecture

### Design Philosophy

The stimulus modules follow these principles:

1. **Functional API:** Pure functions that generate tensors (no stateful classes)
2. **Composability:** Stimuli can be combined and superposed
3. **Device-agnostic:** Automatic handling of CPU/CUDA/MPS
4. **Batch-ready:** All functions support batch dimensions
5. **Physical units:** Consistent use of mm for distance, ms for time

### Module Structure

```
sensoryforge/stimuli/
├── gaussian.py
│   ├── gaussian_stimulus()          # Single Gaussian bump
│   ├── multi_gaussian_stimulus()     # Multiple Gaussians superposed
│   └── gaussian_stimulus_batch()     # Batched Gaussian generation
├── texture.py
│   ├── grating_stimulus()            # Sinusoidal grating
│   ├── square_wave_grating()         # Square wave grating
│   ├── checkerboard_stimulus()       # 2D checkerboard pattern
│   ├── noise_stimulus()              # White/pink/brown noise
│   └── gabor_stimulus()              # Oriented Gabor patch (future)
└── moving.py
    ├── tap_stimulus()                # Static tap with temporal profile
    ├── tap_sequence()                # Multiple taps at locations
    ├── slide_trajectory()            # Linear sliding motion
    ├── moving_gaussian()             # Gaussian moving along path
    └── stroke_stimulus()             # Simulated brush stroke (future)
```

## Module 1: Gaussian Stimuli

**File:** `sensoryforge/stimuli/gaussian.py`

### Purpose

Generate smooth, localized pressure patterns modeled as 2D Gaussian functions. These are the fundamental building blocks for tactile and visual experiments.

### Core Functions

#### 1. `gaussian_stimulus()`

**Signature:**
```python
def gaussian_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    amplitude: float = 1.0,
    sigma: float = 0.2,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Mathematical Definition:**

$$
G(x, y) = A \cdot \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2}\right)
$$

where:
- $A$ = amplitude (peak height)
- $(x_0, y_0)$ = center coordinates
- $\sigma$ = standard deviation (spatial spread)

**Parameters:**
- `xx, yy`: Meshgrid coordinate tensors [H, W] in mm
- `center_x, center_y`: Gaussian center in mm
- `amplitude`: Peak amplitude (arbitrary units, typically pressure in mA)
- `sigma`: Standard deviation in mm (controls spread)
- `device`: Target PyTorch device (auto-detects from xx if None)

**Returns:**
- Gaussian stimulus [H, W] with same shape as coordinate grids

**Example:**
```python
import torch
from sensoryforge.stimuli import gaussian_stimulus

# Create spatial grid (10mm x 10mm, 64x64 resolution)
xx, yy = torch.meshgrid(
    torch.linspace(-5, 5, 64),
    torch.linspace(-5, 5, 64),
    indexing='ij'
)

# Generate Gaussian centered at origin, sigma=1mm, amplitude=2.0
stim = gaussian_stimulus(xx, yy, center_x=0.0, center_y=0.0, 
                         amplitude=2.0, sigma=1.0)

print(stim.shape)      # torch.Size([64, 64])
print(stim.max())      # ~2.0 (peak amplitude)
print(stim[32, 32])    # ~2.0 (center value)
```

#### 2. `multi_gaussian_stimulus()`

**Purpose:** Superpose multiple Gaussian bumps to create complex patterns

**Signature:**
```python
def multi_gaussian_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    centers: List[Tuple[float, float]],
    amplitudes: Optional[List[float]] = None,
    sigmas: Optional[List[float]] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Algorithm:**
$$
S(x, y) = \sum_{i=1}^{N} A_i \cdot \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma_i^2}\right)
$$

**Example:**
```python
# Create three-dot Braille pattern
centers = [(0, 0), (0, 2), (0, 4)]  # Vertical line, 2mm spacing
amplitudes = [1.0, 1.0, 1.0]
sigmas = [0.3, 0.3, 0.3]

stim = multi_gaussian_stimulus(xx, yy, centers, amplitudes, sigmas)
# Result: Three distinct bumps with overlapping tails
```

#### 3. `gaussian_stimulus_batch()`

**Purpose:** Generate multiple Gaussians in batched form (efficient)

**Signature:**
```python
def gaussian_stimulus_batch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    centers: torch.Tensor,  # [batch, 2]
    amplitudes: torch.Tensor,  # [batch]
    sigmas: torch.Tensor,  # [batch]
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor  # [batch, H, W]
```

**Use Case:** Generate many stimuli at once for ML training

**Example:**
```python
# Generate 32 random Gaussian stimuli
batch_size = 32
centers = torch.rand(batch_size, 2) * 10 - 5  # Random centers in [-5, 5]
amplitudes = torch.ones(batch_size)
sigmas = torch.rand(batch_size) * 0.5 + 0.2  # Sigmas in [0.2, 0.7]

batch_stim = gaussian_stimulus_batch(xx, yy, centers, amplitudes, sigmas)
print(batch_stim.shape)  # torch.Size([32, 64, 64])
```

### Use Cases

1. **Tactile indentation:** Simulating fingertip pressing on skin
2. **Receptive field probing:** Testing neuron spatial sensitivity
3. **Spatial localization experiments:** Finding detection thresholds
4. **Braille reading:** Multi-dot patterns for texture recognition

## Module 2: Texture Stimuli

**File:** `sensoryforge/stimuli/texture.py`

### Purpose

Generate periodic and stochastic spatial patterns for texture perception experiments. Includes gratings (visual orientation selectivity), checkerboards (spatial frequency tuning), and noise (statistical texture).

### Core Functions

#### 1. `grating_stimulus()`

**Purpose:** Sinusoidal grating for orientation/frequency tuning

**Signature:**
```python
def grating_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    orientation: float = 0.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    amplitude: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Mathematical Definition:**

$$
G(x, y) = A \cdot \sin(2\pi f (x \cos\theta + y \sin\theta) + \phi)
$$

where:
- $f$ = spatial frequency (cycles per mm)
- $\theta$ = orientation (radians)
- $\phi$ = phase offset (radians)
- $A$ = amplitude

**Example:**
```python
from sensoryforge.stimuli import grating_stimulus

# Vertical grating (orientation = 0°)
vert_grating = grating_stimulus(xx, yy, orientation=0.0, frequency=2.0)

# Horizontal grating (orientation = 90° = π/2)
horiz_grating = grating_stimulus(xx, yy, orientation=torch.pi/2, frequency=2.0)

# Diagonal grating (45°)
diag_grating = grating_stimulus(xx, yy, orientation=torch.pi/4, frequency=1.5)
```

**Use Cases:**
- Visual cortex orientation tuning experiments
- Tactile grating orientation discrimination
- Spatial frequency threshold measurements

#### 2. `square_wave_grating()`

**Purpose:** Binary grating for high-contrast edge detection

**Signature:**
```python
def square_wave_grating(
    xx: torch.Tensor,
    yy: torch.Tensor,
    orientation: float = 0.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    amplitude: float = 1.0,
    threshold: float = 0.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Algorithm:**
- Compute sinusoidal grating
- Apply threshold: `sign(sin(...) - threshold)`
- Results in binary [-amplitude, +amplitude] pattern

**Example:**
```python
# Square wave with 1 cycle per mm
sq_grating = square_wave_grating(xx, yy, frequency=1.0, amplitude=1.0)
# Values are either -1.0 or +1.0 (binary)
```

#### 3. `checkerboard_stimulus()`

**Purpose:** 2D checkerboard for spatial frequency tuning

**Signature:**
```python
def checkerboard_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    square_size: float = 1.0,
    amplitude: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Algorithm:**
- Divide space into squares of size `square_size`
- Alternate polarity in checkerboard pattern
- Uses modulo arithmetic for efficiency

**Example:**
```python
# 1mm x 1mm checkerboard
checker = checkerboard_stimulus(xx, yy, square_size=1.0, amplitude=1.0)

# Fine checkerboard (0.5mm squares)
fine_checker = checkerboard_stimulus(xx, yy, square_size=0.5)
```

#### 4. `noise_stimulus()`

**Purpose:** Stochastic texture for statistical pattern analysis

**Signature:**
```python
def noise_stimulus(
    shape: Tuple[int, int],
    noise_type: str = 'white',
    amplitude: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor
```

**Noise Types:**
- **'white':** Uncorrelated Gaussian noise at each pixel
- **'pink':** 1/f power spectrum (naturalistic)
- **'brown':** 1/f² power spectrum (random walk)
- **'binary':** Random binary pattern (±amplitude)

**Example:**
```python
# White noise texture
white = noise_stimulus((64, 64), noise_type='white', amplitude=0.5)

# Pink noise (more naturalistic)
pink = noise_stimulus((64, 64), noise_type='pink', amplitude=0.5)

# Binary random dots
binary = noise_stimulus((64, 64), noise_type='binary', amplitude=1.0)
```

**Use Cases:**
- Texture discrimination experiments
- Reverse correlation (spike-triggered averaging)
- Statistical learning in sensory systems

## Module 3: Moving Stimuli

**File:** `sensoryforge/stimuli/moving.py`

### Purpose

Generate temporal sequences of stimuli to study motion processing, temporal integration, and dynamic touch/vision. Includes taps (transient events), slides (continuous motion), and complex trajectories.

### Core Functions

#### 1. `tap_stimulus()`

**Purpose:** Single localized tap with temporal profile

**Signature:**
```python
def tap_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    duration_ms: float = 50.0,
    rise_time_ms: float = 10.0,
    fall_time_ms: float = 20.0,
    amplitude: float = 1.0,
    sigma: float = 0.5,
    dt_ms: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor  # [time_steps, H, W]
```

**Temporal Profile:**
- Rise phase: Linear ramp from 0 to amplitude over `rise_time_ms`
- Hold phase: Constant amplitude
- Fall phase: Exponential decay over `fall_time_ms`

**Example:**
```python
from sensoryforge.stimuli import tap_stimulus

# Single tap at origin, 50ms duration
tap = tap_stimulus(
    xx, yy,
    center_x=0.0,
    center_y=0.0,
    duration_ms=50.0,
    rise_time_ms=5.0,
    fall_time_ms=15.0,
    amplitude=2.0,
    sigma=0.5,
    dt_ms=1.0
)

print(tap.shape)  # torch.Size([50, 64, 64]) — 50 time frames
```

**Use Cases:**
- Braille reading simulation
- Transient touch detection
- Rapidly adapting (RA) mechanoreceptor activation

#### 2. `tap_sequence()`

**Purpose:** Multiple taps at different locations and times

**Signature:**
```python
def tap_sequence(
    xx: torch.Tensor,
    yy: torch.Tensor,
    tap_locations: List[Tuple[float, float]],
    tap_times_ms: List[float],
    tap_durations_ms: Optional[List[float]] = None,
    amplitude: float = 1.0,
    sigma: float = 0.5,
    total_duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor  # [time_steps, H, W]
```

**Example:**
```python
# Three sequential taps forming a pattern
locations = [(0, 0), (0, 2), (0, 4)]
times = [10, 30, 50]  # Taps at 10ms, 30ms, 50ms
durations = [20, 20, 20]  # Each tap lasts 20ms

sequence = tap_sequence(
    xx, yy,
    tap_locations=locations,
    tap_times_ms=times,
    tap_durations_ms=durations,
    total_duration_ms=100.0,
    dt_ms=1.0
)

print(sequence.shape)  # torch.Size([100, 64, 64])
```

**Use Cases:**
- Braille character presentation
- Temporal pattern recognition
- Sensory memory experiments

#### 3. `slide_trajectory()`

**Purpose:** Continuous linear motion along a path

**Signature:**
```python
def slide_trajectory(
    xx: torch.Tensor,
    yy: torch.Tensor,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    velocity_mm_per_s: float = 10.0,
    amplitude: float = 1.0,
    sigma: float = 0.5,
    dt_ms: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor  # [time_steps, H, W]
```

**Algorithm:**
- Compute total distance: $d = \sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}$
- Compute duration: $T = d / v$
- Generate frames: position at time $t$ is $(x_0, y_0) + (v \cdot t) \cdot \hat{d}$

**Example:**
```python
from sensoryforge.stimuli import slide_trajectory

# Slide from left to right, 10 mm/s velocity
slide = slide_trajectory(
    xx, yy,
    start_pos=(-4.0, 0.0),
    end_pos=(4.0, 0.0),
    velocity_mm_per_s=10.0,
    amplitude=1.5,
    sigma=0.5,
    dt_ms=1.0
)

print(slide.shape)  # torch.Size([800, 64, 64]) — 800ms to traverse 8mm at 10mm/s
```

**Use Cases:**
- Texture scanning (running finger over surface)
- Motion direction selectivity
- Velocity tuning experiments

#### 4. `moving_gaussian()`

**Purpose:** Gaussian moving along arbitrary trajectory

**Signature:**
```python
def moving_gaussian(
    xx: torch.Tensor,
    yy: torch.Tensor,
    trajectory: torch.Tensor,  # [time_steps, 2]
    amplitude_profile: Optional[torch.Tensor] = None,  # [time_steps]
    sigma: float = 0.5,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor  # [time_steps, H, W]
```

**Example:**
```python
# Circular motion
t = torch.linspace(0, 2*torch.pi, 100)
radius = 3.0
trajectory = torch.stack([
    radius * torch.cos(t),
    radius * torch.sin(t)
], dim=1)  # [100, 2]

# Modulated amplitude (fade in/out)
amplitude_profile = torch.sin(torch.linspace(0, torch.pi, 100))

circular_motion = moving_gaussian(
    xx, yy,
    trajectory=trajectory,
    amplitude_profile=amplitude_profile,
    sigma=0.5
)

print(circular_motion.shape)  # torch.Size([100, 64, 64])
```

**Use Cases:**
- Complex motion patterns (cursive writing simulation)
- Predictive coding experiments
- Smooth pursuit eye movement analogs

## Implementation Details

### Device Handling

All functions automatically handle device placement:

```python
# Coordinates on CPU
xx_cpu, yy_cpu = torch.meshgrid(...)

# Generate on CUDA
stim = gaussian_stimulus(xx_cpu, yy_cpu, ..., device='cuda')
print(stim.device)  # cuda:0

# Or use coordinate device
xx_gpu = xx_cpu.to('cuda')
stim = gaussian_stimulus(xx_gpu, yy_cpu, ...)  # Auto-detects cuda from xx_gpu
```

### Batch Compatibility

Most functions support batch dimensions:

```python
# Batch of coordinate grids [batch, H, W]
xx_batch = torch.stack([xx for _ in range(8)])
yy_batch = torch.stack([yy for _ in range(8)])

# Batch of Gaussians with different parameters
batch_stim = gaussian_stimulus_batch(
    xx_batch, yy_batch,
    centers=torch.randn(8, 2),
    amplitudes=torch.ones(8),
    sigmas=torch.ones(8) * 0.5
)

print(batch_stim.shape)  # [8, 64, 64]
```

### Physical Units

**Spatial:**
- Coordinates (xx, yy): millimeters (mm)
- sigma, square_size, distances: millimeters (mm)

**Temporal:**
- Time steps: milliseconds (ms)
- Velocities: millimeters per second (mm/s)
- dt: milliseconds (ms)

**Amplitude:**
- Arbitrary units (context-dependent)
- Tactile: typically pressure or current (mA)
- Visual: typically contrast or luminance

### Performance Optimizations

1. **Vectorization:** All operations use PyTorch tensor operations (no Python loops)
2. **Pre-allocation:** Temporal sequences preallocate output tensors
3. **In-place operations:** Minimal tensor copying
4. **GPU parallelism:** Full CUDA support for all functions

## Testing Coverage

### Test Statistics

- **Total tests:** 86
- **Pass rate:** 100%
- **Runtime:** 0.15 seconds (all tests)
- **Modules tested:**
  - `test_gaussian_stimulus.py`: 24 tests
  - `test_texture_stimulus.py`: 31 tests
  - `test_moving_stimulus.py`: 31 tests

### Test Categories

**Shape and Dimensionality Tests:**
- Output shapes match expected dimensions
- Batch dimensions preserved correctly
- Temporal sequences have correct length

**Numerical Correctness:**
- Gaussian peaks at expected amplitude
- Grating frequency matches specification
- Trajectory endpoints correct

**Error Handling:**
- Invalid parameters raise appropriate errors
- Mismatched shapes detected
- Negative/zero values validated

**Device Compatibility:**
- CPU tensors handled correctly
- Explicit device specification works
- Mixed device inputs raise errors

**Edge Cases:**
- Zero amplitude
- Very small/large sigma
- Single time step temporal sequences

## Usage Examples

### Example 1: Tactile Localization Experiment

```python
import torch
from sensoryforge.stimuli import gaussian_stimulus
from sensoryforge.core import CompositeGrid
from sensoryforge.filters import SAFilterTorch

# Setup
grid = CompositeGrid(xlim=(-5, 5), ylim=(-5, 5))
grid.add_population('SA1', density=100, arrangement='grid')

# Generate stimulus at random location
center_x, center_y = torch.rand(2) * 8 - 4  # Random in [-4, 4]
xx, yy = torch.meshgrid(torch.linspace(-5, 5, 64), torch.linspace(-5, 5, 64), indexing='ij')
stim = gaussian_stimulus(xx, yy, center_x.item(), center_y.item(), 
                         amplitude=2.0, sigma=0.8)

# Pass through SA filter
sa_filter = SAFilterTorch({'num_neurons': grid.get_population_count('SA1')})
response = sa_filter(stim.unsqueeze(0).unsqueeze(0))

# Decode location from response
# ... (population coding analysis)
```

### Example 2: Texture Discrimination

```python
from sensoryforge.stimuli import grating_stimulus, noise_stimulus

# Texture A: Fine grating
texture_a = grating_stimulus(xx, yy, orientation=0, frequency=3.0)

# Texture B: Noise
texture_b = noise_stimulus((64, 64), noise_type='white', amplitude=0.5)

# Train classifier to distinguish textures
# ... (ML pipeline)
```

### Example 3: Motion Direction Selectivity

```python
from sensoryforge.stimuli import slide_trajectory

# Rightward motion
right_slide = slide_trajectory(xx, yy, (-4, 0), (4, 0), velocity_mm_per_s=10)

# Leftward motion
left_slide = slide_trajectory(xx, yy, (4, 0), (-4, 0), velocity_mm_per_s=10)

# Upward motion
up_slide = slide_trajectory(xx, yy, (0, -4), (0, 4), velocity_mm_per_s=10)

# Pass through neuron model, measure preferred direction
# ... (direction tuning curve)
```

## Integration with SensoryForge

### Current Integration

Extended stimuli are **standalone modules** that can be used independently or with the main pipeline:

```python
# Standalone usage
from sensoryforge.stimuli import gaussian_stimulus, slide_trajectory

# Pipeline integration (future)
from sensoryforge.core import TactileEncodingPipeline

pipeline = TactileEncodingPipeline(
    stimulus_generator='gaussian',  # Use Gaussian stimulus
    grid_type='composite',
    # ... other config
)
```

### Future Enhancements

1. **Stimulus class wrappers:**
   - `GaussianStimulus` class for state management
   - `MovingStimulus` base class for trajectory-based stimuli

2. **Parameterized stimulus sets:**
   - Random stimulus generators for ML training
   - Systematic parameter sweeps for tuning curves

3. **Visualization utilities:**
   - Animation helpers for moving stimuli
   - Interactive parameter tuning (Jupyter widgets)

## Conclusion

The Extended Stimuli modules provide SensoryForge with:

- ✅ **Comprehensive stimulus coverage:** Gaussian, textures, motion
- ✅ **Scientifically grounded:** Based on neuroscience experiments
- ✅ **Production-ready:** 86 tests, 100% pass rate
- ✅ **PyTorch-native:** GPU-compatible, differentiable
- ✅ **Well-documented:** Extensive docstrings and examples
- ✅ **Flexible API:** Functional design, easy composition

**Status:** Merged and ready for experimental use.
