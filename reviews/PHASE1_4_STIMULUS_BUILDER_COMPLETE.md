# Phase 1.4 Complete: Composable Stimulus Builder API

**Status:** ✅ Complete  
**Date:** February 9, 2025  
**Author:** Phase 2 Architecture Remediation

## Overview

Phase 1.4 implements a fluent builder API for creating composable stimuli with optional motion patterns and multi-stimulus composition. This API makes stimulus design intuitive and expressive while maintaining full backward compatibility with existing functional APIs.

## Implementation Summary

### New Files Created

1. **sensoryforge/stimuli/builder.py** (739 lines)
   - `StaticStimulus`: Wraps functional stimulus generation APIs
   - `MovingStimulus`: Adds temporal motion trajectories to any stimulus
   - `CompositeStimulus`: Combines multiple stimuli with configurable modes
   - `Stimulus`: Fluent builder class with static factory methods
   - `with_motion()`: Functional API for adding motion

2. **tests/unit/test_stimulus_builder.py** (530 lines)
   - 32 comprehensive tests covering all builder features
   - Tests for static, moving, and composite stimuli
   - Serialization and deserialization tests
   - Backward compatibility verification

### Modified Files

1. **sensoryforge/stimuli/__init__.py**
   - Added builder API exports
   - Updated module docstring with recommended usage
   - Exported: `Stimulus`, `StaticStimulus`, `MovingStimulus`, `CompositeStimulus`, `with_motion`

## Architecture

### Class Hierarchy

```
BaseStimulus (ABC)
├── StaticStimulus
│   ├── Wraps: gaussian_stimulus(), gabor_texture(), edge_grating(), etc.
│   └── Factory methods: Stimulus.gaussian(), .point(), .edge(), .gabor(), .edge_grating()
│
├── MovingStimulus
│   ├── Wraps: StaticStimulus + motion trajectory
│   ├── Motion types: linear, circular, stationary
│   └── Fluent interface: .with_motion(type, **params)
│
└── CompositeStimulus
    ├── Combines: List[BaseStimulus]
    ├── Modes: add, max, mean, multiply
    └── Factory: Stimulus.compose([...], mode='add')
```

### Fluent Builder Pattern

```python
# Static stimulus
s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.5, 0.5))

# Static stimulus with motion (method chaining)
s2 = Stimulus.gaussian(amplitude=2.0, sigma=0.2).with_motion(
    'linear',
    start=(0.0, 0.0),
    end=(1.0, 1.0),
    num_steps=100
)

# Gabor with circular motion
s3 = Stimulus.gabor(
    wavelength=0.5,
    orientation=0.0
).with_motion(
    'circular',
    center=(0.0, 0.0),
    radius=0.5,
    num_steps=200
)

# Compose multiple stimuli
combined = Stimulus.compose([s1, s2, s3], mode='add')
```

## Key Features

### 1. Static Stimulus Factory Methods

- `Stimulus.gaussian()`: Gaussian bump
- `Stimulus.point()`: Binary disc
- `Stimulus.edge()`: Oriented edge
- `Stimulus.gabor()`: Gabor texture
- `Stimulus.edge_grating()`: Edge grating

All factory methods support:
- Amplitude control
- Center positioning
- Device placement (CPU, CUDA, MPS)

### 2. Motion Attachment

Any static stimulus can have motion attached via `.with_motion()`:

**Linear Motion:**
```python
stim.with_motion('linear', start=(0, 0), end=(1, 1), num_steps=100)
```

**Circular Motion:**
```python
stim.with_motion('circular', center=(0, 0), radius=0.5, num_steps=200)
```

**Stationary (no motion):**
```python
stim.with_motion('stationary', center=(0.5, 0.5), num_steps=10)
```

### 3. Stimulus Composition

Combine multiple stimuli with configurable modes:

**Addition (superposition):**
```python
Stimulus.compose([s1, s2, s3], mode='add')
```

**Maximum:**
```python
Stimulus.compose([s1, s2], mode='max')
```

**Mean:**
```python
Stimulus.compose([s1, s2], mode='mean')
```

**Multiplication:**
```python
Stimulus.compose([s1, s2], mode='multiply')
```

### 4. Serialization Support

All stimuli support `to_dict()` and `from_config()`:

```python
# Serialize
config = stimulus.to_dict()

# Deserialize
reloaded = StaticStimulus.from_config(config)
# or
reloaded = MovingStimulus.from_config(config)
# or
reloaded = CompositeStimulus.from_config(config)
```

### 5. State Management

All stimuli implement `reset_state()`:

```python
moving_stim.step()  # Advance to next time step
moving_stim.reset_state()  # Return to time 0
```

## Test Coverage

### Test Suite: test_stimulus_builder.py

**32 tests, all passing:**

1. **TestStaticStimulus (8 tests)**
   - Gaussian, point, edge, gabor, edge_grating generation
   - Device placement
   - Serialization/deserialization
   - State reset

2. **TestMovingStimulus (8 tests)**
   - Linear motion trajectory
   - Circular motion trajectory
   - Stationary motion
   - Step advancement
   - State reset
   - Error handling (missing params, unknown type)
   - Serialization

3. **TestCompositeStimulus (9 tests)**
   - Composition with add/max/mean/multiply modes
   - Static + moving stimulus composition
   - Reset cascading
   - Error handling (empty list, unknown mode)
   - Serialization

4. **TestBuilderPatterns (5 tests)**
   - Chained Gaussian + linear motion
   - Chained Gabor + circular motion
   - Complex multi-stimulus composition
   - Functional with_motion API

5. **TestBackwardCompatibility (2 tests)**
   - Legacy functional APIs still work
   - Modular functional APIs still work

## Backward Compatibility

✅ **Full backward compatibility maintained:**

1. **Legacy functional APIs:**
   - `gaussian_pressure_torch()`
   - `point_pressure_torch()`
   - `gabor_texture_torch()`
   - Still working, no changes required

2. **Modular functional APIs:**
   - `gaussian_stimulus()`
   - `gabor_texture()`
   - `edge_grating()`
   - No changes, builder wraps these

3. **Existing test suite:**
   - 399 tests passing (367 + 32 new)
   - 0 regressions
   - 4 expected deprecation warnings from Phase 1.2

## Benefits

### For Users

1. **Intuitive stimulus design:**
   - Readable, self-documenting code
   - Method chaining for gradual specification
   - Type-safe composition

2. **Flexibility:**
   - Any stimulus can have motion
   - Multiple stimuli can be combined
   - Easy switching between static and dynamic

3. **Consistency:**
   - Uniform interface across all stimulus types
   - Same pattern for serialization
   - Predictable behavior

### For Developers

1. **Extensibility:**
   - Add new stimulus types by implementing `forward()` in `StaticStimulus`
   - Add new motion patterns in `_generate_trajectory()`
   - Add new composition modes in `CompositeStimulus.forward()`

2. **Testability:**
   - Each component independently testable
   - Clear separation of concerns
   - Easy to mock for integration tests

3. **Maintainability:**
   - Single source of truth for each stimulus type
   - Builder pattern isolates construction logic
   - YAML serialization enables configuration-driven workflows

## Integration Points

### GUI (Phase 2)

The builder API will integrate with GUI in Phase 2:

```python
# StimulusDesignerTab will use builder API
stimulus = Stimulus.gaussian(...)
if motion_enabled:
    stimulus = stimulus.with_motion(motion_type, **params)

# Multiple stimuli can be stacked
stimuli_list = [stim1, stim2, stim3]
combined = Stimulus.compose(stimuli_list, mode=composition_mode)
```

### CLI/YAML (Phase 3)

YAML configuration will map to builder API:

```yaml
stimuli:
  - type: gaussian
    params:
      amplitude: 1.0
      sigma: 0.3
      center: [0.5, 0.5]
    motion:
      type: linear
      start: [0.0, 0.0]
      end: [1.0, 1.0]
      num_steps: 100
  
  - type: gabor
    params:
      wavelength: 0.5
      orientation: 0.0
    motion:
      type: circular
      center: [0.0, 0.0]
      radius: 0.5
      num_steps: 200

composition:
  mode: add
```

### Pipeline

Stimuli can be used in pipeline:

```python
from sensoryforge.stimuli import Stimulus

# Create stimulus
stim = Stimulus.gaussian(...).with_motion(...)

# Generate over time
for t in range(num_steps):
    frame = stim(xx, yy)
    # Process frame through pipeline...
    stim.step()
```

## Known Limitations

1. **Motion coordinate system:**
   - Motion is applied as coordinate translation
   - No rotation of oriented features yet

2. **Temporal interpolation:**
   - Motion uses discrete trajectory points
   - No sub-step interpolation

3. **DSL integration:**
   - Builder API uses functional generators, not DSL models
   - DSL can be added later for custom stimulus equations

## Next Steps (Phase 2)

1. **Update MechanoreceptorTab:**
   - Remove filter parameter from composite grid UI
   - Add arrangement selection dropdown

2. **Update StimulusDesignerTab:**
   - Replace single stimulus with list/stack
   - Add builder API integration
   - Motion toggle per stimulus
   - Composition mode selection

3. **Update SpikingTab:**
   - Innervation method selection per population
   - Display innervation weights visualization

## Conclusion

Phase 1.4 successfully implements a composable stimulus builder API that:

✅ Makes stimulus design intuitive and expressive  
✅ Supports motion attachment without duplicating code  
✅ Enables multi-stimulus composition with configurable modes  
✅ Maintains full backward compatibility  
✅ Provides comprehensive test coverage (32 new tests)  
✅ Integrates cleanly with existing architecture  

**Total test suite: 399 passing, 0 regressions**

The fluent builder pattern sets the foundation for Phase 2 GUI integration and Phase 3 YAML configuration support.
