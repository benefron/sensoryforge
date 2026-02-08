# Copilot Instructions for SensoryForge

## Primary Operator Rule
- Always follow the prompt literally: do not implement changes beyond what the user explicitly requests, do not take initiative beyond the provided instructions and the structured plan, and do not edit code or documents until you either confirm with the user or the prompt already authorizes the change.
- Always summarize the plan in its own markdown file before modifying anything, and keep suggestions strictly in chat.
- Running tests or creating commits does not require additional permission.

## Core Philosophy

SensoryForge is a **modular, extensible framework** for simulating sensory encoding across multiple modalities. The framework is:
- **Modality-agnostic:** Touch is the first implementation, but the architecture generalizes to vision, audition, and multi-modal fusion
- **Scientifically grounded:** All implementations must align with biological principles and published neuroscience literature
- **User-friendly:** Clear documentation, intuitive APIs, and comprehensive tutorials are mandatory
- **Production-ready:** Professional code quality suitable for academic publication and open-source adoption

## Core References

### Scientific Foundation
- Start with the biological principles: sensory compression through receptive fields, dual-pathway processing (sustained vs. transient), and feature extraction at multiple architectural stages
- All encoding decisions should be explainable in terms of neuroscience concepts
- Maintain traceability between code and scientific papers (cite in docstrings)

### Documentation Standards
- **Every public API** must have comprehensive docstrings with examples
- **Every module** must have a corresponding section in the user guide
- **Every new feature** requires a tutorial demonstrating usage
- Use MkDocs Material for documentation generation

### Documentation Placement
- **docs/**: User-facing documentation meant to ship publicly (guides, tutorials, API references).
- **docs_root/**: Internal, unexposed documents (research notes, internal reviews, working plans). Keep these out of published docs.
- **reviews/**: Shareable engineering review artifacts and remediation plans; use for summaries that can be committed.

### Code Quality
- **Type hints are mandatory** for all function signatures
- **Tests are required** for all new code (minimum 80% coverage)
- **Docstrings must be Google Style** with Args/Returns/Examples
- **Commit messages must follow Conventional Commits** (feat:, fix:, docs:, refactor:, test:, build:, ci:)

## Architecture Overview

### Package Structure

```
sensoryforge/
├── core/                 # Fundamental components
│   ├── pipeline.py       # Orchestration and coordination
│   ├── grid.py           # Spatial substrate (2D grids, Poisson arrangements, etc.)
│   ├── innervation.py    # Receptive field generation
│   ├── composite_grid.py # Multi-population composable grids
│   └── utils/            # Shared utilities
├── filters/              # Temporal filtering
│   ├── base.py           # Abstract base class (BaseFilter)
│   ├── sa_ra.py          # SA/RA dual-pathway filters
│   ├── center_surround.py # ON/OFF for vision
│   └── custom/           # User extensions (auto-discovered)
├── neurons/              # Spiking neuron models
│   ├── base.py           # Abstract base class (BaseNeuron)
│   ├── izhikevich.py     # Izhikevich model
│   ├── adex.py           # Adaptive exponential model
│   ├── mqif.py           # Multi-quadratic integrate-and-fire
│   ├── model_dsl.py      # Equation DSL → nn.Module compiler (future)
│   └── custom/           # User extensions
├── solvers/              # ODE/SDE integration backends
│   ├── base.py           # Abstract solver interface
│   ├── euler.py          # Forward Euler (current default)
│   ├── adaptive.py       # Wrapper for torchdiffeq/torchode (Dormand-Prince, etc.)
│   └── sde.py            # Stochastic differential equation solvers
├── stimuli/              # Stimulus generation
│   ├── base.py           # Abstract base class (BaseStimulus)
│   ├── gaussian.py       # Gaussian blobs
│   ├── texture.py        # Texture patterns
│   ├── moving.py         # Moving/sliding stimuli
│   └── custom/           # User extensions
├── gui/                  # Optional PyQt5 interface
│   └── (visualization and interactive configuration)
├── config/               # Configuration management
│   ├── schemas/          # JSON schemas for validation
│   └── default_config.yml # Default configuration
└── plugins/              # Plugin system infrastructure
    └── registry.py       # Auto-discovery and registration
```

### Data Flow

```
Raw Sensory Input (high-dim, dense)
  ↓
Spatial Grid (SpatialGrid — single, composite, or multi-layer)
  ↓
Receptive Fields (create_*_innervation)
  ↓
Temporal Filtering (BaseFilter subclasses)
  ┌────────┴────────┐
  ↓                 ↓
Sustained Path     Transient Path
(SA-like)          (RA-like)
  ↓                 ↓
ODE Solver (Euler, Dormand-Prince, or adaptive via torchdiffeq/torchode)
  ↓                 ↓
Spiking Neurons (BaseNeuron subclasses — hand-written or DSL-compiled)
  ↓
Spike Trains (sparse, event-based)
  ↓
[Optional] Training / ML (differentiable end-to-end via adjoint method)
```

### Composable Grids (Future)

SensoryForge supports multi-population spatial substrates for modeling
receptor mosaics (e.g., SA1/RA1/SA2 in touch, L/M/S cones in vision):

```python
# Example: Three-cone-type retinal mosaic
from sensoryforge.core.composite_grid import CompositeGrid

grid = CompositeGrid(
    shape=(64, 64),
    populations={
        'L_cone': {'density': 0.60, 'filter': 'long_wave'},
        'M_cone': {'density': 0.30, 'filter': 'medium_wave'},
        'S_cone': {'density': 0.10, 'filter': 'short_wave'},
    },
    arrangement='poisson',  # or 'hex', 'jittered_grid'
)
```

Each population gets its own filter pathway and neuron model while sharing the
same spatial coordinate system.

## Coding Conventions

### PyTorch Standards

- **Never hand-roll loops** over neurons or spatial dimensions—use tensor broadcasting
- **Always specify device** explicitly (cpu, cuda, mps)
- **Use `.to(device)`** instead of manual tensor placement
- **Batch dimensions first** convention: `[batch, time, ...]` or `[batch, ...]`
- **Avoid in-place operations** unless performance-critical (and document why)

### Physical Units

Maintain consistent units throughout:
- **Time:** milliseconds (ms) for user-facing APIs, seconds (s) for internal computations (document conversions)
- **Currents:** milliamperes (mA)
- **Voltages:** millivolts (mV)
- **Spatial distances:** millimeters (mm)
- **Angles:** radians

Document units in docstrings:
```python
def set_current(self, current: torch.Tensor):
    """Set input current.
    
    Args:
        current: Input currents [num_neurons] in mA
    """
```

### Tensor Shapes

Always document tensor shapes in docstrings:
```python
def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """Apply filter to stimulus.
    
    Args:
        x: Input stimulus [batch, time, grid_h, grid_w] or [batch, grid_h, grid_w]
        dt: Time step in seconds
    
    Returns:
        Filtered currents [batch, time, num_neurons] or [batch, num_neurons] in mA
    """
```

### Configuration Management

All components must be constructible from YAML config:

```python
class MyFilter(BaseFilter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tau_ms = config['tau_ms']
        self.gain = config.get('gain', 1.0)  # Optional with default
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MyFilter':
        """Factory method for YAML instantiation."""
        return cls(config)
```

Corresponding YAML:
```yaml
filter:
  type: MyFilter
  tau_ms: 10.0
  gain: 1.5
```

### Error Handling

Fail fast with informative messages:

```python
def validate_grid_shape(shape: Tuple[int, int]):
    """Validate grid dimensions.
    
    Args:
        shape: (height, width) tuple
    
    Raises:
        ValueError: If shape invalid
    """
    if len(shape) != 2:
        raise ValueError(f"Grid shape must be 2D, got {len(shape)}D")
    
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Grid dimensions must be positive, got {shape}")
    
    if shape[0] > 1000 or shape[1] > 1000:
        raise ValueError(f"Grid too large (max 1000x1000), got {shape}")
```

## Documentation Standards

### Docstring Template

```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional_param: Type3 = default_value
) -> ReturnType:
    """One-line summary of function purpose.
    
    Extended description providing more context about what this function does,
    when to use it, and any important caveats. Link to relevant papers or
    documentation sections where appropriate.
    
    Args:
        param1: Description of parameter 1. Include tensor shape if applicable,
            e.g., [batch, time, channels]. Units: specify physical units.
        param2: Description of parameter 2.
        optional_param: Description of optional parameter. Default: default_value.
    
    Returns:
        Description of return value. Include tensor shape and units.
    
    Raises:
        ValueError: When and why this exception is raised.
        RuntimeError: When and why this exception is raised.
    
    Example:
        >>> # Show realistic usage
        >>> result = function_name(
        ...     param1=torch.randn(10, 5),
        ...     param2=0.5
        ... )
        >>> result.shape
        torch.Size([10, 20])
    
    References:
        Author et al. (Year). "Paper Title". Journal. DOI or URL.
    """
    pass
```

### Markdown Documentation

Every module needs a corresponding markdown guide:

**User Guide Structure:**
```markdown
# Module Name

## Overview
Brief description of what this module does and when to use it.

## Concepts
Explain the scientific or algorithmic concepts.

## Usage

### Basic Example
\```python
# Simple usage example
\```

### Advanced Example
\```python
# More complex usage
\```

## Configuration
Explain YAML configuration options.

## API Reference
Link to auto-generated API docs.

## See Also
Links to related modules and tutorials.
```

## Testing Requirements

### Test Organization

```
tests/
├── unit/              # Isolated component tests
│   ├── test_filters.py
│   ├── test_neurons.py
│   ├── test_innervation.py
│   ├── test_model_dsl.py
│   ├── test_solvers.py
│   └── test_composite_grid.py
├── integration/       # Multi-component tests
│   └── test_pipeline.py
└── fixtures/          # Shared test data and configs
    ├── configs/
    └── data/
```

### Test Coverage Requirements

- **Minimum 80% overall coverage**
- **100% coverage for base classes** (critical infrastructure)
- **All public APIs must have tests**
- **Edge cases must be tested** (empty inputs, extreme values, errors)

### Test Style

```python
import pytest
import torch
from sensoryforge.filters import SAFilter

class TestSAFilter:
    """Test suite for SA filter implementation."""
    
    @pytest.fixture
    def standard_config(self):
        """Standard configuration for most tests."""
        return {
            'num_neurons': 100,
            'tau_ms': 10.0,
            'gain': 1.0,
        }
    
    @pytest.fixture
    def small_stimulus(self):
        """Small test stimulus."""
        return torch.randn(1, 64, 64)
    
    def test_initialization_with_valid_config(self, standard_config):
        """Test filter initializes correctly with valid config."""
        filter = SAFilter(standard_config)
        assert filter.num_neurons == 100
        assert filter.tau_ms == 10.0
    
    def test_initialization_rejects_negative_tau(self):
        """Test initialization fails with negative time constant."""
        config = {'num_neurons': 100, 'tau_ms': -1.0}
        with pytest.raises(ValueError, match="tau_ms must be positive"):
            SAFilter(config)
    
    def test_forward_preserves_batch_dimension(self, standard_config, small_stimulus):
        """Test forward pass preserves batch dimension."""
        filter = SAFilter(standard_config)
        output = filter(small_stimulus)
        assert output.shape[0] == small_stimulus.shape[0]
    
    @pytest.mark.parametrize("dt", [0.0001, 0.001, 0.01])
    def test_forward_with_different_timesteps(self, standard_config, small_stimulus, dt):
        """Test forward pass works with various time steps."""
        filter = SAFilter(standard_config)
        output = filter(small_stimulus, dt=dt)
        assert output.shape == (1, 100)
```

## Extensibility Architecture

### Base Class Requirements

All base classes (BaseFilter, BaseNeuron, BaseStimulus) must:

1. **Inherit from `torch.nn.Module`** (for PyTorch compatibility)
2. **Define abstract methods** that subclasses must implement
3. **Provide `from_config()` class method** for YAML instantiation
4. **Provide `to_dict()` method** for serialization
5. **Implement state management** (get_state, set_state, reset_state)
6. **Include comprehensive docstrings** with examples

### Plugin Discovery

The plugin system automatically discovers and registers custom components:

```python
# sensoryforge/plugins/my_custom_filter.py
from sensoryforge.filters.base import BaseFilter
import torch

class MyCustomFilter(BaseFilter):
    """Custom filter implementation.
    
    This filter does X by applying Y algorithm.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
    
    def forward(self, x, dt=None):
        """Apply custom filtering."""
        return torch.relu(x - self.threshold)
    
    def reset_state(self):
        """No state to reset."""
        pass
```

User code:
```python
from sensoryforge.plugins.registry import registry

# Discover all plugins in custom directory
registry.discover_plugins(Path('sensoryforge/plugins'))

# Instantiate custom filter
FilterClass = registry.get_filter('MyCustomFilter')
filter = FilterClass({'threshold': 0.3})
```

## Simulation Backend: PyTorch + Adaptive ODE Solvers

### Philosophy

SensoryForge is **pure PyTorch**. There is no Brian2 bridge, no NEST integration,
no external simulation framework. Every component is a `torch.nn.Module`, which
means the entire encoding pipeline is:

- **Differentiable** end-to-end (for ML training via adjoint method)
- **GPU-accelerated** (CUDA, MPS) with zero serialization overhead
- **Batchable** for parallel simulation of multiple stimuli
- **Composable** with standard PyTorch optimizers, loss functions, and data loaders

### ODE Integration

All neuron models currently use **forward Euler** for simplicity. The architecture
supports swapping in adaptive solvers when needed:

| Solver | Library | When to Use |
|--------|---------|-------------|
| Forward Euler | built-in | Default; fast; sufficient for non-stiff systems |
| Dormand-Prince (RK45) | torchdiffeq | Stiff systems (AdEx), accuracy-critical work |
| Adaptive stepping | torchode | GPU-parallel batched solving |
| Adjoint method | torchdiffeq | Memory-efficient backprop through long time series |

```python
# Example: Using adaptive solver with a neuron model
from sensoryforge.solvers.adaptive import AdaptiveODESolver

solver = AdaptiveODESolver(method='dopri5', rtol=1e-5, atol=1e-7)
neuron = IzhikevichNeuronTorch(config, solver=solver)
```

### Equation DSL (Future)

A Brian2-inspired declarative interface for neuroscientists who prefer to define
models via equations rather than writing `nn.Module` subclasses:

```python
from sensoryforge.neurons.model_dsl import NeuronModel

izh = NeuronModel(
    equations='''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
        du/dt = (a * (b*v - u)) / ms
    ''',
    threshold='v >= 30 * mV',
    reset='v = c; u = u + d',
    parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
)

# Compiles to a standard nn.Module — same interface as hand-written models
module = izh.compile(solver='euler', device='cuda')
```

The DSL parses equations with sympy, extracts the ODE system, and compiles it
into a PyTorch-native `nn.Module` that is fully compatible with the rest of the
pipeline. This is a **plug-in addition** — existing hand-written models continue
working unchanged.

### Two Paths to Neuron Models

| Path | Audience | How |
|------|----------|-----|
| **Hand-written `nn.Module`** | Engineers, performance-critical | Subclass `BaseNeuron`, implement `forward()` |
| **Equation DSL / `NeuronModel`** | Neuroscientists, rapid prototyping | Write equation strings, call `.compile()` |

Both paths produce objects with the same interface (`forward()`, `reset_state()`,
`from_config()`, serializable). The pipeline does not care which path created the
model.

## Commit Workflow

### Conventional Commits

Use semantic prefixes:

- `feat:` New feature or capability
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code restructuring without behavior change
- `test:` Adding or modifying tests
- `build:` Build system or dependencies
- `ci:` CI/CD configuration
- `perf:` Performance improvement
- `style:` Code style/formatting only

Examples:
```
feat: Add center-surround filter for vision modality
fix: Correct innervation weight normalization
docs: Add tutorial for custom neuron models
refactor: Simplify pipeline initialization logic
test: Add edge case tests for equation DSL parser
build: Update PyTorch requirement to >=1.12
ci: Add Python 3.11 to test matrix
perf: Vectorize receptive field computation
```

### Commit Frequency

Commit after:
- Each completed feature or fix
- Each documentation update
- Each test addition
- End of work session (even if incomplete)

Do NOT commit:
- Broken code (unless explicitly WIP)
- Failing tests
- Large uncommented changes
- Multiple unrelated changes together

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(filters): Add adaptive threshold to SA filter

Implements dynamic threshold adjustment based on recent activity history.
Uses exponential moving average with configurable time constant.

Closes #42
```

## Release Management

### Versioning

Follow Semantic Versioning (semver): MAJOR.MINOR.PATCH

- **MAJOR:** Breaking changes (incompatible API)
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.2.0`: Add new modality support (new feature)
- `0.2.0` → `0.2.1`: Fix innervation bug (patch)
- `0.2.1` → `1.0.0`: Refactor base classes (breaking change)

### Release Checklist

Before releasing:
- [ ] All tests passing
- [ ] Documentation up to date
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Git tag created (`git tag v0.2.0`)
- [ ] GitHub release created
- [ ] Package published to PyPI

## Common Patterns

### Creating a New Filter

1. Create file in `sensoryforge/filters/`
2. Inherit from `BaseFilter`
3. Implement required methods
4. Add comprehensive docstrings
5. Create unit tests
6. Add to documentation
7. Add example usage

```python
# sensoryforge/filters/my_filter.py
from sensoryforge.filters.base import BaseFilter
import torch
from typing import Dict, Any

class MyFilter(BaseFilter):
    """One-line description.
    
    Extended description...
    
    Attributes:
        param1: Description
        param2: Description
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.param1 = config['param1']
        self.param2 = config.get('param2', default)
    
    def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """Apply filter (documented)."""
        # Implementation
        pass
    
    def reset_state(self):
        """Reset filter state (documented)."""
        pass
```

### Creating a New Neuron Model

SensoryForge supports two paths for creating neuron models:

**Path 1: Hand-written nn.Module** (recommended for performance-critical models)

1. Create file in `sensoryforge/neurons/`
2. Inherit from `BaseNeuron`
3. Implement `forward()`, `reset_state()`
4. Add comprehensive docstrings with equations, units, references
5. Create unit tests
6. Add to documentation

```python
from sensoryforge.neurons.base import BaseNeuron
from typing import Tuple, Dict

class MyNeuron(BaseNeuron):
    """Neuron model description.
    
    Implements the XYZ model from Author et al. (Year).
    
    Equations:
        dv/dt = f(v, u, I)
        du/dt = g(v, u)
    """
    
    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate one time step.
        
        Args:
            current: Input current [batch, num_neurons] in mA
        
        Returns:
            spikes: Binary spike tensor [batch, num_neurons]
            state: Dictionary with 'voltage', etc.
        """
        # Implementation using self.solver (Euler or adaptive)
        pass
```

**Path 2: Equation DSL** (recommended for rapid prototyping)

```python
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations='''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
        du/dt = (a * (b*v - u)) / ms
    ''',
    threshold='v >= 30 * mV',
    reset='v = c; u = u + d',
    parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
)
module = model.compile(solver='euler')  # Produces standard nn.Module
```

## Troubleshooting Common Issues

### Import Errors

If getting import errors:
1. Check package installed in development mode: `pip install -e .`
2. Verify `__init__.py` files exist in all packages
3. Use absolute imports: `from sensoryforge.core import grid`

### ODE Solver Issues

If adaptive solvers fail:
1. Ensure torchdiffeq or torchode installed: `pip install torchdiffeq` or `pip install torchode`
2. Check that the neuron model's `ode_func` returns the correct tensor shape
3. Try reducing `rtol`/`atol` if solver diverges
4. Fall back to forward Euler for debugging (set `solver='euler'` in config)
5. For stiff systems (AdEx), prefer `method='dopri5'` over `'euler'`

### Test Failures

If tests failing:
1. Run specific test: `pytest tests/unit/test_filters.py::TestSAFilter::test_name -v`
2. Check fixture paths correct
3. Verify test isolation (no state leaking between tests)
4. Use `pytest --pdb` to debug

### Documentation Not Building

If MkDocs fails:
1. Check all links valid
2. Verify code examples are syntactically correct
3. Ensure all referenced files exist
4. Run `mkdocs build --strict` to see all warnings

## Performance Optimization

### When to Optimize

Only optimize when:
- Profiling shows bottleneck
- Performance unacceptable for real use
- Large-scale simulations required

### Optimization Strategies

1. **Vectorization:** Replace loops with tensor operations
2. **GPU acceleration:** Move computations to CUDA
3. **Batch processing:** Process multiple stimuli simultaneously
4. **Sparse operations:** Use sparse tensors for sparse data
5. **JIT compilation:** Use `torch.jit.script` for hot paths

### Profiling

```python
import torch.utils.benchmark as benchmark

def profile_function():
    """Profile critical code path."""
    timer = benchmark.Timer(
        stmt='filter(stimulus)',
        setup='from my_module import filter; import torch; stimulus = torch.randn(1, 100, 64, 64)',
        num_threads=1
    )
    print(timer.timeit(100))
```

## Getting Help

### Internal Resources

- `docs/`: User-facing documentation
- `examples/`: Working code examples
- `tests/`: Test examples showing usage
- `CONTRIBUTING.md`: Contribution guidelines

### External Resources

- PyTorch docs: https://pytorch.org/docs/
- torchdiffeq (adaptive ODE solvers): https://github.com/rtqichen/torchdiffeq
- torchode (parallel ODE solving): https://github.com/martenlienen/torchode
- sympy (equation parsing for DSL): https://docs.sympy.org/
- MkDocs Material: https://squidfunk.github.io/mkdocs-material/

### Asking Questions

When asking for help:
1. Describe what you're trying to do
2. Show minimal reproducible example
3. Include error messages (full traceback)
4. State what you've already tried

---

**Remember:** This is a professional, publishable package. Every commit should reflect production-quality code suitable for academic publication and community adoption. When in doubt, prioritize clarity, documentation, and correctness over cleverness or brevity.
