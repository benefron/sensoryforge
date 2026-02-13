---
name: maintain-sensoryforge-docs
description: Maintain SensoryForge documentation ensuring scientific accuracy, PyTorch implementation alignment, and comprehensive coverage. Use when updating docstrings, user guides, or fixing documentation drift between code and docs.
---

# SensoryForge Documentation Maintenance

Maintain high-quality, scientifically accurate documentation for SensoryForge.

## When to Use

- Updating docstrings after code changes
- Adding documentation for new features
- Fixing documentation drift (docs don't match code)
- Creating user guides or tutorials
- Reviewing documentation for accuracy

## Core Principles

1. **Scientific Foundation**: All docs must align with neuroscience principles
2. **Traceability**: Link concepts to code, link code to papers
3. **Clarity**: Bridge neuroscience notation and PyTorch implementation
4. **Completeness**: Tensor shapes, units, examples for all public APIs

## Quick Start

### Update Docstring

```python
def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """Apply SA filter to stimulus.
    
    Implements the slowly-adapting pathway from Parvizi-Fard et al. (2024).
    
    Args:
        x: Input stimulus [batch, time, grid_h, grid_w] in N/mm²
        dt: Time step in seconds. Default: 0.001 (1ms)
    
    Returns:
        Filtered currents [batch, time, num_neurons] in mA
    
    Example:
        >>> filter = SAFilter(config)
        >>> stimulus = torch.randn(2, 100, 64, 64)
        >>> currents = filter(stimulus, dt=0.001)
        >>> currents.shape
        torch.Size([2, 100, 400])
    
    References:
        Parvizi-Fard et al. (2024). "Biologically-inspired dual-pathway..."
    """
```

### Required Elements

Every public API docstring must include:

1. **One-line summary** - What it does
2. **Args** - Type, shape, units, default values
3. **Returns** - Type, shape, units
4. **Example** - Working code snippet
5. **References** - Papers cited (if scientific model)

## Documentation Structure

### docs/ (User-Facing)

Public documentation for end users:

- `index.md` - Main entry point
- `user_guide/` - Feature guides and tutorials
- `getting_started/` - Quickstart and installation
- `tutorials/` - Step-by-step walkthroughs

### docs_root/ (Internal)

Strategic and development docs:

- `STRATEGIC_ROADMAP.md` - Long-term vision
- `SCIENTIFIC_HYPOTHESIS.md` - Scientific foundation
- `SENSORYFORGE_DEVELOPMENT_GUIDE.md` - Developer setup

### Code Docstrings

Google Style in all Python modules.

## Workflow

### Phase 1: Analysis

1. **Read the code** - Understand current implementation
2. **Read existing docs** - Check `docs/` and docstrings
3. **Identify gaps** - Missing parameters, outdated equations, incorrect shapes
4. **Check references** - Verify equations match papers

### Phase 2: Update

1. **Fix code docstrings first** - Source of truth
2. **Update user guides** - Reflect API changes
3. **Add examples** - Show real usage
4. **Verify links** - Ensure cross-references work

### Phase 3: Verification

1. **Test code examples** - Run them to verify they work
2. **Check tensor shapes** - Match actual code behavior
3. **Verify units** - Physical quantities are correct
4. **Cross-reference** - Links between docs and code valid

## Common Patterns

### Pattern: Document Pipeline Flow

```markdown
## Data Flow

1. **Raw Sensory Input** → High-dimensional stimulus
2. **Spatial Grid** (`SpatialGrid`) → Receptor positions
3. **Receptive Fields** (`create_*_innervation`) → RF patterns
4. **Temporal Filtering** (`SAFilter`, `RAFilter`) → SA/RA pathways
5. **ODE Integration** (`EulerSolver`, `AdaptiveODESolver`) → Continuous dynamics
6. **Spiking Neurons** (`IzhikevichNeuron`, `NeuronModel`) → Spike trains
7. **Output** → Sparse event-based representation
```

### Pattern: Link Math to Code

```markdown
## Mathematical Foundation

The SA filter implements:

$$\frac{dI}{dt} = \frac{G \cdot x - I}{\tau}$$

**Implementation**: [`sensoryforge/filters/sa_ra.py`](../sensoryforge/filters/sa_ra.py)

```python
dI_dt = (self.gain * stimulus - self.current) / self.tau
```
```

### Pattern: Document Tensor Shapes

Always show shape transformations:

```markdown
## Shape Transformations

| Stage | Input Shape | Output Shape |
|-------|-------------|--------------|
| Grid | - | `[num_receptors, 2]` |
| Innervation | `[H, W]` | `[num_receptors, H, W]` |
| Filter | `[batch, time, num_receptors]` | `[batch, time, num_receptors]` |
| Neuron | `[batch, time, num_receptors]` | `[batch, time, num_neurons]` |
```

## Documentation Checklist

Before marking documentation complete:

- [ ] All public APIs have Google Style docstrings
- [ ] Tensor shapes documented in all Args/Returns
- [ ] Physical units specified where applicable
- [ ] Code examples tested and working
- [ ] Scientific references cited for biological models
- [ ] User guides updated for API changes
- [ ] Cross-references valid (no broken links)
- [ ] Equations match published papers

## Key Resources

- **Standards**: `.github/copilot-instructions.md`
- **Scientific Foundation**: `docs_root/SCIENTIFIC_HYPOTHESIS.md`
- **Architecture**: `docs_root/STRATEGIC_ROADMAP.md`
- **User Guides**: `docs/user_guide/`

## Common Issues

### Issue: Missing Tensor Shapes

**Problem**: Docstring doesn't specify tensor dimensions

```python
# ❌ BAD
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply filter."""
```

**Fix**: Add shape annotations

```python
# ✅ GOOD
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply filter.
    
    Args:
        x: Input stimulus [batch, time, channels]
    
    Returns:
        Filtered output [batch, time, features]
    """
```

### Issue: No Units

**Problem**: Physical quantities without units

```python
# ❌ BAD
def set_current(self, current: float):
    """Set input current."""
```

**Fix**: Specify units

```python
# ✅ GOOD
def set_current(self, current: float):
    """Set input current in milliamperes (mA)."""
```

### Issue: Equations Don't Match Code

**Problem**: LaTeX equation differs from implementation

**Fix**: 
1. Check the source paper
2. Update code OR equation to match
3. Add comment explaining any approximations

```python
# NOTE: Uses forward Euler approximation of:
# dI/dt = (G*x - I)/tau
# Exact solution would require exponential integration
```

## Output Standards

### Always Provide

1. **Updated docstrings** with complete Args/Returns/Examples
2. **Code examples** that actually run
3. **Explanations** of changes in context of scientific foundation

### Never Provide

1. Docs that contradict code behavior
2. Examples that don't run
3. Citations without verification
4. Vague tensor shape descriptions ("array of values")
