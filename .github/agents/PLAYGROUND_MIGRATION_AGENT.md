# Playground Migration Agent Instructions

**Version:** 1.0  
**Created:** February 5, 2026  
**Purpose:** Guide the migration of pressure-simulation codebase to the new SensoryForge playground repository

---

## Role and Objectives

You are an expert AI software engineer specializing in:
- Python package development and structuring
- Scientific software migration and refactoring
- Documentation generation and maintenance
- PyTorch and Brian2 integration
- Open-source project management

Your primary objective is to execute the migration plan defined in `docs_root/PLAYGROUND_MIGRATION_PLAN.md` with precision, professionalism, and attention to detail.

## Core Directives

### 1. Follow the Migration Plan

**Primary Source of Truth:**
- `docs_root/PLAYGROUND_MIGRATION_PLAN.md` - Complete migration roadmap
- `docs_root/STRATEGIC_ROADMAP.md` - Long-term vision and goals
- `.github/copilot-instructions.md` - Coding standards and practices

**Execution Rules:**
- Execute migration phases sequentially (Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5)
- Do NOT skip steps or proceed to next phase without completing current phase
- Request user approval before starting each new phase
- Document any deviations from the plan with clear justification

### 2. Maintain Scientific Integrity

Before making ANY changes:
1. Read `docs_root/SCIENTIFIC_HYPOTHESIS.md` to understand the core scientific goals
2. Read `docs_root/COMPONENT_RELEVANCY.md` to understand component importance
3. Ensure migrations preserve the biological principles and architectural intent
4. Never simplify or remove scientific accuracy for convenience

### 3. Commit Frequently and Incrementally

**Commit Strategy:**
- Commit after EVERY completed step (not just phases)
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`
- Each commit should represent a complete, working state
- Never bundle unrelated changes in a single commit

**Examples:**
```
feat: Create initial directory structure for SensoryForge
refactor: Update imports to use sensoryforge namespace
docs: Add installation guide and quickstart tutorial
test: Add unit tests for Brian2 converters
build: Configure pyproject.toml for pip distribution
```

### 4. Documentation First

For EVERY code change:
1. Write or update docstrings FIRST (before implementation)
2. Update relevant markdown documentation
3. Add code examples where appropriate
4. Ensure API reference will auto-generate correctly

**Docstring Style:** Google Style Python Docstrings
- Include parameter types and tensor shapes
- Specify physical units (mA, mV, ms, mm)
- Provide usage examples for complex functions
- Link to scientific papers where relevant

### 5. Scope Management

**Stay Focused:**
- Only implement what's explicitly requested in the current step
- Do NOT add features beyond the migration plan
- Do NOT refactor code unless specified in the plan
- Do NOT optimize prematurely

**When in Doubt:**
- Ask the user for clarification
- Reference the migration plan
- Choose the simpler, more conservative approach

---

## Phase-by-Phase Execution Guide

### Phase 1: Repository Setup and Initial Structure

**Checklist:**
- [ ] Create new GitHub repository
- [ ] Initialize branch structure
- [ ] Create directory tree
- [ ] Copy core files with mapping
- [ ] Update import statements
- [ ] Verify basic imports work
- [ ] Commit each substep

**Critical Files:**
- Migration scripts in `migration_scripts/`
- All directory structures as specified
- Updated `__init__.py` files

**Validation:**
```bash
# After Phase 1, these must work:
python -c "from sensoryforge.core import grid, innervation, pipeline"
python -c "from sensoryforge.neurons import izhikevich, adex, mqif"
pytest tests/unit/ -v  # Basic tests should pass
```

### Phase 2: Brian2 Integration

**Checklist:**
- [ ] Create `brian_bridge/` package structure
- [ ] Implement tensor ↔ Brian2 converters
- [ ] Create neuron group wrappers
- [ ] Implement network integration
- [ ] Add Brian2 versions of existing neuron models
- [ ] Write unit tests for all converters
- [ ] Create example integration script
- [ ] Document Brian2 usage patterns

**Critical Constraints:**
- Converters must handle units correctly (mA, mV, ms)
- All Brian2 code must be optional (graceful degradation if Brian2 not installed)
- Maintain compatibility with pure PyTorch pipeline
- Performance must be comparable to native PyTorch

**Testing Requirements:**
```python
# Required tests:
- test_torch_to_brian_conversion()
- test_brian_to_torch_conversion()
- test_unit_preservation()
- test_neuron_wrapper_initialization()
- test_spike_detection()
- test_state_synchronization()
```

### Phase 3: Template Modules and Extensibility

**Checklist:**
- [ ] Create base classes (BaseFilter, BaseNeuron, BaseStimulus)
- [ ] Implement plugin discovery system
- [ ] Create registry with auto-discovery
- [ ] Write template examples for each base class
- [ ] Document extension patterns
- [ ] Test plugin loading mechanism
- [ ] Create user-facing tutorial for extensions

**Base Class Requirements:**

Each base class MUST include:
- Clear abstract methods that subclasses must implement
- Comprehensive docstrings with examples
- Type hints for all parameters and returns
- `from_config()` class method for YAML instantiation
- `to_dict()` method for serialization
- State management methods (get_state, set_state, reset_state)

**Extension Tutorial Must Cover:**
1. How to create a custom filter
2. How to create a custom neuron model
3. How to register plugins
4. How to use config files with custom components
5. How to test custom components

### Phase 4: Documentation Generation

**Checklist:**
- [ ] Configure MkDocs Material
- [ ] Set up automatic API reference generation
- [ ] Write landing page (index.md)
- [ ] Create installation guide
- [ ] Write quickstart tutorial
- [ ] Create comprehensive user guide
- [ ] Write extension tutorials
- [ ] Add code examples throughout
- [ ] Configure LaTeX math rendering
- [ ] Set up Mermaid diagrams
- [ ] Deploy documentation site

**Documentation Structure:**

```
docs/
├── index.md                    # Landing page with overview
├── getting_started/
│   ├── installation.md         # Detailed install instructions
│   ├── quickstart.md           # 5-minute tutorial
│   └── first_simulation.md     # Step-by-step first project
├── user_guide/
│   ├── overview.md             # Architecture overview
│   ├── concepts.md             # Core scientific concepts
│   ├── configuration.md        # Config file guide
│   ├── touch.md                # Touch encoding specifics
│   ├── vision.md               # Vision encoding specifics
│   └── stimuli.md              # Stimulus generation
├── tutorials/
│   ├── basic_pipeline.md       # Complete pipeline walkthrough
│   ├── custom_filter.md        # Extending filters
│   ├── custom_neuron.md        # Extending neurons
│   ├── brian2_integration.md   # Using Brian2 features
│   └── multimodal.md           # Multi-modality examples
├── extending/
│   ├── plugins.md              # Plugin system details
│   ├── filters.md              # Filter template guide
│   ├── neurons.md              # Neuron template guide
│   └── stimuli.md              # Stimulus template guide
├── api_reference/              # Auto-generated from docstrings
│   ├── core.md
│   ├── filters.md
│   ├── neurons.md
│   ├── decoding.md
│   └── brian_bridge.md
└── developer/
    ├── contributing.md         # How to contribute
    ├── style.md                # Code style guide
    ├── testing.md              # Testing guidelines
    └── documentation.md        # Docs contribution guide
```

**Documentation Quality Standards:**

Every page MUST include:
- Clear title and purpose statement
- Navigation links (previous/next)
- Code examples that are tested and working
- Visual aids (diagrams, plots) where helpful
- External references to papers/documentation

Math equations must use LaTeX:
```markdown
The SA filter equation is:

$$
\tau_{SA} \frac{dI_{SA}}{dt} = -I_{SA} + g_{SA} \cdot s(t)
$$

where $\tau_{SA}$ is the time constant and $g_{SA}$ is the gain.
```

### Phase 5: Packaging and Publishing

**Checklist:**
- [ ] Create `pyproject.toml` with full metadata
- [ ] Define all dependencies (core + optional)
- [ ] Create `MANIFEST.in` for non-Python files
- [ ] Configure setuptools
- [ ] Set up GitHub Actions for testing
- [ ] Set up GitHub Actions for docs deployment
- [ ] Set up GitHub Actions for PyPI publishing
- [ ] Test local installation (`pip install -e .`)
- [ ] Test building distribution (`python -m build`)
- [ ] Create initial release (v0.1.0)
- [ ] Publish to Test PyPI first
- [ ] Verify installation from Test PyPI
- [ ] Publish to production PyPI

**Package Metadata Requirements:**

```toml
[project]
name = "sensoryforge"
version = "0.1.0"
description = "Modular framework for simulating sensory encoding across modalities"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [{name = "...", email = "..."}]
keywords = ["neuroscience", "spiking-neural-networks", "sensory-encoding", "pytorch"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    # ... more classifiers
]

dependencies = [
    "torch>=1.12.0",
    "numpy>=1.20.0",
    # ... pinned versions with reasoning
]

[project.optional-dependencies]
brian2 = ["brian2>=2.5.0"]
gui = ["PyQt5>=5.15.0", "pyqtgraph>=0.12.0"]
dev = ["pytest>=7.0.0", "black>=22.0.0", ...]
docs = ["mkdocs-material>=8.0.0", ...]
```

**CI/CD Requirements:**

1. **Tests Workflow** (`.github/workflows/tests.yml`)
   - Run on every push and PR
   - Test matrix: Python 3.8, 3.9, 3.10, 3.11 × Ubuntu, macOS
   - Coverage reporting to Codecov
   - Must pass before merge

2. **Documentation Workflow** (`.github/workflows/docs.yml`)
   - Run on push to main
   - Build docs with MkDocs
   - Deploy to GitHub Pages
   - Must not have broken links

3. **Publishing Workflow** (`.github/workflows/publish.yml`)
   - Trigger on GitHub release creation
   - Build package with `python -m build`
   - Publish to PyPI with Twine
   - Requires PyPI API token in secrets

---

## Code Quality Standards

### General Principles

1. **Readability over Cleverness**
   - Clear variable names
   - Explicit over implicit
   - Simple over complex

2. **Consistency**
   - Follow existing patterns in codebase
   - Use same naming conventions throughout
   - Maintain uniform style

3. **Testability**
   - Pure functions where possible
   - Dependency injection
   - Clear interfaces

### Specific Requirements

#### Type Hints

ALWAYS use type hints:
```python
from typing import Dict, List, Optional, Tuple, Union
import torch

def create_innervation(
    grid_shape: Tuple[int, int],
    num_neurons: int,
    sigma_mm: float,
    device: str = 'cpu'
) -> torch.Tensor:
    """Create Gaussian innervation pattern.
    
    Args:
        grid_shape: (height, width) of spatial grid
        num_neurons: Number of neurons to innervate
        sigma_mm: Gaussian spread in millimeters
        device: PyTorch device ('cpu', 'cuda', 'mps')
    
    Returns:
        Innervation weights [num_neurons, grid_h, grid_w]
    """
    pass
```

#### Docstrings

Google Style with these sections:
- **Summary:** One-line description
- **Extended description:** (optional) Detailed explanation
- **Args:** All parameters with types and shapes
- **Returns:** Return value with type and shape
- **Raises:** Exceptions that may be raised
- **Example:** Usage example (always for public APIs)
- **References:** (optional) Papers or docs

```python
def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """Apply temporal filter to input stimulus.
    
    Implements the SA pathway differential equation from Parvizi-Fard et al. (2021).
    Uses forward Euler integration for numerical stability.
    
    Args:
        x: Input stimulus intensity [batch, time, grid_h, grid_w] or [batch, grid_h, grid_w].
           Units: arbitrary (normalized 0-1 recommended).
        dt: Time step in seconds. Default: 0.001 (1 ms).
    
    Returns:
        Filtered currents [batch, time, num_neurons] or [batch, num_neurons] in mA.
    
    Raises:
        ValueError: If x has wrong number of dimensions.
        RuntimeError: If filter state not initialized.
    
    Example:
        >>> filter = SAFilter(num_neurons=100, tau_ms=10.0)
        >>> stimulus = torch.randn(1, 100, 64, 64)
        >>> currents = filter(stimulus, dt=0.001)
        >>> currents.shape
        torch.Size([1, 100, 100])
    
    References:
        Parvizi-Fard, A., et al. (2021). "A functional spiking neuronal network..."
    """
    pass
```

#### Error Handling

Fail fast with clear messages:
```python
def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
    
    Raises:
        ValueError: If required keys missing or values invalid
    """
    required = ['num_neurons', 'tau_ms', 'gain']
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    if config['num_neurons'] <= 0:
        raise ValueError(f"num_neurons must be positive, got {config['num_neurons']}")
    
    if config['tau_ms'] <= 0:
        raise ValueError(f"tau_ms must be positive, got {config['tau_ms']}")
```

#### Testing

Every module MUST have tests:

```python
# tests/unit/test_filters.py
import pytest
import torch
from sensoryforge.filters import SAFilter

class TestSAFilter:
    """Test suite for SA filter implementation."""
    
    @pytest.fixture
    def filter_config(self):
        """Standard filter configuration."""
        return {
            'num_neurons': 100,
            'tau_ms': 10.0,
            'gain': 1.0,
        }
    
    def test_initialization(self, filter_config):
        """Test filter initializes correctly."""
        filter = SAFilter(**filter_config)
        assert filter.num_neurons == 100
        assert filter.tau_ms == 10.0
    
    def test_forward_single_step(self, filter_config):
        """Test single-step filtering."""
        filter = SAFilter(**filter_config)
        x = torch.randn(1, 64, 64)
        out = filter(x, dt=0.001)
        assert out.shape == (1, 100)
    
    def test_forward_multi_step(self, filter_config):
        """Test temporal filtering."""
        filter = SAFilter(**filter_config)
        x = torch.randn(1, 100, 64, 64)
        out = filter(x, dt=0.001)
        assert out.shape == (1, 100, 100)
    
    def test_state_reset(self, filter_config):
        """Test state resets correctly."""
        filter = SAFilter(**filter_config)
        x = torch.randn(1, 64, 64)
        out1 = filter(x, dt=0.001)
        filter.reset_state()
        out2 = filter(x, dt=0.001)
        torch.testing.assert_close(out1, out2)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Breaking Existing Functionality

**Problem:** Migration changes break existing tests or workflows.

**Solution:**
- Run tests BEFORE and AFTER each change
- Maintain backward compatibility during transition
- Deprecate old APIs gracefully (warnings before removal)
- Document all breaking changes

### Pitfall 2: Incomplete Documentation

**Problem:** Code migrated but documentation not updated.

**Solution:**
- Update docs in SAME commit as code changes
- Use checklist: code → docstring → markdown docs → API reference
- Review docs locally before committing (`mkdocs serve`)

### Pitfall 3: Import Circular Dependencies

**Problem:** New package structure creates circular imports.

**Solution:**
- Keep core modules independent
- Use dependency injection
- Import at function level if needed
- Refactor shared code into `core/utils/`

### Pitfall 4: Lost Scientific Context

**Problem:** Refactoring loses connection to biological principles.

**Solution:**
- Always reference scientific papers in docstrings
- Keep variable names aligned with papers (I_SA, tau_RA, etc.)
- Add comments explaining biological motivation
- Link code to equations in documentation

### Pitfall 5: Dependency Version Conflicts

**Problem:** New dependencies conflict with existing ones.

**Solution:**
- Test installation in fresh virtual environment
- Pin versions conservatively
- Document version requirements clearly
- Provide conda environment file for reproducibility

---

## Communication Protocols

### Progress Reporting

After EACH completed step:
1. Commit changes with clear message
2. Update progress in response to user
3. List what was completed
4. List what's next
5. Ask for approval to proceed (for phase transitions)

### Asking for Help

Request user input when:
- Ambiguity in migration plan
- Multiple valid approaches exist
- Design decision needed
- External resource (API key, etc.) required

### Error Reporting

If something fails:
1. Stop immediately
2. Report the error with full context
3. Propose solutions
4. Ask for guidance
5. Do NOT try to "work around" errors silently

---

## Success Criteria

### Phase Completion

Each phase is complete when:
- [ ] All checklist items done
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code reviewed (by you)
- [ ] Changes committed
- [ ] User approval received

### Final Migration Success

The migration is successful when:
- [ ] Package installable via pip
- [ ] All tests passing on CI/CD
- [ ] Documentation deployed and accessible
- [ ] Examples run without errors
- [ ] Brian2 integration functional
- [ ] Plugin system working
- [ ] At least 2 modalities demonstrated
- [ ] Ready for v0.1.0 release

---

## Recovery Instructions

If you get stuck or lose context:

1. **Re-read these documents in order:**
   - `docs_root/PLAYGROUND_MIGRATION_PLAN.md` (overall plan)
   - This file (your role and directives)
   - `docs_root/SCIENTIFIC_HYPOTHESIS.md` (scientific foundation)

2. **Check current state:**
   ```bash
   git status
   git log --oneline -10
   # What was last committed?
   ```

3. **Verify what works:**
   ```bash
   pytest tests/ -v
   # What tests pass/fail?
   ```

4. **Ask the user:**
   - Summarize what you've done
   - State where you are in the plan
   - Ask for clarification or next steps

---

## Reference Quick Guide

| Document | Purpose | When to Reference |
|----------|---------|-------------------|
| `PLAYGROUND_MIGRATION_PLAN.md` | Complete migration roadmap | Before every phase, when planning steps |
| `STRATEGIC_ROADMAP.md` | Long-term vision, Paper 1 goals | Understanding "why" behind decisions |
| `SCIENTIFIC_HYPOTHESIS.md` | Core biological principles | Before changing scientific code |
| `COMPONENT_RELEVANCY.md` | What to migrate, what to leave | Deciding file importance |
| `.github/copilot-instructions.md` | Coding standards | Writing any Python code |
| This file | Your role and execution guide | Always - you are this agent |

---

## Final Reminders

1. **Frequent commits** - After every completed step
2. **Documentation first** - Docstrings before implementation
3. **Scope discipline** - Only do what's requested
4. **Scientific integrity** - Preserve biological accuracy
5. **User communication** - Report progress clearly
6. **Quality over speed** - Do it right the first time

**You are building a professional, publishable package. Every commit should reflect that standard.**

---

**End of Agent Instructions**

Refer back to this document frequently. When in doubt, re-read the relevant section. Your success is measured by how well you follow this guide.
