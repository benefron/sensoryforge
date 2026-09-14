# SensoryForge — Full Code Review Report

**Date:** 2026-04-08  
**Reviewer:** Claude (claude-sonnet-4-6)  
**Standards reference:** `.github/copilot-instructions.md`  
**Status:** Open — pending mitigation

---

## Table of Contents

- [Critical Issues](#critical-issues)
- [Major Issues](#major-issues)
- [Medium Issues](#medium-issues)
- [Low / Backlog](#low--backlog)
- [Documentation Issues](#documentation-issues)
- [Code Quality Issues](#code-quality-issues)
- [Security Considerations](#security-considerations)
- [Performance Issues](#performance-issues)
- [Testing Gaps](#testing-gaps)
- [Summary Table](#summary-table)
- [Recommended Mitigation Order](#recommended-mitigation-order)

---

## Critical Issues

> Must fix before any feature merge. These either crash at runtime or directly contradict a core architectural guarantee.

---

### C-1: CompositeGrid not implemented in SimulationEngine

**File:** `sensoryforge/core/simulation_engine.py` (~line 92)  
**Rule violated:** Documented, user-facing feature  
**Effort:** Medium (1–2 days)

`_build_grids()` raises `NotImplementedError` for `arrangement == "composite"`. Composite grids are a prominently documented feature in both the copilot instructions and the user guide. Any user who follows the docs and provides a composite grid config will hit a hard crash with no useful error message.

**Current behaviour:**
```python
if arrangement == "composite":
    raise NotImplementedError("Composite grids not yet implemented in SimulationEngine")
```

**Fix:**  
Implement the composite branch — read the per-population configs from the YAML, instantiate `CompositeGrid` with the appropriate layer configs, and return it. The `CompositeGrid` class already exists in `sensoryforge/core/composite_grid.py`; the SimulationEngine just never wires it up.

---

### C-2: DSL neurons use numpy/lambdify — no GPU or autograd support

**File:** `sensoryforge/neurons/model_dsl.py`  
**Rule violated:** "Pure PyTorch", "differentiable end-to-end", "GPU-accelerated"  
**Effort:** Large (3–5 days)

The copilot instructions state the DSL "compiles to a standard nn.Module — same interface as hand-written models" and that the whole pipeline is "differentiable end-to-end via adjoint method". The current implementation uses `sympy.lambdify` with numpy, which:

- Silently breaks (or errors) when tensors are on CUDA or MPS
- Has zero gradient flow — backprop through DSL neurons is impossible
- Violates the "pure PyTorch" architectural guarantee

The instructions themselves acknowledge this as "known technical debt" — which means it is a known violation, not an intentional design choice.

**Fix:**  
Rewrite the DSL code generator to emit PyTorch operations. Use `sympy.lambdify` with `torch` as the backend (or custom `sympy` → `torch.*` code generation). Validate with `torch.autograd.gradcheck` and a CUDA forward pass test.

---

## Major Issues

> Should fix soon. These violate stated project standards or will cause subtle bugs in real usage.

---

### M-1: `reset_state()` vs `reset_states()` naming — breaks polymorphism

**File:** `sensoryforge/filters/sa_ra.py`  
**Rule violated:** BaseFilter interface contract (`reset_state()` singular)  
**Effort:** Small (30 min)

`SAFilterTorch` exposes `reset_states()` (plural). The `BaseFilter` contract and the copilot instructions both specify `reset_state()` (singular). Any code that iterates a list of `BaseFilter` instances and calls `.reset_state()` will get `AttributeError` on an SA filter.

**Fix:**  
Rename `reset_states()` → `reset_state()` in `SAFilterTorch`, update all callers, run the test suite to confirm.

---

### M-2: Type hints missing on many public APIs

**Files:** `sensoryforge/core/innervation.py`, `sensoryforge/neurons/*.py`, `sensoryforge/gui/tabs/*.py`, and others  
**Rule violated:** "Type hints are mandatory for all function signatures"  
**Effort:** Medium (1 day)

Multiple `forward()` methods, filter callbacks, and GUI handlers lack type annotations entirely.

**Fix:**  
Run `mypy --disallow-untyped-defs sensoryforge/` and add hints to every flagged signature. Prioritise the public API surface: filters, neurons, pipeline, grids. Add mypy as a CI check so this cannot regress.

---

### M-3: Docstrings inconsistent — many violate the Google Style mandate

**Files:** `sensoryforge/neurons/izhikevich.py`, `sensoryforge/core/generalized_pipeline.py`, others  
**Rule violated:** "Docstrings must be Google Style with Args/Returns/Examples"  
**Effort:** Medium (1–2 days)

Violations found:

- Some files use numpy-style (`Parameters`, `Returns`) instead of Google-style (`Args:`, `Returns:`)
- Many `forward()` methods omit `Returns:` entirely
- Documented return shapes are wrong — batch dimension absent (e.g., docstring says `[time_steps, num_sa_neurons]` but actual shape is `[batch, time_steps, num_neurons]`)
- Physical units missing from many `Args:` and `Returns:` lines

**Fix:**  
Priority-1 sweep on base classes and all public `forward()` methods. Add `pydocstyle --convention=google` as a linting step. Correct all shape annotations to include the batch dimension.

---

### M-4: Assertions used in production code instead of explicit validation

**Files:** `sensoryforge/neurons/sa.py:136` and others  
**Rule violated:** "Fail fast with informative messages" — copilot instructions show `raise ValueError(...)` as the canonical pattern  
**Effort:** Small (1 hour)

```python
# Current — broken with python -O
assert input_current.dim() == 3, "input_current must be [B,T,F]"
```

Assertions are stripped with `python -O`. In optimised mode these checks silently disappear, allowing malformed tensors to propagate and produce cryptic downstream errors.

**Fix:**
```python
if input_current.dim() != 3:
    raise ValueError(
        f"input_current must be 3-D [batch, time, features], got shape {list(input_current.shape)}"
    )
```

Replace all `assert` guards in `sensoryforge/` with explicit `if ... raise`.

---

### M-5: `to_dict()` missing from base classes and most implementations

**Files:** `BaseFilter`, `BaseNeuron`, most concrete subclasses  
**Rule violated:** "Provide `to_dict()` method for serialization" — required on all base classes  
**Effort:** Medium (1 day)

The copilot instructions list `to_dict()` as a mandatory base-class method. It is absent. The GUI currently re-implements ad hoc serialization in each tab because there is no unified serialisation contract, creating duplicated logic and divergence risk.

**Fix:**  
1. Add `to_dict()` as an `@abstractmethod` to `BaseFilter` and `BaseNeuron`
2. Implement in every concrete subclass
3. Simplify the GUI's serialisation paths to call `component.to_dict()`

---

### M-6: Bare `except Exception` throughout codebase

**Files:** `sensoryforge/neurons/izhikevich.py:111`, `sensoryforge/gui/main.py` (multiple), `sensoryforge/cli.py` (multiple)  
**Rule violated:** Fail-fast philosophy; bare catches mask bugs and swallow tracebacks  
**Effort:** Small (1–2 hours)

**Example:**
```python
try:
    t_val = torch.tensor(val, dtype=dtype, device=device)
except Exception:   # swallows everything
    t_val = torch.tensor(float(val), dtype=dtype, device=device)
```

**Fix:**  
Replace with specific exception types (`TypeError`, `ValueError`, `RuntimeError`). In the CLI, add a `--verbose` / `--debug` flag that re-raises with full traceback for debugging sessions.

---

### M-7: `from_config()` missing or incomplete in several classes

**Files:** `sensoryforge/stimuli/texture.py`, some filter subclasses  
**Rule violated:** "All components must be constructible from YAML config via `from_config()`"  
**Effort:** Small (1 hour per class, ~1 day total)

Some stimuli and filter subclasses lack `from_config()` entirely. These components cannot be used in YAML-driven pipelines, breaking one of the framework's core workflows.

**Fix:**  
Audit every subclass of `BaseFilter`, `BaseNeuron`, and `BaseStimulus`. Add `from_config()` classmethods where missing. Add a unit test for each that round-trips through YAML → `from_config()` → `to_dict()`.

---

## Medium Issues

> Fix when convenient. Standards violations or edge-case bugs that don't affect everyday usage but will cause problems at scale or for new contributors.

---

### Med-1: `CompositeGrid.add_population()` accepts deprecated `filter` parameter

**File:** `sensoryforge/core/composite_grid.py`  
**Effort:** Small

The `filter` parameter is deprecated and silently ignored, but the method still accepts it without warning. Tests emit `DeprecationWarning` noise. Confusing for new contributors who will find the parameter in old examples and wonder why it has no effect.

**Fix:**  
Either remove the parameter outright (if no callers remain) or emit an explicit `DeprecationWarning` with a migration message. Update all call sites in tests and examples.

---

### Med-2: Potential crash in blue noise generation for very small grids

**File:** `sensoryforge/core/grid.py` (~line 159)  
**Effort:** Small

```python
k = min(6, points.shape[0] - 1)   # → k=0 if only 1 point
_, nearest_idx = torch.topk(dists, k + 1, largest=False, dim=1)
```

`k=0` makes `torch.topk` with `k+1=1` work, but `k=-1` (zero points) would crash. More importantly there is no user-facing error for grids that are pathologically small.

**Fix:**
```python
if points.shape[0] < 2:
    raise ValueError(f"Grid must contain at least 2 points for blue-noise relaxation, got {points.shape[0]}")
k = max(1, min(6, points.shape[0] - 1))
```

---

### Med-3: README and docs show wrong tensor shapes for pipeline outputs

**Files:** `README.md`, `docs/getting_started/first_simulation.md`  
**Effort:** Small (1 hour)

```python
# Documented as:
sa_spikes = results['sa_spikes']  # [time_steps, num_sa_neurons]

# Actual shape:
# [batch, time_steps, num_sa_neurons]
```

Any user copying the example code and writing `sa_spikes.shape[0]` expecting `time_steps` will get `batch` and produce silently wrong downstream analysis.

**Fix:**  
Correct all shape annotations in docs to include the batch dimension. Add a squeeze example for the single-stimulus case.

---

### Med-4: No end-to-end DSL tutorial despite it being a documented feature

**File:** `docs/tutorials/custom_neurons.md` (sparse)  
**Rule violated:** "Every new feature requires a tutorial demonstrating usage"  
**Effort:** Medium (1 day, after C-2 is resolved)

The DSL is a Phase 2 feature. `custom_neurons.md` exists but does not show a complete working example: define equations → `compile()` → run through pipeline → inspect spike trains.

**Fix:**  
After C-2 lands, write a complete tutorial with a working Izhikevich example defined via DSL, run on a small composite grid, with a plot of the output raster.

---

### Med-5: Dense distance matrix in innervation won't scale to large grids

**File:** `sensoryforge/core/innervation.py` (~line 203)  
**Effort:** Medium (1–2 days)

```python
receptor_exp = self.receptor_coords.unsqueeze(0)   # [1, N_receptors, 2]
neuron_exp   = self.neuron_centers.unsqueeze(1)    # [N_neurons, 1, 2]
d2 = ((receptor_exp - neuron_exp) ** 2).sum(-1)   # [N_neurons, N_receptors]
```

At 1000×1000 receptors + 500 neurons: 500M float32 elements ≈ 2 GB. Silently exhausts RAM or VRAM with no useful error.

**Fix (short-term):**  
Add a size check that warns when the matrix would exceed a configurable threshold (default 512 MB):
```python
matrix_bytes = N_neurons * N_receptors * 4
if matrix_bytes > self.max_matrix_bytes:
    warnings.warn(f"Distance matrix will be {matrix_bytes/1e9:.1f} GB — consider reducing grid size")
```

**Fix (long-term):**  
Replace with a sparse k-NN lookup (e.g., `torch.cdist` in chunks or a scipy `cKDTree` for the index, PyTorch for the weights).

---

### Med-6: Default parameter values scattered across source — should be centralised

**Files:** `sensoryforge/neurons/izhikevich.py`, `sensoryforge/filters/sa_ra.py`, `sensoryforge/core/generalized_pipeline.py`  
**Effort:** Medium (1 day)

Izhikevich defaults (`a=0.02, b=0.2, c=-65.0, d=8.0`), filter time constants, pipeline defaults — all hardcoded in source. Users must read source to discover defaults. Changes require modifying code, not config.

**Fix:**  
Centralise in `sensoryforge/config/default_config.yml` (or per-module YAML files). Load at import time. This also enables predefined neuron presets (e.g., `regular_spiking`, `fast_spiking`, `bursting`) as named entries in the defaults file.

---

## Low / Backlog

> Nice to have. Clean these up incrementally.

---

### L-1: Missing `__all__` in several submodules

**Files:** `sensoryforge/filters/__init__.py` and others

Without `__all__`, the public API surface is ambiguous and IDE autocomplete is noisy. Add explicit `__all__` lists to all public modules.

---

### L-2: Undocumented in-place tensor operations in forward passes

**Files:** Various

Copilot instructions: "Avoid in-place operations unless performance-critical (and document why)." A few forward passes use `x += ...` which can silently break autograd. Audit and either remove or add an explanatory comment.

---

### L-3: CLI has no `--verbose` / `--debug` flag

**File:** `sensoryforge/cli.py`

All errors are reported as a single line (`Error: <message>`), discarding the traceback. Add `--verbose` that prints the full exception with traceback.

---

### L-4: Edge-case tests missing for grid and neuron boundary conditions

**Files:** `tests/unit/test_grid.py`, `tests/unit/test_neurons.py`

Rule: "Edge cases must be tested (empty inputs, extreme values, errors)." Missing:
- 1×1 grid
- Very large spacing (>1 mm)
- Negative spacing (should raise)
- NaN/Inf input to neuron models

---

### L-5: `CHANGELOG.md` does not exist

The release checklist in copilot instructions requires a `CHANGELOG.md`. It is absent. Create it and populate retroactively from git history for the major milestones.

---

### L-6: Unused import in `register_components.py`

**File:** `sensoryforge/register_components.py` (~line 54)

```python
from sensoryforge.stimuli.moving import MovingStimulus as MovingStimulusLegacy
```

This alias is imported but never referenced in the registration logic. Dead code creates confusion about whether the legacy class is intentionally preserved.

**Fix:** Remove the import, or if it is kept for backward compatibility, document why explicitly.

---

### L-7: Numerical stability tests missing for long simulations

**Files:** `tests/unit/test_neurons.py`

There are no tests verifying that neuron models remain numerically stable over long simulations (e.g., 10,000 time steps), or that they recover gracefully from NaN/Inf inputs. Euler integration on stiff systems (AdEx) can diverge silently.

**Fix:** Add parametrised long-run tests with stability assertions (`assert not torch.isnan(v).any()`). Add tests that feed NaN/Inf input and expect a clean `RuntimeError`, not silent propagation.

---

### L-8: No tests for out-of-memory / very large grid conditions

**Files:** `tests/unit/test_grid.py`, `tests/unit/test_innervation.py`

No test verifies the warning emitted when the distance matrix would exceed the memory threshold (Med-5). Without a test, the warning can be silently removed or broken without CI catching it.

**Fix:** Add a test that constructs a large grid and asserts a `UserWarning` is raised, using `pytest.warns(UserWarning)`.

---

## Documentation Issues

> These are places where the written documentation is wrong or missing relative to what the code actually does.

---

### DOC-1: Configuration schema dataclass fields have no per-field documentation

**File:** `sensoryforge/config/schema.py` (~lines 25–249)

`PopulationConfig`, `GridConfig`, and related dataclasses have many fields with no explanation of valid values, units, or constraints.

**Example:**
```python
@dataclass
class PopulationConfig:
    neuron_arrangement: str = "grid"   # What are the valid strings? "grid", "poisson", "hex"?
    sigma_d_mm: float = 1.0            # Units documented nowhere in the class
```

Without per-field docstrings, users must trace the validation logic to discover what values are legal. This violates the rule that "every public API must have comprehensive docstrings with examples."

**Fix:** Add an inline docstring or Attributes block to each dataclass describing valid values, units, and defaults. Consider using `dataclasses.field(metadata=...)` to make constraints machine-readable.

---

### DOC-2: `compute_weights()` `**kwargs` undocumented in GaussianInnervation

**File:** `sensoryforge/core/innervation.py` (~line 191)

```python
def compute_weights(self, **kwargs) -> torch.Tensor:
    """Compute Gaussian-weighted random connections..."""
    # No documentation of what kwargs are expected or valid
```

A caller has no way to discover what keyword arguments are accepted without reading the implementation.

**Fix:** Replace `**kwargs` with explicit typed parameters, or add a dedicated `Kwargs:` block to the docstring listing every accepted key with type, description, and default.

---

### DOC-3: `copilot-instructions.md` `CompositeGrid` example uses the deprecated `filter` parameter

**File:** `.github/copilot-instructions.md` (~line 139)

```python
grid = CompositeGrid(
    shape=(64, 64),
    populations={
        'L_cone': {'density': 0.60, 'filter': 'long_wave'},   # 'filter' is deprecated!
```

The instructions that guide all contributors use the very parameter that emits `DeprecationWarning`. Every new contributor following this example will write deprecated code on their first day.

**Fix:** Update the example in `copilot-instructions.md` to use the `add_layer()` API. This should be done in the same PR that removes the deprecated parameter (Med-1).

---

## Code Quality Issues

> These are not correctness bugs but make the code harder to maintain, review, or extend.

---

### QUA-1: Complex nested conditionals in `ReceptorGrid.__init__`

**File:** `sensoryforge/core/grid.py` (~lines 89–210)

The `__init__` method has deeply nested `if/elif` chains handling every arrangement type inline. The method is long enough that the top-level structure is hard to follow at a glance.

**Current pattern:**
```python
def __init__(self, ...):
    if arrangement in ["grid", "jittered_grid"]:
        # 30 lines
        if jitter:
            # 15 more lines
    elif arrangement == "poisson":
        # 25 lines
    elif arrangement == "hex":
        # 20 lines
    # ... etc
```

**Fix:** Extract arrangement-specific logic into private factory methods:
```python
def __init__(self, ...):
    self._init_common(...)
    if arrangement in ["grid", "jittered_grid"]:
        self._init_regular_grid(jitter=arrangement == "jittered_grid")
    elif arrangement == "poisson":
        self._init_poisson()
    elif arrangement == "hex":
        self._init_hex()
    else:
        raise ValueError(f"Unknown arrangement: {arrangement!r}")
```

This makes `__init__` a dispatcher, each sub-method testable in isolation.

---

### QUA-2: Hard cutoff in Gaussian innervation connection weights

**File:** `sensoryforge/core/innervation.py` (~lines 214–216)

```python
if self.max_sigma_distance > 0:
    max_dist = self.max_sigma_distance * self.sigma_d_mm
    gaussian_weights[distances > max_dist] = 0.0   # hard step cutoff
```

A hard binary cutoff means receptors just inside the threshold have full Gaussian weight, receptors just outside have zero weight. This creates a discontinuity in the receptive field boundary that is biologically implausible and can cause edge artefacts in spatial responses.

**Fix (optional):** Replace with a smooth Hann or cosine taper over the final 10% of the cutoff radius so weights fall to zero continuously:
```python
taper_start = 0.9 * max_dist
in_taper = (distances > taper_start) & (distances <= max_dist)
taper_factor = 0.5 * (1 + torch.cos(
    torch.pi * (distances[in_taper] - taper_start) / (max_dist - taper_start)
))
gaussian_weights[in_taper] *= taper_factor
gaussian_weights[distances > max_dist] = 0.0
```

---

## Security Considerations

> Low risk in current usage (no network exposure, no multi-tenant environment), but worth fixing as good practice before any open-source or web-facing deployment.

---

### SEC-1: No input size or depth validation on YAML loading

**File:** `sensoryforge/config/yaml_utils.py` (~lines 31–43)

`load_yaml()` performs duplicate-key detection but does not validate:
- File size — a 500 MB YAML file would be parsed into memory without warning
- Maximum nesting depth — a pathologically deep config could cause a stack overflow in the parser
- Anchor/alias bombs — YAML allows exponential expansion via nested aliases (`&anchor [*anchor, *anchor]`)

**Impact:** Low risk today (users supply their own configs). Becomes relevant if the GUI ever accepts configs from untrusted sources (web upload, shared experiment links).

**Fix:**
```python
def load_yaml(stream, max_size_bytes: int = 10 * 1024 * 1024):
    content = stream.read() if hasattr(stream, 'read') else open(stream, 'rb').read()
    if len(content) > max_size_bytes:
        raise ValueError(
            f"Config file too large ({len(content) / 1e6:.1f} MB). "
            f"Maximum allowed: {max_size_bytes / 1e6:.1f} MB"
        )
    return yaml.safe_load(content)  # safe_load already disables Python-object tags
```

---

### SEC-2: Potential path traversal in batch executor output directory

**File:** `sensoryforge/core/batch_executor.py` (~lines 92–94)

```python
output_dir = self.batch_config.get('output_dir', './batch_results')
self.output_dir = Path(output_dir)
```

If `output_dir` is sourced from a user-supplied YAML and contains `../` sequences, batch results could be written outside the intended directory. This becomes a problem if configs are ever shared or loaded from untrusted sources.

**Fix:**
```python
output_dir = Path(output_dir).resolve()
base_dir = Path('.').resolve()
if not str(output_dir).startswith(str(base_dir)):
    raise ValueError(
        f"output_dir '{output_dir}' must be within the working directory '{base_dir}'. "
        f"Absolute or traversal paths are not permitted."
    )
self.output_dir = output_dir
```

---

## Performance Issues

> Not blockers for correctness, but will become bottlenecks as grid sizes and batch sizes grow.

---

### PERF-1: Dense distance matrix in innervation doesn't scale

*(See Med-5 above — listed here for cross-reference.)*

**File:** `sensoryforge/core/innervation.py` (~line 203)  
At large grid sizes this creates a multi-gigabyte dense matrix. Short-term: add a size warning. Long-term: replace with sparse k-NN.

---

### PERF-2: Redundant `.to(device)` calls inside every forward pass

**Files:** Multiple — filters, neuron models

```python
# Called every forward pass, even when device hasn't changed
self.x = self.x.to(device)
self.y = self.y.to(device)
```

`.to(device)` is a no-op when the tensor is already on the correct device, but it still incurs a Python call and a device check on every step. In a 10,000-step simulation with 100 neurons this adds up.

**Fix:** Cache the device at state initialisation and skip the transfer if already correct:
```python
def _ensure_state(self, batch_size, num_neurons, device):
    if self._state_device != device or self._state_batch != batch_size:
        self._init_state(batch_size, num_neurons, device)
        self._state_device = device
        self._state_batch = batch_size
```

---

### PERF-3: Memory inefficiency in Poisson grid generation

**File:** `sensoryforge/core/grid.py` (~lines 340–373)

Intermediate jitter tensors are created without cleanup:
```python
jitter = (torch.rand_like(coordinates) - 0.5) * jitter_scale
coordinates = coordinates + jitter   # allocates a new tensor, old one GC'd lazily
```

For typical grid sizes (80×80 = 6400 points) this is negligible. For 1000×1000 it creates unnecessary GC pressure.

**Fix (only if profiling shows this as a bottleneck):** Use in-place addition:
```python
coordinates.add_((torch.rand_like(coordinates) - 0.5) * jitter_scale)
```

Note: only do this if `coordinates` is not needed for autograd.

---

## Testing Gaps

> Issues with test coverage beyond the edge cases listed in L-4/L-7/L-8.

---

### TEST-1: No round-trip YAML tests for `from_config()` / `to_dict()`

Once M-5 and M-7 are fixed (adding `to_dict()` and `from_config()` everywhere), there are no round-trip tests that verify:
```
YAML file → from_config() → to_dict() → write YAML → reload → compare
```
Without these, silent serialisation drift will go undetected.

**Fix:** Add parametrised round-trip tests for every public component in `tests/integration/test_yaml_roundtrip.py`.

---

### TEST-2: No tests covering the GUI↔pipeline YAML sync

**Files:** `tests/` (missing)

The GUI generates YAML from its form fields and hands it to the pipeline. There are no tests verifying that the YAML the GUI emits is valid for the pipeline to consume. Breakage only shows up at runtime.

**Fix:** Add headless GUI tests (using `QTest` or by extracting the YAML-generation logic into a testable pure function) that verify the emitted YAML passes `validate_config()`.

---

### TEST-3: Solver accuracy not validated against known analytical solutions

**Files:** `tests/unit/test_solvers.py`

The Euler and adaptive solvers are tested for shape and basic execution, but not for numerical accuracy against a system with a known analytical solution (e.g., a linear ODE). Without this, a broken solver that produces wrong values but correct shapes will pass all tests.

**Fix:** Add a test integrating `dy/dt = -y, y(0) = 1` (solution: `y(t) = e^{-t}`) and asserting the numerical solution stays within a known error bound for each solver.

---

## Summary Table

| ID | Severity | File(s) | Rule Violated | Effort |
|----|----------|---------|---------------|--------|
| C-1 | **Critical** | `core/simulation_engine.py` | Documented feature crashes at runtime | M |
| C-2 | **Critical** | `neurons/model_dsl.py` | PyTorch-native + differentiable mandate | L |
| M-1 | Major | `filters/sa_ra.py` | `reset_state()` interface contract | S |
| M-2 | Major | Multiple | "Type hints mandatory" | M |
| M-3 | Major | Multiple | Google Style docstring mandate + shape accuracy | M |
| M-4 | Major | `neurons/sa.py` + others | Fail fast with `ValueError`, not `assert` | S |
| M-5 | Major | Multiple base classes | `to_dict()` required on all base classes | M |
| M-6 | Major | Multiple | Bare `except Exception` | S |
| M-7 | Major | Stimuli, filters | `from_config()` required on all classes | S |
| Med-1 | Medium | `core/composite_grid.py` | API cleanliness / deprecation | S |
| Med-2 | Medium | `core/grid.py` | Edge-case robustness | S |
| Med-3 | Medium | `README.md`, `docs/` | Wrong tensor shapes in examples | S |
| Med-4 | Medium | `docs/tutorials/` | "Every feature needs a tutorial" | M |
| Med-5 | Medium | `core/innervation.py` | Scalability / silent OOM | M |
| Med-6 | Medium | Multiple | Config centralisation | M |
| L-1 | Low | `filters/__init__.py` + others | API surface clarity | S |
| L-2 | Low | Various | Undocumented in-place ops | S |
| L-3 | Low | `cli.py` | Developer ergonomics | S |
| L-4 | Low | `tests/unit/` | Edge-case test coverage | S |
| L-5 | Low | (missing) | Release checklist: CHANGELOG | S |
| L-6 | Low | `register_components.py` | Dead code / unused import | S |
| L-7 | Low | `tests/unit/test_neurons.py` | Numerical stability tests missing | S |
| L-8 | Low | `tests/unit/test_grid.py` | OOM warning test missing | S |
| DOC-1 | Docs | `config/schema.py` | Per-field dataclass documentation | S |
| DOC-2 | Docs | `core/innervation.py` | `**kwargs` undocumented | S |
| DOC-3 | Docs | `.github/copilot-instructions.md` | Example uses deprecated API | S |
| QUA-1 | Quality | `core/grid.py` | Complex nested conditionals | M |
| QUA-2 | Quality | `core/innervation.py` | Hard cutoff discontinuity | S |
| SEC-1 | Security | `config/yaml_utils.py` | No YAML size/depth validation | S |
| SEC-2 | Security | `core/batch_executor.py` | Path traversal in output_dir | S |
| PERF-1 | Perf | `core/innervation.py` | Dense matrix OOM (see Med-5) | M |
| PERF-2 | Perf | Filters, neurons | Redundant `.to(device)` per step | S |
| PERF-3 | Perf | `core/grid.py` | Intermediate tensor allocation | S |
| TEST-1 | Testing | `tests/integration/` | No YAML round-trip tests | S |
| TEST-2 | Testing | `tests/` | No GUI↔pipeline YAML sync tests | M |
| TEST-3 | Testing | `tests/unit/test_solvers.py` | Solver accuracy not validated | S |

**Effort key:** S = Small (<2 hrs) · M = Medium (half–1 day) · L = Large (3–5 days)

| ID | Severity | File(s) | Rule Violated | Effort |
|----|----------|---------|---------------|--------|
| C-1 | **Critical** | `core/simulation_engine.py` | Documented feature crashes at runtime | M |
| C-2 | **Critical** | `neurons/model_dsl.py` | PyTorch-native + differentiable mandate | L |
| M-1 | Major | `filters/sa_ra.py` | `reset_state()` interface contract | S |
| M-2 | Major | Multiple | "Type hints mandatory" | M |
| M-3 | Major | Multiple | Google Style docstring mandate + shape accuracy | M |
| M-4 | Major | `neurons/sa.py` + others | Fail fast with `ValueError`, not `assert` | S |
| M-5 | Major | Multiple base classes | `to_dict()` required on all base classes | M |
| M-6 | Major | Multiple | Bare `except Exception` | S |
| M-7 | Major | Stimuli, filters | `from_config()` required on all classes | S |
| Med-1 | Medium | `core/composite_grid.py` | API cleanliness / deprecation | S |
| Med-2 | Medium | `core/grid.py` | Edge-case robustness | S |
| Med-3 | Medium | `README.md`, `docs/` | Docstring accuracy (wrong shapes) | S |
| Med-4 | Medium | `docs/tutorials/` | "Every feature needs a tutorial" | M |
| Med-5 | Medium | `core/innervation.py` | Scalability / silent OOM | M |
| Med-6 | Medium | Multiple | Config centralisation | M |
| L-1 | Low | `filters/__init__.py` + others | API surface clarity | S |
| L-2 | Low | Various | In-place op documentation | S |
| L-3 | Low | `cli.py` | Developer ergonomics | S |
| L-4 | Low | `tests/unit/` | Edge-case test coverage | S |
| L-5 | Low | (missing) | Release checklist compliance | S |

**Effort key:** S = Small (<2 hrs) · M = Medium (half–1 day) · L = Large (3–5 days)

---

## Recommended Mitigation Order

### Sprint 1 — Correctness & Interface Contract

| # | Task | ID | Effort |
|---|------|----|--------|
| 1 | Rename `reset_states()` → `reset_state()` | M-1 | S |
| 2 | Replace all `assert` with `raise ValueError` | M-4 | S |
| 3 | Implement `to_dict()` on base classes + subclasses | M-5 | M |
| 4 | Audit and fill in `from_config()` on all classes | M-7 | S–M |
| 5 | Implement CompositeGrid in SimulationEngine | C-1 | M |
| 6 | Fix path traversal in batch executor output_dir | SEC-2 | S |

### Sprint 2 — Standards Compliance

| # | Task | ID | Effort |
|---|------|----|--------|
| 7 | Type hints sweep + mypy CI gate | M-2 | M |
| 8 | Docstring sweep (Google style, units, shapes) + pydocstyle gate | M-3 | M |
| 9 | Add per-field docs to config schema dataclasses | DOC-1 | S |
| 10 | Replace bare `except Exception` with specific types | M-6 | S |
| 11 | Fix README/docs tensor shape annotations | Med-3 | S |
| 12 | Update copilot-instructions.md example to use `add_layer()` | DOC-3 | S |
| 13 | Remove/warn deprecated `filter` param in `add_population()` | Med-1 | S |
| 14 | Add blue noise guard for tiny grids | Med-2 | S |
| 15 | Add `--verbose` flag to CLI | L-3 | S |
| 16 | Add YAML size validation to `load_yaml()` | SEC-1 | S |
| 17 | Remove unused import in `register_components.py` | L-6 | S |

### Sprint 3 — Feature Completion & Tech Debt

| # | Task | ID | Effort |
|---|------|----|--------|
| 18 | Rewrite DSL code generation to use PyTorch ops | C-2 | L |
| 19 | Write end-to-end DSL tutorial | Med-4 | M |
| 20 | Add OOM guard + sparse fallback for innervation | Med-5 / PERF-1 | M |
| 21 | Centralise magic defaults to `default_config.yml` | Med-6 | M |
| 22 | Cache device in filter/neuron state to avoid redundant `.to()` | PERF-2 | S |
| 23 | Refactor `ReceptorGrid.__init__` into dispatcher + sub-methods | QUA-1 | M |
| 24 | Replace hard cutoff with smooth taper in innervation | QUA-2 | S |

### Sprint 4 — Testing & Documentation

| # | Task | ID | Effort |
|---|------|----|--------|
| 25 | Add YAML round-trip integration tests | TEST-1 | S |
| 26 | Add headless GUI↔pipeline YAML sync tests | TEST-2 | M |
| 27 | Add solver accuracy tests vs analytical solution | TEST-3 | S |
| 28 | Add edge-case tests (1×1 grid, NaN inputs, etc.) | L-4 | S |
| 29 | Add numerical stability tests for long simulations | L-7 | S |
| 30 | Add OOM warning test | L-8 | S |
| 31 | Add `__all__` to all public modules | L-1 | S |
| 32 | Audit and document in-place tensor ops | L-2 | S |
| 33 | Document `**kwargs` in `compute_weights()` | DOC-2 | S |
| 34 | Create `CHANGELOG.md` from git history | L-5 | S |

---

## Issue Status Tracker

Use this table to track progress as issues are resolved. Update `Status` and `Resolved In` as work lands.

| ID | Status | Resolved In |
|----|--------|-------------|
| C-1 | Open | — |
| C-2 | Open | — |
| M-1 | Open | — |
| M-2 | Open | — |
| M-3 | Open | — |
| M-4 | Open | — |
| M-5 | Open | — |
| M-6 | Open | — |
| M-7 | Open | — |
| Med-1 | Open | — |
| Med-2 | Open | — |
| Med-3 | Open | — |
| Med-4 | Open | — |
| Med-5 | Open | — |
| Med-6 | Open | — |
| L-1 | Open | — |
| L-2 | Open | — |
| L-3 | Open | — |
| L-4 | Open | — |
| L-5 | Open | — |
| L-6 | Open | — |
| L-7 | Open | — |
| L-8 | Open | — |
| DOC-1 | Open | — |
| DOC-2 | Open | — |
| DOC-3 | Open | — |
| QUA-1 | Open | — |
| QUA-2 | Open | — |
| SEC-1 | Open | — |
| SEC-2 | Open | — |
| PERF-1 | Open | — |
| PERF-2 | Open | — |
| PERF-3 | Open | — |
| TEST-1 | Open | — |
| TEST-2 | Open | — |
| TEST-3 | Open | — |

---

*Generated by full codebase review on 2026-04-08. Update this file as issues are resolved.*
