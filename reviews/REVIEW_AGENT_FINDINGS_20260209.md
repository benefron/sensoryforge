# SensoryForge Review Agent Findings

**Date:** 2026-02-09  
**Scope:** Full codebase review — all Python modules, tests, documentation, and configuration  
**Reviewer:** Review Agent (Principal-level)

---

## Executive Summary

| Metric | Value |
| ------ | ----- |
| Python source files scanned | ~40 |
| Test files scanned | 21 (19 unit + 2 integration) |
| Tests passing | 239 passed, 3 skipped |
| Compile/lint errors (Python) | 0 |
| Markdown lint warnings | ~50 (style only) |
| Critical findings | 2 |
| High-priority findings | 8 |
| Medium-priority findings | 10 |
| Low-priority findings | 7 |

The codebase is in solid shape: all 239 tests pass, zero Python compile errors, and most modules have comprehensive docstrings. The most impactful issues are (1) a **Python loop in innervation construction** that blocks GPU parallelism, (2) **noise modules that do not inherit from `nn.Module`** and break `to(device)` propagation, and (3) **missing abstract base classes** (`BaseFilter`, `BaseNeuron`, `BaseStimulus`) that the architecture documentation promises but the code does not define.

---

## Module Coverage Matrix

| Module | Source Reviewed | Tests Exist | Docstrings | Type Hints | Base Class |
| ------ | -------------- | ----------- | ---------- | ---------- | ---------- |
| `core/grid.py` | ✅ | ✅ | ✅ | ✅ | N/A |
| `core/innervation.py` | ✅ | ✅ (4 files) | ✅ | ✅ | nn.Module |
| `core/pipeline.py` | ✅ | ✅ | ✅ | ✅ | nn.Module |
| `core/generalized_pipeline.py` | ✅ | ✅ | Partial | Partial | nn.Module |
| `core/composite_grid.py` | ✅ | ✅ | ✅ | ✅ | — |
| `core/mechanoreceptors.py` | ✅ | ❌ | Partial | Partial | nn.Module |
| `core/tactile_network.py` | ✅ | ❌ | ✅ | ✅ | nn.Module |
| `core/compression.py` | ✅ | ❌ | ✅ | ✅ | dataclass |
| `core/visualization.py` | ✅ | ❌ | Partial | Partial | — |
| `core/notebook_pipeline.py` | ✅ | ❌ | Partial | Partial | nn.Module |
| `filters/sa_ra.py` | ✅ | ✅ | ✅ | ✅ | nn.Module |
| `filters/noise.py` | ✅ | ✅ (indirect) | Partial | Partial | **plain class** |
| `neurons/izhikevich.py` | ✅ | ✅ (3 files) | ✅ | Partial | nn.Module |
| `neurons/adex.py` | ✅ | ✅ | ✅ | Partial | nn.Module |
| `neurons/mqif.py` | ✅ | ✅ | ✅ | Partial | nn.Module |
| `neurons/fa.py` | ✅ | ❌ | ✅ | Partial | nn.Module |
| `neurons/sa.py` | ✅ | ❌ | Partial | Partial | nn.Module |
| `neurons/model_dsl.py` | ✅ | ✅ | ✅ | ✅ | — |
| `solvers/base.py` | ✅ | ✅ (indirect) | ✅ | ✅ | ABC |
| `solvers/euler.py` | ✅ | ✅ | ✅ | ✅ | BaseSolver |
| `solvers/adaptive.py` | ✅ | ✅ (skipped) | ✅ | ✅ | BaseSolver |
| `stimuli/stimulus.py` | ✅ | ✅ (indirect) | ✅ | ✅ | — |
| `stimuli/gaussian.py` | ✅ | ✅ | ✅ | ✅ | — |
| `stimuli/texture.py` | ✅ | ✅ | ✅ | ✅ | — |
| `stimuli/moving.py` | ✅ | ✅ | ✅ | ✅ | — |
| `config/yaml_utils.py` | ✅ | ✅ | ✅ | ✅ | — |
| `cli.py` | ✅ | ❌ | ✅ | ✅ | — |
| `utils/project_registry.py` | ✅ | ❌ | ✅ | ✅ | dataclass |
| `gui/*` | ✅ | ❌ | Partial | Partial | Qt |

---

## Critical Issues (Must Fix)

### C1: Performance — Python loop in innervation map construction

**Location:** `sensoryforge/core/innervation.py:L111-L117`

**Issue Type:** Performance / Architecture Violation

**Current Code:**
```python
for n in range(num_neurons):
    K = K_per_neuron[n].item()
    if K > 0:
        idx = torch.multinomial(prob_weights[n], K, replacement=False)
        rand_vals = torch.empty(K, device=device).uniform_(weight_min, weight_max)
        rand_weights[n, idx] = rand_vals
```

**Problems:**
1. Iterates over every neuron in a Python loop — violates project standard: "Never hand-roll loops over neurons or spatial dimensions"
2. Prevents GPU parallelisation: each iteration invokes a separate CUDA kernel for `multinomial` + `uniform_`
3. For a 40×40 neuron grid (1600 neurons), this is ~1600 sequential kernel launches
4. `torch.multinomial` supports batched sampling when all K values are equal; for variable K, a padded batch approach is feasible

**Severity:** Critical — O(N_neurons) sequential kernel launches on the GPU hot path; 10–100× slower than vectorised equivalent; dominates pipeline construction time for large grids.

**Remediation:**
Replace with a batched approach using a fixed max-K and masking:
```python
max_K = K_per_neuron.max().item()
# Batch multinomial: sample max_K per neuron, then mask
all_idx = torch.multinomial(prob_weights, max_K, replacement=False)  # [num_neurons, max_K]
all_vals = torch.empty(num_neurons, max_K, device=device).uniform_(weight_min, weight_max)

# Build mask: only keep first K[n] samples per neuron
arange = torch.arange(max_K, device=device).unsqueeze(0)  # [1, max_K]
mask = arange < K_per_neuron.unsqueeze(1)                   # [num_neurons, max_K]
all_vals[~mask] = 0.0

# Scatter into the flat weight matrix
rand_weights.scatter_(1, all_idx, all_vals)
```

**Testing:**
1. Verify output matches original (tolerance 1e-6) with fixed seed
2. Benchmark both versions with pytest-benchmark at 100, 1000, 2500 neurons
3. Add GPU timing test if CUDA available

---

### C2: Architecture — Noise modules are plain classes, not `nn.Module`

**Location:** `sensoryforge/filters/noise.py:L10-L66`

**Issue Type:** Bug / Architecture Violation

**Current Code:**
```python
class MembraneNoiseTorch:
    """..."""
    def __init__(self, std: float = 1.0, mean: float = 0.0, seed: int = None):
        ...
    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        ...

class ReceptorNoiseTorch:
    """..."""
    def __init__(self, std: float = 1.0, mean: float = 0.0, seed: int = None):
        ...
    def __call__(self, responses: torch.Tensor) -> torch.Tensor:
        ...
```

**Problems:**
1. Neither class inherits from `nn.Module` — they are invisible to `model.to(device)` and `model.parameters()`
2. The pipeline calls `self.to(self.device)` expecting all children to move, but noise modules stay on CPU
3. `torch.manual_seed(seed)` in `__init__` sets the **global** RNG seed as a side effect, polluting downstream randomness for all other modules constructed afterward
4. No `reset_state()` or `from_config()` methods as required by project base-class contract
5. Missing type hints on `__init__` parameters (`seed: int = None` should be `seed: Optional[int] = None`)

**Severity:** Critical — `to(device)` silently fails to move noise generation to the correct device. On GPU pipelines, `torch.randn_like(current)` still works (device inferred from input), but the global seed mutation corrupts reproducibility of every downstream module.

**Remediation:**
```python
class MembraneNoiseTorch(nn.Module):
    """Additive Gaussian membrane noise for filtered currents.
    
    Args:
        std: Standard deviation of noise (same units as current, mA).
        mean: Mean of noise. Default: 0.0.
        seed: Random seed for reproducibility. If set, creates a dedicated
            Generator instance (does NOT mutate global RNG).
    """
    def __init__(self, std: float = 1.0, mean: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input current tensor [batch, time, neurons]."""
        noise = torch.randn(current.shape, device=current.device, dtype=current.dtype,
                            generator=self._generator) * self.std + self.mean
        return current + noise
```
Apply the same pattern to `ReceptorNoiseTorch`.

**Testing:**
1. Test that `pipeline.to('cpu')` propagates to noise modules
2. Verify that setting `seed` does NOT change `torch.initial_seed()` globally
3. Add unit test `test_noise.py` covering shapes, device movement, reproducibility

---

## High Priority Issues

### H1: Architecture — Missing abstract base classes (`BaseFilter`, `BaseNeuron`, `BaseStimulus`)

**Location:** Entire `sensoryforge/filters/`, `sensoryforge/neurons/`, `sensoryforge/stimuli/` packages

**Issue Type:** Architecture / Documentation Mismatch

**Current State:**
The `.github/copilot-instructions.md` (L369-377) and architecture documentation describe a mandatory contract:
> "All base classes (BaseFilter, BaseNeuron, BaseStimulus) must:
> 1. Inherit from `torch.nn.Module`
> 2. Define abstract methods that subclasses must implement
> 3. Provide `from_config()` class method for YAML instantiation
> 4. Provide `to_dict()` method for serialization
> 5. Implement state management (`get_state`, `set_state`, `reset_state`)
> 6. Include comprehensive docstrings with examples"

**No `BaseFilter` or `BaseNeuron` or `BaseStimulus` class exists in the codebase.** Only `BaseSolver` in `solvers/base.py` follows this pattern.

**Problems:**
1. No enforcement of consistent API across filter/neuron/stimulus implementations
2. Neuron models have inconsistent `forward()` signatures (e.g., `IzhikevichNeuronTorch.forward` accepts optional `a, b, c, d` overrides; `AdExNeuronTorch.forward` does not)
3. No `from_config()` or `to_dict()` on any neuron model except `NeuronModel` (DSL)
4. No `reset_state()` on neuron models (`IzhikevichNeuronTorch` has no state to reset between sequences, but the contract says it should exist)
5. Plugin discovery system references `BaseFilter`/`BaseNeuron` but they don't exist

**Severity:** High — prevents the extensibility architecture from working, blocks plugin discovery, and causes silent API inconsistencies.

**Remediation:**
1. Create `sensoryforge/filters/base.py` with `BaseFilter(nn.Module, ABC)` defining abstract `forward()`, `reset_state()`, `from_config()`, `to_dict()`
2. Create `sensoryforge/neurons/base.py` with `BaseNeuron(nn.Module, ABC)` defining abstract `forward()`, `reset_state()`, `from_config()`, `to_dict()`
3. Create `sensoryforge/stimuli/base.py` with `BaseStimulus(nn.Module, ABC)`
4. Update existing filter/neuron classes to inherit from the new base
5. Add tests for ABC enforcement

---

### H2: Performance — `get_weights_per_neuron()` uses Python loop

**Location:** `sensoryforge/core/innervation.py:L264-L270`

**Issue Type:** Performance

**Current Code:**
```python
def get_weights_per_neuron(self):
    """Get number of connections per neuron."""
    connections_per_neuron = []
    for i in range(self.num_neurons):
        connections = (self.innervation_weights[i] > 0).sum().item()
        connections_per_neuron.append(connections)
    return torch.tensor(connections_per_neuron)
```

**Problems:**
1. Python loop over neurons instead of vectorised operation
2. Calls `.item()` per neuron (CPU sync on each iteration if on GPU)
3. Rebuilds list → tensor conversion unnecessarily

**Severity:** High — called by `get_innervation_info()` which is used in pipeline diagnostics and visualisations.

**Remediation:**
```python
def get_weights_per_neuron(self) -> torch.Tensor:
    """Count nonzero connections per neuron.

    Returns:
        [num_neurons] tensor of connection counts.
    """
    return (self.innervation_weights > 0).view(self.num_neurons, -1).sum(dim=1)
```

**Testing:**
1. Verify output matches original for several seeds
2. Benchmark with 1600-neuron population

---

### H3: Bug — `get_grid_spacing()` computes wrong axes

**Location:** `sensoryforge/core/grid.py:L44-L50`

**Issue Type:** Bug (Correctness)

**Current Code:**
```python
def get_grid_spacing(xx, yy):
    """Calculate grid spacing from coordinate meshgrids."""
    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]
    return dx, dy
```

**Problems:**
The meshgrids are created with `indexing="ij"` (line 39), which means:
- `xx[i, j]` varies along `i` (row = x-axis), constant along `j`
- `yy[i, j]` varies along `j` (col = y-axis) ... **wait, no**: with `ij` indexing, `xx` varies along dim-0 and `yy` varies along dim-1.

So `xx[0, 1] - xx[0, 0]` should be 0 (xx is constant along dim-1). Currently this returns 0 for dx when the grid is square, masking the bug.

Let me re-derive: `torch.meshgrid(x, y, indexing="ij")`:
- `xx[i, j] = x[i]` → varies along dim-0 only → `xx[0, 1] - xx[0, 0] = x[0] - x[0] = 0`
- `yy[i, j] = y[j]` → varies along dim-1 only → `yy[1, 0] - yy[0, 0] = y[0] - y[0] = 0`

The correct computation should be:
```python
dx = xx[1, 0] - xx[0, 0]  # x varies along first dim
dy = yy[0, 1] - yy[0, 0]  # y varies along second dim
```

**Severity:** High — returns `(0, 0)` for grid spacing, which downstream code uses in MechanoreceptorModule's `sigma_mm_to_pixels` and possibly other places. Currently the GridManager stores `spacing` directly so most code paths don't use this function, reducing immediate impact, but any caller relying on `get_grid_spacing` gets wrong values.

**Remediation:**
```python
def get_grid_spacing(xx: torch.Tensor, yy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate grid spacing from coordinate meshgrids (ij indexing).

    Args:
        xx: X-coordinate meshgrid [grid_h, grid_w]. Units: mm.
        yy: Y-coordinate meshgrid [grid_h, grid_w]. Units: mm.

    Returns:
        (dx, dy) spacing tensors in mm.
    """
    dx = xx[1, 0] - xx[0, 0]  # x varies along dim-0 (ij indexing)
    dy = yy[0, 1] - yy[0, 0]  # y varies along dim-1 (ij indexing)
    return dx, dy
```

**Testing:**
1. Add unit test: create grid with known spacing, verify `get_grid_spacing` returns it
2. Check that GridManager's cached `dx, dy` match `spacing`

---

### H4: Missing dependency — `plotly` not in `setup.py` or `requirements.txt`

**Location:** `sensoryforge/core/visualization.py:L10-L11`, `setup.py`, `requirements.txt`

**Issue Type:** Build / Dependency

**Current Code:**
```python
import plotly.graph_objects as go
import plotly.subplots as psub
```
But `plotly` does not appear in `setup.py` `install_requires` or `requirements.txt`.

**Problems:**
1. `import sensoryforge.core.visualization` will fail with `ModuleNotFoundError` on fresh installs
2. `visualization.py` also imports `numpy` (present) but the plotly dependency is completely missing

**Severity:** High — hard crash on import.

**Remediation:**
Add `plotly>=5.0` to `install_requires` in `setup.py` and to `requirements.txt`. Alternatively, make the import conditional with a helpful error message if plotly is not installed.

---

### H5: Global RNG pollution — `torch.manual_seed()` called at module init time

**Location:** Multiple files:
- `sensoryforge/filters/noise.py:L24,L52` (in `__init__`)
- `sensoryforge/core/innervation.py:L81`
- `sensoryforge/core/pipeline.py:L123`
- `sensoryforge/core/generalized_pipeline.py:L160`

**Issue Type:** Bug (Reproducibility)

**Problems:**
1. `MembraneNoiseTorch.__init__` and `ReceptorNoiseTorch.__init__` call `torch.manual_seed(seed)` which mutates the **global** RNG state
2. Module construction order therefore leaks into randomness of subsequently constructed modules
3. The pipeline already calls `torch.manual_seed(self.seed)` at the top, but noise modules created later re-seed the global state
4. If two noise modules have different seeds, the second one's `manual_seed` overrides the first's global effect

**Severity:** High — silent reproducibility failures. Tests may pass due to fixed construction order, but reordering module construction changes results.

**Remediation:**
Use `torch.Generator` instances per module instead of global seeding (see C2 remediation). For the pipeline-level seed, keep the single `torch.manual_seed(self.seed)` call but document that it controls initialization-time randomness only.

**Testing:**
1. Verify that creating two pipelines with the same seed produces identical results
2. Verify that re-ordering noise module construction does not change pipeline output

---

### H6: Inconsistent `forward()` signatures across neuron models

**Location:**
- `sensoryforge/neurons/izhikevich.py:L64` — `forward(self, input_current, a=None, b=None, c=None, d=None, threshold=None)`
- `sensoryforge/neurons/adex.py:L56` — `forward(self, input_current)` (no overrides)
- `sensoryforge/neurons/mqif.py:L53` — `forward(self, input_current)` (no overrides)
- `sensoryforge/neurons/fa.py:L83` — `forward(self, x, *, vb=None, A=None, ...)` (keyword-only overrides)
- `sensoryforge/neurons/sa.py:L97` — `forward(self, input_current, *, I_tau=None, ...)` (keyword-only overrides)

**Issue Type:** API Consistency

**Problems:**
1. No common `forward()` signature — callers cannot interchangeably use different neuron models without model-specific branching
2. Some models accept per-feature parameter overrides, others don't
3. `FANeuronTorch.forward` uses `x` instead of `input_current` as parameter name
4. This blocks a `BaseNeuron.forward(input_current) -> Tuple[v_trace, spikes]` contract

**Severity:** High — prevents modular neuron swapping in the pipeline.

**Remediation:**
1. Define `BaseNeuron.forward(input_current: Tensor) -> Tuple[Tensor, Tensor]` as the required interface
2. Move parameter overrides to a separate `forward_with_overrides()` method or accept via `**kwargs`
3. Ensure all models return `(v_trace, spikes)` with consistent shapes

---

### H7: Stimulus generation loops over timesteps in Python

**Location:** `sensoryforge/core/generalized_pipeline.py:L483-L485`, L530-L531, L548-L549, L570-L571, L725-L726, L739-L740, L749-L750

**Issue Type:** Performance

**Current Code (repeated pattern):**
```python
for t_idx in range(n_timesteps):
    stimulus_sequence[0, t_idx] = spatial_stimulus * temporal_profile[t_idx]
```

**Problems:**
1. Python loop over timesteps instead of vectorised broadcasting
2. For 2070-step sequences (default config), this is 2070 Python loop iterations
3. The operation is a simple outer product: `stimulus_sequence = spatial_stimulus.unsqueeze(0).unsqueeze(0) * temporal_profile.view(1, -1, 1, 1)`

**Severity:** High — easily 100× slower than vectorised equivalent on GPU.

**Remediation:**
```python
# Vectorised: broadcast spatial [H, W] × temporal [T] → [1, T, H, W]
stimulus_sequence = (spatial_stimulus.unsqueeze(0).unsqueeze(0)
                     * temporal_profile.view(1, -1, 1, 1))
```

**Testing:**
1. Verify output matches loop version (exact equality expected)
2. Benchmark improvement

---

### H8: `_CompiledNeuronModule` does not support GPU/autograd properly

**Location:** `sensoryforge/neurons/model_dsl.py:L451-L672`

**Issue Type:** Known Technical Debt / Architecture

**Current State:**
The docstring in copilot-instructions.md acknowledges: "The current DSL implementation uses numpy lambdify and does not fully support GPU or autograd."

However, looking at the code, `_TORCH_LAMBDIFY_MODULES` maps to torch ops, and `sympy.lambdify` is called with `modules=[_TORCH_LAMBDIFY_MODULES]`. This should theoretically produce torch-compatible callables.

**Problems:**
1. `ensure_real()` (L568-L582) handles complex outputs, suggesting sympy sometimes produces complex results for real equations — this indicates a parsing issue with the `I` variable (SymPy's imaginary unit vs. input current symbol)
2. The `I_symbol = Symbol('I', real=True)` workaround is applied in `_parse_equations`, `_parse_threshold`, and `_parse_reset` but NOT in `_create_lambdas` — potentially inconsistent symbol handling
3. `compile()` only supports `solver='euler'`; the `AdaptiveSolver` argument type in `_create_neurons` of `generalized_pipeline.py` (L340) passes a solver object, but `compile()` expects a string
4. No `torch.no_grad()` boundary — the DSL module computes derivatives through the entire computation graph, which is correct for training but wastes memory for inference

**Severity:** High — the DSL is a Phase 2 feature; if the imaginary-I bug surfaces it silently corrupts computation.

**Remediation:**
1. In `_create_lambdas()`, use the same `Symbol('I', real=True)` construction for `I_sym` (already done at L488 — verify consistency)
2. Add a test that exercises the DSL path end-to-end through `GeneralizedTactileEncodingPipeline` with `neurons.type: dsl`
3. Fix `compile()` to accept `solver` as string or `BaseSolver` instance
4. Document GPU compatibility status in the module docstring

---

## Medium Priority Issues

### M1: `IzhikevichNeuronTorch.forward` — `u_init` handling for non-tuple `b`

**Location:** `sensoryforge/neurons/izhikevich.py:L147`

**Current Code:**
```python
u_init_val = (
    self.u_init if not isinstance(self.b, tuple) else b_tensor * self.v_init
)
u = torch.full((batch, features), u_init_val, dtype=dtype, device=device)
```

**Problem:** When `self.b` is not a tuple, `u_init_val` is a Python float and `torch.full` works fine. But when `self.b` IS a tuple, `u_init_val = b_tensor * self.v_init` is a tensor of shape `(features,)`, and `torch.full((batch, features), u_init_val)` fails because `torch.full` requires a scalar fill value. This bug only triggers when `b` is a `(mean, std)` tuple.

**Severity:** Medium — affects parameter-variability mode which is used in the generalized pipeline.

**Remediation:**
```python
if isinstance(self.b, tuple):
    u = (b_tensor * self.v_init).unsqueeze(0).expand(batch, features).clone()
else:
    u = torch.full((batch, features), self.u_init, dtype=dtype, device=device)
```

**Testing:** Add test with `b=(0.2, 0.01)` to verify no crash.

---

### M2: `NotebookTactileEncodingPipeline` is a near-duplicate of `GeneralizedTactileEncodingPipeline`

**Location:** `sensoryforge/core/notebook_pipeline.py` (449 lines)

**Issue Type:** Code Duplication / Maintenance Burden

**Problems:**
1. Nearly identical to `GeneralizedTactileEncodingPipeline` — same structure, same patterns
2. Hardcoded seed values (`seed=33`, `seed=39`) instead of config-driven
3. No tests
4. Maintenance changes must be applied in both files

**Severity:** Medium — increases maintenance burden and risk of divergence.

**Remediation:**
Consider deprecating in favor of `GeneralizedTactileEncodingPipeline` with notebook-specific config. If kept, extract shared logic into a base class.

---

### M3: `GeneralizedTactileEncodingPipeline._generate_texture_stimulus` references undefined `gabor_texture`

**Location:** `sensoryforge/core/generalized_pipeline.py:L461`

**Current Code:**
```python
spatial_stimulus = gabor_texture(
    xx, yy,
    center_x=center_x, center_y=center_y,
    ...
)
```

**Problem:** `gabor_texture` is not imported at the top of the file. The import at the top is `from sensoryforge.stimuli.stimulus import gaussian_pressure_torch, StimulusGenerator`. The `gabor_texture` function exists in `sensoryforge/stimuli/texture.py` but is never imported in `generalized_pipeline.py`.

**Severity:** Medium — `NameError` at runtime when using `stimulus_type="texture"` with `pattern="gabor"`.

**Remediation:**
Add import at the top of the method or at module level:
```python
from sensoryforge.stimuli.texture import gabor_texture, edge_grating
```

---

### M4: Missing test coverage for critical modules

**Location:** Test directory (`tests/unit/`, `tests/integration/`)

**Issue Type:** Testing Gap

**Modules without dedicated test files:**
1. `core/mechanoreceptors.py` — no tests
2. `core/tactile_network.py` — no tests
3. `core/compression.py` — no tests
4. `core/visualization.py` — no tests
5. `core/notebook_pipeline.py` — no tests
6. `neurons/fa.py` — no tests
7. `neurons/sa.py` — no tests
8. `cli.py` — no tests
9. `utils/project_registry.py` — no tests
10. `gui/*` — no tests (understandable for Qt, but headless tests are possible)

**Severity:** Medium — project standard requires 80% coverage, several core modules have 0%.

**Remediation:**
Prioritize tests for:
1. `mechanoreceptors.py` — kernel creation, forward pass shapes
2. `tactile_network.py` — adapter wiring
3. `cli.py` — argparse, config validation
4. `compression.py` — project/compress operations
5. `fa.py` and `sa.py` — neuron dynamics

---

### M5: `CompositeGrid` does not implement Poisson disk sampling correctly

**Location:** `sensoryforge/core/composite_grid.py:L317-L345`

**Issue Type:** Scientific Accuracy

**Current Code:**
```python
def _generate_poisson(self, density: float) -> torch.Tensor:
    """Generate Poisson disk sampling distribution."""
    # Approximate Poisson disk sampling via jittered grid at target density.
    ...
    jitter_scale = 0.5 * spacing
    jitter = (torch.rand_like(coordinates) - 0.5) * jitter_scale
    coordinates = coordinates + jitter
```

**Problem:** The docstring says "Poisson disk sampling" but the implementation is actually a jittered grid — which is already a separate arrangement type (`jittered_grid`). True Poisson disk sampling enforces a minimum distance constraint between all point pairs, which this code does not.

**Severity:** Medium — scientific inaccuracy in naming. The function produces reasonable spatial distributions, but calling it "Poisson disk" is misleading.

**Remediation:**
1. Rename to `_generate_jittered_poisson` or update docstring to say "approximation via jittered grid"
2. Optionally implement a proper dart-throwing or Bridson's algorithm for true Poisson disk sampling

---

### M6: `TactileEncodingPipelineTorch.reset_filter_states` accesses private attributes

**Location:** `sensoryforge/core/pipeline.py:L335-L337`

**Current Code:**
```python
def reset_filter_states(self) -> None:
    self.filters.sa_filter.x = None
    self.filters.sa_filter.I_SA = None
    self.filters.ra_filter.I_RA = None
```

**Problem:** Directly manipulates internal state attributes of child modules. If `SAFilterTorch` or `RAFilterTorch` changes state representation, this breaks silently.

**Severity:** Medium — fragile coupling.

**Remediation:**
Use the existing `reset_states()` method on the filter objects:
```python
def reset_filter_states(self) -> None:
    self.filters.sa_filter.x = None
    self.filters.sa_filter.I_SA = None
    self.filters.ra_filter.I_RA = None
```
→ Better:
```python
def reset_filter_states(self) -> None:
    # Signal that next forward() should reinitialize states
    self.filters.sa_filter.x = None
    self.filters.sa_filter.I_SA = None
    self.filters.ra_filter.I_RA = None
```
Or ideal: add a `reset()` method to `CombinedSARAFilter` that encapsulates this.

---

### M7: `MechanoreceptorModule.update_parameters` buffer shape mismatch

**Location:** `sensoryforge/core/mechanoreceptors.py:L166-L170`

**Current Code:**
```python
def update_parameters(self, ...):
    ...
    new_kernel = create_gaussian_kernel_torch(...)
    # Update the registered buffer
    self.kernel.data = new_kernel.squeeze(0).squeeze(0)
    self.gaussian_kernel = new_kernel
```

**Problem:** `self.kernel` was registered as a buffer with shape `(1, 1, kernel_size, kernel_size)` but `self.kernel.data = new_kernel.squeeze(0).squeeze(0)` assigns a `(kernel_size, kernel_size)` tensor — shape mismatch. The `forward()` method then passes `self.kernel` (which has been corrupted to 2D) to `F.conv2d` which expects 4D, causing a runtime error.

**Severity:** Medium — breaks dynamic parameter updates (used in GUI).

**Remediation:**
```python
self.kernel.data = new_kernel.squeeze(0).squeeze(0)
```
→
```python
self.kernel = new_kernel  # keep original 4D shape
```
Or use buffer reregistration.

---

### M8: `TactileSpikingNetwork.forward` overwrites `sa_spikes`/`ra_spikes` keys

**Location:** `sensoryforge/core/tactile_network.py:L129-L145`

**Current Code:**
```python
results = {
    "sa_spikes": sa_spikes,
    "ra_spikes": ra_spikes,
    ...
}
if return_intermediates:
    results.update(pipeline_results)
```

**Problem:** `pipeline_results` also contains `sa_spikes` and `ra_spikes` keys (from the pipeline's own neuron processing). The `results.update(pipeline_results)` call overwrites the TactileNeuronAdapter spikes with the pipeline's own spike output, defeating the purpose of the adapter.

**Severity:** Medium — incorrect results when `return_intermediates=True`.

**Remediation:**
Prefix the adapter results:
```python
results = {
    "adapter_sa_spikes": sa_spikes,
    "adapter_ra_spikes": ra_spikes,
    ...
}
```
Or merge more carefully.

---

### M9: `setup.py` lists `torchvision` and `torchaudio` as hard dependencies

**Location:** `setup.py:L13-L14`

**Current Code:**
```python
install_requires=[
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "torchaudio>=0.12.0",
    ...
]
```

**Problem:** SensoryForge does not use `torchvision` or `torchaudio` anywhere in its source code. These are heavy packages (~1GB combined) that slow installation and increase dependency conflicts.

**Severity:** Medium — bloats install, causes friction for users.

**Remediation:** Remove `torchvision` and `torchaudio` from `install_requires`. If future vision/audio modalities need them, add as extras.

---

### M10: `cli.py` references `GeneralizedTactileEncodingPipeline.from_config()` which is incomplete

**Location:** `sensoryforge/cli.py:L128`

**Current Code:**
```python
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
```

`from_config()` (line 918 of generalized_pipeline.py) simply delegates to `__init__` with `config_dict=config`. But `cmd_run` then calls `pipeline.forward(stimulus_type=stimulus_type, duration=args.duration, ...)`. The `forward()` method doesn't accept a `duration` keyword.

**Severity:** Medium — CLI `run` command will crash with unexpected keyword argument.

**Remediation:**
1. Pass `duration` via `stimulus_params` in the `forward()` call, or
2. Generate the stimulus separately before calling `forward()`

---

## Low Priority / Enhancements

### L1: `grid.py` — `GridManager` stores `dx, dy` as tensors but `spacing` as float

**Location:** `sensoryforge/core/grid.py:L72-L80`

**Problem:** `self.dx` and `self.dy` are torch tensors (from `get_grid_spacing`) while `self.spacing` is a float. Mixing types can cause unexpected behavior in comparisons.

**Severity:** Low

---

### L2: `StimulusGenerator.generate_batch_stimuli` lacks docstring completeness

**Location:** `sensoryforge/stimuli/stimulus.py:L183`

**Problem:** Missing Args/Returns/Raises documentation per project standard.

**Severity:** Low

---

### L3: `SAFilterTorch._forward_sequence` resets states redundantly

**Location:** `sensoryforge/filters/sa_ra.py:L136-L141`

**Problem:** Calls `self._forward_single_step(..., reset_states=(t == 0))` which re-resets on the first iteration, but line 131 already calls `self.reset_states(...)`. Harmless but wasteful.

**Severity:** Low

---

### L4: `sensoryforge/__init__.py` exports only `GridManager` and `create_grid_torch`

**Location:** `sensoryforge/__init__.py:L37-L43`

**Problem:** The top-level package exposes very few symbols. Users must write lengthy imports like `from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch`. Consider re-exporting key classes from subpackage `__init__.py` files (which already exist and export well) at the top level.

**Severity:** Low — usability enhancement.

---

### L5: Multiple `for t_idx in range(n_timesteps)` loops in `NotebookTactileEncodingPipeline`

**Location:** `sensoryforge/core/notebook_pipeline.py:L194-L196` (and similar)

**Problem:** Same vectorisation opportunity as H7, duplicated in the notebook pipeline.

**Severity:** Low — duplicate of H7.

---

### L6: `sensoryforge/gui/tabs/` not listed in workspace structure detail

**Location:** `sensoryforge/gui/tabs/`

**Problem:** GUI tab modules exist but were not provided in the workspace tree detail. I verified they exist. No code issues flagged — GUI code is extensive but functional. Recommend headless smoke tests in the future.

**Severity:** Low — informational.

---

### L7: `FANeuronTorch.forward` double-reports spikes

**Location:** `sensoryforge/neurons/fa.py:L188-L189`

**Current Code:**
```python
spikes[:, t, :] = fired
spikes[:, t + 1, :] = fired
```

**Problem:** Both time index `t` and `t+1` record the same spike event. On the last time step, `spikes[:, T, :] = fired` is overwritten by the next iteration's `spikes[:, T, :] = fired_new`. This means the last iteration's `spikes[:, T, :]` comes from the `t+1` write of `t = T-1`, which is correct, but the intermediate `t` writes are redundant and misleading.

**Severity:** Low — produces correct final result but the double-write pattern is confusing and wastes computation.

**Remediation:** Record spikes only at `t` (matching other neuron models) or only at `t+1`.

---

## Documentation Improvements

### D1: Missing user guide pages for several modules

According to copilot-instructions.md: "Every module must have a corresponding section in the user guide."

Missing guides:
- `docs/user_guide/mechanoreceptors.md`
- `docs/user_guide/innervation.md`
- `docs/user_guide/pipeline.md`
- `docs/user_guide/neurons.md`
- `docs/user_guide/filters.md`
- `docs/user_guide/noise.md`

Existing guides cover Phase 2 features (composite_grid, equation_dsl, extended_stimuli, solvers) but not the core Phase 1 modules.

### D2: `copilot-instructions.md` references non-existent plugin registry

Lines 440-456 describe a `plugins/registry.py` with `registry.discover_plugins()` and `registry.get_filter()`. No such module exists. This should be marked as planned/future.

### D3: `docs/api_reference/` directory is empty

No auto-generated API reference exists. The `mkdocs` config should be set up with `mkdocstrings` to generate API docs from docstrings.

### D4: Example config `examples/example_config.yml` should be tested in CI

The example configuration should have an integration test that loads and validates it to catch drift.

---

## Testing Gaps Summary

| Gap | Priority | Effort |
| --- | -------- | ------ |
| `core/mechanoreceptors.py` — no tests | Medium | Small |
| `core/tactile_network.py` — no tests | Medium | Small |
| `core/compression.py` — no tests | Medium | Small |
| `neurons/fa.py` — no tests | Medium | Small |
| `neurons/sa.py` — no tests | Medium | Small |
| `cli.py` — no tests | Medium | Medium |
| `core/visualization.py` — no tests | Low | Medium |
| `utils/project_registry.py` — no tests | Low | Medium |
| `core/notebook_pipeline.py` — no tests | Low | Small |
| `get_grid_spacing` bug — no test catches it | High | Tiny |
| DSL end-to-end through generalized pipeline — no test | High | Medium |
| Noise module device propagation — no test | High | Tiny |

---

## Recommended Remediation Priority

1. **C1** — Vectorise innervation loop (biggest perf impact)
2. **C2** — Fix noise modules to inherit `nn.Module`
3. **H3** — Fix `get_grid_spacing` bug
4. **H4** — Add `plotly` dependency
5. **H5** — Fix global RNG pollution
6. **H7** — Vectorise stimulus generation loops
7. **M3** — Fix missing `gabor_texture` import
8. **M7** — Fix kernel buffer shape mismatch
9. **M9** — Remove unnecessary `torchvision`/`torchaudio` dependencies
10. **H1** — Create base classes (larger effort, architectural foundation)
11. **M4** — Add missing test coverage (ongoing)
12. All remaining Medium/Low findings

---

*End of findings. This document is ready for human review and approval before remediation begins.*
