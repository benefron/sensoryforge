# SensoryForge Review Agent Findings
**Date:** 2026-02-11
**Scope:** Full codebase review (sensoryforge/, tests/, docs/, docs_root/, reviews/)

## Executive Summary
- Total findings: 6 (Critical: 1, High: 1, Medium: 2, Low: 1, Testing Gaps: 1)
- Most severe: Protocol Suite execution crashes when debug is enabled due to an undefined attribute.
- Coverage gaps: Compression operator and project registry lack direct tests.

## Critical Issues (Must Fix)
### C1: Protocol Suite crashes in debug mode due to undefined attribute
**Location:** [sensoryforge/gui/protocol_backend.py](sensoryforge/gui/protocol_backend.py#L185-L193)
**Issue Type:** Bug
**Current Code:**
```python
    def run(self) -> None:  # pragma: no cover - executed in worker thread
        self._debug(
            "Worker run invoked",
            populations=[getattr(p, "name", "Population") for p in self._populations],
            protocols=[spec.key for spec in self._protocol_specs],
            base_dt_ms=self._packet_dt_ms,
            device=str(self._device),
            perform_fit=self._perform_fit,
        )
```
**Problems:**
1. `_perform_fit` is not initialized anywhere in `ProtocolWorker`, so attribute access raises `AttributeError`.
2. The Protocol Suite always constructs the worker with `debug=True`, so this crash occurs deterministically when runs start.

**Severity:** Critical (Protocol Suite execution fails immediately)

**Remediation:**
1. Add `_perform_fit: bool` initialization in `ProtocolWorker.__init__` with a safe default (e.g., `False`).
2. If the flag is meant to be configurable, add an optional `perform_fit: bool = False` parameter to `__init__` and pass it from the controller.
3. Alternatively, remove the `perform_fit` field from the `_debug()` call until the feature is implemented.

**Testing:**
- Add a unit test that constructs a `ProtocolWorker` with `debug=True` and calls `run()` with a minimal stubbed population and protocol list; assert no `AttributeError` is raised.
- Add a GUI integration test to the Protocol Suite queue flow to ensure the worker thread starts without crashing.

## High Priority Issues
### H1: Compression operator assumes grid-shaped weights and fails for flat innervation
**Location:** [sensoryforge/core/compression.py](sensoryforge/core/compression.py#L19-L125)
**Issue Type:** Bug
**Current Code:**
```python
@dataclass
class CompressionOperator:
    sa_weights: torch.Tensor
    ra_weights: torch.Tensor
    grid_shape: Tuple[int, int]

    @property
    def num_receptors(self) -> int:
        return int(self.grid_shape[0] * self.grid_shape[1])


def build_compression_operator(
    pipeline: "TactileEncodingPipelineTorch",
) -> CompressionOperator:
    sa_weights = pipeline.sa_innervation.innervation_weights
    ra_weights = pipeline.ra_innervation.innervation_weights
    grid_shape = tuple(sa_weights.shape[1:])
    return CompressionOperator(
        sa_weights=sa_weights,
        ra_weights=ra_weights,
        grid_shape=grid_shape,
    )
```
**Problems:**
1. For `FlatInnervationModule`, weights are `[num_neurons, num_receptors]`; `grid_shape` becomes a 1-tuple, so `num_receptors` raises `IndexError`.
2. `compression_ratio()` and downstream consumers silently assume a 2D grid, which is invalid for composite/flat receptor layouts.

**Severity:** High (breaks compression metrics and projections for composite grids)

**Remediation:**
1. Update `CompressionOperator` to store `num_receptors` explicitly (or accept `grid_shape: Optional[Tuple[int, int]]`).
2. In `build_compression_operator`, detect flat weights via `weights.ndim == 2` and set `num_receptors = weights.shape[1]` with `grid_shape=None` (or `(num_receptors, 1)` if you must keep a tuple).
3. Update `num_receptors` property to fall back to a stored `num_receptors` when `grid_shape` is absent.
4. Document in docstring that flat innervation is supported and grid shape may be absent.

**Testing:**
- Add a unit test using a mock `FlatInnervationModule` weight tensor `[num_neurons, num_receptors]` and assert `CompressionOperator.num_receptors` and `compression_ratio()` behave correctly.
- Add a unit test for standard grid weights to ensure backward compatibility.

## Medium Priority Issues
### M1: BaseFilter.from_config passes a dict into a float-only constructor
**Location:** [sensoryforge/filters/base.py](sensoryforge/filters/base.py#L88-L99)
**Issue Type:** Bug / API consistency
**Current Code:**
```python
@classmethod
def from_config(cls, config: Dict[str, Any]) -> "BaseFilter":
    return cls(config)
```
**Problems:**
1. `BaseFilter.__init__` expects `dt: float`, so `cls(config)` raises `TypeError` for any subclass that does not override `from_config`.
2. This violates the documented base-class contract for YAML instantiation.

**Severity:** Medium (breaks YAML instantiation for custom filters that rely on the base implementation)

**Remediation:**
1. Change the base implementation to `return cls(**config)`.
2. Alternatively, make `from_config` abstract to force subclasses to implement it.
3. Update the docstring example to reflect the preferred constructor signature.

**Testing:**
- Add a unit test in tests/unit/test_base_classes.py that defines a minimal `BaseFilter` subclass, calls `from_config({'dt': 0.001})`, and asserts `dt` is set.

### M2: Documentation references a non-existent `from_yaml()` API
**Locations:**
- [docs/index.md](docs/index.md#L45-L51)
- [docs/user_guide/gui_phase2_access.md](docs/user_guide/gui_phase2_access.md#L107-L113)
**Issue Type:** Documentation
**Current Code:**
```python
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('config.yml')
```
**Problems:**
1. `GeneralizedTactileEncodingPipeline` does not implement `from_yaml()`, so the documented example fails at runtime.
2. This appears in multiple user-facing docs, likely leading to early onboarding failures.

**Severity:** Medium (documentation example is broken)

**Remediation:**
1. Update docs to use `load_yaml()` + `from_config()` instead, or
2. Implement a `from_yaml(path: str)` classmethod that loads YAML and delegates to `from_config()`.

**Testing:**
- Add a doc test snippet (or a small unit test) that loads a YAML fixture and instantiates the pipeline via the documented API.

## Low Priority / Enhancements
### L1: Composite grid documentation still describes filter tags and legacy API
**Locations:**
- [docs/user_guide/composite_grid.md](docs/user_guide/composite_grid.md#L6-L20)
- [docs/user_guide/yaml_configuration.md](docs/user_guide/yaml_configuration.md#L108-L140)
**Issue Type:** Documentation
**Current Code:**
```markdown
Populations differ by density, arrangement, and metadata (e.g., filter tags).
```
```yaml
populations:
  sa1:
    density: 100.0
    arrangement: grid
    filter: SA
```
**Problems:**
1. The `filter` field is deprecated/ignored in code (`CompositeReceptorGrid.add_layer`), so the docs imply behavior that does not exist.
2. The YAML guide still says CompositeGrid support is “coming in a future update,” but the code already supports composite grids and flat innervation.

**Severity:** Low (documentation inconsistency)

**Remediation:**
1. Remove or clearly mark `filter` as deprecated and ignored.
2. Update the YAML guide note to reflect current support and the recommended `add_layer` / `CompositeReceptorGrid` usage.

**Testing:**
- Build docs (MkDocs) and validate that examples match current API names.

## Testing Gaps
### T1: No direct tests for compression operator and project registry
**Locations:**
- [sensoryforge/core/compression.py](sensoryforge/core/compression.py#L19-L125)
- [sensoryforge/utils/project_registry.py](sensoryforge/utils/project_registry.py#L1-L200)
**Issue Type:** Testing gap
**Problems:**
1. `CompressionOperator` behavior is not covered by unit tests, especially for flat innervation weights.
2. `ProjectRegistry` serialization/deserialization paths have no direct tests, despite being core to persistence.

**Severity:** Medium (risk of regression in core infrastructure)

**Remediation:**
1. Add unit tests for `CompressionOperator.project()`, `compression_ratio()`, and flat-weight handling.
2. Add tests for `ProjectRegistry` round-trip of `ProtocolRunRecord` and `STAAnalysisRecord` using a temp directory.

**Testing:**
- `pytest tests/unit/test_compression_operator.py`
- `pytest tests/unit/test_project_registry.py`
