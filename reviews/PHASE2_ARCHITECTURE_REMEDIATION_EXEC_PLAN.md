# Phase 2 GUI Remediation Execution Plan

Date: 2026-02-09
Status: Draft (pending execution)

## Goals
- Align GUI behavior with Phase 2 Architecture Remediation Plan.
- Remove ProtocolSuiteTab from GUI.
- Fix composite grid propagation to Stimulus/Spiking tabs.
- Expose innervation method selection and remove composite-layer filter remnants.
- Replace Stimulus GUI with stackable builder-based design.
- Prevent DSL compile with unsupported solver selections.

## Scope
### 1) GUI Wiring Cleanup
- Remove ProtocolSuiteTab from tab exports and main window.
- Decouple protocol execution controller wiring.

### 2) Composite Grid Propagation Fix
- Make grid change signals carry a consistent grid object or a typed wrapper.
- Update Stimulus/Spiking tabs to handle composite grids without errors.

### 3) Mechanoreceptor Tab: Innervation & Grid UI
- Remove remaining composite grid filter field usage in config.
- Add innervation method selection per neuron population.
- Integrate create_innervation() / strategy selection.

### 4) Stimulus Tab: Stackable Builder UI
- Replace single-stimulus UI with stackable stimulus list.
- Add composition mode (add/max/mean).
- Add motion dropdown per stimulus (static/linear/circular/custom).
- Map UI to Stimulus builder API (Stimulus.compose + with_motion).

### 5) DSL Solver Guard
- Prevent selecting adaptive solver for DSL compile in GUI.
- Add fallback to euler when DSL selected.

### 6) Tests & Validation
- Update GUI integration tests to expect 3 tabs.
- Add tests for composite grid GUI payloads (no filter, correct arrangement).
- Add tests for new stimulus config schema if covered by existing suite.

## Deliverables
- Updated GUI code paths and config serialization.
- Removal of ProtocolSuiteTab.
- Updated tests reflecting new GUI state.

## Risks / Assumptions
- Stimulus builder API is already implemented and stable.
- CompositeReceptorGrid support in core is present.
- GUI test suite may need adjustments for headless runs.

## Execution Order
1) Remove ProtocolSuiteTab wiring.
2) Fix composite grid propagation and related errors.
3) Update Mechanoreceptor tab innervation + config cleanup.
4) Replace Stimulus tab UI with stackable builder flow.
5) Guard DSL solver selection.
6) Update tests.
