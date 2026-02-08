# Phase 2 Agent Task Plan

## Purpose
Create parallel, digestible assignments for Phase 2 so multiple coding agents can work independently. Provide a shared reference file with only necessary, non-sensitive guidance from the development guide and copilot instructions, then create per-agent task files that reference that shared file.

## Steps
1. Create a dedicated folder for Phase 2 agent assignments.
2. Create a shared reference markdown file with Phase 2 goals, coding standards, and constraints (sanitized for non-sensitive content).
3. Create separate task files for each parallelizable Phase 2 deliverable (solvers, equation DSL, composite grid, extended stimuli), each with clear scope, inputs/outputs, tests, and boundaries.

## Phase 2 Tasks

### Core Features (Parallel Track)
- [TASK_SOLVERS.md](TASK_SOLVERS.md) — ODE solver architecture (Euler + Adaptive)
- [TASK_COMPOSITE_GRID.md](TASK_COMPOSITE_GRID.md) — Multi-population spatial substrate
- [TASK_EQUATION_DSL.md](TASK_EQUATION_DSL.md) — Neuron model equation DSL
- [TASK_EXTENDED_STIMULI.md](TASK_EXTENDED_STIMULI.md) — Texture and moving stimulus types

### Integration & Exposure (Sequential)
- [TASK_GUI_CLI_PIPELINE_INTEGRATION.md](TASK_GUI_CLI_PIPELINE_INTEGRATION.md) — Hook features into GUI, CLI, and pipeline with YAML config

## Shared Context
All tasks reference [SHARED_CONTEXT.md](SHARED_CONTEXT.md) for coding standards, structure, and non-sensitive guidance.
