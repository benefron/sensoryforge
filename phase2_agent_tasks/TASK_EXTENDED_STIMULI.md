# Task: Extended Stimuli (Gaussian, Texture, Moving)

## Goal
Add stimulus modules under `sensoryforge/stimuli/` to support Gaussian, texture, and moving stimuli.

## Shared Context
Read [phase2_agent_tasks/SHARED_CONTEXT.md](phase2_agent_tasks/SHARED_CONTEXT.md) first.

## Scope (Do Only These)
1. Create `sensoryforge/stimuli/gaussian.py` (refactor existing Gaussian logic if present).
2. Create `sensoryforge/stimuli/texture.py` for patterns and gratings.
3. Create `sensoryforge/stimuli/moving.py` for moving contacts/trajectories.
4. Add unit tests for each stimulus type under `tests/unit/`.

## Requirements
- Support superposition of multiple stimuli.
- Support temporal sequences (tap, slide, vibration).
- Accept `device` and return tensors with batch-first conventions.
- Expose minimal, clear public APIs with docstrings.

## Tests (Minimum)
- Gaussian output shape and parameter validation.
- Texture patterns deterministic with a fixed seed.
- Moving stimuli update position over time.

## Non-Goals
- Do not change pipeline code.
- Do not change docs or packaging.

## Deliverables
- `sensoryforge/stimuli/gaussian.py`, `texture.py`, `moving.py`.
- Tests in `tests/unit/`.

## Notes
- Use type hints and Google Style docstrings.
- Keep functionality minimal but complete.
