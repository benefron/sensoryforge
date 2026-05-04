Maintain SensoryForge documentation — update docstrings, user guides, and fix drift between code and docs.

$ARGUMENTS should specify the scope, e.g.:
- `sensoryforge/filters/sa_ra.py` — update docstrings for a specific file
- `docs/user_guide/composite_grid.md` — update a user guide
- `full` — audit the entire docs/ and docstrings

## Core principles

- **Code is the source of truth.** Fix docstrings first, then propagate changes to markdown guides.
- **Tensor shapes and units are mandatory** in every Args/Returns block.
- **Every Example: block must run.** If you can't verify it, flag it with `# TODO: verify`.
- **Scientific claims must cite the paper** they derive from.
- **Do not publish internal docs.** `docs/` ships publicly. `docs_root/` stays internal.

## Phase 1 — Analysis

1. Read the specified file(s) and their corresponding tests.
2. Read any existing markdown guide for the module (in `docs/user_guide/`).
3. List every gap found:
   - Missing `Args:` / `Returns:` / `Example:` blocks
   - Wrong tensor shapes (compare docstring to actual code)
   - Missing physical units
   - Broken examples (API has changed)
   - Documentation drift (guide describes old behaviour)
   - Missing scientific citations

## Phase 2 — Fix docstrings

Apply Google Style throughout. Required elements for every public API:

```python
def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """One-line summary of what this does.

    Extended description linking to the pipeline stage and scientific
    motivation. Reference the paper the equation is drawn from.

    Args:
        x: Input stimulus [batch, time, grid_h, grid_w] in N/mm²
        dt: Time step in seconds. Default: 0.001 (1 ms).

    Returns:
        Filtered currents [batch, time, num_neurons] in mA.

    Raises:
        ValueError: If x is not 4-D.

    Example:
        >>> f = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        >>> out = f(torch.randn(2, 100, 64, 64))
        >>> out.shape
        torch.Size([2, 100, 400])

    References:
        Parvizi-Fard et al. (2024). "Biologically-inspired dual-pathway...". bioRxiv.
    """
```

## Phase 3 — Fix user guides

Each user guide in `docs/user_guide/` should follow this structure:

```markdown
# Module Name

## Overview
What it does and when to use it.

## Mathematical Foundation
LaTeX equations ($$...$$) with variables defined.
Implementation note if Euler approximation is used.

## Usage

### Basic example
\```python
# Minimal working example
\```

### Advanced example
\```python
# More realistic usage
\```

## Configuration (YAML)
Show the YAML block that instantiates this component.

## Shape Reference
| Stage | Shape | Units |
|-------|-------|-------|

## See Also
Links to related guides and API reference.
```

## Phase 4 — Verification

- Check `mkdocs.yml` includes any new files added to `docs/`.
- Verify all internal cross-links resolve.
- Confirm no broken imports in example blocks.

## Output

For docstring-only changes: apply edits directly and commit with `docs(<scope>): update docstrings for <module>`.

For guide changes: apply edits and commit with `docs: update <guide name> user guide`.

If drift is severe (many issues), write a summary to the conversation listing each gap found before making changes, so the scope can be confirmed.
