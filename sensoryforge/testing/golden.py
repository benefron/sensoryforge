"""Comparing results against committed golden fixtures, honestly across platforms.

A golden fixture records what the code produced once, on one machine. Several
of SensoryForge's were generated on macOS arm64 and compared with
``torch.equal``, which silently assumes every other machine rounds float32
identically. It does not. Measured on the first CI run on GitHub's Linux
x86_64 runners (torch 2.14.0, 2026-09-17), with
``scripts/dev/cross_platform_diagnostics.py``:

* **Structure was identical everywhere.** The same receptors connected to the
  same neurons, the same neuron centres, the same receptor coordinates. No
  comparison differed in which entries are non-zero.
* **Values differed by at most 5.96e-8**, one float32 unit in the last place
  for values below one, in the receptive-field weights and all four ported
  stimuli. macOS arm64 on the same torch version matched exactly.

So a bit-exact comparison against these fixtures tests the processor's
rounding, not SensoryForge. :func:`assert_matches_golden` compares at a stated
precision instead, and can require the structure to match exactly, which is
the part a real regression in wiring would change. :data:`FLOAT32_GOLDEN_ATOL`
is about three times the largest difference measured, and some seven orders of
magnitude below any change a person would call different.

Comparisons of two results computed in the same process are not golden
comparisons and should stay bit-exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

FLOAT32_GOLDEN_ATOL = 2e-7


def _as_double(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double()
    return torch.as_tensor(np.asarray(value)).double()


def assert_matches_golden(
    actual: Any,
    expected: Any,
    *,
    what: str,
    atol: float = FLOAT32_GOLDEN_ATOL,
    rtol: float = 0.0,
    exact_structure: bool = False,
) -> None:
    """Assert *actual* reproduces a golden fixture to a stated precision.

    Args:
        actual: The freshly computed array or tensor.
        expected: The value loaded from the fixture.
        what: A short description for the failure message.
        atol: Absolute tolerance per element.
        rtol: Relative tolerance per element, applied to ``|expected|``.
        exact_structure: Also require the non-zero pattern to match exactly.
            Use it for weight matrices, where which entries are non-zero is
            the wiring and must not change on any platform.

    Raises:
        AssertionError: With the shape mismatch, the number of structural
            mismatches, or the largest absolute and relative differences and
            the tolerance they exceeded.
    """
    a = _as_double(actual)
    e = _as_double(expected)
    if a.shape != e.shape:
        raise AssertionError(
            f"{what}: shape {list(a.shape)} does not match golden {list(e.shape)}"
        )
    if exact_structure:
        mismatched = int(((a != 0) != (e != 0)).sum())
        if mismatched:
            raise AssertionError(
                f"{what}: {mismatched} entries differ in whether they are zero -- "
                "the structure changed, which rounding cannot explain"
            )
    diff = (a - e).abs()
    tolerance = atol + rtol * e.abs()
    violations = diff > tolerance
    if bool(violations.any()):
        nonzero = e != 0
        max_rel = (
            float((diff[nonzero] / e[nonzero].abs()).max()) if nonzero.any() else 0.0
        )
        raise AssertionError(
            f"{what}: {int(violations.sum())} of {diff.numel()} values differ from "
            f"golden beyond atol={atol:g}, rtol={rtol:g}; max absolute difference "
            f"{float(diff.max()):.3e}, max relative difference {max_rel:.3e}"
        )
