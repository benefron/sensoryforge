"""The CI regression guard tells a slower machine from a slower engine (F-067).

The guard's baseline was measured on an Apple M3 Pro laptop and the guard
runs on a GitHub Linux runner. Comparing raw milliseconds made its verdict
depend on hardware as much as on code: a runner several times slower than
the laptop would fail it with nothing changed, and a false alarm on a
guard's first run is how it gets disabled.

The guard now divides the engine's time by a reference kernel's time, both
measured in the same process. These tests pin the property that matters,
using the pure decision function so nothing here depends on timing.
"""

import pytest

from benchmarks.check_regression import evaluate

BASELINE_RUN = 0.038
BASELINE_REF = 0.020
LIMIT = 3.0


class TestHardwareIsCancelled:
    @pytest.mark.parametrize("machine_slowdown", [1.0, 2.5, 4.0, 10.0])
    def test_a_uniformly_slower_machine_passes(self, machine_slowdown):
        """Everything slower by the same amount is not a regression."""
        verdict = evaluate(
            BASELINE_RUN * machine_slowdown,
            BASELINE_REF * machine_slowdown,
            BASELINE_RUN,
            BASELINE_REF,
            LIMIT,
        )
        assert verdict.passed, verdict.explanation
        assert verdict.calibrated
        assert verdict.factor == pytest.approx(1.0)

    def test_the_uncalibrated_comparison_is_what_used_to_fail(self):
        """Pins the defect: raw milliseconds on a 4x slower runner fail."""
        verdict = evaluate(BASELINE_RUN * 4.0, None, BASELINE_RUN, None, LIMIT)
        assert not verdict.passed
        assert not verdict.calibrated


class TestEngineRegressionsAreCaught:
    @pytest.mark.parametrize("machine_slowdown", [1.0, 3.0])
    def test_the_engine_ten_times_slower_fails_on_any_machine(self, machine_slowdown):
        """The spec's own example, on the laptop and on a slower runner."""
        verdict = evaluate(
            BASELINE_RUN * 10.0 * machine_slowdown,
            BASELINE_REF * machine_slowdown,
            BASELINE_RUN,
            BASELINE_REF,
            LIMIT,
        )
        assert not verdict.passed, verdict.explanation

    def test_just_over_the_limit_fails_and_just_under_passes(self):
        over = evaluate(
            BASELINE_RUN * 3.01, BASELINE_REF, BASELINE_RUN, BASELINE_REF, LIMIT
        )
        under = evaluate(
            BASELINE_RUN * 2.99, BASELINE_REF, BASELINE_RUN, BASELINE_REF, LIMIT
        )
        assert not over.passed
        assert under.passed

    def test_a_small_regression_is_honestly_not_caught(self):
        """Documented limitation: a 2x engine slowdown passes this guard."""
        verdict = evaluate(
            BASELINE_RUN * 2.0, BASELINE_REF, BASELINE_RUN, BASELINE_REF, LIMIT
        )
        assert verdict.passed


class TestInputs:
    def test_an_old_baseline_without_a_reference_still_works_uncalibrated(self):
        verdict = evaluate(BASELINE_RUN, BASELINE_REF, BASELINE_RUN, None, LIMIT)
        assert verdict.passed
        assert not verdict.calibrated
        assert "NOT calibrated" in verdict.explanation

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_timing_is_rejected(self, bad):
        with pytest.raises(ValueError):
            evaluate(bad, BASELINE_REF, BASELINE_RUN, BASELINE_REF, LIMIT)
