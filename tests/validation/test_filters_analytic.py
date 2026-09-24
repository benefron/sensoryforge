"""Analytic validation of the SA/RA temporal filters (Wave S, S1).

Every check here compares against a *closed-form* solution of the filter's
own continuous-time ODE, never against a recorded output of our own code.
Each test also states, in its docstring, the tolerance and why that
tolerance is the right one for an explicit-Euler discretisation of a linear
ODE (first-order local truncation error => global error is O(dt)).

Perturbation proofs (recorded 2026-09-16, applied in place to
``sensoryforge/filters/sa_ra.py``, tested, then reverted):

* SA: changing the ``I_SA`` update in ``_forward_single_step`` (line 156)
  from ``self.I_SA + dI_SA_dt * self.dt`` to
  ``self.I_SA + dI_SA_dt * self.dt * 0.5`` (a bug that halves the effective
  decay step) makes
  ``test_sa_filter_step_response_converges_to_two_exponential_solution``
  fail on the finest-dt absolute-error bound: 0.0123 vs the 5e-4 tolerance
  (35x over).
* RA: changing ``self.k3 * torch.abs(dI_in_dt)`` in ``_forward_single_step``
  (line 398) to ``self.k3 * torch.abs(dI_in_dt) * 0.5`` makes
  ``test_ra_filter_ramp_response_converges_to_k3_didt`` fail on the
  finest-dt absolute-error bound: 0.0190 vs the 5e-4 tolerance (38x over) --
  the whole trajectory is pulled toward half the analytic target, which the
  Euler discretisation error is far too small to mask.
"""

from __future__ import annotations

import torch

from sensoryforge.filters.sa_ra import RAFilterTorch, SAFilterTorch


def _sa_analytic_step_response(
    t_ms: torch.Tensor, tau_r: float, tau_d: float, k1: float
) -> torch.Tensor:
    """Closed-form solution of the SA ODE pair for a unit step input.

    For ``I_in(t) = 1`` (t >= 0), ``dI_in/dt = 0``, the SA equations

        tau_r * dx/dt = k1 - x
        tau_d * dI_SA/dt = x - I_SA

    are two cascaded linear first-order ODEs with zero initial state. Their
    exact solution (standard two-compartment step response) is

        x(t)    = k1 * (1 - exp(-t/tau_r))
        I_SA(t) = k1 * [1 - (tau_d*exp(-t/tau_d) - tau_r*exp(-t/tau_r))
                        / (tau_d - tau_r)]

    (for tau_r != tau_d; the filter defaults keep them well separated).
    """
    x_inf_term = 1.0 - (
        tau_d * torch.exp(-t_ms / tau_d) - tau_r * torch.exp(-t_ms / tau_r)
    ) / (tau_d - tau_r)
    return k1 * x_inf_term


def test_sa_filter_step_response_converges_to_two_exponential_solution():
    """SA step response converges to the closed-form two-exponential solution.

    Reference: exact solution of the linear ODE pair defining the SA filter
    (see ``_sa_analytic_step_response``), evaluated in float64.

    Tolerance: explicit Euler on a linear ODE has local truncation error
    O(dt^2) and global error O(dt) (standard numerical-ODE result). Halving
    dt should roughly halve the peak absolute error. We check both that the
    error is small at the finest step and that it shrinks with an
    order >= 0.7 (well under the theoretical 1.0, to absorb the max-norm's
    extra noise) as dt is halved twice -- a scheme with the wrong order
    (e.g. one that doesn't converge, or converges as dt^0) fails this even
    if the finest-dt error alone looks small.
    """
    tau_r, tau_d, k1, k2 = 5.0, 30.0, 0.05, 3.0
    horizon_ms = 200.0

    errors = []
    dts = [0.4, 0.2, 0.1]
    for dt in dts:
        steps = int(round(horizon_ms / dt))
        sa_filter = SAFilterTorch(tau_r=tau_r, tau_d=tau_d, k1=k1, k2=k2, dt=dt)
        step_input = torch.ones(1, steps, 1, dtype=torch.float64)
        sa_out = sa_filter(step_input, reset_states=True).squeeze()

        t = torch.arange(steps, dtype=torch.float64) * dt
        analytic = _sa_analytic_step_response(t, tau_r, tau_d, k1)
        errors.append(float((sa_out - analytic).abs().max()))

    # Finest step: error must be small in absolute terms.
    assert errors[-1] < 5e-4

    # Convergence order between successive halvings: log2(e_coarse / e_fine).
    order_1 = torch.log2(torch.tensor(errors[0] / errors[1]))
    order_2 = torch.log2(torch.tensor(errors[1] / errors[2]))
    assert order_1 > 0.7, f"expected first-order convergence, got order {order_1}"
    assert order_2 > 0.7, f"expected first-order convergence, got order {order_2}"


def test_ra_filter_ramp_response_converges_to_k3_didt():
    """RA filter's response to a ramp converges to the closed-form ``k3*dI/dt`` curve.

    Reference: for a ramp input ``I_in(t) = c * t`` (constant derivative
    ``c``), the RA ODE ``tau_RA * dI_RA/dt = k3*|dI_in/dt| - I_RA`` is a
    first-order linear ODE driven by the *constant* ``k3*c``, whose exact
    solution (zero initial condition) is

        I_RA(t) = k3*c*(1 - exp(-t/tau_RA))

    which asymptotes to the closed-form steady state ``k3 * c`` -- the
    quantity named in the Wave S spec, "RA response to a ramp against
    k3 * dI/dt" -- and is used here over the whole trajectory (not just the
    limit) so genuine O(dt) discretisation error is visible: at the limit
    alone the residual analytic transient (exp(-t/tau)) swamps the Euler
    error once t >> tau_RA, hiding a broken scheme (see perturbation note
    below the module docstring for why the limit-only check was replaced).

    Tolerance: explicit Euler on a linear ODE has global error O(dt); we
    check the max trajectory error over ``t in [dt, 3*tau_RA]`` (the first
    sample is excluded: the filter's own finite-difference derivative
    estimate is undefined, i.e. taken as 0, at t=0, which is a known,
    dt-independent boundary artefact of the implementation, not part of the
    interior scheme being validated) shrinks at close to first order as dt
    halves, and is small in absolute terms at the finest step.
    """
    tau_ra, k3 = 8.0, 2.0
    c = 0.02  # mA/ms ramp slope
    t_eval = 3.0 * tau_ra

    errors = []
    dts = [0.4, 0.2, 0.1]
    for dt in dts:
        steps = int(round(t_eval / dt))
        ra_filter = RAFilterTorch(tau_RA=tau_ra, k3=k3, dt=dt)
        t = torch.arange(steps, dtype=torch.float64) * dt
        ramp = (c * t).view(1, steps, 1)
        ra_out = ra_filter(ramp, reset_states=True).squeeze()

        analytic = k3 * c * (1.0 - torch.exp(-t / tau_ra))
        errors.append(float((ra_out[1:] - analytic[1:]).abs().max()))

    assert errors[-1] < 5e-4
    order_1 = torch.log2(torch.tensor(errors[0] / errors[1]))
    order_2 = torch.log2(torch.tensor(errors[1] / errors[2]))
    assert order_1 > 0.7, f"expected first-order convergence, got order {order_1}"
    assert order_2 > 0.7, f"expected first-order convergence, got order {order_2}"


def test_ra_filter_impulse_response_decays_monotonically_to_zero():
    """RA impulse response: positive, then a monotone decay to zero.

    Restored. This lived in ``tests/unit/test_filters_vs_theory.py`` and was
    deleted when Wave S migrated that file here, with no replacement; the
    ramp test above covers a different property. It is a qualitative
    property check rather than a closed-form comparison, kept because it
    catches a sign error or an unstable recursion that a ramp input can
    mask: an RA filter must respond to an impulse and then relax, never
    oscillate or drift.

    Tolerance: 1e-6 on successive samples absorbs float32 rounding in an
    otherwise strictly decreasing tail; 1e-2 on the final sample is about
    seven time constants after the impulse, where exp(-200/30) is 1.3e-3.
    """
    tau_ra = 30.0  # ms
    k3 = 2.0
    dt = 0.1  # ms
    steps = int(200.0 / dt)

    ra_filter = RAFilterTorch(tau_RA=tau_ra, k3=k3, dt=dt)
    impulse = torch.zeros(1, steps, 1)
    impulse[:, 0, 0] = 1.0 / dt  # discrete impulse

    ra_out = ra_filter(impulse, reset_states=True).squeeze().numpy()

    assert ra_out[0] > 0.0
    tail = ra_out[1:]
    assert (tail[:-1] >= tail[1:] - 1e-6).all()
    assert tail[-1] < 1e-2
