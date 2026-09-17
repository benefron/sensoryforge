"""Worked example: define, register, and run a minimal solver component (U3).

Runnable companion to ``docs/extending/add_solver.md``.

1. Define a solver class inheriting :class:`~sensoryforge.solvers.base.BaseSolver`,
   implementing Heun's method (explicit trapezoidal RK2) instead of the
   built-in ``EulerSolver``'s forward Euler.
2. Register it with ``SOLVER_REGISTRY``.
3. Run it through the shared contract check
   (``sensoryforge.testing.contracts.check_component``).
4. Integrate a known linear ODE (``dx/dt = -x``) and compare against the
   analytic solution ``x(t) = x0 * exp(-t)`` -- RK2 should track it more
   closely than forward Euler at the same step size, which is the actual
   reason to add a second-order solver.

Run it directly: ``python docs/examples/plugin_solver.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import torch

from sensoryforge.registry import SOLVER_REGISTRY
from sensoryforge.solvers.base import BaseSolver
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.stimuli.base import ParamSpec


class HeunSolver(BaseSolver):
    """Heun's method (explicit trapezoidal RK2) ODE solver.

    ``state_{t+1} = state_t + dt/2 * (f(state_t, t) + f(state_t + dt*f(state_t, t), t+dt))``

    Second-order accurate, unlike the built-in forward-Euler ``EulerSolver``
    (first-order): for the same ``dt`` its local truncation error is
    ``O(dt^3)`` instead of ``O(dt^2)``.

    Args:
        dt: Default time step size in ms.
    """

    def __init__(self, dt: float = 0.05) -> None:
        super().__init__(dt=dt)

    def step(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """One Heun step.

        Args:
            ode_func: ``f(state, t) -> dstate/dt``.
            state: Current state, ``[batch, ...]``.
            t: Current time in ms.
            dt: Step size in ms.

        Returns:
            Updated state, same shape as ``state``.
        """
        k1 = ode_func(state, t)
        k2 = ode_func(state + dt * k1, t + dt)
        return state + (dt / 2.0) * (k1 + k2)

    def integrate(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float,
    ) -> torch.Tensor:
        """Integrate over ``t_span`` by repeated :meth:`step` calls.

        Args:
            ode_func: ``f(state, t) -> dstate/dt``.
            state: Initial state, ``[batch, ...]``.
            t_span: ``(t_start, t_end)`` in ms.
            dt: Step size in ms.

        Returns:
            ``[batch, num_steps + 1, ...]`` trajectory, initial state first.
        """
        t0, t1 = t_span
        n_steps = math.ceil((t1 - t0) / dt)
        traj = [state]
        s, t = state, t0
        for _ in range(n_steps):
            s = self.step(ode_func, s, t, dt)
            t += dt
            traj.append(s)
        return torch.stack(traj, dim=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeunSolver":
        return cls(dt=config.get("dt", 0.05))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "heun", "dt": self.dt}

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "dt",
                dtype="float",
                default=0.05,
                min_val=0.001,
                max_val=10.0,
                unit="ms",
                choices=None,
                help="Integration time step.",
                group="Solver",
                advanced=False,
            )
        ]


def main() -> None:
    SOLVER_REGISTRY.register("heun_demo", HeunSolver)

    from sensoryforge.testing.contracts import check_component

    check_component("solver", HeunSolver)

    def decay(state: torch.Tensor, t: float) -> torch.Tensor:
        return -state

    x0 = torch.tensor([[1.0]])
    dt = 0.5
    t_end = 3.0

    heun = SOLVER_REGISTRY.create("heun_demo", dt=dt)
    euler = EulerSolver(dt=dt)

    heun_traj = heun.integrate(decay, x0, t_span=(0.0, t_end), dt=dt)
    euler_traj = euler.integrate(decay, x0, t_span=(0.0, t_end), dt=dt)
    analytic = x0 * math.exp(-t_end)

    heun_err = abs(heun_traj[0, -1, 0].item() - analytic.item())
    euler_err = abs(euler_traj[0, -1, 0].item() - analytic.item())
    print(f"analytic x(3) = {analytic.item():.6f}")
    print(f"Heun   error = {heun_err:.6f}")
    print(f"Euler  error = {euler_err:.6f}")
    assert (
        heun_err < euler_err
    ), "Heun (2nd order) should beat Euler (1st order) at this dt"


if __name__ == "__main__":
    main()
