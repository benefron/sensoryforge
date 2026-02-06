"""encoding.filters_torch
===========================

Temporal filtering utilities implementing the Parvizi–Fard SA/RA differential
equations.  These filters are shared by the canonical tactile encoding
pipelines and the analytical reconstruction demos, providing reproducible
implementations of the slowly adapting (SA) and rapidly adapting (RA)
population dynamics described in the project hypothesis.

The module exposes individual filters (:class:`SAFilterTorch`,
:class:`RAFilterTorch`) plus :class:`CombinedSARAFilter` for convenience when a
single component needs to keep both pathways in sync.  Each filter supports
batched single-frame inputs, full temporal sequences, and helper routines for
steady-state and edge-response handling used by the GUI and tests.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SAFilterTorch(nn.Module):
    r"""Slowly adapting filter implementing Parvizi–Fard Equations 6 and 7.

    Continuous-time dynamics (currents in mA, time in ms):

    * τ_r · d x/dt = k₂ · dI_in/dt + k₁ · I_in − x
    * τ_d · d I_SA/dt = x − I_SA

    Explicit Euler updates with step Δt = ``dt`` give

    * x_{t+1} = x_t + (Δt / τ_r) · (k₂ · \dot{I}_{in,t} + k₁ · I_{in,t} − x_t)
    * I_{SA,t+1} = I_{SA,t} + (Δt / τ_d) · (x_{t+1} − I_{SA,t})

    Inputs are either ``(batch, neurons)`` single steps or
    ``(batch, time, neurons)`` sequences. If the derivative tensor
    ``dI_in_dt`` is omitted it is estimated with finite differences along the
    time axis.
    """

    def __init__(
        self,
        tau_r: float = 5.0,
        tau_d: float = 30.0,
        k1: float = 0.05,
        k2: float = 3.0,
        dt: float = 0.1,
    ) -> None:
        """Initialise the SA filter with biophysical parameters.

        Args:
            tau_r: Rise time constant (ms) governing auxiliary state dynamics.
            tau_d: Decay time constant (ms) governing output current dynamics.
            k1: Gain applied to the input current inside equation 7.
            k2: Gain applied to the derivative term in equation 7.
            dt: Integration step size in milliseconds.
        """
        super().__init__()

        self.tau_r = tau_r
        self.tau_d = tau_d
        self.k1 = k1
        self.k2 = k2
        self.dt = dt

        # State variables will be initialized when needed
        self.x = None  # auxiliary variable
        self.I_SA = None  # output current

    def reset_states(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Reset state buffers prior to processing a new sequence."""
        self.x = torch.zeros(batch_size, num_neurons, device=device)
        self.I_SA = torch.zeros(batch_size, num_neurons, device=device)

    def forward(
        self,
        I_in: torch.Tensor,
        dI_in_dt: torch.Tensor | None = None,
        reset_states: bool = True,
    ) -> torch.Tensor:
        """Filter input currents according to the SA dynamics.

        Args:
            I_in: Input tensor with shape ``(batch, neurons)`` or
                ``(batch, time, neurons)``.
            dI_in_dt: Optional derivative tensor matching ``I_in``.
            reset_states: Reset internal buffers before processing.

        Returns:
            Filtered tensor matching ``I_in``'s shape.
        """
        if len(I_in.shape) == 2:
            # Single time step: (batch_size, num_neurons)
            return self._forward_single_step(I_in, dI_in_dt, reset_states)
        elif len(I_in.shape) == 3:
            # Time sequence: (batch_size, time_steps, num_neurons)
            return self._forward_sequence(I_in, reset_states)
        else:
            raise ValueError(f"Unsupported input shape: {I_in.shape}")

    def _forward_single_step(
        self,
        I_in: torch.Tensor,
        dI_in_dt: torch.Tensor | None,
        reset_states: bool,
    ) -> torch.Tensor:
        """Process a single SA time step."""
        batch_size, num_neurons = I_in.shape

        if self.x is None or reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        if dI_in_dt is None:
            dI_in_dt = torch.zeros_like(I_in)

        # Equation 7: τ_r * dx/dt = k2 * dI_in/dt + k1 * I_in - x
        dx_dt = (self.k2 * dI_in_dt + self.k1 * I_in - self.x) / self.tau_r
        self.x = self.x + dx_dt * self.dt

        # Equation 6: τ_d * dI_SA/dt = x - I_SA
        dI_SA_dt = (self.x - self.I_SA) / self.tau_d
        self.I_SA = self.I_SA + dI_SA_dt * self.dt

        return self.I_SA.clone()

    def _forward_sequence(
        self,
        I_in: torch.Tensor,
        reset_states: bool,
    ) -> torch.Tensor:
        """Iterate over a temporal sequence using finite differencing."""
        batch_size, time_steps, num_neurons = I_in.shape

        if reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        # Compute derivatives using finite differences
        dI_in_dt = torch.zeros_like(I_in)
        dI_in_dt[:, 1:, :] = (I_in[:, 1:, :] - I_in[:, :-1, :]) / self.dt

        # Process each time step
        outputs = torch.zeros_like(I_in)
        for t in range(time_steps):
            outputs[:, t, :] = self._forward_single_step(
                I_in[:, t, :], dI_in_dt[:, t, :], reset_states=(t == 0)
            )

        return outputs

    def forward_steady_state(self, I_in: torch.Tensor) -> torch.Tensor:
        """
        Compute steady-state SA filter response for constant input.
        For constant input I_in with dI_in/dt = 0:
        - At steady state: dx/dt = 0, so x = k1 * I_in
        - At steady state: dI_SA/dt = 0, so I_SA = x = k1 * I_in

        Args:
            I_in: input current tensor (batch_size, num_neurons)

        Returns:
            I_SA: steady-state SA filter output
        """
        return self.k1 * I_in

    def forward_multi_step(
        self,
        I_in: torch.Tensor,
        num_steps: int = 50,
        reset_states: bool = True,
    ) -> torch.Tensor:
        """Simulate repeated steps so the filter reaches steady state.

        Args:
            I_in: Input current tensor ``(batch_size, num_neurons)``.
            num_steps: Number of time steps to simulate.
            reset_states: Whether to reset states initially.

        Returns:
            Final SA filter output after ``num_steps`` updates.
        """
        batch_size, num_neurons = I_in.shape

        if reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        # Simulate num_steps with constant input
        dI_in_dt = torch.zeros_like(I_in)  # Zero derivative for constant input

        for _ in range(num_steps):
            # Equation 7: τ_r * dx/dt = k2 * dI_in/dt + k1 * I_in - x
            dx_dt = (self.k2 * dI_in_dt + self.k1 * I_in - self.x) / self.tau_r
            self.x = self.x + dx_dt * self.dt

            # Equation 6: τ_d * dI_SA/dt = x - I_SA
            dI_SA_dt = (self.x - self.I_SA) / self.tau_d
            self.I_SA = self.I_SA + dI_SA_dt * self.dt

        return self.I_SA.clone()


class RAFilterTorch(nn.Module):
    r"""Rapidly adapting filter implementing Parvizi–Fard Equation 8.

    Continuous-time dynamics:

    * τ_RA · d I_RA/dt = k₃ · |dI_in/dt| − I_RA

    where currents are in mA and τ_RA is in ms. The implementation performs an
    explicit Euler step with Δt = ``dt`` and estimates ``dI_in/dt`` via finite
    differences when a derivative tensor is not supplied. Inputs can be shaped
    ``(batch, neurons)`` for a single step or ``(batch, time, neurons)`` for
    sequences. States are reset between sequences unless
    ``reset_states=False``.
    """

    def __init__(
        self,
        tau_RA: float = 30.0,
        k3: float = 2.0,
        dt: float = 0.1,
    ) -> None:
        """Initialise with derivative-sensitive parameters."""
        super().__init__()

        self.tau_RA = tau_RA
        self.k3 = k3
        self.dt = dt

        # State variable
        self.I_RA = None  # output current

    def reset_states(
        self,
        batch_size: int,
        num_neurons: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Reset internal state buffers for a fresh sequence."""
        self.I_RA = torch.zeros(batch_size, num_neurons, device=device)

    def forward(
        self,
        I_in: torch.Tensor,
        dI_in_dt: torch.Tensor | None = None,
        reset_states: bool = True,
    ) -> torch.Tensor:
        """Filter inputs using the RA equation with derivative sensitivity."""
        if len(I_in.shape) == 2:
            # Single time step: (batch_size, num_neurons)
            batch_size, num_neurons = I_in.shape
            if self.I_RA is None or reset_states:
                self.reset_states(batch_size, num_neurons, I_in.device)
            if dI_in_dt is None:
                # For a step input, the correct impulse area is I_in / dt
                dI_in_dt = I_in / self.dt
            # Equation 8: τ_RA * dI_RA/dt = k3 * |dI_in/dt| - I_RA
            dI_RA_dt = (self.k3 * torch.abs(dI_in_dt) - self.I_RA) / self.tau_RA
            self.I_RA = self.I_RA + dI_RA_dt * self.dt
            return self.I_RA.clone()
        elif len(I_in.shape) == 3:
            # Time sequence: (batch_size, time_steps, num_neurons)
            batch_size, time_steps, num_neurons = I_in.shape
            if reset_states:
                self.reset_states(batch_size, num_neurons, I_in.device)
            # Compute derivatives using finite differences
            dI_in_dt = torch.zeros_like(I_in)
            dI_in_dt[:, 1:, :] = (I_in[:, 1:, :] - I_in[:, :-1, :]) / self.dt
            # First time step uses the step-size derivative (step / dt)
            dI_in_dt[:, 0, :] = I_in[:, 0, :] / self.dt
            # Process each time step
            outputs = torch.zeros_like(I_in)
            for t in range(time_steps):
                outputs[:, t, :] = self._forward_single_step(
                    I_in[:, t, :], dI_in_dt[:, t, :], reset_states=(t == 0)
                )
            return outputs
        else:
            raise ValueError(f"Unsupported input shape: {I_in.shape}")

    def _forward_single_step(
        self,
        I_in: torch.Tensor,
        dI_in_dt: torch.Tensor | None,
        reset_states: bool,
    ) -> torch.Tensor:
        """Process a single RA step with optional derivative override."""
        batch_size, num_neurons = I_in.shape

        if self.I_RA is None or reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        if dI_in_dt is None:
            dI_in_dt = torch.zeros_like(I_in)

        # Equation 8: τ_RA * dI_RA/dt = k3 * |dI_in/dt| - I_RA
        dI_RA_dt = (self.k3 * torch.abs(dI_in_dt) - self.I_RA) / self.tau_RA
        self.I_RA = self.I_RA + dI_RA_dt * self.dt

        return self.I_RA.clone()

    def _forward_sequence(
        self,
        I_in: torch.Tensor,
        reset_states: bool,
    ) -> torch.Tensor:
        """Iterate over a temporal sequence with finite differencing."""
        batch_size, time_steps, num_neurons = I_in.shape

        if reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        # Compute derivatives using finite differences
        dI_in_dt = torch.zeros_like(I_in)
        dI_in_dt[:, 1:, :] = (I_in[:, 1:, :] - I_in[:, :-1, :]) / self.dt

        # Process each time step
        outputs = torch.zeros_like(I_in)
        for t in range(time_steps):
            outputs[:, t, :] = self._forward_single_step(
                I_in[:, t, :], dI_in_dt[:, t, :], reset_states=(t == 0)
            )

        return outputs

    def forward_edge_response(
        self,
        I_in: torch.Tensor,
        num_steps: int = 10,
        reset_states: bool = True,
    ) -> torch.Tensor:
        """
        Simulate RA response to a step change in input (0 → ``I_in``).

            Args:
                I_in: Final input current tensor ``(batch_size, num_neurons)``.
                num_steps: Number of steps to simulate the transition.
                reset_states: Whether to reset states initially.

            Returns:
                I_RA: RA filter output after simulating the edge
        """
        batch_size, num_neurons = I_in.shape

        if reset_states:
            self.reset_states(batch_size, num_neurons, I_in.device)

        # Simulate step input: 0 -> I_in over num_steps
        for step in range(num_steps):
            if step == 0:
                # First step: large positive derivative
                current_input = I_in / num_steps
                dI_in_dt = current_input / self.dt  # Large derivative
            else:
                # Subsequent steps: smaller derivatives
                current_input = I_in * (step + 1) / num_steps
                prev_input = I_in * step / num_steps
                dI_in_dt = (current_input - prev_input) / self.dt

            # Equation 8: τ_RA * dI_RA/dt = k3 * |dI_in/dt| - I_RA
            dI_RA_dt = (self.k3 * torch.abs(dI_in_dt) - self.I_RA) / self.tau_RA
            self.I_RA = self.I_RA + dI_RA_dt * self.dt

        return self.I_RA.clone()

    def forward_steady_state(self, I_in: torch.Tensor) -> torch.Tensor:
        """
        RA filters respond to derivatives, so steady-state for constant input
        is zero.  Included for API compatibility.

            Args:
                I_in: input current tensor (batch_size, num_neurons)

            Returns:
                I_RA: zero tensor (RA doesn't respond to constant input)
        """
        return torch.zeros_like(I_in)


class CombinedSARAFilter(nn.Module):
    """Composite helper that keeps SA and RA filters aligned."""

    def __init__(
        self,
        sa_params: Dict[str, float] | None = None,
        ra_params: Dict[str, float] | None = None,
    ) -> None:
        """Initialise both filters with optional parameter overrides."""
        super().__init__()

        # Default parameters from Parvizi-Fard paper
        default_sa = {"tau_r": 5, "tau_d": 30, "k1": 0.05, "k2": 3.0, "dt": 0.1}
        default_ra = {"tau_RA": 30, "k3": 2.0, "dt": 0.1}

        sa_params = sa_params or default_sa
        ra_params = ra_params or default_ra

        self.sa_filter = SAFilterTorch(**sa_params)
        self.ra_filter = RAFilterTorch(**ra_params)

    def forward(
        self,
        sa_inputs: torch.Tensor,
        ra_inputs: torch.Tensor,
        reset_states: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter SA/RA inputs in lockstep using the child filters."""
        sa_outputs = self.sa_filter(sa_inputs, reset_states=reset_states)
        ra_outputs = self.ra_filter(ra_inputs, reset_states=reset_states)

        return sa_outputs, ra_outputs

    def forward_enhanced(
        self,
        sa_inputs: torch.Tensor,
        ra_inputs: torch.Tensor,
        method: str = "multi_step",
        sa_steps: int = 50,
        ra_steps: int = 10,
        reset_states: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with better single-frame processing.

        Args:
            sa_inputs: SA neuron inputs.
            ra_inputs: RA neuron inputs.
            method: 'multi_step', 'steady_state', or 'edge_response'.
            sa_steps: Number of steps for SA multi-step processing.
            ra_steps: Number of steps for RA edge response.
            reset_states: Whether to reset filter states.

        Returns:
            sa_outputs, ra_outputs: filtered outputs
        """
        if method == "steady_state":
            # Use steady-state approximations
            sa_outputs = self.sa_filter.forward_steady_state(sa_inputs)
            ra_outputs = self.ra_filter.forward_steady_state(ra_inputs)

        elif method == "multi_step":
            # Use multi-step simulation
            sa_outputs = self.sa_filter.forward_multi_step(
                sa_inputs, num_steps=sa_steps, reset_states=reset_states
            )
            ra_outputs = self.ra_filter.forward_edge_response(
                ra_inputs, num_steps=ra_steps, reset_states=reset_states
            )

        elif method == "edge_response":
            # Edge response for both; SA still uses multi-step settling
            sa_outputs = self.sa_filter.forward_multi_step(
                sa_inputs, num_steps=sa_steps, reset_states=reset_states
            )
            ra_outputs = self.ra_filter.forward_edge_response(
                ra_inputs, num_steps=ra_steps, reset_states=reset_states
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        return sa_outputs, ra_outputs

    def reset_all_states(
        self,
        sa_batch_neurons: Tuple[int, int],
        ra_batch_neurons: Tuple[int, int],
        device: torch.device | str = "cpu",
    ) -> None:
        """Reset states for both filters using ``(batch, neurons)`` tuples."""
        sa_batch, sa_neurons = sa_batch_neurons
        ra_batch, ra_neurons = ra_batch_neurons

        self.sa_filter.reset_states(sa_batch, sa_neurons, device)
        self.ra_filter.reset_states(ra_batch, ra_neurons, device)


def compute_finite_difference_torch(current, previous, dt):
    """
    Compute finite difference derivative for PyTorch tensors.

    Args:
        current: current values
        previous: previous values (None for first step)
        dt: time step

    Returns:
        derivative: finite difference derivative
    """
    if previous is None:
        return torch.zeros_like(current)
    return (current - previous) / dt
