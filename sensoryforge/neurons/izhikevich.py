import math
import torch
import torch.nn as nn


class IzhikevichNeuronTorch(nn.Module):
    r"""Project-compatible Izhikevich neuron with optional parameter noise.

    Continuous dynamics per feature ``f`` and time ``t``:

    - Membrane: ``dv/dt = 0.04 v^2 + 5 v + 140 - u + I_t``
    - Recovery: ``du/dt = a (b v - u)``

    Forward Euler integrates both states via
    ``state_{t+1} = state_t + dt * dstate/dt``.
    Additive Langevin noise ``η_t ~ N(0, noise_std · sqrt(dt))`` perturbs
    ``v`` before threshold detection.

    When ``v >= threshold`` the neuron emits a spike, resets ``v`` to ``c`` and
    increments ``u`` by ``d`` (with broadcast over batch/features). Parameters
    ``a, b, c, d, threshold`` may be floats or ``(mean, std)`` tuples that are
    sampled per feature each forward pass.

    Inputs ``I`` use shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(v_trace, spikes)`` where ``v_trace`` tracks
    voltages for ``steps+1`` samples and ``spikes`` is a boolean tensor of the
    same shape showing threshold crossings.
    """

    def __init__(
        self,
        a=0.02,
        b=0.2,
        c=-65.0,
        d=8.0,
        v_init=-65.0,
        u_init=None,
        dt=0.05,
        threshold=30.0,
        a_std=0.0,
        b_std=0.0,
        c_std=0.0,
        d_std=0.0,
        threshold_std=0.0,
        seed=None,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_init = v_init
        # Handle tuple params: use mean for u_init calculation
        b_val = b[0] if isinstance(b, tuple) else b
        self.u_init = b_val * v_init if u_init is None else u_init
        self.dt = dt
        self.threshold = threshold
        self.a_std = a_std
        self.b_std = b_std
        self.c_std = c_std
        self.d_std = d_std
        self.threshold_std = threshold_std
        self.seed = seed
        # Langevin noise intensity (additive, mV/sqrt(ms))
        self.noise_std = noise_std

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless Izhikevich).

        The Izhikevich model re-initialises v and u each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    def forward(
        self,
        input_current,
        a=None,
        b=None,
        c=None,
        d=None,
        threshold=None,
    ):
        """
        input_current: torch.Tensor, shape [batch, steps, features]
        Optional per-neuron parameters a, b, c, d: each can be a float or
        (mean, std) tuple.
        Returns:
            v_trace: [batch, steps+1, features]
            spikes: [batch, steps+1, features] (bool)
        """
        batch, steps, features = input_current.shape
        device = input_current.device
        dtype = input_current.dtype

        # Helper to sample per-neuron parameters if (mean, std) tuple is given
        def get_param(val, shape):
            if isinstance(val, tuple) and len(val) == 2:
                mean, std = val
                return torch.normal(
                    mean=torch.full(shape, mean, dtype=dtype, device=device),
                    std=torch.full(shape, std, dtype=dtype, device=device),
                )
            else:
                # Handle tensor/list inputs that might need broadcasting or are already correct
                if val is not None:
                    if not torch.is_tensor(val):
                        try:
                            t_val = torch.tensor(val, dtype=dtype, device=device)
                        except Exception:
                            # Fallback for scalar float/int
                            t_val = torch.tensor(
                                float(val), dtype=dtype, device=device
                            )
                    else:
                        t_val = val.to(dtype=dtype, device=device)

                    if t_val.ndim == 0:
                        return t_val.expand(shape)
                    if t_val.shape == shape:
                        return t_val
                    # Try broadcasting
                    try:
                        return t_val.expand(shape)
                    except RuntimeError:
                        # If shapes don't match and can't broadcast, we have a problem.
                        # But let's return t_val and let the caller crash or handle it,
                        # or try to force it if it's a size mismatch issue.
                        pass

                return torch.full(
                    shape,
                    val if val is not None else 0.0,
                    dtype=dtype,
                    device=device,
                )

        # Use provided or default parameters
        a_val = a if a is not None else self.a
        b_val = b if b is not None else self.b
        c_val = c if c is not None else self.c
        d_val = d if d is not None else self.d
        threshold_val = threshold if threshold is not None else self.threshold
        a_tensor = get_param(a_val, (features,))
        b_tensor = get_param(b_val, (features,))
        c_tensor = get_param(c_val, (features,))
        d_tensor = get_param(d_val, (features,))
        threshold_tensor = get_param(threshold_val, (features,))

        v = torch.full((batch, features), self.v_init, dtype=dtype, device=device)
        # Fix tuple-b u_init: torch.full requires scalar, use expand for tensor
        # (resolves ReviewFinding#M1)
        if isinstance(self.b, tuple):
            u = (b_tensor * self.v_init).unsqueeze(0).expand(batch, features).clone()
        else:
            u = torch.full((batch, features), self.u_init, dtype=dtype, device=device)
        v_trace = torch.zeros((batch, steps + 1, features), dtype=dtype, device=device)
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        v_trace[:, 0, :] = v

        for t in range(steps):
            fired = v >= threshold_tensor.unsqueeze(0).expand_as(v)
            v_vis = v.clone()
            v_vis[fired] = threshold_tensor.unsqueeze(0).expand_as(v)[fired]
            v_trace[:, t, :] = v_vis
            spikes[:, t, :] = fired
            v_next = torch.where(fired, c_tensor.unsqueeze(0).expand_as(v), v)
            u_next = torch.where(fired, u + d_tensor.unsqueeze(0).expand_as(u), u)
            not_fired = ~fired
            dv = 0.04 * v**2 + 5 * v + 140 - u + input_current[:, t, :]
            du = a_tensor.unsqueeze(0) * (b_tensor.unsqueeze(0) * v - u)
            # Add Langevin noise when integrating v (not during reset)
            if self.noise_std != 0.0:
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            u_next = torch.where(not_fired, u + self.dt * du, u_next)
            v = v_next
            u = u_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= threshold_tensor.unsqueeze(0).expand_as(v)
        return v_trace, spikes
