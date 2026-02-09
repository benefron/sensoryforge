import math
import torch
import torch.nn as nn


class SANeuronTorch(nn.Module):
    """
    Simplified Silicon (SA) neuron, faithful to paper eq.8 in normalized form.

    Normalization (current domain):
        r = I_th / I_xi,   z = I_mem / r.
        ODE: tau * dz/dt = - (1 - a) * z + I_in_net,
        where a = I_a / I_xi and I_in_net = I_in - I_ahp - I_ref.

    - Time constants from subthreshold: tau = C_mem * U_T / (kappa * I_xi).
        - Adaptation/refactory currents subtract from input and decay
            exponentially.
    - Spike when z >= z_th; on spike: z -> z_reset, inject I_th_ahp and
        I_tau_refractory into I_ahp and I_ref respectively.

    Inputs/Outputs:
        - input_current: [B,T,F] numeric; scaled to Amps by current_scale.
        - Returns (Imem_trace [B,T+1,F] in Amps, spikes [B,T+1,F] bool).

        Noise:
                        - Additive Langevin noise on z integration with
                            std=noise_std*sqrt(dt_s) (units of z). For clarity,
                            noise_std is in A/sqrt(s) on z.
    """

    def __init__(
        self,
        # Bias currents (Amperes)
        I_tau: float = 25e-12,  # 25 pA
        I_th: float = 8.3e-9,  # 8.3 nA
        I_tau_ahp: float = 20e-12,  # 20 pA
        I_th_ahp: float = 16.6e-12,  # 16.6 pA (increment on spike)
        I_tau_refractory: float = 1.6e-9,  # 1.6 nA (also injected on spike)
        # Capacitors (Farads)
        C_mem: float = 100e-15,  # 100 fF
        C_adap: float = 250e-15,  # 250 fF
        C_refractory: float = 200e-15,  # 200 fF
        # Thermal params
        U_T: float = 26e-3,  # 26 mV
        kappa: float = 0.7,
        # Eq.8 intrinsic positive feedback fraction a = I_a / I_xi
        Ia_frac: float = 0.8,  # dimensionless a in [0,1)
        # Integration and thresholds
        dt: float | None = None,  # ms; if None, auto dt = tau/20
        z_reset: float = 0.0,  # reset value for z (A)
        I_in_op: float = 500e-12,  # operating I_in (A) for z_th
        # Input scaling: numeric inputs * current_scale -> Amperes
        current_scale: float = 1e-12,  # interpret inputs as pA by default
        # Langevin noise (on z) in A/sqrt(s)
        noise_std: float = 0.0,
    ):
        super().__init__()
        # Store parameters
        self.I_tau = float(I_tau)  # I_xi
        self.I_th = float(I_th)
        self.I_tau_ahp = float(I_tau_ahp)
        self.I_th_ahp = float(I_th_ahp)
        self.I_tau_ref = float(I_tau_refractory)
        self.C_mem = float(C_mem)
        self.C_adap = float(C_adap)
        self.C_ref = float(C_refractory)
        self.U_T = float(U_T)
        self.kappa = float(kappa)
        self.a = float(Ia_frac)
        # Physical time constant (s)
        self.tau_s = (self.C_mem * self.U_T) / max(self.kappa * self.I_tau, 1e-30)
        # dt in seconds (auto from tau if not provided)
        if dt is None:
            self.dt_ms = max(1e-6, (self.tau_s * 1000.0) / 20.0)
        else:
            self.dt_ms = float(dt)
        self.dt_s = self.dt_ms * 1e-3
        # Normalization gain and threshold in normalized coordinates
        self.r = self.I_th / max(self.I_tau, 1e-30)
        # steady-state z* = I_in / (1-a); choose threshold at op point
        self.z_th = float(I_in_op) / max(1.0 - self.a, 1e-12)
        self.z_reset = float(z_reset)
        self.current_scale = float(current_scale)
        self.noise_std = float(noise_std)

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless SA neuron).

        The SA model re-initialises z, I_ahp, I_ref each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    @staticmethod
    def _param_tensor(val, shape, device, dtype):
        """Float or (mean, std) -> tensor of given shape on device/dtype."""
        if isinstance(val, tuple) and len(val) == 2:
            mean, std = val
            mean_t = torch.full(shape, float(mean), dtype=dtype, device=device)
            std_t = torch.full(shape, float(std), dtype=dtype, device=device)
            return torch.normal(mean=mean_t, std=std_t)
        return torch.full(shape, float(val), dtype=dtype, device=device)

    def forward(
        self,
        input_current: torch.Tensor,
        *,
        # Optional per-call overrides (float or (mean,std) sampled per-feature)
        I_tau=None,
        I_th=None,
        I_tau_ahp=None,
        I_th_ahp=None,
        I_tau_refractory=None,
        C_mem=None,
        C_adap=None,
        C_refractory=None,
        U_T=None,
        kappa=None,
        Ia_frac=None,
        z_reset=None,
        I_in_op=None,
        current_scale=None,
        reset_states: bool = True,
    ):
        """
        Args:
            input_current: [batch, steps, features] numeric. Interpreted as
                currents in Amperes after scaling by current_scale.
            Optional overrides: pass floats or (mean,std) to sample per-feature
                values for parameters.
            reset_states: If True, resets state at start of sequence.
        Returns:
            Imem_trace (A), spikes: both [batch, steps+1, features]
        """
        assert input_current.dim() == 3, "input_current must be [B,T,F]"
        B, T, F = input_current.shape
        device, dtype = input_current.device, input_current.dtype

        # Resolve per-feature parameters for time constants (allow overrides)
        I_tau_f = self._param_tensor(
            self.I_tau if I_tau is None else I_tau,
            (F,),
            device,
            dtype,
        )
        I_tau_ahp_f = self._param_tensor(
            self.I_tau_ahp if I_tau_ahp is None else I_tau_ahp,
            (F,),
            device,
            dtype,
        )
        I_tau_ref_f = self._param_tensor(
            self.I_tau_ref if I_tau_refractory is None else I_tau_refractory,
            (F,),
            device,
            dtype,
        )
        C_mem_f = self._param_tensor(
            self.C_mem if C_mem is None else C_mem,
            (F,),
            device,
            dtype,
        )
        C_adap_f = self._param_tensor(
            self.C_adap if C_adap is None else C_adap,
            (F,),
            device,
            dtype,
        )
        C_ref_f = self._param_tensor(
            self.C_ref if C_refractory is None else C_refractory,
            (F,),
            device,
            dtype,
        )
        U_T_f = self._param_tensor(
            self.U_T if U_T is None else U_T,
            (F,),
            device,
            dtype,
        )
        kappa_f = self._param_tensor(
            self.kappa if kappa is None else kappa,
            (F,),
            device,
            dtype,
        )
        a_val = self.a if Ia_frac is None else float(Ia_frac)
        z_reset_val = self.z_reset if z_reset is None else float(z_reset)
        z_th_val = (
            self.z_th if I_in_op is None else (float(I_in_op) / max(1.0 - a_val, 1e-12))
        )
        scale = self.current_scale if current_scale is None else float(current_scale)

        # Time constants (s) per feature
        tau_m_s = C_mem_f * U_T_f / torch.clamp(kappa_f * I_tau_f, min=1e-30)
        tau_ahp_s = C_adap_f * U_T_f / torch.clamp(kappa_f * I_tau_ahp_f, min=1e-30)
        tau_ref_s = C_ref_f * U_T_f / torch.clamp(kappa_f * I_tau_ref_f, min=1e-30)

        # Broadcast helpers [1,F]
        tau_m_b = tau_m_s.view(1, -1)
        tau_ahp_b = tau_ahp_s.view(1, -1)
        tau_ref_b = tau_ref_s.view(1, -1)
        I_th_ahp_b = torch.full(
            (1, F),
            self.I_th_ahp if I_th_ahp is None else float(I_th_ahp),
            dtype=dtype,
            device=device,
        )
        I_tau_ref_b = torch.full(
            (1, F),
            self.I_tau_ref if I_tau_refractory is None else float(I_tau_refractory),
            dtype=dtype,
            device=device,
        )
        one_minus_a = torch.full(
            (1, F),
            max(1.0 - a_val, 1e-12),
            dtype=dtype,
            device=device,
        )
        z_th_b = torch.full((1, F), z_th_val, dtype=dtype, device=device)

        # States (z in A since z = Imem/r with r dimensionless)
        z = torch.zeros((B, F), dtype=dtype, device=device)
        I_ahp = torch.zeros((B, F), dtype=dtype, device=device)
        I_ref = torch.zeros((B, F), dtype=dtype, device=device)

        # Outputs (Imem trace = z * r)
        imem_trace = torch.zeros((B, T + 1, F), dtype=dtype, device=device)
        spikes = torch.zeros((B, T + 1, F), dtype=torch.bool, device=device)
        # initial Imem
        imem_trace[:, 0, :] = z * self.r

        sqrt_dt_s = math.sqrt(max(self.dt_s, 1e-12))

        for t in range(T):
            # Scale input (numeric) to Amps and subtract adaptation/refractory
            I_in = input_current[:, t, :] * scale
            I_net = I_in - I_ahp - I_ref

            # Normalized ODE: tau * dz/dt = - (1 - a) * z + I_net
            dz_dt = (-(one_minus_a * z) + I_net) / torch.clamp(tau_m_b, min=1e-12)
            if self.noise_std != 0.0:
                eta = torch.randn_like(z) * (self.noise_std * sqrt_dt_s)
                z_next = z + self.dt_s * dz_dt + eta
            else:
                z_next = z + self.dt_s * dz_dt

            # Spike: threshold in normalized domain
            fired = z_next >= z_th_b
            # Reset z
            z_next = torch.where(fired, torch.full_like(z_next, z_reset_val), z_next)

            # Update AHP and refractory currents (decay + injection on spike)
            I_ahp_next = I_ahp + self.dt_s * (
                -(I_ahp / torch.clamp(tau_ahp_b, min=1e-12))
            )
            I_ref_next = I_ref + self.dt_s * (
                -(I_ref / torch.clamp(tau_ref_b, min=1e-12))
            )
            I_ahp_next = torch.where(fired, I_ahp_next + I_th_ahp_b, I_ahp_next)
            I_ref_next = torch.where(fired, I_ref_next + I_tau_ref_b, I_ref_next)

            # Commit state
            z = z_next
            I_ahp = I_ahp_next
            I_ref = I_ref_next

            # Record (Imem = r * z)
            imem_trace[:, t + 1, :] = z * self.r
            spikes[:, t, :] = fired
            spikes[:, t + 1, :] = fired

        return imem_trace, spikes
