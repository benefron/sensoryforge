"""Tests for _dynamics() extraction and solver parameter in neuron models (item 8).

Pins current behavior before the refactor so that any regression is immediately
visible. After the refactor all tests must still pass.

Covers:
- Each model exposes a _dynamics() method returning (dv, du) or (dv, dw)
- _dynamics() values match the closed-form equations in the class docstring
- forward() with no solver matches forward() with explicit EulerSolver
- forward() accepts solver= keyword argument without crashing
"""

import math
import pytest
import torch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch
from sensoryforge.solvers.euler import EulerSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 0.05  # ms — safe Euler step


def _run(model, drive_amplitude: float = 15.0, steps: int = 200) -> tuple:
    """Run a neuron model with constant suprathreshold drive."""
    drive = torch.full((1, steps, 1), float(drive_amplitude))
    return model(drive)


# ---------------------------------------------------------------------------
# 1. _dynamics() method exists and returns the correct shapes
# ---------------------------------------------------------------------------

class TestDynamicsMethodExists:
    def test_izhikevich_has_dynamics(self):
        model = IzhikevichNeuronTorch(dt=DT)
        assert hasattr(model, "_dynamics"), (
            "IzhikevichNeuronTorch must expose _dynamics()"
        )

    def test_adex_has_dynamics(self):
        model = AdExNeuronTorch(dt=DT)
        assert hasattr(model, "_dynamics"), (
            "AdExNeuronTorch must expose _dynamics()"
        )

    def test_mqif_has_dynamics(self):
        model = MQIFNeuronTorch(dt=DT)
        assert hasattr(model, "_dynamics"), (
            "MQIFNeuronTorch must expose _dynamics()"
        )


# ---------------------------------------------------------------------------
# 2. _dynamics() values match the closed-form equations
# ---------------------------------------------------------------------------

class TestDynamicsValues:
    def test_izhikevich_dv_formula(self):
        """dv = 0.04*v^2 + 5*v + 140 - u + I_ext (subthreshold)."""
        model = IzhikevichNeuronTorch(dt=DT, a=0.02, b=0.2)
        v = torch.tensor([[-60.0]])
        u = torch.tensor([[-14.0]])  # b * v_rest
        I = torch.tensor([[5.0]])
        dv, du = model._dynamics(v, u, I)

        expected_dv = 0.04 * (-60.0)**2 + 5 * (-60.0) + 140 - (-14.0) + 5.0
        assert abs(dv.item() - expected_dv) < 1e-3, (
            f"Izhikevich dv={dv.item():.4f}, expected {expected_dv:.4f}"
        )

    def test_izhikevich_du_formula(self):
        """du = a*(b*v - u)."""
        model = IzhikevichNeuronTorch(dt=DT, a=0.02, b=0.2)
        v = torch.tensor([[-60.0]])
        u = torch.tensor([[-14.0]])
        I = torch.tensor([[0.0]])
        _, du = model._dynamics(v, u, I)

        expected_du = 0.02 * (0.2 * (-60.0) - (-14.0))
        assert abs(du.item() - expected_du) < 1e-6, (
            f"Izhikevich du={du.item():.6f}, expected {expected_du:.6f}"
        )

    def test_adex_dv_formula(self):
        """dv = (-(v-EL) + DeltaT*exp((v-VT)/DeltaT) - w + R*I) / tau_m."""
        model = AdExNeuronTorch(dt=DT)
        v = torch.tensor([[-65.0]])
        w = torch.tensor([[0.0]])
        I = torch.tensor([[10.0]])
        dv, dw = model._dynamics(v, w, I)

        EL, VT, DeltaT, R, tau_m = (
            model.EL, model.VT, model.DeltaT, model.R, model.tau_m
        )
        exp_term = DeltaT * math.exp((v.item() - VT) / DeltaT)
        expected_dv = (-(v.item() - EL) + exp_term - w.item() + R * I.item()) / tau_m
        assert abs(dv.item() - expected_dv) < 1e-3, (
            f"AdEx dv={dv.item():.4f}, expected {expected_dv:.4f}"
        )

    def test_adex_dw_formula(self):
        """dw = (a*(v-EL) - w) / tau_w."""
        model = AdExNeuronTorch(dt=DT)
        v = torch.tensor([[-65.0]])
        w = torch.tensor([[5.0]])
        I = torch.tensor([[0.0]])
        _, dw = model._dynamics(v, w, I)

        expected_dw = (model.a * (v.item() - model.EL) - w.item()) / model.tau_w
        assert abs(dw.item() - expected_dw) < 1e-6, (
            f"AdEx dw={dw.item():.6f}, expected {expected_dw:.6f}"
        )

    def test_mqif_dv_formula(self):
        """dv = (a*(v-vr)*(v-vt) - u + I) / tau_m."""
        model = MQIFNeuronTorch(dt=DT)
        v = torch.tensor([[-70.0]])
        u = torch.tensor([[0.0]])
        I = torch.tensor([[5.0]])
        dv, du = model._dynamics(v, u, I)

        quad = model.a * (v.item() - model.vr) * (v.item() - model.vt)
        expected_dv = (quad - u.item() + I.item()) / model.tau_m
        assert abs(dv.item() - expected_dv) < 1e-3, (
            f"MQIF dv={dv.item():.4f}, expected {expected_dv:.4f}"
        )

    def test_mqif_du_formula(self):
        """du = (b*(v-vr) - u) / tau_u."""
        model = MQIFNeuronTorch(dt=DT)
        v = torch.tensor([[-70.0]])
        u = torch.tensor([[0.0]])
        I = torch.tensor([[0.0]])
        _, du = model._dynamics(v, u, I)

        expected_du = (model.b * (v.item() - model.vr) - u.item()) / model.tau_u
        assert abs(du.item() - expected_du) < 1e-6, (
            f"MQIF du={du.item():.6f}, expected {expected_du:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. forward() with no solver is identical to pre-refactor behavior
# ---------------------------------------------------------------------------

class TestForwardUnchanged:
    """forward() must return the same result with and without explicit solver."""

    @pytest.mark.parametrize("model_cls,amplitude", [
        (IzhikevichNeuronTorch, 20.0),
        (AdExNeuronTorch, 500.0),   # AdEx rheobase is ~100+ with default R=1.0
        (MQIFNeuronTorch, 100.0),
    ])
    def test_forward_no_solver_produces_spikes(self, model_cls, amplitude):
        """forward() without solver still produces spikes as before."""
        model = model_cls(dt=DT)
        v_trace, spikes = _run(model, drive_amplitude=amplitude)
        assert spikes.any(), (
            f"{model_cls.__name__}: no spikes produced with amplitude={amplitude}"
        )

    @pytest.mark.parametrize("model_cls,kwargs", [
        (IzhikevichNeuronTorch, {}),
        (AdExNeuronTorch, {}),
        (MQIFNeuronTorch, {}),
    ])
    def test_forward_with_euler_solver_matches_default(self, model_cls, kwargs):
        """forward(x, solver=EulerSolver) must match forward(x) exactly."""
        model = model_cls(dt=DT, **kwargs)
        drive = torch.full((1, 100, 1), 20.0)

        v1, s1 = model(drive)
        solver = EulerSolver(dt=DT)
        v2, s2 = model(drive, solver=solver)

        assert torch.allclose(v1, v2, atol=1e-5), (
            f"{model_cls.__name__}: v_trace differs when solver=EulerSolver"
        )
        assert (s1 == s2).all(), (
            f"{model_cls.__name__}: spikes differ when solver=EulerSolver"
        )


# ---------------------------------------------------------------------------
# 4. forward() accepts solver= kwarg without crashing (interface test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [
    IzhikevichNeuronTorch,
    AdExNeuronTorch,
    MQIFNeuronTorch,
])
def test_forward_accepts_solver_kwarg(model_cls):
    """Passing solver=EulerSolver must not raise any exception."""
    model = model_cls(dt=DT)
    drive = torch.full((1, 50, 1), 20.0)
    solver = EulerSolver(dt=DT)
    # Should not raise
    v_trace, spikes = model(drive, solver=solver)
    assert v_trace.shape == (1, 51, 1), (
        f"{model_cls.__name__}: unexpected v_trace shape {v_trace.shape}"
    )
