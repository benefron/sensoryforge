"""The event converters' contract with pressure-simulation (its brief, 2.4), run here.

SensoryForge v1.3.0 gives the signed level-crossing unit an optional leaky
reference (``reference_leak_tau_ms``) and records in every bundle the input
floor each population's converter received (schema 2.3.0). Pressure-simulation
emulates both converters bit for bit and decodes their events, so each test
below pins one statement of ``docs/reference/converter_contract.md`` and
carries the brief's number. Known answers are exact (or to the precision the
brief states); nothing here is fitted.

Test 1's engine golden was recorded on an export of tag v1.2.1 by
``tests/fixtures/make_converter_v1_2_1_golden.py``. **A failure in test 1
means an old design's events changed.** Do not re-record the golden and do
not loosen a tolerance: fix the change so that a unit with the leak off runs
as v1.2.1 did.
"""

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.cli import cmd_run, create_parser
from sensoryforge.config.defaults import resolve_input_floor
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import SCHEMA_VERSION
from sensoryforge.io.design import load_design
from sensoryforge.neurons.event_encoders import LevelCrossingNeuron, SigmaDeltaNeuron
from sensoryforge.testing.contracts import check_component
from sensoryforge.testing.golden import assert_matches_golden

h5py = pytest.importorskip("h5py")

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "converter_v1_2_1"
GOLDEN_DESIGN = GOLDEN_DIR / "design"
GOLDEN = json.loads((GOLDEN_DIR / "golden.json").read_text())
ON_REFERENCE_PLATFORM = (
    sys.platform == GOLDEN["platform"]["sys_platform"]
    and platform.machine() == GOLDEN["platform"]["machine"]
    and torch.__version__ == GOLDEN["platform"]["torch"]
)

DELTA = 0.05  # ms: the converter sub-step pressure-simulation uses
CROSSING = 1.0 - 1e-5  # a crossing is |d| >= theta * (1 - 1e-5)
#: The unit's own refusal, not a TypeError for an unknown keyword.
LEAK_REFUSED = r"reference_leak_tau_ms must be None \(no leak\) or a finite number"


# --------------------------------------------------------------------------- #
# v1.2.1's two forward loops, frozen verbatim (test 1). Never edit these.
# --------------------------------------------------------------------------- #

_V121_CROSSING_EPS = 1e-5


def _v121_refractory_steps(refractory_ms: float, dt: float) -> int:
    """Refractory period in integration steps (0 = none, else >= 1)."""
    if refractory_ms <= 0.0:
        return 0
    return max(1, int(round(refractory_ms / dt)))


def _v121_level_crossing_forward(self, input_current):
    """``LevelCrossingNeuron.forward`` at tag v1.2.1 (0fafeff), verbatim."""
    if input_current.dim() != 3:
        raise ValueError(
            "Expected 3-D input [batch, steps, N], got shape "
            f"{list(input_current.shape)}"
        )
    batch, steps, n = input_current.shape
    device = input_current.device
    dtype = input_current.dtype if input_current.is_floating_point() else torch.float32
    x_all = input_current.to(dtype)
    if self.noise_std > 0.0:
        x_all = x_all + torch.randn_like(x_all) * self.noise_std

    if self.initial_reference == "first" and steps > 0:
        ref = x_all[:, 0, :].clone()
    else:
        ref = torch.zeros((batch, n), dtype=dtype, device=device)

    ref_trace = torch.empty((batch, steps + 1, n), dtype=dtype, device=device)
    events = torch.zeros((batch, steps + 1, n), dtype=torch.int16, device=device)
    ref_trace[:, 0, :] = ref

    theta = self.theta
    ref_steps = _v121_refractory_steps(self.refractory_ms, self.dt)
    blocked = (
        torch.zeros((batch, n), dtype=torch.int64, device=device)
        if ref_steps > 0
        else None
    )
    for t in range(steps):
        d = x_all[:, t, :] - ref
        magnitude = torch.floor(d.abs() / theta + _V121_CROSSING_EPS)
        if blocked is None:
            k = torch.sign(d) * magnitude
        else:
            allowed = blocked == 0
            k = torch.sign(d) * ((magnitude >= 1) & allowed).to(dtype)
            fired = k != 0
            blocked = torch.where(
                fired,
                torch.full_like(blocked, ref_steps - 1),
                (blocked - 1).clamp(min=0),
            )
        ref = ref + k * theta
        ref_trace[:, t + 1, :] = ref
        events[:, t + 1, :] = k.to(torch.int16)
    return ref_trace, events


def _v121_sigma_delta_forward(self, input_current):
    """``SigmaDeltaNeuron.forward`` at tag v1.2.1 (0fafeff), verbatim."""
    if input_current.dim() != 3:
        raise ValueError(
            "Expected 3-D input [batch, steps, N], got shape "
            f"{list(input_current.shape)}"
        )
    batch, steps, n = input_current.shape
    device = input_current.device
    dtype = input_current.dtype if input_current.is_floating_point() else torch.float32
    x_all = input_current.to(dtype)
    if self.noise_std > 0.0:
        x_all = x_all + torch.randn_like(x_all) * self.noise_std

    u = torch.zeros((batch, n), dtype=dtype, device=device)
    u_trace = torch.empty((batch, steps + 1, n), dtype=dtype, device=device)
    spikes = torch.zeros((batch, steps + 1, n), dtype=torch.int16, device=device)
    u_trace[:, 0, :] = u

    dt = self.dt
    theta = self.theta
    decay = None if self.leak_tau_ms is None else dt / self.leak_tau_ms
    ref_steps = _v121_refractory_steps(self.refractory_ms, dt)
    blocked = (
        torch.zeros((batch, n), dtype=torch.int64, device=device)
        if ref_steps > 0
        else None
    )
    for t in range(steps):
        if decay is None:
            u = u + dt * x_all[:, t, :]
        else:
            u = u + dt * x_all[:, t, :] - decay * u
        count = torch.floor(u / theta + _V121_CROSSING_EPS).clamp(min=0)
        if blocked is not None:
            allowed = blocked == 0
            count = ((count >= 1) & allowed).to(dtype)
            blocked = torch.where(
                count > 0,
                torch.full_like(blocked, ref_steps - 1),
                (blocked - 1).clamp(min=0),
            )
        u = u - count * theta
        if blocked is not None:
            u = u.clamp(max=theta)
        u_trace[:, t + 1, :] = u
        spikes[:, t + 1, :] = count.to(torch.int16)
    return u_trace, spikes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _seeded_drives(theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Seeded drives ``[4, 400, 3]``: ramps, steps, a signed multi-scale drive, noise.

    Batch 0 holds ramps (one rising by exactly ``theta`` per step, one falling
    slowly, one rising fast then falling), batch 1 steps of several sizes and
    signs, batch 2 a signed multi-scale drive, batch 3 white noise of std
    ``0.3 * theta`` around three levels. Units are the drive's (mA).
    """
    gen = torch.Generator().manual_seed(20261005)
    steps = 400
    n = torch.arange(steps, dtype=torch.float64)
    ramps = torch.stack(
        [
            n * theta,
            -0.037 * theta * n,
            torch.where(n < 150, 0.9 * theta * n, 0.9 * theta * (300 - n)),
        ],
        dim=-1,
    )
    jumps = torch.zeros(steps, 3, dtype=torch.float64)
    sizes = torch.randn(12, 3, generator=gen, dtype=torch.float64) * 3.0 * theta
    for j in range(12):
        jumps[j * 33 :, :] += sizes[j]
    t = n.unsqueeze(-1) * 0.05
    freqs = torch.tensor([0.31, 2.9, 17.0], dtype=torch.float64)
    multi = (
        4.0 * theta * torch.sin(2 * math.pi * freqs[0] * t / 10.0)
        + 1.5 * theta * torch.sin(2 * math.pi * freqs[1] * t + 0.4)
        + 0.6 * theta * torch.sin(2 * math.pi * freqs[2] * t + 1.1)
    ).expand(steps, 3) * torch.tensor([1.0, -0.7, 2.3], dtype=torch.float64)
    levels = torch.tensor([0.0, 2.2 * theta, -5.1 * theta], dtype=torch.float64)
    noise = levels + 0.3 * theta * torch.randn(
        steps, 3, generator=gen, dtype=torch.float64
    )
    return torch.stack([ramps, jumps, multi, noise]).to(dtype)


def _first_event(events: torch.Tensor) -> int:
    """Sub-step (1-based) of a 1-D event trace's first event, or -1."""
    nonzero = torch.nonzero(events[1:]).flatten()
    return int(nonzero[0]) + 1 if nonzero.numel() else -1


def _run_design_cli(design_dir: Path, bundle_dir: Path, stimulus: str, duration):
    args = create_parser().parse_args(
        [
            "run",
            "--design",
            str(design_dir),
            "--stimulus",
            stimulus,
            "--duration",
            str(duration),
            "--bundle",
            str(bundle_dir),
        ]
    )
    assert cmd_run(args) == 0
    return bundle_dir


def _bundle_arrays(bundle_dir: Path) -> dict:
    arrays = {}
    with h5py.File(bundle_dir / "data.h5", "r") as f:
        for name, group in f["populations"].items():
            for key in group:
                arrays[f"{name}__{key}"] = group[key][()]
    return arrays


def _bundle_entries(bundle_dir: Path) -> dict:
    cfg = json.loads((bundle_dir / "config.json").read_text())
    return cfg, {p["name"]: p for p in cfg["populations"]}


def _copy_golden_design(tmp_path: Path) -> Path:
    design = tmp_path / "design"
    shutil.copytree(GOLDEN_DESIGN, design)
    return design


def _edit_design(design: Path, edit) -> None:
    manifest = json.loads((design / "design.json").read_text())
    edit(manifest)
    (design / "design.json").write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# 1. Leak off is v1.2.1, bit for bit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("initial_reference", ["zero", "first"])
@pytest.mark.parametrize("refractory_ms", [0.0, 2.0])
@pytest.mark.parametrize("spelling", ["absent", "none"])
def test_1_leak_off_is_v1_2_1_bit_for_bit_unit(
    dtype, initial_reference, refractory_ms, spelling
):
    theta = 0.37
    kwargs = dict(
        dt=DELTA,
        theta=theta,
        refractory_ms=refractory_ms,
        initial_reference=initial_reference,
    )
    if spelling == "none":
        kwargs["reference_leak_tau_ms"] = None
    model = LevelCrossingNeuron(**kwargs)
    x = _seeded_drives(theta, dtype)
    ref_now, ev_now = model(x)
    ref_then, ev_then = _v121_level_crossing_forward(model, x)
    assert ref_now.dtype == ref_then.dtype == dtype
    assert torch.equal(ref_now, ref_then)
    assert torch.equal(ev_now, ev_then)
    assert int(ev_now.abs().sum()) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("leak_tau_ms", [None, 200.0])
@pytest.mark.parametrize("refractory_ms", [0.0, 2.0])
def test_1_sigma_delta_is_v1_2_1_bit_for_bit_unit(dtype, leak_tau_ms, refractory_ms):
    theta = 0.8
    model = SigmaDeltaNeuron(
        dt=DELTA, theta=theta, leak_tau_ms=leak_tau_ms, refractory_ms=refractory_ms
    )
    x = _seeded_drives(theta, dtype).abs() * 5.0
    u_now, sp_now = model(x)
    u_then, sp_then = _v121_sigma_delta_forward(model, x)
    assert torch.equal(u_now, u_then)
    assert torch.equal(sp_now, sp_then)
    assert int(sp_now.sum()) > 0


@pytest.mark.parametrize("cls", [LevelCrossingNeuron, SigmaDeltaNeuron])
def test_1_comparator_noise_is_drawn_as_in_v1_2_1(cls):
    frozen = {
        LevelCrossingNeuron: _v121_level_crossing_forward,
        SigmaDeltaNeuron: _v121_sigma_delta_forward,
    }[cls]
    model = cls(dt=DELTA, theta=0.5, noise_std=0.15)
    x = _seeded_drives(0.5, torch.float32).abs()
    torch.manual_seed(7)
    state_now, ev_now = model(x)
    torch.manual_seed(7)
    state_then, ev_then = frozen(model, x)
    assert torch.equal(state_now, state_then)
    assert torch.equal(ev_now, ev_then)


@pytest.mark.parametrize("leak_spelling", ["absent", "null"])
def test_1_leak_off_is_v1_2_1_bit_for_bit_engine(tmp_path, leak_spelling):
    design = _copy_golden_design(tmp_path)
    if leak_spelling == "null":

        def edit(manifest):
            for prec in manifest["populations"]:
                if prec["neuron_model"] == "level_crossing":
                    prec["model_params"]["reference_leak_tau_ms"] = None

        _edit_design(design, edit)
    bundle = _run_design_cli(
        design, tmp_path / "bundle", GOLDEN["stimulus"], GOLDEN["duration_ms"]
    )
    arrays = _bundle_arrays(bundle)
    golden = np.load(GOLDEN_DIR / "golden.npz")
    assert sorted(arrays) == sorted(golden.files) == sorted(GOLDEN["arrays"])
    for key in golden.files:
        got, want = arrays[key], golden[key]
        assert got.dtype == want.dtype, key
        if ON_REFERENCE_PLATFORM:
            assert np.array_equal(got, want), key
            digest = hashlib.sha256(got.tobytes()).hexdigest()
            assert digest == GOLDEN["arrays"][key]["sha256"], key
        else:
            assert_matches_golden(got, want, what=key)


# --------------------------------------------------------------------------- #
# 2. A steady slope (known answer)
# --------------------------------------------------------------------------- #

SLOPE_MULTIPLES = (1.3, 2.0, 3.7, 7.1, 20.0)


@pytest.mark.parametrize("theta", [0.3, 1.7])
@pytest.mark.parametrize("tau_over_delta", [2.5, 10.0, 64.0, 400.0])
def test_2_a_steady_slope_fires_iff_faster_than_theta_over_tau(theta, tau_over_delta):
    tau = tau_over_delta * DELTA
    lam = DELTA / tau
    rheo = theta / tau
    slopes = [rheo * (1 - 1e-3), rheo * (1 + 1e-3)] + [
        rheo * m for m in SLOPE_MULTIPLES
    ]
    steps = int(math.ceil(50 * tau / DELTA))
    n = torch.arange(1, steps + 1, dtype=torch.float64).unsqueeze(-1)
    s = torch.tensor(slopes, dtype=torch.float64)
    x = (n * DELTA * s).unsqueeze(0)  # x_n = s * n * delta, [1, steps, slopes]
    model = LevelCrossingNeuron(dt=DELTA, theta=theta, reference_leak_tau_ms=tau)
    ref, ev = model(x)

    assert int(ev[0, :, 0].abs().sum()) == 0, "s just below theta/tau fired"
    assert _first_event(ev[0, :, 1]) > 0, "s just above theta/tau never fired"

    for j, multiple in enumerate(SLOPE_MULTIPLES, start=2):
        slope = slopes[j]
        ratio = math.log(1 - theta * CROSSING / (slope * tau)) / math.log1p(-lam)
        assert abs(ratio - round(ratio)) > 1e-9, "choose another slope"
        n_star = math.ceil(ratio)
        first = _first_event(ev[0, :, j])
        assert first == n_star, (multiple, first, n_star)
        assert int(ev[0, first, j]) > 0
        # Before the first event: u_n = (1 - lam) s tau (1 - (1 - lam)^n).
        for k in range(1, n_star):
            u = float(x[0, k - 1, j] - ref[0, k, j])
            want = (1 - lam) * slope * tau * -math.expm1(k * math.log1p(-lam))
            assert u == pytest.approx(want, rel=1e-9), (multiple, k)
        # The continuous interval is the delta -> 0 limit of n_star * delta.
        t_cont = -tau * math.log(1 - theta * CROSSING / (slope * tau))
        assert t_cont * (1 - lam) <= n_star * DELTA < t_cont + DELTA


@pytest.mark.parametrize("tau", [2.0, 8.0])
def test_2_with_held_bins_the_threshold_is_theta_one_minus_rho_over_dt(tau):
    """The engine holds each bin's drive over n_sub steps (converter contract, 2).

    Over bins without an event the difference is ``rho * u + jump`` per bin,
    ``rho = (1 - delta / tau) ** n_sub``, so a slope ``s`` (per ms, stepped once
    per bin) fires iff ``s * dt_ms / (1 - rho) >= theta * (1 - 1e-5)``.
    """
    theta, dt_ms = 0.3, 1.0
    n_sub = round(dt_ms / DELTA)
    rho = (1 - DELTA / tau) ** n_sub
    edge = theta * CROSSING * (1 - rho) / dt_ms
    assert edge < theta / tau
    bins = int(50 * tau / dt_ms)
    k = torch.arange(1, bins + 1, dtype=torch.float64).unsqueeze(-1)
    s = torch.tensor([edge * (1 - 1e-3), edge * (1 + 1e-3)], dtype=torch.float64)
    x = (k * dt_ms * s).repeat_interleave(n_sub, dim=0).unsqueeze(0)
    _, ev = LevelCrossingNeuron(dt=DELTA, theta=theta, reference_leak_tau_ms=tau)(x)
    assert int(ev[0, :, 0].abs().sum()) == 0
    fired = torch.nonzero(ev[0, :, 1]).flatten()
    assert fired.numel() > 0
    # With no dead time, events fall only on a bin's first step.
    assert torch.all((fired - 1) % n_sub == 0)


# --------------------------------------------------------------------------- #
# 3. A step fires floor(dx / theta) events, then falls silent
# --------------------------------------------------------------------------- #

STEP_RATIOS = (3.0, 3.5, 0.99, 1.0, -2.0, -4.25)


@pytest.mark.parametrize("tau", [None, 2.0])
def test_3_a_step_fires_its_quanta_at_once_then_falls_silent(tau):
    theta = 0.25
    hold = int(50 * 2.0 / DELTA)
    dx = torch.tensor(STEP_RATIOS, dtype=torch.float32) * theta
    x = dx.view(1, 1, -1).expand(1, hold, -1).contiguous()
    model = LevelCrossingNeuron(dt=DELTA, theta=theta, reference_leak_tau_ms=tau)
    _, ev = model(x)
    for j, ratio in enumerate(STEP_RATIOS):
        want = int(math.copysign(math.floor(abs(ratio) + 1e-5), ratio))
        assert int(ev[0, 1, j]) == want, ratio
        assert int(ev[0, 2:, j].abs().sum()) == 0, ratio


def test_3_the_leak_forgets_a_held_level():
    theta, tau = 0.25, 2.0
    hold = int(20 * tau / DELTA)
    tail = int(5 * tau / DELTA)
    x = torch.cat(
        [
            torch.full((hold,), 0.6 * theta, dtype=torch.float32),
            torch.full((tail,), 1.2 * theta, dtype=torch.float32),
        ]
    ).view(1, -1, 1)
    _, ev_off = LevelCrossingNeuron(dt=DELTA, theta=theta)(x)
    _, ev_on = LevelCrossingNeuron(dt=DELTA, theta=theta, reference_leak_tau_ms=tau)(x)
    assert int(ev_off.abs().sum()) == 1
    assert int(ev_off[0, hold + 1, 0]) == 1
    assert int(ev_on.abs().sum()) == 0


# --------------------------------------------------------------------------- #
# 4. The dead time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dead_steps", [1, 2, 3, 5])
def test_4_level_crossing_pays_a_fast_ramp_out_late_without_a_leak(dead_steps):
    theta, ramp = 0.5, 40
    slope = 3 * theta / (dead_steps * DELTA)
    rise = ramp * slope * DELTA
    total = math.floor(rise / theta + 1e-5)
    steps = ramp + dead_steps * total + 20
    n = torch.arange(1, steps + 1, dtype=torch.float64).clamp(max=ramp)
    x = (n * slope * DELTA).view(1, -1, 1)
    model = LevelCrossingNeuron(dt=DELTA, theta=theta, refractory_ms=dead_steps * DELTA)
    ref, ev = model(x)
    times = torch.nonzero(ev[0, :, 0]).flatten()
    assert torch.all(ev[0, times, 0] == 1)
    assert torch.all(torch.diff(times) == dead_steps)
    assert int(times[-1]) > ramp, "the late pay-out continues into the hold"
    assert int(ev.sum()) == total
    assert abs(float(x[0, -1, 0] - ref[0, -1, 0])) < theta


@pytest.mark.parametrize("dead_steps", [1, 2, 4])
def test_4_level_crossing_with_a_leak_loses_part_of_a_pending_step(dead_steps):
    theta, tau, dx = 0.5, 1.0, 4.3 * 0.5
    lam = DELTA / tau
    expected, c = [], dx
    while c >= theta * CROSSING:
        assert abs(c / theta - CROSSING) > 1e-6, "choose another step"
        expected.append(1 + len(expected) * dead_steps)
        c = (1 - lam) ** dead_steps * (c - theta)
    assert abs(c / theta - CROSSING) > 1e-6
    assert len(expected) >= 2
    steps = int(20 * tau / DELTA)
    x = torch.full((1, steps, 1), dx, dtype=torch.float64)
    model = LevelCrossingNeuron(
        dt=DELTA,
        theta=theta,
        refractory_ms=dead_steps * DELTA,
        reference_leak_tau_ms=tau,
    )
    _, ev = model(x)
    times = torch.nonzero(ev[0, :, 0]).flatten().tolist()
    assert times == expected
    assert torch.all(ev[0, expected, 0] == 1)


@pytest.mark.parametrize("leak_tau_ms", [None, 200.0])
def test_4_sigma_delta_dead_time_loses_charge_and_never_bursts(leak_tau_ms):
    theta, dead_steps, on = 2.0, 3, 61
    drive = 3 * theta / DELTA
    x = torch.zeros((1, on + 30, 1), dtype=torch.float64)
    x[0, :on, 0] = drive
    model = SigmaDeltaNeuron(
        dt=DELTA,
        theta=theta,
        leak_tau_ms=leak_tau_ms,
        refractory_ms=dead_steps * DELTA,
    )
    u, sp = model(x)
    times = torch.nonzero(sp[0, :, 0]).flatten()
    during = times[times <= on]
    assert during.tolist() == list(range(1, on + 1, dead_steps))
    assert torch.all(sp[0, during, 0] == 1)
    assert float(u[0, 1:, 0].max()) <= theta
    assert int(sp[0, on + 1 :, 0].sum()) <= 1


# --------------------------------------------------------------------------- #
# 5. The floor is recorded
# --------------------------------------------------------------------------- #


def test_5_every_bundle_records_the_floor_each_converter_received(tmp_path):
    design = _copy_golden_design(tmp_path)

    def add_adex_sa(manifest):
        sa = next(p for p in manifest["populations"] if p["name"] == "sa")
        adex = dict(sa, name="sa_adex", neuron_model="adex", model_params={})
        adex["filter_method"], adex["filter_params"] = "sa", {}
        manifest["populations"].append(adex)

    _edit_design(design, add_adex_sa)
    bundle = _run_design_cli(design, tmp_path / "bundle", "ramp_gaussian", 20)
    cfg, entries = _bundle_entries(bundle)
    assert cfg["schema_version"] == SCHEMA_VERSION == "2.3.0"
    assert entries["sa"]["encoder"]["input_floor_ma"] == 0.0
    assert entries["ra"]["encoder"]["input_floor_ma"] is None
    assert entries["sa_adex"]["encoder"]["input_floor_ma"] == 0.0

    dt = cfg["config"]["simulation"]["integrate_dt_ms"]
    manifest = json.loads((design / "design.json").read_text())
    params = {p["name"]: p["model_params"] for p in manifest["populations"]}
    want_ra = LevelCrossingNeuron(dt=dt, noise_std=0.0, **params["ra"]).to_dict()
    want_sa = SigmaDeltaNeuron(dt=dt, noise_std=0.0, **params["sa"]).to_dict()
    assert entries["ra"]["encoder"]["params"] == want_ra
    assert entries["ra"]["encoder"]["params"]["reference_leak_tau_ms"] is None
    assert entries["sa"]["encoder"]["params"] == want_sa


def _floor_config(floors) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=6, cols=6, spacing=0.2)
        ],
        populations=[
            PopulationConfig(
                name=f"SA {i}",
                neuron_type="SA",
                neuron_model="sigma_delta",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                input_gain=5.0,
                model_params={"theta": 5.0},
                input_floor=floor,
                seed=4,
            )
            for i, floor in enumerate(floors)
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=0.1),
    )


def test_5_an_explicit_floor_is_recorded_as_applied(tmp_path):
    floors = [0.5, float("-inf"), None]
    engine = SimulationEngine(_floor_config(floors))
    engine.run(torch.rand(1, 12, 6, 6), bundle_dir=tmp_path / "bundle")
    _, entries = _bundle_entries(tmp_path / "bundle")
    assert entries["SA 0"]["encoder"]["input_floor_ma"] == 0.5
    assert entries["SA 1"]["encoder"]["input_floor_ma"] is None
    assert entries["SA 2"]["encoder"]["input_floor_ma"] == 0.0
    for pop in engine.populations:
        cfg = pop["config"]
        floor = resolve_input_floor(cfg.neuron_type, cfg.neuron_model, cfg.input_floor)
        assert pop["input_floor"] == floor
        assert entries[pop["name"]]["encoder"]["input_floor_ma"] == floor


# --------------------------------------------------------------------------- #
# 6. The design key round-trips
# --------------------------------------------------------------------------- #


def _set_ra_params(design: Path, **params) -> None:
    def edit(manifest):
        for prec in manifest["populations"]:
            if prec["neuron_model"] == "level_crossing":
                prec["model_params"].update(params)

    _edit_design(design, edit)


def test_6_the_design_key_reaches_the_unit_and_the_bundle(tmp_path):
    design = _copy_golden_design(tmp_path)
    _set_ra_params(design, reference_leak_tau_ms=4.0)
    config = load_design(design)
    ra = next(p for p in config.populations if p.neuron_model == "level_crossing")
    assert ra.model_params["reference_leak_tau_ms"] == 4.0
    engine = SimulationEngine(config)
    unit = next(p["neuron"] for p in engine.populations if p["name"] == ra.name)
    assert unit.reference_leak_tau_ms == 4.0
    bundle = _run_design_cli(design, tmp_path / "bundle", "ramp_gaussian", 20)
    _, entries = _bundle_entries(bundle)
    assert entries[ra.name]["encoder"]["params"]["reference_leak_tau_ms"] == 4.0


@pytest.mark.parametrize("bad", [0, -1.0, 0.01, float("inf"), float("nan"), "10"])
def test_6_a_bad_leak_is_refused_at_load_naming_the_population(tmp_path, bad):
    design = _copy_golden_design(tmp_path)
    _set_ra_params(design, reference_leak_tau_ms=bad)
    manifest = json.loads((design / "design.json").read_text())
    index = next(
        i
        for i, p in enumerate(manifest["populations"])
        if p["neuron_model"] == "level_crossing"
    )
    with pytest.raises(ValueError, match=LEAK_REFUSED) as excinfo:
        load_design(design)
    assert f"populations[{index}]" in str(excinfo.value)


def test_6_keys_beside_model_params_are_ignored(tmp_path):
    design = _copy_golden_design(tmp_path)
    before = load_design(design).to_dict()

    def add_keys(manifest):
        for prec in manifest["populations"]:
            prec["sub_step_ms"] = 0.05
            prec["input_floor_ma"] = (
                None if prec["neuron_model"] == "level_crossing" else 0.0
            )
            prec["encoder"] = {"kind": prec["neuron_model"], "note": "PS's own"}

    _edit_design(design, add_keys)
    assert load_design(design).to_dict() == before


# --------------------------------------------------------------------------- #
# 7. Determinism
# --------------------------------------------------------------------------- #

_DIGEST_SCRIPT = """
import hashlib, sys, torch
from sensoryforge.neurons.event_encoders import LevelCrossingNeuron
gen = torch.Generator().manual_seed(11)
x = torch.cumsum(torch.randn(2, 600, 5, generator=gen, dtype=torch.float64), 1)
model = LevelCrossingNeuron(dt=0.05, theta=0.4, noise_std=0.1,
                            refractory_ms=0.1, reference_leak_tau_ms=1.5)
torch.manual_seed(3)
ref, ev = model(x)
print(hashlib.sha256(ev.numpy().tobytes() + ref.numpy().tobytes()).hexdigest())
"""


def _digest_here() -> str:
    gen = torch.Generator().manual_seed(11)
    x = torch.cumsum(torch.randn(2, 600, 5, generator=gen, dtype=torch.float64), 1)
    model = LevelCrossingNeuron(
        dt=0.05,
        theta=0.4,
        noise_std=0.1,
        refractory_ms=0.1,
        reference_leak_tau_ms=1.5,
    )
    torch.manual_seed(3)
    ref, ev = model(x)
    return hashlib.sha256(ev.numpy().tobytes() + ref.numpy().tobytes()).hexdigest()


def _digest_in_a_new_process() -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_7_same_input_and_seed_give_the_same_events_in_two_calls_and_processes():
    first, second = _digest_here(), _digest_here()
    assert first == second
    assert _digest_in_a_new_process() == first


@pytest.mark.parametrize("tau", [None, 1.5])
@pytest.mark.parametrize("refractory_ms", [0.0, 0.1])
def test_7_one_neuron_alone_equals_it_in_a_population_and_a_batch(tau, refractory_ms):
    gen = torch.Generator().manual_seed(5)
    x = torch.cumsum(torch.randn(3, 500, 4, generator=gen, dtype=torch.float32), 1)
    model = LevelCrossingNeuron(
        dt=DELTA,
        theta=0.4,
        refractory_ms=refractory_ms,
        reference_leak_tau_ms=tau,
    )
    ref, ev = model(x)
    assert int(ev.abs().sum()) > 0
    for b in range(3):
        for i in range(4):
            ref_1, ev_1 = model(x[b : b + 1, :, i : i + 1])
            assert torch.equal(ev_1[0, :, 0], ev[b, :, i])
            assert torch.equal(ref_1[0, :, 0], ref[b, :, i])


def _noisy_config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=6, cols=6, spacing=0.2)
        ],
        populations=[
            PopulationConfig(
                name="RA events",
                neuron_type="RA",
                neuron_model="level_crossing",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                input_gain=5.0,
                model_params={"theta": 0.2, "reference_leak_tau_ms": 3.0},
                membrane_noise_std=0.05,
                noise_seed=123,
                seed=3,
            ),
            PopulationConfig(
                name="SA sigma-delta",
                neuron_type="SA",
                neuron_model="sigma_delta",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                input_gain=5.0,
                model_params={"theta": 5.0},
                membrane_noise_std=0.05,
                noise_seed=321,
                seed=4,
            ),
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=0.1),
    )


def test_7_comparator_noise_with_a_noise_seed_is_reproducible():
    stimulus = torch.rand(1, 40, 6, 6, generator=torch.Generator().manual_seed(2))
    runs = [SimulationEngine(_noisy_config()).run(stimulus) for _ in range(2)]
    assert torch.equal(runs[0]["RA events"]["events"], runs[1]["RA events"]["events"])
    assert torch.equal(
        runs[0]["SA sigma-delta"]["spikes"], runs[1]["SA sigma-delta"]["spikes"]
    )
    assert float(runs[0]["RA events"]["events"].abs().sum()) > 0


# --------------------------------------------------------------------------- #
# 8. Validation and round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad", [0, 0.0, -1.0, 0.01, float("inf"), float("nan"), "10", True]
)
def test_8_bad_leak_values_raise_at_construction(bad):
    with pytest.raises(ValueError, match=LEAK_REFUSED):
        LevelCrossingNeuron(dt=DELTA, reference_leak_tau_ms=bad)


def test_8_the_leak_is_accepted_from_dt_up_and_round_trips():
    assert LevelCrossingNeuron(dt=DELTA).reference_leak_tau_ms is None
    assert LevelCrossingNeuron(reference_leak_tau_ms=None).reference_leak_tau_ms is None
    assert LevelCrossingNeuron(dt=DELTA, reference_leak_tau_ms=DELTA).to_dict()[
        "reference_leak_tau_ms"
    ] == pytest.approx(DELTA)
    with pytest.raises(ValueError, match=LEAK_REFUSED):
        LevelCrossingNeuron(dt=0.1, reference_leak_tau_ms=0.0999)
    model = LevelCrossingNeuron(dt=DELTA, theta=0.3, reference_leak_tau_ms=7)
    config = model.to_dict()
    assert config["reference_leak_tau_ms"] == 7.0
    assert isinstance(config["reference_leak_tau_ms"], float)
    assert LevelCrossingNeuron.from_config(config).to_dict() == config
    assert LevelCrossingNeuron().to_dict()["reference_leak_tau_ms"] is None
    check_component("neuron", LevelCrossingNeuron)
    check_component("neuron", LevelCrossingNeuron, model)


def test_8_the_leak_has_a_param_spec():
    specs = {s.name: s for s in LevelCrossingNeuron.get_param_spec()}
    spec = specs["reference_leak_tau_ms"]
    assert spec.default is None and spec.unit == "ms" and spec.advanced
