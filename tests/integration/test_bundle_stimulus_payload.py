"""The bundle's stimulus payload is tagged and regenerates (Wave J, J7).

``stimuli/stimulus.json`` used to be whatever dict the caller passed, written
straight through with no schema tag. pressure-simulation's
``generate_stimulus_from_json`` reads every field with a ``.get`` default, so
handed an empty or foreign payload it does not raise: it silently yields a
static Gaussian blob at the origin, and its viewer encodes that and draws
plausible plots of the wrong stimulus. These tests pin the fix.

``_generate_stimulus_from_json`` below re-implements
``encoding/encode_runner.py::generate_stimulus_from_json`` from
`~/Documents/pressure simulation` so the round trip runs in CI without that
repository present. It is a transcription, not a reinterpretation; if the two
drift, this test is what should be updated, in the same commit that records
why.
"""

import json
import math

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import build_stimulus_payload, write_bundle

h5py = pytest.importorskip("h5py")

ROWS = COLS = 12
SPACING = 0.15
DT_MS = 1.0


# --------------------------------------------------------------------------- #
# Transcribed from pressure-simulation
# --------------------------------------------------------------------------- #
def _grid_coords(rows, cols, spacing, center=(0.0, 0.0)):
    """``encoding/grid_torch.py::create_grid_torch`` -- meshgrid(indexing="ij")."""
    total_x = (rows - 1) * spacing
    total_y = (cols - 1) * spacing
    x = torch.linspace(center[0] - total_x / 2, center[0] + total_x / 2, rows)
    y = torch.linspace(center[1] - total_y / 2, center[1] + total_y / 2, cols)
    return torch.meshgrid(x, y, indexing="ij")


def _generate_stimulus_from_json(payload, xx, yy):
    """``encoding/encode_runner.py::generate_stimulus_from_json``, transcribed."""
    dt = max(float(payload.get("dt_ms", 1.0)), 0.1)
    total_ms = max(float(payload.get("total_ms", 300.0)), dt)
    time_axis = torch.arange(0.0, total_ms + 0.5 * dt, dt)

    ramp_up = max(float(payload.get("ramp_up_ms", 50.0)), 0.0)
    plateau = max(float(payload.get("plateau_ms", 200.0)), 0.0)
    ramp_down = max(float(payload.get("ramp_down_ms", 50.0)), 0.0)
    peak = float(payload.get("amplitude", 1.0))
    down_start = ramp_up + plateau
    total_dur = ramp_up + plateau + ramp_down

    amp = torch.zeros_like(time_axis)
    if ramp_up > 0:
        up_mask = time_axis < ramp_up
        amp[up_mask] = time_axis[up_mask] / ramp_up
    else:
        amp[time_axis < ramp_up + 1e-6] = 1.0
    amp[(time_axis >= ramp_up) & (time_axis < down_start)] = 1.0
    if ramp_down > 0:
        dm = (time_axis >= down_start) & (time_axis <= down_start + ramp_down)
        amp[dm] = torch.clamp(1.0 - (time_axis[dm] - down_start) / ramp_down, 0, 1)
    amp = torch.where(time_axis > total_dur, torch.zeros_like(amp), amp)
    amp = amp.clamp(0, 1) * peak

    stim_type = str(payload.get("type", "gaussian"))
    start_raw = payload.get("start", [0.0, 0.0])
    end_raw = payload.get("end", start_raw)
    spread = float(payload.get("spread", 0.3))
    motion = str(payload.get("motion", "static"))
    orientation_deg = float(payload.get("orientation_deg", 0.0))

    sx, sy = float(start_raw[0]), float(start_raw[1])
    ex, ey = float(end_raw[0]), float(end_raw[1])
    dist = math.hypot(ex - sx, ey - sy)

    frames = torch.zeros(time_axis.numel(), *xx.shape)
    for idx, t_val in enumerate(time_axis):
        t = float(t_val)
        if motion == "moving" and plateau > 0 and dist > 1e-6:
            if t <= ramp_up:
                alpha = 0.0
            elif t >= ramp_up + plateau:
                alpha = 1.0
            else:
                alpha = (t - ramp_up) / plateau
            cx = sx + alpha * (ex - sx)
            cy = sy + alpha * (ey - sy)
        else:
            cx, cy = sx, sy

        if stim_type == "gaussian":
            frame = torch.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * max(spread, 1e-6) ** 2)
            )
        elif stim_type == "point":
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            frame = (r <= (max(spread, 1e-6) / 2)).float()
        else:  # edge
            theta = torch.tensor(math.radians(orientation_deg), dtype=xx.dtype)
            projection = (xx - cx) * torch.sin(theta) + (yy - cy) * torch.cos(theta)
            frame = torch.exp(-(projection**2) / (2 * max(spread, 1e-6) ** 2))
        frames[idx] = frame * amp[idx]
    return frames, time_axis


# --------------------------------------------------------------------------- #
def _config():
    return SensoryForgeConfig(
        grids=[
            GridConfig(
                name="Main", arrangement="grid", rows=ROWS, cols=COLS, spacing=SPACING
            )
        ],
        populations=[
            PopulationConfig(
                name="SA",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="gaussian",
                neurons_per_row=2,
                seed=3,
            )
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=DT_MS),
    )


def _write(tmp_path, stimulus_config, frames):
    config = _config()
    engine = SimulationEngine(config)
    results = engine.run(frames.unsqueeze(0), return_intermediates=True)
    bundle_dir = write_bundle(
        tmp_path / "b",
        config,
        engine,
        results,
        frames.unsqueeze(0),
        stimulus_config=stimulus_config,
    )
    with open(bundle_dir / "stimuli" / "stimulus.json") as f:
        payload = json.load(f)
    return bundle_dir, payload


def _gaussian_frames(amplitude, sigma, n_frames):
    xx, yy = _grid_coords(ROWS, COLS, SPACING)
    blob = amplitude * torch.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
    return blob.unsqueeze(0).repeat(n_frames, 1, 1)


class TestPressureSimRoundTrip:
    """A mappable stimulus regenerates its own frames, at zero tolerance."""

    def test_gaussian_payload_regenerates_the_stored_frames(self, tmp_path):
        n_frames = 20
        frames = _gaussian_frames(2.0, 1.0, n_frames)
        bundle_dir, payload = _write(
            tmp_path,
            {"type": "gaussian", "amplitude": 2.0, "spread": 1.0},
            frames,
        )

        assert payload["kind"] == "stimulus"
        assert payload["schema_version"] == "1.0.0"

        xx, yy = _grid_coords(ROWS, COLS, SPACING)
        regenerated, _ = _generate_stimulus_from_json(payload, xx, yy)

        with h5py.File(bundle_dir / "data.h5", "r") as f:
            stored = torch.from_numpy(f["stimulus"]["frames"][()])

        assert regenerated.shape == stored.shape, (
            f"pressure-simulation would build {list(regenerated.shape)} frames "
            f"from this payload, but the bundle stores {list(stored.shape)}"
        )
        assert torch.equal(regenerated, stored), (
            "regenerating the payload gives different frames than the bundle "
            f"stores; max abs diff {float((regenerated - stored).abs().max())}"
        )

    def test_final_frame_is_not_dropped(self, tmp_path):
        """The plateau mask is strict, so the last sample needs care.

        pressure-simulation's plateau mask is ``t < ramp_up + plateau``. A
        plateau of exactly ``total_ms`` therefore leaves the final sample at
        zero while ours is not, which is a one-frame difference easy to miss
        in an aggregate comparison.
        """
        frames = _gaussian_frames(1.0, 0.8, 12)
        _, payload = _write(tmp_path, {"type": "gaussian", "spread": 0.8}, frames)
        xx, yy = _grid_coords(ROWS, COLS, SPACING)
        regenerated, _ = _generate_stimulus_from_json(payload, xx, yy)
        assert float(regenerated[-1].max()) > 0.0, (
            "the last regenerated frame is all zeros: plateau_ms must exceed "
            "total_ms because the plateau mask is a strict inequality"
        )


class TestTagging:
    """Every payload says which schema it is, so nothing is read by accident."""

    def test_unmappable_type_is_tagged_and_flagged(self, tmp_path):
        frames = _gaussian_frames(1.0, 1.0, 8)
        _, payload = _write(tmp_path, {"type": "texture", "pattern": "gabor"}, frames)
        assert payload["kind"] == "sensoryforge_stimulus"
        assert payload["reconstructible_by_pressure_simulation"] is False
        assert payload["sensoryforge"]["pattern"] == "gabor"

    def test_missing_stimulus_config_is_never_written_as_empty(self, tmp_path):
        frames = _gaussian_frames(1.0, 1.0, 8)
        _, payload = _write(tmp_path, None, frames)
        assert payload != {}
        assert payload["kind"] == "sensoryforge_stimulus"
        assert payload["reconstructible_by_pressure_simulation"] is False
        assert payload["type"] == "unspecified"
        assert payload["n_frames"] == 8

    def test_original_config_is_preserved_losslessly(self, tmp_path):
        frames = _gaussian_frames(3.0, 0.5, 6)
        original = {
            "type": "gaussian",
            "amplitude": 3.0,
            "spread": 0.5,
            "extra": [1, 2],
        }
        _, payload = _write(tmp_path, original, frames)
        assert payload["sensoryforge"] == original


class TestPayloadBuilder:
    """Unit-level checks on the builder itself."""

    @pytest.mark.parametrize("stim_type", ["gaussian", "point", "edge"])
    def test_mappable_types_use_the_pressure_sim_schema(self, stim_type):
        payload = build_stimulus_payload(
            {"type": stim_type},
            dt_ms=1.0,
            n_frames=10,
            grid_section={
                "rows": 4,
                "cols": 5,
                "spacing_mm": 0.2,
                "center_mm": [1.0, -1.0],
            },
        )
        assert payload["kind"] == "stimulus"
        assert payload["grid"] == {
            "rows": 4,
            "cols": 5,
            "spacing": 0.2,
            "center_x": 1.0,
            "center_y": -1.0,
        }

    def test_total_ms_is_the_last_sample_time_not_the_duration(self):
        payload = build_stimulus_payload(
            {"type": "gaussian"}, dt_ms=0.5, n_frames=11, grid_section={}
        )
        assert payload["total_ms"] == pytest.approx(5.0)
        assert payload["dt_ms"] == pytest.approx(0.5)
