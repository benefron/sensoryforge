"""Record the spikes the presets (and the deprecated ``noise_std``) produce, for C-130.

C-130 separated the sensor (receptor-current) noise from the neuron's membrane
noise (``PopulationConfig.sensor_noise_std`` / ``membrane_noise_std``) and kept
``noise_std`` as a deprecated alias that sets both. This script records, **with
the code from before that change**, the spikes every shipped preset produces and
the spikes and filtered drive of two configs that use ``noise_std``;
``tests/integration/test_noise_split_golden.py`` then requires the new code to
reproduce them.

Run it against the pre-change checkout, never the worktree under test::

    PYTHONPATH=/path/to/pre-change/checkout conda run -n sensoryforge \\
        python scripts/dev/export_noise_split_golden.py

It writes ``tests/fixtures/noise_split_golden.pt`` (in the checkout this script
lives in), with the platform it was recorded on.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

#: Duration of every recorded run (ms). Short, to keep the test fast.
DURATION_MS = 120.0


def cases() -> List[Tuple[str, Dict[str, Any]]]:
    """``(case name, canonical config dict)`` for every recorded run.

    Returns:
        One case per shipped preset, unchanged, plus two
        ``tactile_sa1_ra1_adex`` variants using the deprecated ``noise_std``:
        one with a per-population ``noise_seed``, one drawing from the run seed.
    """
    import copy

    from sensoryforge.presets import list_presets, load_preset

    out: List[Tuple[str, Dict[str, Any]]] = []
    for name in list_presets():
        out.append((f"preset:{name}", load_preset(name)))

    base = load_preset("tactile_sa1_ra1_adex")
    seeded = copy.deepcopy(base)
    for i, pop in enumerate(seeded["populations"]):
        pop["noise_std"] = 2.0 + i
        pop["noise_seed"] = 7 + i
    out.append(("alias:noise_std_seeded", seeded))

    unseeded = copy.deepcopy(base)
    for pop in unseeded["populations"]:
        pop["noise_std"] = 1.5
    unseeded.setdefault("simulation", {})["seed"] = 3
    out.append(("alias:noise_std_run_seed", unseeded))
    return out


def run_case(config_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Run one case through ``SimulationEngine`` and keep its spikes and drive.

    Args:
        config_dict: Canonical config dict.

    Returns:
        ``{"<population>/spikes": sparse int16 [1, T, N] spike counts,
        "<population>/filtered": filtered_summary(...)}`` of the filtered
        current in mA after gain and noise.
    """
    import warnings

    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.core.simulation_engine import SimulationEngine
    from sensoryforge.stimuli.render import render_for_config

    with warnings.catch_warnings():
        # The deprecated noise_std alias warns on the new code; irrelevant here.
        warnings.simplefilter("ignore", FutureWarning)
        config = SensoryForgeConfig.from_dict(config_dict)
    config.simulation.device = "cpu"
    stimulus, _, _, _ = render_for_config(
        config, duration_ms=DURATION_MS, dt_ms=config.simulation.dt_ms
    )
    engine = SimulationEngine(config)
    results = engine.run(stimulus, return_intermediates=True)
    out: Dict[str, Any] = {}
    for pop_name, pop_results in results.items():
        spikes = pop_results["spikes"].detach().cpu()
        filtered = pop_results["filtered"].detach().cpu().contiguous()
        # Spikes are sparse integer counts; stored sparse to keep the fixture
        # small. The filtered current is summarised: its exact bytes' SHA-256
        # (same-platform check) and float64 moments (cross-platform check).
        out[f"{pop_name}/spikes"] = spikes.to(torch.int16).to_sparse()
        out[f"{pop_name}/filtered"] = filtered_summary(filtered)
    return out


def filtered_summary(filtered: torch.Tensor) -> Dict[str, Any]:
    """A compact fingerprint of a filtered-current tensor.

    Args:
        filtered: ``[batch, time, N]`` current in mA.

    Returns:
        ``{"sha256": hex digest of the float32 bytes, "shape": list,
        "sum": float, "sq_sum": float}`` (moments in float64).
    """
    import hashlib

    data = filtered.to(torch.float32).contiguous()
    as_double = data.double()
    return {
        "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        "shape": list(data.shape),
        "sum": float(as_double.sum()),
        "sq_sum": float((as_double**2).sum()),
    }


def platform_signature() -> Dict[str, str]:
    """The properties that decide whether exact reproduction is expected."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
    }


def main() -> int:
    import sensoryforge

    out_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "noise_split_golden.pt"
    payload = {
        "platform": platform_signature(),
        "sensoryforge_file": sensoryforge.__file__,
        "duration_ms": DURATION_MS,
        "cases": {name: run_case(cfg) for name, cfg in cases()},
    }
    torch.save(payload, out_path)
    print(f"wrote {out_path} from {sensoryforge.__file__}")
    for name, tensors in payload["cases"].items():
        counts = {
            k: int(v.to_dense().sum()) for k, v in tensors.items() if k.endswith("/spikes")
        }
        print(f"  {name}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
