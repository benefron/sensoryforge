"""Time the world renderer on pressure-simulation's RA design load (spec §5.6).

4.1 M triples (t - tau, t, t + tau) at 40x40, 0.15 mm, float64, in chunks of
8000 draws, from a four-class world shaped like pressure-simulation's today
(dots, edges, random braille cells, signed textures, always sliding at
0.02 mm/ms). Times a few chunks after a warm-up and extrapolates.

    conda run -n sensoryforge python benchmarks/world_render.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from sensoryforge.world import Canvas, World, render, sample
from sensoryforge.world import rng

WORLD = {
    "world": {
        "name": "ps_today_like",
        "defaults": {
            "hold_ms": {"value": 0},
            "slide_ms": {"value": 1000},
            "speed_mm_per_ms": {"value": 0.02},
            "direction_deg": {"range": [0, 360], "circular": True},
            "x_mm": {"range": [-2.925, 2.925]},
            "y_mm": {"range": [-2.925, 2.925]},
        },
        "classes": {
            "dots": {
                "layer": {"shape": {"kind": "gaussian"}},
                "axes": {"sigma_mm": {"range": [0.15, 0.45]}},
            },
            "edges": {
                "layer": {"shape": {"kind": "bar", "length_mm": 0}},
                "axes": {
                    "width_mm": {"range": [0.05, 0.15]},
                    "orientation_deg": {"range": [0, 180], "circular": True},
                },
            },
            "braille": {
                "layer": {
                    "shape": {"kind": "gaussian", "sigma_mm": 0.15},
                    "pattern": {"kind": "braille", "dot_spacing_mm": 0.35},
                },
                "axes": {"dots": {"dist": "braille_cells"}},
            },
            "textures": {
                "layer": {"shape": {"kind": "gabor", "signed": True}},
                "axes": {
                    "wavelength_mm": {"range": [0.3, 0.8]},
                    "sigma_mm": {"range": [0.3, 0.6]},
                    "orientation_deg": {"range": [0, 360], "circular": True},
                    "phase_deg": {"range": [0, 360], "circular": True},
                },
            },
        },
    }
}
TRIPLES = 4_096_000
CHUNK = 8_000
TAU_MS = 8.0
TIMED_CHUNKS = 6
TARGET_S = 600.0


def main() -> None:
    world = World.from_dict(WORLD)
    canvas = Canvas.from_grid(40, 40, 0.15)
    seconds = []
    for c in range(TIMED_CHUNKS + 1):
        start = time.perf_counter()
        indices = range(c * CHUNK, (c + 1) * CHUNK)
        draws = sample(world, indices=indices, seed=0)
        t = 100.0 + 800.0 * rng.uniforms(
            rng.draw_seeds(1, np.arange(c * CHUNK, (c + 1) * CHUNK)), "t"
        )
        times = torch.from_numpy(np.stack([t - TAU_MS, t, t + TAU_MS], axis=1))
        frames = render(draws, canvas, times, dtype=torch.float64)
        assert frames.shape == (CHUNK, 3, 40, 40)
        seconds.append(time.perf_counter() - start)
    per_chunk = statistics.median(seconds[1:])  # the first chunk warms up
    total = per_chunk * TRIPLES / CHUNK
    print(f"threads: {torch.get_num_threads()}, torch {torch.__version__}")
    print(f"per chunk of {CHUNK} triples: {per_chunk:.3f} s (median of {TIMED_CHUNKS})")
    print(f"4.1 M triples at 40x40, float64: {total:.0f} s ({total / 60:.1f} min)")
    print("PASS" if total <= TARGET_S else f"OVER the {TARGET_S:.0f} s target")


if __name__ == "__main__":
    main()
