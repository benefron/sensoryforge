"""A check on the world's sampler (outside the audit's import wall: it reads the world).

It needs no declaration of what a draw should show: only that the sampler keeps its
promise that draw ``i`` depends on the world, the seed and ``i``, never on how many
draws were asked for.
"""

from __future__ import annotations

import json
from typing import Any, List

from sensoryforge.evaluate.checks.base import (
    Check,
    CheckContext,
    CheckResult,
    CheckSkipped,
    Failure,
)

#: Draws compared (the long run is twice this), and the indices regenerated alone.
N_DRAWS = 32
SINGLES = (0, 5, 17, 31, 47, 63)


def _key(draw: Any) -> Any:
    return (
        draw.class_name,
        json.dumps(draw.values, sort_keys=True, default=repr),
        draw.draw_seed,
    )


class PrefixStable(Check):
    """Draw ``i`` does not depend on how many draws are sampled."""

    id = "core.world.prefix_stable"
    pack = "core"
    scope = "dataset"
    title = "Draw i does not depend on n"
    what = (
        f"The data set's own world and seed sample {N_DRAWS} and {2 * N_DRAWS} "
        "draws; the "
        f"first {N_DRAWS} must be identical, and draws regenerated one at a time by "
        "index must equal the same draws of the long run."
    )

    def run(self, ctx: CheckContext) -> CheckResult:
        from sensoryforge.world import World
        from sensoryforge.world.sampling import sample

        path = ctx.folder / "dataset.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            raw = dict(meta["world"])
            raw.pop("format", None)
            world = World.from_dict(raw)
            seed = int(meta.get("seed", 0))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckSkipped(f"dataset.json holds no usable world ({exc})") from exc
        short = sample(world, N_DRAWS, seed)
        long = sample(world, 2 * N_DRAWS, seed)
        failures: List[Failure] = []
        for i in range(N_DRAWS):
            if _key(short[i]) != _key(long[i]):
                failures.append(
                    Failure(
                        f"draw {i}", f"differs between n={N_DRAWS} and n={2 * N_DRAWS}"
                    )
                )
        for i in SINGLES:
            alone = sample(world, indices=[i], seed=seed)[0]
            if _key(alone) != _key(long[i]):
                failures.append(Failure(f"draw {i}", "differs when regenerated alone"))
        return CheckResult(
            "fail" if failures else "pass",
            tested=N_DRAWS + len(SINGLES),
            failed=len(failures),
            failures=failures,
            numbers={"seed": seed},
        )
