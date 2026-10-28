"""Run the agent-skill scenarios with fresh agents and write their scores (roadmap B9).

Each scenario costs a real agent run, so nothing here runs in the default test suite.
Pick what to run, build the install once, run, and read ``scores.md``::

    python tests/agents/run.py --list
    python tests/agents/run.py --skill sensoryforge-design-world --out agent-runs/world
    python tests/agents/run.py --scenario world.dots-and-edges --model sonnet
    python tests/agents/run.py --preflight

``--preflight`` asks a cheap agent to run ``sensoryforge`` and ``python`` in an empty
workspace and checks that both resolve into the throwaway environment (a shell profile
that puts another Python first would make every scenario test the wrong install). Every
run does it first unless ``--no-preflight`` is given. See
``docs/developer_guide/agent_scenarios.md``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.agents import harness as h  # noqa: E402

PREFLIGHT = h.Scenario(
    id="harness.preflight",
    skill="sensoryforge-setup",
    kind="normal",
    prompt=(
        "Run exactly these two shell commands and show their output verbatim, nothing "
        "else: `which sensoryforge` and "
        '`python -c "import sensoryforge, sys; print(sys.prefix); '
        'print(sensoryforge.__file__)"`'
    ),
    skip_defaults=["ran", "no_repo_paths"],
)


def preflight(install: h.Install, model: str) -> str:
    """Return an empty string when the agent's shell resolves into ``install``."""
    runner = h.ClaudeRunner(model=model, budget_usd=0.3, timeout_s=300)
    workspace = h.prepare_workspace(PREFLIGHT, install, install.root / "workspaces")
    transcript = runner.run(PREFLIGHT, workspace, install)
    outputs = "\n".join(c.result for c in transcript.calls if c.command)
    venv = str(install.venv)
    if transcript.errors:
        return "; ".join(transcript.errors)
    if outputs.count(venv) < 2 and str(install.venv.resolve()) not in outputs:
        return f"the agent's shell does not use {venv}:\n{outputs[:800]}"
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skill", action="append", default=[], help="a skill's scenarios"
    )
    parser.add_argument(
        "--scenario", action="append", default=[], help="one scenario id"
    )
    parser.add_argument("--kind", action="append", default=[], choices=h.KINDS)
    parser.add_argument("--model", default="sonnet", help="the agent's model")
    parser.add_argument("--judge-model", default="haiku", help="the judge's model")
    parser.add_argument(
        "--no-judge", action="store_true", help="leave judge lines unscored"
    )
    parser.add_argument(
        "--budget", type=float, default=2.0, help="USD per turn at most"
    )
    parser.add_argument("--jobs", type=int, default=1, help="scenarios run at once")
    parser.add_argument("--root", type=Path, help="where the install and workspaces go")
    parser.add_argument("--out", type=Path, help="where results.json and scores.md go")
    parser.add_argument(
        "--list", action="store_true", help="list the scenarios and stop"
    )
    parser.add_argument(
        "--preflight", action="store_true", help="only check the install"
    )
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args(argv)

    chosen = [
        s
        for s in h.load_scenarios()
        if (not args.skill or s.skill in args.skill)
        and (not args.scenario or s.id in args.scenario)
        and (not args.kind or s.kind in args.kind)
    ]
    if args.list:
        for s in chosen:
            print(f"{s.id:40s} {s.kind:14s} {s.skill}")
        return 0
    if not chosen and not args.preflight:
        print("no scenario matches", file=sys.stderr)
        return 2

    root = args.root or Path(tempfile.mkdtemp(prefix="sf-agent-runs-"))
    out = args.out or root / "report"
    print(f"building the install in {root} ...", flush=True)
    install = h.build_install(root)
    print(f"sensoryforge resolves to {install.package}", flush=True)
    if args.preflight or not args.no_preflight:
        problem = preflight(install, "haiku")
        if problem:
            print(f"preflight failed: {problem}", file=sys.stderr)
            return 1
        print("preflight: the agent's shell uses the install", flush=True)
        if args.preflight:
            return 0

    runner = h.ClaudeRunner(model=args.model, budget_usd=args.budget)
    judge = None if args.no_judge else h.claude_judge(args.judge_model)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(h.run_scenario, s, install, runner, judge, out): s
            for s in chosen
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result.scenario.id}: {'pass' if result.passed else 'FAIL'} "
                f"{result.score:.2f} (${result.transcript.cost_usd:.3f}, "
                f"{result.seconds:.0f} s)",
                flush=True,
            )
    order = {s.id: i for i, s in enumerate(chosen)}
    results.sort(key=lambda r: order[r.scenario.id])
    path = h.write_report(results, out)
    print(f"scores: {path}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
