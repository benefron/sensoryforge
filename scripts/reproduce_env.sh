#!/bin/bash
# S3: the reproducibility proof, "from a clean checkout and a fresh environment"
# (Wave S, docs/development/handover/phase4_tasks.md).
#
# Builds a throwaway virtualenv, installs SensoryForge into it with no other
# state carried over from the caller's environment, and runs
# scripts/reproduce_figure.py --check inside it. Exit code is that script's:
# 0 = the fresh install reproduces the committed reference bit-for-bit
# (same platform) / within the stated tolerance (cross-platform); non-zero
# otherwise.
#
# Usage:
#   scripts/reproduce_env.sh
#
# This script is the one meant to run in CI's reproducibility job and to be
# handed to a reader verifying the tool-paper reproducibility claim by hand;
# it is intentionally not exercised by the SensoryForge dev-environment test
# suite itself, because F-053 forbids this project's own agents from running
# pip/conda install commands against the shared dev environment.
# tests/validation/test_reproducibility.py instead calls
# scripts/reproduce_figure.py directly (skipping the venv/install step,
# since the dev environment already has the package importable) to prove
# the underlying determinism claim in CI without violating F-053.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$(mktemp -d)/sensoryforge-repro-venv"

echo "Repository: $REPO_ROOT"
echo "Fresh venv: $VENV_DIR"

python3 -m venv --system-site-packages "$VENV_DIR"
# --system-site-packages reuses the base Python's torch build (large binary
# wheel; not re-downloaded) while still installing SensoryForge itself and
# its own metadata fresh, matching the B1/B2 wheel-check pattern in
# docs/development/handover/phase1_tasks.md's Appendix.

"$VENV_DIR/bin/pip" install --quiet "$REPO_ROOT"
"$VENV_DIR/bin/python" -c "import sensoryforge; print('sensoryforge:', sensoryforge.__file__)"

"$VENV_DIR/bin/python" "$REPO_ROOT/scripts/reproduce_figure.py" --check
status=$?

rm -rf "$(dirname "$VENV_DIR")"
exit $status
