"""``sensoryforge run --design`` end to end, and the provenance it stamps
into the bundle (Phase 2a, T2).

Drives the real CLI entry point (``cli.create_parser`` + ``cli.cmd_run``, in
process -- no subprocess needed) against ``tests/fixtures/design_8x8/``,
then reads the written bundle back with :func:`sensoryforge.io.bundle.load_bundle`
and checks:

* ``config_json["design"]`` equals :func:`sensoryforge.io.design.read_manifest`
  of the fixture, verbatim (not re-derived from the config).
* each imported population's bank provenance carries ``design_id`` and the
  source-repo fields.
* each bank's ``weights`` is bit-identical to the corresponding ``.npz``'s
  ``H``, **after** the documented y-slow -> x-slow column re-index
  :class:`~sensoryforge.core.rf_builders.imported.ImportedRFBuilder` performs
  (pressure-simulation flattens ``(height, width)`` row-major/y-slow --
  column ``j = iy * n_x + ix``; SensoryForge's receptor grid is x-slow --
  ``k = ix * n_y + iy``). This fixture's grid is square (8x8) so the
  re-index is ``H.reshape(N, n_y, n_x).transpose(1, 2).reshape(N, -1)``,
  exactly what ``ImportedRFBuilder._read_npz`` does; comparing raw,
  un-reindexed columns would fail even though the import is correct.

Also checks the backward-compat guarantee: a run through the same CLI with
no ``--design`` writes a bundle with no ``"design"`` key and no
``design_id``/``source_repo*`` in any population's provenance -- i.e.
``write_bundle``'s new ``design_manifest`` plumbing changes nothing when
unused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.cli import cmd_run, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.io.design import read_manifest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "design_8x8"


def _run_cli(argv):
    parser = create_parser()
    args = parser.parse_args(argv)
    return cmd_run(args)


class TestRunDesignEndToEnd:
    def test_cli_runs_and_writes_bundle(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        rc = _run_cli(
            [
                "run",
                "--design",
                str(FIXTURE_DIR),
                "--stimulus",
                "ramp_gaussian",
                "--duration",
                "20",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc == 0
        assert (bundle_dir / "config.json").exists()

    def test_design_manifest_stamped_verbatim(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        rc = _run_cli(
            [
                "run",
                "--design",
                str(FIXTURE_DIR),
                "--stimulus",
                "ramp_gaussian",
                "--duration",
                "20",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc == 0

        bundle = load_bundle(bundle_dir)
        manifest = read_manifest(FIXTURE_DIR)
        assert bundle.meta["config_json"]["design"] == manifest

    def test_imported_bank_provenance_carries_design_fields(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        rc = _run_cli(
            [
                "run",
                "--design",
                str(FIXTURE_DIR),
                "--stimulus",
                "ramp_gaussian",
                "--duration",
                "20",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc == 0

        bundle = load_bundle(bundle_dir)
        manifest = read_manifest(FIXTURE_DIR)
        assert bundle.banks, "expected at least one population bank"
        for name, bank in bundle.banks.items():
            prov = bank.provenance
            assert prov.get("source_format") == "npz"
            assert prov["design_id"] == manifest["design_id"]
            assert prov["source_repo"] == "pressure-simulation"
            assert prov["source_repo_git_sha"] == manifest["git_sha"]

    def test_bank_weights_bit_identical_to_npz_after_reindex(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        rc = _run_cli(
            [
                "run",
                "--design",
                str(FIXTURE_DIR),
                "--stimulus",
                "ramp_gaussian",
                "--duration",
                "20",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc == 0

        bundle = load_bundle(bundle_dir)
        manifest = read_manifest(FIXTURE_DIR)
        rows, cols = manifest["decisions"]["grid"]

        for prec in manifest["populations"]:
            name = prec["name"]
            bank = bundle.banks[name]
            with np.load(FIXTURE_DIR / prec["rf_file"]) as data:
                H = np.ascontiguousarray(data["H"], dtype=np.float32)
            n = H.shape[0]
            # pressure-simulation's y-slow -> SensoryForge's x-slow, the
            # same reshape/transpose ImportedRFBuilder._read_npz performs.
            h_reindexed = H.reshape(n, rows, cols).transpose(0, 2, 1).reshape(n, -1)
            assert torch.equal(bank.weights, torch.from_numpy(h_reindexed))

    def test_unknown_stimulus_name_errors_clearly(self, tmp_path, capsys):
        bundle_dir = tmp_path / "bundle_unused"
        rc = _run_cli(
            [
                "run",
                "--design",
                str(FIXTURE_DIR),
                "--stimulus",
                "not_a_real_stimulus",
                "--duration",
                "20",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc != 0
        captured = capsys.readouterr()
        assert "not_a_real_stimulus" in captured.err

    def test_design_and_preset_are_mutually_exclusive(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--design",
                    str(FIXTURE_DIR),
                    "--preset",
                    "tactile_sa1_ra1",
                ]
            )


class TestNoDesignBackwardCompat:
    """A run through the same CLI without --design must be unaffected."""

    def test_no_design_key_and_no_design_provenance(self, tmp_path):
        bundle_dir = tmp_path / "bundle_plain"
        rc = _run_cli(
            [
                "run",
                "--preset",
                "tactile_sa1_ra1",
                "--duration",
                "5",
                "--bundle",
                str(bundle_dir),
            ]
        )
        assert rc == 0

        bundle = load_bundle(bundle_dir)
        assert "design" not in bundle.meta["config_json"]
        for bank in bundle.banks.values():
            assert "design_id" not in bank.provenance
            assert "source_repo" not in bank.provenance
            assert "source_repo_git_sha" not in bank.provenance
