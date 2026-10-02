"""SensoryForge's own sha: from git in a checkout, else pip's direct_url.json."""

import json
import subprocess
from pathlib import Path

from sensoryforge import provenance
from sensoryforge.provenance import read_source_info, source_info

ROOT = Path(__file__).resolve().parents[2]


def test_a_git_checkout_reports_its_head():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    info = read_source_info()
    assert info["source"] == "git" and info["sha"] == head
    assert isinstance(info["dirty"], bool)
    assert source_info()["sha"] == head


def test_a_pip_install_from_git_reports_its_commit(tmp_path, monkeypatch):
    class FakeDistribution:
        def read_text(self, name):
            if name == "direct_url.json":
                return json.dumps(
                    {
                        "url": "file:///x",
                        "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                    }
                )
            return None

    monkeypatch.setattr(
        provenance.metadata, "distribution", lambda name: FakeDistribution()
    )
    (tmp_path / "sensoryforge").mkdir()
    info = read_source_info(tmp_path / "sensoryforge")
    assert info == {"sha": "abc123", "dirty": False, "source": "direct_url"}


def test_with_neither_the_sha_is_unknown(tmp_path, monkeypatch):
    def missing(name):
        raise provenance.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "distribution", missing)
    (tmp_path / "sensoryforge").mkdir()
    info = read_source_info(tmp_path / "sensoryforge")
    assert info == {"sha": "unknown", "dirty": None, "source": "unknown"}
