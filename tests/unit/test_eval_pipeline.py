"""The one-command pipeline (roadmap A11a): world file in, report folder out.

A small tactile world goes through every stage on the 8x8 test design. The tests check
the files the stages leave, ``pipeline.json`` against its schema, the explicit pack
rule, what happens when a stage fails or an input is wrong, and that a resumed run
reuses the bundles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sensoryforge.cli import create_parser, cmd_evaluate
from sensoryforge.evaluate import pipeline
from sensoryforge.evaluate.pipeline import (
    PIPELINE_FILE,
    REPORT_DIR,
    STAGES,
    dataset_mapping,
    run_pipeline,
)
from sensoryforge.io.design import load_design
from sensoryforge.world.dataset import load_dataset

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WORLD = FIXTURES / "worlds" / "tactile_small.yml"
DESIGN = FIXTURES / "design_8x8"
STAMP = "2026-10-23T18:00:00Z"
DRAWS = 4
DURATION_MS = 100.0


def make_spec(draws=DRAWS):
    return load_dataset(dataset_mapping(WORLD, draws, 3, DURATION_MS))


def make_config():
    return load_design(DESIGN)


def validate(doc):
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    root = Path(__file__).resolve().parents[2] / "schemas" / "evaluate"
    registry = Registry()
    for path in root.glob("*.json"):
        schema = json.loads(path.read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
        registry = registry.with_resource(path.name, Resource.from_contents(schema))
    schema = json.loads((root / "pipeline.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(doc)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """``(pipeline.json content, data-set folder)`` with every stage, tactile pack."""
    out = tmp_path_factory.mktemp("pipeline_full")
    doc = run_pipeline(
        make_config(),
        make_spec(),
        out,
        sensor_desc="--design design_8x8",
        source=str(WORLD),
        packs=["core", "tactile"],
        check_packs=["core"],
        check_sample=2,
        created_at=STAMP,
    )
    return doc, out


def test_world_to_report_folder(full):
    doc, out = full
    assert doc["status"] == "ok" and doc["error"] is None
    assert [s["name"] for s in doc["stages"]] == list(STAGES)
    assert all(s["status"] == "ok" for s in doc["stages"])
    report = out / REPORT_DIR
    for name in (
        "audit.json",
        "audit_entries.jsonl",
        "declared.json",
        "declared_entries.jsonl",
        "compare.json",
        "compare_entries.jsonl",
        "checks.json",
        PIPELINE_FILE,
    ):
        assert (report / name).is_file(), name
    assert (out / "manifest.jsonl").is_file() and (out / "dataset.json").is_file()
    assert doc["batch"] == {"ok": DRAWS, "failed": 0, "skipped": 0}
    assert doc["counts"]["entries"] == DRAWS and doc["counts"]["audited"] == DRAWS
    assert doc["counts"]["compared_pairs"] > 0
    assert doc["counts"]["checks"]["fail"] == 0
    assert doc["inputs"]["adapter"] == "tactile" and doc["notes"] == []
    assert json.loads((report / PIPELINE_FILE).read_text()) == doc


def test_pipeline_json_is_valid_and_hashes_the_stage_files(full):
    doc, out = full
    validate(doc)
    from sensoryforge.evaluate.stage_io import file_sha256

    assert set(doc["files"]) >= {"audit.json", "declared.json", "compare.json"}
    for name, sha in doc["files"].items():
        assert file_sha256(out / REPORT_DIR / name) == sha


def test_the_audit_runs_in_its_own_process(tmp_path, monkeypatch):
    calls = []
    real = pipeline._run_audit_process

    def recording(folder, out, packs):
        calls.append((folder, out, list(packs)))
        return real(folder, out, packs)

    monkeypatch.setattr(pipeline, "_run_audit_process", recording)
    doc = run_pipeline(
        make_config(),
        make_spec(1),
        tmp_path,
        packs=["core"],
        run_check_stage=False,
        created_at=STAMP,
    )
    assert doc["status"] == "ok"
    assert calls == [(tmp_path, tmp_path / REPORT_DIR, ["core"])]


def test_resume_reuses_the_bundles(full):
    _, out = full
    doc = run_pipeline(
        make_config(),
        make_spec(),
        out,
        packs=["core", "tactile"],
        run_check_stage=False,
        resume=True,
        created_at=STAMP,
    )
    assert doc["status"] == "ok"
    assert doc["batch"] == {"ok": 0, "failed": 0, "skipped": DRAWS}
    assert doc["stages"][-1] == {
        "name": "checks",
        "status": "skipped",
        "detail": None,
        "seconds": 0.0,
    }
    assert doc["inputs"]["check_packs"] is None
    validate(doc)


def test_a_pack_that_is_not_named_is_noted_not_detected(tmp_path):
    doc = run_pipeline(
        make_config(),
        make_spec(2),
        tmp_path,
        packs=["core"],
        run_check_stage=False,
        created_at=STAMP,
    )
    assert doc["status"] == "ok"
    assert len(doc["notes"]) == 1 and "--packs core,tactile" in doc["notes"][0]
    validate(doc)


def test_unknown_packs_are_refused_before_anything_runs(tmp_path):
    with pytest.raises(ValueError, match="metric pack 'nope'"):
        run_pipeline(make_config(), make_spec(), tmp_path, packs=["nope"])
    with pytest.raises(ValueError, match="check pack 'nope'"):
        run_pipeline(make_config(), make_spec(), tmp_path, check_packs=["nope"])
    assert not list(tmp_path.iterdir())


def test_failed_entries_stop_the_pipeline_and_say_why(tmp_path, monkeypatch):
    def failing(config, spec, out, **kwargs):
        raise ValueError("stimulus does not fit the canvas")

    monkeypatch.setattr(pipeline, "run_dataset", failing)
    doc = run_pipeline(
        make_config(), make_spec(), tmp_path, packs=["core"], created_at=STAMP
    )
    assert doc["status"] == "failed"
    assert doc["error"] == "batch: stimulus does not fit the canvas"
    status = {s["name"]: s["status"] for s in doc["stages"]}
    assert status == {
        "build": "ok",
        "batch": "failed",
        "audit": "skipped",
        "declared": "skipped",
        "compare": "skipped",
        "checks": "skipped",
    }
    assert (
        "ValueError" in next(s for s in doc["stages"] if s["name"] == "batch")["detail"]
    )
    validate(doc)


def test_an_unexpected_error_still_writes_pipeline_json(tmp_path, monkeypatch):
    # A TypeError is not a stage error: it propagates, but only after pipeline.json
    # records the stage as failed (it used to stay "ok", with no file written).
    def broken(config, spec, out, **kwargs):
        raise TypeError("run_dataset() got an unexpected keyword argument")

    monkeypatch.setattr(pipeline, "run_dataset", broken)
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_pipeline(
            make_config(), make_spec(), tmp_path, packs=["core"], created_at=STAMP
        )
    doc = json.loads((tmp_path / REPORT_DIR / PIPELINE_FILE).read_text())
    assert doc["status"] == "failed" and doc["error"].startswith("TypeError: ")
    batch = next(s for s in doc["stages"] if s["name"] == "batch")
    assert batch["status"] == "failed" and "TypeError" in batch["detail"]
    validate(doc)


def test_entries_that_fail_to_simulate_fail_the_batch_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "run_dataset",
        lambda *a, **k: {"ok": 3, "failed": 1, "skipped": 0},
    )
    doc = run_pipeline(
        make_config(), make_spec(), tmp_path, packs=["core"], created_at=STAMP
    )
    assert doc["status"] == "failed" and "1 of 4 entries failed" in doc["error"]
    assert next(s for s in doc["stages"] if s["name"] == "audit")["status"] == "skipped"


def test_dataset_mapping_checks_its_draws():
    with pytest.raises(ValueError, match="at least 1"):
        dataset_mapping(WORLD, 0, 1, 100.0)
    mapping = dataset_mapping(WORLD, 5, 7, 100.0)["dataset"]
    assert mapping["splits"] == {"train": {"n": 5}} and mapping["seed"] == 7
    assert Path(mapping["world"]).is_absolute()


# ----------------------------------------------------------------- the command line


def run_cli(*argv):
    args = create_parser().parse_args(["evaluate", "run", *argv])
    return cmd_evaluate(args)


def test_cli_world_file(tmp_path, capsys):
    out = tmp_path / "run"
    code = run_cli(
        str(WORLD),
        "--out",
        str(out),
        "--design",
        str(DESIGN),
        "--draws",
        "3",
        "--duration-ms",
        "100",
        "--packs",
        "core,tactile",
        "--no-checks",
    )
    text = capsys.readouterr().out
    assert code == 0, text
    assert f"report folder: {out / REPORT_DIR}" in text
    assert "checks   skipped" in text
    doc = json.loads((out / REPORT_DIR / PIPELINE_FILE).read_text())
    assert doc["batch"]["ok"] == 3 and doc["counts"]["compared_pairs"] > 0


def test_cli_dataset_file(tmp_path, capsys):
    spec = tmp_path / "set.yml"
    spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "mini",
                    "world": str(WORLD),
                    "seed": 1,
                    "duration_ms": 100,
                    "splits": {"train": {"n": 2}, "validation": {"n": 1}},
                }
            }
        )
    )
    out = tmp_path / "run"
    code = run_cli(str(spec), "--out", str(out), "--design", str(DESIGN), "--no-checks")
    assert code == 0, capsys.readouterr().out
    doc = json.loads((out / REPORT_DIR / PIPELINE_FILE).read_text())
    assert doc["batch"]["ok"] == 3 and doc["inputs"]["dataset_name"] == "mini"


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "needs --draws N and --duration-ms MS"),
        (["--draws", "2", "--duration-ms", "50"], "no config file and no --preset"),
    ],
)
def test_cli_world_file_needs_its_arguments(tmp_path, capsys, extra, message):
    code = run_cli(str(WORLD), "--out", str(tmp_path / "run"), *extra)
    assert code == 1 and message in capsys.readouterr().err


def test_cli_data_set_file_refuses_draws(tmp_path, capsys):
    spec = tmp_path / "set.yml"
    spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "world": str(WORLD),
                    "seed": 1,
                    "duration_ms": 100,
                    "splits": {"train": {"n": 1}},
                }
            }
        )
    )
    code = run_cli(
        str(spec),
        "--out",
        str(tmp_path / "run"),
        "--design",
        str(DESIGN),
        "--draws",
        "2",
    )
    assert code == 1 and "apply to a world file" in capsys.readouterr().err


def test_cli_exit_code_follows_a_failed_stage(tmp_path, capsys, monkeypatch):
    def failing(config, spec, out, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pipeline, "run_dataset", failing)
    code = run_cli(
        str(WORLD),
        "--out",
        str(tmp_path / "run"),
        "--design",
        str(DESIGN),
        "--draws",
        "2",
        "--duration-ms",
        "50",
    )
    captured = capsys.readouterr()
    assert code == 1 and "batch: boom" in captured.err
    assert "batch    failed" in captured.out
