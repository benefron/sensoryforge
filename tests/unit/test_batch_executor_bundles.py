"""BatchExecutor writes one bundle per stimulus, and the SLURM script's
flags all exist on the CLI parser (Wave J, J3, F-011, F-013).
"""

import shlex

import pytest
import torch

from sensoryforge.cli import create_parser
from sensoryforge.core.batch_executor import BatchExecutor
from sensoryforge.io.bundle import load_bundle


def _canonical_batch_config(output_dir, n_amplitudes=3):
    return {
        "metadata": {"batch_name": "bundle_test"},
        "base_config": {
            "grids": [
                {
                    "name": "test_grid",
                    "rows": 6,
                    "cols": 6,
                    "spacing": 1.0,
                    "arrangement": "grid",
                }
            ],
            "populations": [
                {
                    "name": "SA Pop",
                    "target_grid": "test_grid",
                    "neuron_type": "SA",
                    "neurons_per_row": 2,
                    "innervation_method": "gaussian",
                    "connections_per_neuron": 4,
                    "sigma_d_mm": 2.0,
                    "filter_method": "none",
                    "neuron_model": "Izhikevich",
                    "input_gain": 1.0,
                    "noise_std": 0.0,
                    "seed": 42,
                }
            ],
            "simulation": {"dt_ms": 1.0, "device": "cpu"},
        },
        "batch": {
            "output_dir": str(output_dir),
            "stimuli": [
                {
                    "type": "gaussian",
                    "parameters": {
                        "amplitude": [float(i) for i in range(10, 10 + n_amplitudes)]
                    },
                    "repetitions": 1,
                }
            ],
        },
    }


class TestOneBundlePerStimulus:
    def test_three_stimuli_three_bundles_distinct_spikes(self, tmp_path):
        config = _canonical_batch_config(tmp_path / "out", n_amplitudes=3)
        executor = BatchExecutor(config)
        assert len(executor.stimulus_configs) == 3

        results = executor.execute()
        assert results["num_stimuli"] == 3
        assert results["failed_stimuli"] == []

        bundle_dirs = sorted((executor.batch_root).glob("stim_*"))
        assert len(bundle_dirs) == 3
        assert [d.name for d in bundle_dirs] == [
            "stim_0000",
            "stim_0001",
            "stim_0002",
        ]

        spikes_by_idx = {}
        for d in bundle_dirs:
            bundle = load_bundle(d)
            spikes_by_idx[d.name] = bundle.populations["SA Pop"]["spikes"]

        # Distinct amplitudes -> not all three spike arrays identical.
        arrays = list(spikes_by_idx.values())
        all_equal = all(torch.equal(arrays[0], a) for a in arrays[1:])
        assert not all_equal, "Different amplitudes produced identical spikes"

        assert (executor.batch_root / "batch_metadata.json").exists()
        assert (executor.batch_root / "stimulus_index.json").exists()

    def test_task_index_reproduces_matching_bundle_exactly(self, tmp_path):
        config = _canonical_batch_config(tmp_path / "out", n_amplitudes=3)

        executor_full = BatchExecutor(config)
        executor_full.execute()
        full_bundle = load_bundle(executor_full.batch_root / "stim_0001")

        # Fresh executor (same batch_name -> same batch_id only if run in the
        # same second; instead compare against a fresh single-task executor
        # writing under its own root, using stim_config directly).
        executor_task = BatchExecutor(
            _canonical_batch_config(tmp_path / "out2", n_amplitudes=3)
        )
        result = executor_task.execute(task_index=1)
        assert result["num_stimuli"] == 1
        task_bundle_dir = executor_task.batch_root / "stim_0001"
        assert task_bundle_dir.exists()
        task_bundle = load_bundle(task_bundle_dir)

        assert torch.equal(
            full_bundle.populations["SA Pop"]["spikes"],
            task_bundle.populations["SA Pop"]["spikes"],
        )
        assert torch.equal(
            full_bundle.banks["SA Pop"].weights, task_bundle.banks["SA Pop"].weights
        )
        # Only stim_0001 was written by the task-index run.
        assert not (executor_task.batch_root / "stim_0000").exists()
        assert not (executor_task.batch_root / "stim_0002").exists()


class TestSlurmScriptFlagsExist:
    def test_every_flag_in_generated_script_is_accepted(self, tmp_path):
        config = _canonical_batch_config(tmp_path / "out", n_amplitudes=3)
        executor = BatchExecutor(config)
        script = executor.generate_slurm_script(
            "/tmp/fake_config.yml", gpus=0, output_dir=str(tmp_path / "out")
        )

        # Find the (possibly line-continued) `sensoryforge batch ...` call.
        lines = script.splitlines()
        start = next(
            i for i, l in enumerate(lines) if l.strip().startswith("sensoryforge")
        )
        call_lines = [lines[start]]
        i = start
        while call_lines[-1].rstrip().endswith("\\"):
            i += 1
            call_lines.append(lines[i])
        joined = " ".join(l.rstrip(" \\") for l in call_lines)

        # Substitute the shell variables the array job would have set.
        joined = joined.replace("$STIM_IDX", "0").replace(
            '"$OUTPUT_DIR"', str(tmp_path / "out")
        )
        argv = shlex.split(joined)
        assert argv[0] == "sensoryforge"
        argv = argv[1:]

        parser = create_parser()
        args = parser.parse_args(argv)
        assert args.command == "batch"
        assert args.task_index == 0
        assert args.output == str(tmp_path / "out")


class TestMonolithicFileRemoved:
    def test_no_consolidated_pt_or_h5_at_output_root(self, tmp_path):
        config = _canonical_batch_config(tmp_path / "out", n_amplitudes=2)
        executor = BatchExecutor(config)
        executor.execute()
        top_level_files = [
            p
            for p in (tmp_path / "out").iterdir()
            if p.is_file() and p.suffix in (".pt", ".h5")
        ]
        assert top_level_files == [], (
            f"expected no consolidated .pt/.h5 directly under output_dir, "
            f"got {top_level_files}"
        )
