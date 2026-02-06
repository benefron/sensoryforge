import torch

from encoding.pipeline_torch import create_small_pipeline


def test_filter_methods_produce_expected_shapes():
    pipeline = create_small_pipeline(grid_size=16, device="cpu", seed=13)

    stimulus = torch.zeros(1, 16, 16)
    stimulus[..., 5:11, 5:11] = 5.0

    for method in ("steady_state", "multi_step", "edge_response"):
        outputs = pipeline(stimulus, filter_method=method)
        sa = outputs["sa_outputs"]
        ra = outputs["ra_outputs"]

        assert sa.ndim == 3 and ra.ndim == 3
        assert sa.shape[0] == ra.shape[0] == 1
        assert sa.shape[-1] == pipeline.sa_innervation.num_neurons
        assert ra.shape[-1] == pipeline.ra_innervation.num_neurons


def test_temporal_input_preserves_time_axis():
    pipeline = create_small_pipeline(grid_size=10, device="cpu", seed=21)

    temporal = torch.zeros(1, 6, 10, 10)
    temporal[:, :, 3:7, 3:7] = 2.0

    outputs = pipeline(temporal)

    assert outputs["sa_outputs"].shape[1] == temporal.shape[1]
    assert outputs["ra_outputs"].shape[1] == temporal.shape[1]
