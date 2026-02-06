import torch

from sensoryforge.core.pipeline import (
    TactileEncodingPipelineTorch,
    create_small_pipeline,
)


def test_forward_static_gaussian_produces_spikes():
    pipeline = TactileEncodingPipelineTorch(
        overrides={"pipeline": {"device": "cpu", "seed": 7}},
    )
    stimulus = pipeline.generate_stimulus(
        stimulus_type="gaussian",
        amplitude=2.0,
        sigma=0.3,
    )

    outputs = pipeline(stimulus, filter_method="multi_step")

    assert "sa_spikes" in outputs and "ra_spikes" in outputs
    sa_spikes = outputs["sa_spikes"]
    ra_spikes = outputs["ra_spikes"]

    assert sa_spikes.ndim == 3
    assert ra_spikes.ndim == 3
    assert sa_spikes.shape[0] == ra_spikes.shape[0] == 1
    assert sa_spikes.dtype == torch.bool
    assert ra_spikes.dtype == torch.bool


def test_create_small_pipeline_overrides_grid_and_device():
    pipeline = create_small_pipeline(grid_size=24, device="cpu", seed=11)
    info = pipeline.get_pipeline_info()

    assert info["grid"]["grid_size"] == (24, 24)
    assert info["device"] == "cpu"
    assert info["neurons"]["sa_neurons"] == pipeline.sa_innervation.num_neurons
