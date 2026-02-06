from sensoryforge.core.pipeline import create_small_pipeline


def test_visualize_receptive_field_returns_grid_shape():
    pipeline = create_small_pipeline(grid_size=12, device="cpu", seed=5)

    sa_field = pipeline.visualize_neuron_receptive_field("SA", 0)
    ra_field = pipeline.visualize_neuron_receptive_field("RA", 0)

    assert sa_field.shape == pipeline.grid_manager.grid_size
    assert ra_field.shape == pipeline.grid_manager.grid_size
