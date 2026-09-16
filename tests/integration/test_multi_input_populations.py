"""Multi-input populations: one bank per input, sum/concat combine (Phase 2, Wave M2).

Per the Wave M spec's anti-plausibility guardrail: summing and concatenating
drives produces a plausible-looking tensor no matter how wrong the wiring is,
so these tests compare against an independently computed reference rather
than only checking shapes.
"""

from __future__ import annotations

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import SimulationEngine


class TestCombineBanksSum:
    def test_sum_combine_equals_single_bank_on_summed_response(self):
        """Two inputs carrying identical banks and complementary stimuli give
        exactly the same drive as one input carrying their summed stimulus.

        Uses small-integer-valued float32 tensors so every intermediate sum
        is exact regardless of BLAS reduction order, making torch.equal a
        valid check of the *combination logic itself* (not floating-point
        luck).
        """
        weights = torch.tensor([[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
        centers = torch.zeros(2, 2)
        coords = torch.zeros(4, 2)
        bank = ReceptiveFieldBank(weights, centers, coords)

        combined = SimulationEngine._combine_banks(
            [bank, bank], [1.0, 1.0], "sum", ["a", "b"]
        )

        resp_a = torch.tensor([[[1.0, 0.0, 2.0, 3.0]]])  # [1, 1, 4]
        resp_b = torch.tensor([[[4.0, 1.0, 0.0, 2.0]]])
        resp_total = resp_a + resp_b

        drive_two_input = combined(torch.cat([resp_a, resp_b], dim=-1))
        drive_single_input = bank(resp_total)

        assert torch.equal(drive_two_input, drive_single_input)
        # And it isn't trivially all-zero -- a real check, not a vacuous one.
        assert drive_two_input.abs().sum().item() > 0

    def test_sum_requires_matching_neuron_count(self):
        w1 = torch.rand(3, 4)
        w2 = torch.rand(5, 4)
        b1 = ReceptiveFieldBank(w1, torch.zeros(3, 2), torch.zeros(4, 2))
        b2 = ReceptiveFieldBank(w2, torch.zeros(5, 2), torch.zeros(4, 2))
        try:
            SimulationEngine._combine_banks([b1, b2], [1.0, 1.0], "sum", ["a", "b"])
            raise AssertionError("expected ValueError for mismatched N")
        except ValueError as exc:
            assert "sum" in str(exc)


class TestCombineBanksConcat:
    def test_concat_blocks_match_single_input_runs(self):
        """concat produces the expected shape AND each block equals the
        single-input run it came from -- not merely N * len(inputs)."""
        torch.manual_seed(0)
        w1 = torch.rand(3, 5)
        w2 = torch.rand(4, 6)
        bank1 = ReceptiveFieldBank(w1, torch.rand(3, 2), torch.rand(5, 2))
        bank2 = ReceptiveFieldBank(w2, torch.rand(4, 2), torch.rand(6, 2))

        combined = SimulationEngine._combine_banks(
            [bank1, bank2], [1.0, 1.0], "concat", ["a", "b"]
        )
        assert combined.num_neurons == 3 + 4
        assert combined.num_receptors == 5 + 6

        resp1 = torch.rand(2, 7, 5)
        resp2 = torch.rand(2, 7, 6)
        drive_combined = combined(torch.cat([resp1, resp2], dim=-1))

        drive1_alone = bank1(resp1)
        drive2_alone = bank2(resp2)

        # The zero-padded block contributes literal 0.0 terms to the
        # matmul, so this is exact in the *math*; a block-tiled BLAS GEMM
        # can still reorder the nonzero terms' additions differently when
        # the total reduction width changes, so compare numerically
        # (tight) rather than demanding bit-for-bit equality here.
        assert torch.allclose(drive_combined[..., :3], drive1_alone, rtol=0, atol=1e-6)
        assert torch.allclose(drive_combined[..., 3:], drive2_alone, rtol=0, atol=1e-6)

    def test_concat_with_gain_scales_each_block(self):
        torch.manual_seed(1)
        w1 = torch.rand(2, 3)
        w2 = torch.rand(2, 3)
        bank1 = ReceptiveFieldBank(w1, torch.rand(2, 2), torch.rand(3, 2))
        bank2 = ReceptiveFieldBank(w2, torch.rand(2, 2), torch.rand(3, 2))
        combined = SimulationEngine._combine_banks(
            [bank1, bank2], [2.0, 0.5], "concat", ["a", "b"]
        )
        resp1 = torch.rand(1, 4, 3)
        resp2 = torch.rand(1, 4, 3)
        drive_combined = combined(torch.cat([resp1, resp2], dim=-1))
        assert torch.allclose(
            drive_combined[..., :2], 2.0 * bank1(resp1), rtol=0, atol=1e-6
        )
        assert torch.allclose(
            drive_combined[..., 2:], 0.5 * bank2(resp2), rtol=0, atol=1e-6
        )


class _RGBConfig:
    """Shared config builder: one grid with 3 channels, two multi-input
    populations (sum and concat) reading it."""

    @staticmethod
    def build(combine: str) -> SensoryForgeConfig:
        grid = GridConfig(
            name="RGB Grid",
            arrangement="grid",
            rows=6,
            cols=6,
            spacing=0.2,
            channels=["R", "G", "B"],
        )
        pop = PopulationConfig(
            name="pop",
            neuron_type="SA",
            neurons_per_row=3,
            neuron_model="Izhikevich",
            filter_method="none",
            inputs=[
                PopulationInput(
                    grid="RGB Grid",
                    channel="R",
                    rf=RFBuilderConfig(method="one_to_one"),
                ),
                PopulationInput(
                    grid="RGB Grid",
                    channel="G",
                    rf=RFBuilderConfig(method="one_to_one"),
                ),
            ],
            combine=combine,
        )
        return SensoryForgeConfig(
            grids=[grid],
            populations=[pop],
            simulation=SimulationConfig(device="cpu", dt_ms=1.0),
        )


class TestEngineMultiInputIntegration:
    def test_engine_builds_one_bank_per_input(self):
        config = _RGBConfig.build("concat")
        engine = SimulationEngine(config)
        pop = engine.populations[0]
        assert len(pop["inputs"]) == 2
        assert pop["inputs"][0]["channel"] == "R"
        assert pop["inputs"][1]["channel"] == "G"

    def test_concat_neuron_count_doubles(self):
        config_single = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="RGB Grid",
                    arrangement="grid",
                    rows=6,
                    cols=6,
                    spacing=0.2,
                    channels=["R", "G", "B"],
                )
            ],
            populations=[
                PopulationConfig(
                    name="pop",
                    neuron_type="SA",
                    neurons_per_row=3,
                    inputs=[
                        PopulationInput(
                            grid="RGB Grid",
                            channel="R",
                            rf=RFBuilderConfig(method="one_to_one"),
                        )
                    ],
                )
            ],
        )
        config_concat = _RGBConfig.build("concat")
        engine_single = SimulationEngine(config_single)
        engine_concat = SimulationEngine(config_concat)
        n_single = engine_single.populations[0]["bank"].num_neurons
        n_concat = engine_concat.populations[0]["bank"].num_neurons
        assert n_concat == 2 * n_single

    def test_run_produces_drive_matching_manual_combination(self):
        """End-to-end: engine.run() on a 3-channel stimulus produces a
        drive equal to manually sampling+combining each channel."""
        config = _RGBConfig.build("sum")
        engine = SimulationEngine(config)
        torch.manual_seed(0)
        stimulus = torch.rand(1, 5, 3, 6, 6)  # [batch, T, C, H, W]

        results = engine.run(stimulus, return_intermediates=True)
        drive = results["pop"]["drive"]

        pop = engine.populations[0]
        bank = pop["bank"]
        ctx_r, ctx_g = pop["inputs"]
        resp_r = stimulus[:, :, 0].reshape(1, 5, 36)
        resp_g = stimulus[:, :, 1].reshape(1, 5, 36)
        manual_drive = bank(torch.cat([resp_r, resp_g], dim=-1))
        assert torch.allclose(drive, manual_drive)
        assert drive.shape == (1, 5, bank.num_neurons)
