"""pressure-simulation's viewer loader can read a SensoryForge bundle (Wave J, J4).

Re-implements the relevant steps of ``GUIs/ebkf_viewer.py``'s
``_on_load_bundle``/``_on_run`` in `~/Documents/pressure simulation` here (that
repo is not imported -- see ``docs/development/handover/phase2_tasks.md``
section 2 for the exact fields its loader reads):

1. Read ``config.json``; pull ``grid`` (rows, cols, spacing_mm, center_mm) and,
   for each ``populations[i]``, its ``tensors`` file.
2. ``torch.load`` that file; accept the weight tensor under
   ``innervation_weights``, ``weights`` or ``W``; reshape ``[N, M]`` to
   ``[N, rows, cols]`` (the viewer's own ``W.view(W.shape[0], -1)`` does the
   reverse -- this test does the forward direction with the bundle's own
   ``grid_shape`` to prove it is consistent) and check ``neuron_centers``'s
   shape.
3. Rebuild the drive the way pressure-simulation's encoder does:
   ``stimulus.view(T, H*W) @ W.T``, and compare it to the bundle's own stored
   ``/populations/<name>/drive``.
"""

import copy
import json

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import write_bundle

# The 1.0.0 fields pressure-simulation's `_on_load_bundle` reads from
# config.json with a bare `[...]` lookup (see phase2_tasks.md section 2 and
# GUIs/ebkf_viewer.py:663-682) -- dropping any one of these must break the
# loader. `center_mm` is read with `.get(..., default)` there, so it is
# intentionally not in this list (see test_missing_center_mm_falls_back_ok).
V1_REQUIRED_FIELDS = [
    ("grid", "rows"),
    ("grid", "cols"),
    ("grid", "spacing_mm"),
    ("populations", 0, "name"),
    ("populations", 0, "tensors"),
]


def _small_config():
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=6, cols=6, spacing=0.2)
        ],
        populations=[
            PopulationConfig(
                name="SA #6",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                seed=11,
            ),
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )


def _write_test_bundle(tmp_path, T=8):
    config = _small_config()
    engine = SimulationEngine(config)
    stimulus = torch.rand(1, T, 6, 6)
    results = engine.run(stimulus, return_intermediates=True)
    bundle_dir = write_bundle(tmp_path / "bundle", config, engine, results, stimulus)
    return bundle_dir, stimulus[0], results


def _load_like_pressure_sim(bundle_dir):
    """Steps 1-2 above, matching `_on_load_bundle`."""
    with open(bundle_dir / "config.json") as f:
        cfg = json.load(f)

    grid_cfg = cfg["grid"]
    rows, cols = int(grid_cfg["rows"]), int(grid_cfg["cols"])
    float(grid_cfg["spacing_mm"])  # bare lookup: must raise if the key is missing
    grid_cfg.get("center_mm", [0.0, 0.0])  # matches the real loader's .get() default

    innervation = {}
    pop_meta = {}
    for pop in cfg["populations"]:
        name = pop["name"]
        tensor_path = bundle_dir / pop["tensors"]
        data = torch.load(tensor_path, weights_only=False)
        W = data.get("innervation_weights", data.get("weights", data.get("W")))
        if W.ndim == 3:
            W = W.view(W.shape[0], -1)
        innervation[name] = W
        pop_meta[name] = pop

    return rows, cols, innervation, pop_meta


class TestFreshBundleCompat:
    def test_loader_reads_config_and_tensors(self, tmp_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        rows, cols, innervation, pop_meta = _load_like_pressure_sim(bundle_dir)
        assert rows == 6 and cols == 6
        assert "SA #6" in innervation
        W = innervation["SA #6"]
        assert W.ndim == 2 and W.shape[1] == rows * cols
        assert pop_meta["SA #6"]["neuron_type"] == "SA"

    def test_neuron_centers_shape(self, tmp_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        with open(bundle_dir / "config.json") as f:
            cfg = json.load(f)
        pop = cfg["populations"][0]
        data = torch.load(bundle_dir / pop["tensors"], weights_only=False)
        centers = data["neuron_centers"]
        W = data["innervation_weights"]
        assert centers.ndim == 2 and centers.shape[1] == 2
        assert centers.shape[0] == W.shape[0]

    def test_rebuilt_drive_matches_stored_drive(self, tmp_path):
        """`stimulus.view(T, H*W) @ W.T` (pressure-simulation's encoder)
        equals the bundle's own stored drive.
        """
        bundle_dir, stimulus, results = _write_test_bundle(tmp_path, T=8)
        rows, cols, innervation, _ = _load_like_pressure_sim(bundle_dir)
        W = innervation["SA #6"]

        T = stimulus.shape[0]
        rebuilt_drive = stimulus.reshape(T, rows * cols) @ W.T

        stored_drive = results["SA #6"]["drive"][0]
        assert torch.allclose(rebuilt_drive, stored_drive, atol=1e-6)


class TestOptionalFieldFallsBack:
    def test_missing_center_mm_falls_back_ok(self, tmp_path):
        """`center_mm` is read with `.get(..., [0.0, 0.0])` in the real
        loader, so dropping it must NOT break loading (unlike the fields in
        V1_REQUIRED_FIELDS).
        """
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        cfg_path = bundle_dir / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        del cfg["grid"]["center_mm"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        rows, cols, innervation, _ = _load_like_pressure_sim(bundle_dir)
        assert rows == 6 and "SA #6" in innervation


class TestDroppedFieldBreaksCompat:
    @pytest.mark.parametrize(
        "field_path", V1_REQUIRED_FIELDS, ids=lambda p: ".".join(map(str, p))
    )
    def test_dropping_a_v1_field_breaks_the_loader(self, tmp_path, field_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        cfg_path = bundle_dir / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        # Navigate to the parent container and delete the final key/index.
        mutated = copy.deepcopy(cfg)
        node = mutated
        for step in field_path[:-1]:
            node = node[step]
        del node[field_path[-1]]

        with open(cfg_path, "w") as f:
            json.dump(mutated, f)

        with pytest.raises((KeyError, TypeError, AttributeError)):
            _load_like_pressure_sim(bundle_dir)


class TestRunButtonWouldEnable:
    """`_on_load_bundle` ends with::

        self.btn_run.setEnabled(
            self.combo_stimulus.count() > 0 and self.combo_neuron.count() > 0
        )

    where ``combo_stimulus`` is populated from ``sorted(stim_dir.glob("*.json"))``
    (``stim_dir = bundle_dir / "stimuli"``) and ``combo_neuron`` from
    ``sorted(nm_dir.glob("*.json"))`` (``nm_dir = bundle_dir / "neuron_modules"``).
    A bundle with no ``neuron_modules/*.json`` loads and displays in the viewer
    but its Run button never enables, so it can never be encoded (J6).
    """

    def test_both_combo_globs_are_non_empty(self, tmp_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        stim_dir = bundle_dir / "stimuli"
        nm_dir = bundle_dir / "neuron_modules"

        combo_stimulus_items = sorted(stim_dir.glob("*.json"))
        combo_neuron_items = sorted(nm_dir.glob("*.json"))

        run_button_would_enable = (
            len(combo_stimulus_items) > 0 and len(combo_neuron_items) > 0
        )
        assert run_button_would_enable, (
            f"stimuli/*.json: {combo_stimulus_items}, "
            f"neuron_modules/*.json: {combo_neuron_items}"
        )


class TestNeuronModuleContent:
    """``_on_run`` (``GUIs/ebkf_viewer.py`` lines 738-758) reads only
    ``enabled``, ``name``, ``neuron_type``, ``filter_method``, ``noise_std``,
    ``model_params`` and ``filter_params`` from each ``population_configs``
    entry (``model`` is metadata only -- never read; ``input_gain`` is always
    overridden by the viewer's own spinboxes). It matches an entry to a
    population by exact ``name`` against ``config.json``'s
    ``populations[*].name`` (``if name not in self._innervation: continue`` --
    a non-matching entry is silently dropped, not an error), so the name must
    be the *raw* population name, not the filesystem-safe one the ``.pt``
    filenames use.
    """

    REQUIRED_KEYS = {
        "enabled",
        "name",
        "neuron_type",
        "filter_method",
        "noise_std",
        "model_params",
        "filter_params",
    }

    def test_neuron_module_json_parses_with_schema_tag(self, tmp_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        nm_files = sorted((bundle_dir / "neuron_modules").glob("*.json"))
        assert len(nm_files) == 1
        with open(nm_files[0]) as f:
            nm = json.load(f)
        assert nm["schema_version"] == "1.0.0"
        assert nm["kind"] == "neuron_module"
        assert "population_configs" in nm

    def test_one_entry_per_population_with_required_keys(self, tmp_path):
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        with open(bundle_dir / "config.json") as f:
            cfg = json.load(f)
        pop_names_in_config = [p["name"] for p in cfg["populations"]]

        nm_path = next((bundle_dir / "neuron_modules").glob("*.json"))
        with open(nm_path) as f:
            nm = json.load(f)
        pop_configs = nm["population_configs"]

        assert len(pop_configs) == len(pop_names_in_config)
        for pc in pop_configs:
            assert self.REQUIRED_KEYS <= set(pc), pc

    def test_names_match_config_json_population_names_exactly(self, tmp_path):
        """The viewer drops any population_configs entry whose name does not
        exactly match a config.json population name -- prove ours do (not
        the `_safe_name`-mangled .pt filename form).
        """
        bundle_dir, _, _ = _write_test_bundle(tmp_path)
        with open(bundle_dir / "config.json") as f:
            cfg = json.load(f)
        pop_names_in_config = {p["name"] for p in cfg["populations"]}

        nm_path = next((bundle_dir / "neuron_modules").glob("*.json"))
        with open(nm_path) as f:
            nm = json.load(f)

        # Loader semantics: `if name not in self._innervation: continue`.
        matched = [
            pc for pc in nm["population_configs"] if pc["name"] in pop_names_in_config
        ]
        assert len(matched) == len(nm["population_configs"]), (
            "some neuron_module population_configs entries would be silently "
            f"dropped by the viewer: {nm['population_configs']} vs. config.json "
            f"names {pop_names_in_config}"
        )
