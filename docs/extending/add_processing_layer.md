# Adding a Processing Layer

A processing layer sits between receptor sampling and the receptive-field bank, per population
input (`PopulationInput.processing`, Phase 2 Wave M3) -- see
[Populations and Inputs](../concepts/populations_and_inputs.md) for where it fits in the pipeline.
The shipped `OnOffLayer` (`sensoryforge/core/processing.py`) is the worked example this page walks
through, and the complete runnable script is `docs/examples/onoff_processing_layer.py`: it
defines a minimal custom layer, registers it, then uses `OnOffLayer` standalone and wired into a
real `SimulationEngine` run. It is executed by `tests/docs/test_docs_examples.py`.

---

## 1. The contract

| Method | Purpose |
|---|---|
| `forward(receptor_responses, metadata=None)` | Transform `[..., M]` receptor responses. May change the last dimension (`OnOffLayer`: `M -> 2M`). |
| `to_dict()` / `from_config()` | `PopulationInput.processing` entries are `{"method": <name>, "params": {...}}` (the same shape `RFBuilderConfig` uses); `from_config` should accept both a flat dict and this nested `params` form. |
| `get_param_spec()` | A `ParamSpec` per user-facing parameter, for the GUI (default: `[]`). |
| `REQUIRES_RECEPTOR_COORDS` | Class attribute, default `False`. Set `True` when the layer's `from_config` needs the input's receptor coordinates (e.g. to compute inter-receptor distances) -- `ProcessingPipeline.from_config`/`.expand_receptor_coords` then pass `receptor_coords=` to it. |
| `expand_receptor_coords(receptor_coords)` | Classmethod. Default: identity. Override when `forward()` changes `M`, so `SimulationEngine._build_input_bank` builds the receptive-field bank on the matching post-processing receptor axis, column for column. |

Register with `PROCESSING_REGISTRY` (`sensoryforge.registry`) -- either in-repo
(`register_components.py`'s `register_all()`, one line, what `OnOffLayer` does) or from a plugin
package's own entry point, the same as any other component kind (see
`docs/developer_guide/plugins.md`).

A layer that does **not** change `M` needs nothing beyond `forward()`/`to_dict()`/`from_config()`
-- `expand_receptor_coords` and `REQUIRES_RECEPTOR_COORDS` both default correctly. The empty list
(no processing at all, `PopulationInput.processing == []`) never even constructs a
`ProcessingPipeline` at run time -- the identity path allocates nothing, matching the same
convention `GridConfig.channels`/`PopulationConfig.inputs` use for their own defaults.

---

## 2. `OnOffLayer`, read as the example

A centre-surround receptive field is the textbook first stage of vision: a difference-of-Gaussians
(DoG) kernel, positive at the centre and negative in the surround (or vice versa), splits a scene
into an **ON** channel (responds to brightness increases) and an **OFF** channel (responds to
decreases). `OnOffLayer` computes the DoG directly over *receptor coordinates* -- not spatial grid
indices -- so it works for any arrangement (grid, hex, Poisson, imported), and applies it as one
matmul:

```python
class OnOffLayer(BaseProcessingLayer):
    REQUIRES_RECEPTOR_COORDS = True

    def __init__(self, receptor_coords, *, sigma_center_mm=0.15, sigma_surround_mm=0.45):
        super().__init__()
        coords = receptor_coords.detach().to(torch.float32)
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(dim=-1)  # [M, M]
        dog = _gaussian(d2, sigma_center_mm) - _gaussian(d2, sigma_surround_mm)
        self.register_buffer("dog_kernel", dog.contiguous())

    def forward(self, receptor_responses, metadata=None):
        center_surround = torch.matmul(receptor_responses, self.dog_kernel.T)
        on = torch.clamp(center_surround, min=0.0)
        off = torch.clamp(-center_surround, min=0.0)
        return torch.cat([on, off], dim=-1)          # [..., 2M]: ON then OFF

    @classmethod
    def expand_receptor_coords(cls, receptor_coords):
        return torch.cat([receptor_coords, receptor_coords], dim=0)   # [2M, 2]
```

Two things worth internalising before writing your own layer:

- **`REQUIRES_RECEPTOR_COORDS = True`** is what makes `ProcessingPipeline.from_config` pass
  `receptor_coords=` to `OnOffLayer.from_config` -- a layer that only needs its own constructor
  parameters (a plain gain, say) does not set this and is constructed the ordinary way.
- **`expand_receptor_coords` must mirror `forward()` exactly.** `OnOffLayer` doubles `M` by
  concatenating the ON and OFF planes; its coordinate expansion concatenates the *same* receptor
  coordinates twice, in the same order, so the receptive-field bank built on the doubled
  coordinates has its weight column `k` line up with `forward()`'s output column `k` for every
  `k`. Get this wrong (e.g. interleave instead of concatenate) and the bank silently wires ON-plane
  weights onto OFF-plane responses -- a plausible-looking, wrong result no shape test catches.

## 3. The anti-plausibility check this layer needs

An ON/OFF split with the two planes accidentally swapped, or collapsed to the same values, passes
every shape test (`[..., 2M]` either way) and every round-trip test. The check that actually
exercises the *sign* logic: drive the layer with a stimulus unambiguously asymmetric in sign, and
assert **which** plane responds.

```python
resp = torch.zeros(1, 1, 6)
resp[0, 0, 3] = 5.0  # a bright spot at receptor 3
on, off = layer(resp)[..., :6], layer(resp)[..., 6:]
assert on[0, 0, 3] > 0 and off[0, 0, 3] == 0   # bright -> ON only

resp[0, 0, 3] = -5.0  # a dark spot at the same receptor
on, off = layer(resp)[..., :6], layer(resp)[..., 6:]
assert off[0, 0, 3] > 0 and on[0, 0, 3] == 0   # dark -> OFF only
```

`tests/unit/test_onoff_layer.py` runs exactly this (plus registration, `get_param_spec`,
`to_dict`/`from_config` round trip, and an engine-level check that a population input with
`processing: [{method: onoff}]` builds its bank on the doubled receptor axis).

## 4. Using it from a config

```yaml
populations:
  - name: "RG OnOff Population"
    neuron_type: SA
    neurons_per_row: 8
    inputs:
      - grid: "RGB Grid"
        channel: "R"
        processing:
          - method: onoff
      - grid: "RGB Grid"
        channel: "G"
        processing:
          - method: onoff
    combine: sum
```

See `sensoryforge/presets/vision_onoff_rgb.yml` and `examples/vision_rgb_onoff.py` for the full
demo this powers.
