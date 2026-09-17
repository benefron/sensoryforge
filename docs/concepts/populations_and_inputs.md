# Populations and inputs

A population is a set of neurons. Before Phase 2 Wave M, each population read exactly one
receptor bank: one grid, one channel, one receptive-field builder. Wave M generalises this --
a population's neurons can read from **several inputs**, each its own grid/channel/receptive
field, with an optional processing stage per input, combined into one population drive. This is
what makes SimulationEngine no longer tactile-only: see
`sensoryforge/presets/vision_onoff_rgb.yml` and `examples/vision_rgb_onoff.py` for the concrete
demo, a population reading R/G/B channels of one vision-style sensor array.

See [Configuration Schema](../user_guide/configuration_schema.md#multi-input-populations-phase-2-wave-m1m2)
for the field reference and [Sensor Arrays](sensor_arrays.md) for what a grid/channel is.

## One neuron, several inputs

`PopulationConfig.inputs` is a list of `PopulationInput`, each with:

- `grid` -- which `GridConfig` this input samples,
- `channel` -- which named plane of that grid (`GridConfig.channels`; `"value"` is the
  single-channel default),
- `rf` -- an `RFBuilderConfig` (`method` + `params`) selecting the receptive-field builder for
  *this* input, independently of any other input's,
- `gain` -- a scalar multiplier on this input's drive before combining,
- `layers` -- for a composite grid, the named layer subset this input reads,
- `processing` -- an ordered list of processing-layer specs applied to this input's receptor
  responses before its receptive-field bank (see below).

`SimulationEngine._build_populations` builds one `ReceptiveFieldBank` per input
(`_build_input_bank`), on that input's own grid/channel/builder. Every input that does not derive
its own neuron lattice (`BaseInnervation.DERIVES_NEURON_CENTERS`) shares one lattice, computed
once from the population's first such input -- the population's neurons are one set of cells that
happen to read several signals, not several independent populations glued together.

## Sum versus concat

`PopulationConfig.combine` says how the per-input drives become the population's one drive:

- **`"sum"`**: element-wise. Every input must produce the same neuron count `N` -- the same
  neurons, driven by the sum of what each input tells them. This is the natural mode for several
  *channels of the same sensor modality* (e.g. R and G planes of a vision-style array) landing on
  one set of neurons.
- **`"concat"`**: the neuron axis grows to `sum(N_i)` -- each input gets its own block of neurons.
  This is the natural mode for inputs that should stay *distinguishable downstream* (e.g. reading
  R, G and B as three separate populations of neurons rather than blending them).

Internally, `SimulationEngine._combine_banks` folds the per-input banks into a single combined
`ReceptiveFieldBank` whose `forward()` on the concatenation (along the receptor axis, in the same
order) of every input's own (post-processing) receptor response reproduces the combination
exactly:

- `"sum"`: the combined weights are `hstack(gain_i * weights_i)` (receptor axis concatenated,
  neuron axis shared) -- one matmul equals `sum_i gain_i * bank_i(response_i)`.
- `"concat"`: the combined weights are block-diagonal (each input's own `[N_i, M_i]` block, zero
  elsewhere) -- one matmul equals `cat([gain_i * bank_i(response_i) for i], dim=-1)`, and the
  blocks cannot interact even under floating-point rounding, since the padding is literal `0.0`.

A population with no `inputs` (the common case, sugar fields only) is unaffected: `_combine_banks`
returns the single bank unchanged when there is exactly one input and its gain is `1.0`.

`combine` is a fixed two-way choice (`"sum"` or `"concat"`), not a registry-backed plugin point
like the seven component kinds in `docs/extending/` -- there is no `COMBINE_REGISTRY` and no way to
add a third combination rule today without editing `SimulationEngine._combine_banks` itself.

## Where processing sits in the spine

```
Stimulus  [batch, time, C, H, W]
    ↓  per-input: select this input's channel plane, sample at its grid's receptor coordinates
    ↓  [batch, time, M]
    ↓  per-input: ProcessingPipeline (PopulationInput.processing) -- empty by default, no allocation
    ↓  [batch, time, M']   (M' == M unless a layer changes the receptor axis, e.g. OnOffLayer -> 2M)
    ↓  per-input: ReceptiveFieldBank, built on the *post-processing* receptor axis
    ↓  [batch, time, N_i]
    ↓  combine (sum | concat) across inputs
    ↓  [batch, time, N]
    ↓  Filter -> Neuron  (population-level, same as before Wave M)
```

A processing layer sits strictly between receptor sampling and the receptive-field bank, per
input -- it never sees another input's response, and the population's filter/neuron stage never
sees per-input structure at all (by the time it runs, everything is one combined drive). This is
why a processing layer that changes the receptor count (`OnOffLayer`'s ON/OFF split, doubling `M`)
only has to get the *bank* right: `BaseProcessingLayer.expand_receptor_coords` tells
`_build_input_bank` how to grow the receptor coordinates so the bank's weight columns line up with
the pipeline's output, column for column. See
[Adding a Processing Layer](../extending/add_processing_layer.md) for the full worked example.

## The sugar fields (single-input populations)

Every population field that existed before Wave M -- `target_grid`, `target_layers`,
`innervation_method`, `sigma_d_mm`, `connections_per_neuron`, `use_distance_weights`,
`resolvable_distance_mm`, `innervation_params` -- still works exactly as before, and is what
`PopulationConfig.effective_inputs()` expands into one implicit `PopulationInput` when `inputs`
is empty. `to_dict()` does the reverse: a population with exactly one input that fits the sugar
shape (default channel, gain `1.0`, no processing) serializes back to those same fields, so every
config written before Wave M round-trips to byte-identical YAML. Setting both `inputs` and one of
the sugar fields on the same population raises `ValueError` -- pick one form.
