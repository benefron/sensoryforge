# Scientific Hypothesis and Conceptual Foundation

**Project:** Bio-Inspired Sensory Data Encoding and Transmission  
**Date:** October 2025  
**Core Research Question:** Can biologically-inspired coding schemes enable efficient, event-based sensor communication?

---

## Central Hypothesis

> **"Sensory neurons' coding scheme of Slowly Adapting (SA) and Fast Adapting (FA) is sufficient to reconstruct spatiotemporal stimuli from spike trains when encoding parameters are known."**

This hypothesis posits that the dual-pathway strategy observed in biological sensory systems—where SA neurons encode sustained stimulus properties and FA neurons encode temporal dynamics—provides a complete and efficient representation for stimulus reconstruction, bridging the gap between dense sensor arrays and sparse, event-driven communication.

---

## Biological Principles as Engineering Framework

The nervous system solves a fundamental communication problem: transmitting high-dimensional sensory data through bandwidth-limited neural pathways while preserving behaviorally-relevant information. This project adopts two complementary principles from sensory neuroscience to build an encoding pipeline applicable to tactile sensing and extensible to other modalities, sensor fusion, and hierarchical abstraction:

### 1. Hardwired Attributes: The "Labeled Line" Principle

**Biological context:** Müller's doctrine of specific nerve energies states that the *identity* of information (modality, location, feature type) is determined by the physical pathway, not the signal itself. In the somatosensory system, this manifests as:

- **Spatial labeled lines**: Each neuron's receptive field (RF) defines its "where" information—the region of skin it monitors.
- **Feature-selective labeled lines**: RFs can encode not just location but also *stimulus features* (edge orientation, texture frequency, curvature)—sensory systems often transmit extracted features rather than raw sensor values.
- **Modality labeled lines**: SA vs. RA/FA pathways separate touch, pressure, and vibration by their temporal filtering properties.
- **Population coding**: Overlapping receptive fields provide spatial coverage; population statistics encode fine spatial detail through ensemble activity. Each sensory neuron integrates input from an *average number of receptors*, defining the innervation density and spatial resolution.

**Engineering translation:**

- **Receptive field optimization**: Design spatially distributed, Gaussian-weighted RFs that tile the sensor grid with controlled overlap. RF parameters include:
  - **Gaussian spread (σ)**: Spatial extent of each RF.
  - **Weight range**: Magnitude of connection strengths.
  - **Mean number of connections**: Average number of sensors each neuron receives input from, controlling innervation density and spatial resolution.
  
  Each neuron integrates signals from multiple sensors, reducing dimensionality while preserving spatial structure. RFs can also be tuned to extract specific features (edges, gradients, local patterns).
  
- **Population architecture**: Initially, the system uses **two populations** (SA and RA/FA). Future extensions may implement a **four-population design** inspired by human fingertip mechanoreceptors:
  - **Type 1 (high-resolution)**: Small RFs, dense innervation (e.g., SA-I, RA-I analogs) for fine spatial detail.
  - **Type 2 (low-resolution)**: Large RFs, sparse innervation (e.g., SA-II, RA-II analogs) for global stimulus properties and integration.
- **Fixed architecture**: The innervation tensors (connections from sensors → neurons) are *learned* or *designed* once, then frozen during deployment. This "wiring diagram" encodes spatial priors (location, coverage, feature selectivity) without requiring dynamic reconfiguration.
- **Reconstruction via innervation maps**: During decoding, the known RF structure allows analytical inversion—responses can be mapped back to sensor space using the fixed innervation weights.

### 2. Dynamic Attributes: Real-Time Feature Decomposition

**Biological context:** SA and FA neurons decompose tactile stimuli into complementary temporal features:

- **SA neurons** (Slowly Adapting):
  - Act as low-pass filters—respond sustained pressure and static indentation.
  - Encode **intensity** (magnitude of stimulus) and **slow changes** (e.g., gradual force increases).
  - Enable online monitoring of contact state and force magnitude.
  
- **RA/FA neurons** (Rapidly/Fast Adapting):
  - Act as high-pass or band-pass filters—respond to stimulus *onset*, *offset*, and dynamic changes.
  - Encode **timing** (when events occur), **rate of change** (derivative of stimulus), and **transients** (edges, vibrations).
  - Provide temporal precision for detecting motion, texture, and vibration frequencies.

**Engineering translation:**

- **Dual-pathway filtering**: Implement biologically-inspired SA and RA temporal filters (equations from Parvizi-Fard et al. and experimental data) that decompose raw sensor signals into:
  - **SA pathway**: Sustained response component (low-pass filtered mechanoreceptor output → intensity coding).
  - **RA pathway**: Transient response component (high-pass or derivative-like filtered output → change detection, timing).
  
- **Sparse, event-driven encoding**: Convert filtered signals into spike trains using spiking neuron models (Izhikevich, AdEx). Sparsity arises naturally:
  - SA neurons spike mainly during sustained stimulation → temporal sparsity during static phases.
  - RA neurons spike primarily at stimulus edges and changes → sparsity during constant-velocity or static phases.
  
- **Temporal compression**: Event-based (spike) representation eliminates redundant samples—data is transmitted *only when information changes*, dramatically reducing bandwidth compared to fixed-rate sampling.

---

## System Architecture: From Sensors to Reconstruction

The project pipeline instantiates these principles in four stages:

### Stage 1: Sensor Grid → Stimulus Generation

- **Input**: 2D pressure/tactile sensor array (e.g., 40×40 grid).
- **Stimulus module**: Generates spatiotemporal stimuli with built-in spatial spread characteristics. The stimulus equations govern spatial propagation, providing sufficient detail for proof-of-concept validation. (Future real-system implementations may require learned mechanoreceptor models.)
- **Output**: Spatially distributed stimulus ready for innervation stage.

### Stage 2: Innervation → Population Encoding

- **Receptive fields**: Each neuron receives weighted input from a subset of mechanoreceptors, defined by a Gaussian RF centered at a specific grid location. RF parameters (σ, weight range) control spatial resolution and overlap.
- **Innervation tensors**: Dense 3D tensors `[num_neurons, grid_h, grid_w]` encode the "wiring diagram"—the labeled lines. Separate innervation maps exist for SA and RA populations.
- **Population response**: Weighted sum of mechanoreceptor outputs across each neuron's RF yields the population activation vector for SA and RA pathways.

### Stage 3: Temporal Filtering → SA/RA Decomposition

- **SA filter**: Low-pass dynamics (linear or adaptation-based) extract sustained stimulus components. Neurons in the SA population encode integrated pressure over time.
- **RA filter**: High-pass or derivative dynamics emphasize stimulus *changes*. RA neurons spike at edges, onsets, and during motion.
- **Implementation**: `filters_torch.CombinedSARAFilter` provides three operational modes:
  - **`steady_state`**: Computes analytical steady-state response for constant inputs (SA: `I_SA = k1 * I_in`; RA: zero response for constant input). Used for spatial-only stimuli.
  - **`multi_step`**: Simulates filter dynamics over multiple time steps to approach steady state from initial conditions. Allows filters to "settle" before spiking stage.
  - **`edge_response`**: Simulates transient response to step changes (0 → I_in transition). Captures RA neurons' sensitivity to stimulus onset and SA neurons' rise dynamics.

### Stage 4: Spiking Encoding → Sparse Event Stream

- **Neuron models**: Biophysically-inspired spiking models (Izhikevich by default, with AdEx/MQIF/FA alternatives) convert filtered currents into discrete spike trains.
- **Noise & variability**: Optional intrinsic noise (`noise_std`) models biological stochasticity, improving robustness.
- **Output**: Spike rasters `[population, time_steps]`—the event-based sensory representation. Sparse by design: spikes occur only when neurons cross threshold, concentrating information at behaviorally-relevant moments.

### Stage 5: Decoding → Stimulus Reconstruction

The final stage maps the population responses (continuous currents or spike
rasters) back to the stimulus domain using a progression of decoders that grow
in complexity:

1. **Analytical baselines** – pseudo-inverse and ridge decoders that assume the
  innervation map is known and directly invertible under mild regularisation.
2. **Probabilistic decoders** – GLM/MAP models that learn stimulus-to-spike
  mappings while accounting for noise and biological variability.

These readouts operate on the same labeled-line architecture established in the
earlier stages; no additional sampling layers are required beyond the existing
innervation design.

#### Probabilistic Decoding Framework

While analytical inversion provides a baseline, biological systems operate in noisy, uncertain environments. We adopt a **State-Space Model (SSM)** framework to formalize the decoding problem as optimal estimation under uncertainty.

The stimulus evolution and neural encoding are modeled as:

$$
\begin{aligned}
\mathbf{x}_k &= \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k, & \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k) \\
\mathbf{z}_k &= \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, & \mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R}_k)
\end{aligned}
$$

Where:
- $\mathbf{x}_k$ is the latent stimulus state at time $k$.
- $\mathbf{z}_k$ is the observed neural activity (currents or spikes).
- $\mathbf{F}_k$ represents the stimulus dynamics (e.g., diffusion, drift).
- $\mathbf{H}_k$ is the observation matrix (the innervation weights).
- $\mathbf{w}_k, \mathbf{v}_k$ are process and measurement noise.

**Why Bayesian methods?**
1.  **Priors**: Unlike static inversion (Pseudoinverse), Bayesian decoders incorporate *prior knowledge* about stimulus physics (smoothness, continuity) via $\mathbf{F}_k$ and $\mathbf{Q}_k$.
2.  **Dynamics**: They explicitly model time, allowing the decoder to "predict" the stimulus state even during sparse spiking events.
3.  **Uncertainty**: They provide a measure of confidence (covariance $\mathbf{P}_k$), essential for sensor fusion and decision making.

#### Phase 1: Analytical inversion of innervation (proof-of-concept)

- **Target**: Reconstruct the *filtered stimulus* before any spiking or additional filtering. This phase ignores spike generation, mechanoreceptor double-Gaussian responses, and SA/RA temporal filtering. We work directly with the filtered stimulus input to sensory neurons.
- **Method**: Using *known* innervation maps (RF structure):
  1. **Forward model**: Apply innervation matrix to filtered stimulus → neuron responses.
  2. **Inverse model**: Apply pseudo-inverse of innervation matrix to neuron responses → reconstructed filtered stimulus.
  3. **Validation**: Compare reconstructed vs. original filtered stimulus; measure reconstruction error.
- **Success criteria**: Establish proof-of-concept that RF-based spatial coding enables invertible stimulus representation. Determine candidate population sizes and innervation profiles (RF σ, mean connections, neuron density) for subsequent phases. This validates the sufficiency of the labeled-line architecture for spatial encoding.

#### Phase 2: GLM-based spike-to-input reconstruction

- **Method**: Train a Generalized Linear Model (GLM) to predict sensory neuron input directly from spike trains, bypassing explicit filter inversion. GLM can capture nonlinearities and noise correlations missed by linear analytical methods.
- **Advantage**: More robust to parameter mismatch and biological variability; learns optimal decoding weights from data.

#### Phase 3: Advanced reconstruction enhancements

Further improvements to reconstruction fidelity and biological realism:

- **Spatial filters**: Convolutional layers to exploit spatial correlations in the stimulus.
- **Decompression algorithms**: Learned upsampling or super-resolution to recover fine details from sparse spike patterns.
- **Neuroscience-inspired mechanisms**:
  - **Lateral inhibition**: Sharpen spatial contrast by suppressing neighbors.
  - **Winner-takes-all**: Sparse coding via competitive selection of most active neurons.
  - **On-off cells**: Separate pathways for stimulus increases vs. decreases (analogous to retinal ON/OFF ganglion cells).
  - **Adaptation**: Model long-term gain control and dynamic range adjustment.

#### Phase 4: Real-world validation

- **Simulation**: Test reconstruction pipeline on realistic, noisy sensor data and temporal stimuli.
- **Neuromorphic hardware**: Deploy encoding/decoding on spiking neural network hardware (e.g., Intel Loihi, SpiNNaker, Akida) to demonstrate real-time, event-driven operation with actual spike generation and energy measurements.

---

## Key Scientific Insights and Predictions

### 1. Sufficiency of SA/FA Coding

**Prediction**: SA and FA pathways together provide a *complete basis* for stimulus reconstruction. SA captures the "what" (magnitude, static shape), and FA captures the "when" and "how fast" (dynamics, edges). Loss of either pathway degrades but does not eliminate reconstruction—consistent with psychophysical studies showing complementary roles.

**Test**: Compare reconstruction fidelity using:

- SA-only populations (expect poor edge/transient recovery).
- RA-only populations (expect poor sustained intensity tracking).
- Combined SA+RA populations (expect high-fidelity reconstruction).

### 2. Receptive Field Structure Determines Spatial Encoding

**Prediction**: Reconstruction fidelity depends on RF parameters (σ, mean connections, neuron density) and their match to stimulus spatial statistics. The labeled line (innervation architecture) defines the spatial encoding capacity of the system.

**Test**: Systematically vary RF parameters and measure reconstruction error. Phase 1 analytical inversion will establish baseline performance and identify candidate architectures for subsequent phases.

### 3. Temporal Sparsity from Event-Based Encoding

**Prediction**: Spike-based encoding provides temporal sparsity—spikes occur primarily during stimulus changes (RA pathway) and sustained contact (SA pathway). Data throughput scales with stimulus dynamics rather than fixed sampling rate.

**Test**: Measure spike counts and reconstruction fidelity across stimulus types (static, step, dynamic). Quantify sparsity and efficiency gains compared to traditional sampling methods.

### 4. Generalization to Other Modalities and Sensor Fusion

**Claim**: The SA/FA decomposition is *not* specific to touch—it reflects a general principle of sensory coding:

- **Vision**: Magnocellular (transient, motion) vs. parvocellular (sustained, color/form) pathways.
- **Audition**: Onset vs. sustained sound responses; temporal envelope vs. fine structure.
- **Proprioception**: Muscle spindle Ia (velocity-sensitive) vs. II (length-sensitive) fibers.

**Generalized sensor arrays**: This framework is not strictly biological and can be extended to:

- **Non-human/non-biological modalities**: Any sensor array (thermal, chemical, acoustic, electromagnetic) can be encoded using SA/FA principles.
- **Pre-extracted features**: Sensors that already perform feature extraction (RGB channels, edge detectors, frequency analyzers, odorant receptors, speed/acceleration sensors) can be directly mapped onto labeled lines. The encoding module inherently performs *feature integration and abstraction*, creating a compressed, abstract representation of the stimulus.

**Multi-modal sensor fusion**: Multi-modal sensor arrays (e.g., vision + tactile, pressure + temperature, RGB + depth) can share the same encoding framework. Each modality gets SA/RA pathways; receptive fields define cross-modal integration zones; decoding fuses information from all pathways.

**Hierarchical abstraction and convergence/divergence**:

If this encoding scheme successfully reconstructs stimuli, it implies the module performs a form of *feature extraction and integration*. This enables **hierarchical building blocks**:

- **Convergence**: Multiple lower-level feature detectors (e.g., oriented edges, color channels) feed into higher-level neurons with larger, composite RFs.
- **Divergence**: A single sensory neuron's output can fan out to multiple downstream targets. Divergence enables:
  - **Feature re-extraction**: Different downstream pathways extract complementary features from the same input.
  - **Computational operations**: Non-linear transformations, gain control, normalization.
  - **Temporal dynamics**: Delays, integration windows, adaptation mechanisms.
  - **Reconstruction and decoding**: Multiple readout pathways for different behavioral or computational goals.
  - **Code transformation**: Shuffling, reformatting, or recombining information for downstream processing.
- **Hierarchical stacking**: Stack encoding modules to build progressively more abstract representations (analogous to cortical hierarchy: V1 edges → V2 shapes → V4 objects). Each layer maintains *invertibility*—higher layers can be decoded back through intermediate representations to the original sensor input, enabling probing and interpretation at any abstraction level.

**Test**: Implement vision encoding (2D intensity → SA pathway, optical flow → RA pathway) and demonstrate reconstruction from combined tactile + visual spike trains. Extend to a 3-layer hierarchy (raw sensors → first-order features → second-order objects) with end-to-end reconstruction.

---

## Relationship to Existing Neuroscience Literature

This project synthesizes insights from:

1. **Receptive field theory** (Hubel & Wiesel, Mountcastle): Spatial labeled lines as computational units.
2. **Efficient coding hypothesis** (Barlow, Simoncelli): Sensory systems optimize information transmission under constraints (bandwidth, noise, metabolic cost). SA/FA decomposition maximizes information per spike by decorrelating temporal features.
3. **Predictive coding** (Rao & Ballard): RA neurons can be interpreted as encoding *prediction errors*—deviations from expected (static) input. SA neurons encode the prediction itself.
4. **Neuromorphic engineering** (Mead, Indiveri): Event-based sensors (DVS cameras) and spiking neural networks implement similar principles. This project bridges neuroscience theory and neuromorphic hardware design.

---

## Engineering and Scientific Impact

### Immediate Contributions

1. **Validate sufficiency hypothesis**: Demonstrate that SA/FA coding enables near-perfect reconstruction of spatiotemporal stimuli, given known encoding parameters.
2. **Quantify efficiency**: Benchmark spike-based encoding vs. traditional sampling (bits/s, compression ratios, reconstruction error vs. data rate).
3. **Design principles**: Derive guidelines for RF structure, population size, and filter parameters to achieve target spatial/temporal resolution and robustness.

### Broader Vision

1. **Neuromorphic sensor networks**: Deploy event-based tactile/visual sensors with embedded spike encoding for low-power, high-bandwidth applications (robotics, prosthetics, IoT).
2. **Sensory integration**: Extend to multi-modal fusion—e.g., robotic skin combining pressure, temperature, and strain; autonomous vehicles fusing vision, LiDAR, and tactile feedback.
3. **Sensory abstraction**: Build hierarchical architectures where downstream layers learn feature detectors (texture, shape, motion patterns) directly from spike trains, analogous to cortical processing.
4. **Biological insights**: Use the model to generate predictions for neurophysiology experiments—e.g., how should RF structure adapt to optimize reconstruction in different tasks?

---

## Implementation Status (October 2025)

- ✅ **Encoding pipeline**: PyTorch-based, GPU-accelerated, with SA/RA filtering and Izhikevich spiking neurons. Validated against theoretical filter responses.
- ✅ **Analytical decoding**: Pseudo-inverse reconstruction using known innervation maps. Demonstrated for Gaussian and point stimuli to sensory neuron input level.
- ✅ **GUI & registry**: Interactive tools for exploring parameter space and persisting protocol/run/STA data.
- ✅ **Automated tests**: `tests/test_pytorch_pipeline.py`, `tests/test_enhanced_filters.py`, and `tests/test_pytorch_neurons.py` exercise the active pipeline, filter implementations, and spiking neuron modules end-to-end.
- 🔄 **In progress**:
  - Establish minimal population sizes and innervation profiles for reconstruction.
  - Systematic parameter sweeps to map reconstruction fidelity vs. RF density, filter parameters, and noise levels.
- 🔜 **Next steps**:
  - GLM-based spike-to-input reconstruction.
  - Spatial filters and neuroscience-inspired enhancements (lateral inhibition, on-off cells, adaptation).
  - Multi-modal extension (vision + tactile fusion).
  - Real sensor validation using simulation or neuromorphic hardware (Loihi, SpiNNaker, Akida).
  - Hierarchical architecture with multi-layer abstraction and invertibility.

---

## References to Supporting Documentation

- **Architecture details**: `docs_root/README.md`, `docs_root/PYTORCH_README.md`
- **Component relevancy**: `docs_root/COMPONENT_RELEVANCY.md` (module map, testing guidelines)
- **Filter equations**: `docs_archive/Parvizi_Fard_Implementation_Guide.md`, `encoding/filters_torch.py` docstrings
- **STA & inversion**: `docs_root/STA_pipeline_plan.md`, `decoding/README.md`
- **Neuron models**: `neurons/README.md`, `docs_root/PyTorch_Neuron_Implementation_Guide.md`

---

## Usage by AI Agents and Contributors

**This document defines the scientific core of the project and serves as the highest-priority reference.** All architectural decisions, feature implementations, and experimental designs should align with the central hypothesis and phased roadmap outlined above.

When implementing features, debugging, or proposing changes:

1. **Anchor decisions to the hypothesis**: Does this change improve our ability to test sufficiency of SA/FA coding? Does it clarify the role of labeled lines vs. dynamic attributes? Does it advance toward reconstruction, generalization, or hierarchical abstraction?
2. **Respect biological plausibility as a starting point**: Current default configurations use biologically-inspired filter dynamics, RF structures, and neuron parameters derived from neurophysiology. However, the framework is *not strictly biological*—we may explore parameter regimes beyond biological constraints (e.g., faster membrane time constants, extended dynamic ranges) to achieve improved performance. Non-biological sensor modalities, abstract features, and hierarchical stacking are explicitly within scope.
3. **Prioritize reconstruction phases**: Focus first on analytical inversion (Phase 1), then GLM-based methods (Phase 2), before adding advanced enhancements (Phase 3). Real-world validation (Phase 4) is the ultimate goal.
4. **Document extensions and generalizations**: When adapting the pipeline to new modalities, multi-modal fusion, or hierarchical layers, update this document and link to relevant implementation guides.
5. **Prioritize reconstruction fidelity**: The end-to-end pipeline's success is measured by stimulus reconstruction error. Intermediate metrics (spike counts, filter accuracy) serve this goal.
6. **Maintain modularity**: Encoding, decoding, and neuron models should remain separable to enable systematic ablation studies (e.g., testing SA-only or RA-only pathways).

When in doubt about design trade-offs, return to the central hypothesis and ask: *"Does this help prove or refine our understanding of how SA/FA coding enables reconstruction?"*

---

**Document Authorship:**  
Ben Efron (Principal Investigator)  
GitHub Copilot (Documentation Assistant)  
Created: October 6, 2025  
Last Updated: October 6, 2025
