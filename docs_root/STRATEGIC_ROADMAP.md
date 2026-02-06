# Strategic Roadmap: Sensory Compression & Hierarchical Abstraction

**Date:** January 30, 2026  
**Status:** Active planning phase following colleague meeting  
**Last Updated:** January 30, 2026

---

## Executive Summary

**Core Vision:** Build modular, extensible building blocks for sensory information processing that compress and abstract sensory streams through learned and structured priors. The framework is **modality-agnostic**—touch is the initial testbed, but the architecture generalizes to vision, audition, and multi-modal fusion.

**Current State:** Functional sensory encoding playground with PyTorch pipeline, GUI tools, and multiple neuron/filter models. Originally developed for tactile simulation, but designed for generalization.

**Immediate Goals:** 
1. **Paper 1:** Release extensible framework as standalone package (new repo, catchy name)
2. **Paper 2:** Event-based Kalman filter for low-dimensional control and active sensing
3. **Paper 3:** Scalable reconstruction methods with edge/analog computing considerations

---

## I. The Larger Vision: Sensory Compression & Abstraction

### Core Hypothesis

Efficient sensory processing emerges from:
1. **Compression** via receptive fields (many inputs → fewer outputs)
2. **Feature extraction** at multiple architectural stages (receptors, receptive fields, filters, priors, latent spaces)
3. **Dual-pathway dynamics** (sustained vs. bursting, rate vs. timing, monitoring vs. novelty detection)
4. **Abstraction** as a byproduct of compression + structured priors
5. **Adaptability** via tunable parameters, learned priors, and attention
6. **Bayesian framework** enabling incorporation of physics-aware models, cognitive biases, and use-case-specific priors

**Feature extraction is distributed across architectural stages:**
- **Receptor distribution:** Color, contrast, change detection, temporal dynamics, frequency sensitivity, motion, orientation
- **Receptive fields:** Spatial pooling, feature selectivity, coverage properties
- **Temporal filtering:** Slope, magnitude, change rate, frequency decomposition, sustained vs. transient
- **Spiking dynamics:** Rate coding, spike timing, burst patterns
- **Priors and interpretation:** Task-relevant feature emphasis
- **Latent space design:** Learned or designed feature representations
- **Parallel pathways:** Simultaneous extraction of complementary features

**SA/RA as a universal dual-pathway concept** (not touch-specific):
- **Sustained vs. Bursting:** Continuous monitoring vs. change detection
- **Rate coding vs. Spike timing:** Population averaging vs. temporal precision
- **Attention:** Ongoing monitoring vs. novelty-triggered focus
- **Modality translations:**
  - Touch: SA (pressure amplitude) / RA (change, slip, texture)
  - Vision: ON/OFF (intensity increase/decrease), sustained/transient
  - Audition: Frequency-specific receptive fields (cochlear decomposition) + sustained/transient activation/intensity
  - General: Any modality can extract dual channels for amplitude vs. change

Receptive fields perform dimensionality reduction while extracting features. Combined with structured priors (spatial smoothness, temporal dynamics), this produces abstracted representations suitable for downstream tasks.

### On Reconstruction

Reconstruction serves as a **validation tool**: if we can reconstruct the original signal, information is preserved during compression. The reconstruction capability also:
- Enables analysis and interpretation of encoded representations
- Supports learning objectives (decoder training)
- Allows "zooming in" to details when needed (though typically we operate on compressed/abstracted representations)

### System Architecture

```
Raw Sensory Input (high-dim, dense)
  ↓ [Feature Extraction Stage 1: Receptor distribution]
Receptor Array (spatial arrangement, modality-specific properties)
  ↓ [Feature Extraction Stage 2: Receptive field pooling]
Receptive Field Layer: Compression + Spatial Features
  ↓ [Feature Extraction Stage 3: Temporal filtering]
  ┌─────────┴─────────┐
  ↓                   ↓
Sustained (SA-like)   Transient (RA-like)
(amplitude,           (change, novelty,
 monitoring)           attention trigger)
  ↓                   ↓         [Feature Extraction Stage 4: Spiking dynamics]
Spiking: Event-based, sparse representation
  ↓
Task-Specific Processing (control, classification, reconstruction, etc.)
  ↓                           [Feature Extraction Stage 5: Priors/latent space]
Task Output / State Estimate

Reconstruction (offline, for validation/learning):
Latent State → Expansion → Original Signal Estimate

Key: Feature extraction occurs at EVERY stage.
```

### Neuroscience-Inspired Extensions

Because the framework is grounded in neuroscience principles, we can leverage the rich knowledge base of systems, theoretical, and computational neuroscience to extend capabilities:

- **Divergence/Convergence:** Parallel streams (one signal → multiple processing pathways) and pooling operations (many → few with mathematical operations)
- **ON/OFF pathways:** Separate channels for increases/decreases (efficient coding, edge enhancement)
- **Dendritic computation:** Feature extraction in receptive field structure
- **Spike timing vs. rate coding:** Temporal precision vs. population averaging
- **Frequency response:** Single-cell or population-level oscillatory dynamics
- **Energy efficiency:** Sparse coding, minimal spike counts while preserving information
- **Lateral inhibition, normalization, adaptation:** Standard neuroscience mechanisms applicable when beneficial

These are **extensions**, not core requirements. The base framework is sensor agnostic.

### Flexibility Through Modular Design

**Core principle:** The framework is flexible and extensible, but **specific use cases require design choices** tailored to the system:

- **Sensors:** Mechanoreceptors, event cameras (DVS), microphones, pressure arrays, any sensor grid
- **Filters:** SA/RA (touch), ON/OFF (vision), bandpass (audio), custom temporal dynamics
- **Priors:** Spatial smoothness, temporal coherence, spectral structure, physics-aware constraints, learned priors
- **Innervation:** Gaussian overlap, distance-weighted, topographic maps, spatial Poisson process distributions, constrained variance (one-to-one or uniform), non-grid arrangements
- **Neurons:** Izhikevich, AdEx, MQIF (already implemented), rate-based, custom models
- **Feature extraction:** Configurable per use case—what features to transmit from sensors

**Why Bayesian methods:** Flexibility to incorporate priors at all levels—physics-aware models, cognitive biases, predictive coding, fading memory. Supports both designed systems and learned components (dynamics, forward models, features). Enables online optimization and calibration.

**Target modalities for demonstration:**
- Touch (current): pressure grid → SA/RA neurons
- Vision: event cameras (DVS) → ON/OFF cells (connects to lab projects)
- Multi-modal: audio-tactile fusion (simple example, connects to previous PhD work)

**Hardware integration:** Framework designed to interface with custom neuromorphic hardware (e.g., lab's RA/SA hardware implementation).

---

## II. Current Project State

### What We Have (Functional)

| Component | Status | Location |
|-----------|--------|----------|
| Receptor grid generator | ✅ Complete | `encoding/grid_torch.py` |
| Stimulus synthesis | ✅ Complete | `encoding/stimulus_torch.py` |
| Innervation (labeled-line compression) | ✅ Complete | `encoding/innervation_torch.py` |
| SA/RA filters (Parvizi-Fard) | ✅ Complete | `encoding/filters_torch.py` |
| Multiple neuron models | ✅ Complete | `neurons/` (Izhikevich, AdEx, MQIF, FA) |
| Noise models | ✅ Complete | `encoding/noise_torch.py` |
| Encoding pipeline orchestration | ✅ Complete | `encoding/pipeline_torch.py` |
| Project registry (data persistence) | ✅ Complete | `utils/project_registry.py` |
| GUI (5 tabs: grid, stimulus, neurons, protocols, STA) | ✅ Complete | `GUIs/` |
| Pseudoinverse decoder (static baseline) | ✅ Complete | `decoding/pipeline.py` |
| STA analysis backend | ✅ Complete | `decoding/modules/sta_*.py` |
| Metrics & visualization | ✅ Complete | `metrics/`, `encoding/visualization_torch.py` |

### What We're Developing

| Component | Status | Location |
|-----------|--------|----------|
| Event-based Kalman filter (AM-KF) | 🔄 Design complete, awaiting implementation | `docs/concepts/bayesian_decoding.md` |
| Geodesic Q interpolation | 🔄 Math documented, code pending | `docs_root/COLLEAGUE_FEEDBACK_INTEGRATION.md` |
| Coupled noise dynamics | 🔄 Design phase | `docs/concepts/bayesian_decoding.md` |
| Local/sparse KF updates | 🔄 Concept validated, implementation needed | — |

### What We Need (For Publications)

#### For Paper 1 (Methods/Framework)
- [ ] Brian2 integration for ODE solving (optional enhancement)
- [ ] Event-based camera simulation on receptor grid
- [ ] Extended documentation (usage tutorials, developer guide)
- [ ] Benchmark suite (performance, scalability)
- [ ] Example workflows (notebooks + scripts)

#### For Paper 2 (Event-based KF)
- [ ] Full AM-KF implementation with local updates
- [ ] Validation on low-dimensional test cases (1D, 2D small grids)
- [ ] Comparative analysis (vs standard KF, vs particle filter)
- [ ] Sleep/wake/reset logic implementation
- [ ] Computational cost profiling

#### For Paper 3 (Alternative Reconstruction)
- [ ] Diffusion model decoder (collaboration needed)
- [ ] Fading memory model (collaboration needed)
- [ ] Low-dim latent dynamics + expansion (collaboration needed)
- [ ] Comparative framework (benchmark all methods)

---

## III. Three Publication Tracks

### Paper 1: Extensible Sensory Encoding Framework

**Working Name Ideas:** 
- "SensoryForge" / "NeuroForge"
- "SpikeFlow" / "SenseFlow"
- "Receptor" / "ReceptorKit"
- (needs catchy, memorable name for new repo)

**Title (working):** "SensoryForge: A Modular, Extensible Framework for Simulating Sensory Encoding Across Modalities"

**Target Audience:** Computational neuroscience, neuroengineering, ML for robotics, neuromorphic computing

**Core Contribution:** Open-source, GPU-accelerated, highly extensible framework for exploring sensory encoding schemes. Touch is the first modality, but architecture is modality-agnostic.

**Key Technical Goals:**

1. **New GitHub Repository**
   - Clean separation from current development repo
   - Professional packaging (pip installable)
   - Clear documentation and tutorials
   - Example notebooks for each modality

2. **Extended Stimulus Options**
   - Multiple simultaneous stimuli (parallel contacts)
   - Texture generation (regular patterns, random textures)
   - Moving textures (sliding contact, exploration)
   - Temporal sequences (tap, slide, vibration)
   - Custom stimulus injection API

3. **Extensibility Architecture**
   - Base classes / templates for each component type:
     - `BaseFilter` → extend for new temporal dynamics
     - `BaseNeuron` → extend for new spiking models
     - `BaseStimulus` → extend for new stimulus types
     - `BaseInnervation` → extend for new connectivity patterns
   - Plugin discovery: drop new class in folder, auto-registers
   - Configuration-driven: YAML/JSON defines which components to use
   - Intermediate interfaces for common operations

4. **Brian2 Integration**
   - Link functions: PyTorch tensors ↔ Brian2 arrays
   - Leverage Brian2's C++ code generation for speed
   - Access Brian2's neuron/synapse/plasticity library
   - Optional: run filters as Brian2 equations for consistency
   - Keep PyTorch as core (GPU, learning, gradients)

5. **Grid and Innervation Extensions**
   - Non-grid spatial arrangements (Poisson process, hexagonal, irregular)
   - Distance-weighted innervation (constraint on receptive field shapes)
   - Connection variance control (one-to-one, uniform distribution, sparse/dense)
   - Configurable receptive field constraints per use case

6. **ON/OFF Pathway Support**
   - Separate increase/decrease channels as first-class extension
   - Template for dual-pathway architectures

7. **Multi-Modality Demonstration**
   - Touch: SA/RA implementation (current)
   - Vision: Event cameras (DVS) with ON/OFF cells
   - Multi-modal: Audio-tactile (if feasible, simple example)
   - Show same framework, different design choices per modality

**Architecture Sketch:**

```
sensoryforge/
├── core/
│   ├── grid.py           # Spatial substrate (PyTorch)
│   ├── innervation.py    # Receptive field generation
│   └── pipeline.py       # Orchestration
├── filters/
│   ├── base.py           # BaseFilter template
│   ├── sa_ra.py          # SA/RA implementation
│   ├── center_surround.py # ON/OFF for vision
│   └── custom/           # User extensions (auto-discovered)
├── neurons/
│   ├── base.py           # BaseNeuron template
│   ├── izhikevich.py     # Already implemented
│   ├── adex.py           # Already implemented
│   ├── mqif.py           # Already implemented
│   └── custom/
├── stimuli/
│   ├── base.py           # BaseStimulus template
│   ├── gaussian.py
│   ├── texture.py
│   ├── moving.py
│   └── custom/
├── brian_bridge/
│   ├── converters.py     # Tensor ↔ Brian2 arrays
│   ├── neuron_groups.py  # Wrap Brian2 NeuronGroups
│   └── network.py        # Brian2 Network integration
├── gui/                  # PyQt interface
├── config/               # YAML schemas
└── examples/
    ├── touch_demo.ipynb
    ├── vision_demo.ipynb
    └── custom_filter_tutorial.ipynb
```

**Narrative Arc:**
- Problem: Existing sensory simulations are either modality-specific or not extensible
- Solution: Modular toolkit with clear extension points and multi-modality support
- Validation: Reproduce SA/RA responses, demonstrate ON/OFF vision encoding
- Extension: Show how to add new components (tutorial)
- Impact: Enable rapid prototyping for neuromorphic sensing research

**Target Venues:**
- *Journal of Open Source Software* (JOSS) - fast, good for tools
- *Journal of Neural Engineering*
- *Frontiers in Neurorobotics*

**Timeline:** 4-5 months
- Month 1: Refactor into extensible architecture, create templates
- Month 2: Brian2 integration, extended stimuli
- Month 3: Vision modality example, documentation
- Month 4: Tutorial notebooks, benchmark suite
- Month 5: Write manuscript, prepare release

**Dependencies:** None (can proceed immediately)

---

### Paper 2: Attention-Modulated Kalman Filter with Sparse Localized Updates

**Title (working):** "Attention-Modulated Kalman Filtering for Event-Driven Sparse Sensory Reconstruction"

**Target Audience:** Control systems, robotics, computational neuroscience, neuromorphic engineering, event-based sensing

**Core Contribution:** An event-based Kalman filter that uses **transient neural activity (RA-like signals) as a spatial attention mechanism** to localize computation and modulate trust in observations and dynamics. The framework is **grid-size independent**: computational cost scales with active region size, not total sensor count.

**Alignment with Event_Based_Kalman_Filter.tex specification:**

See `written_outcomes/Event_Based_Kalman_Filter.tex` for full mathematical specification and `written_outcomes/benelux_abstract/New version/intial_abstract.tex` for conceptual framing.

**Core Innovation: RA as Spatial Attention for Local Q/R Modulation**

The key innovation is using RA (transient/change-detection) neurons to:
1. **Localize where computation happens** (sparse updates, grid-size independent)
2. **Modulate Q locally** (process noise): higher Q where RA fires → allow rapid state changes
3. **Modulate R locally** (measurement noise): lower R for RA, adjust for SA → change trust in observations
4. **Trigger updates only where needed** (event-driven efficiency)

This enables a single Kalman filter to operate in two regimes:
- **Slow mode** (RA inactive): SA monitoring, high stability, low Q, smooth tracking
- **Fast mode** (RA active): RA localization, high responsiveness, elevated Q, trust change detection

**The "Low-Dimensional Control" Framing:**

By "low-dimensional control" we mean:
1. **Choose systems with inherently low dimensions** (pendulum, small grids 10×10 or 20×20, simple manipulation tasks)
2. **Full state estimation on these small systems** with tractable covariance

This is NOT about extracting low-dimensional latent states from high-dimensional grids (that's a future extension, different objective). For this paper, we demonstrate the attention-modulated KF on systems where the full state space is tractable.

**For larger grids:** Solve locally on small patches where RA indicates activity (sparse localized updates, primary design goal per Event_Based_Kalman_Filter.tex).

**Scientific Hypotheses:**

**H1 (Attention-Modulated Trust - PRIMARY INNOVATION):** Locally modulating Q and R based on RA activity enables a linear Kalman filter to adaptively trust different observation sources (SA vs. RA) and adjust confidence in dynamics spatially and temporally. This attention mechanism improves reconstruction accuracy and computational efficiency.

**H2 (Event-Driven Sparse Updates):** Restricting computation to regions where RA neurons fire (sparse localized updates) enables grid-size independence. Computational cost scales with active area, not total grid size, enabling efficient processing of large sensor arrays.

**H3 (SA/RA Complementarity):** SA neurons provide continuous amplitude monitoring (stable baseline). RA neurons detect change and trigger localized high-gain updates. Combining both populations minimizes spike count while maintaining reconstruction quality across diverse stimulus regimes.

**Key Features (from Event_Based_Kalman_Filter.tex):**
1. **Spatial activity mask** derived from RA receptive fields (Section 4.2)
2. **Local Q/R modulation** via geodesic interpolation on SPD manifolds (Section 4.3)
3. **Sign inference** for RA observations using SA temporal trends (Section 5.1)
4. **Sparse localized updates** (target architecture, grid-size independent) (Section 5.2)
5. **Dual-regime operation:** Slow mode (SA-dominant) / Fast mode (RA-active)
6. **Full covariance tracking** for uncertainty quantification
7. **Temporal decay** of RA attention mask (persistent but fading)

**Validation Strategy:**
- **Systems:** Low-dimensional grids (10×10, 20×20), pendulum, simple manipulation
- **Stimuli:** Moving contacts, sliding textures, vibration, multi-contact
- **Metrics:** 
  - Reconstruction error (MSE, SSIM) vs. ground truth
  - Computational cost (FLOPs, wall-clock time) vs. grid size
  - Sparse update efficiency (active region fraction)
  - Uncertainty calibration (predicted vs. actual error)
- **Baselines:** 
  - Standard KF (no attention, full grid always)
  - Pseudoinverse reconstruction (non-Bayesian)
  - Thresholding heuristics (hard cutoffs)
- **Ablations:** 
  - No Q modulation (constant process noise)
  - No R modulation (constant measurement trust)
  - No spatial masking (full grid updates)
  - No temporal decay (instantaneous attention)
  - SA-only (disable RA, verify stable amplitude estimation)
- **Alignment:** Verify implementation matches `Event_Based_Kalman_Filter.tex` specification (state model, observation model, gated covariance, sign inference, sparse updates)

**Narrative Arc:**
- **Problem:** Real-time sensory processing on large grids is computationally expensive
- **Biological insight:** Transient neurons (RA) signal *where* and *when* attention is needed
- **Solution:** Use RA as spatial attention to localize KF updates and modulate Q/R locally
- **Key result:** Computational cost scales with active area, not total grid size
- **Validation:** Demonstrate on small systems and local patches of larger grids
- **Extensions:** Active sensing, control integration, multi-modal fusion

**Target Venues:**
- *IEEE Transactions on Robotics*
- *Autonomous Robots*
- *Conference on Robot Learning (CoRL)*
- *IEEE/RSJ IROS*

**Timeline:** 4-6 months
- Month 1: Implement core AM-KF with control state space
- Month 2: Active sensing integration, sleep/wake logic
- Month 3: Control task benchmarks (simulated)
- Month 4: Comparative analysis
- Month 5-6: Write manuscript

**Collaborators Needed:**
- Robotics/control expert (optional, for real-world grounding)
- Access to robotic hand simulator (optional)

---

### Paper 3: Scalable Reconstruction & Edge Computing

**Title (working):** "Efficient High-Dimensional Sensory Reconstruction: From Diffusion Models to Analog Computing"

**Target Audience:** ML, neuromorphic engineering, edge computing, computational neuroscience

**Core Contribution:** Comparative study of reconstruction methods with emphasis on **computational constraints**: edge deployment, analog feasibility, real-time requirements.

**Key Framing Shift:** Not just "which method is most accurate" but "which methods are feasible under different computational constraints?"

**Methods to Compare:**

| Method | Learning | Inference | Hardware | Collaborator |
|--------|----------|-----------|----------|--------------|
| **Pseudoinverse** | None | <1ms | Any | None (baseline) |
| **AM-KF** (low-dim/local) | None | <1ms | Any | None (Paper 2) |
| **Latent Encoder + Decoder** | Offline | ~10ms | GPU/Edge | You |
| **Diffusion Models** | Offline (slow) | Real-time? | GPU | **Yes** |
| **Neural Network Decoder** | Offline | <10ms | GPU/Edge | Possible |
| **Fading Memory** | Minimal | <1ms | **Analog** | **Yes** |

**Key Questions:**
- **Diffusion models:** Are they real-time feasible for inference after training? Or too slow?
- **Alternative ML:** Could simpler neural networks (CNN, RNN) provide recursive Bayesian estimation after learning, enabling generalized reconstruction independent of grid size with low online computation?
- **Two-phase approach:** Offline learning (diffusion, VAE, other) + online lightweight inference (linear decoder, fading memory)

**Edge Computing Considerations:**

For real-world deployment (prosthetics, robots, wearables), we need:
- Low power consumption
- Low latency (<10ms for control)
- Small memory footprint
- Potentially analog implementation

**Fading memory** is particularly interesting here:
- Can be implemented in analog circuits
- Natural fit for streaming data
- Biologically plausible
- Limited accuracy but very efficient

**Analog Feasibility Analysis:**
- Which operations can be done in analog? (weighted sums, thresholding, leaky integration)
- Which require digital? (matrix inversion, iteration)
- Hybrid architectures: analog front-end, digital refinement

**Latent Dynamics Approach (Your Lead)**

This aligns with your vision of compression + priors yielding abstraction:

1. **Compression already happened:** Spikes are the latent representation
2. **Learn to decode latent state:** $\mathbf{s}_k = f(\text{spikes}_k)$ where $\mathbf{s}$ is low-dim
3. **Optional: run dynamics in latent space** (KF on $\mathbf{s}$)
4. **Expand for validation:** Decoder $g(\mathbf{s}_k) \approx \text{stimulus}_k$

The latent state $\mathbf{s}$ IS the abstraction. We don't "extract features then abstract"—the compression + learned decoder directly produces abstracted representations.

**Multi-Modality Requirement:**

To demonstrate generality, Paper 3 should show reconstruction on:
- Touch (primary)
- Vision with event cameras (DVS)
- (Optional) Audio-tactile if simple example available

This proves the framework isn't modality-specific.

**Validation Strategy:**
- Reconstruction accuracy (MSE, SSIM)
- Computational cost (FLOPs, latency, power)
- Hardware feasibility analysis
- Scaling behavior (how does cost grow with dimension?)
- Failure modes (where does each method break?)

**Target Venues:**
- *NeurIPS* / *ICML* (ML audience)
- *IEEE Journal on Emerging and Selected Topics in Circuits and Systems* (edge computing)
- *Frontiers in Neuroscience* (computational neuroscience)

**Timeline:** 6-9 months (depends on collaborations)
- Months 1-2: Collaborator recruitment, literature review, ML method assessment
- Months 3-4: Baseline implementations (diffusion or alternative ML, fading memory)
- Months 5-6: Latent dynamics development, multi-modality examples
- Months 7-8: Comparative experiments, hardware feasibility analysis
- Month 9: Write manuscript

**Dependencies:**
- ML expert for reconstruction methods
- Fading memory / analog computing expert
- Training data generation pipeline
- **Computational resources:** HPC with GPU nodes (available)

**Collaborators Needed:**
- **Critical:** ML/generative models expert (diffusion or alternatives)
- **Critical:** Fading memory / analog computing expert
- **Helpful:** Neuromorphic hardware expert (possible lab connection)

**Validation Strategy:**
- Reconstruction accuracy (MSE, SSIM)
- Computational cost (FLOPs, latency, power)
- Hardware feasibility analysis
- Scaling behavior (how does cost grow with dimension?)
- Failure modes (where does each method break?)

**Target Venues:**
- *NeurIPS* / *ICML* (ML audience)
- *IEEE Journal on Emerging and Selected Topics in Circuits and Systems* (edge computing)
- *Frontiers in Neuroscience* (computational neuroscience)

**Timeline:** 6-9 months
- Months 1-2: Collaborator recruitment, literature review
- Months 3-4: Baseline implementations (diffusion, fading memory)
- Months 5-6: Latent dynamics development, multi-modality
- Months 7-8: Comparative experiments, hardware analysis
- Month 9: Write manuscript

---

## IV. Connecting the Papers

### How They Build on Each Other

```
Paper 1 (Framework)
    ↓
Provides: Extensible encoding infrastructure, multi-modality
    ↓
Paper 2 (AM-KF for Control)        Paper 3 (Reconstruction Methods)
    ↓                                    ↓
Provides: Low-dim state estimation,     Provides: High-dim reconstruction,
          control integration,                    edge computing analysis,
          active sensing                          hardware feasibility
    ↓                                    ↓
    └──────────────┬─────────────────────┘
                   ↓
          Future: Complete System
          (sense → compress → control/reconstruct)
```

### The Abstraction is Built-In

Your insight is important: **we don't need a separate abstraction layer.**

- **Receptive fields** already perform dimensionality reduction
- **Temporal filters** impose priors (SA for steady-state, RA for change)
- **Compression** forces information to be encoded efficiently
- **Solving for latent state** (whether via KF or learned decoder) extracts the abstraction

The question isn't "how do we add abstraction?" but "what latent variables are implicitly encoded by the compression, and how do we read them out?"

### Divergence and Convergence in Practice

**Divergence Example:**
- Same spike stream → control pathway (AM-KF, Paper 2) + reconstruction pathway (Paper 3)
- Different downstream tasks, same encoding
- Parallel computation possible

**Convergence Example:**
- SA and RA streams → combined observation (fused in KF)
- Multiple receptors → single neuron (compression)
- Mathematical operation: weighted sum defined by innervation

---

## V. Technical Implementation Notes

### Paper 1: Extensibility Architecture

**Base Class Template Pattern:**

```
BaseFilter
├── __init__(config)      # Load parameters from config dict
├── forward(x)            # Core computation
├── reset()               # Clear internal state
├── to_device(device)     # Move to CPU/GPU
└── get_state_dict()      # For serialization

Concrete implementations override forward(), optionally others.
```

**Auto-Discovery System:**
- Each module folder has `custom/` subdirectory
- On import, scan for classes inheriting base class
- Register in global registry (dict mapping name → class)
- Config file specifies component by name

**Brian2 Bridge Design:**
- `tensor_to_brian(t: torch.Tensor) -> brian2.array`
- `brian_to_tensor(a: brian2.array, device) -> torch.Tensor`
- Wrapper class `BrianNeuronGroup` that implements `BaseNeuron` interface
- Optional: compile Brian2 to C++ for speed, then call from Python

### Paper 2: Control State Space

**Example low-dimensional state:**

$$\mathbf{x} = \begin{bmatrix} x_c \\ y_c \\ p \\ v_x \\ v_y \\ \dot{p} \end{bmatrix}$$

Where:
- $(x_c, y_c)$: contact centroid
- $p$: pressure magnitude
- $(v_x, v_y)$: lateral velocity (slip)
- $\dot{p}$: pressure rate of change

This is 6-dimensional—full covariance is $6 \times 6 = 36$ parameters. Trivially tractable.

**Observation model:**
Spikes → binned rates → nonlinear mapping to state (learned or analytical)

### Paper 3: Hardware Feasibility Matrix

| Operation | Digital | Analog | Hybrid |
|-----------|---------|--------|--------|
| Weighted sum | ✓ | ✓ (resistor network) | — |
| Thresholding | ✓ | ✓ (comparator) | — |
| Leaky integration | ✓ | ✓ (RC circuit) | — |
| Matrix inversion | ✓ | ✗ | Iterative |
| Nonlinear activation | ✓ | Limited | Lookup |
| Gradient descent | ✓ | ✗ | — |

Fading memory is attractive because it only needs weighted sums and leaky integration—fully analog-compatible.

---

## VI. Concrete Next Steps (Pre-Meeting Preparation)

**1. Clarify Collaboration Needs**

Prepare a 1-slide summary for each potential collaborator:

**For Diffusion Model Expert:**
- "We have spike trains from sensory neurons. Need to reconstruct 2D pressure fields."
- "Currently using Kalman filter (works but doesn't scale). Want to try diffusion models."
- "Data: Can generate unlimited training pairs (stimulus → spikes)"
- "Ask: Would you be interested in co-authoring? What's the scope?"

**For Fading Memory Expert:**
- "Online reconstruction problem: streaming spikes → real-time pressure estimate."
- "Fading memory could be lightweight alternative to Kalman filter."
- "Ask: Is there existing work on fading memory for inverse problems?"

**For Bayesian ML Researcher:**
- "High-dimensional state estimation from sparse observations."
- "Want to compare variational inference to our Kalman approach."
- "Ask: Would amortized VI be applicable here?"

**2. Define Authorship & Contribution Model**

Decide now to avoid issues later:

- **Paper 1 (Methods):** You lead, others can contribute specific modules (e.g., Brian2 integration)
- **Paper 2 (AM-KF):** You lead, optional co-author for theoretical analysis
- **Paper 3 (Reconstruction):** Joint leadership with ML collaborator(s)

**3. Create a Visual Roadmap**

Prepare a Gantt chart showing:
- 3 parallel tracks (Papers 1, 2, 3)
- Dependencies and milestones
- Where collaborations are needed
- Target submission dates

**4. Prioritize**

Based on your resources and interests:

**High Priority (Start Now):**
- Paper 1 (you can do this solo, builds foundation)
- AM-KF implementation for Paper 2 (critical for your thesis/project)

**Medium Priority (After collaborations secured):**
- Latent dynamics implementation (Paper 3)
- Comparative benchmarking

**Low Priority (Future work):**
- Multi-layer hierarchy
- Alternative modalities (vision, audition)
- Neuromorphic hardware deployment

---

## VII. Success Criteria

### Short-term (6 months)
- [ ] Paper 1 submitted or in revision
- [ ] AM-KF implemented and validated on low-dim cases
- [ ] At least one collaboration secured for Paper 3

### Medium-term (12 months)
- [ ] Paper 1 published
- [ ] Paper 2 submitted
- [ ] Paper 3 in active development with collaborators
- [ ] Framework adopted by at least 2 external groups (citations, GitHub stars)

### Long-term (18-24 months)
- [ ] All 3 papers published
- [ ] Multi-layer hierarchical system prototype working
- [ ] Demonstrated on at least 2 sensory modalities (touch + one other)
- [ ] External funding secured (grant proposal based on this work)
- [ ] Community forming around framework (workshop, tutorial sessions)

---

## VIII. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Collaborations fall through | Paper 3 delayed | Start with methods you can implement solo (latent dynamics) |
| AM-KF doesn't scale to full grid | Paper 2 scope reduced | Frame as proof-of-concept, emphasize principles over scale |
| Diffusion models outperform everything | AM-KF seems obsolete | Position AM-KF as interpretable/biological baseline; Paper 2 still valid |
| Framework too complex for adoption | Paper 1 impact limited | Invest in tutorials, simplified entry points, example notebooks |
| Computational resources insufficient | Papers 2 & 3 limited | Seek cloud credits, collaborate with groups that have GPUs |

---

## IX. Key Questions for Your Meeting

**Strategic:**
1. Which of the 3 papers should be highest priority?
2. Should we pursue conferences (NeurIPS, ICML) or journals first?
3. Is the latent dynamics approach worth pursuing, or focus on diffusion models?

**Collaborative:**
4. Who is interested in co-authoring which paper?
5. What resources can collaborators provide (compute, data, expertise)?
6. What's a realistic timeline for joint work?

**Technical:**
7. For high-dim reconstruction, is diffusion the state-of-art or are there other methods?
8. Can we use off-the-shelf diffusion models or need custom architecture?
9. Is there existing work on fading memory for inverse problems we should know about?

**Practical:**
10. Should we open-source everything immediately or after Paper 1 acceptance?
11. Do we need IRB approval for any future human data collection?
12. What conferences should we target for presenting preliminary work?

---

## X. Vision Statement (For Funding / Broader Impact)

*"We are developing modular building blocks for sensory information processing that achieve efficient compression while preserving the ability to reconstruct original signals when needed.*

*The key insight: abstraction emerges naturally from compression combined with structured priors. Receptive fields reduce dimensionality while extracting features. Temporal filters impose dynamics. The result is a sparse, event-driven representation that captures task-relevant information.*

*This framework is modality-agnostic—the same architecture processes touch, vision, or any sensor array by changing the parameters, not the structure. It's hardware-aware—we analyze feasibility for edge computing, analog circuits, and neuromorphic chips. And it's control-oriented—providing uncertainty-aware state estimates for closed-loop systems.*

*Applications span robotics (tactile sensing for manipulation), prosthetics (neural interface decoding), autonomous systems (efficient perception), and neuromorphic computing (brain-inspired hardware). The framework bridges neuroscience insights with engineering requirements.*

*We're building the sensory front-end for embodied AI."*

---

## XI. Recommended Immediate Actions

**This Week:**
1. Review `Event_Based_Kalman_Filter.tex` and `intial_abstract.tex` for framing alignment
2. Verify all cited papers in this roadmap are accurate
3. Create presentation slides for meeting
4. List specific asks for each potential collaborator
5. Brainstorm package name for Paper 1 (catchy, memorable, available on PyPI)

**Next 2 Weeks:**
6. Set up new GitHub repo with clean structure
7. Draft base class templates (BaseFilter, BaseNeuron, BaseStimulus)
8. Implement one extended stimulus (texture or multiple contacts)
9. Prototype Brian2 bridge (tensor conversion functions)
10. Write 1-page abstract for each paper

**Next Month:**
9. Complete extensibility refactor for Paper 1
10. Implement AM-KF with 6D control state
11. Test on simple control task (contact localization)
12. Have follow-up meetings with collaborators

---

## XII. Resources & References

### Key Papers to Cite (Conceptual Foundations)

**Tactile Encoding:**
- Parvizi-Fard et al. (2021) - "Functional spiking neuronal model for tactile perception," *J. Neurophysiol.*, vol. 126, pp. 1385–1397
- Johansson & Flanagan (2009) - tactile sensory coding review
- Weber et al. (2013) - SA/RA responses to edges

**Kalman Filtering:**
- Kalman (1960) - original KF paper
- Julier & Uhlmann (2004) - unscented KF
- Särkkä (2013) - Bayesian filtering textbook

**Attention Mechanisms:**
- Itti & Koch (2001) - computational models of attention
- Reynolds & Heeger (2009) - normalization model of attention
- Mnih et al. (2014) - recurrent attention models (ML)

**Diffusion Models:**
- Ho et al. (2020) - denoising diffusion probabilistic models
- Song et al. (2021) - score-based generative models
- Dhariwal & Nichol (2021) - improved diffusion models

**Fading Memory:**
- Principe et al. (2000) - information theoretic learning
- Jaeger (2001) - echo state networks
- Maass et al. (2002) - liquid state machines

**Hierarchical Sensory Processing:**
- Rao & Ballard (1999) - predictive coding
- Friston (2005) - free energy principle
- Lee & Mumford (2003) - hierarchical Bayesian inference

### Existing Codebases to Study

- **Spiking networks:** Brian2, NEST, BindsNET
- **Kalman filters:** FilterPy, PyKalman
- **Diffusion models:** Hugging Face Diffusers
- **Tactile simulation:** TacTip (Bristol Robotics Lab)

### Potential Funding Sources

- **NSF:** Robust Intelligence, Cyber-Physical Systems
- **NIH:** BRAIN Initiative (if neuroscience angle strong)
- **DARPA:** Biological Technologies Office, Microsystems Technology Office
- **Industry:** Meta Reality Labs, Google DeepMind, Microsoft Research
- **Foundations:** Simons Foundation, Sloan Foundation

---

## XIII. Conclusion

You're at an inflection point: the playground is built, now it's time to generalize and publish. The three-paper strategy provides:

1. **Paper 1:** Establishes the extensible platform and proves modality-agnostic design
2. **Paper 2:** Demonstrates efficient state estimation for control (the practical use case)
3. **Paper 3:** Explores scalable reconstruction with hardware feasibility analysis

Key reframings from your feedback:
- **Abstraction is built-in:** Compression + priors, not a separate layer
- **Control, not reconstruction:** AM-KF is for low-dimensional control states
- **Modality-agnostic:** Touch is inspiration, not limitation
- **Hardware-aware:** Edge computing and analog feasibility matter
- **Extensibility first:** Templates and auto-discovery for community adoption

**Your next milestone:** Create the new repo with extensibility architecture and demonstrate on two modalities. This single deliverable validates Paper 1's core claim and provides infrastructure for Papers 2 and 3.

---

**This document is a living roadmap. Update after your meeting and revisit quarterly.**
