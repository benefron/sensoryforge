# SensoryForge Documentation

**Placeholder - Documentation in progress**

SensoryForge is a modular, extensible framework for simulating sensory encoding across multiple modalities.

## Quick Links

- [Installation](getting_started/installation.md) *(coming soon)*
- [Quick Start](getting_started/quickstart.md) *(coming soon)*
- [User Guide](user_guide/overview.md) *(coming soon)*
- [API Reference](api_reference/) *(coming soon)*

## Overview

SensoryForge provides:

- **Modality-agnostic architecture:** Same framework for touch, vision, audition
- **GPU-accelerated:** Built on PyTorch for efficient tensor operations
- **Highly extensible:** Plugin system for custom components
- **Production-ready:** Clear documentation, comprehensive testing, CI/CD

## Getting Started

```python
from sensoryforge import SensoryPipeline
from sensoryforge.stimuli import gaussian_pressure_torch

# Create pipeline
pipeline = SensoryPipeline.from_config('config.yml')

# Generate stimulus
# ...

# Run encoding
results = pipeline.encode(stimulus)
```

## Documentation Structure

**Phase 4 will add:**
- Getting started guides
- User guide with scientific concepts
- Tutorials for extending the framework
- Complete API reference
- Developer documentation

---

*This is a placeholder. Complete documentation will be generated in Phase 4 of the development plan.*
