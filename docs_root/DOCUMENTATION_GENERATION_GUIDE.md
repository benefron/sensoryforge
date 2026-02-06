# Documentation Generation Guide for SensoryForge

**Purpose:** Comprehensive guide for generating, maintaining, and deploying documentation for the SensoryForge playground repository

---

## I. Documentation Philosophy

### Core Principles

1. **User-First:** Documentation serves users at all levels (beginners, researchers, contributors)
2. **Completeness:** Every public API must be documented with examples
3. **Accuracy:** Docs must stay synchronized with code (automatic verification)
4. **Discoverability:** Users should easily find what they need
5. **Maintainability:** Documentation should be easy to update and extend

### Documentation Layers

1. **In-Code Documentation:** Docstrings, type hints, inline comments
2. **User Guides:** Conceptual explanations and workflows
3. **Tutorials:** Step-by-step learning paths
4. **API Reference:** Auto-generated from docstrings
5. **Developer Guides:** Contributing, architecture, testing

---

## II. Documentation Stack

### Tools and Technologies

- **MkDocs Material:** Static site generator with beautiful theme
- **mkdocstrings:** Auto-generate API reference from docstrings
- **Python-Markdown:** Extended markdown with LaTeX, code highlighting
- **GitHub Pages:** Free hosting for documentation
- **GitHub Actions:** Automated building and deployment

### File Structure

```
sensoryforge/
├── docs/                           # MkDocs source files
│   ├── index.md                    # Landing page
│   ├── getting_started/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── first_simulation.md
│   ├── user_guide/
│   │   ├── overview.md
│   │   ├── concepts.md
│   │   ├── configuration.md
│   │   ├── touch.md
│   │   ├── vision.md
│   │   └── stimuli.md
│   ├── tutorials/
│   │   ├── basic_pipeline.md
│   │   ├── custom_filter.md
│   │   ├── custom_neuron.md
│   │   ├── brian2_integration.md
│   │   └── multimodal.md
│   ├── extending/
│   │   ├── plugins.md
│   │   ├── filters.md
│   │   ├── neurons.md
│   │   └── stimuli.md
│   ├── api_reference/
│   │   ├── core.md
│   │   ├── filters.md
│   │   ├── neurons.md
│   │   ├── decoding.md
│   │   └── brian_bridge.md
│   ├── developer/
│   │   ├── contributing.md
│   │   ├── style.md
│   │   ├── testing.md
│   │   └── documentation.md
│   ├── javascripts/
│   │   └── mathjax.js             # LaTeX math rendering
│   └── stylesheets/
│       └── extra.css               # Custom styling
├── mkdocs.yml                      # MkDocs configuration
├── sensoryforge/                   # Source code with docstrings
└── examples/                       # Code examples for docs
```

---

## III. Writing Documentation

### A. In-Code Documentation

#### Docstring Standards

**Template:**
```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional: Type3 = default
) -> ReturnType:
    """One-line summary ending with period.
    
    Extended description providing more context. Can span multiple
    paragraphs. Link to relevant concepts or papers.
    
    Args:
        param1: Description. Include tensor shape [batch, time, channels]
            and physical units (e.g., "in mA"). Can span multiple lines.
        param2: Another parameter description.
        optional: Optional parameter. Default: default.
    
    Returns:
        Description of return value. Include shape and units.
        For complex returns, describe structure.
    
    Raises:
        ValueError: When this is raised and why.
        RuntimeError: Another exception case.
    
    Example:
        >>> # Simple usage
        >>> result = function_name(
        ...     param1=value1,
        ...     param2=value2
        ... )
        >>> result.shape
        torch.Size([10, 20])
        
        >>> # Advanced usage
        >>> result = function_name(param1=value1, param2=value2, optional=custom)
    
    Note:
        Important caveats or warnings about usage.
    
    References:
        Author et al. (Year). "Paper Title". Journal.
        https://doi.org/10.xxxx/xxxxx
    """
    pass
```

**Class Docstrings:**
```python
class MyClass:
    """One-line summary of class purpose.
    
    Extended description of what this class does, when to use it,
    and how it fits into the larger architecture.
    
    Attributes:
        attr1: Description of public attribute
        attr2: Description of another attribute
    
    Example:
        >>> obj = MyClass(param=value)
        >>> result = obj.process(data)
    """
    
    def __init__(self, param: Type):
        """Initialize MyClass.
        
        Args:
            param: Parameter description
        """
        pass
```

#### Type Hints

Always include type hints:
```python
from typing import Dict, List, Optional, Tuple, Union
import torch

def process_data(
    data: torch.Tensor,
    config: Dict[str, Any],
    optional: Optional[List[str]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process input data."""
    pass
```

### B. Markdown Documentation

#### Page Structure Template

```markdown
# Page Title

Brief overview of what this page covers (1-2 sentences).

## Section 1: Topic Name

Explanation of the topic. Use clear, concise language.

### Subsection

More detailed content.

## Code Examples

Always include working code examples:

\```python
from sensoryforge import SomeClass

# Create instance
instance = SomeClass(param=value)

# Use it
result = instance.method(data)
print(f"Result: {result}")
\```

## Visual Aids

Include diagrams where helpful:

\```mermaid
graph LR
    A[Input] --> B[Process]
    B --> C[Output]
\```

## Mathematical Notation

Use LaTeX for equations:

The filter equation is:

$$
\\tau \\frac{dI}{dt} = -I + g \\cdot s(t)
$$

where $\\tau$ is the time constant and $g$ is the gain.

## See Also

- [Related Topic 1](../other_page.md)
- [API Reference](../../api_reference/module.md)
- [Tutorial](../../tutorials/example.md)
```

#### Writing Guidelines

**DO:**
- Start with the simplest example
- Build complexity gradually
- Explain WHY not just HOW
- Link to related content
- Include expected output
- Use realistic examples
- Test all code examples

**DON'T:**
- Assume prior knowledge
- Use jargon without explanation
- Leave code untested
- Have broken links
- Forget to update when code changes

### C. Tutorials

#### Tutorial Structure

```markdown
# Tutorial: [Descriptive Title]

**Learning Objectives:**
- Objective 1
- Objective 2
- Objective 3

**Prerequisites:**
- Prerequisite 1
- Prerequisite 2

**Estimated Time:** 20 minutes

---

## Introduction

What you'll build and why it's useful.

## Step 1: Setup

\```python
# Setup code
import sensoryforge as sf
\```

**Explanation:** Why we need this.

## Step 2: Create Components

\```python
# Component creation
grid = sf.SpatialGrid(size_mm=(100, 100))
\```

**Explanation:** What this does and why.

## Step 3: Run Simulation

\```python
# Run code
results = pipeline.encode(stimulus)
\```

**Expected Output:**
\```
Generated 1250 spikes across 100 neurons
\```

## Step 4: Visualize Results

\```python
import matplotlib.pyplot as plt
plt.plot(results['spikes'].sum(dim=-1))
plt.show()
\```

## Summary

What you learned and next steps.

## Next Steps

- [Advanced Tutorial](advanced.md)
- [Related Concept](../user_guide/concept.md)
```

### D. API Reference

#### Auto-Generation with mkdocstrings

API reference pages use mkdocstrings to auto-generate from docstrings:

```markdown
# Core API Reference

## SensoryPipeline

::: sensoryforge.core.pipeline.SensoryPipeline
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - encode
        - decode
        - reset

## SpatialGrid

::: sensoryforge.core.grid.SpatialGrid
    options:
      show_root_heading: true
      show_source: true

## Functions

::: sensoryforge.core.innervation.create_sa_innervation
::: sensoryforge.core.innervation.create_ra_innervation
```

This automatically pulls docstrings and creates formatted API documentation.

---

## IV. MkDocs Configuration

### mkdocs.yml Structure

```yaml
# Site metadata
site_name: SensoryForge
site_description: Modular framework for sensory encoding
site_author: Your Name
site_url: https://benefron.github.io/sensoryforge

# Repository
repo_name: benefron/sensoryforge
repo_url: https://github.com/benefron/sensoryforge
edit_uri: edit/main/docs/

# Theme configuration
theme:
  name: material
  palette:
    # Light/dark mode toggle
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  
  features:
    - navigation.tabs          # Top-level tabs
    - navigation.sections      # Collapsible sections
    - navigation.expand        # Expand all by default
    - navigation.top           # Back to top button
    - search.suggest           # Search suggestions
    - search.highlight         # Highlight search terms
    - content.code.annotate    # Code annotations
    - content.code.copy        # Copy code button
  
  icon:
    repo: fontawesome/brands/github

# Plugins
plugins:
  - search                     # Search functionality
  - mkdocstrings:              # API reference auto-generation
      handlers:
        python:
          paths: [sensoryforge]
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
            show_category_heading: true
            members_order: source
            filters:
              - "!^_"          # Exclude private members
  - autorefs                   # Auto-reference linking

# Markdown extensions
markdown_extensions:
  - admonition                 # Call-out boxes
  - pymdownx.details           # Collapsible sections
  - pymdownx.superfences:      # Code fences + Mermaid
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:        # Syntax highlighting
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite      # Inline code highlighting
  - pymdownx.snippets          # Include external files
  - pymdownx.arithmatex:       # LaTeX math
      generic: true
  - toc:                       # Table of contents
      permalink: true
      toc_depth: 3
  - tables                     # Table support
  - attr_list                  # Add attributes to elements
  - md_in_html                 # Markdown in HTML
  - pymdownx.tabbed:           # Tabbed content
      alternate_style: true

# Extra JavaScript for math rendering
extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

# Extra CSS
extra_css:
  - stylesheets/extra.css

# Navigation structure
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting_started/installation.md
    - Quick Start: getting_started/quickstart.md
    - First Simulation: getting_started/first_simulation.md
  - User Guide:
    - Overview: user_guide/overview.md
    - Core Concepts: user_guide/concepts.md
    - Configuration: user_guide/configuration.md
    - Touch Encoding: user_guide/touch.md
    - Vision Encoding: user_guide/vision.md
    - Custom Stimuli: user_guide/stimuli.md
  - Tutorials:
    - Basic Pipeline: tutorials/basic_pipeline.md
    - Custom Filter: tutorials/custom_filter.md
    - Custom Neuron: tutorials/custom_neuron.md
    - Brian2 Integration: tutorials/brian2_integration.md
    - Multi-Modal: tutorials/multimodal.md
  - Extending:
    - Plugin System: extending/plugins.md
    - Filter Templates: extending/filters.md
    - Neuron Templates: extending/neurons.md
    - Stimulus Templates: extending/stimuli.md
  - API Reference:
    - Core: api_reference/core.md
    - Filters: api_reference/filters.md
    - Neurons: api_reference/neurons.md
    - Decoding: api_reference/decoding.md
    - Brian Bridge: api_reference/brian_bridge.md
  - Developer Guide:
    - Contributing: developer/contributing.md
    - Code Style: developer/style.md
    - Testing: developer/testing.md
    - Documentation: developer/documentation.md
```

---

## V. Building and Deploying

### Local Development

```bash
# Install documentation dependencies
pip install mkdocs-material mkdocstrings[python] mkdocs-autorefs

# Serve locally with live reload
mkdocs serve

# Open browser to http://localhost:8000
```

### Building Static Site

```bash
# Build documentation
mkdocs build

# Output in site/ directory
# Check site/index.html
```

### GitHub Actions Deployment

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install mkdocs-material mkdocstrings[python] mkdocs-autorefs
          pip install -e .
      
      - name: Build documentation
        run: mkdocs build --strict
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

**Setup:**
1. Create workflow file
2. Push to main branch
3. Enable GitHub Pages in repository settings
4. Select "gh-pages" branch as source
5. Documentation available at `https://benefron.github.io/sensoryforge`

---

## VI. Documentation Maintenance

### Regular Tasks

**Weekly:**
- [ ] Review open issues for documentation gaps
- [ ] Update FAQ based on common questions
- [ ] Fix broken links (use `mkdocs build --strict`)

**Per Release:**
- [ ] Update version numbers in examples
- [ ] Add changelog entry
- [ ] Update API reference
- [ ] Review and update getting started guide
- [ ] Add migration guide for breaking changes

**Monthly:**
- [ ] Review analytics (if enabled)
- [ ] Identify under-documented areas
- [ ] Update based on community feedback

### Quality Checks

**Automated:**
```bash
# Build with strict mode (fails on warnings)
mkdocs build --strict

# Check for broken links
# (Use linkchecker or similar tool)
linkchecker site/index.html

# Verify code examples
pytest docs/ --doctest-modules
```

**Manual:**
- [ ] All code examples run without errors
- [ ] All links work
- [ ] Math renders correctly
- [ ] Images load properly
- [ ] Mobile view looks good
- [ ] Search works
- [ ] Navigation makes sense

---

## VII. Best Practices

### Writing Tips

1. **Start Simple:** Begin with simplest use case
2. **Progressive Disclosure:** Add complexity gradually
3. **Show Output:** Always show expected results
4. **Explain Why:** Don't just show what, explain why
5. **Link Generously:** Connect related topics
6. **Update Together:** Code and docs in same commit

### Common Patterns

**Admonitions (call-out boxes):**
```markdown
!!! note
    Important information that deserves attention.

!!! warning
    Caution about potential issues.

!!! tip
    Helpful suggestion or best practice.

!!! example
    Example usage or application.
```

**Code Tabs:**
```markdown
=== "Python"
    \```python
    import sensoryforge as sf
    \```

=== "Configuration"
    \```yaml
    pipeline:
      type: basic
    \```
```

**Math Inline and Block:**
```markdown
The time constant $\tau$ controls decay rate.

The full equation is:

$$
\tau \frac{dI}{dt} = -I + g \cdot s(t)
$$
```

---

## VIII. Troubleshooting

### Common Issues

**Math not rendering:**
- Check `mathjax.js` is loaded
- Verify `pymdownx.arithmatex` configured
- Use `$$...$$` for block, `$...$` for inline

**API reference not generating:**
- Ensure `mkdocstrings` installed
- Check Python path in mkdocs.yml
- Verify docstrings use Google style
- Check for syntax errors in docstrings

**Broken links:**
- Run `mkdocs build --strict`
- Use relative paths
- Check file extensions (.md not .html)

**Search not working:**
- Verify search plugin enabled
- Rebuild site
- Clear browser cache

---

## IX. Examples

### Example Page: Getting Started

See `docs/getting_started/installation.md` in this repository for a complete example.

### Example API Reference

See `docs/api_reference/core.md` for auto-generated API documentation example.

### Example Tutorial

See `docs/tutorials/basic_pipeline.md` for a complete tutorial example.

---

## X. Resources

### Official Documentation
- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)

### Examples
- [FastAPI Docs](https://fastapi.tiangolo.com/) - Excellent example
- [PyTorch Docs](https://pytorch.org/docs/) - Comprehensive reference
- [Pydantic Docs](https://docs.pydantic.dev/) - Clean structure

### Tools
- [Mermaid Live Editor](https://mermaid.live/) - Create diagrams
- [LaTeX Math Editor](https://latex.codecogs.com/) - Test equations

---

**Remember:** Documentation is as important as code. Great documentation makes the difference between a tool used by dozens and a tool used by thousands. Invest the time to do it right.
