---
name: sensoryforge-plugin-lifecycle
description: Take a SensoryForge plugin from scaffold to install: version it, name the SensoryForge release it needs, build it, install it in the right environment and confirm SensoryForge found it. Covers updating, sharing and removing it too. Use when someone wants to package, install, ship, version, share, update or uninstall a component they wrote, asks why their plugin or new component does not show up, or says it was installed but SensoryForge cannot see it, which version of SensoryForge it works with, or how to hand it to a colleague, in whatever words they put it.
---

# The life of a plugin

A plugin is a small Python package that registers one or more components (a neuron, a
filter, a world shape, a metric pack, and so on) with SensoryForge when it is installed.
Nothing inside SensoryForge is edited. This skill is the whole path around the component:
the package, its version, the install and the proof that SensoryForge found it. What
goes inside the component is the job of `sensoryforge-extend-pipeline` (neuron, filter,
stimulus, grid, receptive-field builder, solver, processing layer, event converter),
`sensoryforge-extend-world` (world shapes, patterns, modulations, distributions, class
kinds) and `sensoryforge-extend-evaluation` (metric packs, declared adapters); use the
one that fits for step 3 and come back here. You end with a package that passes its
contract check, is installed in the environment SensoryForge runs in, and shows up in
SensoryForge's own list, and with the user knowing which release it needs.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Which route, and which environment?

- **One experiment, no sharing:** a module next to the config and a `plugins:` entry
  naming it (`my_module:register`) need no package at all. Offer it first when the user
  has one config and no wish to install anything. The rest of this skill is for a
  package: a component reused across configs or projects, or handed to someone.
- **Find the environment first.** The plugin must be installed in the same Python that
  runs the sensoryforge command. Check with `python -c "import sensoryforge, sys;
  print(sensoryforge.__file__, sys.executable)"`, and use `python -m pip <command>`, never a bare
  pip that may belong to another Python. A plugin installed into a different
  environment is the commonest reason it "does not show up".

## 2. What the user decides

Ask for these; do not choose them. A request that leaves them open ("get it to my
colleague") is a question to answer first: change and build nothing until it is answered. Use `sensoryforge-ground-assumptions` for a value
that is a fact about the science, which these are not:

- the **component's name and kind** (the registry name configs will use; it is the
  user's to keep stable, see step 6);
- the **package name** (the scaffold writes `sensoryforge-NAME`) and who may use it
  (a licence is theirs to pick; the scaffold writes none);
- **where it goes**: this environment only, a colleague's, or published. For a
  colleague, ask how they will get it and whether they already have SensoryForge in
  their environment, and which release. Handing a
  wheel or a repository link to someone is fine to offer; uploading to an index or
  pushing to a remote is outward-facing, so ask first and only when they say to.

## 3. Scaffold and fill in

```
sensoryforge new-component neuron NAME --dest .
```

The kinds with a scaffold are neuron, filter, stimulus, solver, grid and innervation (a
receptive-field builder). It writes `sensoryforge-NAME/` with `pyproject.toml`, a
module whose `register()` function does the registering, a contract test and a README,
and refuses to write inside an installed package. A processing layer, an event
converter, a world component and an evaluation component have no scaffold: write the
module by hand and copy the scaffold's `pyproject.toml` (the `[project.entry-points]`
section is the whole mechanism; the module path and the `register()` function are what
it names). Then fill in the component with the skill for its kind, from the user's
definition and no one else's. A package may register several components from one
`register()`. The scaffold's constructor takes `dt=` (the integration step, which the
engine passes in) and has a placeholder default: keep it, and tell the user it is there
and that it is not one of their parameters.

## 4. Check it before it is built

Run the contract check for the kind on the source folder, not an installed copy: `check_component(kind, Class, instance=...)` from `sensoryforge.testing.contracts`
for the pipeline kinds (give it an instance built with non-default values, so the
`to_dict()` round trip can fail), `check_world_component()` from
`sensoryforge.testing.world_contracts` for a world component, and
`check_evaluation_component()` from `sensoryforge.testing.evaluate_contracts` for an
evaluation component. The scaffold's own test calls the first of these; pytest is
not installed with SensoryForge, so run that test only if the user has pytest (ask before
installing it) and otherwise call the check from a few lines of Python, as the example
does. Say that the check passed, by name. It shows the component is well formed, not
that it is right: also run the case the user can check by hand.

## 5. Name the version and the SensoryForge it needs

Open `pyproject.toml` with the user, and set:

- `version = "0.1.0"`, the plugin's own, starting there;
- `dependencies = ["sensoryforge"]`: change it to a lower bound, the release
  the plugin was checked against, which is what
  `python -c "import sensoryforge; print(sensoryforge.__version__)"` prints (for a
  source install this is also the Version line of `python -m pip show sensoryforge | head -3`). Write
  `"sensoryforge>=1.3.0"`, not an exact pin and not an upper bound, unless the user knows
  a release that breaks it.

Say this to the user in so many words, because nothing in SensoryForge enforces more
than pip does: **SensoryForge has no plugin version number of its own and checks nothing
when it loads a plugin.** The bound is read only when pip installs the plugin, and you
tested the plugin on that one release only: you cannot say it works on any other. A later
release is not warned about; check it again after the user updates SensoryForge. Also, an editable source install keeps the version it had when it
was installed, so pip show can lag `sensoryforge.__version__` until it is reinstalled;
compare both before trusting the bound. And SensoryForge is not on PyPI: if pip says it
cannot find or wants to install SensoryForge, the environment is not the one
SensoryForge is in, or its version is below the bound. Stop and tell the user; do not
install anything to get past it.

## 6. Build and install it

```
python -m pip wheel --no-deps -w dist ./sensoryforge-NAME
python -m pip install dist/sensoryforge_NAME-0.1.0-py3-none-any.whl
```

The folder needs `./` before it, or pip reads it as a name to look up. A wheel is what
you give a colleague: ask whether they have SensoryForge, and which release, before you
tell them the steps. Offline, add
`--no-build-isolation` to the first line (it then needs `setuptools>=77` and the wheel package).
While the user is still editing, `python -m pip install --editable ./sensoryforge-NAME`
uses the folder as it is. Installing changes the user's environment: say which one.

## 7. Prove SensoryForge found it

A plugin that fails to load is skipped with a warning, so an install that "worked" proves
nothing. In a fresh process, import SensoryForge and look:

- **a pipeline component** appears in `sensoryforge list-components` under its kind;
- **a world or evaluation component** appears in `sensoryforge describe --section world`
  or `--section evaluation`;
- then use it once, from a config naming it, or from Python as the example does.

If it is missing, read the warning (run `python -W always -c "import
sensoryforge.register_components"`), and check in this order: the environment of step 1;
`python -m pip show sensoryforge-NAME` (installed at all, at the version you expect); the
entry point's `module:function` path against the real module and function name; and that
`register()` runs without error when called by hand. Fix the cause, not the check.

## 8. After it is installed

- **A change to it** is a higher version number and a reinstall (`python -m pip install --upgrade
  ...` with the new wheel; an editable install picks up the source without one, but not a
  new entry point). Configs name a component by its registry key, so a **renamed key, a
  removed parameter or a changed unit breaks every config that uses it**: that is the
  case to tell the user about, with a bump of the leading number; a new parameter with a
  default is the middle number; a fix is the last. This is a convention, not a rule
  SensoryForge enforces.
- **Removing it** is `python -m pip uninstall sensoryforge-NAME`; the component's name
  then vanishes from the list, and a config that names it no longer runs.

## Verify

The step 4 check passed by name, `pyproject.toml` holds the version and lower bound the
user agreed, the plugin is in SensoryForge's environment and a fresh process lists its
names, nothing was uploaded or pushed, and every number in it came from the user.

## Worked example

A filter scaffolded, given a gain the user asked for, checked, versioned with the
SensoryForge it needs, built into a wheel, installed into a folder (so the example leaves
the environment alone), and confirmed from a fresh process:

```bash
sensoryforge new-component filter Bandpass --dest .
# prints: Scaffolded new filter plugin package
```

```python
# file: sensoryforge-bandpass/sensoryforge_bandpass/component.py
"""A single-pole lowpass with an output gain (the user's definition)."""

from typing import List, Optional

import torch

from sensoryforge.filters.base import BaseFilter
from sensoryforge.stimuli.base import ParamSpec


class BandpassFilterTorch(BaseFilter):
    """Single-pole lowpass (tau in ms) scaled by a unitless gain."""

    def __init__(self, tau: float = 10.0, gain: float = 1.0, dt: float = 0.1) -> None:
        super().__init__(dt=dt)
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.gain = gain
        self._state: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self._state = None

    def forward(self, x: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """Filter the drive ``[batch, time, N_neurons]`` (mA); same shape out."""
        step = dt if dt is not None else self.dt
        batch, steps, n = x.shape
        state = (
            self._state
            if self._state is not None
            else torch.zeros(batch, n, device=x.device, dtype=x.dtype)
        )
        outputs = []
        for t in range(steps):
            state = state + (x[:, t, :] - state) / self.tau * step
            outputs.append(state.unsqueeze(1))
        self._state = state.detach()
        return self.gain * torch.cat(outputs, dim=1)

    def to_dict(self) -> dict:
        return {"tau": self.tau, "gain": self.gain, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict) -> "BandpassFilterTorch":
        return cls(
            tau=config.get("tau", 10.0),
            gain=config.get("gain", 1.0),
            dt=config.get("dt", 0.1),
        )

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("tau", dtype="float", default=10.0, min_val=0.01, unit="ms"),
            ParamSpec("gain", dtype="float", default=1.0, min_val=0.0, unit=""),
        ]


def register() -> None:
    """Entry-point target."""
    from sensoryforge.registry import FILTER_REGISTRY

    FILTER_REGISTRY.register("bandpass", BandpassFilterTorch)
```

```toml
# file: sensoryforge-bandpass/pyproject.toml
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sensoryforge-bandpass"
version = "0.2.0"
description = "SensoryForge filter plugin: BandpassFilterTorch"
requires-python = ">=3.10"
dependencies = ["sensoryforge>=1.3.0"]

[project.entry-points."sensoryforge.components"]
bandpass = "sensoryforge_bandpass.component:register"

[tool.setuptools.packages.find]
where = ["."]
include = ["sensoryforge_bandpass*"]
```

```python
# file: check_it.py
import sys

sys.path.insert(0, "sensoryforge-bandpass")
from sensoryforge.testing.contracts import check_component
from sensoryforge_bandpass.component import BandpassFilterTorch

check_component(
    "filter", BandpassFilterTorch, instance=BandpassFilterTorch(tau=7.0, gain=2.5, dt=0.2)
)
print("contract ok")
```

```python
# file: verify.py
import importlib.metadata as md
import sys

sys.path.insert(0, "site")  # the install target; a real install needs no line like this
import torch

import sensoryforge_bandpass
from sensoryforge.registry import FILTER_REGISTRY

print("from the install", "site" in sensoryforge_bandpass.__file__)
print("registered", "bandpass" in FILTER_REGISTRY.list_registered())
print("version", md.version("sensoryforge-bandpass"), md.requires("sensoryforge-bandpass"))
y = FILTER_REGISTRY.create("bandpass", tau=5.0, gain=2.0, dt=0.1)(torch.ones(1, 50, 2))
print("step response", round(float(y[0, -1, 0]), 3))
```

```bash
python check_it.py
# prints: contract ok
python -m pip wheel --no-deps --no-build-isolation -w dist ./sensoryforge-bandpass
# prints: Successfully built sensoryforge-bandpass
python -m pip install --no-deps --target site dist/sensoryforge_bandpass-0.2.0-py3-none-any.whl
# prints: Successfully installed sensoryforge-bandpass-0.2.0
python verify.py
# prints: from the install True
# prints: registered True
# prints: version 0.2.0 ['sensoryforge>=1.3.0']
# prints: step response 1.272
```
