"""Component registry system for SensoryForge extensibility.

This module provides a unified registry pattern for all component types
(neurons, filters, innervation methods, stimuli, solvers, grids, processing layers).
Components register themselves and can be instantiated by name via config.

The registry pattern ensures:
- No hardcoded if/else chains for component lookup
- Easy extensibility: new components just register themselves
- Consistent from_config() pattern across all components
- Clear error messages for unregistered components

Example:
    >>> from sensoryforge.registry import NEURON_REGISTRY
    >>> NEURON_REGISTRY.register("custom_neuron", MyCustomNeuron)
    >>> neuron = NEURON_REGISTRY.create("custom_neuron", **config)
"""

from __future__ import annotations

from typing import Any, Dict, List, Type, Optional, Callable


class ComponentRegistry:
    """Generic registry for component classes.

    Components register themselves with a string name, then can be
    instantiated by name with keyword arguments.

    Attributes:
        _registry: Dict mapping component name → (class, factory_func)
            factory_func is optional - if None, class is instantiated directly
    """

    def __init__(self, registry_name: str = "ComponentRegistry"):
        """Initialize empty registry.

        Args:
            registry_name: Name for error messages (e.g., "NEURON_REGISTRY").
        """
        # Keyed by casefolded name -> (display_name, cls, factory_func). The
        # display_name is the first spelling a component was registered
        # under (F-046: lookups are case-insensitive, but list_registered()
        # still shows one canonical spelling per component).
        self._registry: Dict[str, tuple[str, Type, Optional[Callable]]] = {}
        self._name = registry_name

    def register(
        self,
        name: str,
        cls: Type,
        factory_func: Optional[Callable] = None,
    ) -> None:
        """Register a component class.

        Args:
            name: String identifier for this component (e.g., "izhikevich").
                Lookups (``get_class``, ``is_registered``, ``create``) are
                case-insensitive (F-046): ``"Izhikevich"`` and
                ``"izhikevich"`` refer to the same registration.
            cls: Component class (must have from_config() classmethod).
            factory_func: Optional factory function. If provided, this is called
                instead of cls(**kwargs). Useful for components that need
                special instantiation logic.

        Note:
            Registration is idempotent - if the same class is already registered
            under a name that case-folds the same, this is a no-op that keeps
            the original display spelling (even if factory_func differs).
            Raises ``ValueError`` if a *different* class is registered under a
            name that case-folds to an existing registration.

        Raises:
            ValueError: If ``name`` case-folds to an existing registration
                for a different class.
        """
        key = name.casefold()
        if key in self._registry:
            existing_name, existing_cls, existing_factory = self._registry[key]
            # If same class, skip silently (idempotent) -- keep the
            # original display spelling.
            # Note: We don't compare factory_func because it may be recreated
            # on each call to register_all(), but the class is what matters
            if existing_cls is cls:
                return
            # Otherwise this is a genuine name collision between two
            # different classes -- fail loudly rather than silently
            # shadowing one plugin with another.
            raise ValueError(
                f"{self._name}: Component '{name}' already registered as "
                f"'{existing_name}' with {existing_cls.__name__}; cannot "
                f"register {cls.__name__} under a name that case-folds the "
                f"same"
            )
        self._registry[key] = (name, cls, factory_func)

    def create(self, name: str, **kwargs) -> Any:
        """Create a component instance by name.

        Uses factory function if provided, otherwise instantiates class directly.
        Supports both direct instantiation and `from_config()` pattern.

        Args:
            name: Registered component name (e.g., "izhikevich", "sa", "gaussian").
            **kwargs: Arguments passed to component constructor or factory_func.
                If `config` key is present and component has `from_config()`, uses that.

        Returns:
            Component instance (type depends on registered class).

        Raises:
            KeyError: If name is not registered. Error message includes available names.

        Example:
            >>> # Direct instantiation
            >>> neuron = NEURON_REGISTRY.create("izhikevich", dt=1.0, a=0.02, b=0.2)
            >>>
            >>> # Using from_config pattern
            >>> neuron = NEURON_REGISTRY.create("izhikevich", config={"dt": 1.0, "a": 0.02})
            >>>
            >>> # Factory function (for innervation)
            >>> innervation = INNERVATION_REGISTRY.create(
            ...     "gaussian",
            ...     receptor_coords=coords,
            ...     neuron_centers=centers,
            ...     connections_per_neuron=28.0,
            ...     device="cpu"
            ... )
        """
        key = name.casefold()
        if key not in self._registry:
            available = ", ".join(sorted(n for n, _, _ in self._registry.values()))
            raise KeyError(
                f"{self._name}: Component '{name}' not registered. "
                f"Available: {available}"
            )

        _, cls, factory_func = self._registry[key]

        if factory_func is not None:
            return factory_func(**kwargs)
        else:
            # Try from_config() first if kwargs looks like a config dict
            if "config" in kwargs and hasattr(cls, "from_config"):
                return cls.from_config(kwargs["config"])
            elif (
                hasattr(cls, "from_config") and len(kwargs) == 1 and "config" in kwargs
            ):
                return cls.from_config(kwargs["config"])
            else:
                return cls(**kwargs)

    def list_registered(self) -> List[str]:
        """List all registered component names.

        One entry per case-folded key, using the first-registered spelling
        as the display name (F-046) -- e.g. registering both "Izhikevich"
        and "izhikevich" for the same class produces a single "Izhikevich"
        entry, not two.

        Returns:
            Sorted list of display names, one per component.
        """
        return sorted(name for name, _, _ in self._registry.values())

    def is_registered(self, name: str) -> bool:
        """Check if a component name is registered.

        Args:
            name: Component name to check (case-insensitive).

        Returns:
            True if registered, False otherwise.
        """
        return name.casefold() in self._registry

    def get_class(self, name: str) -> Type:
        """Get the registered class for a component name.

        Args:
            name: Registered component name (case-insensitive).

        Returns:
            Component class.

        Raises:
            KeyError: If name is not registered.
        """
        key = name.casefold()
        if key not in self._registry:
            available = ", ".join(sorted(n for n, _, _ in self._registry.values()))
            raise KeyError(
                f"{self._name}: Component '{name}' not registered. "
                f"Available: {available}"
            )
        _, cls, _ = self._registry[key]
        return cls

    def name_for(self, cls: Type) -> Optional[str]:
        """Return the display name ``cls`` is registered under, or ``None``.

        The first registration whose class is exactly ``cls`` wins (aliases
        registered later map to the same class but do not override it).
        """
        for display_name, registered_cls, _ in self._registry.values():
            if registered_cls is cls:
                return display_name
        return None

    def get_param_spec(self, name: str) -> list:
        """Return the ``get_param_spec()`` list for a registered component.

        Calls ``cls.get_param_spec()`` on the registered class.  If the class
        does not implement ``get_param_spec`` the method returns an empty list
        rather than raising.

        Args:
            name: Registered component name.

        Returns:
            List of :class:`~sensoryforge.stimuli.base.ParamSpec` instances,
            or an empty list if the class does not expose one.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        cls = self.get_class(name)
        spec_fn = getattr(cls, "get_param_spec", None)
        if spec_fn is None or not callable(spec_fn):
            return []
        return spec_fn()


# Global registries for each component type
NEURON_REGISTRY = ComponentRegistry("NEURON_REGISTRY")
FILTER_REGISTRY = ComponentRegistry("FILTER_REGISTRY")
INNERVATION_REGISTRY = ComponentRegistry("INNERVATION_REGISTRY")
STIMULUS_REGISTRY = ComponentRegistry("STIMULUS_REGISTRY")
SOLVER_REGISTRY = ComponentRegistry("SOLVER_REGISTRY")
GRID_REGISTRY = ComponentRegistry("GRID_REGISTRY")
PROCESSING_REGISTRY = ComponentRegistry("PROCESSING_REGISTRY")
