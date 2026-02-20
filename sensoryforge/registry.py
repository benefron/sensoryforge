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
import warnings


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
        self._registry: Dict[str, tuple[Type, Optional[Callable]]] = {}
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
            cls: Component class (must have from_config() classmethod).
            factory_func: Optional factory function. If provided, this is called
                instead of cls(**kwargs). Useful for components that need
                special instantiation logic.
        
        Note:
            Registration is idempotent - if the same class is already registered
            with the same name, this is a no-op (even if factory_func differs).
            Only warns if overwriting with a different class.
        """
        if name in self._registry:
            existing_cls, existing_factory = self._registry[name]
            # If same class, skip silently (idempotent)
            # Note: We don't compare factory_func because it may be recreated
            # on each call to register_all(), but the class is what matters
            if existing_cls is cls:
                return
            # Otherwise warn about overwriting with different class
            warnings.warn(
                f"{self._name}: Component '{name}' already registered with "
                f"{existing_cls.__name__}, overwriting with {cls.__name__}",
                UserWarning,
            )
        self._registry[name] = (cls, factory_func)
    
    def create(self, name: str, **kwargs) -> Any:
        """Create a component instance by name.
        
        Args:
            name: Registered component name.
            **kwargs: Arguments passed to component constructor or factory_func.
        
        Returns:
            Component instance.
        
        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: Component '{name}' not registered. "
                f"Available: {available}"
            )
        
        cls, factory_func = self._registry[name]
        
        if factory_func is not None:
            return factory_func(**kwargs)
        else:
            # Try from_config() first if kwargs looks like a config dict
            if "config" in kwargs and hasattr(cls, "from_config"):
                return cls.from_config(kwargs["config"])
            elif hasattr(cls, "from_config") and len(kwargs) == 1 and "config" in kwargs:
                return cls.from_config(kwargs["config"])
            else:
                return cls(**kwargs)
    
    def list_registered(self) -> List[str]:
        """List all registered component names.
        
        Returns:
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if a component name is registered.
        
        Args:
            name: Component name to check.
        
        Returns:
            True if registered, False otherwise.
        """
        return name in self._registry
    
    def get_class(self, name: str) -> Type:
        """Get the registered class for a component name.
        
        Args:
            name: Registered component name.
        
        Returns:
            Component class.
        
        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: Component '{name}' not registered. "
                f"Available: {available}"
            )
        cls, _ = self._registry[name]
        return cls


# Global registries for each component type
NEURON_REGISTRY = ComponentRegistry("NEURON_REGISTRY")
FILTER_REGISTRY = ComponentRegistry("FILTER_REGISTRY")
INNERVATION_REGISTRY = ComponentRegistry("INNERVATION_REGISTRY")
STIMULUS_REGISTRY = ComponentRegistry("STIMULUS_REGISTRY")
SOLVER_REGISTRY = ComponentRegistry("SOLVER_REGISTRY")
GRID_REGISTRY = ComponentRegistry("GRID_REGISTRY")
PROCESSING_REGISTRY = ComponentRegistry("PROCESSING_REGISTRY")
