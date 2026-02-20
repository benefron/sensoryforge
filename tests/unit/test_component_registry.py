"""Unit tests for component registry system."""

import pytest
import torch
from sensoryforge.registry import (
    ComponentRegistry,
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
)
from sensoryforge.register_components import register_all


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""
    
    def test_register_and_get_class(self):
        """Test registering and retrieving a class."""
        registry = ComponentRegistry("test")
        
        class TestComponent:
            pass
        
        registry.register("test_component", TestComponent)
        assert registry.is_registered("test_component")
        assert registry.get_class("test_component") == TestComponent
    
    def test_unregistered_component(self):
        """Test that unregistered components raise KeyError."""
        registry = ComponentRegistry("test")
        
        with pytest.raises(KeyError):
            registry.get_class("nonexistent")
    
    def test_list_registered(self):
        """Test listing registered components."""
        registry = ComponentRegistry("test")
        
        class Component1:
            pass
        class Component2:
            pass
        
        registry.register("comp1", Component1)
        registry.register("comp2", Component2)
        
        registered = registry.list_registered()
        assert "comp1" in registered
        assert "comp2" in registered
    
    def test_factory_function(self):
        """Test using factory function for creation."""
        registry = ComponentRegistry("test")
        
        class TestComponent:
            def __init__(self, value):
                self.value = value
        
        def factory(**kwargs):
            return TestComponent(kwargs.get("value", 0))
        
        registry.register("test", TestComponent, factory)
        
        # Create should use factory
        instance = registry.create("test", value=42)
        assert instance.value == 42
        
        # get_class should return class
        cls = registry.get_class("test")
        assert cls == TestComponent


class TestNeuronRegistry:
    """Test NEURON_REGISTRY."""
    
    def test_registered_neurons(self):
        """Test that standard neurons are registered."""
        register_all()
        
        assert NEURON_REGISTRY.is_registered("izhikevich")
        assert NEURON_REGISTRY.is_registered("adex")
        assert NEURON_REGISTRY.is_registered("mqif")
        assert NEURON_REGISTRY.is_registered("fa")
        assert NEURON_REGISTRY.is_registered("sa")
    
    def test_create_neuron(self):
        """Test creating a neuron via registry."""
        register_all()
        
        neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        assert neuron_cls is not None
        
        # Verify it can be instantiated
        neuron = neuron_cls(dt=1.0)
        assert neuron is not None


class TestFilterRegistry:
    """Test FILTER_REGISTRY."""
    
    def test_registered_filters(self):
        """Test that standard filters are registered."""
        register_all()
        
        assert FILTER_REGISTRY.is_registered("sa")
        assert FILTER_REGISTRY.is_registered("ra")
        assert FILTER_REGISTRY.is_registered("none")
    
    def test_create_filter(self):
        """Test creating a filter via registry."""
        register_all()
        
        filter_cls = FILTER_REGISTRY.get_class("sa")
        assert filter_cls is not None


class TestInnervationRegistry:
    """Test INNERVATION_REGISTRY."""
    
    def test_registered_innervation_methods(self):
        """Test that standard innervation methods are registered."""
        register_all()
        
        assert INNERVATION_REGISTRY.is_registered("gaussian")
        assert INNERVATION_REGISTRY.is_registered("uniform")
        assert INNERVATION_REGISTRY.is_registered("one_to_one")
        assert INNERVATION_REGISTRY.is_registered("distance_weighted")
    
    def test_create_innervation(self):
        """Test creating innervation via registry."""
        register_all()
        
        innervation_cls = INNERVATION_REGISTRY.get_class("gaussian")
        assert innervation_cls is not None
