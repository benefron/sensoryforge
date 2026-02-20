"""Unit tests for extensibility patterns."""

import pytest
import torch
from typing import Dict, Any
from sensoryforge.registry import ComponentRegistry, NEURON_REGISTRY
from sensoryforge.register_components import register_all


class CustomNeuron:
    """Example custom neuron for testing extensibility."""
    
    def __init__(self, param1: float = 1.0, param2: float = 2.0, dt: float = 1.0):
        self.param1 = param1
        self.param2 = param2
        self.dt = dt
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.zeros_like(input_current)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CustomNeuron':
        """Create from config."""
        return cls(
            param1=config.get("param1", 1.0),
            param2=config.get("param2", 2.0),
            dt=config.get("dt", 1.0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "custom_neuron",
            "param1": float(self.param1),
            "param2": float(self.param2),
            "dt": float(self.dt),
        }


class TestExtensibility:
    """Test extensibility patterns."""
    
    def test_register_custom_component(self):
        """Test registering a custom component."""
        registry = ComponentRegistry("test")
        
        registry.register("custom_neuron", CustomNeuron)
        assert registry.is_registered("custom_neuron")
        
        neuron_cls = registry.get_class("custom_neuron")
        assert neuron_cls == CustomNeuron
    
    def test_create_from_registry(self):
        """Test creating component via registry."""
        registry = ComponentRegistry("test")
        registry.register("custom_neuron", CustomNeuron)
        
        neuron = registry.create("custom_neuron", param1=1.5, param2=2.5, dt=0.5)
        assert isinstance(neuron, CustomNeuron)
        assert neuron.param1 == 1.5
        assert neuron.param2 == 2.5
    
    def test_config_round_trip(self):
        """Test config serialization round-trip."""
        neuron = CustomNeuron(param1=1.5, param2=2.5, dt=0.5)
        
        # Serialize
        config = neuron.to_dict()
        assert config["type"] == "custom_neuron"
        assert config["param1"] == 1.5
        
        # Deserialize
        neuron2 = CustomNeuron.from_config(config)
        assert neuron2.param1 == neuron.param1
        assert neuron2.param2 == neuron.param2
        assert neuron2.dt == neuron.dt
    
    def test_registry_lookup_vs_if_else(self):
        """Test that registry lookup works better than if/else chains."""
        register_all()
        
        # Registry lookup (extensible)
        neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        assert neuron_cls is not None
        
        # If/else chain (not extensible without code changes)
        # This test demonstrates why registry is better
        model_name = "izhikevich"
        if model_name == "izhikevich":
            neuron_cls2 = neuron_cls  # Would need to import here
        else:
            neuron_cls2 = None
        
        assert neuron_cls2 == neuron_cls
