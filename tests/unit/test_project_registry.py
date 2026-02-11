"""Unit tests for project registry.

Tests for ReviewFindings#T1.
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from sensoryforge.utils.project_registry import (
    ProtocolDefinition,
    NeuronModuleManifest,
    ProtocolRunRecord,
    STAAnalysisRecord,
    ProjectRegistry,
)


class TestProtocolDefinition:
    """Test suite for ProtocolDefinition serialization."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization roundtrip for ProtocolDefinition."""
        protocol = ProtocolDefinition(
            protocol_id="gaussian_tap",
            version="1.0",
            name="Gaussian Tap",
            description="A Gaussian pressure tap stimulus",
            stimulus={"type": "gaussian", "sigma": 2.0},
            execution={"duration_ms": 500, "dt_ms": 1.0},
            tags=["spatial", "transient"],
            metadata={"author": "test"}
        )
        
        # Serialize
        data = protocol.to_dict()
        
        # Deserialize
        restored = ProtocolDefinition.from_dict(data)
        
        # Verify
        assert restored.protocol_id == protocol.protocol_id
        assert restored.version == protocol.version
        assert restored.name == protocol.name
        assert restored.description == protocol.description
        assert restored.stimulus == protocol.stimulus
        assert restored.execution == protocol.execution
        assert list(restored.tags) == list(protocol.tags)
        assert restored.metadata == protocol.metadata

    def test_validates_protocol_id(self):
        """Test that invalid protocol_id raises ValueError."""
        with pytest.raises(ValueError, match="invalid characters"):
            ProtocolDefinition(
                protocol_id="invalid id!",  # Space and ! are invalid
                version="1.0",
                name="Test",
                description="Test",
                stimulus={},
                execution={}
            )


class TestNeuronModuleManifest:
    """Test suite for NeuronModuleManifest serialization."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization roundtrip for NeuronModuleManifest."""
        manifest = NeuronModuleManifest(
            name="sa_izhikevich",
            model="IzhikevichNeuronTorch",
            filter="SAFilterTorch",
            parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
            file="sa_izhikevich.pt",
            tags=["sa", "izhikevich"],
            metadata={"created": "2026-02-11"}
        )
        
        # Serialize
        data = manifest.to_dict()
        
        # Deserialize
        restored = NeuronModuleManifest.from_dict(data)
        
        # Verify
        assert restored.name == manifest.name
        assert restored.model == manifest.model
        assert restored.filter == manifest.filter
        assert restored.parameters == manifest.parameters
        assert restored.file == manifest.file
        assert list(restored.tags) == list(manifest.tags)
        assert restored.metadata == manifest.metadata


class TestProtocolRunRecord:
    """Test suite for ProtocolRunRecord serialization.
    
    Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#T1
    """

    @pytest.fixture
    def sample_neuron_modules(self):
        """Create sample neuron module manifests."""
        return [
            NeuronModuleManifest(
                name="sa_test",
                model="IzhikevichNeuronTorch",
                filter="SAFilterTorch",
                parameters={"a": 0.02},
                file="sa_test.pt"
            ),
            NeuronModuleManifest(
                name="ra_test",
                model="IzhikevichNeuronTorch",
                filter="RAFilterTorch",
                parameters={"a": 0.02},
                file="ra_test.pt"
            )
        ]

    def test_to_dict_and_from_dict_roundtrip(self, sample_neuron_modules):
        """Test serialization roundtrip for ProtocolRunRecord."""
        record = ProtocolRunRecord.new(
            run_id="run_001",
            protocol_id="gaussian_tap",
            stimulus_reference="protocols/gaussian_tap/stimulus.pt",
            neuron_modules=sample_neuron_modules,
            tensors={
                "sa_spikes": "runs/run_001/sa_spikes.pt",
                "ra_spikes": "runs/run_001/ra_spikes.pt"
            },
            metrics={"spike_count": 1000},
            notes="Test run",
            metadata={"device": "cpu"}
        )
        
        # Serialize
        data = record.to_dict()
        
        # Deserialize
        restored = ProtocolRunRecord.from_dict(data)
        
        # Verify
        assert restored.run_id == record.run_id
        assert restored.protocol_id == record.protocol_id
        assert restored.stimulus_reference == record.stimulus_reference
        assert len(restored.neuron_modules) == len(record.neuron_modules)
        assert restored.tensors == record.tensors
        assert restored.metrics == record.metrics
        assert restored.notes == record.notes
        assert restored.metadata == record.metadata

    def test_created_at_is_automatically_set(self, sample_neuron_modules):
        """Test that created_at timestamp is set automatically."""
        record = ProtocolRunRecord.new(
            run_id="run_002",
            protocol_id="gaussian_tap",
            stimulus_reference="test.pt",
            neuron_modules=sample_neuron_modules,
            tensors={}
        )
        
        # Verify created_at is a valid ISO timestamp
        created_at = datetime.fromisoformat(record.created_at.replace('Z', '+00:00'))
        assert isinstance(created_at, datetime)


class TestSTAAnalysisRecord:
    """Test suite for STAAnalysisRecord serialization."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization roundtrip for STAAnalysisRecord."""
        from sensoryforge.utils.project_registry import STAConfiguration, STAConfigurationResult
        
        result = STAConfigurationResult(
            kernel="sta_analyses/sta_001/kernel_0.pt",
            metrics={"spike_count": 10},
            parameters={"neuron_idx": 0}
        )
        
        config = STAConfiguration(
            name="standard_window",
            method="standard",
            signal_source="spikes",
            parameters={"window_ms": 100},
            results=[result]
        )
        
        record = STAAnalysisRecord.new(
            analysis_id="sta_001",
            source_run="run_001",
            population="sa_test",
            configurations=[config],
            metadata={"note": "test"}
        )
        
        # Serialize
        data = record.to_dict()
        
        # Deserialize
        restored = STAAnalysisRecord.from_dict(data)
        
        # Verify
        assert restored.analysis_id == record.analysis_id
        assert restored.source_run == record.source_run
        assert restored.population == record.population
        assert len(restored.configurations) == 1
        assert restored.configurations[0].name == "standard_window"
        assert restored.metadata == record.metadata


class TestProjectRegistry:
    """Test suite for ProjectRegistry file operations.
    
    Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#T1
    """

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a temporary project registry."""
        return ProjectRegistry(root=tmp_path)

    @pytest.fixture
    def sample_protocol(self):
        """Create a sample protocol definition."""
        return ProtocolDefinition(
            protocol_id="test_protocol",
            version="1.0",
            name="Test Protocol",
            description="A test protocol",
            stimulus={"type": "gaussian"},
            execution={"duration_ms": 500}
        )

    def test_registry_directory_structure(self, temp_registry):
        """Test that registry creates expected directory structure."""
        # Initialize should create base directories
        assert temp_registry.root.exists()
        assert temp_registry.paths["protocols"].exists()
        assert temp_registry.paths["runs"].exists()

    def test_save_and_load_protocol(self, temp_registry, sample_protocol):
        """Test saving and loading protocol definitions."""
        # Save
        temp_registry.save_protocol(sample_protocol)
        
        # Verify file exists
        protocol_file = temp_registry.paths["protocols"] / f"{sample_protocol.protocol_id}.json"
        assert protocol_file.exists()
        
        # Load
        loaded = temp_registry.load_protocol(sample_protocol.protocol_id)
        
        # Verify
        assert loaded.protocol_id == sample_protocol.protocol_id
        assert loaded.name == sample_protocol.name
        assert loaded.stimulus == sample_protocol.stimulus

    def test_save_and_load_run_record(self, temp_registry):
        """Test saving and loading protocol run records."""
        record = ProtocolRunRecord.new(
            run_id="test_run",
            protocol_id="test_protocol",
            stimulus_reference="test.pt",
            neuron_modules=[],
            tensors={"spikes": "runs/test_run/spikes.pt"}
        )
        
        # Save
        temp_registry.save_run(record)
        
        # Verify file exists
        run_file = temp_registry.paths["runs"] / f"{record.run_id}.json"
        assert run_file.exists()
        
        # Load
        loaded = temp_registry.load_run(record.run_id)
        
        # Verify
        assert loaded.run_id == record.run_id
        assert loaded.protocol_id == record.protocol_id
        assert loaded.tensors == record.tensors

    def test_list_protocols(self, temp_registry, sample_protocol):
        """Test listing all saved protocols."""
        # Save multiple protocols
        temp_registry.save_protocol(sample_protocol)
        
        protocol2 = ProtocolDefinition(
            protocol_id="test_protocol_2",
            version="1.0",
            name="Test Protocol 2",
            description="Another test",
            stimulus={},
            execution={}
        )
        temp_registry.save_protocol(protocol2)
        
        # List
        protocols = temp_registry.list_protocols()
        
        # Verify
        assert len(protocols) >= 2
        protocol_ids = [p.protocol_id for p in protocols]
        assert "test_protocol" in protocol_ids
        assert "test_protocol_2" in protocol_ids

    def test_json_roundtrip_preserves_data(self, tmp_path):
        """Test that JSON serialization preserves all data correctly."""
        record = ProtocolRunRecord.new(
            run_id="roundtrip_test",
            protocol_id="test_protocol",
            stimulus_reference="test.pt",
            neuron_modules=[
                NeuronModuleManifest(
                    name="test_neuron",
                    model="TestModel",
                    filter=None,
                    parameters={"param": 1.5},
                    file="test.pt"
                )
            ],
            tensors={"spikes": "spikes.pt"},
            metrics={"count": 100},
            notes="Test notes",
            metadata={"key": "value"}
        )
        
        # Write to JSON file
        json_path = tmp_path / "test_record.json"
        with open(json_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
        
        # Read back
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        restored = ProtocolRunRecord.from_dict(data)
        
        # Verify all fields preserved
        assert restored.run_id == record.run_id
        assert restored.tensors == record.tensors
        assert restored.metrics == record.metrics
        assert restored.notes == record.notes
        assert restored.metadata == record.metadata
        assert len(restored.neuron_modules) == 1
        assert restored.neuron_modules[0].name == "test_neuron"
