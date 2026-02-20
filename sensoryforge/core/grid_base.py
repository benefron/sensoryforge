"""Base class for receptor grid implementations.

This module provides the abstract base class for all grid types
(ReceptorGrid, CompositeReceptorGrid, etc.) to ensure consistent
interfaces and enable registry-based instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import torch


class BaseGrid(ABC):
    """Abstract base class for receptor grid implementations.
    
    All grid implementations must:
    1. Provide get_coordinates() or get_all_coordinates() methods
    2. Support from_config() class method for YAML instantiation
    3. Provide to_dict() method for serialization
    4. Expose xlim, ylim spatial bounds
    
    Attributes:
        xlim: Spatial bounds along x-axis (min, max) in mm.
        ylim: Spatial bounds along y-axis (min, max) in mm.
        device: PyTorch device for tensor storage.
    """
    
    def __init__(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize grid with spatial bounds.
        
        Args:
            xlim: Spatial bounds along x-axis (min, max) in mm.
            ylim: Spatial bounds along y-axis (min, max) in mm.
            device: PyTorch device identifier.
        """
        self.xlim = xlim
        self.ylim = ylim
        self.device = torch.device(device) if isinstance(device, str) else device
    
    @abstractmethod
    def get_all_coordinates(self) -> torch.Tensor:
        """Get all receptor coordinates.
        
        Returns:
            Tensor [N_receptors, 2] with (x, y) positions in mm.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseGrid":
        """Create grid instance from config dict.
        
        Args:
            config: Dictionary with grid configuration parameters.
        
        Returns:
            BaseGrid instance.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid parameters to dict.
        
        Returns:
            Dictionary with grid type and parameters.
        """
        return {
            "type": self.__class__.__name__.lower().replace("grid", ""),
            "xlim": list(self.xlim),
            "ylim": list(self.ylim),
            "device": str(self.device),
        }
