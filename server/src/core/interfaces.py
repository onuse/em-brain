"""
Core interfaces for the dynamic brain architecture.

This module defines the clean interfaces between layers, ensuring proper
separation of concerns throughout the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch


# Data structures

@dataclass
class SensorChannel:
    """Definition of a single sensor channel."""
    index: int
    name: str
    range_min: float
    range_max: float
    unit: str
    description: str


@dataclass
class MotorChannel:
    """Definition of a single motor channel."""
    index: int
    name: str
    range_min: float
    range_max: float
    unit: str
    description: str


@dataclass
class Robot:
    """Pure data class representing a robot's configuration."""
    robot_id: str
    robot_type: str
    sensory_channels: List[SensorChannel]
    motor_channels: List[MotorChannel]
    capabilities: Dict[str, Any]
    
    def get_profile_key(self) -> str:
        """Generate a unique key for this robot profile."""
        # Simple implementation - can be made more sophisticated
        return f"{self.robot_type}_{len(self.sensory_channels)}s_{len(self.motor_channels)}m"


@dataclass
class BrainSessionInfo:
    """Information about an active brain session."""
    session_id: str
    robot_id: str
    brain_dimensions: int
    created_at: float


# Interfaces

class IConnectionHandler(ABC):
    """Manages client connections and routes to appropriate services."""
    
    @abstractmethod
    def handle_handshake(self, client_id: str, capabilities: List[float]) -> List[float]:
        """Handle handshake and return response capabilities."""
        pass
    
    @abstractmethod
    def handle_sensory_input(self, client_id: str, sensory_data: List[float]) -> List[float]:
        """Process sensory input and return motor commands."""
        pass
    
    @abstractmethod
    def handle_disconnect(self, client_id: str) -> None:
        """Clean up when client disconnects."""
        pass


class IRobotRegistry(ABC):
    """Manages robot profiles and types."""
    
    @abstractmethod
    def register_robot(self, capabilities: List[float]) -> Robot:
        """Parse capabilities and create/retrieve robot profile."""
        pass
    
    @abstractmethod
    def get_robot(self, robot_id: str) -> Optional[Robot]:
        """Retrieve robot by ID."""
        pass


class IBrainService(ABC):
    """Manages brain lifecycle and sessions."""
    
    @abstractmethod
    def create_session(self, robot: Robot) -> 'IBrainSession':
        """Create a new brain session for a robot."""
        pass
    
    @abstractmethod
    def get_session_info(self, session_id: str) -> Optional[BrainSessionInfo]:
        """Get information about a session."""
        pass


class IBrainSession(ABC):
    """Handles one robot's interaction with a brain."""
    
    @abstractmethod
    def process_sensory_input(self, raw_sensory: List[float]) -> List[float]:
        """Process sensory input and return motor commands."""
        pass
    
    @abstractmethod
    def get_handshake_response(self) -> List[float]:
        """Get handshake response for this session."""
        pass
    
    @abstractmethod
    def get_session_id(self) -> str:
        """Get unique session identifier."""
        pass


class IBrainPool(ABC):
    """Manages brain instances."""
    
    @abstractmethod
    def get_brain_for_profile(self, profile_key: str) -> 'IBrain':
        """Get or create brain for a robot profile."""
        pass
    
    @abstractmethod
    def get_active_brains(self) -> Dict[str, 'IBrain']:
        """Get all active brain instances."""
        pass


class IBrainFactory(ABC):
    """Creates brain instances."""
    
    @abstractmethod
    def create(self, field_dimensions: int, spatial_resolution: int, 
               sensory_dim: Optional[int] = None, motor_dim: Optional[int] = None) -> 'IBrain':
        """Create a new brain with specified dimensions."""
        pass


class IBrain(ABC):
    """Pure brain interface - knows nothing about robots."""
    
    @abstractmethod
    def process_field_dynamics(self, field_input: torch.Tensor) -> torch.Tensor:
        """Process field dynamics and return field output."""
        pass
    
    @abstractmethod
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get brain state for persistence."""
        pass
    
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load brain state from persistence."""
        pass


class ISensoryAdapter(ABC):
    """Translates robot-specific sensors to abstract field space."""
    
    @abstractmethod
    def to_field_space(self, sensory: List[float]) -> torch.Tensor:
        """Convert robot sensory input to field coordinates."""
        pass
    
    @abstractmethod
    def get_field_dimensions(self) -> int:
        """Get expected field dimensions."""
        pass


class IMotorAdapter(ABC):
    """Translates abstract field space to robot-specific motors."""
    
    @abstractmethod
    def from_field_space(self, field_state: torch.Tensor) -> List[float]:
        """Convert field output to motor commands."""
        pass
    
    @abstractmethod
    def get_motor_dimensions(self) -> int:
        """Get expected motor dimensions."""
        pass


class IAdapterFactory(ABC):
    """Creates adapters for robot-brain translation."""
    
    @abstractmethod
    def create_sensory_adapter(self, robot: Robot, field_dimensions: int) -> ISensoryAdapter:
        """Create sensory adapter for robot."""
        pass
    
    @abstractmethod
    def create_motor_adapter(self, robot: Robot, field_dimensions: int) -> IMotorAdapter:
        """Create motor adapter for robot."""
        pass