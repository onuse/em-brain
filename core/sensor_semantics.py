"""
Sensor Semantics Interface - Defines how brainstem communicates sensor meanings to brain.

This provides a clean interface for the brainstem to describe what sensors are available
and what they can be used for, allowing drives to discover relevant sensors naturally.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum


class SensorModality(Enum):
    """Different types of sensor modalities."""
    DISTANCE = "distance"
    VISION = "vision"
    CHEMICAL = "chemical"  # smell, taste
    INTERNAL = "internal"  # health, energy, orientation
    TEMPORAL = "temporal"  # time-based sensors
    TACTILE = "tactile"    # touch, pressure
    AUDITORY = "auditory"  # sound
    

class SensorPurpose(Enum):
    """What the sensor can be used for."""
    NAVIGATION = "navigation"
    SURVIVAL = "survival"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    ENERGY_MANAGEMENT = "energy_management"
    THREAT_DETECTION = "threat_detection"
    FOOD_DETECTION = "food_detection"
    SELF_STATE = "self_state"


@dataclass
class SensorDescriptor:
    """Describes a sensor's characteristics and potential uses."""
    index: int
    name: str
    modality: SensorModality
    purposes: Set[SensorPurpose]
    value_range: tuple  # (min, max) expected values
    description: str
    is_critical: bool = False  # Critical for survival
    change_frequency: str = "medium"  # "low", "medium", "high"
    

class SensorSemantics:
    """
    Interface for brainstem to communicate sensor meanings to brain.
    
    This allows drives to discover which sensors are relevant to their purposes
    without hardcoding sensor indices.
    """
    
    def __init__(self):
        self.sensor_descriptors: Dict[int, SensorDescriptor] = {}
        self.purpose_to_sensors: Dict[SensorPurpose, List[int]] = {}
        self.modality_to_sensors: Dict[SensorModality, List[int]] = {}
        
    def register_sensor(self, descriptor: SensorDescriptor):
        """Register a sensor with its semantic description."""
        self.sensor_descriptors[descriptor.index] = descriptor
        
        # Update purpose mappings
        for purpose in descriptor.purposes:
            if purpose not in self.purpose_to_sensors:
                self.purpose_to_sensors[purpose] = []
            self.purpose_to_sensors[purpose].append(descriptor.index)
        
        # Update modality mappings
        if descriptor.modality not in self.modality_to_sensors:
            self.modality_to_sensors[descriptor.modality] = []
        self.modality_to_sensors[descriptor.modality].append(descriptor.index)
    
    def get_sensors_for_purpose(self, purpose: SensorPurpose) -> List[int]:
        """Get sensor indices that serve a specific purpose."""
        return self.purpose_to_sensors.get(purpose, [])
    
    def get_sensors_for_modality(self, modality: SensorModality) -> List[int]:
        """Get sensor indices of a specific modality."""
        return self.modality_to_sensors.get(modality, [])
    
    def get_critical_sensors(self) -> List[int]:
        """Get indices of sensors critical for survival."""
        return [idx for idx, desc in self.sensor_descriptors.items() if desc.is_critical]
    
    def get_sensor_descriptor(self, index: int) -> Optional[SensorDescriptor]:
        """Get descriptor for a specific sensor index."""
        return self.sensor_descriptors.get(index)
    
    def get_all_sensors(self) -> Dict[int, SensorDescriptor]:
        """Get all registered sensor descriptors."""
        return self.sensor_descriptors.copy()
    
    def has_sensor_for_purpose(self, purpose: SensorPurpose) -> bool:
        """Check if any sensors serve a specific purpose."""
        return purpose in self.purpose_to_sensors and len(self.purpose_to_sensors[purpose]) > 0
    
    def get_sensor_count(self) -> int:
        """Get total number of registered sensors."""
        return len(self.sensor_descriptors)
    
    def get_discovery_priority_sensors(self) -> List[int]:
        """
        Get sensors that should be explored first for discovery.
        
        Returns sensors in priority order:
        1. Critical sensors (survival)
        2. High-change frequency sensors (dynamic environment)
        3. Self-state sensors (understanding self)
        4. Others
        """
        critical = [idx for idx, desc in self.sensor_descriptors.items() if desc.is_critical]
        high_change = [idx for idx, desc in self.sensor_descriptors.items() 
                      if desc.change_frequency == "high" and idx not in critical]
        self_state = [idx for idx, desc in self.sensor_descriptors.items() 
                     if SensorPurpose.SELF_STATE in desc.purposes and idx not in critical]
        others = [idx for idx in self.sensor_descriptors.keys() 
                 if idx not in critical and idx not in high_change and idx not in self_state]
        
        return critical + high_change + self_state + others