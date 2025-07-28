"""
Simplified Brain Serializer - Unified Field Brain Only

This serializer handles persistence for the UnifiedFieldBrain only.
All legacy brain type complexity has been removed.
"""

import time
import torch
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SerializedBrainState:
    """Simplified brain state for UnifiedFieldBrain only."""
    brain_type: str
    field_dimensions: int
    brain_cycles: int
    total_factory_cycles: int
    field_parameters: Dict[str, Any]
    timestamp: float
    version: str = "1.0"


class BrainSerializer:
    """Simplified serializer for UnifiedFieldBrain only."""
    
    def __init__(self, compression_enabled: bool = True):
        self.compression_enabled = compression_enabled
    
    def serialize_brain_state(self, brain_factory) -> SerializedBrainState:
        """Extract brain state for persistence - simplified for UnifiedFieldBrain only."""
        start_time = time.perf_counter()
        
        try:
            # Get state from the simplified factory method
            state_data = brain_factory.get_brain_state_for_persistence()
            
            # Create serialized state
            brain_state = SerializedBrainState(
                brain_type=state_data['brain_type'],
                field_dimensions=state_data['field_dimensions'],
                brain_cycles=state_data['brain_cycles'],
                total_factory_cycles=state_data['total_factory_cycles'],
                field_parameters=state_data['field_parameters'],
                timestamp=time.time()
            )
            
            duration = time.perf_counter() - start_time
            print(f"✅ Brain serialization completed in {duration*1000:.1f}ms")
            
            return brain_state
            
        except Exception as e:
            print(f"⚠️ Brain serialization failed: {e}")
            raise
    
    def restore_brain_state(self, brain_factory, brain_state: SerializedBrainState) -> bool:
        """Restore brain state - simplified for UnifiedFieldBrain only."""
        start_time = time.perf_counter()
        
        try:
            # Validate compatibility
            if brain_state.brain_type != 'unified_field':
                print(f"⚠️ Cannot restore non-unified brain state: {brain_state.brain_type}")
                return False
            
            # Convert to dict format expected by factory
            state_dict = {
                'brain_type': brain_state.brain_type,
                'field_dimensions': brain_state.field_dimensions,
                'brain_cycles': brain_state.brain_cycles,
                'total_factory_cycles': brain_state.total_factory_cycles,
                'field_parameters': brain_state.field_parameters,
            }
            
            # Restore using factory method
            success = brain_factory.restore_brain_state(state_dict)
            
            if success:
                duration = time.perf_counter() - start_time
                print(f"✅ Brain restoration completed in {duration*1000:.1f}ms")
            
            return success
            
        except Exception as e:
            print(f"⚠️ Brain restoration failed: {e}")
            return False
    
    def serialize_to_dict(self, brain_state: SerializedBrainState) -> Dict[str, Any]:
        """Convert brain state to dictionary for JSON serialization."""
        return {
            'brain_type': brain_state.brain_type,
            'field_dimensions': brain_state.field_dimensions,
            'brain_cycles': brain_state.brain_cycles,
            'total_factory_cycles': brain_state.total_factory_cycles,
            'field_parameters': brain_state.field_parameters,
            'timestamp': brain_state.timestamp,
            'version': brain_state.version
        }
    
    def deserialize_from_dict(self, data: Dict[str, Any]) -> SerializedBrainState:
        """Create brain state from dictionary."""
        return SerializedBrainState(
            brain_type=data['brain_type'],
            field_dimensions=data['field_dimensions'],
            brain_cycles=data['brain_cycles'],
            total_factory_cycles=data['total_factory_cycles'],
            field_parameters=data['field_parameters'],
            timestamp=data['timestamp'],
            version=data.get('version', '1.0')
        )
    
    def get_serialization_info(self, brain_factory) -> Dict[str, Any]:
        """Get information about what will be serialized."""
        return {
            'serializable_components': [
                'field_dimensions',
                'brain_cycles', 
                'field_parameters',
                'factory_cycles'
            ],
            'field_dimensions': brain_factory.brain.total_dimensions,
            'brain_cycles': brain_factory.brain.brain_cycles,
            'note': 'Field tensor not serialized - too large, reconstructed during operation'
        }