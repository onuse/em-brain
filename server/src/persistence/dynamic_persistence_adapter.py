"""
Dynamic Persistence Adapter

Adapts the existing persistence system to work with the new dynamic brain architecture.
This bridges the gap between the old BrainFactory-based persistence and the new
DynamicBrainFactory/DynamicUnifiedFieldBrain architecture.
"""

import time
import torch
import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..core.interfaces import IBrain


@dataclass
class DynamicBrainState:
    """Extended brain state for dynamic unified field brain persistence."""
    # Core identification
    brain_type: str = "dynamic_unified_field"
    version: str = "2.0"
    
    # Dynamic architecture info
    conceptual_dimensions: int = 0
    tensor_shape: List[int] = field(default_factory=list)
    dimension_mapping: Dict[str, Any] = field(default_factory=dict)
    
    # Field state
    unified_field: Optional[np.ndarray] = None  # Numpy array for JSON serialization
    field_energy: float = 0.0
    
    # Brain state
    brain_cycles: int = 0
    total_cycles: int = 0
    
    # Memory and learning
    topology_regions: Dict[str, Dict] = field(default_factory=dict)
    active_constraints: List[Dict] = field(default_factory=list)
    working_memory: List[Any] = field(default_factory=list)
    
    # Prediction and confidence
    prediction_confidence: float = 0.5
    prediction_confidence_history: List[float] = field(default_factory=list)
    
    # Spontaneous dynamics state
    spontaneous_enabled: bool = True
    last_imprint_strength: float = 0.0
    
    # Blended reality state
    blended_reality_state: Dict[str, Any] = field(default_factory=dict)
    
    # Session tracking
    session_count: int = 0
    total_experiences: int = 0
    timestamp: float = field(default_factory=time.time)


class DynamicPersistenceAdapter:
    """
    Adapter to make the persistence system work with dynamic brains.
    
    This handles:
    - Extracting comprehensive state from DynamicUnifiedFieldBrain
    - Serializing high-dimensional tensors efficiently
    - Restoring state to newly created brains
    - Managing backward compatibility
    """
    
    def __init__(self, compression_enabled: bool = True):
        self.compression_enabled = compression_enabled
        
    def extract_brain_state(self, brain_wrapper: IBrain) -> DynamicBrainState:
        """Extract comprehensive state from a dynamic brain."""
        
        # Get the actual brain instance
        brain = brain_wrapper.brain if hasattr(brain_wrapper, 'brain') else brain_wrapper
        
        # Create state object
        state = DynamicBrainState()
        
        # Basic info
        state.brain_cycles = brain.brain_cycles
        state.total_cycles = brain_wrapper.total_cycles if hasattr(brain_wrapper, 'total_cycles') else 0
        
        # Dynamic architecture info
        state.conceptual_dimensions = brain.total_dimensions
        state.tensor_shape = list(brain.tensor_shape)
        if hasattr(brain, 'dimension_mapping'):
            # Convert dimension mapping to serializable format
            # Skip complex objects that can't be serialized
            state.dimension_mapping = {
                'conceptual_to_tensor': brain.dimension_mapping.get('conceptual_to_tensor', {}),
                'tensor_to_conceptual': brain.dimension_mapping.get('tensor_to_conceptual', {})
            }
        
        # Field state (convert to numpy for serialization)
        state.unified_field = brain.unified_field.cpu().numpy()
        state.field_energy = float(torch.mean(torch.abs(brain.unified_field)))
        print(f"💾 Saving unified field: shape {state.unified_field.shape}, mean energy {state.field_energy:.6f}")
        
        # Memory and topology
        if hasattr(brain, 'topology_regions'):
            # Limit to most recent/important regions
            state.topology_regions = dict(list(brain.topology_regions.items())[-100:])
        
        # Constraints
        if hasattr(brain, 'constraint_field') and hasattr(brain.constraint_field, 'active_constraints'):
            # Serialize constraint info (not the full constraint objects)
            state.active_constraints = [
                {
                    'id': f"constraint_{i}",  # Generate ID since constraints don't have one
                    'type': c.constraint_type.value if hasattr(c.constraint_type, 'value') else str(c.constraint_type),
                    'strength': float(c.strength) if hasattr(c, 'strength') else 0.5,
                    'dimensions': c.dimensions if hasattr(c, 'dimensions') else [],
                    'discovery_timestamp': float(c.discovery_timestamp) if hasattr(c, 'discovery_timestamp') else 0.0
                }
                for i, c in enumerate(brain.constraint_field.active_constraints[:50])  # Limit to top 50
            ]
        
        # Working memory
        if hasattr(brain, 'working_memory'):
            # Serialize recent experiences
            state.working_memory = []
            for exp in list(brain.working_memory)[-20:]:  # Last 20 items
                if hasattr(exp, 'experience_id'):
                    state.working_memory.append({
                        'id': exp.experience_id,
                        'timestamp': exp.timestamp,
                        'field_intensity': float(exp.field_intensity)
                    })
        
        # Prediction confidence
        if hasattr(brain, '_current_prediction_confidence'):
            state.prediction_confidence = brain._current_prediction_confidence
        if hasattr(brain, '_prediction_confidence_history'):
            state.prediction_confidence_history = list(brain._prediction_confidence_history)[-100:]
        
        # Spontaneous dynamics
        if hasattr(brain, 'spontaneous_enabled'):
            state.spontaneous_enabled = brain.spontaneous_enabled
        if hasattr(brain, '_last_imprint_strength'):
            state.last_imprint_strength = brain._last_imprint_strength
        
        # Blended reality
        if hasattr(brain, 'blended_reality'):
            state.blended_reality_state = brain.blended_reality.get_blend_state()
        
        # Count total experiences
        if hasattr(brain, 'field_experiences'):
            state.total_experiences = len(brain.field_experiences)
        
        return state
    
    def restore_brain_state(self, brain_wrapper: IBrain, state: DynamicBrainState) -> bool:
        """Restore state to a dynamic brain."""
        
        try:
            # Get the actual brain instance
            brain = brain_wrapper.brain if hasattr(brain_wrapper, 'brain') else brain_wrapper
            
            # Restore basic counters
            brain.brain_cycles = state.brain_cycles
            if hasattr(brain_wrapper, 'total_cycles'):
                brain_wrapper.total_cycles = state.total_cycles
            
            # Restore unified field
            if state.unified_field is not None:
                # Convert numpy back to tensor
                field_tensor = torch.tensor(state.unified_field, dtype=torch.float32)
                
                # Verify shape matches
                if field_tensor.shape == brain.unified_field.shape:
                    brain.unified_field = field_tensor.to(brain.device)
                    print(f"✅ Restored unified field: shape {field_tensor.shape}, mean energy {torch.mean(torch.abs(field_tensor)).item():.6f}")
                else:
                    print(f"⚠️ Field shape mismatch: saved {field_tensor.shape} vs current {brain.unified_field.shape}")
                    # Could implement shape adaptation here if needed
            else:
                print("⚠️ No unified field data in saved state")
            
            # Restore topology regions
            if hasattr(brain, 'topology_regions') and state.topology_regions:
                brain.topology_regions.update(state.topology_regions)
            
            # Restore working memory (just the count, not full experiences)
            if hasattr(brain, 'field_experiences') and state.total_experiences > 0:
                # This gives the brain awareness of its history even if we don't restore all experiences
                print(f"📝 Brain has {state.total_experiences} historical experiences")
            
            # Restore prediction confidence
            if hasattr(brain, '_current_prediction_confidence'):
                brain._current_prediction_confidence = state.prediction_confidence
            if hasattr(brain, '_prediction_confidence_history') and state.prediction_confidence_history:
                brain._prediction_confidence_history.extend(state.prediction_confidence_history[-50:])
            
            # Restore spontaneous dynamics state
            if hasattr(brain, '_last_imprint_strength'):
                brain._last_imprint_strength = state.last_imprint_strength
            
            # Restore blended reality state
            if hasattr(brain, 'blended_reality') and state.blended_reality_state:
                if 'smoothed_confidence' in state.blended_reality_state:
                    brain.blended_reality._smoothed_confidence = state.blended_reality_state['smoothed_confidence']
                if 'cycles_without_input' in state.blended_reality_state:
                    brain.blended_reality._cycles_without_input = state.blended_reality_state['cycles_without_input']
            
            print(f"✅ Restored brain state: {state.brain_cycles} cycles, {len(state.topology_regions)} memory regions")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore brain state: {e}")
            return False
    
    def serialize_to_dict(self, state: DynamicBrainState) -> Dict[str, Any]:
        """Convert brain state to dictionary for JSON serialization."""
        
        result = {
            'brain_type': state.brain_type,
            'version': state.version,
            'conceptual_dimensions': state.conceptual_dimensions,
            'tensor_shape': state.tensor_shape,
            'dimension_mapping': state.dimension_mapping,
            'brain_cycles': state.brain_cycles,
            'total_cycles': state.total_cycles,
            'field_energy': state.field_energy,
            'topology_regions': state.topology_regions,
            'active_constraints': state.active_constraints,
            'working_memory': state.working_memory,
            'prediction_confidence': state.prediction_confidence,
            'prediction_confidence_history': state.prediction_confidence_history,
            'spontaneous_enabled': state.spontaneous_enabled,
            'last_imprint_strength': state.last_imprint_strength,
            'blended_reality_state': state.blended_reality_state,
            'session_count': state.session_count,
            'total_experiences': state.total_experiences,
            'timestamp': state.timestamp
        }
        
        # Handle unified field separately due to size
        if state.unified_field is not None:
            if self.compression_enabled:
                # Could implement compression here
                result['unified_field_shape'] = state.unified_field.shape
                result['unified_field_compressed'] = None  # Binary persistence handles compression
            else:
                # Convert to nested lists for JSON
                result['unified_field'] = state.unified_field.tolist()
        
        return result
    
    def deserialize_from_dict(self, data: Dict[str, Any]) -> DynamicBrainState:
        """Create brain state from dictionary."""
        
        state = DynamicBrainState()
        
        # Copy all basic fields
        for key in ['brain_type', 'version', 'conceptual_dimensions', 'tensor_shape',
                    'dimension_mapping', 'brain_cycles', 'total_cycles', 'field_energy',
                    'topology_regions', 'active_constraints', 'working_memory',
                    'prediction_confidence', 'prediction_confidence_history',
                    'spontaneous_enabled', 'last_imprint_strength', 'blended_reality_state',
                    'session_count', 'total_experiences', 'timestamp']:
            if key in data:
                setattr(state, key, data[key])
        
        # Handle unified field
        if 'unified_field' in data:
            state.unified_field = np.array(data['unified_field'])
        elif 'unified_field_compressed' in data:
            # Binary persistence handles decompression
            pass
        
        return state