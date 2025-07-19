"""
Brain Pattern Serializer

Extracts and restores complete brain state from vector stream architectures.
This is the core component that actually saves/loads the brain's learned knowledge.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SerializedPattern:
    """A pattern extracted from vector streams with metadata."""
    pattern_id: str
    stream_type: str  # 'sensory', 'motor', 'temporal', 'cross_stream'
    pattern_data: Dict[str, Any]
    activation_count: int
    last_accessed: float
    success_rate: float
    energy_level: float
    creation_time: float
    importance_score: float


@dataclass
class SerializedBrainState:
    """Complete serialized brain state for persistence."""
    version: str
    session_count: int
    total_cycles: int
    total_experiences: int
    save_timestamp: float
    
    # Core learned content
    patterns: List[SerializedPattern]
    confidence_state: Dict[str, Any]
    hardware_adaptations: Dict[str, Any]
    cross_stream_associations: Dict[str, Any]
    
    # Architecture metadata
    brain_type: str
    sensory_dim: int
    motor_dim: int
    temporal_dim: int
    
    # Learning trajectories
    learning_history: List[Dict[str, Any]]
    emergence_events: List[Dict[str, Any]]


class BrainSerializer:
    """Extracts and restores complete brain patterns from vector stream architectures."""
    
    def __init__(self):
        self.version = "1.0"
        self.serialization_stats = {
            'total_serializations': 0,
            'total_patterns_extracted': 0,
            'total_patterns_restored': 0,
            'avg_serialization_time_ms': 0.0,
            'avg_restoration_time_ms': 0.0
        }
    
    def serialize_brain_state(self, brain) -> SerializedBrainState:
        """Extract complete brain state for persistence."""
        start_time = time.perf_counter()
        
        try:
            # Extract patterns from vector brain
            patterns = self._extract_all_patterns(brain.vector_brain)
            
            # Extract confidence dynamics
            confidence_state = self._extract_confidence_state(brain)
            
            # Extract hardware adaptations
            hardware_adaptations = self._extract_hardware_adaptations(brain)
            
            # Extract cross-stream associations
            cross_stream_associations = self._extract_cross_stream_associations(brain)
            
            # Extract learning history
            learning_history = self._extract_learning_history(brain)
            
            # Extract emergence events
            emergence_events = self._extract_emergence_events(brain)
            
            # Create complete brain state
            brain_state = SerializedBrainState(
                version=self.version,
                session_count=getattr(brain, 'session_count', 1),
                total_cycles=brain.total_cycles,
                total_experiences=brain.total_experiences,
                save_timestamp=time.time(),
                patterns=patterns,
                confidence_state=confidence_state,
                hardware_adaptations=hardware_adaptations,
                cross_stream_associations=cross_stream_associations,
                brain_type=brain.brain_type,
                sensory_dim=brain.sensory_dim,
                motor_dim=brain.motor_dim,
                temporal_dim=brain.temporal_dim,
                learning_history=learning_history,
                emergence_events=emergence_events
            )
            
            # Update statistics
            serialization_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_serialization_stats(len(patterns), serialization_time_ms)
            
            return brain_state
            
        except Exception as e:
            print(f"⚠️ Brain serialization failed: {e}")
            raise
    
    def restore_brain_state(self, brain, brain_state: SerializedBrainState) -> bool:
        """Restore complete brain state from serialized data."""
        start_time = time.perf_counter()
        
        try:
            # Validate compatibility
            if not self._validate_compatibility(brain, brain_state):
                print("⚠️ Brain state incompatible with current brain architecture")
                return False
            
            # Restore patterns to vector brain
            restored_patterns = self._restore_patterns_to_vector_brain(
                brain.vector_brain, brain_state.patterns
            )
            
            # Restore confidence state
            self._restore_confidence_state(brain, brain_state.confidence_state)
            
            # Restore hardware adaptations
            self._restore_hardware_adaptations(brain, brain_state.hardware_adaptations)
            
            # Restore cross-stream associations
            self._restore_cross_stream_associations(brain, brain_state.cross_stream_associations)
            
            # Restore experience counters
            brain.total_experiences = brain_state.total_experiences
            
            # Update statistics
            restoration_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_restoration_stats(restored_patterns, restoration_time_ms)
            
            print(f"🧠 Brain state restored: {restored_patterns} patterns, {brain_state.total_experiences} experiences")
            return True
            
        except Exception as e:
            print(f"⚠️ Brain restoration failed: {e}")
            return False
    
    def _extract_all_patterns(self, vector_brain) -> List[SerializedPattern]:
        """Extract all patterns from vector brain streams."""
        patterns = []
        current_time = time.time()
        
        # Extract from sparse goldilocks brain if available
        if hasattr(vector_brain, 'sparse_representations'):
            patterns.extend(self._extract_sparse_patterns(vector_brain, current_time))
        
        # Extract from stream-specific storage
        for stream_name in ['sensory', 'motor', 'temporal']:
            if hasattr(vector_brain, f'{stream_name}_stream'):
                stream = getattr(vector_brain, f'{stream_name}_stream')
                patterns.extend(self._extract_stream_patterns(stream, stream_name, current_time))
        
        # Extract cross-stream patterns
        if hasattr(vector_brain, 'cross_stream_coactivation'):
            patterns.extend(self._extract_cross_stream_patterns(vector_brain, current_time))
        
        return patterns
    
    def _extract_sparse_patterns(self, vector_brain, current_time: float) -> List[SerializedPattern]:
        """Extract patterns from sparse representations system."""
        patterns = []
        
        if hasattr(vector_brain, 'sparse_representations'):
            sparse_system = vector_brain.sparse_representations
            
            # Extract active patterns
            if hasattr(sparse_system, 'active_patterns'):
                for i, pattern in enumerate(sparse_system.active_patterns):
                    if pattern is not None:
                        pattern_data = {
                            'pattern_vector': pattern.tolist() if hasattr(pattern, 'tolist') else pattern,
                            'pattern_index': i,
                            'sparsity_level': np.count_nonzero(pattern) / len(pattern) if hasattr(pattern, '__len__') else 0.02
                        }
                        
                        serialized_pattern = SerializedPattern(
                            pattern_id=f"sparse_{i}_{int(current_time)}",
                            stream_type="sparse_distributed",
                            pattern_data=pattern_data,
                            activation_count=getattr(sparse_system, 'pattern_activations', {}).get(i, 1),
                            last_accessed=current_time,
                            success_rate=0.5,  # Default for now
                            energy_level=1.0,  # Default for now
                            creation_time=current_time,
                            importance_score=0.5
                        )
                        patterns.append(serialized_pattern)
        
        return patterns
    
    def _extract_stream_patterns(self, stream, stream_name: str, current_time: float) -> List[SerializedPattern]:
        """Extract patterns from a specific vector stream."""
        patterns = []
        
        # Try different pattern storage approaches
        pattern_sources = [
            'patterns', 'stored_patterns', 'pattern_memory', 
            'learned_patterns', 'active_patterns'
        ]
        
        for source in pattern_sources:
            if hasattr(stream, source):
                source_patterns = getattr(stream, source)
                if source_patterns:
                    patterns.extend(self._convert_stream_patterns_to_serialized(
                        source_patterns, stream_name, current_time
                    ))
                break
        
        return patterns
    
    def _convert_stream_patterns_to_serialized(self, source_patterns, stream_name: str, 
                                             current_time: float) -> List[SerializedPattern]:
        """Convert stream-specific patterns to serialized format."""
        patterns = []
        
        # Handle different pattern storage formats
        if isinstance(source_patterns, dict):
            for pattern_id, pattern_data in source_patterns.items():
                patterns.append(self._create_serialized_pattern(
                    pattern_id, stream_name, pattern_data, current_time
                ))
        elif isinstance(source_patterns, list):
            for i, pattern_data in enumerate(source_patterns):
                pattern_id = f"{stream_name}_{i}_{int(current_time)}"
                patterns.append(self._create_serialized_pattern(
                    pattern_id, stream_name, pattern_data, current_time
                ))
        
        return patterns
    
    def _create_serialized_pattern(self, pattern_id: str, stream_name: str, 
                                 pattern_data: Any, current_time: float) -> SerializedPattern:
        """Create a SerializedPattern from raw pattern data."""
        # Convert numpy arrays to lists for JSON serialization
        if hasattr(pattern_data, 'tolist'):
            serializable_data = {'pattern': pattern_data.tolist()}
        elif isinstance(pattern_data, dict):
            serializable_data = self._make_serializable(pattern_data)
        else:
            serializable_data = {'pattern': pattern_data}
        
        return SerializedPattern(
            pattern_id=pattern_id,
            stream_type=stream_name,
            pattern_data=serializable_data,
            activation_count=1,
            last_accessed=current_time,
            success_rate=0.5,
            energy_level=1.0,
            creation_time=current_time,
            importance_score=0.5
        )
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data
    
    def _extract_cross_stream_patterns(self, vector_brain, current_time: float) -> List[SerializedPattern]:
        """Extract cross-stream association patterns."""
        patterns = []
        
        if hasattr(vector_brain, 'cross_stream_coactivation'):
            coactivation = vector_brain.cross_stream_coactivation
            
            # Extract coactivation patterns
            if hasattr(coactivation, 'association_patterns'):
                associations = coactivation.association_patterns
                for assoc_id, assoc_data in associations.items():
                    pattern_data = self._make_serializable(assoc_data)
                    
                    serialized_pattern = SerializedPattern(
                        pattern_id=f"cross_stream_{assoc_id}",
                        stream_type="cross_stream",
                        pattern_data=pattern_data,
                        activation_count=assoc_data.get('activation_count', 1),
                        last_accessed=current_time,
                        success_rate=assoc_data.get('success_rate', 0.5),
                        energy_level=assoc_data.get('energy_level', 1.0),
                        creation_time=current_time,
                        importance_score=assoc_data.get('importance_score', 0.5)
                    )
                    patterns.append(serialized_pattern)
        
        return patterns
    
    def _extract_confidence_state(self, brain) -> Dict[str, Any]:
        """Extract confidence system state."""
        confidence_state = {}
        
        if hasattr(brain.vector_brain, 'emergent_confidence'):
            confidence_system = brain.vector_brain.emergent_confidence
            confidence_state = {
                'current_confidence': getattr(confidence_system, 'current_confidence', 0.7),
                'confidence_history': getattr(confidence_system, 'confidence_history', []),
                'total_updates': getattr(confidence_system, 'total_updates', 0),
                'volatility_history': getattr(confidence_system, 'volatility_history', []),
                'coherence_history': getattr(confidence_system, 'coherence_history', [])
            }
        
        return confidence_state
    
    def _extract_hardware_adaptations(self, brain) -> Dict[str, Any]:
        """Extract hardware adaptation state."""
        hardware_state = {}
        
        if hasattr(brain, 'hardware_adaptation'):
            adapter = brain.hardware_adaptation
            hardware_state = {
                'working_memory_limit': getattr(adapter, 'working_memory_limit', 671),
                'similarity_search_limit': getattr(adapter, 'similarity_search_limit', 16777),
                'cognitive_energy_budget': getattr(adapter, 'cognitive_energy_budget', 20800),
                'cycle_time_history': getattr(adapter, 'cycle_time_history', []),
                'adaptation_events': getattr(adapter, 'adaptation_events', [])
            }
        
        return hardware_state
    
    def _extract_cross_stream_associations(self, brain) -> Dict[str, Any]:
        """Extract cross-stream association data."""
        associations = {}
        
        if hasattr(brain.vector_brain, 'cross_stream_coactivation'):
            coactivation = brain.vector_brain.cross_stream_coactivation
            associations = {
                'stream_connections': getattr(coactivation, 'stream_connections', {}),
                'association_strength': getattr(coactivation, 'association_strength', {}),
                'coactivation_history': getattr(coactivation, 'coactivation_history', [])
            }
        
        return associations
    
    def _extract_learning_history(self, brain) -> List[Dict[str, Any]]:
        """Extract learning trajectory history."""
        history = []
        
        if hasattr(brain, 'recent_learning_outcomes'):
            for outcome in brain.recent_learning_outcomes[-100:]:  # Last 100 events
                if isinstance(outcome, dict):
                    history.append(outcome)
        
        return history
    
    def _extract_emergence_events(self, brain) -> List[Dict[str, Any]]:
        """Extract emergence event history."""
        events = []
        
        # This would be extracted from logger if available
        if hasattr(brain, 'logger') and brain.logger:
            # Extract from logger's emergence events if available
            pass
        
        return events
    
    def _validate_compatibility(self, brain, brain_state: SerializedBrainState) -> bool:
        """Validate that brain state is compatible with current brain."""
        # For fresh states with no patterns, allow dimensional mismatches
        # (this happens when default dimensions don't match config)
        if len(brain_state.patterns) == 0 and brain_state.total_experiences == 0:
            # Fresh state - only check brain type
            return brain.brain_type == brain_state.brain_type
        
        # For actual saved states, require exact match
        return (
            brain.brain_type == brain_state.brain_type and
            brain.sensory_dim == brain_state.sensory_dim and
            brain.motor_dim == brain_state.motor_dim and
            brain.temporal_dim == brain_state.temporal_dim
        )
    
    def _restore_patterns_to_vector_brain(self, vector_brain, patterns: List[SerializedPattern]) -> int:
        """Restore patterns to vector brain streams."""
        restored_count = 0
        
        # Group patterns by stream type
        patterns_by_stream = {}
        for pattern in patterns:
            stream_type = pattern.stream_type
            if stream_type not in patterns_by_stream:
                patterns_by_stream[stream_type] = []
            patterns_by_stream[stream_type].append(pattern)
        
        # Restore patterns to appropriate streams
        for stream_type, stream_patterns in patterns_by_stream.items():
            if stream_type == "sparse_distributed":
                restored_count += self._restore_sparse_patterns(vector_brain, stream_patterns)
            elif stream_type in ["sensory", "motor", "temporal"]:
                restored_count += self._restore_stream_patterns(vector_brain, stream_type, stream_patterns)
            elif stream_type == "cross_stream":
                restored_count += self._restore_cross_stream_patterns(vector_brain, stream_patterns)
        
        return restored_count
    
    def _restore_sparse_patterns(self, vector_brain, patterns: List[SerializedPattern]) -> int:
        """Restore patterns to sparse representations system."""
        restored_count = 0
        
        if hasattr(vector_brain, 'sparse_representations'):
            sparse_system = vector_brain.sparse_representations
            
            # Initialize pattern storage if needed
            if not hasattr(sparse_system, 'active_patterns'):
                sparse_system.active_patterns = []
            
            for pattern in patterns:
                try:
                    pattern_vector = np.array(pattern.pattern_data['pattern'])
                    sparse_system.active_patterns.append(pattern_vector)
                    restored_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to restore sparse pattern {pattern.pattern_id}: {e}")
        
        return restored_count
    
    def _restore_stream_patterns(self, vector_brain, stream_type: str, 
                                patterns: List[SerializedPattern]) -> int:
        """Restore patterns to a specific vector stream."""
        restored_count = 0
        
        stream_attr = f"{stream_type}_stream"
        if hasattr(vector_brain, stream_attr):
            stream = getattr(vector_brain, stream_attr)
            
            # Initialize pattern storage if needed
            if not hasattr(stream, 'patterns'):
                stream.patterns = {}
            
            for pattern in patterns:
                try:
                    stream.patterns[pattern.pattern_id] = pattern.pattern_data
                    restored_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to restore {stream_type} pattern {pattern.pattern_id}: {e}")
        
        return restored_count
    
    def _restore_cross_stream_patterns(self, vector_brain, patterns: List[SerializedPattern]) -> int:
        """Restore cross-stream association patterns."""
        restored_count = 0
        
        if hasattr(vector_brain, 'cross_stream_coactivation'):
            coactivation = vector_brain.cross_stream_coactivation
            
            # Initialize association storage if needed
            if not hasattr(coactivation, 'association_patterns'):
                coactivation.association_patterns = {}
            
            for pattern in patterns:
                try:
                    coactivation.association_patterns[pattern.pattern_id] = pattern.pattern_data
                    restored_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to restore cross-stream pattern {pattern.pattern_id}: {e}")
        
        return restored_count
    
    def _restore_confidence_state(self, brain, confidence_state: Dict[str, Any]):
        """Restore confidence system state."""
        if hasattr(brain.vector_brain, 'emergent_confidence') and confidence_state:
            confidence_system = brain.vector_brain.emergent_confidence
            
            for attr, value in confidence_state.items():
                if hasattr(confidence_system, attr):
                    setattr(confidence_system, attr, value)
    
    def _restore_hardware_adaptations(self, brain, hardware_state: Dict[str, Any]):
        """Restore hardware adaptation state."""
        if hasattr(brain, 'hardware_adaptation') and hardware_state:
            adapter = brain.hardware_adaptation
            
            for attr, value in hardware_state.items():
                if hasattr(adapter, attr):
                    setattr(adapter, attr, value)
    
    def _restore_cross_stream_associations(self, brain, associations: Dict[str, Any]):
        """Restore cross-stream association data."""
        if hasattr(brain.vector_brain, 'cross_stream_coactivation') and associations:
            coactivation = brain.vector_brain.cross_stream_coactivation
            
            for attr, value in associations.items():
                if hasattr(coactivation, attr):
                    setattr(coactivation, attr, value)
    
    def _update_serialization_stats(self, pattern_count: int, time_ms: float):
        """Update serialization performance statistics."""
        stats = self.serialization_stats
        stats['total_serializations'] += 1
        stats['total_patterns_extracted'] += pattern_count
        
        # Update average time
        total_ops = stats['total_serializations']
        stats['avg_serialization_time_ms'] = (
            (stats['avg_serialization_time_ms'] * (total_ops - 1) + time_ms) / total_ops
        )
    
    def _update_restoration_stats(self, pattern_count: int, time_ms: float):
        """Update restoration performance statistics."""
        stats = self.serialization_stats
        stats['total_patterns_restored'] += pattern_count
        
        # Update average restoration time (approximate)
        if stats['total_serializations'] > 0:
            stats['avg_restoration_time_ms'] = (
                (stats['avg_restoration_time_ms'] + time_ms) / 2
            )
        else:
            stats['avg_restoration_time_ms'] = time_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serialization performance statistics."""
        return self.serialization_stats.copy()
    
    def to_dict(self, brain_state: SerializedBrainState) -> Dict[str, Any]:
        """Convert SerializedBrainState to dictionary for JSON serialization."""
        state_dict = asdict(brain_state)
        
        # Ensure all pattern data is serializable
        for pattern_dict in state_dict['patterns']:
            pattern_dict['pattern_data'] = self._make_serializable(pattern_dict['pattern_data'])
        
        return state_dict
    
    def from_dict(self, state_dict: Dict[str, Any]) -> SerializedBrainState:
        """Create SerializedBrainState from dictionary."""
        try:
            # Convert pattern dictionaries to SerializedPattern objects
            patterns = []
            patterns_data = state_dict.get('patterns', [])
            
            if patterns_data is None:
                patterns_data = []
            
            for pattern_dict in patterns_data:
                if isinstance(pattern_dict, dict):
                    patterns.append(SerializedPattern(**pattern_dict))
                else:
                    print(f"⚠️ Skipping invalid pattern data: {type(pattern_dict)}")
            
            # Define valid fields for SerializedBrainState to filter out old/invalid keys
            valid_fields = {
                'version', 'session_count', 'total_cycles', 'total_experiences', 
                'save_timestamp', 'patterns', 'confidence_state', 'hardware_adaptations',
                'cross_stream_associations', 'brain_type', 'sensory_dim', 'motor_dim',
                'temporal_dim', 'learning_history', 'emergence_events'
            }
            
            # Filter state_dict to only include valid fields (backward compatibility)
            brain_state_dict = {}
            for key, value in state_dict.items():
                if key in valid_fields:
                    brain_state_dict[key] = value
                else:
                    print(f"⚠️ Skipping unknown field '{key}' from saved data (backward compatibility)")
            
            # Set patterns from processed data
            brain_state_dict['patterns'] = patterns
            
            # Ensure required fields exist with defaults
            brain_state_dict.setdefault('version', '1.0')
            brain_state_dict.setdefault('session_count', 1)
            brain_state_dict.setdefault('total_cycles', 0)
            brain_state_dict.setdefault('total_experiences', 0)
            brain_state_dict.setdefault('save_timestamp', time.time())
            brain_state_dict.setdefault('confidence_state', {})
            brain_state_dict.setdefault('hardware_adaptations', {})
            brain_state_dict.setdefault('cross_stream_associations', {})
            brain_state_dict.setdefault('brain_type', 'sparse_goldilocks')
            brain_state_dict.setdefault('sensory_dim', 16)
            brain_state_dict.setdefault('motor_dim', 4)
            brain_state_dict.setdefault('temporal_dim', 4)
            brain_state_dict.setdefault('learning_history', [])
            brain_state_dict.setdefault('emergence_events', [])
            
            return SerializedBrainState(**brain_state_dict)
            
        except Exception as e:
            print(f"❌ Error in brain_serializer.from_dict: {e}")
            print(f"   State dict keys: {list(state_dict.keys()) if state_dict else 'None'}")
            if state_dict and 'patterns' in state_dict:
                print(f"   Patterns type: {type(state_dict['patterns'])}")
                print(f"   Patterns length: {len(state_dict['patterns']) if hasattr(state_dict['patterns'], '__len__') else 'No length'}")
            raise