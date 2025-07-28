#!/usr/bin/env python3
"""
Constraint Propagation System

Implements Phase 7c: Enhanced cross-stream constraint propagation where constraints
from one stream naturally affect processing in other streams, creating emergent coordination.

Key principles:
- Constraints propagate through biological mechanisms (not explicit rules)
- Processing load in one stream affects resource allocation in others
- Urgency signals cascade across streams
- Interference patterns create natural inhibition
- All coordination emerges from constraint physics, not programming
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .stream_types import StreamType, ConstraintType, StreamState


@dataclass
class PropagatedConstraint:
    """A constraint that propagates from one stream to others."""
    constraint_type: ConstraintType
    source_stream: StreamType
    intensity: float  # 0.0-1.0 constraint strength
    decay_rate: float  # How quickly constraint weakens over time
    creation_time: float
    propagation_pattern: Dict[StreamType, float]  # How much affects each stream
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstraintPropagationSystem:
    """
    System for propagating constraints between streams to create emergent coordination.
    
    This implements biological-like constraint propagation where processing challenges
    in one stream naturally create adaptive pressure in other streams.
    """
    
    def __init__(self, biological_oscillator=None, quiet_mode: bool = False):
        """
        Initialize constraint propagation system.
        
        Args:
            biological_oscillator: Biological timing for natural propagation rhythms
            quiet_mode: Suppress debug output
        """
        self.biological_oscillator = biological_oscillator
        self.quiet_mode = quiet_mode
        
        # Active propagated constraints
        self._active_constraints: List[PropagatedConstraint] = []
        
        # Constraint propagation matrix (how constraints spread between streams)
        self._propagation_matrix = self._initialize_propagation_matrix()
        
        # Stream sensitivity to different constraint types
        self._stream_sensitivities = self._initialize_stream_sensitivities()
        
        # Constraint decay dynamics (biological memory of constraints)
        self._decay_dynamics = {
            ConstraintType.PROCESSING_LOAD: 0.95,      # Fast decay - immediate pressure
            ConstraintType.RESOURCE_SCARCITY: 0.90,    # Medium decay - resource competition
            ConstraintType.URGENCY_SIGNAL: 0.80,       # Faster decay - urgency fades
            ConstraintType.INTERFERENCE: 0.92,         # Slow decay - interference persists
            ConstraintType.COHERENCE_PRESSURE: 0.85,   # Medium decay - coherence needs
            ConstraintType.ENERGY_DEPLETION: 0.88      # Medium decay - energy recovery
        }
        
        # Historical constraint tracking for adaptation
        self._constraint_history = defaultdict(list)
        
        # Emergent propagation patterns (learned from constraint interactions)
        self._emergent_patterns = {}
        
        if not quiet_mode:
            print(f"🔗 ConstraintPropagationSystem initialized")
            print(f"   Constraint types: {len(ConstraintType)}")
            print(f"   Stream interactions: {len(self._propagation_matrix)}")
    
    def _initialize_propagation_matrix(self) -> Dict[StreamType, Dict[StreamType, float]]:
        """Initialize how constraints propagate between streams (biological connectivity)."""
        # Based on biological neural connectivity patterns
        matrix = {}
        
        for source in StreamType:
            matrix[source] = {}
            for target in StreamType:
                if source == target:
                    matrix[source][target] = 1.0  # Self-constraint always propagates
                else:
                    # Biological connectivity strengths (based on neural anatomy)
                    connectivity = self._get_biological_connectivity(source, target)
                    matrix[source][target] = connectivity
        
        return matrix
    
    def _get_biological_connectivity(self, source: StreamType, target: StreamType) -> float:
        """Get biological connectivity strength between stream types."""
        # Based on neuroanatomical connection strengths
        connectivity_map = {
            # Sensory -> Others
            (StreamType.SENSORY, StreamType.MOTOR): 0.8,      # Strong sensorimotor links
            (StreamType.SENSORY, StreamType.TEMPORAL): 0.6,   # Sensory timing integration
            (StreamType.SENSORY, StreamType.CONFIDENCE): 0.7, # Sensory confidence signals
            (StreamType.SENSORY, StreamType.ATTENTION): 0.9,  # Sensory captures attention
            
            # Motor -> Others  
            (StreamType.MOTOR, StreamType.SENSORY): 0.7,      # Motor-sensory feedback
            (StreamType.MOTOR, StreamType.TEMPORAL): 0.8,     # Motor timing critical
            (StreamType.MOTOR, StreamType.CONFIDENCE): 0.6,   # Motor execution confidence
            (StreamType.MOTOR, StreamType.ATTENTION): 0.5,    # Motor needs attention
            
            # Temporal -> Others
            (StreamType.TEMPORAL, StreamType.SENSORY): 0.5,   # Temporal context to sensory
            (StreamType.TEMPORAL, StreamType.MOTOR): 0.7,     # Timing affects motor control
            (StreamType.TEMPORAL, StreamType.CONFIDENCE): 0.8, # Timing affects confidence
            (StreamType.TEMPORAL, StreamType.ATTENTION): 0.6,  # Temporal focus attention
            
            # Confidence -> Others
            (StreamType.CONFIDENCE, StreamType.SENSORY): 0.4, # Confidence modulates sensing
            (StreamType.CONFIDENCE, StreamType.MOTOR): 0.6,   # Confidence affects action
            (StreamType.CONFIDENCE, StreamType.TEMPORAL): 0.5, # Confidence affects timing
            (StreamType.CONFIDENCE, StreamType.ATTENTION): 0.7, # Confidence drives attention
            
            # Attention -> Others
            (StreamType.ATTENTION, StreamType.SENSORY): 0.8,  # Attention enhances sensing
            (StreamType.ATTENTION, StreamType.MOTOR): 0.4,    # Attention can inhibit motor
            (StreamType.ATTENTION, StreamType.TEMPORAL): 0.6, # Attention affects timing
            (StreamType.ATTENTION, StreamType.CONFIDENCE): 0.3, # Attention modulates confidence
        }
        
        return connectivity_map.get((source, target), 0.3)  # Default weak connectivity
    
    def _initialize_stream_sensitivities(self) -> Dict[StreamType, Dict[ConstraintType, float]]:
        """Initialize how sensitive each stream is to different constraint types."""
        sensitivities = {}
        
        for stream in StreamType:
            sensitivities[stream] = {}
            for constraint_type in ConstraintType:
                # Biological sensitivity patterns
                sensitivity = self._get_biological_sensitivity(stream, constraint_type)
                sensitivities[stream][constraint_type] = sensitivity
        
        return sensitivities
    
    def _get_biological_sensitivity(self, stream: StreamType, constraint_type: ConstraintType) -> float:
        """Get biological sensitivity of stream to constraint type."""
        # Based on neuroscience research on stream-specific sensitivities
        sensitivity_map = {
            # Sensory stream sensitivities
            (StreamType.SENSORY, ConstraintType.PROCESSING_LOAD): 0.9,    # Very sensitive to load
            (StreamType.SENSORY, ConstraintType.RESOURCE_SCARCITY): 0.7,  # Needs resources
            (StreamType.SENSORY, ConstraintType.URGENCY_SIGNAL): 0.8,     # Responds to urgency
            (StreamType.SENSORY, ConstraintType.INTERFERENCE): 0.6,       # Moderate interference
            (StreamType.SENSORY, ConstraintType.COHERENCE_PRESSURE): 0.5, # Less coherence pressure
            (StreamType.SENSORY, ConstraintType.ENERGY_DEPLETION): 0.8,   # Energy dependent
            
            # Motor stream sensitivities  
            (StreamType.MOTOR, ConstraintType.PROCESSING_LOAD): 0.6,      # Less load sensitive
            (StreamType.MOTOR, ConstraintType.RESOURCE_SCARCITY): 0.8,    # Needs resources for action
            (StreamType.MOTOR, ConstraintType.URGENCY_SIGNAL): 0.9,       # Very urgent responsive
            (StreamType.MOTOR, ConstraintType.INTERFERENCE): 0.8,         # Interference critical
            (StreamType.MOTOR, ConstraintType.COHERENCE_PRESSURE): 0.9,   # Needs coherent output
            (StreamType.MOTOR, ConstraintType.ENERGY_DEPLETION): 0.7,     # Moderate energy dependency
            
            # Temporal stream sensitivities
            (StreamType.TEMPORAL, ConstraintType.PROCESSING_LOAD): 0.7,   # Moderate load sensitivity
            (StreamType.TEMPORAL, ConstraintType.RESOURCE_SCARCITY): 0.5, # Less resource dependent
            (StreamType.TEMPORAL, ConstraintType.URGENCY_SIGNAL): 0.6,    # Moderate urgency response
            (StreamType.TEMPORAL, ConstraintType.INTERFERENCE): 0.4,      # Less interference sensitive
            (StreamType.TEMPORAL, ConstraintType.COHERENCE_PRESSURE): 0.8, # High coherence needs
            (StreamType.TEMPORAL, ConstraintType.ENERGY_DEPLETION): 0.6,  # Moderate energy dependency
            
            # Confidence stream sensitivities
            (StreamType.CONFIDENCE, ConstraintType.PROCESSING_LOAD): 0.8, # Load affects confidence
            (StreamType.CONFIDENCE, ConstraintType.RESOURCE_SCARCITY): 0.6, # Moderate resource sensitivity
            (StreamType.CONFIDENCE, ConstraintType.URGENCY_SIGNAL): 0.5,  # Less urgency sensitive
            (StreamType.CONFIDENCE, ConstraintType.INTERFERENCE): 0.9,    # Very interference sensitive
            (StreamType.CONFIDENCE, ConstraintType.COHERENCE_PRESSURE): 0.7, # Moderate coherence needs
            (StreamType.CONFIDENCE, ConstraintType.ENERGY_DEPLETION): 0.5, # Less energy dependent
            
            # Attention stream sensitivities
            (StreamType.ATTENTION, ConstraintType.PROCESSING_LOAD): 0.5,  # Less load sensitive
            (StreamType.ATTENTION, ConstraintType.RESOURCE_SCARCITY): 0.9, # Very resource sensitive
            (StreamType.ATTENTION, ConstraintType.URGENCY_SIGNAL): 0.8,   # High urgency response
            (StreamType.ATTENTION, ConstraintType.INTERFERENCE): 0.7,     # Moderate interference sensitivity
            (StreamType.ATTENTION, ConstraintType.COHERENCE_PRESSURE): 0.6, # Moderate coherence needs
            (StreamType.ATTENTION, ConstraintType.ENERGY_DEPLETION): 0.8, # High energy dependency
        }
        
        return sensitivity_map.get((stream, constraint_type), 0.5)  # Default moderate sensitivity
    
    def propagate_constraint(self, source_stream: StreamType, constraint_type: ConstraintType,
                           intensity: float, metadata: Dict[str, Any] = None) -> bool:
        """
        Propagate a constraint from source stream to other streams.
        
        Args:
            source_stream: Stream originating the constraint
            constraint_type: Type of constraint to propagate
            intensity: Strength of constraint (0.0-1.0)
            metadata: Additional constraint information
            
        Returns:
            True if constraint was propagated successfully
        """
        if metadata is None:
            metadata = {}
        
        current_time = time.time()
        
        # Calculate propagation pattern based on biological connectivity
        propagation_pattern = {}
        for target_stream in StreamType:
            if target_stream != source_stream:
                # Biological connectivity * constraint intensity * stream sensitivity
                connectivity = self._propagation_matrix[source_stream][target_stream]
                sensitivity = self._stream_sensitivities[target_stream][constraint_type]
                propagation_strength = connectivity * intensity * sensitivity
                
                # Add biological noise and nonlinearity
                propagation_strength *= (0.8 + 0.4 * np.random.random())  # 20% noise
                propagation_strength = min(1.0, propagation_strength)  # Cap at 1.0
                
                if propagation_strength > 0.1:  # Only propagate significant constraints
                    propagation_pattern[target_stream] = propagation_strength
        
        # Create propagated constraint
        constraint = PropagatedConstraint(
            constraint_type=constraint_type,
            source_stream=source_stream,
            intensity=intensity,
            decay_rate=self._decay_dynamics[constraint_type],
            creation_time=current_time,
            propagation_pattern=propagation_pattern,
            metadata=metadata
        )
        
        # Add to active constraints
        self._active_constraints.append(constraint)
        
        # Update constraint history for learning
        self._constraint_history[constraint_type].append({
            'source': source_stream,
            'intensity': intensity,
            'propagation_count': len(propagation_pattern),
            'time': current_time
        })
        
        # Learn emergent propagation patterns
        self._update_emergent_patterns(constraint)
        
        if not self.quiet_mode and len(propagation_pattern) > 0:
            print(f"🔗 Constraint propagated: {constraint_type.value} from {source_stream.value}")
            print(f"   Intensity: {intensity:.2f}, Targets: {list(propagation_pattern.keys())}")
        
        return len(propagation_pattern) > 0
    
    def get_stream_constraints(self, stream: StreamType) -> Dict[ConstraintType, float]:
        """
        Get current constraint pressures affecting a specific stream.
        
        Args:
            stream: Stream to get constraints for
            
        Returns:
            Dictionary mapping constraint types to their current intensities
        """
        current_time = time.time()
        constraints = defaultdict(float)
        
        for constraint in self._active_constraints:
            if stream in constraint.propagation_pattern:
                # Calculate current intensity with decay
                age = current_time - constraint.creation_time
                decay_factor = constraint.decay_rate ** age
                current_intensity = constraint.intensity * decay_factor
                
                # Apply propagation strength
                propagated_intensity = current_intensity * constraint.propagation_pattern[stream]
                
                # Accumulate constraints of same type (biological summation)
                constraints[constraint.constraint_type] += propagated_intensity
        
        # Cap constraints at 1.0 (biological saturation)
        for constraint_type in constraints:
            constraints[constraint_type] = min(1.0, constraints[constraint_type])
        
        return dict(constraints)
    
    def get_total_constraint_pressure(self, stream: StreamType) -> float:
        """Get total constraint pressure on a stream (sum of all constraint types)."""
        constraints = self.get_stream_constraints(stream)
        return min(1.0, sum(constraints.values()))
    
    def update_constraint_dynamics(self):
        """Update constraint decay and cleanup expired constraints."""
        current_time = time.time()
        active_constraints = []
        
        for constraint in self._active_constraints:
            # Calculate current intensity with decay
            age = current_time - constraint.creation_time
            decay_factor = constraint.decay_rate ** age
            current_intensity = constraint.intensity * decay_factor
            
            # Keep constraint if still significant
            if current_intensity > 0.05:  # 5% threshold
                active_constraints.append(constraint)
        
        self._active_constraints = active_constraints
        
        # Adapt propagation dynamics based on biological oscillations
        if self.biological_oscillator:
            timing = self.biological_oscillator.get_current_timing()
            if timing.binding_window_active:
                # During binding windows, enhance constraint propagation
                self._enhance_propagation_during_binding()
    
    def _enhance_propagation_during_binding(self):
        """Enhance constraint propagation during gamma binding windows."""
        # During binding windows, constraints propagate more strongly
        for constraint in self._active_constraints:
            # Temporarily boost propagation strength by 20%
            for target in constraint.propagation_pattern:
                constraint.propagation_pattern[target] *= 1.2
                constraint.propagation_pattern[target] = min(1.0, constraint.propagation_pattern[target])
    
    def _update_emergent_patterns(self, constraint: PropagatedConstraint):
        """Learn emergent propagation patterns from constraint interactions."""
        pattern_key = f"{constraint.source_stream.value}_{constraint.constraint_type.value}"
        
        if pattern_key not in self._emergent_patterns:
            self._emergent_patterns[pattern_key] = {
                'frequency': 0,
                'avg_intensity': 0.0,
                'successful_propagations': 0,
                'target_preferences': defaultdict(float)
            }
        
        pattern = self._emergent_patterns[pattern_key]
        pattern['frequency'] += 1
        pattern['avg_intensity'] = (pattern['avg_intensity'] + constraint.intensity) / 2
        
        if len(constraint.propagation_pattern) > 0:
            pattern['successful_propagations'] += 1
            
            # Learn target preferences
            for target, strength in constraint.propagation_pattern.items():
                pattern['target_preferences'][target.value] += strength
    
    def get_propagation_stats(self) -> Dict[str, Any]:
        """Get constraint propagation statistics for monitoring."""
        current_time = time.time()
        
        # Count active constraints by type
        constraint_counts = defaultdict(int)
        total_intensity = 0.0
        
        for constraint in self._active_constraints:
            constraint_counts[constraint.constraint_type.value] += 1
            age = current_time - constraint.creation_time
            decay_factor = constraint.decay_rate ** age
            total_intensity += constraint.intensity * decay_factor
        
        # Historical statistics
        total_propagations = sum(len(history) for history in self._constraint_history.values())
        
        return {
            'active_constraints': len(self._active_constraints),
            'constraints_by_type': dict(constraint_counts),
            'total_constraint_intensity': total_intensity,
            'total_propagations': total_propagations,
            'emergent_patterns': len(self._emergent_patterns),
            'constraint_history_size': len(self._constraint_history),
            'propagation_matrix_complexity': len(self._propagation_matrix) * len(StreamType)
        }
    
    def cleanup_expired_constraints(self, max_age_seconds: float = 10.0):
        """Clean up very old constraints to prevent memory growth."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        # Remove old active constraints
        self._active_constraints = [
            c for c in self._active_constraints 
            if c.creation_time >= cutoff_time
        ]
        
        # Trim constraint history
        for constraint_type in self._constraint_history:
            self._constraint_history[constraint_type] = [
                entry for entry in self._constraint_history[constraint_type]
                if entry['time'] >= cutoff_time
            ]


def create_constraint_propagation_system(biological_oscillator=None, quiet_mode: bool = False) -> ConstraintPropagationSystem:
    """
    Factory function to create constraint propagation system.
    
    Args:
        biological_oscillator: Optional biological oscillator for timing
        quiet_mode: Suppress debug output
        
    Returns:
        Configured ConstraintPropagationSystem instance
    """
    return ConstraintPropagationSystem(biological_oscillator, quiet_mode)


# Example usage and testing
if __name__ == "__main__":
    print("🔗 Testing Constraint Propagation System")
    
    # Create constraint propagation system
    propagation_system = create_constraint_propagation_system(quiet_mode=False)
    
    # Test constraint propagation
    print(f"\n🔬 Testing constraint propagation...")
    
    # Simulate high processing load in sensory stream
    success = propagation_system.propagate_constraint(
        source_stream=StreamType.SENSORY,
        constraint_type=ConstraintType.PROCESSING_LOAD,
        intensity=0.8,
        metadata={'cause': 'high_input_complexity'}
    )
    print(f"Sensory load propagation: {'✅' if success else '❌'}")
    
    # Check constraints on motor stream
    motor_constraints = propagation_system.get_stream_constraints(StreamType.MOTOR)
    print(f"Motor stream constraints: {motor_constraints}")
    
    # Simulate urgency in motor stream
    propagation_system.propagate_constraint(
        source_stream=StreamType.MOTOR,
        constraint_type=ConstraintType.URGENCY_SIGNAL,
        intensity=0.9,
        metadata={'cause': 'obstacle_detected'}
    )
    
    # Check constraints on all streams
    for stream in StreamType:
        pressure = propagation_system.get_total_constraint_pressure(stream)
        print(f"{stream.value} total pressure: {pressure:.3f}")
    
    # Test constraint decay
    print(f"\n🔬 Testing constraint decay...")
    propagation_system.update_constraint_dynamics()
    
    # Get statistics
    stats = propagation_system.get_propagation_stats()
    print(f"\n📊 Propagation Statistics:")
    print(f"  Active constraints: {stats['active_constraints']}")
    print(f"  Total propagations: {stats['total_propagations']}")
    print(f"  Emergent patterns: {stats['emergent_patterns']}")
    
    print("✅ Constraint propagation system test completed!")