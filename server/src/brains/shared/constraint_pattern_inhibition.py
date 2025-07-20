#!/usr/bin/env python3
"""
Constraint-Based Pattern Inhibition and Selection System

Implements emergent pattern selection where patterns compete for limited activation
resources through constraint dynamics, leading to natural pattern hierarchies.

Key principles:
- Pattern selection emerges from constraint competition, not explicit algorithms
- Interference between patterns creates natural inhibition
- Coherence pressures strengthen related pattern groups
- Resource competition determines which patterns remain active
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .stream_types import StreamType, ConstraintType


@dataclass
class PatternActivation:
    """Represents an active pattern with its current state."""
    pattern_id: int
    stream_type: StreamType
    activation_strength: float    # Current activation level (0.0-1.0)
    base_activation: float       # Original activation from stream processing
    coherence_score: float       # How well it fits with other patterns (0.0-1.0)
    interference_penalty: float  # Penalty from interfering patterns (0.0+)
    resource_consumption: float  # How much resources this pattern uses (0.0-1.0)
    age_cycles: int             # How many cycles this pattern has been active
    last_reinforcement: float   # Timestamp of last reinforcement
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternInteraction:
    """Represents interaction between two patterns."""
    pattern_a: int
    pattern_b: int
    interaction_type: str        # "inhibition", "reinforcement", "neutral"
    interaction_strength: float  # Magnitude of interaction (0.0-1.0)
    constraint_basis: ConstraintType  # Which constraint created this interaction


class PatternSelectionMode(Enum):
    """Different modes of pattern selection."""
    COMPETITIVE_INHIBITION = "competitive_inhibition"     # Strongest patterns inhibit weaker ones
    COHERENCE_CLUSTERING = "coherence_clustering"         # Coherent patterns group together
    RESOURCE_OPTIMIZATION = "resource_optimization"       # Minimize resource consumption
    INTERFERENCE_MINIMIZATION = "interference_minimization"  # Minimize pattern interference


class ConstraintPatternInhibitor:
    """
    Manages constraint-based pattern inhibition and selection.
    
    Implements biologically-inspired pattern competition where patterns naturally
    compete for limited activation resources based on constraint dynamics.
    """
    
    def __init__(self, max_active_patterns: int = 10, 
                 selection_mode: PatternSelectionMode = PatternSelectionMode.COMPETITIVE_INHIBITION,
                 quiet_mode: bool = False):
        """
        Initialize constraint-based pattern inhibitor.
        
        Args:
            max_active_patterns: Maximum number of patterns that can remain active
            selection_mode: How patterns compete and get selected
            quiet_mode: Suppress debug output
        """
        self.max_active_patterns = max_active_patterns
        self.selection_mode = selection_mode
        self.quiet_mode = quiet_mode
        
        # Active patterns across all streams
        self.active_patterns: Dict[int, PatternActivation] = {}
        
        # Pattern interaction network
        self.pattern_interactions: Dict[Tuple[int, int], PatternInteraction] = {}
        
        # Inhibition parameters
        self.inhibition_strength = 0.8      # How strongly patterns inhibit each other
        self.coherence_boost = 0.3          # Boost for coherent pattern groups
        self.resource_penalty = 0.5         # Penalty for high resource consumption
        self.age_decay_rate = 0.02          # How quickly old patterns decay
        self.activation_threshold = 0.1     # Minimum activation to remain active
        
        # Selection statistics
        self.selection_stats = {
            'total_selections': 0,
            'patterns_inhibited': 0,
            'coherence_groups_formed': 0,
            'resource_optimizations': 0,
            'avg_active_patterns': 0.0
        }
        
        # History for analysis
        self.selection_history = []
        
        if not quiet_mode:
            print(f"🧠 ConstraintPatternInhibitor initialized")
            print(f"   Max active patterns: {max_active_patterns}")
            print(f"   Selection mode: {selection_mode.value}")
            print(f"   Inhibition strength: {self.inhibition_strength:.2f}")
    
    def update_active_patterns(self, stream_patterns: Dict[StreamType, List[int]], 
                              stream_activations: Dict[StreamType, List[float]],
                              constraint_pressures: Dict[StreamType, Dict[ConstraintType, float]]) -> Dict[StreamType, List[int]]:
        """
        Update active patterns with constraint-based inhibition and selection.
        
        Args:
            stream_patterns: Active patterns from each stream
            stream_activations: Activation strengths for each pattern
            constraint_pressures: Current constraint pressures for each stream
            
        Returns:
            Selected patterns after constraint-based inhibition
        """
        current_time = time.time()
        
        # Update pattern activations from stream input
        self._update_pattern_activations(stream_patterns, stream_activations, constraint_pressures, current_time)
        
        # Compute pattern interactions based on constraints
        self._compute_pattern_interactions(constraint_pressures)
        
        # Apply constraint-based inhibition and selection
        selected_patterns = self._apply_constraint_selection()
        
        # Update statistics
        self._update_selection_stats(selected_patterns)
        
        # Clean up inactive patterns
        self._cleanup_inactive_patterns()
        
        return selected_patterns
    
    def _update_pattern_activations(self, stream_patterns: Dict[StreamType, List[int]], 
                                   stream_activations: Dict[StreamType, List[float]],
                                   constraint_pressures: Dict[StreamType, Dict[ConstraintType, float]],
                                   current_time: float):
        """Update activations for all patterns based on stream input."""
        # Process patterns from each stream
        for stream_type, patterns in stream_patterns.items():
            activations = stream_activations.get(stream_type, [])
            constraints = constraint_pressures.get(stream_type, {})
            
            for i, pattern_id in enumerate(patterns):
                base_activation = activations[i] if i < len(activations) else 0.5
                
                # Create or update pattern activation
                if pattern_id in self.active_patterns:
                    pattern = self.active_patterns[pattern_id]
                    pattern.base_activation = base_activation
                    pattern.last_reinforcement = current_time
                    pattern.age_cycles += 1
                else:
                    # New pattern
                    pattern = PatternActivation(
                        pattern_id=pattern_id,
                        stream_type=stream_type,
                        activation_strength=base_activation,
                        base_activation=base_activation,
                        coherence_score=0.5,  # Neutral initially
                        interference_penalty=0.0,
                        resource_consumption=self._compute_resource_consumption(
                            pattern_id, stream_type, base_activation, constraints
                        ),
                        age_cycles=0,
                        last_reinforcement=current_time
                    )
                    self.active_patterns[pattern_id] = pattern
                
                # Update activation strength based on constraints
                self._update_pattern_constraint_effects(pattern, constraints)
    
    def _compute_resource_consumption(self, pattern_id: int, stream_type: StreamType, 
                                    activation: float, constraints: Dict[ConstraintType, float]) -> float:
        """Compute how much resources this pattern consumes."""
        # Base consumption from activation level
        base_consumption = activation * 0.3
        
        # Pattern complexity factor (higher pattern IDs = more complex)
        complexity_factor = min(1.0, (pattern_id % 100) / 100.0)
        base_consumption += complexity_factor * 0.2
        
        # Stream-specific consumption
        stream_factors = {
            StreamType.SENSORY: 1.0,      # Baseline
            StreamType.MOTOR: 1.2,        # Motor patterns use more resources
            StreamType.TEMPORAL: 0.8,     # Temporal patterns use fewer resources
            StreamType.CONFIDENCE: 0.6,   # Confidence patterns are lightweight
            StreamType.ATTENTION: 0.5     # Attention patterns are minimal
        }
        base_consumption *= stream_factors.get(stream_type, 1.0)
        
        # Constraint-based modulation
        load_pressure = constraints.get(ConstraintType.PROCESSING_LOAD, 0.0)
        base_consumption *= (1.0 + load_pressure * 0.5)  # High load increases consumption
        
        return min(1.0, base_consumption)
    
    def _update_pattern_constraint_effects(self, pattern: PatternActivation, 
                                         constraints: Dict[ConstraintType, float]):
        """Update pattern activation based on constraint pressures."""
        # Interference pressure reduces activation
        interference = constraints.get(ConstraintType.INTERFERENCE, 0.0)
        pattern.interference_penalty = interference * 0.4
        
        # Coherence pressure improves activation
        coherence_pressure = constraints.get(ConstraintType.COHERENCE_PRESSURE, 0.0)
        if coherence_pressure > 0.3:
            pattern.coherence_score *= (1.0 - coherence_pressure * 0.2)  # High coherence pressure = less coherent
        
        # Resource scarcity affects high-consumption patterns more
        scarcity = constraints.get(ConstraintType.RESOURCE_SCARCITY, 0.0)
        if pattern.resource_consumption > 0.6 and scarcity > 0.4:
            pattern.activation_strength *= (1.0 - scarcity * 0.3)
        
        # Age decay
        age_factor = max(0.1, 1.0 - pattern.age_cycles * self.age_decay_rate)
        pattern.activation_strength = pattern.base_activation * age_factor - pattern.interference_penalty
        pattern.activation_strength = max(0.0, pattern.activation_strength)
    
    def _compute_pattern_interactions(self, constraint_pressures: Dict[StreamType, Dict[ConstraintType, float]]):
        """Compute interactions between active patterns based on constraints."""
        patterns = list(self.active_patterns.values())
        
        for i, pattern_a in enumerate(patterns):
            for pattern_b in patterns[i+1:]:
                interaction_key = (min(pattern_a.pattern_id, pattern_b.pattern_id),
                                 max(pattern_a.pattern_id, pattern_b.pattern_id))
                
                # Compute interaction based on pattern properties
                interaction = self._compute_pattern_interaction(pattern_a, pattern_b, constraint_pressures)
                
                if interaction:
                    self.pattern_interactions[interaction_key] = interaction
    
    def _compute_pattern_interaction(self, pattern_a: PatternActivation, pattern_b: PatternActivation,
                                   constraint_pressures: Dict[StreamType, Dict[ConstraintType, float]]) -> Optional[PatternInteraction]:
        """Compute interaction between two specific patterns."""
        # Same stream patterns may interfere
        if pattern_a.stream_type == pattern_b.stream_type:
            # Strong patterns inhibit weaker ones in same stream
            if abs(pattern_a.activation_strength - pattern_b.activation_strength) > 0.3:
                stronger = pattern_a if pattern_a.activation_strength > pattern_b.activation_strength else pattern_b
                interference_strength = min(0.8, abs(pattern_a.activation_strength - pattern_b.activation_strength))
                
                return PatternInteraction(
                    pattern_a=pattern_a.pattern_id,
                    pattern_b=pattern_b.pattern_id,
                    interaction_type="inhibition",
                    interaction_strength=interference_strength,
                    constraint_basis=ConstraintType.INTERFERENCE
                )
        
        # Cross-stream patterns may reinforce if coherent
        else:
            # Check for coherence based on activation similarity and timing
            activation_similarity = 1.0 - abs(pattern_a.activation_strength - pattern_b.activation_strength)
            timing_similarity = 1.0 - min(1.0, abs(pattern_a.last_reinforcement - pattern_b.last_reinforcement) / 0.1)
            
            coherence = (activation_similarity + timing_similarity) / 2.0
            
            if coherence > 0.6:
                return PatternInteraction(
                    pattern_a=pattern_a.pattern_id,
                    pattern_b=pattern_b.pattern_id,
                    interaction_type="reinforcement",
                    interaction_strength=coherence * 0.5,
                    constraint_basis=ConstraintType.COHERENCE_PRESSURE
                )
        
        return None
    
    def _apply_constraint_selection(self) -> Dict[StreamType, List[int]]:
        """Apply constraint-based pattern selection and inhibition."""
        if self.selection_mode == PatternSelectionMode.COMPETITIVE_INHIBITION:
            return self._competitive_inhibition_selection()
        elif self.selection_mode == PatternSelectionMode.COHERENCE_CLUSTERING:
            return self._coherence_clustering_selection()
        elif self.selection_mode == PatternSelectionMode.RESOURCE_OPTIMIZATION:
            return self._resource_optimization_selection()
        elif self.selection_mode == PatternSelectionMode.INTERFERENCE_MINIMIZATION:
            return self._interference_minimization_selection()
        else:
            return self._competitive_inhibition_selection()  # Default
    
    def _competitive_inhibition_selection(self) -> Dict[StreamType, List[int]]:
        """Select patterns through competitive inhibition."""
        # Apply mutual inhibition
        for interaction in self.pattern_interactions.values():
            if interaction.interaction_type == "inhibition":
                pattern_a = self.active_patterns.get(interaction.pattern_a)
                pattern_b = self.active_patterns.get(interaction.pattern_b)
                
                if pattern_a and pattern_b:
                    # Stronger pattern inhibits weaker one
                    if pattern_a.activation_strength > pattern_b.activation_strength:
                        inhibition = interaction.interaction_strength * self.inhibition_strength
                        pattern_b.activation_strength *= (1.0 - inhibition)
                    else:
                        inhibition = interaction.interaction_strength * self.inhibition_strength
                        pattern_a.activation_strength *= (1.0 - inhibition)
        
        # Select patterns above threshold, up to maximum
        active_patterns = [p for p in self.active_patterns.values() 
                          if p.activation_strength > self.activation_threshold]
        active_patterns.sort(key=lambda p: p.activation_strength, reverse=True)
        
        selected = active_patterns[:self.max_active_patterns]
        
        # Group by stream
        selected_by_stream = defaultdict(list)
        for pattern in selected:
            selected_by_stream[pattern.stream_type].append(pattern.pattern_id)
        
        return dict(selected_by_stream)
    
    def _coherence_clustering_selection(self) -> Dict[StreamType, List[int]]:
        """Select patterns by forming coherent clusters."""
        # Apply reinforcement between coherent patterns
        for interaction in self.pattern_interactions.values():
            if interaction.interaction_type == "reinforcement":
                pattern_a = self.active_patterns.get(interaction.pattern_a)
                pattern_b = self.active_patterns.get(interaction.pattern_b)
                
                if pattern_a and pattern_b:
                    boost = interaction.interaction_strength * self.coherence_boost
                    pattern_a.activation_strength *= (1.0 + boost)
                    pattern_b.activation_strength *= (1.0 + boost)
                    pattern_a.coherence_score = min(1.0, pattern_a.coherence_score + boost)
                    pattern_b.coherence_score = min(1.0, pattern_b.coherence_score + boost)
        
        # Select patterns with highest coherence scores
        active_patterns = [p for p in self.active_patterns.values() 
                          if p.activation_strength > self.activation_threshold]
        active_patterns.sort(key=lambda p: p.coherence_score * p.activation_strength, reverse=True)
        
        selected = active_patterns[:self.max_active_patterns]
        
        selected_by_stream = defaultdict(list)
        for pattern in selected:
            selected_by_stream[pattern.stream_type].append(pattern.pattern_id)
        
        return dict(selected_by_stream)
    
    def _resource_optimization_selection(self) -> Dict[StreamType, List[int]]:
        """Select patterns to optimize resource consumption."""
        # Calculate efficiency (activation / resource consumption)
        active_patterns = []
        for pattern in self.active_patterns.values():
            if pattern.activation_strength > self.activation_threshold:
                efficiency = pattern.activation_strength / max(0.1, pattern.resource_consumption)
                pattern.metadata['efficiency'] = efficiency
                active_patterns.append(pattern)
        
        # Select most efficient patterns
        active_patterns.sort(key=lambda p: p.metadata['efficiency'], reverse=True)
        selected = active_patterns[:self.max_active_patterns]
        
        selected_by_stream = defaultdict(list)
        for pattern in selected:
            selected_by_stream[pattern.stream_type].append(pattern.pattern_id)
        
        return dict(selected_by_stream)
    
    def _interference_minimization_selection(self) -> Dict[StreamType, List[int]]:
        """Select patterns to minimize total interference."""
        # Calculate interference cost for each pattern
        for pattern in self.active_patterns.values():
            interference_cost = 0.0
            for interaction in self.pattern_interactions.values():
                if (interaction.pattern_a == pattern.pattern_id or 
                    interaction.pattern_b == pattern.pattern_id) and \
                   interaction.interaction_type == "inhibition":
                    interference_cost += interaction.interaction_strength
            pattern.metadata['interference_cost'] = interference_cost
        
        # Select patterns with lowest interference cost
        active_patterns = [p for p in self.active_patterns.values() 
                          if p.activation_strength > self.activation_threshold]
        active_patterns.sort(key=lambda p: p.metadata.get('interference_cost', 0.0))
        
        selected = active_patterns[:self.max_active_patterns]
        
        selected_by_stream = defaultdict(list)
        for pattern in selected:
            selected_by_stream[pattern.stream_type].append(pattern.pattern_id)
        
        return dict(selected_by_stream)
    
    def _update_selection_stats(self, selected_patterns: Dict[StreamType, List[int]]):
        """Update selection statistics."""
        self.selection_stats['total_selections'] += 1
        
        total_selected = sum(len(patterns) for patterns in selected_patterns.values())
        total_stats = self.selection_stats['total_selections']
        prev_avg = self.selection_stats['avg_active_patterns']
        self.selection_stats['avg_active_patterns'] = (
            (prev_avg * (total_stats - 1) + total_selected) / total_stats
        )
        
        # Count inhibited patterns
        all_pattern_ids = set(self.active_patterns.keys())
        selected_pattern_ids = set()
        for patterns in selected_patterns.values():
            selected_pattern_ids.update(patterns)
        
        inhibited_count = len(all_pattern_ids - selected_pattern_ids)
        self.selection_stats['patterns_inhibited'] += inhibited_count
        
        # Record selection event
        self.selection_history.append({
            'timestamp': time.time(),
            'selected_patterns': selected_patterns.copy(),
            'total_active': len(self.active_patterns),
            'total_selected': total_selected,
            'inhibited_count': inhibited_count
        })
        
        # Keep history manageable
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-50:]
    
    def _cleanup_inactive_patterns(self):
        """Remove patterns that have fallen below activation threshold."""
        inactive_patterns = [
            pattern_id for pattern_id, pattern in self.active_patterns.items()
            if pattern.activation_strength <= self.activation_threshold
        ]
        
        for pattern_id in inactive_patterns:
            del self.active_patterns[pattern_id]
            
            # Remove interactions involving this pattern
            interactions_to_remove = [
                key for key in self.pattern_interactions.keys()
                if pattern_id in key
            ]
            for key in interactions_to_remove:
                del self.pattern_interactions[key]
    
    def get_active_patterns_info(self) -> Dict[str, Any]:
        """Get detailed information about currently active patterns."""
        pattern_info = {}
        for pattern_id, pattern in self.active_patterns.items():
            pattern_info[str(pattern_id)] = {
                'stream_type': pattern.stream_type.value,
                'activation_strength': pattern.activation_strength,
                'base_activation': pattern.base_activation,
                'coherence_score': pattern.coherence_score,
                'interference_penalty': pattern.interference_penalty,
                'resource_consumption': pattern.resource_consumption,
                'age_cycles': pattern.age_cycles
            }
        
        return {
            'patterns': pattern_info,
            'total_active': len(self.active_patterns),
            'total_interactions': len(self.pattern_interactions),
            'selection_stats': self.selection_stats.copy()
        }
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get pattern selection statistics."""
        stats = self.selection_stats.copy()
        
        # Add derived metrics
        if stats['total_selections'] > 0:
            stats['inhibition_rate'] = stats['patterns_inhibited'] / stats['total_selections']
        else:
            stats['inhibition_rate'] = 0.0
        
        # Recent selection patterns
        if self.selection_history:
            recent_selections = self.selection_history[-10:]
            recent_totals = [entry['total_selected'] for entry in recent_selections]
            stats['recent_avg_selected'] = np.mean(recent_totals) if recent_totals else 0.0
            stats['selection_variance'] = np.var(recent_totals) if len(recent_totals) > 1 else 0.0
        
        return stats


# Factory function
def create_pattern_inhibitor(max_patterns: int = 10,
                           mode: PatternSelectionMode = PatternSelectionMode.COMPETITIVE_INHIBITION,
                           quiet_mode: bool = False) -> ConstraintPatternInhibitor:
    """Create a constraint-based pattern inhibitor."""
    return ConstraintPatternInhibitor(max_patterns, mode, quiet_mode)


if __name__ == "__main__":
    # Example usage
    print("🧠 Testing Constraint-Based Pattern Inhibition")
    
    inhibitor = create_pattern_inhibitor(max_patterns=5, quiet_mode=False)
    
    # Mock stream patterns and activations
    stream_patterns = {
        StreamType.SENSORY: [1, 2, 3, 4, 5, 6],
        StreamType.MOTOR: [10, 11, 12],
        StreamType.TEMPORAL: [20, 21]
    }
    
    stream_activations = {
        StreamType.SENSORY: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        StreamType.MOTOR: [0.8, 0.6, 0.4],
        StreamType.TEMPORAL: [0.7, 0.5]
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.INTERFERENCE: 0.6,
            ConstraintType.COHERENCE_PRESSURE: 0.3
        },
        StreamType.MOTOR: {
            ConstraintType.RESOURCE_SCARCITY: 0.4
        },
        StreamType.TEMPORAL: {}
    }
    
    # Run pattern selection
    selected = inhibitor.update_active_patterns(stream_patterns, stream_activations, constraint_pressures)
    
    print(f"\n📊 Pattern Selection Results:")
    for stream_type, patterns in selected.items():
        print(f"  {stream_type.value}: {patterns}")
    
    # Get detailed info
    info = inhibitor.get_active_patterns_info()
    print(f"\n📈 Active Patterns Info:")
    print(f"  Total active: {info['total_active']}")
    print(f"  Total interactions: {info['total_interactions']}")
    
    stats = inhibitor.get_selection_stats()
    print(f"\n📊 Selection Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")