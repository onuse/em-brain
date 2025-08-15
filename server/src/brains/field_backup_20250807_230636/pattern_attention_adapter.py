"""
Pattern Attention Adapter

Adapts unified pattern extraction for attention selection.
Selects most salient patterns for focused processing.
"""

import torch
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import time

from .unified_pattern_system import UnifiedPatternSystem, FieldPattern


@dataclass
class AttentionFocus:
    """Current attention state."""
    attended_patterns: List[FieldPattern]
    focus_strength: float
    total_salience: float
    timestamp: float
    
    def get_locations(self) -> Set[Optional[tuple]]:
        """Get attended locations."""
        return {p.location for p in self.attended_patterns}


class PatternAttentionAdapter:
    """
    Manages attention using unified pattern extraction.
    
    Selects the most salient patterns for focused processing
    while maintaining biological attention limits.
    """
    
    def __init__(self,
                 pattern_system: UnifiedPatternSystem,
                 attention_capacity: int = 5,
                 device: torch.device = torch.device('cpu')):
        """Initialize attention adapter."""
        self.pattern_system = pattern_system
        self.attention_capacity = attention_capacity
        self.device = device
        
        # Attention state
        self.current_focus = None
        self.attention_history = []
        
        # Salience modulation
        self.novelty_weight = 0.4
        self.energy_weight = 0.3
        self.coherence_weight = 0.3
        
        # Attention inertia (prefer stable focus)
        self.focus_inertia = 0.7
        
    def process_field_patterns(self,
                              field: torch.Tensor,
                              sensory_patterns: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Process field and select patterns for attention.
        
        Args:
            field: Current field state
            sensory_patterns: Optional sensory inputs to prioritize
            
        Returns:
            Attention state dictionary
        """
        # Extract patterns from field
        field_patterns = self.pattern_system.extract_patterns(
            field, 
            n_patterns=self.attention_capacity * 3  # Extract more than we'll attend to
        )
        
        # Process sensory patterns if provided
        if sensory_patterns:
            sensory_saliences = self._process_sensory_patterns(sensory_patterns)
            # Boost field patterns near sensory inputs
            field_patterns = self._boost_sensory_aligned_patterns(
                field_patterns, sensory_saliences
            )
        
        # Select patterns for attention
        attended_patterns = self._select_attention_targets(field_patterns)
        
        # Create attention focus
        total_salience = sum(p.salience for p in attended_patterns)
        focus_strength = self._compute_focus_strength(attended_patterns)
        
        self.current_focus = AttentionFocus(
            attended_patterns=attended_patterns,
            focus_strength=focus_strength,
            total_salience=total_salience,
            timestamp=time.time()
        )
        
        # Update history
        self.attention_history.append(self.current_focus)
        if len(self.attention_history) > 100:
            self.attention_history.pop(0)
        
        # Return attention state
        return {
            'attended_patterns': attended_patterns,
            'attended_locations': self.current_focus.get_locations(),
            'focus_strength': focus_strength,
            'total_salience': total_salience,
            'pattern_count': len(attended_patterns),
            'suppression_level': self._compute_suppression_level(field_patterns, attended_patterns)
        }
    
    def _process_sensory_patterns(self, 
                                 sensory_patterns: Dict[str, torch.Tensor]) -> List[float]:
        """Extract salience from sensory inputs."""
        saliences = []
        
        for modality, pattern in sensory_patterns.items():
            # Simple energy-based salience
            energy = torch.mean(torch.abs(pattern)).item()
            
            # Modality weights
            modality_weight = {
                'visual': 1.2,
                'auditory': 1.1,
                'motor': 1.3,
                'tactile': 1.0
            }.get(modality, 1.0)
            
            saliences.append(energy * modality_weight)
        
        return saliences
    
    def _boost_sensory_aligned_patterns(self,
                                       field_patterns: List[FieldPattern],
                                       sensory_saliences: List[float]) -> List[FieldPattern]:
        """Boost patterns that align with sensory inputs."""
        if not sensory_saliences:
            return field_patterns
        
        # Simple boost based on overall sensory salience
        sensory_boost = min(1.5, 1.0 + sum(sensory_saliences))
        
        # Boost patterns with high energy (likely sensory-driven)
        for pattern in field_patterns:
            if pattern.energy > 0.5:
                pattern.salience *= sensory_boost
        
        return field_patterns
    
    def _select_attention_targets(self, 
                                 patterns: List[FieldPattern]) -> List[FieldPattern]:
        """Select patterns for attention focus."""
        # Sort by salience
        patterns.sort(key=lambda p: p.salience, reverse=True)
        
        selected = []
        
        # If we have previous focus, consider attention inertia
        if self.current_focus and self.focus_inertia > 0:
            # Try to maintain focus on similar patterns
            for old_pattern in self.current_focus.attended_patterns:
                for new_pattern in patterns:
                    if self._patterns_similar(old_pattern, new_pattern):
                        # Boost salience for continuity
                        new_pattern.salience *= (1.0 + self.focus_inertia)
                        break
            
            # Re-sort with inertia boost
            patterns.sort(key=lambda p: p.salience, reverse=True)
        
        # Select top patterns up to capacity
        for pattern in patterns:
            if len(selected) < self.attention_capacity:
                # Additional filtering - minimum salience threshold
                if pattern.salience > 0.1:
                    selected.append(pattern)
            else:
                break
        
        return selected
    
    def _patterns_similar(self, p1: FieldPattern, p2: FieldPattern) -> bool:
        """Check if two patterns are similar enough for attention continuity."""
        # Location-based similarity
        if p1.location and p2.location:
            if p1.location == p2.location:
                return True
        
        # Feature-based similarity
        feature_diff = 0.0
        for feature in ['energy', 'oscillation', 'coherence']:
            diff = abs(getattr(p1, feature) - getattr(p2, feature))
            feature_diff += diff
        
        return feature_diff < 0.3  # Threshold for similarity
    
    def _compute_focus_strength(self, patterns: List[FieldPattern]) -> float:
        """Compute overall attention focus strength."""
        if not patterns:
            return 0.0
        
        # Average salience of attended patterns
        avg_salience = sum(p.salience for p in patterns) / len(patterns)
        
        # Focus is stronger with fewer, more salient patterns
        focus_factor = 1.0 - (len(patterns) / self.attention_capacity) * 0.3
        
        return min(1.0, avg_salience * focus_factor)
    
    def _compute_suppression_level(self,
                                  all_patterns: List[FieldPattern],
                                  attended_patterns: List[FieldPattern]) -> float:
        """Compute how much we're suppressing unattended patterns."""
        if not all_patterns:
            return 0.0
        
        attended_salience = sum(p.salience for p in attended_patterns)
        total_salience = sum(p.salience for p in all_patterns)
        
        if total_salience > 0:
            return 1.0 - (attended_salience / total_salience)
        else:
            return 0.0
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get summary of current attention state."""
        if not self.current_focus:
            return {
                'active': False,
                'pattern_count': 0,
                'focus_strength': 0.0
            }
        
        # Analyze attended patterns
        pattern_types = {
            'oscillatory': 0,
            'flowing': 0,
            'stable': 0,
            'novel': 0
        }
        
        for pattern in self.current_focus.attended_patterns:
            if pattern.oscillation > 0.5:
                pattern_types['oscillatory'] += 1
            if pattern.flow_divergence > 0.3 or pattern.flow_curl > 0.3:
                pattern_types['flowing'] += 1
            if pattern.coherence > 0.7:
                pattern_types['stable'] += 1
            if pattern.novelty > 0.6:
                pattern_types['novel'] += 1
        
        return {
            'active': True,
            'pattern_count': len(self.current_focus.attended_patterns),
            'focus_strength': self.current_focus.focus_strength,
            'total_salience': self.current_focus.total_salience,
            'pattern_types': pattern_types,
            'locations': list(self.current_focus.get_locations())
        }