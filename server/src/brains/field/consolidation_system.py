"""
Consolidation System for Advanced Learning

Implements memory consolidation and dream states during idle periods.
Based on biological sleep consolidation principles.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class ConsolidationMetrics:
    """Metrics tracking consolidation effectiveness."""
    cycles_processed: int = 0
    patterns_strengthened: int = 0
    patterns_pruned: int = 0
    topology_refined: int = 0
    dream_sequences: int = 0
    consolidation_benefit: float = 0.0


class ConsolidationSystem:
    """
    Implements biological-inspired memory consolidation during idle periods.
    
    Key features:
    1. Pattern replay and strengthening
    2. Weak pattern pruning
    3. Dream-like recombination
    4. Topology refinement
    """
    
    def __init__(self, field_shape: Tuple[int, ...], device: torch.device):
        """Initialize consolidation system."""
        self.field_shape = field_shape
        self.device = device
        
        # Consolidation parameters
        self.replay_strength = 0.3
        self.pruning_threshold = 0.1
        self.dream_temperature = 0.5
        self.topology_decay = 0.95
        
        # State tracking
        self.is_consolidating = False
        self.last_consolidation = time.time()
        self.metrics = ConsolidationMetrics()
        
        # Memory buffers for consolidation
        self.recent_patterns = deque(maxlen=100)
        self.important_patterns = deque(maxlen=50)
        
    def start_consolidation(self, brain):
        """Begin consolidation phase."""
        self.is_consolidating = True
        self.consolidation_start = time.time()
        
        # Extract current patterns from brain
        if hasattr(brain, 'pattern_system'):
            current_patterns = brain.pattern_system.extract_patterns(
                brain.unified_field, n_patterns=20
            )
            # Store patterns with metadata
            for pattern in current_patterns:
                self.recent_patterns.append({
                    'pattern': pattern,
                    'timestamp': time.time(),
                    'salience': pattern.salience if hasattr(pattern, 'salience') else 0.5
                })
    
    def consolidate_memories(self, brain, duration_seconds: float = 60.0):
        """
        Perform memory consolidation for specified duration.
        
        This simulates sleep consolidation:
        1. Replay important patterns
        2. Strengthen consistent patterns
        3. Prune weak associations
        4. Generate dream states for exploration
        """
        start_time = time.time()
        cycles = 0
        
        while (time.time() - start_time) < duration_seconds and self.is_consolidating:
            # Phase 1: Pattern replay (40% of time)
            if (time.time() - start_time) / duration_seconds < 0.4:
                self._replay_important_patterns(brain)
                
            # Phase 2: Dream generation (40% of time)
            elif (time.time() - start_time) / duration_seconds < 0.8:
                self._generate_dream_state(brain)
                
            # Phase 3: Topology refinement (20% of time)
            else:
                self._refine_topology(brain)
            
            cycles += 1
            self.metrics.cycles_processed += 1
            
            # Brief pause to prevent CPU spinning
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            time.sleep(0.01)
        
        self.is_consolidating = False
        self.last_consolidation = time.time()
        
        # Calculate consolidation benefit
        self.metrics.consolidation_benefit = self._calculate_benefit()
        
        return self.metrics
    
    def _replay_important_patterns(self, brain):
        """Replay and strengthen important patterns."""
        if not self.recent_patterns:
            return
            
        # Select patterns by importance
        patterns_to_replay = sorted(
            self.recent_patterns,
            key=lambda p: p['salience'],
            reverse=True
        )[:5]
        
        for pattern_data in patterns_to_replay:
            pattern = pattern_data['pattern']
            
            # Create a weak imprint of the pattern
            if hasattr(pattern, 'energy') and hasattr(pattern, 'coherence'):
                # Generate field activation from pattern
                activation = self._pattern_to_field_activation(pattern)
                
                # Blend into current field with replay strength
                brain.unified_field = (
                    brain.unified_field * (1 - self.replay_strength) +
                    activation * self.replay_strength
                )
                
                self.metrics.patterns_strengthened += 1
    
    def _generate_dream_state(self, brain):
        """Generate dream-like states for creative exploration."""
        # Sample from recent patterns
        if len(self.recent_patterns) < 2:
            return
            
        # Randomly combine patterns
        idx1, idx2 = np.random.choice(len(self.recent_patterns), 2, replace=False)
        pattern1 = self.recent_patterns[idx1]['pattern']
        pattern2 = self.recent_patterns[idx2]['pattern']
        
        # Create dream activation by blending patterns with noise
        activation1 = self._pattern_to_field_activation(pattern1)
        activation2 = self._pattern_to_field_activation(pattern2)
        
        # Blend with temperature-controlled randomness
        dream_blend = np.random.rand()
        dream_field = (
            activation1 * dream_blend +
            activation2 * (1 - dream_blend) +
            torch.randn_like(brain.unified_field) * self.dream_temperature
        )
        
        # Softly imprint dream state
        brain.unified_field = brain.unified_field * 0.9 + dream_field * 0.1
        
        self.metrics.dream_sequences += 1
    
    def _refine_topology(self, brain):
        """Refine reward topology by strengthening consistent patterns."""
        if hasattr(brain, 'topology_shaper'):
            # Decay weak attractors
            if hasattr(brain.topology_shaper, 'attractors'):
                for attractor in brain.topology_shaper.attractors:
                    if hasattr(attractor, 'strength'):
                        # Weak attractors decay faster
                        if attractor.strength < self.pruning_threshold:
                            attractor.strength *= 0.9
                            self.metrics.patterns_pruned += 1
                        else:
                            # Strong attractors decay slowly
                            attractor.strength *= self.topology_decay
                
                self.metrics.topology_refined += 1
    
    def _pattern_to_field_activation(self, pattern) -> torch.Tensor:
        """Convert a pattern to field activation."""
        # Create a field-sized activation
        activation = torch.zeros(self.field_shape, device=self.device)
        
        # Use pattern features to modulate activation
        if hasattr(pattern, 'to_dict'):
            features = pattern.to_dict()
            energy = features.get('energy', 0.5)
            coherence = features.get('coherence', 0.5)
            
            # Create coherent activation in center region
            center = [s // 2 for s in self.field_shape[:3]]
            size = 4
            
            activation[
                center[0]-size:center[0]+size,
                center[1]-size:center[1]+size,
                center[2]-size:center[2]+size,
                :
            ] = energy * coherence
            
            # Add some spatial variation
            noise = torch.randn_like(activation) * 0.1
            activation += noise
            
        return activation
    
    def _calculate_benefit(self) -> float:
        """Calculate consolidation benefit score."""
        if self.metrics.cycles_processed == 0:
            return 0.0
            
        # Weighted score based on activities
        benefit = (
            self.metrics.patterns_strengthened * 0.4 +
            self.metrics.dream_sequences * 0.3 +
            self.metrics.topology_refined * 0.2 -
            self.metrics.patterns_pruned * 0.1
        ) / self.metrics.cycles_processed
        
        return max(0.0, min(1.0, benefit))
    
    def get_status(self) -> Dict[str, any]:
        """Get consolidation system status."""
        return {
            'is_consolidating': self.is_consolidating,
            'last_consolidation': self.last_consolidation,
            'metrics': {
                'cycles': self.metrics.cycles_processed,
                'strengthened': self.metrics.patterns_strengthened,
                'pruned': self.metrics.patterns_pruned,
                'dreams': self.metrics.dream_sequences,
                'benefit': self.metrics.consolidation_benefit
            },
            'pattern_memory': {
                'recent': len(self.recent_patterns),
                'important': len(self.important_patterns)
            }
        }