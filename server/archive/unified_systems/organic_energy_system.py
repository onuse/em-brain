"""
Organic Energy System

Energy emerges naturally from field dynamics rather than being tracked separately.
No artificial thresholds or modes - just smooth, continuous relationships.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from collections import deque
import numpy as np


class OrganicEnergySystem:
    """
    Energy as an emergent property of field dynamics.
    
    Key principles:
    1. Energy = field activity intensity
    2. Novelty = distance in pattern space  
    3. Behavior emerges smoothly from energy
    4. No artificial modes or thresholds
    """
    
    def __init__(self, pattern_memory_size: int = 100, device: torch.device = torch.device('cpu')):
        """Initialize organic energy system."""
        self.device = device
        
        # Pattern memory as actual patterns, not hashes
        self.pattern_memory = deque(maxlen=pattern_memory_size)
        self.pattern_energies = deque(maxlen=pattern_memory_size)
        
        # Smooth energy tracking
        self.energy_history = deque(maxlen=1000)
        self.smoothed_energy = 0.5
        
        # No modes, just continuous values
        self._last_novelty = 0.0
        self._last_satisfaction = 0.0
        
    def compute_field_energy(self, field: torch.Tensor) -> float:
        """
        Energy emerges from field dynamics.
        
        High activity = high energy (satiated)
        Low activity = low energy (hungry)
        """
        # Energy is simply the intensity of field activity
        # Could also use gradient magnitude for "how much is changing"
        energy = float(torch.mean(torch.abs(field)))
        
        # Track history for smoothing
        self.energy_history.append(energy)
        
        # Smooth over long timescale (biological inertia)
        if len(self.energy_history) > 100:
            self.smoothed_energy = float(np.mean(list(self.energy_history)[-500:]))
        else:
            self.smoothed_energy = energy
            
        return energy
        
    def compute_pattern_novelty(self, pattern: torch.Tensor) -> float:
        """
        Novelty as distance from known patterns.
        
        No hashing - just continuous similarity in high-D space.
        """
        if len(self.pattern_memory) == 0:
            # First pattern is maximally novel
            self.pattern_memory.append(pattern.detach().clone())
            self.pattern_energies.append(1.0)
            self._last_novelty = 1.0
            return 1.0
            
        # Flatten pattern for comparison
        pattern_flat = pattern.flatten()
        
        # Compute similarity to all known patterns
        similarities = []
        for known_pattern in self.pattern_memory:
            known_flat = known_pattern.flatten()
            
            # Handle different sizes gracefully
            min_size = min(len(pattern_flat), len(known_flat))
            if min_size > 0:
                similarity = F.cosine_similarity(
                    pattern_flat[:min_size].unsqueeze(0),
                    known_flat[:min_size].unsqueeze(0),
                    dim=1
                )
                similarities.append(float(similarity))
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        # Add sufficiently novel patterns to memory
        if novelty > 0.3:  # Soft threshold
            self.pattern_memory.append(pattern.detach().clone())
            self.pattern_energies.append(novelty)
            
        self._last_novelty = novelty
        return novelty
        
    def compute_behavioral_influence(self, energy: float) -> Dict[str, float]:
        """
        Smooth behavioral influence from energy.
        
        No modes or thresholds - just continuous functions.
        """
        # Normalized energy (assumes typical range 0.1 to 1.0)
        norm_energy = np.clip((energy - 0.1) / 0.9, 0, 1)
        
        # Smooth transitions using sigmoid-like functions
        # Low energy → high exploration, high energy → low exploration
        exploration_drive = 1.0 / (1.0 + np.exp(5 * (norm_energy - 0.5)))
        
        # Sensory amplification when hungry (low energy)
        sensory_amplification = 1.0 + exploration_drive
        
        # Motor noise for exploration
        motor_noise = exploration_drive * 0.4
        
        # Spontaneous weight increases with energy (more fantasy when satiated)
        spontaneous_weight = 0.3 + 0.5 * norm_energy
        
        # Decay rate - faster when high energy (consolidation)
        decay_rate = 0.999 - 0.01 * norm_energy
        
        # Attention bias smoothly transitions
        # Low energy: novelty-seeking, High energy: familiar-seeking
        attention_novelty_bias = exploration_drive
        
        return {
            'exploration_drive': exploration_drive,
            'sensory_amplification': sensory_amplification,
            'motor_noise': motor_noise,
            'spontaneous_weight': spontaneous_weight,
            'decay_rate': decay_rate,
            'attention_novelty_bias': attention_novelty_bias
        }
        
    def process_field_dynamics(self,
                             field: torch.Tensor,
                             sensory_pattern: Optional[torch.Tensor] = None,
                             prediction_error: float = 0.5,
                             reward: float = 0.0) -> Dict[str, Any]:
        """
        Process field and return energy-based recommendations.
        
        Everything emerges from field dynamics.
        """
        # Compute current energy from field
        current_energy = self.compute_field_energy(field)
        
        # Compute novelty if pattern provided
        novelty = 0.0
        if sensory_pattern is not None and sensory_pattern.numel() > 0:
            novelty = self.compute_pattern_novelty(sensory_pattern)
            
        # Satisfaction from good predictions (inverse of error)
        satisfaction = (1.0 - prediction_error) * 0.2
        self._last_satisfaction = satisfaction
        
        # Natural energy modulation from experiences
        # Novelty and rewards add energy, satisfaction maintains it
        energy_delta = novelty * 0.1 + abs(reward) * 0.2 + satisfaction * 0.05
        
        # Get behavioral influence
        behavior = self.compute_behavioral_influence(self.smoothed_energy)
        
        return {
            'current_energy': current_energy,
            'smoothed_energy': self.smoothed_energy,
            'energy_delta': energy_delta,
            'novelty': novelty,
            'satisfaction': satisfaction,
            'behavior': behavior,
            'pattern_memory_size': len(self.pattern_memory)
        }
        
    def modulate_field(self, field: torch.Tensor, behavior: Dict[str, float]) -> torch.Tensor:
        """
        Apply energy-based modulation to field.
        
        Minimal intervention - let field dynamics do most of the work.
        """
        # Apply decay based on energy state
        decay = behavior['decay_rate']
        field = field * decay
        
        # Natural pruning through decay - no explicit masking needed
        # Weak patterns naturally fade below numerical precision
        
        return field
        
    def should_consolidate_memory(self) -> bool:
        """
        Determine if memory consolidation should occur.
        
        Based on energy level and pattern memory fullness.
        """
        memory_pressure = len(self.pattern_memory) / self.pattern_memory.maxlen
        energy_pressure = self.smoothed_energy
        
        # Consolidate when both memory and energy are high
        return memory_pressure > 0.8 and energy_pressure > 0.7
        
    def consolidate_patterns(self):
        """
        Consolidate pattern memory by merging similar patterns.
        
        Natural forgetting of low-energy patterns.
        """
        if len(self.pattern_memory) < 10:
            return
            
        # Remove patterns with lowest energy
        # (Natural forgetting of unimportant patterns)
        energy_threshold = np.percentile(list(self.pattern_energies), 20)
        
        new_memory = deque(maxlen=self.pattern_memory.maxlen)
        new_energies = deque(maxlen=self.pattern_energies.maxlen)
        
        for pattern, energy in zip(self.pattern_memory, self.pattern_energies):
            if energy > energy_threshold:
                new_memory.append(pattern)
                new_energies.append(energy * 0.9)  # Gentle decay
                
        self.pattern_memory = new_memory
        self.pattern_energies = new_energies