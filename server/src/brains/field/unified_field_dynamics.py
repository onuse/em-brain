"""
Unified Field Dynamics

Merges energy system and blended reality into a single coherent system.
Energy, confidence, and behavioral modulation are all aspects of the same
underlying field dynamics.

Key insight: Energy and confidence are two views of the same phenomenon:
- Energy: How "full" the field is with patterns
- Confidence: How well we predict what comes next
Both influence the balance between internal and external dynamics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from collections import deque
import numpy as np


class UnifiedFieldDynamics:
    """
    Unified system for field dynamics modulation.
    
    Combines:
    1. Energy emergence from field activity
    2. Confidence from prediction accuracy  
    3. Behavioral modulation (exploration/exploitation)
    4. Reality blending (internal/external balance)
    
    All emerge from the same underlying field state.
    """
    
    def __init__(self, 
                 pattern_memory_size: int = 100,
                 confidence_window: int = 50,
                 device: torch.device = torch.device('cpu')):
        """Initialize unified dynamics."""
        self.device = device
        
        # Pattern memory for novelty detection
        self.pattern_memory = deque(maxlen=pattern_memory_size)
        self.pattern_energies = deque(maxlen=pattern_memory_size)
        
        # Prediction tracking for confidence
        self.prediction_errors = deque(maxlen=confidence_window)
        self.smoothed_confidence = 0.5
        
        # Unified state tracking
        self.energy_history = deque(maxlen=1000)
        self.smoothed_energy = 0.5
        self.cycles_without_input = 0
        
        # No modes, just continuous values
        self._last_novelty = 0.0
        self._last_modulation = {}
        
    def compute_field_state(self, field: torch.Tensor) -> Dict[str, float]:
        """
        Compute all aspects of field state from the field itself.
        
        Returns:
            Dictionary with energy, novelty, confidence metrics
        """
        # Energy emerges from field activity
        energy = float(torch.mean(torch.abs(field)))
        
        # Also consider field variance (structured vs random)
        variance = float(torch.var(field))
        
        # Combined measure: high activity + high structure = high energy
        combined_energy = energy * (1.0 + variance)
        
        # Update history
        self.energy_history.append(combined_energy)
        
        # Smooth energy over time
        self.smoothed_energy = (0.95 * self.smoothed_energy + 
                                0.05 * combined_energy)
        
        return {
            'raw_energy': energy,
            'variance': variance,
            'combined_energy': combined_energy,
            'smoothed_energy': self.smoothed_energy
        }
    
    def compute_novelty(self, field: torch.Tensor) -> float:
        """
        Compute novelty as distance from known patterns.
        
        Uses actual pattern similarity, not hashes.
        """
        if len(self.pattern_memory) == 0:
            return 1.0  # Everything is novel at first
        
        # Downsample field for efficient comparison
        field_small = F.adaptive_avg_pool3d(
            field.unsqueeze(0).unsqueeze(0)[:, :, :32, :32, :32],
            output_size=(8, 8, 8)
        ).squeeze()
        
        field_flat = field_small.flatten()
        field_norm = field_flat / (torch.norm(field_flat) + 1e-8)
        
        # Compare to memory using cosine similarity
        max_similarity = 0.0
        for pattern in self.pattern_memory:
            similarity = float(torch.dot(field_norm, pattern))
            max_similarity = max(max_similarity, similarity)
        
        # Store this pattern
        self.pattern_memory.append(field_norm.detach().clone())
        
        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        self._last_novelty = novelty
        
        return novelty
    
    def update_confidence(self, prediction_error: float):
        """
        Update confidence based on prediction accuracy.
        
        Low error → high confidence → more internal dynamics
        High error → low confidence → more external focus
        """
        # Store error
        self.prediction_errors.append(prediction_error)
        
        # Compute recent average error
        if len(self.prediction_errors) > 5:
            recent_error = np.mean(list(self.prediction_errors)[-10:])
        else:
            recent_error = 0.5  # Neutral starting point
        
        # Convert error to confidence (inverse relationship)
        # Using sigmoid for smooth transition
        raw_confidence = 1.0 / (1.0 + recent_error * 5.0)
        
        # Smooth confidence changes
        self.smoothed_confidence = (0.9 * self.smoothed_confidence + 
                                   0.1 * raw_confidence)
    
    def compute_field_modulation(self, 
                                energy_state: Dict[str, float],
                                has_sensory_input: bool = True) -> Dict[str, float]:
        """
        Compute unified modulation parameters.
        
        This replaces both energy behavioral influence and blended reality.
        """
        energy = energy_state['smoothed_energy']
        novelty = self._last_novelty
        confidence = self.smoothed_confidence
        
        # Update cycles without input
        if not has_sensory_input:
            self.cycles_without_input += 1
        else:
            self.cycles_without_input = 0
        
        # Dream mode after extended idle
        is_dreaming = self.cycles_without_input > 100
        
        # Normalize energy to [0, 1]
        norm_energy = np.clip(energy / 2.0, 0.0, 1.0)
        
        # Primary modulation: balance of internal vs external
        # Influenced by BOTH energy and confidence
        # High energy OR high confidence → more internal
        # Low energy AND low confidence → more external
        internal_drive = (norm_energy + confidence) / 2.0
        
        # Dream mode overrides
        if is_dreaming:
            internal_drive = 0.95
        
        # Exploration emerges from low energy AND high novelty
        exploration_drive = (1.0 - norm_energy) * (0.5 + 0.5 * novelty)
        
        # Sensory modulation
        # Low confidence → amplify sensors (need more info)
        # High confidence → normal sensing (trust internals)
        sensory_amplification = 1.0 + (1.0 - confidence) * 0.5
        
        # Imprint strength (how strongly sensory affects field)
        # Inverse of internal drive
        imprint_strength = 0.1 + 0.7 * (1.0 - internal_drive)
        
        # Motor variability for exploration
        motor_noise = exploration_drive * 0.4
        
        # Field decay (consolidation when high energy)
        decay_rate = 0.999 - 0.01 * norm_energy
        
        # Attention bias
        # Low energy + low confidence: seek novelty
        # High energy + high confidence: prefer familiar
        attention_novelty_bias = (1.0 - internal_drive) * 0.8
        
        self._last_modulation = {
            'internal_drive': internal_drive,
            'spontaneous_weight': internal_drive,  # For compatibility
            'exploration_drive': exploration_drive,
            'sensory_amplification': sensory_amplification,
            'imprint_strength': imprint_strength,
            'motor_noise': motor_noise,
            'decay_rate': decay_rate,
            'attention_novelty_bias': attention_novelty_bias,
            'is_dreaming': is_dreaming,
            'energy': norm_energy,
            'confidence': confidence,
            'novelty': novelty
        }
        
        return self._last_modulation
    
    def modulate_field(self, 
                      field: torch.Tensor,
                      modulation: Dict[str, float]) -> torch.Tensor:
        """
        Apply modulation to field dynamics.
        
        This is mostly decay - the actual dynamics happen elsewhere.
        """
        decay_rate = modulation.get('decay_rate', 0.999)
        
        # Simple exponential decay
        return field * decay_rate
    
    def get_state_description(self) -> str:
        """Get human-readable state description."""
        if not self._last_modulation:
            return "Initializing..."
        
        m = self._last_modulation
        
        if m['is_dreaming']:
            return "💤 DREAM: Pure internal dynamics"
        
        energy_desc = "High energy" if m['energy'] > 0.6 else "Low energy"
        confidence_desc = "confident" if m['confidence'] > 0.6 else "uncertain"
        mode = "exploring" if m['exploration_drive'] > 0.5 else "exploiting"
        
        balance = f"{int(m['internal_drive']*100)}% internal"
        
        return f"🧠 {energy_desc}, {confidence_desc}, {mode} | {balance}"