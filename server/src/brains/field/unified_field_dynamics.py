"""
Unified Field Dynamics

Merges energy system, blended reality, and spontaneous dynamics into a single 
coherent system. Energy, confidence, behavioral modulation, and spontaneous 
activity are all aspects of the same underlying field dynamics.

Key insights:
- Energy: How "full" the field is with patterns
- Confidence: How well we predict what comes next
- Spontaneous: Internal activity that maintains field liveliness
All influence the balance between internal and external dynamics.
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
                 field_shape: Tuple[int, ...],
                 pattern_memory_size: int = 100,
                 confidence_window: int = 50,
                 spontaneous_rate: float = 0.001,
                 resting_potential: float = 0.01,
                 device: torch.device = torch.device('cpu')):
        """Initialize unified dynamics."""
        self.device = device
        self.field_shape = field_shape
        
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
        
        # Spontaneous dynamics parameters
        self.spontaneous_rate = spontaneous_rate
        self.resting_potential = resting_potential
        self.coherence_scale = 3.0
        
        # Traveling wave parameters for spontaneous activity
        self.wave_vectors = self._initialize_wave_vectors()
        self.phase_offset = 0.0
        
        # Criticality parameters (edge of chaos)
        self.inhibition_strength = 1.02
        self.local_excitation = 1.05
        
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
        # Use slicing instead of adaptive_avg_pool3d for MPS compatibility
        field_small = field[:8, :8, :8, :8] if field.shape[0] >= 8 else field
        
        field_flat = field_small.flatten()
        
        # Sample if too large
        if len(field_flat) > 512:
            indices = torch.linspace(0, len(field_flat)-1, 512, dtype=torch.long, device=field.device)
            field_flat = field_flat[indices]
        
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
    
    def _initialize_wave_vectors(self) -> torch.Tensor:
        """Initialize random wave vectors for spontaneous traveling waves."""
        # Create 3-5 traveling wave patterns
        n_waves = np.random.randint(3, 6)
        
        # Only use first 3 dimensions for spatial waves
        wave_dims = min(3, len(self.field_shape))
        wave_vectors = torch.randn(n_waves, wave_dims, device=self.device)
        
        # Normalize to control wave speed
        wave_vectors = wave_vectors / torch.norm(wave_vectors, dim=1, keepdim=True)
        
        return wave_vectors
    
    def evolve_field(self, field: torch.Tensor) -> torch.Tensor:
        """
        Evolve field dynamics including spontaneous activity.
        
        This integrates spontaneous dynamics directly into field evolution,
        making it part of the unified system rather than a separate add-on.
        
        Args:
            field: Current field state
            
        Returns:
            Evolved field with all dynamics applied
        """
        # Get current modulation state
        modulation = self._last_modulation
        if not modulation:
            # Compute if not available
            energy_state = self.compute_field_state(field)
            modulation = self.compute_field_modulation(energy_state)
        
        # 1. Apply decay
        field = field * modulation.get('decay_rate', 0.999)
        
        # 2. Generate spontaneous activity
        # Spontaneous weight determines the balance
        spontaneous_weight = modulation.get('spontaneous_weight', 0.5)
        sensory_gating = 1.0 - spontaneous_weight
        
        # Baseline fluctuations
        baseline_noise = torch.randn_like(field) * self.spontaneous_rate
        
        # Coherent traveling waves
        waves = self._generate_traveling_waves(field.shape)
        
        # Local recurrence (active regions stay active)
        recurrent = field * self.spontaneous_rate * 10.0 * self.local_excitation
        
        # Homeostatic pressure
        homeostatic = self._homeostatic_drive(field)
        
        # Combine spontaneous sources
        spontaneous = (baseline_noise + 
                      waves * 2.0 + 
                      recurrent +
                      homeostatic)
        
        # Gate by sensory input state
        spontaneous *= sensory_gating
        
        # Apply criticality to prevent runaway dynamics
        spontaneous = self._apply_criticality(spontaneous, field)
        
        # 3. Add spontaneous to field
        field = field + spontaneous
        
        return field
    
    def _generate_traveling_waves(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate coherent traveling wave patterns."""
        # Start with zeros for first 3 dimensions only
        wave_shape = shape[:3] if len(shape) >= 3 else shape
        waves = torch.zeros(wave_shape, device=self.device)
        
        # Create coordinate grids for first 3 dimensions
        coords = []
        for i in range(len(wave_shape)):
            coord = torch.arange(wave_shape[i], device=self.device, dtype=torch.float32)
            coord = coord.view([-1 if j == i else 1 for j in range(len(wave_shape))])
            coords.append(coord)
        
        # Generate each traveling wave
        for wave_vec in self.wave_vectors:
            # Calculate phase at each position
            phase = self.phase_offset
            for i in range(len(coords)):
                if i < len(wave_vec):
                    phase = phase + wave_vec[i] * coords[i] / self.coherence_scale
            
            # Add wave contribution
            wave_amplitude = self.spontaneous_rate * 5.0
            waves += torch.sin(phase) * wave_amplitude
        
        # Slowly advance phase
        self.phase_offset += 0.1
        
        # Expand to full field shape if needed
        if len(shape) > len(wave_shape):
            # Add singleton dimensions for remaining dims
            for _ in range(len(shape) - len(wave_shape)):
                waves = waves.unsqueeze(-1)
            
            # Expand to full shape
            waves = waves.expand(shape).clone()
            
            # Apply decay for higher dimensions
            waves *= 0.1
        
        return waves
    
    def _homeostatic_drive(self, field: torch.Tensor) -> torch.Tensor:
        """Maintain average activity level."""
        current_mean = torch.mean(torch.abs(field))
        target_mean = self.resting_potential
        
        # Push toward target activity
        drive = (target_mean - current_mean) * self.spontaneous_rate
        
        return torch.ones_like(field) * drive
    
    def _apply_criticality(self, spontaneous: torch.Tensor, 
                          current_field: torch.Tensor) -> torch.Tensor:
        """Maintain critical dynamics (edge of chaos)."""
        # Global inhibition to prevent runaway
        total_activity = torch.sum(torch.abs(current_field))
        inhibition = torch.sigmoid(total_activity / current_field.numel() - 0.1)
        
        # Apply inhibition
        spontaneous = spontaneous * (1.0 - inhibition * 0.5)
        
        # Add small anti-correlation to prevent synchrony
        mean_activity = torch.mean(current_field)
        spontaneous = spontaneous - mean_activity * self.inhibition_strength
        
        return spontaneous
    
    def reset_waves(self):
        """Reset traveling wave patterns (like sleep state transitions)."""
        self.wave_vectors = self._initialize_wave_vectors()
        self.phase_offset = 0.0