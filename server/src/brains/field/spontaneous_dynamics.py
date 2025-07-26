"""
Spontaneous Field Dynamics

Minimal biological primitive that enables internal activity without sensory input.
Based on how even simple nervous systems maintain ongoing activity.
"""

import torch
import numpy as np
from typing import Optional, Tuple


class SpontaneousDynamics:
    """
    Adds spontaneous activity to field dynamics.
    
    Biological inspiration:
    - Even isolated neurons show spontaneous firing
    - Neural networks maintain baseline activity
    - This activity isn't random - it explores state space
    - Creates the substrate for internal experience
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 resting_potential: float = 0.01,
                 spontaneous_rate: float = 0.001,
                 coherence_scale: float = 3.0,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize spontaneous dynamics.
        
        Args:
            field_shape: Shape of the unified field
            resting_potential: Baseline activation level
            spontaneous_rate: Rate of spontaneous activation
            coherence_scale: Spatial scale of coherent activations
            device: Computation device
        """
        self.field_shape = field_shape
        self.resting_potential = resting_potential
        self.spontaneous_rate = spontaneous_rate
        self.coherence_scale = coherence_scale
        self.device = device
        
        # Traveling wave parameters (creates coherent internal dynamics)
        self.wave_vectors = self._initialize_wave_vectors()
        self.phase_offset = 0.0
        
        # Criticality parameters (edge of chaos)
        self.inhibition_strength = 1.02  # Slightly above critical
        self.local_excitation = 1.05     # Local positive feedback
        
    def _initialize_wave_vectors(self) -> torch.Tensor:
        """Initialize random wave vectors for traveling waves."""
        # Create 3-5 traveling wave patterns
        n_waves = np.random.randint(3, 6)
        
        # Only use first 3 dimensions for spatial waves
        wave_dims = min(3, len(self.field_shape))
        wave_vectors = torch.randn(n_waves, wave_dims, device=self.device)
        
        # Normalize to control wave speed
        wave_vectors = wave_vectors / torch.norm(wave_vectors, dim=1, keepdim=True)
        
        return wave_vectors
    
    def generate_spontaneous_activity(self, 
                                    current_field: torch.Tensor,
                                    sensory_gating: float = 1.0) -> torch.Tensor:
        """
        Generate spontaneous activity for the field.
        
        Args:
            current_field: Current field state
            sensory_gating: 0-1, reduces spontaneous activity when processing input
            
        Returns:
            Spontaneous activation to add to field
        """
        # 1. Baseline fluctuations (like resting potential variations)
        baseline_noise = torch.randn_like(current_field) * self.spontaneous_rate
        
        # 2. Coherent traveling waves (organized spontaneous patterns)
        waves = self._generate_traveling_waves(current_field.shape)
        
        # 3. Local recurrence (what fires together, wires together)
        recurrent = self._local_recurrence(current_field)
        
        # 4. Homeostatic pressure (maintain average activity)
        homeostatic = self._homeostatic_drive(current_field)
        
        # Combine all spontaneous sources
        spontaneous = (baseline_noise + 
                      waves * 2.0 + 
                      recurrent * self.local_excitation +
                      homeostatic)
        
        # Gate by sensory input (less spontaneous activity during strong input)
        spontaneous *= (1.0 - sensory_gating * 0.8)
        
        # Maintain critical dynamics (edge of chaos)
        spontaneous = self._apply_criticality(spontaneous, current_field)
        
        return spontaneous
    
    def _generate_traveling_waves(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate coherent traveling wave patterns."""
        waves = torch.zeros(shape, device=self.device)
        
        # Create coordinate grids for first 3 dimensions
        coords = []
        for i in range(min(3, len(shape))):
            coord = torch.arange(shape[i], device=self.device, dtype=torch.float32)
            coord = coord.view([-1 if j == i else 1 for j in range(len(shape))])
            coords.append(coord)
        
        # Generate each traveling wave
        for wave_vec in self.wave_vectors:
            # Calculate phase at each position
            phase = self.phase_offset
            for i, (coord, k) in enumerate(zip(coords, wave_vec)):
                phase = phase + k * coord / self.coherence_scale
            
            # Add wave contribution
            wave_amplitude = self.spontaneous_rate * 5.0
            waves += torch.sin(phase) * wave_amplitude
        
        # Slowly advance phase
        self.phase_offset += 0.1
        
        # Apply to full field shape
        if len(shape) > 3:
            # Broadcast to higher dimensions with decay
            for dim in range(3, len(shape)):
                decay = 0.5 ** (dim - 2)
                waves = waves.unsqueeze(-1).expand(shape) * decay
        
        return waves
    
    def _local_recurrence(self, field: torch.Tensor) -> torch.Tensor:
        """Local positive feedback - active regions stay active."""
        # Simple local excitation
        return field * self.spontaneous_rate * 10.0
    
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


# Minimal integration into existing brain
def add_spontaneous_dynamics(brain):
    """
    Add spontaneous dynamics to existing brain.
    
    This is a minimal change that enables internal activity.
    """
    
    # Create spontaneous dynamics module
    brain.spontaneous = SpontaneousDynamics(
        field_shape=brain.unified_field.shape,
        device=brain.device
    )
    
    # Override the field evolution method
    original_evolve = brain._evolve_unified_field
    
    def evolve_with_spontaneous():
        # Original evolution
        original_evolve()
        
        # Add spontaneous activity
        # Reduce spontaneous activity if recent sensory input
        sensory_gating = 0.0
        if hasattr(brain, '_last_imprint_indices') and brain._last_imprint_indices:
            # Recent sensory input gates spontaneous activity
            recency = min(1.0, (brain.brain_cycles % 10) / 10.0)
            sensory_gating = 1.0 - recency
        
        spontaneous = brain.spontaneous.generate_spontaneous_activity(
            brain.unified_field,
            sensory_gating
        )
        
        brain.unified_field += spontaneous
    
    brain._evolve_unified_field = evolve_with_spontaneous
    
    return brain