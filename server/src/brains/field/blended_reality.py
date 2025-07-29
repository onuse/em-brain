"""
Blended Reality Implementation

Implements confidence-based blending of spontaneous dynamics (fantasy) 
and sensory input (reality) in a single unified field.
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BlendedRealityConfig:
    """Configuration for reality blending behavior."""
    # Base strength for sensory imprints (when confidence = 0)
    base_imprint_strength: float = 0.8
    
    # Minimum strength for sensory imprints (when confidence = 1)
    min_imprint_strength: float = 0.1
    
    # Base weight for spontaneous activity
    base_spontaneous_weight: float = 0.3
    
    # Maximum weight for spontaneous activity (when confidence = 1)
    max_spontaneous_weight: float = 0.9
    
    # Confidence smoothing rate (0 = no smoothing, 1 = no change)
    confidence_smoothing: float = 0.9
    
    # Cycles without input before entering dream mode
    dream_threshold: int = 100
    
    # Dream mode spontaneous weight
    dream_spontaneous_weight: float = 0.95


class BlendedReality:
    """
    Manages the blending of spontaneous dynamics and sensory input
    based on prediction confidence.
    
    High confidence → More fantasy (spontaneous dynamics)
    Low confidence → More reality (sensory input)
    """
    
    def __init__(self, config: Optional[BlendedRealityConfig] = None):
        """Initialize blended reality system."""
        self.config = config or BlendedRealityConfig()
        
        # State tracking
        self._smoothed_confidence = 0.5
        self._cycles_without_input = 0
        self._dream_mode = False
        self._last_sensory_strength = 0.0
        
    def update_confidence(self, raw_confidence: float) -> float:
        """
        Apply temporal smoothing to confidence changes.
        
        Args:
            raw_confidence: Current prediction confidence (0-1)
            
        Returns:
            Smoothed confidence value
        """
        # Temporal smoothing for gradual transitions
        # Handle both field names for compatibility
        smoothing = getattr(self.config, 'confidence_smoothing', 
                          getattr(self.config, 'confidence_smoothing_rate', 0.9))
        self._smoothed_confidence = (
            smoothing * self._smoothed_confidence +
            (1.0 - smoothing) * raw_confidence
        )
        
        return self._smoothed_confidence
    
    def calculate_imprint_strength(self, 
                                 base_intensity: float,
                                 has_sensory_input: bool = True) -> float:
        """
        Calculate sensory imprint strength based on confidence.
        
        Args:
            base_intensity: Base field intensity from experience
            has_sensory_input: Whether we have actual sensory input
            
        Returns:
            Scaled imprint strength
        """
        if not has_sensory_input:
            self._cycles_without_input += 1
            # Check for dream mode transition
            if self._cycles_without_input >= self.config.dream_threshold:
                self._dream_mode = True
            return 0.0
        else:
            self._cycles_without_input = 0
            self._dream_mode = False
        
        # Linear interpolation based on confidence
        # High confidence = weak imprint, Low confidence = strong imprint
        strength_range = self.config.base_imprint_strength - self.config.min_imprint_strength
        imprint_strength = (
            self.config.base_imprint_strength - 
            (strength_range * self._smoothed_confidence)
        )
        
        # Scale by base intensity
        final_strength = imprint_strength * base_intensity
        self._last_sensory_strength = final_strength
        
        return final_strength
    
    def calculate_spontaneous_weight(self) -> float:
        """
        Calculate weight for spontaneous activity based on confidence.
        
        Returns:
            Weight for spontaneous dynamics contribution
        """
        if self._dream_mode:
            # Pure fantasy in dream mode
            return getattr(self.config, 'dream_spontaneous_weight', 0.95)
        
        # Linear interpolation based on confidence
        # High confidence = strong spontaneous, Low confidence = weak spontaneous
        # Handle both field names for compatibility
        base_weight = getattr(self.config, 'base_spontaneous_weight', 0.3)
        max_weight = getattr(self.config, 'max_spontaneous_weight', 0.9)
        weight_range = max_weight - base_weight
        spontaneous_weight = (
            base_weight + 
            (weight_range * self._smoothed_confidence)
        )
        
        return spontaneous_weight
    
    def get_blend_state(self) -> dict:
        """Get current blending state for monitoring."""
        spontaneous_weight = self.calculate_spontaneous_weight()
        
        # Calculate actual reality/fantasy balance based on spontaneous weight
        # When spontaneous_weight is high, we have more fantasy
        fantasy_percent = spontaneous_weight * 100
        reality_percent = (1.0 - spontaneous_weight) * 100
        
        return {
            'smoothed_confidence': self._smoothed_confidence,
            'imprint_strength': self._last_sensory_strength,
            'spontaneous_weight': spontaneous_weight,
            'dream_mode': self._dream_mode,
            'cycles_without_input': self._cycles_without_input,
            'reality_balance': f"{reality_percent:.0f}% reality / {fantasy_percent:.0f}% fantasy"
        }
    
    def reset_dream_state(self):
        """Reset dream mode state (e.g., when robot wakes up)."""
        self._dream_mode = False
        self._cycles_without_input = 0


def integrate_blended_reality(brain):
    """
    Integrate blended reality into existing brain.
    
    This modifies the brain's imprinting and spontaneous dynamics
    to use confidence-based blending.
    
    Args:
        brain: DynamicUnifiedFieldBrain instance
        
    Returns:
        Modified brain with blended reality
    """
    # Create blended reality system
    brain.blended_reality = BlendedReality()
    
    # Store original methods
    original_imprint = brain._imprint_unified_experience
    original_evolve = brain._evolve_unified_field
    
    def imprint_with_blending(experience):
        """Modified imprint that uses confidence-based strength."""
        # Update smoothed confidence
        brain.blended_reality.update_confidence(brain._current_prediction_confidence)
        
        # Calculate confidence-based imprint strength
        base_intensity = experience.field_intensity
        
        # Check for meaningful input by looking at variance from neutral (0.5)
        # Neutral input has all sensors at 0.5, real input varies from this
        has_meaningful_input = False
        # Handle both field names for compatibility
        raw_input = getattr(experience, 'raw_input_stream', None)
        if raw_input is None:
            raw_input = getattr(experience, 'raw_sensory_input', None)
        if raw_input is not None:
            # Calculate variance from neutral baseline
            neutral_baseline = 0.5
            variance_from_neutral = torch.mean(torch.abs(raw_input[:-1] - neutral_baseline))
            # Consider it meaningful if variance > 0.05 (allowing for some noise)
            has_meaningful_input = variance_from_neutral > 0.05
        
        # Only imprint if we have meaningful input
        if has_meaningful_input:
            # Get scaled imprint strength
            scaled_intensity = brain.blended_reality.calculate_imprint_strength(
                base_intensity, True
            )
            
            # Temporarily modify field intensity
            original_intensity = experience.field_intensity
            experience.field_intensity = scaled_intensity
            
            # Call original imprint
            original_imprint(experience)
            
            # Restore original intensity
            experience.field_intensity = original_intensity
        else:
            # No meaningful input - skip imprinting entirely
            brain.blended_reality.calculate_imprint_strength(base_intensity, False)
    
    def evolve_with_weighted_spontaneous():
        """Modified evolution that weights spontaneous activity."""
        # Call original evolution first (which handles decay and diffusion)
        original_evolve()
        
        # The spontaneous dynamics are already applied in _evolve_unified_field
        # We just need to adjust the blend based on confidence
        # This happens through the sensory gating calculation in the main evolution
        
        # Log state occasionally
        if brain.brain_cycles % 100 == 0 and hasattr(brain, 'blended_reality'):
            state = brain.blended_reality.get_blend_state()
            if not brain.quiet_mode:
                print(f"🌀 BLEND: {state['reality_balance']} | "
                      f"Dream: {state['dream_mode']} | "
                      f"Confidence: {state['smoothed_confidence']:.2f}")
    
    # Replace methods
    brain._imprint_unified_experience = imprint_with_blending
    brain._evolve_unified_field = evolve_with_weighted_spontaneous
    
    return brain