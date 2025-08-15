#!/usr/bin/env python3
"""
Vocal Interface - Hardware Abstraction for Digital Vocal Cords

Defines the interface contract between brain emotional systems and
vocal hardware. Enables testing with mock implementations and enforces
safety constraints at the HAL level.

This is the first HAL component, treating vocal cords as a complex
actuator parallel to camera as a complex sensor.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class VocalParameters:
    """Digital vocal cord parameter set."""
    
    # Core vocal characteristics
    fundamental_frequency: float  # Hz (80-1000 typical animal range)
    amplitude: float             # 0.0-1.0 normalized volume
    duration: float              # seconds
    
    # Harmonic content (richness of sound)
    harmonics: List[float]       # Amplitude of each harmonic (0.0-1.0)
    
    # Modulation (natural variation)
    frequency_modulation: Tuple[float, float]  # (rate_hz, depth_cents)
    amplitude_modulation: Tuple[float, float]  # (rate_hz, depth_0_to_1)
    
    # Vocal quality
    noise_component: float       # 0.0-0.5 breathiness/roughness
    attack_time: float          # seconds (0.01-0.5)
    decay_time: float           # seconds (0.01-2.0)
    
    # Resonant characteristics (formants)
    formant_frequencies: List[float]  # Hz [F1, F2, F3] typical 250-4000Hz
    formant_amplitudes: List[float]   # 0.0-1.0 relative amplitudes


@dataclass
class VocalSafetyConstraints:
    """Safety limits for vocal output."""
    
    max_volume: float = 0.8          # Maximum amplitude (0.0-1.0)
    max_duration: float = 5.0        # Maximum single vocalization (seconds)
    max_frequency: float = 2000.0    # Maximum fundamental frequency (Hz)
    min_frequency: float = 50.0      # Minimum fundamental frequency (Hz)
    max_duty_cycle: float = 0.3      # Maximum fraction of time vocalizing
    cooldown_period: float = 0.1     # Minimum silence between vocalizations


class VocalInterface(ABC):
    """Abstract interface for digital vocal cord hardware."""
    
    def __init__(self, safety_constraints: Optional[VocalSafetyConstraints] = None):
        """Initialize vocal interface with safety constraints."""
        self.safety_constraints = safety_constraints or VocalSafetyConstraints()
        self._is_active = False
        self._last_vocalization_time = 0.0
        
    @abstractmethod
    def initialize_vocal_system(self) -> bool:
        """
        Initialize the vocal hardware/software system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def synthesize_vocalization(self, params: VocalParameters) -> bool:
        """
        Generate and output a vocalization with given parameters.
        
        Args:
            params: VocalParameters defining the sound characteristics
            
        Returns:
            bool: True if vocalization started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_vocalization(self) -> bool:
        """
        Immediately stop any current vocalization.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def is_vocalizing(self) -> bool:
        """
        Check if currently producing sound.
        
        Returns:
            bool: True if actively vocalizing, False otherwise
        """
        pass
    
    @abstractmethod
    def get_vocal_capabilities(self) -> Dict[str, any]:
        """
        Get information about vocal system capabilities.
        
        Returns:
            Dict containing capability information
        """
        pass
    
    @abstractmethod
    def set_volume(self, volume: float) -> bool:
        """
        Set overall vocal output volume.
        
        Args:
            volume: Volume level (0.0-1.0)
            
        Returns:
            bool: True if volume set successfully, False otherwise
        """
        pass
    
    def validate_parameters(self, params: VocalParameters) -> bool:
        """
        Validate vocal parameters against safety constraints.
        
        Args:
            params: VocalParameters to validate
            
        Returns:
            bool: True if parameters are safe, False otherwise
        """
        constraints = self.safety_constraints
        
        # Check amplitude limits
        if params.amplitude > constraints.max_volume:
            return False
            
        # Check duration limits  
        if params.duration > constraints.max_duration:
            return False
            
        # Check frequency limits
        if (params.fundamental_frequency > constraints.max_frequency or 
            params.fundamental_frequency < constraints.min_frequency):
            return False
            
        # Check harmonic amplitudes
        if any(h > 1.0 or h < 0.0 for h in params.harmonics):
            return False
            
        # Check noise component
        if params.noise_component > 0.5 or params.noise_component < 0.0:
            return False
            
        # Check timing parameters
        if (params.attack_time > 0.5 or params.attack_time < 0.001 or
            params.decay_time > 2.0 or params.decay_time < 0.001):
            return False
            
        return True
    
    def apply_safety_constraints(self, params: VocalParameters) -> VocalParameters:
        """
        Apply safety constraints to parameters, clamping to safe values.
        
        Args:
            params: Original VocalParameters
            
        Returns:
            VocalParameters: Safety-constrained parameters
        """
        constraints = self.safety_constraints
        
        # Create safe copy
        safe_params = VocalParameters(
            fundamental_frequency=max(constraints.min_frequency, 
                                    min(constraints.max_frequency, params.fundamental_frequency)),
            amplitude=max(0.0, min(constraints.max_volume, params.amplitude)),
            duration=max(0.01, min(constraints.max_duration, params.duration)),
            harmonics=[max(0.0, min(1.0, h)) for h in params.harmonics],
            frequency_modulation=params.frequency_modulation,
            amplitude_modulation=params.amplitude_modulation,
            noise_component=max(0.0, min(0.5, params.noise_component)),
            attack_time=max(0.001, min(0.5, params.attack_time)),
            decay_time=max(0.001, min(2.0, params.decay_time)),
            formant_frequencies=params.formant_frequencies.copy(),
            formant_amplitudes=[max(0.0, min(1.0, a)) for a in params.formant_amplitudes]
        )
        
        return safe_params


class EmotionalVocalMapper:
    """Maps brain emotional states to vocal parameters."""
    
    def __init__(self):
        """Initialize emotional vocal mapping system."""
        self.emotional_patterns = self._create_emotional_patterns()
    
    def _create_emotional_patterns(self) -> Dict[str, VocalParameters]:
        """Create mapping from emotional states to vocal characteristics."""
        
        patterns = {
            'curiosity': VocalParameters(
                fundamental_frequency=400.0,
                amplitude=0.3,
                duration=0.2,
                harmonics=[1.0, 0.3, 0.1],
                frequency_modulation=(5.0, 20.0),  # Rising questioning tone
                amplitude_modulation=(0.0, 0.0),
                noise_component=0.1,
                attack_time=0.01,
                decay_time=0.1,
                formant_frequencies=[600.0, 1200.0, 2400.0],
                formant_amplitudes=[0.8, 0.6, 0.3]
            ),
            
            'confidence': VocalParameters(
                fundamental_frequency=300.0,
                amplitude=0.5,
                duration=0.4,
                harmonics=[1.0, 0.5, 0.3, 0.1],
                frequency_modulation=(0.0, 0.0),   # Steady, assured tone
                amplitude_modulation=(2.0, 0.1),   # Slight rhythmic pulse
                noise_component=0.05,
                attack_time=0.02,
                decay_time=0.3,
                formant_frequencies=[400.0, 800.0, 1600.0],
                formant_amplitudes=[1.0, 0.7, 0.4]
            ),
            
            'confusion': VocalParameters(
                fundamental_frequency=250.0,
                amplitude=0.4,
                duration=0.6,
                harmonics=[1.0, 0.2, 0.4, 0.1, 0.3],  # Irregular harmonics
                frequency_modulation=(3.0, 50.0),      # Warbling uncertainty
                amplitude_modulation=(4.0, 0.3),       # Unsteady volume
                noise_component=0.25,                   # Breathy confusion
                attack_time=0.05,
                decay_time=0.4,
                formant_frequencies=[350.0, 700.0, 1400.0],
                formant_amplitudes=[0.7, 0.5, 0.2]
            ),
            
            'achievement': VocalParameters(
                fundamental_frequency=500.0,
                amplitude=0.6,
                duration=1.0,
                harmonics=[1.0, 0.7, 0.5, 0.3, 0.2],  # Rich, bright harmonics
                frequency_modulation=(1.0, 10.0),      # Slight triumphant rise
                amplitude_modulation=(0.5, 0.05),      # Steady celebration
                noise_component=0.02,                   # Clean, pure tone
                attack_time=0.01,
                decay_time=0.8,
                formant_frequencies=[500.0, 1000.0, 2000.0],
                formant_amplitudes=[1.0, 0.8, 0.6]
            ),
            
            'distress': VocalParameters(
                fundamental_frequency=600.0,
                amplitude=0.7,
                duration=0.3,
                harmonics=[1.0, 0.1, 0.6, 0.1, 0.4],  # Harsh, urgent harmonics
                frequency_modulation=(8.0, 30.0),      # Urgent fluctuation
                amplitude_modulation=(6.0, 0.4),       # Alarmed variation
                noise_component=0.35,                   # Harsh, distressed quality
                attack_time=0.005,                      # Sharp onset
                decay_time=0.1,
                formant_frequencies=[700.0, 1400.0, 2800.0],
                formant_amplitudes=[0.9, 0.8, 0.7]
            ),
            
            'contentment': VocalParameters(
                fundamental_frequency=200.0,
                amplitude=0.2,
                duration=0.8,
                harmonics=[1.0, 0.6, 0.3, 0.1],       # Warm, gentle harmonics
                frequency_modulation=(0.5, 5.0),       # Very gentle variation
                amplitude_modulation=(1.0, 0.05),      # Soft breathing rhythm
                noise_component=0.08,                   # Slightly breathy, relaxed
                attack_time=0.1,                        # Gentle onset
                decay_time=0.6,
                formant_frequencies=[300.0, 600.0, 1200.0],
                formant_amplitudes=[0.8, 0.5, 0.2]
            )
        }
        
        return patterns
    
    def map_brain_state_to_vocal_params(self, brain_state: Dict[str, any]) -> VocalParameters:
        """
        Convert brain emotional state to vocal parameters.
        
        Args:
            brain_state: Dictionary containing brain state information
            
        Returns:
            VocalParameters: Appropriate vocal characteristics for the state
        """
        # Extract relevant emotional indicators from brain state
        prediction_confidence = brain_state.get('prediction_confidence', 0.5)
        prediction_method = brain_state.get('prediction_method', 'unknown')
        total_experiences = brain_state.get('total_experiences', 0)
        consensus_rate = brain_state.get('consensus_rate', 0.0)
        
        # Determine primary emotional state
        if prediction_method == 'bootstrap_random' or total_experiences < 20:
            # Robot is still learning, express curiosity
            emotional_state = 'curiosity'
        elif prediction_confidence > 0.8 and consensus_rate > 0.7:
            # Robot is confident and successful
            emotional_state = 'confidence'
        elif prediction_confidence < 0.3:
            # Robot is uncertain or confused
            emotional_state = 'confusion'
        elif brain_state.get('goal_achieved', False):
            # Robot completed a task successfully
            emotional_state = 'achievement'
        elif brain_state.get('collision_detected', False) or brain_state.get('error_state', False):
            # Robot encountered a problem
            emotional_state = 'distress'
        else:
            # Default calm state
            emotional_state = 'contentment'
        
        # Get base parameters for the emotional state
        base_params = self.emotional_patterns[emotional_state]
        
        # Modulate parameters based on specific brain state values
        modulated_params = self._modulate_parameters(base_params, brain_state)
        
        return modulated_params
    
    def _modulate_parameters(self, base_params: VocalParameters, brain_state: Dict[str, any]) -> VocalParameters:
        """Apply fine modulation based on specific brain state values."""
        
        # Copy base parameters
        modulated = VocalParameters(
            fundamental_frequency=base_params.fundamental_frequency,
            amplitude=base_params.amplitude,
            duration=base_params.duration,
            harmonics=base_params.harmonics.copy(),
            frequency_modulation=base_params.frequency_modulation,
            amplitude_modulation=base_params.amplitude_modulation,
            noise_component=base_params.noise_component,
            attack_time=base_params.attack_time,
            decay_time=base_params.decay_time,
            formant_frequencies=base_params.formant_frequencies.copy(),
            formant_amplitudes=base_params.formant_amplitudes.copy()
        )
        
        # Modulate based on confidence level
        confidence = brain_state.get('prediction_confidence', 0.5)
        confidence_factor = (confidence - 0.5) * 2  # -1 to +1
        
        # Higher confidence = slightly higher pitch and volume
        modulated.fundamental_frequency *= (1.0 + confidence_factor * 0.2)
        modulated.amplitude *= (1.0 + confidence_factor * 0.1)
        
        # Modulate based on experience level
        experiences = brain_state.get('total_experiences', 0)
        experience_factor = min(1.0, experiences / 100.0)  # 0 to 1
        
        # More experience = richer harmonics
        for i in range(len(modulated.harmonics)):
            if i > 0:  # Don't change fundamental
                modulated.harmonics[i] *= (0.5 + experience_factor * 0.5)
        
        # Modulate based on activity level
        motor_speed = brain_state.get('motor_speed', 0.0)
        activity_factor = abs(motor_speed) / 100.0  # Assuming 0-100 speed range
        
        # Higher activity = slightly faster modulation
        freq_mod_rate, freq_mod_depth = modulated.frequency_modulation
        modulated.frequency_modulation = (freq_mod_rate * (1.0 + activity_factor * 0.5), freq_mod_depth)
        
        return modulated