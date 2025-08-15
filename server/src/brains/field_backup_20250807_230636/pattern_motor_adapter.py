"""
Pattern Motor Adapter

Adapts unified pattern extraction for motor generation.
Maps pattern features to motor tendencies.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .unified_pattern_system import UnifiedPatternSystem, FieldPattern
from .motor_cortex import MotorCortex


@dataclass 
class MotorMapping:
    """Maps pattern features to motor tendencies."""
    # Oscillation → movement
    oscillation_forward: float = 1.0
    oscillation_lateral: float = 0.5
    
    # Flow → rotation
    divergence_turn: float = 1.0
    curl_spin: float = 0.8
    
    # Energy → speed
    energy_speed: float = 0.5
    variance_urgency: float = 2.0
    
    # Coherence → confidence
    coherence_confidence: float = 1.0
    novelty_exploration: float = 0.3


class PatternMotorAdapter:
    """
    Generates motor commands from unified pattern features.
    
    This replaces the pattern extraction in PatternBasedMotorGenerator
    with the unified system.
    """
    
    def __init__(self,
                 pattern_system: UnifiedPatternSystem,
                 motor_dim: int,
                 motor_cortex: MotorCortex,
                 device: torch.device):
        """Initialize motor adapter."""
        self.pattern_system = pattern_system
        self.motor_dim = motor_dim
        self.motor_cortex = motor_cortex
        self.device = device
        
        # Motor mapping configuration
        self.mapping = MotorMapping()
        
        # Boredom tracking
        self.recent_patterns = []
        self.boredom_threshold = 0.95
        
    def generate_motor_action(self,
                             field: torch.Tensor,
                             spontaneous_activity: torch.Tensor,
                             attention_state: Optional[Dict[str, Any]] = None,
                             exploration_params: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Generate motor commands from field patterns.
        
        Args:
            field: Current field state
            spontaneous_activity: Internal dynamics
            attention_state: What to focus on
            exploration_params: Exploration/exploitation balance
            
        Returns:
            Motor commands tensor
        """
        # Extract patterns using unified system
        patterns = self.pattern_system.extract_patterns(field, n_patterns=10)
        
        if not patterns:
            # No patterns - return neutral
            return torch.zeros(self.motor_dim, device=self.device)
        
        # Get dominant pattern (highest salience)
        dominant = patterns[0]
        
        # Check for attended patterns if attention provided
        if attention_state and 'attended_patterns' in attention_state:
            # Find most salient attended pattern
            for pattern in patterns:
                if pattern.location in attention_state['attended_patterns']:
                    dominant = pattern
                    break
        
        # Convert pattern to motor tendencies
        tendencies = self._pattern_to_motor_tendencies(dominant)
        
        # Add boredom factor
        boredom = self._compute_boredom(dominant)
        tendencies['boredom'] = boredom
        
        # Add spontaneous influence (now comes from field dynamics)
        if spontaneous_activity is not None:
            spont_influence = torch.mean(torch.abs(spontaneous_activity)).item()
            tendencies['spontaneous'] = spont_influence
        else:
            # Spontaneous is now integrated into field - use field variance as proxy
            tendencies['spontaneous'] = dominant.variance * 0.5
        
        # Convert tendencies to motor commands
        motor_commands = self._tendencies_to_motors(tendencies, exploration_params)
        
        # Apply motor cortex processing
        # Check if this is adaptive motor cortex
        if hasattr(self.motor_cortex, 'process_intentions'):
            # Build brain state for adaptive cortex
            brain_state = {
                'exploration_drive': exploration_params.get('exploration_drive', 0.5) if exploration_params else 0.5,
                'information': dominant.energy,  # Use pattern energy as proxy for field information
                'confidence': dominant.coherence
            }
            
            # Check method signature
            import inspect
            sig = inspect.signature(self.motor_cortex.process_intentions)
            if 'brain_state' in sig.parameters:
                # Adaptive motor cortex
                motor_commands, feedback = self.motor_cortex.process_intentions(
                    intentions=motor_commands,
                    confidence=dominant.coherence,
                    brain_state=brain_state
                )
            else:
                # Original motor cortex
                motor_commands, feedback = self.motor_cortex.process_intentions(
                    intentions=motor_commands,
                    confidence=dominant.coherence,
                    pattern_features={'energy': dominant.energy}
                )
        else:
            # Fallback
            feedback = None
        
        # Update pattern memory
        self.recent_patterns.append(dominant)
        if len(self.recent_patterns) > 20:
            self.recent_patterns.pop(0)
        
        return motor_commands
    
    def _pattern_to_motor_tendencies(self, pattern: FieldPattern) -> Dict[str, float]:
        """Convert pattern features to motor tendencies."""
        tendencies = {}
        
        # Movement from oscillation (with baseline to ensure some movement)
        tendencies['forward'] = pattern.oscillation * self.mapping.oscillation_forward + 0.2
        tendencies['lateral'] = pattern.oscillation * self.mapping.oscillation_lateral
        
        # Rotation from flow
        tendencies['turn'] = pattern.flow_divergence * self.mapping.divergence_turn
        tendencies['spin'] = pattern.flow_curl * self.mapping.curl_spin
        
        # Speed from energy (with baseline to prevent complete stillness)
        tendencies['speed'] = pattern.energy * self.mapping.energy_speed + 0.3
        tendencies['urgency'] = pattern.variance * self.mapping.variance_urgency
        
        # Confidence and exploration
        tendencies['confidence'] = pattern.coherence * self.mapping.coherence_confidence
        tendencies['exploration'] = pattern.novelty * self.mapping.novelty_exploration
        
        # Uncertainty-driven attention (for camera control)
        # High variance + low coherence = high uncertainty
        tendencies['uncertainty'] = pattern.variance * (1.0 - pattern.coherence)
        
        # Attention shift based on salience gradient
        # This could be enhanced with actual spatial gradient computation
        tendencies['attention_shift'] = pattern.salience * (1.0 - tendencies['confidence'])
        
        # Vertical interest (for tilt) - based on flow patterns
        tendencies['vertical_interest'] = abs(pattern.flow_divergence) * 0.5
        
        return tendencies
    
    def _compute_boredom(self, pattern: FieldPattern) -> float:
        """Detect repetitive patterns."""
        if len(self.recent_patterns) < 5:
            return 0.0
        
        # Compare to recent patterns
        similarities = []
        pattern_dict = pattern.to_dict()
        
        for past in self.recent_patterns[-10:]:
            past_dict = past.to_dict()
            
            # Feature-wise similarity
            sim = 0.0
            for key in ['energy', 'oscillation', 'flow_divergence']:
                if key in pattern_dict and key in past_dict:
                    diff = abs(pattern_dict[key] - past_dict[key])
                    sim += 1.0 - min(1.0, diff * 2)
            
            similarities.append(sim / 3)  # Average over features
        
        # High similarity = high boredom
        max_similarity = max(similarities) if similarities else 0.0
        
        if max_similarity > self.boredom_threshold:
            return min(1.0, (max_similarity - self.boredom_threshold) * 10)
        else:
            return 0.0
    
    def _tendencies_to_motors(self, 
                             tendencies: Dict[str, float],
                             exploration_params: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Convert motor tendencies to actual motor commands."""
        # Initialize motor commands
        motors = torch.zeros(self.motor_dim, device=self.device)
        
        # Get exploration parameters
        if exploration_params:
            base_exploration = exploration_params.get('exploration_drive', 0.5)
            motor_noise = exploration_params.get('motor_noise', 0.2)
        else:
            base_exploration = 0.5
            motor_noise = 0.2
        
        # Total exploration drive
        total_exploration = (
            base_exploration + 
            tendencies.get('exploration', 0) * 0.3 +
            tendencies.get('boredom', 0) * 0.5 +
            tendencies.get('spontaneous', 0) * 0.2
        )
        total_exploration = np.clip(total_exploration, 0, 1)
        
        # Map tendencies to motor dimensions
        if self.motor_dim >= 2:
            # Differential drive (2 motors)
            forward = tendencies.get('forward', 0) * tendencies.get('confidence', 0.5)
            turn = tendencies.get('turn', 0) + tendencies.get('spin', 0) * 0.5
            
            # Add exploration noise
            forward += np.random.normal(0, motor_noise * total_exploration)
            turn += np.random.normal(0, motor_noise * total_exploration * 2)
            
            # Convert to left/right motors
            motors[0] = forward - turn  # Left motor
            motors[1] = forward + turn  # Right motor
            
        if self.motor_dim >= 3:
            # Additional motors (e.g., arm, gripper)
            motors[2] = tendencies.get('lateral', 0) + \
                       np.random.normal(0, motor_noise * total_exploration)
        
        if self.motor_dim >= 4:
            # Camera control (motors 3-4 for pan/tilt)
            # These are driven by uncertainty and attention
            uncertainty = tendencies.get('uncertainty', 0.5)
            attention_shift = tendencies.get('attention_shift', 0)
            
            # Pan (motor 3) - horizontal scanning driven by uncertainty
            motors[3] = (attention_shift * 0.8 + 
                        uncertainty * np.random.normal(0, 0.3) +
                        np.random.normal(0, motor_noise * total_exploration * 0.5))
            
        if self.motor_dim >= 5:
            # Tilt (motor 4) - vertical scanning, less frequent
            motors[4] = (tendencies.get('vertical_interest', 0) * 0.5 +
                        uncertainty * np.random.normal(0, 0.2) +
                        np.random.normal(0, motor_noise * total_exploration * 0.3))
            
        if self.motor_dim > 5:
            # Additional motors beyond camera
            for i in range(5, self.motor_dim):
                influence = 1.0 / (i - 3)
                motors[i] = (tendencies.get('urgency', 0) * influence +
                            np.random.normal(0, motor_noise * total_exploration))
        
        # Scale by overall speed tendency
        speed_factor = 0.5 + tendencies.get('speed', 0.5) * 0.5
        motors = motors * speed_factor
        
        # Clamp to reasonable range
        motors = torch.clamp(motors, -1.0, 1.0)
        
        return motors