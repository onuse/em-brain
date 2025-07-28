#!/usr/bin/env python3
"""
Pattern-Based Motor Generation

Generates motor commands from field patterns rather than gradients.
This provides a coordinate-free alternative to gradient-based motor control.

Key principles:
1. Motor commands emerge from field evolution patterns
2. No spatial gradients or coordinates required
3. Pattern dynamics map to motor tendencies
4. Field coherence drives action confidence
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .field_types import FieldNativeAction, UnifiedFieldExperience, FieldDynamicsFamily


@dataclass
class PatternMotorMapping:
    """Maps field pattern characteristics to motor tendencies."""
    # Pattern oscillations to movement
    oscillation_to_forward: float = 1.0
    oscillation_to_lateral: float = 0.5
    
    # Pattern flow to rotation
    flow_divergence_to_turn: float = 1.0
    flow_curl_to_spin: float = 0.8
    
    # Pattern energy to speed
    energy_gradient_to_speed: float = 0.5
    energy_variance_to_urgency: float = 2.0
    
    # Pattern coherence to confidence
    coherence_to_confidence: float = 1.0
    novelty_to_exploration: float = 0.3


class PatternBasedMotorGenerator:
    """
    Generates motor commands from field patterns without coordinates.
    
    This is a key component for reducing coordinate dependencies.
    Instead of following spatial gradients, it extracts motor tendencies
    from the intrinsic patterns in field evolution.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...],
                 motor_dim: int,
                 device: torch.device,
                 quiet_mode: bool = False):
        """
        Initialize pattern-based motor generator.
        
        Args:
            field_shape: Shape of the unified field tensor
            motor_dim: Number of motor outputs
            device: Computation device
            quiet_mode: Suppress debug output
        """
        self.field_shape = field_shape
        self.motor_dim = motor_dim
        self.device = device
        self.quiet_mode = quiet_mode
        
        # Pattern-motor mapping
        self.mapping = PatternMotorMapping()
        
        # Field history for evolution tracking
        self.field_history = []
        self.history_length = 3
        
        # Motor smoothing
        self.previous_motor = None
        self.smoothing_factor = 0.7
        
        # Debug - store last pattern features
        self._last_pattern_features = None
        
        if not quiet_mode:
            print(f"🎮 Pattern-Based Motor Generator initialized")
            print(f"   Field shape: {field_shape}")
            print(f"   Motor dimensions: {motor_dim}")
            print(f"   No coordinates or gradients!")
    
    def generate_motor_action(self,
                            current_field: torch.Tensor,
                            experience: Optional[UnifiedFieldExperience] = None) -> FieldNativeAction:
        """
        Generate motor commands from field patterns.
        
        Args:
            current_field: Current unified field state
            experience: Optional current experience for context
            
        Returns:
            Motor action based on field patterns
        """
        # Update field history
        self._update_field_history(current_field)
        
        # Extract field evolution patterns
        evolution_patterns = self._extract_evolution_patterns()
        
        # Analyze pattern characteristics
        pattern_features = self._analyze_pattern_features(evolution_patterns)
        self._last_pattern_features = pattern_features  # Store for debugging
        
        # Map patterns to motor tendencies
        motor_tendencies = self._patterns_to_motor_tendencies(pattern_features)
        
        # Convert tendencies to motor commands
        motor_commands = self._tendencies_to_commands(motor_tendencies)
        
        # Apply smoothing
        if self.previous_motor is not None:
            motor_commands = (self.smoothing_factor * self.previous_motor + 
                            (1 - self.smoothing_factor) * motor_commands)
        self.previous_motor = motor_commands.clone()
        
        # Create action
        action = FieldNativeAction(
            timestamp=0.0,  # Will be set by caller
            output_stream=motor_commands,  # Motor commands as output stream
            field_gradients=torch.zeros(3, device=self.device),  # No gradients!
            confidence=pattern_features.get('coherence', 0.5),
            dynamics_family_contributions=self._get_dynamics_contributions(motor_tendencies)
        )
        
        return action
    
    def _update_field_history(self, field: torch.Tensor):
        """Update field history for evolution tracking."""
        self.field_history.append(field.clone().detach())
        if len(self.field_history) > self.history_length:
            self.field_history.pop(0)
    
    def _extract_evolution_patterns(self) -> Dict[str, torch.Tensor]:
        """Extract patterns of field evolution over time."""
        if len(self.field_history) < 2:
            # Not enough history - return zeros
            return {
                'temporal_change': torch.zeros_like(self.field_history[-1]),
                'acceleration': torch.zeros_like(self.field_history[-1])
            }
        
        patterns = {}
        
        # Temporal change (velocity)
        patterns['temporal_change'] = self.field_history[-1] - self.field_history[-2]
        
        # Acceleration (if enough history)
        if len(self.field_history) >= 3:
            velocity_prev = self.field_history[-2] - self.field_history[-3]
            velocity_curr = patterns['temporal_change']
            patterns['acceleration'] = velocity_curr - velocity_prev
        else:
            patterns['acceleration'] = torch.zeros_like(patterns['temporal_change'])
        
        return patterns
    
    def _analyze_pattern_features(self, patterns: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze characteristics of field evolution patterns."""
        features = {}
        
        temporal_change = patterns['temporal_change']
        acceleration = patterns['acceleration']
        
        # 1. Oscillation analysis (rhythmic patterns)
        # Detect zero-crossings in temporal change
        oscillation_score = self._detect_oscillations(temporal_change)
        features['oscillation'] = oscillation_score
        
        # 2. Flow analysis (directional patterns)
        flow_features = self._analyze_flow_patterns(temporal_change)
        features.update(flow_features)
        
        # 3. Energy analysis (intensity patterns)
        features['energy_mean'] = torch.mean(torch.abs(temporal_change)).item()
        features['energy_variance'] = torch.var(torch.abs(temporal_change)).item()
        features['energy_gradient'] = torch.mean(torch.abs(acceleration)).item()
        
        # 4. Coherence analysis (pattern stability)
        features['coherence'] = self._calculate_coherence(temporal_change)
        
        # 5. Novelty analysis (pattern uniqueness)
        features['novelty'] = self._calculate_novelty(patterns)
        
        return features
    
    def _detect_oscillations(self, field: torch.Tensor) -> float:
        """Detect oscillatory patterns in field."""
        # Flatten field for analysis
        flat_field = field.flatten()
        
        # Count sign changes
        signs = torch.sign(flat_field)
        sign_changes = torch.sum(torch.abs(signs[1:] - signs[:-1])) / 2
        
        # Normalize by field size
        oscillation_score = sign_changes.item() / len(flat_field)
        
        return min(1.0, oscillation_score * 10)  # Scale to [0, 1]
    
    def _analyze_flow_patterns(self, field: torch.Tensor) -> Dict[str, float]:
        """Analyze flow-like patterns in field evolution."""
        flow_features = {}
        
        # Flatten the field to analyze overall flow patterns
        flat_field = field.flatten()
        field_size = len(flat_field)
        
        if field_size < 4:
            # Too small for flow analysis
            flow_features['divergence'] = 0.0
            flow_features['curl'] = 0.0
            return flow_features
        
        # Create a pseudo-2D view for flow analysis
        # Approximate square dimensions
        side = int(np.sqrt(field_size))
        if side < 2:
            side = 2
        
        # Truncate to fit square
        truncated_size = side * side
        if truncated_size > field_size:
            side = side - 1
            truncated_size = side * side
        
        # Reshape to 2D
        try:
            field_2d = flat_field[:truncated_size].reshape(side, side)
            
            # Approximate divergence (expansion/contraction)
            if side > 1:
                dx = field_2d[1:, :] - field_2d[:-1, :]
                dy = field_2d[:, 1:] - field_2d[:, :-1]
                
                # Calculate divergence on the overlapping region
                div_x = torch.mean(torch.abs(dx[:, :-1]))
                div_y = torch.mean(torch.abs(dy[:-1, :]))
                divergence = (div_x + div_y) / 2
                flow_features['divergence'] = divergence.item()
                
                # Approximate curl (rotation) - difference in perpendicular gradients
                # Use center differences for better approximation
                if side > 2:
                    curl_component = torch.mean(torch.abs(dx[:-1, :-1]) - torch.abs(dy[:-1, :-1]))
                    flow_features['curl'] = curl_component.item()
                else:
                    flow_features['curl'] = 0.0
            else:
                flow_features['divergence'] = 0.0
                flow_features['curl'] = 0.0
                
        except:
            # Fallback if reshape fails
            flow_features['divergence'] = 0.0
            flow_features['curl'] = 0.0
        
        return flow_features
    
    def _calculate_coherence(self, field: torch.Tensor) -> float:
        """Calculate coherence/organization of field pattern."""
        # Coherence based on spatial autocorrelation
        flat_field = field.flatten()
        
        if len(flat_field) < 2:
            return 0.5
        
        # Normalize
        if torch.std(flat_field) < 1e-6:
            return 1.0  # Uniform field is perfectly coherent
        
        normalized = (flat_field - torch.mean(flat_field)) / torch.std(flat_field)
        
        # Autocorrelation at multiple lags
        correlations = []
        for lag in [1, 2, 4]:
            if lag < len(normalized):
                corr = torch.sum(normalized[:-lag] * normalized[lag:]) / (len(normalized) - lag)
                correlations.append(abs(corr.item()))
        
        # Average correlation as coherence measure
        coherence = np.mean(correlations) if correlations else 0.5
        
        return coherence
    
    def _calculate_novelty(self, patterns: Dict[str, torch.Tensor]) -> float:
        """Calculate novelty of current pattern."""
        # Simple novelty: magnitude of acceleration
        # (How much the pattern is changing its change)
        acceleration_magnitude = torch.mean(torch.abs(patterns['acceleration'])).item()
        
        # Normalize to [0, 1]
        novelty = np.tanh(acceleration_magnitude * 5)
        
        return novelty
    
    def _patterns_to_motor_tendencies(self, features: Dict[str, float]) -> Dict[str, float]:
        """Map pattern features to motor tendencies."""
        tendencies = {}
        
        # Forward/backward from oscillations
        tendencies['forward'] = features['oscillation'] * self.mapping.oscillation_to_forward
        
        # Lateral movement from oscillation + divergence
        tendencies['lateral'] = (features['oscillation'] * self.mapping.oscillation_to_lateral +
                                features.get('divergence', 0) * 0.3)
        
        # Turning from flow curl
        tendencies['turn'] = features.get('curl', 0) * self.mapping.flow_divergence_to_turn
        
        # Speed from energy gradient
        tendencies['speed'] = features['energy_gradient'] * self.mapping.energy_gradient_to_speed
        
        # Urgency from energy variance
        tendencies['urgency'] = features['energy_variance'] * self.mapping.energy_variance_to_urgency
        
        # Exploration from novelty
        tendencies['exploration'] = features['novelty'] * self.mapping.novelty_to_exploration
        
        # Overall confidence from coherence
        tendencies['confidence'] = features['coherence'] * self.mapping.coherence_to_confidence
        
        return tendencies
    
    def _tendencies_to_commands(self, tendencies: Dict[str, float]) -> torch.Tensor:
        """Convert motor tendencies to actual motor commands."""
        commands = torch.zeros(self.motor_dim, device=self.device)
        
        # Map tendencies to motor dimensions
        if self.motor_dim >= 2:
            # First two motors: forward/lateral movement
            commands[0] = np.clip(tendencies['forward'] - tendencies['lateral'], -1.0, 1.0)
            commands[1] = np.clip(tendencies['turn'], -1.0, 1.0)
        
        if self.motor_dim >= 3:
            # Third motor: speed/intensity
            commands[2] = np.clip(tendencies['speed'] * tendencies['confidence'], 0.0, 1.0)
        
        if self.motor_dim >= 4:
            # Fourth motor: action/exploration
            commands[3] = np.clip(tendencies['urgency'] + tendencies['exploration'], 0.0, 1.0)
        
        if self.motor_dim >= 5:
            # Fifth motor: camera pan (exploration-driven)
            commands[4] = np.clip(tendencies['exploration'] * 2.0 - 1.0, -1.0, 1.0)
        
        # Apply confidence scaling (without artificial boost)
        commands = commands * (0.5 + 0.5 * tendencies['confidence'])
        
        return commands
    
    def _get_dynamics_contributions(self, tendencies: Dict[str, float]) -> Dict[FieldDynamicsFamily, float]:
        """Map motor tendencies to dynamics family contributions."""
        return {
            FieldDynamicsFamily.OSCILLATORY: tendencies.get('forward', 0.0),
            FieldDynamicsFamily.FLOW: tendencies.get('turn', 0.0) + tendencies.get('lateral', 0.0),
            FieldDynamicsFamily.ENERGY: tendencies.get('speed', 0.0),
            FieldDynamicsFamily.TOPOLOGY: tendencies.get('confidence', 0.0),
            FieldDynamicsFamily.COUPLING: tendencies.get('urgency', 0.0),
            FieldDynamicsFamily.EMERGENCE: tendencies.get('exploration', 0.0),
            FieldDynamicsFamily.SPATIAL: 0.0  # No spatial coordinates!
        }
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern-based motor generation."""
        if not self.field_history:
            return {'history_length': 0}
        
        current_patterns = self._extract_evolution_patterns()
        features = self._analyze_pattern_features(current_patterns)
        
        return {
            'history_length': len(self.field_history),
            'pattern_features': features,
            'motor_smoothing': self.smoothing_factor,
            'pattern_based': True
        }