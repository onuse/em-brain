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
from collections import deque

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
        
        # Pattern repetition tracking for boredom detection
        self.pattern_history = deque(maxlen=20)  # Track patterns over time
        self.boredom_threshold = 0.8  # Similarity threshold for boredom
        self.boredom_counter = 0
        
        # Motor smoothing
        self.previous_motor = None
        self.smoothing_factor = 0.7
        
        # Debug - store last pattern features
        self._last_pattern_features = None
        self._debug_cycle_count = 0
        self._debug_interval = 500  # Log every N cycles
        
        if not quiet_mode:
            print(f"🎮 Pattern-Based Motor Generator initialized")
            print(f"   Field shape: {field_shape}")
            print(f"   Motor dimensions: {motor_dim}")
            print(f"   No coordinates or gradients!")
    
    def generate_motor_action(self,
                            current_field: torch.Tensor,
                            experience: Optional[UnifiedFieldExperience] = None,
                            improvement_rate: float = 0.0,
                            spontaneous_info: Optional[Dict[str, float]] = None) -> FieldNativeAction:
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
        
        # Calculate boredom from pattern repetition
        boredom = self._calculate_boredom(current_field)
        pattern_features['boredom'] = boredom
        
        # Add improvement rate (negative = getting worse, positive = improving)
        pattern_features['improvement_rate'] = improvement_rate
        pattern_features['learning_stalled'] = abs(improvement_rate) < 0.01  # No significant change
        
        # Add spontaneous dynamics info
        if spontaneous_info:
            pattern_features['spontaneous_magnitude'] = spontaneous_info.get('magnitude', 0.0)
            pattern_features['spontaneous_variance'] = spontaneous_info.get('variance', 0.0)
            pattern_features['spontaneous_weight'] = spontaneous_info.get('weight', 0.0)
        else:
            pattern_features['spontaneous_magnitude'] = 0.0
            pattern_features['spontaneous_variance'] = 0.0
            pattern_features['spontaneous_weight'] = 0.0
        
        self._last_pattern_features = pattern_features  # Store for debugging
        self._last_exploration_drive = 0.0  # Will be updated in tendencies mapping
        
        # Map patterns to motor tendencies
        motor_tendencies = self._patterns_to_motor_tendencies(pattern_features)
        
        # Debug logging
        self._debug_cycle_count += 1
        if not self.quiet_mode and self._debug_cycle_count % self._debug_interval == 0:
            print(f"\n🔍 PATTERN DEBUG (cycle {self._debug_cycle_count}):")
            print(f"   Field patterns:")
            print(f"     - Oscillation: {pattern_features.get('oscillation', 0):.3f}")
            print(f"     - Energy gradient: {pattern_features.get('energy_gradient', 0):.3f}")
            print(f"     - Energy variance: {pattern_features.get('energy_variance', 0):.3f}")
            print(f"     - Coherence: {pattern_features.get('coherence', 0):.3f}")
            print(f"     - Novelty: {pattern_features.get('novelty', 0):.3f}")
            print(f"     - Boredom: {pattern_features.get('boredom', 0):.3f}")
            print(f"   Spontaneous dynamics:")
            print(f"     - Magnitude: {pattern_features.get('spontaneous_magnitude', 0):.3f}")
            print(f"     - Variance: {pattern_features.get('spontaneous_variance', 0):.3f}")
            print(f"     - Weight: {pattern_features.get('spontaneous_weight', 0):.3f}")
            print(f"   Motor tendencies:")
            print(f"     - Forward: {motor_tendencies.get('forward', 0):.3f}")
            print(f"     - Turn: {motor_tendencies.get('turn', 0):.3f}")
            print(f"     - Speed: {motor_tendencies.get('speed', 0):.3f}")
            print(f"     - Exploration: {motor_tendencies.get('exploration', 0):.3f}")
            print(f"     - Confidence: {motor_tendencies.get('confidence', 0):.3f}")
        
        # Convert tendencies to motor commands
        motor_commands = self._tendencies_to_commands(motor_tendencies)
        
        # Adaptive smoothing - less smoothing when exploring
        adaptive_smoothing = self.smoothing_factor * (1.0 - motor_tendencies.get('exploration', 0.0) * 0.5)
        
        # Apply smoothing
        if self.previous_motor is not None:
            motor_commands = (adaptive_smoothing * self.previous_motor + 
                            (1 - adaptive_smoothing) * motor_commands)
        
        # Add exploration noise based on total exploration drive
        exploration_level = motor_tendencies.get('exploration', 0.0)
        if exploration_level > 0.2:  # Lower threshold since spontaneous is always present
            # Noise proportional to exploration level
            exploration_noise = torch.randn_like(motor_commands) * 0.3 * exploration_level
            motor_commands = motor_commands + exploration_noise
            motor_commands = torch.clamp(motor_commands, -1.0, 1.0)
            
            # Debug log exploration events periodically
            # Use boredom counter as a proxy for cycles
            if not self.quiet_mode and hasattr(self, 'boredom_counter') and self.boredom_counter % 50 == 0:
                spont_exp = pattern_features['spontaneous_magnitude'] * (1.0 + pattern_features['spontaneous_variance']) * pattern_features['spontaneous_weight']
                print(f"🎲 EXPLORATION: total={exploration_level:.2f}, " +
                      f"spontaneous={spont_exp:.2f}, boredom={pattern_features['boredom']:.2f}")
        
        self.previous_motor = motor_commands.clone()
        
        # Debug final motor commands
        if not self.quiet_mode and self._debug_cycle_count % self._debug_interval == 0:
            print(f"   Final motor commands: {[f'{cmd:.3f}' for cmd in motor_commands.tolist()]}")
            print(f"   Exploration active: {exploration_level > 0.2}")
        
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
    
    def _calculate_boredom(self, current_pattern: torch.Tensor) -> float:
        """
        Calculate boredom level based on pattern repetition.
        
        High boredom = patterns are too similar/repetitive
        Low boredom = patterns are varied
        
        Returns:
            float: Boredom level [0, 1] where 1 = extremely bored
        """
        # Store current pattern signature (downsampled for efficiency)
        pattern_signature = current_pattern.flatten()[::10]  # Sample every 10th element
        
        if len(self.pattern_history) == 0:
            self.pattern_history.append(pattern_signature)
            return 0.0  # No boredom initially
        
        # Calculate similarity to recent patterns
        similarities = []
        for past_pattern in self.pattern_history:
            if len(past_pattern) == len(pattern_signature):
                # Cosine similarity
                dot_product = torch.sum(pattern_signature * past_pattern)
                norm_current = torch.norm(pattern_signature)
                norm_past = torch.norm(past_pattern)
                
                if norm_current > 1e-6 and norm_past > 1e-6:
                    similarity = (dot_product / (norm_current * norm_past)).item()
                    similarities.append(abs(similarity))
        
        # Track pattern
        self.pattern_history.append(pattern_signature.clone())
        
        if not similarities:
            return 0.0
        
        # Average similarity indicates how repetitive patterns are
        avg_similarity = np.mean(similarities)
        
        # Count how many patterns are too similar
        high_similarity_count = sum(1 for s in similarities if s > self.boredom_threshold)
        
        # Boredom increases with pattern repetition
        repetition_ratio = high_similarity_count / len(similarities)
        
        # Combined boredom metric
        boredom = 0.7 * avg_similarity + 0.3 * repetition_ratio
        
        # Update boredom counter
        if boredom > 0.7:
            self.boredom_counter += 1
        else:
            self.boredom_counter = max(0, self.boredom_counter - 1)
        
        # Amplify boredom if it persists
        if self.boredom_counter > 10:
            boredom = min(1.0, boredom * 1.5)
        
        return float(np.clip(boredom, 0.0, 1.0))
    
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
        
        # === UNIFIED EXPLORATION FRAMEWORK ===
        # Exploration emerges from three sources:
        
        # 1. SPONTANEOUS DYNAMICS (Internal restlessness)
        # High spontaneous activity = natural motor variability
        spontaneous_magnitude = features.get('spontaneous_magnitude', 0.0)
        spontaneous_variance = features.get('spontaneous_variance', 0.0)
        spontaneous_weight = features.get('spontaneous_weight', 0.0)
        
        # Spontaneous exploration: magnitude * variance * weight
        # - Magnitude: how active is the internal dynamics
        # - Variance: how complex/rich are the patterns
        # - Weight: how much the brain is in "fantasy" mode
        spontaneous_exploration = spontaneous_magnitude * (1.0 + spontaneous_variance) * spontaneous_weight
        
        # 2. PATTERN STAGNATION (Environmental boredom)
        # High boredom + low novelty = need new experiences
        boredom_drive = features.get('boredom', 0.0)
        novelty_satisfaction = features['novelty']
        pattern_exploration = boredom_drive * (1.0 - novelty_satisfaction)
        
        # 3. LEARNING PLATEAU (Cognitive stagnation)
        # No improvement = need to try something different
        learning_exploration = 0.0
        if features.get('learning_stalled', False):
            stall_factor = 1.0 - abs(features.get('improvement_rate', 0.0))
            learning_exploration = 0.3 * stall_factor
        
        # COMBINED EXPLORATION with baseline
        # Always maintain some exploration to prevent getting stuck
        EXPLORATION_BASELINE = 0.2  # 20% minimum exploration
        
        # Weighted combination of exploration sources
        weighted_exploration = (
            0.5 * spontaneous_exploration +  # Primary driver
            0.3 * pattern_exploration +       # Environmental feedback
            0.2 * learning_exploration        # Learning feedback
        )
        
        # Add baseline to ensure minimum exploration
        total_exploration = EXPLORATION_BASELINE + (1.0 - EXPLORATION_BASELINE) * weighted_exploration
        
        # Store for debugging
        self._last_exploration_drive = total_exploration
        
        # Ensure reasonable bounds
        tendencies['exploration'] = np.clip(total_exploration, 0.0, 1.0)
        
        # Overall confidence from coherence
        tendencies['confidence'] = features['coherence'] * self.mapping.coherence_to_confidence
        
        return tendencies
    
    def _tendencies_to_commands(self, tendencies: Dict[str, float]) -> torch.Tensor:
        """Convert motor tendencies to actual motor commands."""
        commands = torch.zeros(self.motor_dim, device=self.device)
        
        # Map tendencies to motor dimensions
        # For compatibility with biological_embodied_learning test: [forward, left, right, stop]
        if self.motor_dim >= 4:
            # Forward movement (pure forward tendency)
            commands[0] = np.clip(tendencies['forward'], 0.0, 1.0)
            
            # Left turn (positive turn)
            commands[1] = np.clip(max(0, tendencies['turn']), 0.0, 1.0)
            
            # Right turn (negative turn)
            commands[2] = np.clip(max(0, -tendencies['turn']), 0.0, 1.0)
            
            # Stop (inverse of urgency and forward)
            stop_tendency = 1.0 - max(tendencies['urgency'], tendencies['forward'])
            commands[3] = np.clip(stop_tendency * 0.5, 0.0, 1.0)  # Scale down stop
        
        elif self.motor_dim >= 2:
            # Fallback for 2D motor (forward, turn)
            commands[0] = np.clip(tendencies['forward'], -1.0, 1.0)
            commands[1] = np.clip(tendencies['turn'], -1.0, 1.0)
        
        if self.motor_dim >= 5:
            # Fifth motor: camera pan (exploration-driven)
            commands[4] = np.clip(tendencies['exploration'] * 2.0 - 1.0, -1.0, 1.0)
        
        # Apply confidence scaling with minimum threshold
        # Never scale below 70% to maintain movement capability
        MIN_COMMAND_SCALE = 0.7
        confidence_scale = MIN_COMMAND_SCALE + (1.0 - MIN_COMMAND_SCALE) * tendencies['confidence']
        commands = commands * confidence_scale
        
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
            'pattern_based': True,
            'boredom_level': features.get('boredom', 0.0),
            'boredom_counter': self.boredom_counter,
            'pattern_history_length': len(self.pattern_history)
        }