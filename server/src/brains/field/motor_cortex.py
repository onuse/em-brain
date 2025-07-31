#!/usr/bin/env python3
"""
Motor Cortex - Intention to Action Translation

Translates brain intentions into motor commands with proper thresholding
and feedback. This replaces the brittle boost factor with a biologically
inspired system that requires sufficient intention and confidence.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class MotorFeedback:
    """Feedback from motor cortex to brain about rejected intentions."""
    accepted: bool
    rejection_reason: str = ""
    required_strength: float = 0.0
    current_strength: float = 0.0


class MotorCortex:
    """
    Translates neural intentions into motor commands.
    
    This layer sits between the brain's pattern-based motor generator
    and the actual motor outputs. It enforces minimum activation thresholds
    and provides feedback when intentions are too weak.
    """
    
    def __init__(self, 
                 motor_dim: int,
                 device: torch.device,
                 activation_threshold: float = 0.1,
                 confidence_threshold: float = 0.05,  # Lower threshold for more exploration
                 max_amplification: float = 3.0,
                 quiet_mode: bool = False):
        """
        Initialize motor cortex.
        
        Args:
            motor_dim: Number of motor outputs
            device: Computation device
            activation_threshold: Minimum intention strength to produce movement
            confidence_threshold: Minimum confidence to accept intention
            max_amplification: Maximum signal amplification
            quiet_mode: Suppress debug output
        """
        self.motor_dim = motor_dim
        self.device = device
        self.activation_threshold = activation_threshold
        self.confidence_threshold = confidence_threshold
        self.max_amplification = max_amplification
        self.quiet_mode = quiet_mode
        
        # Track rejection statistics
        self.total_intentions = 0
        self.rejected_weak = 0
        self.rejected_uncertain = 0
        self.accepted = 0
        
        # Feedback for brain
        self.last_feedback = None
        
        if not quiet_mode:
            print(f"🧠 Motor Cortex initialized")
            print(f"   Activation threshold: {activation_threshold}")
            print(f"   Confidence threshold: {confidence_threshold}")
            print(f"   Max amplification: {max_amplification}x")
    
    def process_intentions(self, 
                          intentions: torch.Tensor,
                          confidence: float,
                          pattern_features: Dict[str, float] = None) -> Tuple[torch.Tensor, MotorFeedback]:
        """
        Process motor intentions into actual motor commands.
        
        Args:
            intentions: Raw motor intentions from pattern-based motor
            confidence: Overall confidence in the action
            pattern_features: Optional pattern analysis features
            
        Returns:
            Tuple of (motor_commands, feedback)
        """
        self.total_intentions += 1
        
        # Check if intentions are strong enough
        intention_strength = torch.max(torch.abs(intentions)).detach().item()
        
        # Note: The brain already provides confidence boosted by developmental confidence,
        # so we use it directly without additional boosting
        
        # For low confidence, provide assistance
        if confidence < 0.3:  # Help more often (was confidence_threshold = 0.05)
            self.rejected_uncertain += 1  # Track for statistics
            
            # Scale assistance based on how low confidence is
            # At conf=0.3: boost 1.5x, at conf=0: boost 2.1x
            assistance_factor = 1.5 + (0.3 - confidence) * 2.0
            
            # Apply assistance to intentions
            motor_commands = intentions * assistance_factor
            
            # Ensure minimum forward movement for exploration
            if motor_commands[0] < self.activation_threshold:
                motor_commands[0] = self.activation_threshold
            
            feedback = MotorFeedback(
                accepted=True,  # Accept but note low confidence
                rejection_reason="low_confidence_exploration",
                required_strength=self.confidence_threshold,
                current_strength=confidence
            )
            self.last_feedback = feedback
            return motor_commands, feedback
        
        # Boost weak intentions (only check after confidence)
        if intention_strength < self.activation_threshold:
            self.rejected_weak += 1  # Still track for statistics
            
            if intention_strength < 1e-8:  # Essentially zero
                # Create minimal movement in primary direction
                motor_commands = torch.zeros_like(intentions)
                motor_commands[0] = self.activation_threshold
            else:
                # Boost to minimum threshold
                boost_factor = self.activation_threshold / intention_strength
                motor_commands = intentions * boost_factor
                
            feedback = MotorFeedback(
                accepted=True,  # We're accepting but boosting
                rejection_reason="boosted_weak_intention",
                required_strength=self.activation_threshold,
                current_strength=intention_strength
            )
            self.last_feedback = feedback
            return motor_commands, feedback
        
        # Accept and process intentions
        self.accepted += 1
        
        # Confidence-based amplification
        # Low confidence (0.3-0.5): 1.0-1.5x amplification
        # Medium confidence (0.5-0.7): 1.5-2.0x amplification  
        # High confidence (0.7-1.0): 2.0-3.0x amplification
        amplification = 1.0 + (confidence - self.confidence_threshold) * \
                       (self.max_amplification - 1.0) / (1.0 - self.confidence_threshold)
        
        # Apply amplification with safety clipping
        motor_commands = intentions * amplification
        motor_commands = torch.clamp(motor_commands, -1.0, 1.0)
        
        # Store for debugging
        self._last_amplification = amplification
        
        # Optional: Apply pattern-specific modulation
        if pattern_features:
            # Boost exploration when novelty is high
            if 'novelty' in pattern_features and pattern_features['novelty'] > 0.5:
                motor_commands = motor_commands * 1.2
            
            # Smooth movements when coherence is high
            if 'coherence' in pattern_features and pattern_features['coherence'] > 0.7:
                motor_commands = motor_commands * 0.9  # Slightly dampen for smoothness
        
        feedback = MotorFeedback(
            accepted=True,
            rejection_reason="",
            required_strength=0.0,
            current_strength=intention_strength
        )
        self.last_feedback = feedback
        
        if not self.quiet_mode and self.total_intentions % 1000 == 0:
            acceptance_rate = self.accepted / self.total_intentions
            print(f"⚡ Motor Cortex: {acceptance_rate:.1%} acceptance rate")
        
        return motor_commands, feedback
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get motor cortex statistics."""
        if self.total_intentions == 0:
            return {
                'total_intentions': 0,
                'acceptance_rate': 0.0,
                'rejection_rate_weak': 0.0,
                'rejection_rate_uncertain': 0.0
            }
        
        stats = {
            'total_intentions': self.total_intentions,
            'acceptance_rate': self.accepted / self.total_intentions,
            'rejection_rate_weak': self.rejected_weak / self.total_intentions,
            'rejection_rate_uncertain': self.rejected_uncertain / self.total_intentions,
            'last_feedback': self.last_feedback
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_intentions = 0
        self.rejected_weak = 0
        self.rejected_uncertain = 0
        self.accepted = 0