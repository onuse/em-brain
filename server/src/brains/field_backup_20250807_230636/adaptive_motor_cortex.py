#!/usr/bin/env python3
"""
Adaptive Motor Cortex - Dynamic Intention to Action Translation

Replaces fixed thresholds with adaptive, context-sensitive motor gating.
The threshold emerges from:
1. Recent motor history (habituation/sensitization)
2. Environmental feedback (reward/punishment)
3. Internal state (exploration vs exploitation)
4. Noise floor (adapts to baseline activity)

This creates more nuanced, graded motor outputs instead of binary on/off.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class AdaptiveMotorFeedback:
    """Enhanced feedback with adaptive threshold information."""
    accepted: bool
    effective_threshold: float  # The actual threshold used
    intention_strength: float
    adaptation_factor: float  # How much the threshold was adapted
    gating_mode: str  # "sensitive", "normal", "selective"


class AdaptiveMotorCortex:
    """
    Adaptive motor cortex with dynamic thresholds.
    
    Instead of fixed thresholds, this system adapts based on:
    - Signal-to-noise ratio in motor intentions
    - Recent motor activity patterns
    - Environmental feedback
    - Internal brain state
    """
    
    def __init__(self, 
                 motor_dim: int,
                 device: torch.device,
                 base_sensitivity: float = 0.1,
                 adaptation_rate: float = 0.01,
                 history_window: int = 100,
                 quiet_mode: bool = False):
        """
        Initialize adaptive motor cortex.
        
        Args:
            motor_dim: Number of motor outputs
            device: Computation device
            base_sensitivity: Starting sensitivity (not a fixed threshold!)
            adaptation_rate: How quickly thresholds adapt
            history_window: Window for tracking patterns
            quiet_mode: Suppress debug output
        """
        self.motor_dim = motor_dim
        self.device = device
        self.base_sensitivity = base_sensitivity
        self.adaptation_rate = adaptation_rate
        self.quiet_mode = quiet_mode
        
        # Adaptive thresholds per motor
        self.motor_thresholds = torch.full((motor_dim,), base_sensitivity, device=device)
        self.noise_floor = torch.zeros(motor_dim, device=device)
        
        # History tracking
        self.intention_history = deque(maxlen=history_window)
        self.output_history = deque(maxlen=history_window)
        self.feedback_history = deque(maxlen=history_window)
        
        # Adaptation state
        self.habituation_level = torch.zeros(motor_dim, device=device)
        self.sensitization_level = torch.zeros(motor_dim, device=device)
        
        # Statistics
        self.total_intentions = 0
        self.adaptation_events = 0
        
        if not quiet_mode:
            print(f"🧠 Adaptive Motor Cortex initialized")
            print(f"   Base sensitivity: {base_sensitivity}")
            print(f"   Adaptation rate: {adaptation_rate}")
            print(f"   No fixed thresholds - adapts to context")
    
    def process_intentions(self, 
                          intentions: torch.Tensor,
                          confidence: float,
                          brain_state: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, AdaptiveMotorFeedback]:
        """
        Process motor intentions with adaptive thresholds.
        
        Args:
            intentions: Raw motor intentions
            confidence: Confidence in the action
            brain_state: Optional brain state (exploration, information level, etc.)
            
        Returns:
            Tuple of (motor_commands, feedback)
        """
        self.total_intentions += 1
        
        # Update noise floor estimate
        self._update_noise_floor(intentions)
        
        # Compute adaptive thresholds
        effective_thresholds = self._compute_adaptive_thresholds(
            intentions, confidence, brain_state
        )
        
        # Determine gating mode based on context
        gating_mode = self._determine_gating_mode(confidence, brain_state)
        
        # Apply adaptive gating with smooth transfer function
        motor_commands = self._apply_adaptive_gating(
            intentions, effective_thresholds, gating_mode
        )
        
        # Update history
        self.intention_history.append(intentions.clone())
        self.output_history.append(motor_commands.clone())
        
        # Create feedback
        intention_strength = torch.max(torch.abs(intentions)).item()
        avg_threshold = torch.mean(effective_thresholds).item()
        
        feedback = AdaptiveMotorFeedback(
            accepted=torch.any(motor_commands != 0).item(),
            effective_threshold=avg_threshold,
            intention_strength=intention_strength,
            adaptation_factor=avg_threshold / self.base_sensitivity,
            gating_mode=gating_mode
        )
        
        self.feedback_history.append(feedback)
        
        # Adapt thresholds based on outcomes
        self._adapt_thresholds(intentions, motor_commands, confidence)
        
        return motor_commands, feedback
    
    def _update_noise_floor(self, intentions: torch.Tensor):
        """Estimate noise floor from recent quiet periods."""
        # Handle dimension mismatch gracefully
        if intentions.shape[0] != self.noise_floor.shape[0]:
            # Resize noise floor to match intentions
            self.noise_floor = torch.zeros(intentions.shape[0], device=self.device)
            self.motor_thresholds = torch.full((intentions.shape[0],), self.base_sensitivity, device=self.device)
            self.habituation_level = torch.zeros(intentions.shape[0], device=self.device)
            self.sensitization_level = torch.zeros(intentions.shape[0], device=self.device)
        
        # Simple exponential moving average of small signals
        small_signal_mask = torch.abs(intentions) < self.base_sensitivity * 2
        if torch.any(small_signal_mask):
            noise_estimate = torch.abs(intentions) * small_signal_mask.float()
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * noise_estimate
    
    def _compute_adaptive_thresholds(self, 
                                    intentions: torch.Tensor,
                                    confidence: float,
                                    brain_state: Optional[Dict[str, float]]) -> torch.Tensor:
        """Compute context-sensitive thresholds."""
        # Start with base thresholds
        thresholds = self.motor_thresholds.clone()
        
        # 1. Adapt to noise floor (never gate below noise)
        thresholds = torch.maximum(thresholds, self.noise_floor * 3)
        
        # 2. Confidence modulation (smooth, not stepped)
        # Low confidence → lower thresholds (more permissive)
        # High confidence → normal thresholds
        confidence_factor = 0.5 + 0.5 * confidence  # Range: 0.5-1.0
        thresholds *= confidence_factor
        
        # 3. Exploration modulation
        if brain_state:
            exploration = brain_state.get('exploration_drive', 0.5)
            # High exploration → lower thresholds
            exploration_factor = 1.0 - 0.5 * exploration  # Range: 0.5-1.0
            thresholds *= exploration_factor
            
            # Information state modulation
            information = brain_state.get('information', 0.5)
            # Low information → more permissive (looking for input)
            # High information → more selective (processing)
            info_factor = 0.7 + 0.6 * information  # Range: 0.7-1.3
            thresholds *= info_factor
        
        # 4. Habituation/sensitization
        # Frequently used motors become less sensitive (habituation)
        # Rarely used motors become more sensitive (sensitization)
        thresholds *= (1.0 + self.habituation_level - self.sensitization_level)
        
        # Ensure reasonable bounds
        thresholds = torch.clamp(thresholds, 
                               self.base_sensitivity * 0.1,  # 10x more sensitive
                               self.base_sensitivity * 3.0)  # 3x less sensitive
        
        return thresholds
    
    def _determine_gating_mode(self, confidence: float, 
                              brain_state: Optional[Dict[str, float]]) -> str:
        """Determine gating mode based on context."""
        if brain_state:
            exploration = brain_state.get('exploration_drive', 0.5)
            if exploration > 0.7:
                return "sensitive"  # Low thresholds, high gain
            elif confidence > 0.7:
                return "selective"  # Higher thresholds, precise control
        
        return "normal"
    
    def _apply_adaptive_gating(self, intentions: torch.Tensor,
                              thresholds: torch.Tensor,
                              mode: str) -> torch.Tensor:
        """Apply smooth, adaptive gating function."""
        # Normalize intentions by thresholds
        normalized = intentions / (thresholds + 1e-8)
        
        if mode == "sensitive":
            # Sigmoid with high gain - small signals get through
            # But with smooth gradation, not binary
            gated = torch.tanh(normalized * 3.0) * torch.abs(intentions)
            # Add small exploration noise
            noise = torch.randn_like(gated) * thresholds * 0.1
            gated += noise
            
        elif mode == "selective":
            # Smooth threshold with dead zone
            # Signals must clearly exceed threshold
            smoothness = 0.2
            gated = torch.where(
                torch.abs(normalized) > 1.0,
                intentions * torch.sigmoid((torch.abs(normalized) - 1.0) / smoothness),
                intentions * torch.abs(normalized) * 0.1  # Weak subthreshold response
            )
            
        else:  # normal
            # Smooth sigmoid gating
            # Gradual activation as signal exceeds threshold
            gated = intentions * torch.sigmoid((torch.abs(normalized) - 0.5) * 4)
        
        # Apply gain based on signal strength
        # Weak signals get boosted, strong signals are preserved
        signal_strength = torch.abs(gated)
        gain = torch.where(
            signal_strength < thresholds,
            1.0 + (1.0 - signal_strength / thresholds),  # Boost weak
            1.0  # Preserve strong
        )
        
        return gated * gain
    
    def _adapt_thresholds(self, intentions: torch.Tensor,
                         outputs: torch.Tensor,
                         confidence: float):
        """Adapt thresholds based on recent activity."""
        # Track which motors are active
        active_mask = torch.abs(outputs) > 1e-6
        
        # Update habituation (frequent use → less sensitive)
        self.habituation_level = 0.99 * self.habituation_level
        self.habituation_level[active_mask] += self.adaptation_rate
        
        # Update sensitization (rare use → more sensitive)
        self.sensitization_level = 0.99 * self.sensitization_level
        self.sensitization_level[~active_mask] += self.adaptation_rate * 0.5
        
        # Adapt base thresholds based on success
        # If high confidence but low output → decrease threshold
        # If low confidence but high output → increase threshold
        if confidence > 0.7 and torch.mean(torch.abs(outputs)) < 0.1:
            self.motor_thresholds *= (1.0 - self.adaptation_rate)
            self.adaptation_events += 1
        elif confidence < 0.3 and torch.mean(torch.abs(outputs)) > 0.5:
            self.motor_thresholds *= (1.0 + self.adaptation_rate)
            self.adaptation_events += 1
    
    def inject_reward_feedback(self, reward: float):
        """Update thresholds based on reward feedback."""
        if len(self.output_history) == 0:
            return
            
        recent_output = self.output_history[-1]
        
        if reward > 0:
            # Positive reward → remember what worked
            # Slightly decrease thresholds for active motors
            active_mask = torch.abs(recent_output) > 1e-6
            self.motor_thresholds[active_mask] *= (1.0 - self.adaptation_rate * reward)
        elif reward < 0:
            # Negative reward → increase thresholds for active motors
            # They caused the bad outcome
            active_mask = torch.abs(recent_output) > 1e-6
            self.motor_thresholds[active_mask] *= (1.0 + self.adaptation_rate * abs(reward))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive statistics."""
        recent_feedback = list(self.feedback_history)
        
        if recent_feedback:
            acceptance_rate = sum(f.accepted for f in recent_feedback) / len(recent_feedback)
            avg_threshold = np.mean([f.effective_threshold for f in recent_feedback])
            avg_adaptation = np.mean([f.adaptation_factor for f in recent_feedback])
        else:
            acceptance_rate = 0.0
            avg_threshold = self.base_sensitivity
            avg_adaptation = 1.0
        
        return {
            'acceptance_rate': acceptance_rate,
            'avg_threshold': avg_threshold,
            'avg_adaptation_factor': avg_adaptation,
            'current_thresholds': self.motor_thresholds.detach().cpu().numpy().tolist(),
            'noise_floor': self.noise_floor.mean().item(),
            'habituation': self.habituation_level.detach().cpu().numpy().tolist(),
            'sensitization': self.sensitization_level.detach().cpu().numpy().tolist(),
            'adaptation_events': self.adaptation_events
        }