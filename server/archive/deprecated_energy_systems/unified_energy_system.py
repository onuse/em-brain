"""
Unified Energy System

Implements a biologically-inspired energy management system where:
- Low energy = hungry for exploration and new patterns
- High energy = satiated, time to consolidate and organize
- Energy naturally flows through exploration->pattern discovery->consolidation cycles
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class UnifiedEnergyConfig:
    """Configuration for unified energy system."""
    # Energy thresholds
    hungry_threshold: float = 0.3  # Below this = exploration mode
    satiated_threshold: float = 0.7  # Above this = consolidation mode
    
    # Transition timescales
    energy_smoothing_window: int = 1000  # Cycles for smooth transitions
    mode_transition_cycles: int = 200  # Minimum cycles before mode change
    
    # Energy gain parameters
    novelty_energy_gain: float = 0.1  # Energy from novel patterns
    prediction_success_gain: float = 0.02  # Small energy from good predictions
    reward_energy_scale: float = 0.2  # Reward signal energy multiplier
    habituation_threshold: int = 10  # Times before pattern stops giving energy
    
    # Energy loss parameters
    base_decay_rate: float = 0.999  # Very slow natural decay
    consolidation_decay: float = 0.99  # Faster decay when consolidating
    
    # Behavior modulation
    exploration_sensory_boost: float = 2.0  # Sensory amplification when hungry
    exploration_motor_noise: float = 0.3  # Motor variability when exploring
    consolidation_spontaneous: float = 0.8  # Spontaneous weight when satiated


class EnergyState:
    """Tracks current energy state and transitions."""
    def __init__(self):
        self.current_energy = 0.5  # Start in middle
        self.smoothed_energy = 0.5
        self.mode = "BALANCED"  # HUNGRY, BALANCED, SATIATED
        self.mode_timer = 0
        self.last_mode_change = 0
        
        # Pattern tracking for habituation
        self.pattern_history: Dict[int, int] = {}  # hash -> count
        self.recent_patterns: deque = deque(maxlen=100)
        
        # Energy history for smooth transitions
        self.energy_history: deque = deque(maxlen=1000)


class UnifiedEnergySystem:
    """
    Manages field energy as a unified system driving exploration and consolidation.
    """
    
    def __init__(self, config: Optional[UnifiedEnergyConfig] = None, quiet_mode: bool = False):
        """Initialize unified energy system."""
        self.config = config or UnifiedEnergyConfig()
        self.quiet_mode = quiet_mode
        self.state = EnergyState()
        
        # Track energy sources
        self._last_novelty = 0.0
        self._last_prediction_success = 0.0
        self._last_reward_energy = 0.0
        
    def update_energy(self, 
                     field: torch.Tensor,
                     sensory_pattern: Optional[torch.Tensor] = None,
                     prediction_error: float = 0.5,
                     reward: float = 0.0) -> Dict[str, Any]:
        """
        Update energy based on current state and inputs.
        
        Args:
            field: Current field state
            sensory_pattern: Current sensory input pattern
            prediction_error: How wrong our prediction was (0=perfect, 1=terrible)
            reward: External reward signal (-1 to 1)
            
        Returns:
            Energy update info and behavioral recommendations
        """
        # Calculate current raw energy
        raw_energy = float(torch.mean(torch.abs(field)))
        self.state.energy_history.append(raw_energy)
        
        # Smooth energy over long timescale
        if len(self.state.energy_history) > 100:
            window_size = min(len(self.state.energy_history), self.config.energy_smoothing_window)
            self.state.smoothed_energy = float(np.mean(list(self.state.energy_history)[-window_size:]))
        else:
            self.state.smoothed_energy = raw_energy
            
        # Calculate energy changes from various sources
        energy_delta = 0.0
        
        # 1. Novelty detection adds energy
        if sensory_pattern is not None:
            pattern_hash = hash(tuple(sensory_pattern.flatten().tolist()[:10]))  # Simple hash
            
            # Check if pattern is novel
            if pattern_hash not in self.state.pattern_history:
                # Completely novel pattern!
                energy_delta += self.config.novelty_energy_gain
                self._last_novelty = self.config.novelty_energy_gain
                self.state.pattern_history[pattern_hash] = 1
            else:
                # Seen before - check habituation
                count = self.state.pattern_history[pattern_hash]
                if count < self.config.habituation_threshold:
                    # Still interesting
                    energy_gain = self.config.novelty_energy_gain * (1.0 - count / self.config.habituation_threshold)
                    energy_delta += energy_gain
                    self._last_novelty = energy_gain
                    self.state.pattern_history[pattern_hash] = count + 1
                else:
                    # Habituated - no energy
                    self._last_novelty = 0.0
                    
            self.state.recent_patterns.append(pattern_hash)
            
        # 2. Successful predictions add small energy
        prediction_success = 1.0 - prediction_error
        if prediction_success > 0.7:  # Good prediction
            energy_delta += self.config.prediction_success_gain * prediction_success
            self._last_prediction_success = self.config.prediction_success_gain * prediction_success
        else:
            self._last_prediction_success = 0.0
            
        # 3. Rewards add energy proportional to magnitude
        if abs(reward) > 0.01:
            reward_energy = abs(reward) * self.config.reward_energy_scale
            energy_delta += reward_energy
            self._last_reward_energy = reward_energy
        else:
            self._last_reward_energy = 0.0
            
        # 4. Natural decay based on current mode
        if self.state.mode == "SATIATED":
            decay_rate = self.config.consolidation_decay
        else:
            decay_rate = self.config.base_decay_rate
            
        # Update current energy
        self.state.current_energy = self.state.smoothed_energy + energy_delta
        self.state.current_energy *= decay_rate
        
        # Update mode based on energy level
        self._update_mode()
        
        # Generate behavioral recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'current_energy': self.state.current_energy,
            'smoothed_energy': self.state.smoothed_energy,
            'mode': self.state.mode,
            'mode_timer': self.state.mode_timer,
            'energy_sources': {
                'novelty': self._last_novelty,
                'prediction': self._last_prediction_success,
                'reward': self._last_reward_energy
            },
            'recommendations': recommendations
        }
        
    def _update_mode(self):
        """Update behavioral mode based on energy level."""
        old_mode = self.state.mode
        
        # Determine target mode based on energy
        if self.state.smoothed_energy < self.config.hungry_threshold:
            target_mode = "HUNGRY"
        elif self.state.smoothed_energy > self.config.satiated_threshold:
            target_mode = "SATIATED"
        else:
            target_mode = "BALANCED"
            
        # Handle mode transitions with hysteresis
        if target_mode != self.state.mode:
            if self.state.mode_timer >= self.config.mode_transition_cycles:
                # Enough time has passed - change mode
                self.state.mode = target_mode
                self.state.mode_timer = 0
                self.state.last_mode_change = time.time()
                
                if not self.quiet_mode:
                    energy_desc = f"energy={self.state.smoothed_energy:.3f}"
                    if target_mode == "HUNGRY":
                        print(f"🔋 Energy state: HUNGRY ({energy_desc}) - Seeking new patterns")
                    elif target_mode == "SATIATED":
                        print(f"⚡ Energy state: SATIATED ({energy_desc}) - Consolidating patterns")
                    else:
                        print(f"🔸 Energy state: BALANCED ({energy_desc})")
            else:
                # Increment timer toward transition
                self.state.mode_timer += 1
        else:
            # Reset timer if we're back to current mode
            self.state.mode_timer = 0
            
    def _generate_recommendations(self) -> Dict[str, float]:
        """Generate behavioral recommendations based on energy state."""
        recs = {}
        
        if self.state.mode == "HUNGRY":
            # Low energy - explore and seek patterns
            recs['sensory_amplification'] = self.config.exploration_sensory_boost
            recs['motor_noise'] = self.config.exploration_motor_noise
            recs['spontaneous_weight'] = 0.2  # Less fantasy, more reality
            recs['decay_rate'] = 1.0  # No decay when hungry
            recs['attention_bias'] = 'novelty'  # Focus on new things
            
        elif self.state.mode == "SATIATED":
            # High energy - consolidate and organize
            recs['sensory_amplification'] = 0.5  # Reduced sensory attention
            recs['motor_noise'] = 0.1  # Smooth, refined movements
            recs['spontaneous_weight'] = self.config.consolidation_spontaneous  # More fantasy
            recs['decay_rate'] = self.config.consolidation_decay  # Allow consolidation
            recs['attention_bias'] = 'familiarity'  # Focus on known patterns
            
        else:  # BALANCED
            # Middle ground
            recs['sensory_amplification'] = 1.0
            recs['motor_noise'] = 0.2
            recs['spontaneous_weight'] = 0.5
            recs['decay_rate'] = self.config.base_decay_rate
            recs['attention_bias'] = 'balanced'
            
        return recs
        
    def apply_energy_modulation(self, field: torch.Tensor, 
                               activation_threshold: float = 0.001) -> torch.Tensor:
        """
        Apply energy-based modulation to field directly.
        
        This replaces the complex maintenance system with simple, unified control.
        """
        # Get current recommendations
        recommendations = self._generate_recommendations()
        decay = recommendations.get('decay_rate', 1.0)
        
        # Apply decay based on energy state
        if decay < 1.0:
            field *= decay
            
        # Prune weak activations when consolidating (high energy)
        if self.state.mode == "SATIATED":
            # During consolidation, clean up noise
            mask = torch.abs(field) > activation_threshold
            field *= mask.float()
            # Don't add baseline - let weak activations truly disappear
            field[~mask] = 0.0
        
        # Prevent complete field death only in extreme cases
        elif self.state.mode == "HUNGRY" and self.state.current_energy < 0.01:
            # Emergency baseline to prevent complete death
            # But only for active regions
            active_mask = torch.abs(field) > 0
            field = torch.where(active_mask, 
                              torch.maximum(field, torch.tensor(activation_threshold, device=field.device)),
                              field)
            
        # No artificial energy injection or restoration
        # Energy comes naturally from patterns and exploration
        return field
        
    def get_energy_state(self) -> Dict[str, Any]:
        """Get current energy state for monitoring."""
        return {
            'mode': self.state.mode,
            'current_energy': self.state.current_energy,
            'smoothed_energy': self.state.smoothed_energy,
            'energy_sources': {
                'novelty': self._last_novelty,
                'prediction': self._last_prediction_success,
                'reward': self._last_reward_energy
            },
            'pattern_memory_size': len(self.state.pattern_history),
            'mode_stability': self.state.mode_timer / self.config.mode_transition_cycles
        }