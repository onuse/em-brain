"""
Predictive Action System

Actions are chosen based on predicted outcomes, not just current field state.
The brain can "imagine" action consequences before committing.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .field_types import FieldNativeAction


@dataclass
class ActionCandidate:
    """Represents a potential action with predicted outcome."""
    motor_pattern: torch.Tensor
    predicted_field: torch.Tensor
    predicted_sensory: Optional[torch.Tensor]
    anticipated_value: float
    uncertainty: float
    exploration_bonus: float
    
    @property
    def selection_score(self) -> float:
        """Combined score for action selection."""
        return self.anticipated_value + self.exploration_bonus


class PredictiveActionSystem:
    """
    Selects actions based on predicted outcomes.
    
    Key principles:
    1. Actions condition field evolution
    2. Field evolution predicts sensory outcomes  
    3. Prediction errors drive exploration
    4. Fantasy allows action planning without execution
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 motor_dim: int = 4,
                 prediction_horizon: int = 5,
                 device: torch.device = torch.device('cpu')):
        """Initialize predictive action system."""
        self.field_shape = field_shape
        self.motor_dim = motor_dim
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        # Action-outcome history for learning
        self.action_history = []
        self.outcome_history = []
        self.max_history = 1000
        
        # Learned action effects (simple for now)
        self.action_effects = {}
        
        # Exploration parameters
        self.exploration_weight = 0.3
        self.uncertainty_threshold = 0.5
        
    def generate_action_candidates(self, 
                                 current_field: torch.Tensor,
                                 current_patterns: Dict[str, Any],
                                 n_candidates: int = 5) -> List[ActionCandidate]:
        """
        Generate candidate actions with different strategies.
        
        Returns diverse actions: exploit, explore, random, etc.
        """
        candidates = []
        
        # 1. Exploitation candidate - follow field gradients
        exploit_action = self._generate_exploit_action(current_field, current_patterns)
        candidates.append(exploit_action)
        
        # 2. Exploration candidates - perpendicular to gradients
        for i in range(2):
            explore_action = self._generate_explore_action(current_field, current_patterns, i)
            candidates.append(explore_action)
            
        # 3. Random candidates for diversity
        for _ in range(n_candidates - 3):
            random_action = self._generate_random_action()
            candidates.append(random_action)
            
        return candidates
        
    def preview_action_outcome(self,
                             current_field: torch.Tensor,
                             action: ActionCandidate,
                             evolution_steps: int = None) -> ActionCandidate:
        """
        Preview what would happen if we took this action.
        
        This is where fantasy/imagination happens!
        """
        if evolution_steps is None:
            evolution_steps = self.prediction_horizon
            
        # Clone field for preview (don't modify actual field)
        preview_field = current_field.clone()
        
        # Apply action influence to field
        # Action "pulls" field in certain directions
        action_influence = self._compute_action_influence(action.motor_pattern)
        preview_field = self._apply_action_influence(preview_field, action_influence)
        
        # Evolve field forward in time (imagination)
        for _ in range(evolution_steps):
            preview_field = self._evolve_field_step(preview_field)
            
        # Predict sensory outcome from final field state
        predicted_sensory = self._predict_sensory_from_field(preview_field)
        
        # Estimate value of predicted state
        anticipated_value = self._estimate_field_value(preview_field)
        
        # Estimate uncertainty (how confident are we?)
        uncertainty = self._estimate_prediction_uncertainty(current_field, preview_field)
        
        # Exploration bonus for uncertain outcomes
        exploration_bonus = self.exploration_weight * uncertainty
        
        # Update action candidate with predictions
        action.predicted_field = preview_field
        action.predicted_sensory = predicted_sensory
        action.anticipated_value = anticipated_value
        action.uncertainty = uncertainty
        action.exploration_bonus = exploration_bonus
        
        return action
        
    def select_action(self,
                     candidates: List[ActionCandidate],
                     exploration_drive: float = 0.5) -> ActionCandidate:
        """
        Select best action based on predicted outcomes.
        
        Balances exploitation (value) and exploration (uncertainty).
        """
        # Adjust exploration weight based on energy-driven exploration drive
        current_exploration = self.exploration_weight * exploration_drive
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            # Combine value and exploration
            score = candidate.anticipated_value + current_exploration * candidate.uncertainty
            scores.append(score)
            
        # Softmax selection (not just argmax) for variety
        scores_tensor = torch.tensor(scores)
        probabilities = F.softmax(scores_tensor / 0.1, dim=0)  # Temperature = 0.1
        
        # Sample action based on probabilities
        selected_idx = torch.multinomial(probabilities, 1).item()
        
        return candidates[selected_idx]
        
    def update_predictions(self,
                          action_taken: torch.Tensor,
                          actual_outcome: torch.Tensor,
                          actual_sensory: torch.Tensor):
        """
        Learn from prediction errors to improve future predictions.
        
        This is where the system learns world dynamics!
        """
        # Add to history
        self.action_history.append(action_taken.detach().clone())
        self.outcome_history.append({
            'field': actual_outcome.detach().clone(),
            'sensory': actual_sensory.detach().clone()
        })
        
        # Trim history
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
            self.outcome_history.pop(0)
            
        # Update action effects model (simple for now)
        # In future: could use neural network to learn complex dynamics
        action_key = self._discretize_action(action_taken)
        if action_key not in self.action_effects:
            self.action_effects[action_key] = []
        self.action_effects[action_key].append(actual_sensory)
        
    def _generate_exploit_action(self, field: torch.Tensor, patterns: Dict) -> ActionCandidate:
        """Generate action that follows current field tendencies."""
        # Extract motor tendency from field patterns
        motor_pattern = torch.zeros(self.motor_dim, device=self.device)
        
        # Simple approach: use field gradients in motor-relevant dimensions
        if 'flow' in patterns:
            flow = patterns['flow']
            motor_pattern[0] = flow.get('x', 0.0)  # Forward/back
            motor_pattern[1] = flow.get('y', 0.0)  # Left/right
            
        return ActionCandidate(
            motor_pattern=motor_pattern,
            predicted_field=field,
            predicted_sensory=None,
            anticipated_value=0.0,
            uncertainty=0.5,
            exploration_bonus=0.0
        )
        
    def _generate_explore_action(self, field: torch.Tensor, patterns: Dict, variant: int) -> ActionCandidate:
        """Generate action perpendicular to current tendencies."""
        exploit = self._generate_exploit_action(field, patterns)
        
        # Rotate 90 degrees for exploration
        explore_pattern = exploit.motor_pattern.clone()
        if variant == 0:
            # Swap and negate for perpendicular
            explore_pattern[0], explore_pattern[1] = -explore_pattern[1], explore_pattern[0]
        else:
            # Different perpendicular direction
            explore_pattern[0], explore_pattern[1] = explore_pattern[1], -explore_pattern[0]
            
        return ActionCandidate(
            motor_pattern=explore_pattern,
            predicted_field=field,
            predicted_sensory=None,
            anticipated_value=0.0,
            uncertainty=1.0,  # High uncertainty for exploration
            exploration_bonus=0.0
        )
        
    def _generate_random_action(self) -> ActionCandidate:
        """Generate random action for diversity."""
        random_pattern = torch.randn(self.motor_dim, device=self.device) * 0.5
        
        return ActionCandidate(
            motor_pattern=random_pattern,
            predicted_field=None,
            predicted_sensory=None,
            anticipated_value=0.0,
            uncertainty=1.0,
            exploration_bonus=0.0
        )
        
    def _compute_action_influence(self, motor_pattern: torch.Tensor) -> torch.Tensor:
        """Compute how action influences field evolution."""
        # Create influence tensor same shape as field
        influence = torch.zeros(self.field_shape, device=self.device)
        
        # Motor patterns influence spatial and flow dimensions
        # This is simplified - in reality would be learned
        if len(motor_pattern) >= 2:
            # Forward/backward influences spatial X dimension
            influence[0, :, :] += motor_pattern[0] * 0.1
            # Left/right influences spatial Y dimension  
            influence[:, 0, :] += motor_pattern[1] * 0.1
            
        return influence
        
    def _apply_action_influence(self, field: torch.Tensor, influence: torch.Tensor) -> torch.Tensor:
        """Apply action influence to field."""
        # Actions "pull" the field in certain directions
        return field + influence
        
    def _evolve_field_step(self, field: torch.Tensor) -> torch.Tensor:
        """Single step of field evolution for prediction."""
        # Simplified evolution - in practice would use full field dynamics
        # Decay
        field = field * 0.99
        
        # Diffusion
        for dim in range(min(3, len(field.shape))):
            shifted_forward = torch.roll(field, shifts=1, dims=dim)
            shifted_backward = torch.roll(field, shifts=-1, dims=dim)
            field = field + 0.01 * (shifted_forward + shifted_backward - 2 * field)
            
        return field
        
    def _predict_sensory_from_field(self, field: torch.Tensor) -> torch.Tensor:
        """Predict sensory input from field state."""
        # Simplified: project field to sensory space
        # In reality, this would be learned from experience
        
        # Take statistics from different field regions
        sensory = torch.zeros(25, device=self.device)  # 24 sensors + reward
        
        # Spatial regions → visual sensors
        spatial_activity = torch.mean(field[:3, :3, :3])
        sensory[:8] = spatial_activity
        
        # Oscillatory regions → temporal sensors
        if len(field.shape) > 3:
            oscillatory_activity = torch.mean(field[3:6, :, :])
            sensory[8:16] = oscillatory_activity
            
        return sensory
        
    def _estimate_field_value(self, field: torch.Tensor) -> float:
        """Estimate value/goodness of a field state."""
        # Simple heuristics for now
        # Good states have moderate energy and low noise
        
        energy = torch.mean(torch.abs(field)).item()
        variance = torch.var(field).item()
        
        # Optimal energy around 0.5, low variance is good
        energy_score = 1.0 - abs(energy - 0.5) * 2
        stability_score = 1.0 / (1.0 + variance)
        
        return 0.7 * energy_score + 0.3 * stability_score
        
    def _estimate_prediction_uncertainty(self, current: torch.Tensor, predicted: torch.Tensor) -> float:
        """Estimate uncertainty of prediction."""
        # High change = high uncertainty
        # Also consider history - repeated patterns = low uncertainty
        
        change = torch.mean(torch.abs(predicted - current)).item()
        
        # Normalize to 0-1 range
        uncertainty = min(1.0, change * 2)
        
        return uncertainty
        
    def _discretize_action(self, action: torch.Tensor) -> str:
        """Convert continuous action to discrete key for learning."""
        # Simple discretization for action effects tracking
        discrete = torch.round(action * 2) / 2  # Round to 0.5 increments
        return str(discrete.tolist())