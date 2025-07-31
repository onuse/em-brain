#!/usr/bin/env python3
"""
Action-Prediction Integration System - Phase 4

Every action is an experiment to test predictions. The brain selects actions
based on:
1. Predicted outcomes at multiple timescales
2. Confidence in those predictions
3. Information gain potential (reducing uncertainty)

Key insight: Actions are not outputs, they are queries to test hypotheses
about how the world works.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from ...utils.tensor_ops import create_zeros


@dataclass 
class PredictiveAction:
    """An action with its predicted outcomes at multiple timescales."""
    motor_values: torch.Tensor
    
    # Predicted sensory outcomes
    immediate_outcome: torch.Tensor      # What happens next cycle
    short_term_outcome: torch.Tensor     # Average over ~10 cycles
    long_term_outcome: torch.Tensor      # Average over ~100 cycles
    
    # Confidence in predictions
    immediate_confidence: float
    short_term_confidence: float
    long_term_confidence: float
    
    # Expected information gain
    uncertainty_reduction: float
    
    # Action type for tracking
    action_type: str  # 'exploit', 'explore', 'test'


class ActionPredictionSystem:
    """
    Integrates action selection with hierarchical predictions.
    
    Actions are selected to:
    1. Achieve desired outcomes (exploit predictions)
    2. Test uncertain predictions (explore boundaries) 
    3. Maximize information gain (reduce uncertainty)
    """
    
    def __init__(self,
                 motor_dim: int,
                 sensory_dim: int,
                 device: torch.device):
        self.motor_dim = motor_dim
        self.sensory_dim = sensory_dim
        self.device = device
        
        # Action-outcome history for learning
        self.action_history = deque(maxlen=1000)
        self.outcome_history = deque(maxlen=1000)
        
        # Learned action-outcome mappings at each timescale
        # These predict: given action A, what sensory input S will result?
        self.immediate_action_weights = create_zeros((motor_dim, sensory_dim), device=device)
        self.short_term_action_weights = create_zeros((motor_dim, sensory_dim), device=device)
        self.long_term_action_weights = create_zeros((motor_dim, sensory_dim), device=device)
        
        # Initialize with small random weights
        torch.nn.init.xavier_uniform_(self.immediate_action_weights, gain=0.5)
        torch.nn.init.xavier_uniform_(self.short_term_action_weights, gain=0.3)
        torch.nn.init.xavier_uniform_(self.long_term_action_weights, gain=0.2)
        
        # Learning rates for action-outcome mappings
        self.action_lr = 0.1
        
        # Track prediction accuracy per action type
        self.action_type_errors = {
            'random': deque(maxlen=100),
            'exploit': deque(maxlen=100),
            'explore': deque(maxlen=100),
            'test': deque(maxlen=100)
        }
        
    def generate_action_candidates(self, 
                                 current_predictions: Dict[str, torch.Tensor],
                                 n_candidates: int = 5) -> List[PredictiveAction]:
        """
        Generate candidate actions with their predicted outcomes.
        
        Args:
            current_predictions: Current sensory predictions at all timescales
            n_candidates: Number of action candidates to generate
            
        Returns:
            List of PredictiveAction candidates
        """
        candidates = []
        
        # 1. Random exploration action
        random_action = torch.randn(self.motor_dim, device=self.device) * 0.5
        random_candidate = self._predict_action_outcome(random_action, 'explore')
        candidates.append(random_candidate)
        
        # 2. Exploitation actions - actions that maintain current predictions
        if 'immediate' in current_predictions:
            # Find action that would produce the current prediction
            exploit_action = self._invert_prediction(current_predictions['immediate'])
            exploit_candidate = self._predict_action_outcome(exploit_action, 'exploit')
            candidates.append(exploit_candidate)
        
        # 3. Test actions - systematically vary each motor dimension
        for i in range(min(self.motor_dim, n_candidates - 2)):
            test_action = torch.zeros(self.motor_dim, device=self.device)
            test_action[i] = 1.0  # Unit impulse in one dimension
            test_candidate = self._predict_action_outcome(test_action, 'test')
            candidates.append(test_candidate)
        
        # 4. Information-seeking actions - high uncertainty areas
        for _ in range(n_candidates - len(candidates)):
            # Generate action towards areas of high prediction uncertainty
            info_action = self._generate_information_seeking_action()
            info_candidate = self._predict_action_outcome(info_action, 'explore')
            candidates.append(info_candidate)
        
        return candidates[:n_candidates]
    
    def select_action(self,
                     candidates: List[PredictiveAction],
                     hierarchical_predictions: Any,
                     exploration_drive: float = 0.5) -> Tuple[torch.Tensor, PredictiveAction]:
        """
        Select best action based on predictions and exploration drive.
        
        Args:
            candidates: List of candidate actions
            hierarchical_predictions: Current hierarchical predictions from brain
            exploration_drive: Balance between exploitation (0) and exploration (1)
            
        Returns:
            Selected motor values and full PredictiveAction
        """
        if not candidates:
            # Fallback to random action
            return torch.randn(self.motor_dim, device=self.device) * 0.3, None
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            # Base score from predicted outcome desirability
            outcome_score = self._score_predicted_outcome(candidate)
            
            # Confidence bonus - prefer actions we're confident about
            confidence_score = (candidate.immediate_confidence * 0.5 +
                              candidate.short_term_confidence * 0.3 +
                              candidate.long_term_confidence * 0.2)
            
            # Information gain bonus - prefer actions that reduce uncertainty
            info_score = candidate.uncertainty_reduction
            
            # Combine scores based on exploration drive
            if exploration_drive > 0.7:
                # High exploration: prioritize information gain
                total_score = info_score * 0.7 + outcome_score * 0.3
            elif exploration_drive < 0.3:
                # Low exploration: prioritize confident good outcomes
                total_score = outcome_score * 0.6 + confidence_score * 0.4
            else:
                # Balanced: mix all factors
                total_score = outcome_score * 0.4 + confidence_score * 0.3 + info_score * 0.3
            
            scores.append(total_score)
        
        # Select best action
        best_idx = np.argmax(scores)
        selected = candidates[best_idx]
        
        # Add noise based on exploration drive
        noise = torch.randn_like(selected.motor_values) * exploration_drive * 0.2
        final_motor = selected.motor_values + noise
        
        # Clamp to valid range
        final_motor = torch.clamp(final_motor, -1.0, 1.0)
        
        return final_motor, selected
    
    def update_action_outcome_mapping(self,
                                    action: torch.Tensor,
                                    actual_outcome: torch.Tensor,
                                    predicted_action: Optional[PredictiveAction] = None):
        """
        Update action-outcome mappings based on actual results.
        
        Args:
            action: The action that was taken
            actual_outcome: The sensory outcome that resulted
            predicted_action: The prediction that was made (if any)
        """
        # Store in history - ensure consistent size
        self.action_history.append(action.detach().clone())
        # Pad or truncate outcome to consistent size
        if actual_outcome.shape[0] < self.sensory_dim:
            # Pad with zeros if too small
            padded = torch.zeros(self.sensory_dim, device=self.device)
            padded[:actual_outcome.shape[0]] = actual_outcome
            self.outcome_history.append(padded.detach().clone())
        else:
            # Truncate if too large
            self.outcome_history.append(actual_outcome[:self.sensory_dim].detach().clone())
        
        # Update immediate mapping (gradient descent)
        # Ensure dimensions match
        min_dim = min(actual_outcome.shape[0], self.sensory_dim)
        # Fix: matmul produces full sensory_dim output, we need to slice before comparing
        full_prediction = torch.matmul(action, self.immediate_action_weights)
        prediction = full_prediction[:min_dim]
        error = actual_outcome[:min_dim] - prediction
        
        # Weight update
        weight_grad = torch.outer(action, error)
        self.immediate_action_weights[:, :min_dim] += weight_grad[:, :min_dim] * self.action_lr
        
        # Track error by action type
        if predicted_action:
            error_magnitude = torch.mean(torch.abs(error)).item()
            self.action_type_errors[predicted_action.action_type].append(error_magnitude)
        
        # Update longer timescale mappings if we have enough history
        if len(self.action_history) >= 10:
            # Short-term: average over last 10 actions/outcomes
            recent_actions = torch.stack(list(self.action_history)[-10:])
            recent_outcomes = torch.stack(list(self.outcome_history)[-10:])
            
            avg_action = torch.mean(recent_actions, dim=0)
            avg_outcome = torch.mean(recent_outcomes, dim=0)
            
            # Update short-term mapping
            short_pred = torch.matmul(avg_action, self.short_term_action_weights)
            # Ensure dimensions match for comparison
            min_dim = min(avg_outcome.shape[0], short_pred.shape[0])
            short_error = avg_outcome[:min_dim] - short_pred[:min_dim]
            short_grad = torch.outer(avg_action, short_error)
            self.short_term_action_weights[:, :min_dim] += short_grad[:, :min_dim] * self.action_lr * 0.5
        
        if len(self.action_history) >= 100:
            # Long-term: average over last 100 actions/outcomes
            long_actions = torch.stack(list(self.action_history)[-100:])
            long_outcomes = torch.stack(list(self.outcome_history)[-100:])
            
            avg_action = torch.mean(long_actions, dim=0)
            avg_outcome = torch.mean(long_outcomes, dim=0)
            
            # Update long-term mapping
            long_pred = torch.matmul(avg_action, self.long_term_action_weights)
            # Ensure dimensions match for comparison
            min_dim = min(avg_outcome.shape[0], long_pred.shape[0])
            long_error = avg_outcome[:min_dim] - long_pred[:min_dim]
            long_grad = torch.outer(avg_action, long_error)
            self.long_term_action_weights[:, :min_dim] += long_grad[:, :min_dim] * self.action_lr * 0.1
    
    def _predict_action_outcome(self, action: torch.Tensor, action_type: str) -> PredictiveAction:
        """Predict outcomes for a given action at all timescales."""
        # Immediate prediction
        immediate_pred = torch.tanh(torch.matmul(action, self.immediate_action_weights))
        
        # Short-term prediction (smoothed)
        short_pred = torch.tanh(torch.matmul(action, self.short_term_action_weights) * 0.8)
        
        # Long-term prediction (very smoothed)
        long_pred = torch.tanh(torch.matmul(action, self.long_term_action_weights) * 0.5)
        
        # Calculate confidence based on historical accuracy
        immediate_conf = self._calculate_confidence('immediate', action_type)
        short_conf = self._calculate_confidence('short_term', action_type)
        long_conf = self._calculate_confidence('long_term', action_type)
        
        # Estimate uncertainty reduction
        # Actions that move away from recent actions have higher info gain
        if len(self.action_history) > 0:
            recent_actions = torch.stack(list(self.action_history)[-10:]) if len(self.action_history) >= 10 else torch.stack(list(self.action_history))
            novelty = torch.mean(torch.abs(action.unsqueeze(0) - recent_actions)).item()
            uncertainty_reduction = min(1.0, novelty * 2.0)
        else:
            uncertainty_reduction = 0.5
        
        return PredictiveAction(
            motor_values=action,
            immediate_outcome=immediate_pred,
            short_term_outcome=short_pred,
            long_term_outcome=long_pred,
            immediate_confidence=immediate_conf,
            short_term_confidence=short_conf,
            long_term_confidence=long_conf,
            uncertainty_reduction=uncertainty_reduction,
            action_type=action_type
        )
    
    def _invert_prediction(self, desired_outcome: torch.Tensor) -> torch.Tensor:
        """Find action that would produce desired sensory outcome."""
        # Simple linear inversion (pseudo-inverse)
        # In reality, this could be learned through experience
        try:
            # Use immediate weights to invert
            # Move to CPU for pinverse if on MPS to avoid warning
            if self.device.type == 'mps':
                weights_cpu = self.immediate_action_weights.T.cpu()
                pseudo_inv = torch.pinverse(weights_cpu).to(self.device)
            else:
                pseudo_inv = torch.pinverse(self.immediate_action_weights.T)
            action = torch.matmul(pseudo_inv, desired_outcome)
            return torch.clamp(action, -1.0, 1.0)
        except:
            # Fallback to small random action
            return torch.randn(self.motor_dim, device=self.device) * 0.1
    
    def _generate_information_seeking_action(self) -> torch.Tensor:
        """Generate action that explores uncertain areas."""
        if len(self.action_history) < 10:
            # Not enough history - random exploration
            return torch.randn(self.motor_dim, device=self.device)
        
        # Find dimensions with high variance in recent actions
        recent_actions = torch.stack(list(self.action_history)[-20:])
        action_std = torch.std(recent_actions, dim=0)
        
        # Explore dimensions with low variance (underexplored)
        exploration_weights = 1.0 / (action_std + 0.1)
        exploration_weights = exploration_weights / torch.sum(exploration_weights)
        
        # Generate action biased towards underexplored dimensions
        action = torch.randn(self.motor_dim, device=self.device)
        action = action * exploration_weights
        
        return torch.clamp(action, -1.0, 1.0)
    
    def _score_predicted_outcome(self, candidate: PredictiveAction) -> float:
        """Score how desirable a predicted outcome is."""
        # For now, prefer outcomes that:
        # 1. Are moderate (not extreme)
        # 2. Are predictable (high confidence)
        # 3. Are stable across timescales
        
        # Penalize extreme predictions
        immediate_extremity = torch.mean(torch.abs(candidate.immediate_outcome)).item()
        moderation_score = 1.0 - immediate_extremity
        
        # Reward stability across timescales
        stability = 1.0 - torch.mean(torch.abs(
            candidate.immediate_outcome - candidate.long_term_outcome
        )).item()
        
        # Combine factors
        return moderation_score * 0.6 + stability * 0.4
    
    def _calculate_confidence(self, timescale: str, action_type: str) -> float:
        """Calculate confidence based on historical prediction accuracy."""
        if action_type not in self.action_type_errors:
            return 0.5
        
        errors = self.action_type_errors[action_type]
        if len(errors) < 5:
            return 0.5
        
        # Lower error = higher confidence
        recent_errors = list(errors)[-10:]
        mean_error = sum(recent_errors) / len(recent_errors)
        confidence = 1.0 - min(1.0, mean_error * 2.0)
        
        # Decay confidence for longer timescales
        if timescale == 'short_term':
            confidence *= 0.8
        elif timescale == 'long_term':
            confidence *= 0.6
        
        return confidence
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action-outcome learning."""
        stats = {
            'total_actions': len(self.action_history),
            'action_types': {},
            'prediction_accuracy': {}
        }
        
        # Calculate average error per action type
        for action_type, errors in self.action_type_errors.items():
            if errors:
                stats['action_types'][action_type] = {
                    'count': len(errors),
                    'avg_error': sum(errors) / len(errors),
                    'improving': list(errors)[-1] < list(errors)[0] if len(errors) > 1 else False
                }
        
        # Check if action-outcome mappings are improving
        if len(self.action_history) > 20:
            # Test current mapping accuracy
            recent_actions = torch.stack(list(self.action_history)[-10:])
            recent_outcomes = torch.stack(list(self.outcome_history)[-10:])
            
            predictions = torch.tanh(torch.matmul(recent_actions, self.immediate_action_weights))
            errors = torch.mean(torch.abs(predictions - recent_outcomes), dim=1)
            
            stats['prediction_accuracy'] = {
                'immediate': 1.0 - torch.mean(errors).item(),
                'learning': torch.mean(errors[:5]) > torch.mean(errors[5:])
            }
        
        return stats