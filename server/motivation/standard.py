"""
Standard Motivations - Basic Motivation Set

Implements common motivations for robotic intelligence:
- Curiosity: Seeks information and learning opportunities
- Energy: Maintains energy homeostasis  
- Safety: Avoids predicted harm or damage
"""

import random
import numpy as np
from typing import Any, List
from .base import BaseMotivation, ActionProposal


class CuriosityMotivation(BaseMotivation):
    """
    Seeks states with high prediction uncertainty.
    
    Values actions that lead to learning opportunities by finding
    situations where the brain's predictions are uncertain.
    """
    
    def __init__(self, weight: float = 1.0, exploration_bonus: float = 0.1):
        super().__init__("Curiosity", weight)
        self.exploration_bonus = exploration_bonus
        self.uncertainty_threshold = 0.5  # What counts as "interesting" uncertainty
        
    def propose_action(self, current_state: Any) -> ActionProposal:
        """Propose action that maximizes learning opportunity."""
        
        # Generate possible actions to evaluate
        possible_actions = self._generate_exploration_actions(current_state)
        
        best_action = None
        best_uncertainty = 0.0
        best_reasoning = "No valid actions"
        
        for action in possible_actions:
            # Use brain to predict outcome
            predicted_outcome = self.brain.predict(current_state, action)
            
            # Calculate prediction uncertainty
            uncertainty = self._calculate_uncertainty(predicted_outcome)
            
            if uncertainty > best_uncertainty:
                best_uncertainty = uncertainty
                best_action = action
                best_reasoning = f"High uncertainty ({uncertainty:.2f}) → learning opportunity"
        
        # If no good exploration found, try random action
        if best_action is None or best_uncertainty < self.uncertainty_threshold:
            best_action = self._generate_random_action()
            best_reasoning = f"Random exploration (low uncertainty environment)"
            best_uncertainty = self.exploration_bonus
        
        # Convert uncertainty to motivation value
        motivation_value = min(1.0, best_uncertainty + self.exploration_bonus)
        confidence = 0.8  # Curiosity is usually confident about wanting uncertainty
        
        return ActionProposal(
            action=best_action,
            predicted_outcome=self.brain.predict(current_state, best_action),
            motivation_value=motivation_value,
            confidence=confidence,
            motivation_name=self.name,
            reasoning=best_reasoning
        )
    
    def calculate_value(self, predicted_outcome: Any) -> float:
        """Value outcomes based on learning potential."""
        uncertainty = self._calculate_uncertainty(predicted_outcome)
        
        # High uncertainty = high value for curiosity
        return min(1.0, uncertainty + self.exploration_bonus)
    
    def _generate_exploration_actions(self, current_state: Any) -> List[Any]:
        """Generate actions designed for exploration."""
        # This would be customized based on action space
        # For now, generate some basic movement actions
        
        actions = []
        
        # Try different movement directions
        for direction in ['forward', 'backward', 'left', 'right']:
            for speed in [0.2, 0.5, 0.8]:
                actions.append({'type': 'move', 'direction': direction, 'speed': speed})
        
        # Try rotation actions  
        for angle in [-45, -30, -15, 15, 30, 45]:
            actions.append({'type': 'rotate', 'angle': angle})
        
        # Try stopping/waiting
        actions.append({'type': 'stop', 'duration': 1.0})
        
        return actions
    
    def _generate_random_action(self) -> Any:
        """Generate random exploration action."""
        action_type = random.choice(['move', 'rotate', 'stop'])
        
        if action_type == 'move':
            return {
                'type': 'move',
                'direction': random.choice(['forward', 'backward', 'left', 'right']),
                'speed': random.uniform(0.1, 0.9)
            }
        elif action_type == 'rotate':
            return {
                'type': 'rotate', 
                'angle': random.uniform(-60, 60)
            }
        else:
            return {'type': 'stop', 'duration': random.uniform(0.5, 2.0)}
    
    def _calculate_uncertainty(self, predicted_outcome: Any) -> float:
        """Calculate prediction uncertainty."""
        
        if hasattr(self.brain, 'get_prediction_confidence'):
            # Use brain's confidence measure
            confidence = self.brain.get_prediction_confidence(predicted_outcome)
            uncertainty = 1.0 - confidence
            
            # Add action-based uncertainty factors
            if hasattr(predicted_outcome, 'action'):
                action = predicted_outcome.action
                if isinstance(action, dict):
                    action_type = action.get('type', 'unknown')
                    
                    # Different actions have different learning potential
                    if action_type == 'move':
                        speed = action.get('speed', 0.5)
                        # Faster movement = more uncertain outcomes
                        uncertainty += speed * 0.3
                    elif action_type == 'rotate':
                        angle = abs(action.get('angle', 0))
                        # Larger rotations = more uncertainty about new view
                        uncertainty += (angle / 90) * 0.4
                    elif action_type == 'stop':
                        # Stopping provides little learning
                        uncertainty *= 0.5
            
            return min(1.0, uncertainty)
        
        # Fallback: simulate uncertainty based on outcome novelty
        if hasattr(predicted_outcome, 'similarity_score'):
            # Lower similarity = higher uncertainty
            return 1.0 - predicted_outcome.similarity_score
        
        # Default: medium uncertainty
        return 0.5


class EnergyMotivation(BaseMotivation):
    """
    Maintains energy homeostasis through conservative behavior.
    
    Values actions that preserve energy and seeks charging when needed.
    """
    
    def __init__(self, weight: float = 1.0, energy_threshold: float = 0.3):
        super().__init__("Energy", weight)
        self.energy_threshold = energy_threshold  # Below this = urgent energy seeking
        self.comfort_zone = (0.6, 0.9)  # Preferred energy range
        
    def propose_action(self, current_state: Any) -> ActionProposal:
        """Propose energy-conscious action."""
        
        current_energy = self._get_energy_level(current_state)
        
        if current_energy < self.energy_threshold:
            # Critical energy - seek charging/rest urgently
            action = self._propose_energy_seeking_action(current_state)
            motivation_value = 0.9  # Very high priority
            confidence = 0.95  # Very confident about energy needs
            reasoning = f"Critical energy ({current_energy:.1%}) - seeking power"
            
        elif current_energy < self.comfort_zone[0]:
            # Low energy - prefer conservative actions
            action = self._propose_conservative_action(current_state)
            motivation_value = 0.6
            confidence = 0.8
            reasoning = f"Low energy ({current_energy:.1%}) - conserving power"
            
        elif current_energy > self.comfort_zone[1]:
            # High energy - can afford more activity
            action = self._propose_normal_action(current_state)
            motivation_value = 0.3
            confidence = 0.6
            reasoning = f"Good energy ({current_energy:.1%}) - normal activity OK"
            
        else:
            # Comfortable energy range
            action = self._propose_normal_action(current_state)
            motivation_value = 0.4
            confidence = 0.7
            reasoning = f"Comfortable energy ({current_energy:.1%}) - moderate activity"
        
        return ActionProposal(
            action=action,
            predicted_outcome=self.brain.predict(current_state, action),
            motivation_value=motivation_value,
            confidence=confidence,
            motivation_name=self.name,
            reasoning=reasoning
        )
    
    def calculate_value(self, predicted_outcome: Any) -> float:
        """Value outcomes based on energy impact."""
        
        # Estimate energy cost of predicted outcome
        energy_cost = self._estimate_energy_cost(predicted_outcome)
        current_energy = self._get_current_energy()
        
        if current_energy < self.energy_threshold:
            # Critical energy - heavily penalize energy use
            return max(0.0, 1.0 - (energy_cost * 3.0))
        else:
            # Normal energy - moderate energy consciousness
            return max(0.0, 1.0 - energy_cost)
    
    def _get_energy_level(self, state: Any) -> float:
        """Extract energy level from state (0.0 = empty, 1.0 = full)."""
        
        # Try common energy field names
        if hasattr(state, 'battery'):
            return state.battery
        elif hasattr(state, 'energy'):
            return state.energy
        elif isinstance(state, dict):
            return state.get('battery', state.get('energy', 0.7))  # Default to 70%
        
        # Fallback: assume moderate energy
        return 0.7
    
    def _get_current_energy(self) -> float:
        """Get current energy level (fallback for value calculation)."""
        # This would be improved with access to current state
        return 0.6  # Conservative estimate
    
    def _propose_energy_seeking_action(self, state: Any) -> Any:
        """Propose action to find energy/charging."""
        return {'type': 'seek_charger', 'urgency': 'high'}
    
    def _propose_conservative_action(self, state: Any) -> Any:
        """Propose low-energy action."""
        return {'type': 'move', 'direction': 'forward', 'speed': 0.2}
    
    def _propose_normal_action(self, state: Any) -> Any:
        """Propose normal activity level action."""
        return {'type': 'move', 'direction': 'forward', 'speed': 0.5}
    
    def _estimate_energy_cost(self, predicted_outcome: Any) -> float:
        """Estimate energy cost of predicted outcome."""
        
        # This would be customized based on outcome format
        if hasattr(predicted_outcome, 'energy_cost'):
            return predicted_outcome.energy_cost
        
        # Rough estimates based on action type
        if hasattr(predicted_outcome, 'action'):
            action = predicted_outcome.action
            if isinstance(action, dict):
                if action.get('type') == 'move':
                    speed = action.get('speed', 0.5)
                    return speed * 0.1  # Higher speed = more energy
                elif action.get('type') == 'rotate':
                    return 0.05  # Rotation uses moderate energy
                elif action.get('type') == 'stop':
                    return 0.01  # Minimal energy when stopped
        
        # Default: assume moderate energy cost
        return 0.05


class SafetyMotivation(BaseMotivation):
    """
    Avoids predicted negative outcomes and maintains safe operation.
    
    Values actions that minimize risk of damage, collision, or failure.
    """
    
    def __init__(self, weight: float = 1.0, safety_threshold: float = 0.7):
        super().__init__("Safety", weight)
        self.safety_threshold = safety_threshold  # Minimum acceptable safety
        self.danger_memory = []  # Remember dangerous situations
        
    def propose_action(self, current_state: Any) -> ActionProposal:
        """Propose safest available action."""
        
        # Store current state for safety calculations
        self._current_state = current_state
        
        possible_actions = self._generate_safe_actions(current_state)
        
        best_action = None
        best_safety = 0.0
        best_reasoning = "No safe actions found"
        
        for action in possible_actions:
            predicted_outcome = self.brain.predict(current_state, action)
            safety_score = self._calculate_safety(predicted_outcome)
            
            if safety_score > best_safety:
                best_safety = safety_score
                best_action = action
                best_reasoning = f"Safest option (safety={safety_score:.2f})"
        
        # If no action meets safety threshold, choose least dangerous
        if best_safety < self.safety_threshold:
            best_reasoning = f"Emergency choice (safety={best_safety:.2f} < {self.safety_threshold:.2f})"
        
        # Convert safety to motivation value
        motivation_value = best_safety
        confidence = 0.9 if best_safety > self.safety_threshold else 0.6
        
        return ActionProposal(
            action=best_action or self._emergency_action(),
            predicted_outcome=self.brain.predict(current_state, best_action or self._emergency_action()),
            motivation_value=motivation_value,
            confidence=confidence,
            motivation_name=self.name,
            reasoning=best_reasoning
        )
    
    def calculate_value(self, predicted_outcome: Any) -> float:
        """Value outcomes based on safety."""
        return self._calculate_safety(predicted_outcome)
    
    def _generate_safe_actions(self, current_state: Any) -> List[Any]:
        """Generate actions prioritizing safety."""
        
        actions = []
        
        # Conservative movement actions
        for direction in ['forward', 'backward']:
            for speed in [0.1, 0.3]:  # Only slow speeds for safety
                actions.append({'type': 'move', 'direction': direction, 'speed': speed})
        
        # Small rotation adjustments
        for angle in [-15, -10, 10, 15]:
            actions.append({'type': 'rotate', 'angle': angle})
        
        # Stopping is usually safe
        actions.append({'type': 'stop', 'duration': 0.5})
        
        return actions
    
    def _calculate_safety(self, predicted_outcome: Any) -> float:
        """Calculate safety score for predicted outcome."""
        
        # Start with baseline safety that varies with action risk
        safety_score = 0.7  # Not perfect by default - room for improvement
        
        # Evaluate action risk from the action itself
        if hasattr(predicted_outcome, 'action'):
            action = predicted_outcome.action
            if isinstance(action, dict):
                action_type = action.get('type', 'unknown')
                
                if action_type == 'stop':
                    safety_score = 0.9  # Stopping is usually very safe
                elif action_type == 'move':
                    speed = action.get('speed', 0.5)
                    # Higher speed = lower safety
                    safety_score = max(0.3, 0.9 - (speed * 0.4))
                elif action_type == 'rotate':
                    angle = abs(action.get('angle', 0))
                    # Large rotations are riskier
                    safety_score = max(0.5, 0.8 - (angle / 90) * 0.3)
                elif action_type == 'seek_charger':
                    safety_score = 0.6  # Movement inherently has some risk
        
        # Check similarity to past dangerous situations
        danger_similarity = self._similarity_to_danger(predicted_outcome)
        safety_score -= danger_similarity * 0.8
        
        # Check for collision indicators
        if hasattr(predicted_outcome, 'collision_risk'):
            safety_score -= predicted_outcome.collision_risk * 0.9
        
        # Check for damage indicators  
        if hasattr(predicted_outcome, 'damage_risk'):
            safety_score -= predicted_outcome.damage_risk
        
        # Check distance to obstacles from current state if available
        # This is a heuristic - closer obstacles make any movement riskier
        if hasattr(self, '_current_state'):
            if hasattr(self._current_state, 'obstacle_distance'):
                distance = self._current_state.obstacle_distance
                if distance < 30:  # Within 30cm is concerning
                    risk_factor = max(0, (30 - distance) / 30)
                    safety_score -= risk_factor * 0.4
        
        return max(0.0, min(1.0, safety_score))
    
    def _similarity_to_danger(self, predicted_outcome: Any) -> float:
        """Check similarity to previously dangerous situations."""
        
        if not self.danger_memory:
            return 0.0
        
        # This would use the brain's similarity engine
        # For now, return low similarity
        return 0.1
    
    def _emergency_action(self) -> Any:
        """Emergency fallback action when no safe options exist."""
        return {'type': 'stop', 'duration': 2.0}  # Stop and wait
    
    def record_danger(self, dangerous_situation: Any):
        """Record a dangerous situation to avoid in future."""
        self.danger_memory.append(dangerous_situation)
        
        # Keep memory manageable
        if len(self.danger_memory) > 100:
            self.danger_memory = self.danger_memory[-50:]