"""
Unified Learning Drive - Emergent curiosity and exploration through learning dynamics.

This replaces separate curiosity and exploration drives with a single adaptive system
that naturally exhibits both behaviors based on learning gradients and context.

Key Principles:
1. No artificial thresholds or floors
2. Behavior emerges from learning dynamics
3. Parameters auto-tune to robot's capabilities
4. Maturity emerges from learning efficiency
"""

import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import deque
from .base_drive import BaseDrive, DriveContext, ActionEvaluation


class LearningGradient:
    """Measures how much the robot is learning in different contexts."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.prediction_errors = deque(maxlen=window_size)
        self.learning_events = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add_learning_event(self, prediction_error: float, timestamp: float):
        """Record a learning event."""
        self.prediction_errors.append(prediction_error)
        self.timestamps.append(timestamp)
        
        # Calculate learning event strength (how much we learned from this experience)
        if len(self.prediction_errors) >= 2:
            previous_error = self.prediction_errors[-2]
            learning_strength = max(0, previous_error - prediction_error)  # Positive = learning
            self.learning_events.append(learning_strength)
        else:
            self.learning_events.append(0.1)  # Initial learning assumption
    
    def get_current_learning_rate(self) -> float:
        """Get current rate of learning (how fast we're improving)."""
        if len(self.learning_events) < 3:
            return 1.0  # High learning rate for new brains
        
        # Calculate recent learning trend
        recent_learning = list(self.learning_events)[-10:]
        if len(recent_learning) < 3:
            return np.mean(recent_learning)
        
        # Exponential weighted average favoring recent learning
        weights = np.exp(np.linspace(-1, 0, len(recent_learning)))
        weighted_learning = np.average(recent_learning, weights=weights)
        
        return max(0.0, weighted_learning)
    
    def get_learning_stability(self) -> float:
        """How stable/predictable is our learning? (emergent maturity indicator)."""
        if len(self.prediction_errors) < 5:
            return 0.0  # Unstable when new
        
        recent_errors = list(self.prediction_errors)[-10:]
        variance = np.var(recent_errors)
        
        # Low variance = stable learning = maturity
        stability = 1.0 / (1.0 + variance * 10)  # Sigmoid-like transformation
        return stability


class SpatialLearningMap:
    """Tracks learning efficiency across spatial locations."""
    
    def __init__(self, decay_rate: float = 0.995):
        self.learning_map = {}  # position -> learning_gradient
        self.visit_counts = {}  # position -> visit_count  
        self.decay_rate = decay_rate
    
    def update_spatial_learning(self, position: Tuple[int, int], prediction_error: float, timestamp: float):
        """Update learning info for a spatial location."""
        pos_key = (round(position[0], 1), round(position[1], 1))
        
        if pos_key not in self.learning_map:
            self.learning_map[pos_key] = LearningGradient()
            self.visit_counts[pos_key] = 0
        
        self.learning_map[pos_key].add_learning_event(prediction_error, timestamp)
        self.visit_counts[pos_key] += 1
        
        # Decay older learning to keep map current
        self._decay_learning_map()
    
    def get_spatial_learning_potential(self, position: Tuple[int, int]) -> float:
        """How much learning potential exists at this position?"""
        pos_key = (round(position[0], 1), round(position[1], 1))
        
        if pos_key not in self.learning_map:
            return 1.0  # High potential for unvisited areas
        
        learning_rate = self.learning_map[pos_key].get_current_learning_rate()
        visit_count = self.visit_counts[pos_key]
        
        # Learning potential decreases with visit frequency but increases with learning rate
        visit_penalty = 1.0 / (1.0 + visit_count * 0.1)
        learning_boost = learning_rate * 2.0
        
        return min(1.0, learning_boost * visit_penalty)
    
    def get_exploration_frontiers(self) -> List[Tuple[int, int]]:
        """Get positions adjacent to known areas with high learning potential."""
        frontiers = []
        
        for known_pos in self.learning_map.keys():
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (known_pos[0] + dx, known_pos[1] + dy)
                    if neighbor not in self.learning_map:
                        frontiers.append(neighbor)
        
        return frontiers
    
    def _decay_learning_map(self):
        """Gradually decay old learning information."""
        for gradient in self.learning_map.values():
            # This would be more sophisticated in a full implementation
            pass


class UnifiedLearningDrive(BaseDrive):
    """
    Unified learning drive that exhibits emergent curiosity and exploration behavior.
    
    The drive naturally seeks:
    1. Areas with high learning potential (prediction error)
    2. Unexplored spatial regions  
    3. Novel action-outcome patterns
    4. Efficient learning (consolidating when appropriate)
    
    All behavior emerges from learning dynamics - no artificial thresholds.
    """
    
    def __init__(self, base_weight: float = 0.6):
        super().__init__("UnifiedLearning", base_weight)
        
        # Core learning measurement systems
        self.global_learning_gradient = LearningGradient(window_size=50)
        self.spatial_learning_map = SpatialLearningMap()
        
        # Adaptive parameters (auto-tune based on robot's capabilities)
        self.learning_sensitivity = 1.0  # How sensitive to learning opportunities
        self.spatial_bias = 0.5  # Balance between prediction-seeking and spatial exploration
        self.consolidation_preference = 0.0  # Emerges based on learning efficiency
        
        # Experience tracking for emergent behavior
        self.recent_positions = deque(maxlen=20)
        self.recent_actions = deque(maxlen=20)
        self.recent_learning_events = deque(maxlen=100)
        
        # No artificial floors or thresholds!
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate action based on learning potential."""
        self.total_evaluations += 1
        
        # Update our understanding of current learning context
        self._update_learning_context(context)
        
        # Calculate multiple aspects of learning potential
        prediction_learning_score = self._calculate_prediction_learning_potential(action, context)
        spatial_learning_score = self._calculate_spatial_learning_potential(action, context)
        pattern_learning_score = self._calculate_pattern_learning_potential(action, context)
        consolidation_efficiency = self._calculate_consolidation_efficiency(context)
        
        # Combine scores with adaptive weighting
        total_score = self._combine_learning_scores(
            prediction_learning_score,
            spatial_learning_score, 
            pattern_learning_score,
            consolidation_efficiency,
            context
        )
        
        # Calculate confidence based on learning stability (emergent maturity)
        learning_stability = self.global_learning_gradient.get_learning_stability()
        confidence = 0.3 + learning_stability * 0.7  # More stable = more confident
        
        # Calculate urgency based on learning rate (fast learning = high urgency)
        current_learning_rate = self.global_learning_gradient.get_current_learning_rate()
        urgency = min(1.0, current_learning_rate * 2.0)
        
        # Generate reasoning
        reasoning = self._generate_learning_reasoning(
            prediction_learning_score, spatial_learning_score, 
            pattern_learning_score, consolidation_efficiency
        )
        
        evaluation = ActionEvaluation(
            drive_name=self.name,
            action_score=total_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency
        )
        
        self.log_activation(total_score, context.step_count)
        return evaluation
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update drive based on current learning dynamics."""
        # Record current learning state
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1]
            self.global_learning_gradient.add_learning_event(recent_error, context.step_count)
            self.spatial_learning_map.update_spatial_learning(
                context.robot_position, recent_error, context.step_count
            )
        
        # Auto-tune parameters based on learning dynamics
        self._auto_tune_parameters(context)
        
        # Calculate drive weight based on current learning potential
        global_learning_rate = self.global_learning_gradient.get_current_learning_rate()
        spatial_learning_potential = np.mean([
            self.spatial_learning_map.get_spatial_learning_potential(pos)
            for pos in [context.robot_position] + list(self.recent_positions)[-5:]
        ])
        
        # Drive weight emerges from learning potential - no artificial floors!
        learning_drive_strength = (global_learning_rate + spatial_learning_potential) / 2.0
        
        # Natural rest emerges when learning potential is low everywhere
        self.current_weight = self.base_weight * learning_drive_strength
        
        return self.current_weight
    
    def _update_learning_context(self, context: DriveContext):
        """Update our understanding of the learning context."""
        self.recent_positions.append(context.robot_position)
        
        # Track action patterns
        if hasattr(context, 'recent_action'):
            self.recent_actions.append(context.recent_action)
    
    def _calculate_prediction_learning_potential(self, action: Dict[str, float], context: DriveContext) -> float:
        """How much could we learn about predictions from this action?"""
        if not context.prediction_errors:
            return 0.8  # High potential when we have no data
        
        recent_error = context.prediction_errors[-1] if context.prediction_errors else 0.5
        
        # Areas with higher prediction error have higher learning potential
        prediction_potential = min(1.0, recent_error * 2.0)
        
        # Bonus for actions that might lead to different outcomes
        action_magnitude = abs(action.get('forward_motor', 0)) + abs(action.get('turn_motor', 0))
        action_novelty_bonus = min(0.3, action_magnitude * 0.5)
        
        return min(1.0, prediction_potential + action_novelty_bonus)
    
    def _calculate_spatial_learning_potential(self, action: Dict[str, float], context: DriveContext) -> float:
        """How much could we learn about space from this action?"""
        current_pos = context.robot_position
        
        # Predict where this action might take us
        predicted_positions = self._predict_action_destinations(action, current_pos, context.robot_orientation)
        
        spatial_potentials = []
        for pos in predicted_positions:
            spatial_potential = self.spatial_learning_map.get_spatial_learning_potential(pos)
            spatial_potentials.append(spatial_potential)
        
        # Also check if this moves toward learning frontiers
        frontiers = self.spatial_learning_map.get_exploration_frontiers()
        frontier_bonus = 0.0
        
        if frontiers:
            min_frontier_distance = min([
                abs(pos[0] - frontier[0]) + abs(pos[1] - frontier[1])
                for pos in predicted_positions
                for frontier in frontiers[:10]  # Check closest frontiers
            ])
            frontier_bonus = max(0, 0.5 - min_frontier_distance * 0.1)
        
        avg_spatial_potential = np.mean(spatial_potentials) if spatial_potentials else 0.5
        return min(1.0, avg_spatial_potential + frontier_bonus)
    
    def _calculate_pattern_learning_potential(self, action: Dict[str, float], context: DriveContext) -> float:
        """How much could we learn about action-outcome patterns?"""
        # Simple pattern learning: actions different from recent patterns have higher potential
        if len(self.recent_actions) < 3:
            return 0.6  # Moderate potential when we have little data
        
        recent_actions = list(self.recent_actions)[-5:]
        
        # Calculate how different this action is from recent patterns
        action_signature = (
            action.get('forward_motor', 0),
            action.get('turn_motor', 0), 
            action.get('brake_motor', 0)
        )
        
        pattern_novelty = 0.5  # Default
        
        if recent_actions:
            # Calculate average difference from recent actions
            differences = []
            for recent_action in recent_actions:
                if isinstance(recent_action, dict):
                    recent_signature = (
                        recent_action.get('forward_motor', 0),
                        recent_action.get('turn_motor', 0),
                        recent_action.get('brake_motor', 0)
                    )
                    diff = sum(abs(a - r) for a, r in zip(action_signature, recent_signature))
                    differences.append(diff)
            
            if differences:
                avg_difference = np.mean(differences)
                pattern_novelty = min(1.0, avg_difference * 2.0)  # Scale to 0-1
        
        return pattern_novelty
    
    def _calculate_consolidation_efficiency(self, context: DriveContext) -> float:
        """How efficiently are we consolidating vs. creating new memories?"""
        # This would integrate with the memory system to measure consolidation efficiency
        # For now, estimate based on learning stability
        stability = self.global_learning_gradient.get_learning_stability()
        
        # High stability suggests efficient consolidation
        return stability
    
    def _combine_learning_scores(self, prediction_score: float, spatial_score: float, 
                                pattern_score: float, consolidation_score: float,
                                context: DriveContext) -> float:
        """Adaptively combine different learning scores."""
        # Auto-tune the combination weights based on current learning dynamics
        current_learning_rate = self.global_learning_gradient.get_current_learning_rate()
        
        # When learning rate is high, favor exploration (prediction + spatial)
        # When learning rate is low, favor consolidation and pattern learning
        exploration_weight = current_learning_rate
        consolidation_weight = 1.0 - current_learning_rate
        
        exploration_component = (prediction_score + spatial_score) / 2.0
        consolidation_component = (pattern_score + consolidation_score) / 2.0
        
        combined_score = (
            exploration_component * exploration_weight +
            consolidation_component * consolidation_weight
        )
        
        return combined_score
    
    def _auto_tune_parameters(self, context: DriveContext):
        """Auto-tune parameters based on robot's learning patterns."""
        learning_rate = self.global_learning_gradient.get_current_learning_rate()
        learning_stability = self.global_learning_gradient.get_learning_stability()
        
        # Adjust learning sensitivity based on how much we're actually learning
        if learning_rate > 0.5:
            self.learning_sensitivity = min(2.0, self.learning_sensitivity * 1.01)
        else:
            self.learning_sensitivity = max(0.5, self.learning_sensitivity * 0.99)
        
        # Adjust spatial bias based on learning stability
        # More stable learning = can focus more on spatial exploration
        self.spatial_bias = 0.3 + learning_stability * 0.4
        
        # Consolidation preference emerges from learning efficiency
        self.consolidation_preference = learning_stability
    
    def _predict_action_destinations(self, action: Dict[str, float], 
                                   current_pos: Tuple[int, int], 
                                   orientation: int) -> List[Tuple[int, int]]:
        """Predict where this action might take the robot."""
        # Simple prediction - could be made more sophisticated
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        
        predicted_positions = []
        x, y = current_pos
        
        # Handle turning
        new_orientation = orientation
        if abs(turn_motor) > 0.3:
            new_orientation = (orientation + (1 if turn_motor > 0 else -1)) % 4
        
        # Handle movement
        if abs(forward_motor) > 0.2:
            direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
            dx, dy = direction_vectors[new_orientation]
            
            if forward_motor > 0:
                predicted_positions.append((x + dx, y + dy))
            else:
                predicted_positions.append((x - dx, y - dy))
        else:
            predicted_positions.append(current_pos)
        
        return predicted_positions
    
    def _generate_learning_reasoning(self, prediction_score: float, spatial_score: float,
                                   pattern_score: float, consolidation_score: float) -> str:
        """Generate human-readable reasoning for the learning evaluation."""
        scores = {
            'prediction': prediction_score,
            'spatial': spatial_score, 
            'pattern': pattern_score,
            'consolidation': consolidation_score
        }
        
        dominant_aspect = max(scores.keys(), key=lambda k: scores[k])
        dominant_score = scores[dominant_aspect]
        
        learning_rate = self.global_learning_gradient.get_current_learning_rate()
        
        if dominant_score > 0.7:
            if dominant_aspect == 'prediction':
                return f"High prediction learning potential ({dominant_score:.2f})"
            elif dominant_aspect == 'spatial':
                return f"Good spatial exploration opportunity ({dominant_score:.2f})"
            elif dominant_aspect == 'pattern':
                return f"Novel action pattern to explore ({dominant_score:.2f})"
            else:
                return f"Efficient memory consolidation opportunity ({dominant_score:.2f})"
        elif learning_rate < 0.1:
            return f"Low learning potential - approaching mastery"
        else:
            return f"Moderate learning opportunity"
    
    def get_learning_statistics(self) -> Dict:
        """Get detailed statistics about learning progress."""
        learning_rate = self.global_learning_gradient.get_current_learning_rate()
        learning_stability = self.global_learning_gradient.get_learning_stability()
        
        spatial_coverage = len(self.spatial_learning_map.learning_map)
        avg_spatial_potential = np.mean([
            self.spatial_learning_map.get_spatial_learning_potential(pos)
            for pos in self.spatial_learning_map.learning_map.keys()
        ]) if self.spatial_learning_map.learning_map else 1.0
        
        return {
            'current_learning_rate': learning_rate,
            'learning_stability': learning_stability,
            'emergent_maturity': learning_stability,  # Maturity = stability
            'spatial_coverage': spatial_coverage,
            'avg_spatial_learning_potential': avg_spatial_potential,
            'learning_sensitivity': self.learning_sensitivity,
            'spatial_bias': self.spatial_bias,
            'consolidation_preference': self.consolidation_preference,
            'frontiers_available': len(self.spatial_learning_map.get_exploration_frontiers()),
            'drive_weight': self.current_weight
        }
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """Evaluate the pain/pleasure of a learning experience."""
        # Learning events are generally pleasurable
        # Lack of learning is mildly uncomfortable  
        
        if hasattr(experience, 'prediction_error'):
            error = experience.prediction_error
            
            # High error followed by learning = pleasure
            # High error with no learning = mild discomfort
            # Low error (mastery) = mild pleasure
            
            if error > 0.5:
                # High error - check if we're learning from it
                learning_rate = self.global_learning_gradient.get_current_learning_rate()
                if learning_rate > 0.3:
                    return 0.4  # Learning from mistakes = pleasure
                else:
                    return -0.2  # Not learning from mistakes = mild discomfort
            elif error < 0.2:
                return 0.3  # Mastery = mild pleasure
            else:
                return 0.1  # Normal learning = slight pleasure
        
        return 0.0  # Neutral
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Learning drive's contribution to robot mood."""
        learning_rate = self.global_learning_gradient.get_current_learning_rate()
        learning_stability = self.global_learning_gradient.get_learning_stability()
        
        # Satisfaction based on learning progress
        if learning_rate > 0.5:
            satisfaction = 0.4  # Actively learning = good
        elif learning_rate < 0.1 and learning_stability > 0.7:
            satisfaction = 0.6  # Mastered = very good
        elif learning_rate < 0.1 and learning_stability < 0.3:
            satisfaction = -0.3  # Confused/stuck = bad
        else:
            satisfaction = 0.0  # Normal state
        
        # Urgency based on learning opportunities
        urgency = min(0.8, learning_rate * 1.5)
        
        # Confidence based on stability
        confidence = learning_stability
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency, 
            'confidence': confidence
        }