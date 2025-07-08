"""
Exploration Drive - Motivated by discovering new areas and mapping the environment.
The robot wants to see what's out there and build a comprehensive world model.
"""

import math
from typing import Dict, List, Set, Tuple
from .base_drive import BaseDrive, DriveContext, ActionEvaluation


class ExplorationDrive(BaseDrive):
    """
    Drive to explore new areas and discover the environment.
    
    This drive motivates the robot to:
    - Visit previously unexplored areas
    - Map the environment systematically
    - Avoid repetitive paths and locations
    - Seek variety in experiences and locations
    """
    
    def __init__(self, base_weight: float = 0.2):
        super().__init__("Exploration", base_weight)
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.position_visit_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.exploration_radius = 0  # How far we've explored from start
        self.backtracking_penalty = 0.0
        self.novelty_seeking_boost = 0.0
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate action based on exploration potential."""
        self.total_evaluations += 1
        
        # Update position tracking
        self._update_position_tracking(context.robot_position)
        
        # Novelty score (how new/unexplored the likely destination is)
        novelty_score = self._calculate_novelty_score(action, context)
        
        # Coverage score (how well this contributes to systematic exploration)
        coverage_score = self._calculate_coverage_score(action, context)
        
        # Variety score (avoiding repetitive patterns)
        variety_score = self._calculate_variety_score(action, context)
        
        # Distance exploration score (expanding exploration radius)
        distance_score = self._calculate_distance_exploration_score(action, context)
        
        # Combine scores
        exploration_score = (
            novelty_score * 0.35 +
            coverage_score * 0.25 +
            variety_score * 0.25 +
            distance_score * 0.15
        )
        
        # Apply exploration boosts
        exploration_score = min(1.0, exploration_score + self.novelty_seeking_boost)
        
        # Confidence based on how much we've explored
        total_visited = len(self.visited_positions)
        confidence = min(1.0, total_visited / 20.0)  # More confident as we explore more
        
        # Generate reasoning
        reasoning = self._generate_reasoning(novelty_score, coverage_score, variety_score, distance_score)
        
        # Urgency based on exploration stagnation
        urgency = 0.2 + self.backtracking_penalty * 0.6
        
        evaluation = ActionEvaluation(
            drive_name=self.name,
            action_score=exploration_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency
        )
        
        # Log high activation
        self.log_activation(exploration_score, context.step_count)
        
        return evaluation
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update exploration drive based on recent exploration progress."""
        current_pos = context.robot_position
        
        # Update visit tracking
        self.position_visit_counts[current_pos] = self.position_visit_counts.get(current_pos, 0) + 1
        
        # Track recent positions for backtracking detection
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > 10:
            self.recent_positions = self.recent_positions[-5:]
        
        # Calculate backtracking penalty
        if len(self.recent_positions) >= 5:
            unique_recent = len(set(self.recent_positions[-5:]))
            if unique_recent <= 2:  # Very repetitive
                self.backtracking_penalty = min(1.0, self.backtracking_penalty + 0.1)
            else:
                self.backtracking_penalty = max(0.0, self.backtracking_penalty - 0.05)
        
        # Calculate exploration radius
        if self.visited_positions:
            max_distance = max(abs(pos[0]) + abs(pos[1]) for pos in self.visited_positions)
            self.exploration_radius = max(self.exploration_radius, max_distance)
        
        # Boost novelty seeking if we're in a well-explored area
        current_visit_count = self.position_visit_counts.get(current_pos, 0)
        if current_visit_count > 3:
            self.novelty_seeking_boost = min(0.3, self.novelty_seeking_boost + 0.05)
        else:
            self.novelty_seeking_boost = max(0.0, self.novelty_seeking_boost - 0.02)
        
        # Adjust drive weight based on exploration state
        total_unique_positions = len(self.visited_positions)
        
        if total_unique_positions < 5:
            # Early exploration - boost drive
            weight_multiplier = 1.4
        elif self.backtracking_penalty > 0.5:
            # Stuck in small area - boost drive significantly
            weight_multiplier = 1.8
        elif total_unique_positions > 20:
            # Well explored - reduce drive
            weight_multiplier = 0.7
        else:
            # Normal exploration
            weight_multiplier = 1.0
        
        self.current_weight = min(0.8, self.base_weight * weight_multiplier)
        return self.current_weight
    
    def _update_position_tracking(self, position: Tuple[int, int]):
        """Update tracking of visited positions."""
        self.visited_positions.add(position)
    
    def _calculate_novelty_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how likely this action leads to new/unexplored areas."""
        current_pos = context.robot_position
        orientation = context.robot_orientation
        
        # Predict likely destination based on action
        predicted_positions = self._predict_action_destinations(action, current_pos, orientation)
        
        novelty_scores = []
        for pos in predicted_positions:
            visit_count = self.position_visit_counts.get(pos, 0)
            
            if visit_count == 0:
                # Never been here - highest novelty
                novelty_scores.append(1.0)
            elif visit_count == 1:
                # Been here once - high novelty
                novelty_scores.append(0.8)
            elif visit_count <= 3:
                # Somewhat familiar - medium novelty
                novelty_scores.append(0.5)
            else:
                # Very familiar - low novelty
                novelty_scores.append(0.1)
        
        # Average novelty of predicted destinations
        return sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
    
    def _calculate_coverage_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well this action contributes to systematic coverage."""
        current_pos = context.robot_position
        
        # Encourage movement toward edges of explored area
        if not self.visited_positions:
            return 0.7  # Initial exploration
        
        # Find the centroid of explored area
        if len(self.visited_positions) > 2:
            avg_x = sum(pos[0] for pos in self.visited_positions) / len(self.visited_positions)
            avg_y = sum(pos[1] for pos in self.visited_positions) / len(self.visited_positions)
            centroid = (avg_x, avg_y)
            
            # Distance from centroid (further = better for coverage)
            current_distance = math.sqrt((current_pos[0] - centroid[0])**2 + (current_pos[1] - centroid[1])**2)
            
            # Predict destination
            predicted_positions = self._predict_action_destinations(action, current_pos, context.robot_orientation)
            coverage_scores = []
            
            for pred_pos in predicted_positions:
                pred_distance = math.sqrt((pred_pos[0] - centroid[0])**2 + (pred_pos[1] - centroid[1])**2)
                
                # Bonus for moving away from center
                if pred_distance > current_distance:
                    coverage_scores.append(0.8)
                elif pred_distance == current_distance:
                    coverage_scores.append(0.5)
                else:
                    coverage_scores.append(0.2)
            
            return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.5
        
        return 0.5
    
    def _calculate_variety_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how much this action adds variety to recent behavior."""
        # Encourage different types of movement
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        brake_motor = action.get('brake_motor', 0.0)
        
        # Classify action type
        if abs(turn_motor) > 0.3:
            action_type = "turn"
        elif forward_motor > 0.2:
            action_type = "forward"
        elif forward_motor < -0.2:
            action_type = "backward"
        elif brake_motor > 0.3:
            action_type = "brake"
        else:
            action_type = "idle"
        
        # Simple variety: avoid repeating the same action type too much
        # (In a real implementation, we'd track recent action history)
        variety_score = 0.6  # Baseline variety
        
        # Bonus for moderate, varied movement
        movement_variety = abs(forward_motor) + abs(turn_motor)
        if 0.3 <= movement_variety <= 0.8:
            variety_score += 0.3
        
        return min(1.0, variety_score)
    
    def _calculate_distance_exploration_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate score for expanding the exploration radius."""
        current_pos = context.robot_position
        current_distance_from_origin = abs(current_pos[0]) + abs(current_pos[1])
        
        # Predict destinations
        predicted_positions = self._predict_action_destinations(action, current_pos, context.robot_orientation)
        
        distance_scores = []
        for pred_pos in predicted_positions:
            pred_distance = abs(pred_pos[0]) + abs(pred_pos[1])
            
            if pred_distance > self.exploration_radius:
                # Expanding exploration frontier
                distance_scores.append(1.0)
            elif pred_distance > current_distance_from_origin:
                # Moving outward
                distance_scores.append(0.7)
            elif pred_distance == current_distance_from_origin:
                # Maintaining distance
                distance_scores.append(0.4)
            else:
                # Moving inward
                distance_scores.append(0.2)
        
        return sum(distance_scores) / len(distance_scores) if distance_scores else 0.4
    
    def _predict_action_destinations(self, action: Dict[str, float], 
                                   current_pos: Tuple[int, int], 
                                   orientation: int) -> List[Tuple[int, int]]:
        """Predict where the robot might end up after executing this action."""
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        
        predicted_positions = []
        x, y = current_pos
        
        # Handle turning
        new_orientation = orientation
        if abs(turn_motor) > 0.3:
            if turn_motor > 0:
                new_orientation = (orientation + 1) % 4
            else:
                new_orientation = (orientation - 1) % 4
        
        # Handle movement
        if abs(forward_motor) > 0.2:
            # Direction vectors: [North, East, South, West]
            direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            dx, dy = direction_vectors[new_orientation]
            
            if forward_motor > 0:
                # Forward
                predicted_positions.append((x + dx, y + dy))
            else:
                # Backward
                predicted_positions.append((x - dx, y - dy))
        else:
            # No movement - stay in place
            predicted_positions.append(current_pos)
        
        return predicted_positions
    
    def _generate_reasoning(self, novelty: float, coverage: float, variety: float, distance: float) -> str:
        """Generate human-readable reasoning for exploration evaluation."""
        primary_factor = max(novelty, coverage, variety, distance)
        
        if primary_factor == novelty and novelty > 0.7:
            return f"High novelty opportunity ({novelty:.2f}) - exploring unvisited area"
        elif primary_factor == coverage and coverage > 0.7:
            return f"Good coverage expansion ({coverage:.2f}) - systematic exploration"
        elif primary_factor == distance and distance > 0.7:
            return f"Frontier expansion ({distance:.2f}) - pushing exploration boundaries"
        elif primary_factor == variety and variety > 0.7:
            return f"Behavioral variety ({variety:.2f}) - diverse movement patterns"
        elif self.backtracking_penalty > 0.5:
            return f"Breaking repetitive pattern (penalty: {self.backtracking_penalty:.2f})"
        elif primary_factor < 0.3:
            return "Low exploration value - familiar area"
        else:
            return f"Moderate exploration opportunity"
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """
        Exploration drive's pain/pleasure evaluation.
        
        Stagnation (staying in same place) = PAIN
        Movement and new discoveries = PLEASURE
        """
        # Calculate movement from action taken
        action = experience.action_taken
        forward_movement = abs(action.get('forward_motor', 0.0))
        turn_movement = abs(action.get('turn_motor', 0.0))
        total_movement = forward_movement + turn_movement
        
        # Movement pain/pleasure
        movement_score = 0.0
        
        # PAIN: Low movement (stagnation)
        if total_movement < 0.1:  # Very little movement
            movement_score = -0.8  # Strong pain from being stationary
        elif total_movement < 0.3:  # Moderate movement
            movement_score = -0.3  # Mild discomfort from low activity
        else:  # Good movement
            movement_score = min(0.5, total_movement)  # Pleasure from movement
        
        # PAIN: Repetitive patterns (if we could detect them)
        # For now, penalize very predictable actions
        if (abs(forward_movement - 0.3) < 0.05 and turn_movement < 0.05):
            # Repetitive forward movement pattern
            movement_score -= 0.4
        
        # PLEASURE: Novel directions and actions
        if turn_movement > 0.3:  # Good turning = exploration
            movement_score += 0.3
            
        if forward_movement > 0.5:  # Adventurous movement
            movement_score += 0.2
        
        return max(-1.0, min(1.0, movement_score))
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Exploration drive's contribution to robot mood."""
        # Satisfaction based on recent exploration activity
        satisfaction = 0.0
        
        # Check if we're moving enough
        if len(self.recent_positions) >= 3:
            # Calculate movement variety in recent positions
            unique_recent = len(set(self.recent_positions[-5:]))
            if unique_recent >= 4:
                satisfaction = 0.5  # Good exploration
            elif unique_recent >= 2:
                satisfaction = 0.0  # Some movement
            else:
                satisfaction = -0.6  # Stagnation pain
        
        # Urgency based on how much backtracking we're doing
        urgency = self.backtracking_penalty
        
        # Confidence based on exploration progress
        total_visited = len(self.visited_positions)
        confidence = min(0.5, total_visited / 50.0)  # More exploration = more confidence
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def get_exploration_stats(self) -> Dict:
        """Get detailed exploration drive statistics."""
        stats = self.get_drive_info()
        stats.update({
            'unique_positions_visited': len(self.visited_positions),
            'exploration_radius': self.exploration_radius,
            'backtracking_penalty': self.backtracking_penalty,
            'novelty_seeking_boost': self.novelty_seeking_boost,
            'most_visited_position': max(self.position_visit_counts.items(), key=lambda x: x[1]) if self.position_visit_counts else None,
            'total_position_visits': sum(self.position_visit_counts.values())
        })
        return stats
    
    def reset_exploration(self):
        """Reset exploration tracking (useful for new environments)."""
        self.visited_positions.clear()
        self.position_visit_counts.clear()
        self.recent_positions.clear()
        self.exploration_radius = 0
        self.backtracking_penalty = 0.0
        self.novelty_seeking_boost = 0.0
        self.reset_drive()