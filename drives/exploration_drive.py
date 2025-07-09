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
    
    def __init__(self, base_weight: float = 0.3, world_width: int = 40, world_height: int = 40):
        super().__init__("Exploration", base_weight)
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.position_visit_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.exploration_radius = 0  # How far we've explored from start
        self.backtracking_penalty = 0.0
        self.novelty_seeking_boost = 0.0
        
        # World bounds awareness for complete mapping
        self.world_width = world_width
        self.world_height = world_height
        self.world_bounds = (0, 0, world_width - 1, world_height - 1)  # (min_x, min_y, max_x, max_y)
        self.total_world_positions = world_width * world_height
        self.exploration_completion_threshold = 0.85  # 85% coverage considered "complete"
        
        # Frontier detection for systematic exploration
        self.frontier_positions: Set[Tuple[int, int]] = set()
        self.frontier_update_counter = 0
        
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
        
        # Natural drive weight based on exploration potential (can reach zero)
        total_unique_positions = len(self.visited_positions)
        exploration_completion = self.get_exploration_completion_ratio()
        
        # Calculate exploration potential (higher = more to explore)
        if exploration_completion >= self.exploration_completion_threshold:
            # Exploration is complete - drive naturally goes to zero
            exploration_potential = 0.0
        elif exploration_completion >= 0.8:
            # Very close to completion - minimal exploration drive
            exploration_potential = 0.1
        else:
            # Calculate based on frontiers and novelty
            frontier_count = len(self.frontier_positions)
            max_possible_frontiers = min(100, self.total_world_positions * 0.1)  # Rough estimate
            frontier_ratio = frontier_count / max(1, max_possible_frontiers)
            
            # Early exploration gets higher potential
            if total_unique_positions < 10:
                base_potential = 1.0
            else:
                # NO artificial floor - exploration potential based purely on frontiers
                base_potential = frontier_ratio  # Can reach zero when no frontiers
            
            # Backtracking penalty increases exploration potential
            if self.backtracking_penalty > 0.5:
                stagnation_boost = 1.0 + self.backtracking_penalty
            else:
                stagnation_boost = 1.0
            
            exploration_potential = min(1.0, base_potential * stagnation_boost)
        
        # Natural weight - no artificial minimums, can be zero
        self.current_weight = self.base_weight * exploration_potential
        return self.current_weight
    
    def _update_position_tracking(self, position: Tuple[int, int]):
        """Update tracking of visited positions."""
        self.visited_positions.add(position)
        
        # Update frontier detection periodically
        self.frontier_update_counter += 1
        if self.frontier_update_counter % 5 == 0:  # Update every 5 positions
            self._update_frontier_detection()
    
    def _calculate_novelty_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how likely this action leads to new/unexplored areas."""
        current_pos = context.robot_position
        orientation = context.robot_orientation
        
        # Predict likely destination based on action
        predicted_positions = self._predict_action_destinations(action, current_pos, orientation)
        
        novelty_scores = []
        for pos in predicted_positions:
            visit_count = self.position_visit_counts.get(pos, 0)
            
            # Base novelty score
            if visit_count == 0:
                # Never been here - highest novelty
                base_score = 1.0
            elif visit_count == 1:
                # Been here once - high novelty
                base_score = 0.8
            elif visit_count <= 3:
                # Somewhat familiar - medium novelty
                base_score = 0.5
            else:
                # Very familiar - low novelty
                base_score = 0.1
            
            # Frontier bonus - positions adjacent to explored areas get extra novelty
            frontier_bonus = 0.0
            if pos in self.frontier_positions:
                frontier_bonus = 0.3  # Significant bonus for frontier positions
            
            # Combine base score with frontier bonus
            total_score = min(1.0, base_score + frontier_bonus)
            novelty_scores.append(total_score)
        
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
        self.frontier_positions.clear()
        self.frontier_update_counter = 0
        self.reset_drive()
    
    def _update_frontier_detection(self):
        """Update frontier positions - unvisited areas adjacent to visited areas."""
        self.frontier_positions.clear()
        
        # For each visited position, check its neighbors
        for x, y in self.visited_positions:
            # Check 8-directional neighbors (including diagonals)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the center position
                    
                    neighbor_x, neighbor_y = x + dx, y + dy
                    
                    # Check if neighbor is within world bounds
                    if (0 <= neighbor_x < self.world_width and 
                        0 <= neighbor_y < self.world_height):
                        
                        neighbor_pos = (neighbor_x, neighbor_y)
                        
                        # If neighbor is unvisited, it's a frontier
                        if neighbor_pos not in self.visited_positions:
                            self.frontier_positions.add(neighbor_pos)
    
    def get_exploration_completion_ratio(self) -> float:
        """Get the ratio of world explored (0.0 to 1.0)."""
        return len(self.visited_positions) / self.total_world_positions
    
    def is_exploration_complete(self) -> bool:
        """Check if exploration is sufficiently complete."""
        return self.get_exploration_completion_ratio() >= self.exploration_completion_threshold
    
    def get_nearest_frontier(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get the nearest frontier position to current position."""
        if not self.frontier_positions:
            return None
        
        min_distance = float('inf')
        nearest_frontier = None
        
        for frontier_pos in self.frontier_positions:
            distance = math.sqrt((current_pos[0] - frontier_pos[0])**2 + 
                               (current_pos[1] - frontier_pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_frontier = frontier_pos
        
        return nearest_frontier
    
    def get_exploration_status(self) -> Dict[str, any]:
        """Get comprehensive exploration status for debugging."""
        completion_ratio = self.get_exploration_completion_ratio()
        return {
            'visited_positions': len(self.visited_positions),
            'total_world_positions': self.total_world_positions,
            'completion_ratio': completion_ratio,
            'completion_percentage': completion_ratio * 100,
            'frontier_positions': len(self.frontier_positions),
            'is_complete': self.is_exploration_complete(),
            'exploration_radius': self.exploration_radius,
            'backtracking_penalty': self.backtracking_penalty
        }