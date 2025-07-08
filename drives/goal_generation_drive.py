"""
Goal Generation Drive - Creates temporary objectives based on emergent opportunities.

This drive doesn't compete with other drives directly. Instead, it analyzes
the current brain state and generates focused, temporary objectives that
other drives can work toward. Goals EMERGE from brain state rather than
being externally imposed.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .base_drive import BaseDrive, DriveContext, ActionEvaluation


@dataclass
class TemporaryGoal:
    """Represents a temporary objective for the robot."""
    goal_type: str  # "investigate", "reach", "avoid", "pattern_break"
    target_location: Optional[Tuple[int, int]] = None
    target_pattern: Optional[Dict] = None
    priority: float = 0.5  # 0.0 = low, 1.0 = critical
    steps_remaining: int = 20
    created_by_drive: str = "unknown"
    success_condition: str = "location_reached"
    
    def is_complete(self, current_position: Tuple[int, int], context: DriveContext) -> bool:
        """Check if goal has been achieved."""
        if self.steps_remaining <= 0:
            return True
            
        if self.success_condition == "location_reached" and self.target_location:
            distance = abs(current_position[0] - self.target_location[0]) + \
                      abs(current_position[1] - self.target_location[1])
            return distance <= 1
        
        if self.success_condition == "pattern_broken":
            # Check if robot has broken out of repetitive behavior
            return self._pattern_changed(context)
        
        return False
    
    def _pattern_changed(self, context: DriveContext) -> bool:
        """Check if behavioral pattern has changed significantly."""
        # Simplified: check if recent actions are more varied
        recent_actions = getattr(context, 'recent_actions', [])
        if len(recent_actions) < 5:
            return False
        
        # Calculate action variance in recent steps
        forward_variance = self._calculate_variance([a.get('forward_motor', 0) for a in recent_actions[-5:]])
        turn_variance = self._calculate_variance([a.get('turn_motor', 0) for a in recent_actions[-5:]])
        
        return forward_variance > 0.1 or turn_variance > 0.1
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of action values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class GoalGenerationDrive(BaseDrive):
    """
    Generates temporary objectives based on emergent opportunities and needs.
    
    This meta-drive analyzes the brain state and creates focused goals that
    help direct behavior without hardcoding specific actions.
    """
    
    def __init__(self, base_weight: float = 0.1):
        super().__init__("Goal Generation", base_weight)
        self.active_goals: List[TemporaryGoal] = []
        self.completed_goals: List[TemporaryGoal] = []
        self.goal_history: List[str] = []
        self.last_goal_generation = 0
        self.generation_cooldown = 5  # Steps between goal generation attempts
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate how well action supports current goals."""
        self.total_evaluations += 1
        
        # Update existing goals
        self._update_active_goals(context)
        
        # Generate new goals if needed
        if self._should_generate_goals(context):
            new_goals = self._generate_goals_from_context(context)
            self.active_goals.extend(new_goals)
            self.last_goal_generation = context.step_count
        
        # Score action based on goal alignment
        goal_alignment_score = self._calculate_goal_alignment(action, context)
        
        # Confidence based on goal clarity
        confidence = 0.3 + (len(self.active_goals) * 0.2)
        confidence = min(1.0, confidence)
        
        # Generate reasoning
        reasoning = self._generate_goal_reasoning()
        
        return ActionEvaluation(
            drive_name=self.name,
            action_score=goal_alignment_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=self._calculate_goal_urgency()
        )
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update goal generation based on brain state."""
        # Clean up completed goals
        self._cleanup_completed_goals(context)
        
        # Adjust weight based on goal activity
        if len(self.active_goals) > 0:
            # Active goals - increase influence
            weight_multiplier = 1.5
        elif self._brain_seems_stuck(context):
            # Might need goals to break patterns
            weight_multiplier = 2.0
        else:
            # Normal exploration is working fine
            weight_multiplier = 0.5
        
        self.current_weight = min(0.8, self.base_weight * weight_multiplier)
        return self.current_weight
    
    def _should_generate_goals(self, context: DriveContext) -> bool:
        """Determine if new goals should be generated."""
        # Cooldown check
        if context.step_count - self.last_goal_generation < self.generation_cooldown:
            return False
        
        # Don't overwhelm with too many goals
        if len(self.active_goals) >= 2:
            return False
        
        # Generate goals if opportunities detected
        return (self._curiosity_opportunity_detected(context) or
                self._survival_opportunity_detected(context) or
                self._exploration_opportunity_detected(context) or
                self._pattern_break_needed(context))
    
    def _generate_goals_from_context(self, context: DriveContext) -> List[TemporaryGoal]:
        """Generate goals based on current brain state and opportunities."""
        new_goals = []
        
        # Curiosity-driven goals
        if self._curiosity_opportunity_detected(context):
            goal = self._create_investigation_goal(context)
            if goal:
                new_goals.append(goal)
        
        # Survival-driven goals
        if self._survival_opportunity_detected(context):
            goal = self._create_survival_goal(context)
            if goal:
                new_goals.append(goal)
        
        # Exploration-driven goals
        if self._exploration_opportunity_detected(context):
            goal = self._create_exploration_goal(context)
            if goal:
                new_goals.append(goal)
        
        # Pattern-breaking goals
        if self._pattern_break_needed(context):
            goal = self._create_pattern_break_goal(context)
            if goal:
                new_goals.append(goal)
        
        return new_goals
    
    def _curiosity_opportunity_detected(self, context: DriveContext) -> bool:
        """Check if there's a curiosity-driven opportunity."""
        # High prediction error suggests something interesting
        if context.prediction_errors and len(context.prediction_errors) > 0:
            recent_error = sum(context.prediction_errors[-3:]) / min(3, len(context.prediction_errors))
            return recent_error > 1.0  # Threshold for "interesting"
        return False
    
    def _survival_opportunity_detected(self, context: DriveContext) -> bool:
        """Check if there's a survival-driven opportunity."""
        # Low energy/health but resources might be available
        return (context.robot_energy < 0.4 or context.robot_health < 0.5) and \
               context.threat_level in ['normal', 'alert']
    
    def _exploration_opportunity_detected(self, context: DriveContext) -> bool:
        """Check if there's an exploration opportunity."""
        # Robot seems to be in same area for too long
        recent_positions = getattr(context, 'recent_positions', [])
        if len(recent_positions) >= 10:
            unique_positions = len(set(recent_positions[-10:]))
            return unique_positions < 3  # Very limited movement
        return False
    
    def _pattern_break_needed(self, context: DriveContext) -> bool:
        """Check if robot is stuck in repetitive patterns."""
        recent_actions = getattr(context, 'recent_actions', [])
        if len(recent_actions) < 8:
            return False
        
        # Check for repetitive action patterns
        last_4 = recent_actions[-4:]
        prev_4 = recent_actions[-8:-4]
        
        # Simple pattern detection: are recent actions very similar to previous ones?
        similarity = self._calculate_action_sequence_similarity(last_4, prev_4)
        return similarity > 0.8  # High similarity suggests repetition
    
    def _create_investigation_goal(self, context: DriveContext) -> Optional[TemporaryGoal]:
        """Create a goal to investigate something interesting."""
        # For simplicity, create goal to move toward area of high prediction error
        # In practice, this would be more sophisticated
        current_pos = context.robot_position
        
        # Create goal to move in a different direction to investigate
        target_x = current_pos[0] + (1 if current_pos[0] < 10 else -1)
        target_y = current_pos[1] + (1 if current_pos[1] < 10 else -1)
        
        return TemporaryGoal(
            goal_type="investigate",
            target_location=(target_x, target_y),
            priority=0.7,
            steps_remaining=15,
            created_by_drive="curiosity",
            success_condition="location_reached"
        )
    
    def _create_survival_goal(self, context: DriveContext) -> Optional[TemporaryGoal]:
        """Create a goal to address survival needs."""
        # Simplified: move toward center (safer area) when health/energy low
        return TemporaryGoal(
            goal_type="reach_safety",
            target_location=(10, 10),  # Assume center is safer
            priority=0.9,
            steps_remaining=20,
            created_by_drive="survival",
            success_condition="location_reached"
        )
    
    def _create_exploration_goal(self, context: DriveContext) -> Optional[TemporaryGoal]:
        """Create a goal to explore new areas."""
        current_pos = context.robot_position
        
        # Move toward an unexplored edge
        if current_pos[0] < 10:
            target = (min(20, current_pos[0] + 5), current_pos[1])
        else:
            target = (max(0, current_pos[0] - 5), current_pos[1])
        
        return TemporaryGoal(
            goal_type="explore",
            target_location=target,
            priority=0.6,
            steps_remaining=25,
            created_by_drive="exploration",
            success_condition="location_reached"
        )
    
    def _create_pattern_break_goal(self, context: DriveContext) -> Optional[TemporaryGoal]:
        """Create a goal to break repetitive patterns."""
        return TemporaryGoal(
            goal_type="pattern_break",
            target_location=None,
            priority=0.5,
            steps_remaining=10,
            created_by_drive="pattern_break",
            success_condition="pattern_broken"
        )
    
    def _calculate_goal_alignment(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well action aligns with current goals."""
        if not self.active_goals:
            return 0.5  # Neutral when no goals
        
        total_alignment = 0.0
        total_weight = 0.0
        
        for goal in self.active_goals:
            alignment = self._calculate_single_goal_alignment(action, goal, context)
            weight = goal.priority
            total_alignment += alignment * weight
            total_weight += weight
        
        return total_alignment / total_weight if total_weight > 0 else 0.5
    
    def _calculate_single_goal_alignment(self, action: Dict[str, float], 
                                       goal: TemporaryGoal, context: DriveContext) -> float:
        """Calculate alignment for a specific goal."""
        if goal.goal_type == "reach_safety" and goal.target_location:
            return self._calculate_movement_toward_target(action, goal.target_location, context)
        elif goal.goal_type == "investigate" and goal.target_location:
            return self._calculate_movement_toward_target(action, goal.target_location, context)
        elif goal.goal_type == "explore" and goal.target_location:
            return self._calculate_movement_toward_target(action, goal.target_location, context)
        elif goal.goal_type == "pattern_break":
            return self._calculate_pattern_break_alignment(action, context)
        
        return 0.5
    
    def _calculate_movement_toward_target(self, action: Dict[str, float], 
                                        target: Tuple[int, int], context: DriveContext) -> float:
        """Calculate how well action moves toward target location."""
        # Simplified calculation - in practice would be more sophisticated
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        
        # Encourage forward movement for reaching targets
        if forward_motor > 0.2:
            return 0.7 + min(0.3, forward_motor)
        else:
            return 0.3
    
    def _calculate_pattern_break_alignment(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well action breaks current patterns."""
        recent_actions = getattr(context, 'recent_actions', [])
        if len(recent_actions) < 3:
            return 0.5
        
        # Score higher for actions different from recent ones
        recent_avg_forward = sum(a.get('forward_motor', 0) for a in recent_actions[-3:]) / 3
        recent_avg_turn = sum(a.get('turn_motor', 0) for a in recent_actions[-3:]) / 3
        
        forward_diff = abs(action.get('forward_motor', 0) - recent_avg_forward)
        turn_diff = abs(action.get('turn_motor', 0) - recent_avg_turn)
        
        return min(1.0, (forward_diff + turn_diff) * 2)  # Scale difference
    
    def _calculate_action_sequence_similarity(self, seq1: List[Dict], seq2: List[Dict]) -> float:
        """Calculate similarity between two action sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        similarities = []
        for a1, a2 in zip(seq1, seq2):
            forward_diff = abs(a1.get('forward_motor', 0) - a2.get('forward_motor', 0))
            turn_diff = abs(a1.get('turn_motor', 0) - a2.get('turn_motor', 0))
            action_similarity = 1.0 - min(1.0, (forward_diff + turn_diff) / 2)
            similarities.append(action_similarity)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_goal_urgency(self) -> float:
        """Calculate overall urgency of current goals."""
        if not self.active_goals:
            return 0.0
        
        max_priority = max(goal.priority for goal in self.active_goals)
        return max_priority
    
    def _update_active_goals(self, context: DriveContext):
        """Update goal states and remove completed ones."""
        for goal in self.active_goals[:]:  # Copy list to avoid modification during iteration
            goal.steps_remaining -= 1
            
            if goal.is_complete(context.robot_position, context):
                self.active_goals.remove(goal)
                self.completed_goals.append(goal)
                self.goal_history.append(f"Completed: {goal.goal_type}")
    
    def _cleanup_completed_goals(self, context: DriveContext):
        """Remove old completed goals to prevent memory buildup."""
        if len(self.completed_goals) > 10:
            self.completed_goals = self.completed_goals[-5:]  # Keep only recent ones
        
        if len(self.goal_history) > 20:
            self.goal_history = self.goal_history[-10:]
    
    def _brain_seems_stuck(self, context: DriveContext) -> bool:
        """Check if brain seems stuck in unproductive patterns."""
        return self._pattern_break_needed(context) or self._exploration_opportunity_detected(context)
    
    def _generate_goal_reasoning(self) -> str:
        """Generate human-readable reasoning for current goals."""
        if not self.active_goals:
            return "No active goals - allowing open exploration"
        
        goal_descriptions = []
        for goal in self.active_goals:
            if goal.goal_type == "investigate":
                goal_descriptions.append(f"Investigating anomaly ({goal.steps_remaining} steps)")
            elif goal.goal_type == "reach_safety":
                goal_descriptions.append(f"Moving to safety (priority: {goal.priority:.1f})")
            elif goal.goal_type == "explore":
                goal_descriptions.append(f"Exploring new area ({goal.steps_remaining} steps)")
            elif goal.goal_type == "pattern_break":
                goal_descriptions.append(f"Breaking repetitive behavior")
        
        return f"Active goals: {', '.join(goal_descriptions)}"
    
    def get_goal_statistics(self) -> Dict:
        """Get detailed goal generation statistics."""
        stats = self.get_drive_info()
        stats.update({
            'active_goals': len(self.active_goals),
            'completed_goals': len(self.completed_goals),
            'goal_types_active': [g.goal_type for g in self.active_goals],
            'recent_goal_history': self.goal_history[-5:],
            'average_goal_priority': sum(g.priority for g in self.active_goals) / len(self.active_goals) if self.active_goals else 0.0
        })
        return stats