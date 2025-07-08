"""
Predictive Survival Drive - Survival drive enhanced with sensory prediction.

This transforms survival from reactive to predictive:
- Before: "This action seems safe"  
- After: "This action will lead to safety"

The drive can now avoid dangers before experiencing them by imagining
the consequences of actions.
"""

from typing import Dict, List, Any
import random

from .predictive_base_drive import PredictiveBaseDrive, DriveContext
from prediction.sensory_predictor import SensoryPrediction


class PredictiveSurvivalDrive(PredictiveBaseDrive):
    """
    Survival drive that evaluates actions based on their predicted safety consequences.
    
    This allows the robot to avoid walls, damages, and dangers before hitting them
    by imagining what will happen if it takes each action.
    """
    
    def __init__(self, base_weight: float = 1.0, predictor=None):
        super().__init__("PredictiveSurvival", base_weight, predictor)
        
        # Survival thresholds and preferences
        self.danger_threshold = 0.7      # Sensor level considered dangerous
        self.wall_threshold = 0.6        # Wall sensor level to avoid
        self.health_concern_threshold = 0.3  # Health level that triggers concern
        
        # Learning parameters
        self.learned_danger_patterns = {}  # Learned what sensor patterns mean danger
        self.safety_experiences = []       # Memory of safe outcomes
        
    def evaluate_immediate_action(self, action: Dict[str, float], context: DriveContext) -> float:
        """
        Evaluate action based on immediate properties (fallback when no prediction).
        
        This is the original survival logic - conservative and based on current state.
        """
        score = 0.5  # Neutral starting point
        
        # Current health considerations
        if context.robot_health < self.health_concern_threshold:
            # Low health makes us very conservative
            score -= 0.3
            
            # Prefer gentle actions when health is low
            action_intensity = self._calculate_action_intensity(action)
            if action_intensity < 0.3:
                score += 0.2  # Gentle actions are good when hurt
        
        # Current threat level
        if context.threat_level == 'high':
            score -= 0.4
            # Prefer evasive actions in high threat
            if action.get('brake_motor', 0) > 0.5:
                score += 0.3  # Braking is good in danger
        elif context.threat_level == 'low':
            score += 0.1  # Slight bonus for safe environment
        
        # Avoid obviously risky actions
        if action.get('forward_motor', 0) > 0.7:
            score -= 0.2  # Fast forward movement is risky
        
        # Recent damage makes us cautious
        if context.time_since_last_damage < 5:
            score -= 0.2
            if action.get('forward_motor', 0) > 0.4:
                score -= 0.1  # Recent damage makes forward movement scarier
        
        return max(0.0, min(1.0, score))
    
    def evaluate_predicted_consequences(self, prediction: SensoryPrediction, context: DriveContext) -> float:
        """
        Evaluate action based on its predicted sensory consequences.
        
        This is the key enhancement: we can now avoid dangers before experiencing them
        by imagining what the world will look like after taking the action.
        """
        score = 0.5  # Neutral starting point
        
        # Analyze predicted sensory outcomes for survival threats
        predicted_sensors = prediction.predicted_sensors
        
        # Wall/obstacle avoidance based on prediction
        wall_danger_score = self._evaluate_predicted_walls(predicted_sensors)
        score += wall_danger_score
        
        # Damage/danger avoidance based on prediction  
        damage_danger_score = self._evaluate_predicted_damage(predicted_sensors)
        score += damage_danger_score
        
        # Environmental safety based on prediction
        environment_score = self._evaluate_predicted_environment(predicted_sensors)
        score += environment_score
        
        # Stability and control based on prediction
        stability_score = self._evaluate_predicted_stability(predicted_sensors)
        score += stability_score
        
        # Bonus for high-confidence safe predictions
        if prediction.confidence > 0.7 and score > 0.6:
            score += 0.1  # Reward confident predictions of safety
        
        # Penalty for high-confidence dangerous predictions
        if prediction.confidence > 0.7 and score < 0.4:
            score -= 0.2  # Strongly avoid confident predictions of danger
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_predicted_walls(self, predicted_sensors: Dict[str, float]) -> float:
        """Evaluate wall/obstacle dangers in predicted sensory state."""
        wall_score = 0.0
        
        # Look for wall-related sensors in prediction
        for sensor_name, value in predicted_sensors.items():
            if 'wall' in sensor_name.lower() or 'distance' in sensor_name.lower() or sensor_name == '0':
                # High wall sensor values are dangerous
                if value > self.wall_threshold:
                    danger_level = (value - self.wall_threshold) / (1.0 - self.wall_threshold)
                    wall_score -= danger_level * 0.4  # Strong penalty for predicted wall collision
                elif value < 0.3:
                    wall_score += 0.1  # Bonus for clear path ahead
        
        return wall_score
    
    def _evaluate_predicted_damage(self, predicted_sensors: Dict[str, float]) -> float:
        """Evaluate damage/harm in predicted sensory state."""
        damage_score = 0.0
        
        # Look for damage-related sensors
        for sensor_name, value in predicted_sensors.items():
            if 'damage' in sensor_name.lower() or 'pain' in sensor_name.lower() or 'harm' in sensor_name.lower():
                if value > 0.1:
                    # Any predicted damage is very bad
                    damage_score -= value * 0.6
        
        return damage_score
    
    def _evaluate_predicted_environment(self, predicted_sensors: Dict[str, float]) -> float:
        """Evaluate overall environmental safety in predicted state."""
        env_score = 0.0
        
        # Look for environmental indicators
        for sensor_name, value in predicted_sensors.items():
            if 'food' in sensor_name.lower() or sensor_name == '1':
                # Food proximity is generally safe
                if value > 0.5:
                    env_score += 0.1
            
            elif 'danger' in sensor_name.lower() or 'threat' in sensor_name.lower():
                # Danger sensors are bad
                if value > 0.3:
                    env_score -= value * 0.3
        
        return env_score
    
    def _evaluate_predicted_stability(self, predicted_sensors: Dict[str, float]) -> float:
        """Evaluate predicted stability and control."""
        stability_score = 0.0
        
        # Look for motion/stability related sensors
        for sensor_name, value in predicted_sensors.items():
            if 'speed' in sensor_name.lower() or 'velocity' in sensor_name.lower():
                # Moderate speeds are good, extreme speeds are dangerous
                if 0.2 <= value <= 0.7:
                    stability_score += 0.05
                elif value > 0.8:
                    stability_score -= 0.1  # Too fast is dangerous
            
            elif 'balance' in sensor_name.lower() or 'tilt' in sensor_name.lower():
                # Good balance is safe
                if value < 0.2:  # Low tilt is good
                    stability_score += 0.05
                elif value > 0.7:  # High tilt is dangerous
                    stability_score -= 0.2
        
        return stability_score
    
    def _calculate_action_intensity(self, action: Dict[str, float]) -> float:
        """Calculate how intense/risky an action is."""
        total_intensity = 0.0
        for motor, value in action.items():
            total_intensity += abs(value)
        
        return total_intensity / len(action) if action else 0.0
    
    def _calculate_urgency(self, context: DriveContext) -> float:
        """Calculate survival urgency based on current state."""
        urgency = 0.3  # Base urgency
        
        # Health urgency
        if context.robot_health < 0.3:
            urgency += 0.4  # Low health is urgent
        elif context.robot_health < 0.6:
            urgency += 0.2  # Moderate health concern
        
        # Threat urgency
        if context.threat_level == 'high':
            urgency += 0.3
        elif context.threat_level == 'medium':
            urgency += 0.1
        
        # Recent damage urgency
        if context.time_since_last_damage < 3:
            urgency += 0.2  # Recent damage increases urgency
        
        return min(1.0, urgency)
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Calculate mood contribution for survival drive."""
        # Survival contributes to overall anxiety/safety feelings
        health_satisfaction = context.robot_health  # Higher health = higher satisfaction
        
        threat_dissatisfaction = 0.0
        if context.threat_level == 'high':
            threat_dissatisfaction = 0.6
        elif context.threat_level == 'medium':
            threat_dissatisfaction = 0.3
        
        # Recent damage creates anxiety
        damage_anxiety = max(0.0, 1.0 - context.time_since_last_damage / 10.0) * 0.4
        
        overall_satisfaction = health_satisfaction - threat_dissatisfaction - damage_anxiety
        urgency = self._calculate_urgency(context)
        confidence = 0.8 if context.robot_health > 0.7 else 0.4
        
        return {
            'satisfaction': overall_satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def update_drive_state(self, context: DriveContext):
        """Update survival drive based on current context."""
        super().update_drive_state(context)
        
        # Adapt weight based on danger level
        if context.threat_level == 'high' or context.robot_health < 0.3:
            self.current_weight = self.base_weight * 2.0  # Double weight in danger
        elif context.threat_level == 'low' and context.robot_health > 0.8:
            self.current_weight = self.base_weight * 0.7  # Lower weight when very safe
        else:
            self.current_weight = self.base_weight
        
        # Learn from survival experiences
        self._learn_from_context(context)
    
    def _learn_from_context(self, context: DriveContext):
        """Learn patterns about what leads to danger vs safety."""
        # Simple learning: remember patterns that led to damage
        if context.time_since_last_damage == 0:  # Just took damage
            danger_pattern = {
                'sensory_pattern': context.current_sensory[:],
                'health_level': context.robot_health,
                'threat_level': context.threat_level
            }
            # Store this as a dangerous pattern (simplified learning)
            pattern_key = str(danger_pattern)[:50]  # Simplified key
            self.learned_danger_patterns[pattern_key] = self.learned_danger_patterns.get(pattern_key, 0) + 1
        
        elif context.time_since_last_damage > 20 and context.robot_health > 0.8:
            # Long time without damage and good health = safe pattern
            safe_pattern = {
                'sensory_pattern': context.current_sensory[:],
                'health_level': context.robot_health,
                'threat_level': context.threat_level
            }
            self.safety_experiences.append(safe_pattern)
            
            # Keep only recent safety experiences
            if len(self.safety_experiences) > 50:
                self.safety_experiences = self.safety_experiences[-25:]