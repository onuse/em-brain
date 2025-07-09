"""
Curiosity Drive - Motivated by prediction uncertainty and learning opportunities.
The robot wants to understand and predict its world better.
"""

import math
from typing import Dict, List
from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from collections import defaultdict


class CuriosityDrive(BaseDrive):
    """
    Drive to reduce prediction uncertainty and learn about the world.
    
    This drive motivates the robot to:
    - Seek out situations with high prediction error
    - Try actions that might lead to learning
    - Avoid repetitive, predictable situations
    """
    
    def __init__(self, base_weight: float = 0.35):
        super().__init__("Curiosity", base_weight)
        self.recent_prediction_errors = []
        self.prediction_improvement_rate = 0.0
        self.learning_stagnation_penalty = 0.0
        
        # Pure novelty-seeking: track experience familiarity
        self.experience_familiarity = defaultdict(float)  # Track what's been explored
        self.sensor_pattern_familiarity = defaultdict(float)  # Track sensor patterns
        self.motor_pattern_familiarity = defaultdict(float)  # Track motor patterns
        self.familiarity_decay_rate = 0.01  # Gradual forgetting to allow re-exploration
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate action based on pure novelty-seeking and learning potential."""
        self.total_evaluations += 1
        
        # Pure novelty score - predict what this action will teach us
        novelty_score = self._calculate_pure_novelty_score(action, context)
        
        # Self-discovery score - how much will this teach us about our capabilities
        self_discovery_score = self._calculate_self_discovery_score(action, context)
        
        # Environmental novelty score - how much will this teach us about the world
        environment_novelty_score = self._calculate_environment_novelty_score(action, context)
        
        # Sensor exploration score - how much will this teach us about our sensors
        sensor_exploration_score = self._calculate_sensor_exploration_score(action, context)
        
        # Take maximum available novelty (natural progression from self → world → social)
        curiosity_score = max(
            novelty_score,
            self_discovery_score, 
            environment_novelty_score,
            sensor_exploration_score
        )
        
        # Confidence based on how much data we have
        confidence = min(1.0, len(context.prediction_errors) / 10.0)
        
        # Generate reasoning
        reasoning = self._generate_novelty_reasoning(novelty_score, self_discovery_score, 
                                                   environment_novelty_score, sensor_exploration_score)
        
        # Urgency based on learning stagnation
        urgency = 0.3 + self.learning_stagnation_penalty * 0.7
        
        evaluation = ActionEvaluation(
            drive_name=self.name,
            action_score=curiosity_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency
        )
        
        # Log high activation
        self.log_activation(curiosity_score, context.step_count)
        
        return evaluation
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update curiosity drive based on recent learning progress."""
        # Track recent prediction errors
        if context.prediction_errors:
            self.recent_prediction_errors.extend(context.prediction_errors[-3:])
            if len(self.recent_prediction_errors) > 20:
                self.recent_prediction_errors = self.recent_prediction_errors[-10:]
        
        # Calculate learning progress
        if len(self.recent_prediction_errors) >= 5:
            recent_avg = sum(self.recent_prediction_errors[-5:]) / 5
            older_avg = sum(self.recent_prediction_errors[-10:-5]) / 5 if len(self.recent_prediction_errors) >= 10 else recent_avg
            self.prediction_improvement_rate = older_avg - recent_avg  # Positive = improving
        
        # Detect learning stagnation
        if len(self.recent_prediction_errors) >= 10:
            recent_variance = self._calculate_variance(self.recent_prediction_errors[-10:])
            if recent_variance < 0.01 and sum(self.recent_prediction_errors[-5:]) / 5 > 0.3:
                self.learning_stagnation_penalty = min(1.0, self.learning_stagnation_penalty + 0.1)
            else:
                self.learning_stagnation_penalty = max(0.0, self.learning_stagnation_penalty - 0.05)
        
        # Fluid curiosity strength based on learning dynamics
        curiosity_strength = self._calculate_fluid_curiosity_strength(context)
        
        self.current_weight = curiosity_strength
        return self.current_weight
    
    def _calculate_uncertainty_score(self, context: DriveContext) -> float:
        """Calculate score based on prediction uncertainty - can reach true zero."""
        if not context.prediction_errors:
            return 0.8  # High curiosity when no data
        
        # Recent prediction error indicates uncertainty
        recent_error = sum(context.prediction_errors[-3:]) / min(3, len(context.prediction_errors))
        
        # Normalize to 0-1 scale (assuming errors typically range 0-2)
        base_uncertainty = min(1.0, recent_error / 2.0)
        
        # REMOVED: Artificial curiosity floor - let drive reach true zero when mastered
        # Natural curiosity emerges purely from prediction uncertainty
        
        # Add small exploration bonus that can decay to zero
        if len(context.prediction_errors) < 50:
            # Early learning phase - maintain some exploration
            exploration_bonus = 0.2 * math.exp(-len(context.prediction_errors) / 25.0)
        else:
            # Mature learning phase - exploration bonus can reach zero
            exploration_bonus = 0.1 * math.exp(-len(context.prediction_errors) / 100.0)
        
        # Natural uncertainty - can reach zero when predictions are excellent
        uncertainty_score = base_uncertainty + exploration_bonus
        
        return uncertainty_score
    
    def _calculate_action_novelty(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how novel/unique this action is."""
        # Simple novelty: actions that are different from recent patterns
        # For now, use action magnitude as a proxy for novelty
        
        forward_magnitude = abs(action.get('forward_motor', 0.0))
        turn_magnitude = abs(action.get('turn_motor', 0.0))
        brake_magnitude = action.get('brake_motor', 0.0)
        
        # Moderate actions are most novel (not too extreme, not too conservative)
        forward_novelty = 1.0 - abs(forward_magnitude - 0.4)  # Peak novelty at 0.4 strength
        turn_novelty = 1.0 - abs(turn_magnitude - 0.4)
        brake_novelty = 1.0 - brake_magnitude  # Low braking is more novel
        
        novelty_score = (forward_novelty + turn_novelty + brake_novelty) / 3.0
        return max(0.0, min(1.0, novelty_score))
    
    def _calculate_learning_opportunity(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate potential learning value of this action."""
        # Learning opportunities are higher when:
        # 1. Robot is in good condition (can afford to experiment)
        # 2. Environment seems safe
        # 3. Recent experiences show potential for learning
        
        # Safety factor (can afford to be curious)
        safety_factor = min(context.robot_health, context.robot_energy)
        
        # Environment safety (based on threat level)
        threat_penalty = {
            'normal': 0.0,
            'alert': 0.2,
            'danger': 0.5,
            'critical': 0.8
        }.get(context.threat_level, 0.3)
        
        environment_safety = 1.0 - threat_penalty
        
        # Recent learning potential (variance in recent experiences suggests learning opportunities)
        learning_potential = 0.5
        if len(context.prediction_errors) >= 3:
            recent_variance = self._calculate_variance(context.prediction_errors[-3:])
            learning_potential = min(1.0, recent_variance * 2.0)  # Higher variance = more to learn
        
        opportunity_score = (safety_factor * 0.4 + environment_safety * 0.3 + learning_potential * 0.3)
        return max(0.0, min(1.0, opportunity_score))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _generate_reasoning(self, uncertainty: float, novelty: float, learning: float) -> str:
        """Generate human-readable reasoning for the curiosity score."""
        primary_factor = max(uncertainty, novelty, learning)
        
        if primary_factor == uncertainty and uncertainty > 0.6:
            return f"High prediction uncertainty ({uncertainty:.2f}) - want to learn more"
        elif primary_factor == novelty and novelty > 0.6:
            return f"Novel action opportunity ({novelty:.2f}) - want to try something new"
        elif primary_factor == learning and learning > 0.6:
            return f"Good learning conditions ({learning:.2f}) - safe to experiment"
        elif primary_factor < 0.3:
            return "Low curiosity - situation seems predictable"
        else:
            return f"Moderate curiosity - balanced learning opportunity"
    
    def _calculate_fluid_curiosity_strength(self, context: DriveContext) -> float:
        """
        Calculate fluid, continuous curiosity drive strength based on watched metrics.
        
        This implements the user's vision of drives being "totally fluid" - 
        strength changes gradually and continuously with watched values.
        """
        import math
        
        # Core curiosity metrics (higher = more curiosity needed)
        prediction_error_metric = 0.0
        if context.prediction_errors:
            # Recent prediction error indicates learning opportunities
            recent_error = sum(context.prediction_errors[-3:]) / min(3, len(context.prediction_errors))
            prediction_error_metric = min(1.0, recent_error / 2.0)  # Normalize to 0-1
        
        # Learning stagnation metric (higher = need more curiosity)
        stagnation_metric = self.learning_stagnation_penalty
        
        # Novelty metric - based purely on prediction variance (can reach zero)
        novelty_metric = 0.0  # NO baseline - novelty must be earned through variance
        if len(self.recent_prediction_errors) >= 5:
            recent_variance = self._calculate_variance(self.recent_prediction_errors[-5:])
            novelty_metric = min(1.0, recent_variance * 3.0)  # Higher variance = more novelty
        
        # Safety factor - curiosity should be suppressed when survival is critical
        survival_urgency = 0.0
        if context.robot_health < 0.3 or context.robot_energy < 0.2:
            # Calculate survival urgency similar to survival drive
            health_urgency = (1.0 - context.robot_health) if context.robot_health < 0.5 else 0.0
            energy_urgency = (1.0 - context.robot_energy) if context.robot_energy < 0.5 else 0.0
            survival_urgency = max(health_urgency, energy_urgency)
        
        # Threat suppression - curiosity reduces under threat
        threat_suppression = {
            'normal': 0.0,
            'alert': 0.1,
            'danger': 0.4,
            'critical': 0.8
        }.get(context.threat_level, 0.0)
        
        # Combine curiosity-promoting factors
        curiosity_promotion = (
            prediction_error_metric * 0.4 +    # Prediction errors drive learning
            stagnation_metric * 0.3 +          # Stagnation needs new exploration  
            novelty_metric * 0.3               # Environmental novelty attracts curiosity
        )
        
        # Apply safety and threat suppression (can reduce to zero)
        safety_factor = max(0.0, 1.0 - survival_urgency - threat_suppression)
        modulated_curiosity = curiosity_promotion * safety_factor
        
        # Apply drive strength curve - maps curiosity need to drive weight
        # REMOVED artificial baseline - curiosity can reach true zero when no learning needed
        # When modulated_curiosity = 0 (no learning needed): drive_strength = 0.0 (true rest)
        # When modulated_curiosity = 0.5 (moderate): drive_strength ≈ 0.5 (active)
        # When modulated_curiosity = 1.0 (high need): drive_strength ≈ 0.8 (dominant, but not survival-level)
        
        base_strength = 0.0   # NO baseline - curiosity can be truly zero
        max_strength = 0.8    # Maximum curiosity strength
        
        # Sigmoid-like curve for smooth, natural response
        curiosity_response = 1.0 / (1.0 + math.exp(-6 * (modulated_curiosity - 0.5)))
        drive_strength = base_strength + (max_strength - base_strength) * curiosity_response
        
        return min(max_strength, max(base_strength, drive_strength))
    
    def _calculate_pure_novelty_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate pure novelty score based on predicted experience unfamiliarity."""
        # Predict what experience this action will create
        predicted_experience = self._predict_experience_signature(action, context)
        
        # Check familiarity with this experience type
        familiarity = self.experience_familiarity[predicted_experience]
        
        # Higher novelty for unfamiliar experiences
        novelty_score = max(0.0, 1.0 - familiarity)
        
        return novelty_score
    
    def _calculate_self_discovery_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how much this action will teach us about our own capabilities."""
        # Create motor pattern signature
        motor_signature = self._create_motor_signature(action)
        motor_familiarity = self.motor_pattern_familiarity[motor_signature]
        
        # Self-discovery is high for unfamiliar motor patterns
        motor_novelty = max(0.0, 1.0 - motor_familiarity)
        
        # Bonus for exploring motor limits/combinations
        motor_complexity = self._calculate_motor_complexity(action)
        exploration_bonus = motor_complexity * 0.3
        
        return min(1.0, motor_novelty + exploration_bonus)
    
    def _calculate_environment_novelty_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how much this action will teach us about the environment."""
        if not hasattr(context, 'current_sensory') or not context.current_sensory:
            return 0.5  # Unknown environment = high novelty
        
        # Create current environmental signature
        env_signature = self._create_environment_signature(context.current_sensory)
        env_familiarity = self.sensor_pattern_familiarity[env_signature]
        
        # Environmental novelty for unfamiliar sensor patterns
        env_novelty = max(0.0, 1.0 - env_familiarity)
        
        # Bonus for actions that might change environment significantly
        if abs(action.get('forward_motor', 0.0)) > 0.5:
            env_novelty += 0.2  # Moving might discover new areas
        
        return min(1.0, env_novelty)
    
    def _calculate_sensor_exploration_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how much this action will teach us about our sensors."""
        if not hasattr(context, 'current_sensory') or not context.current_sensory:
            return 0.8  # No sensor data = high exploration value
        
        # Look for sensor indices we haven't explored much
        sensor_exploration_score = 0.0
        sensor_count = len(context.current_sensory)
        
        for i in range(sensor_count):
            sensor_signature = f"sensor_{i}"
            sensor_familiarity = self.sensor_pattern_familiarity[sensor_signature]
            
            # Unfamiliar sensors are interesting
            if sensor_familiarity < 0.5:
                sensor_exploration_score += (1.0 - sensor_familiarity) / sensor_count
        
        # Actions that might activate different sensors get bonuses
        if abs(action.get('turn_motor', 0.0)) > 0.3:
            sensor_exploration_score += 0.2  # Turning changes what we sense
        
        return min(1.0, sensor_exploration_score)
    
    def _predict_experience_signature(self, action: Dict[str, float], context: DriveContext) -> str:
        """Create a signature for the predicted experience this action will create."""
        # Combine action and current context into experience signature
        motor_sig = self._create_motor_signature(action)
        
        if hasattr(context, 'current_sensory') and context.current_sensory:
            env_sig = self._create_environment_signature(context.current_sensory[:10])  # Use first 10 sensors
        else:
            env_sig = "unknown_env"
            
        return f"{motor_sig}|{env_sig}"
    
    def _create_motor_signature(self, action: Dict[str, float]) -> str:
        """Create a discretized signature for motor actions."""
        if not action:
            return "no_action"
        
        # Discretize action values to create meaningful signatures
        signature_parts = []
        for key in sorted(action.keys()):
            value = action[key]
            if abs(value) > 0.1:  # Only include significant action components
                discretized = round(value * 4) / 4  # Discretize to 0.25 increments
                signature_parts.append(f"{key}:{discretized}")
        
        return "|".join(signature_parts) if signature_parts else "no_action"
    
    def _create_environment_signature(self, sensory_data: List[float]) -> str:
        """Create a discretized signature for environmental sensor patterns."""
        if not sensory_data:
            return "empty_sensors"
        
        # Discretize sensor values to create patterns
        signature_parts = []
        for i, value in enumerate(sensory_data[:10]):  # Use first 10 sensors for environment
            discretized = round(value * 4) / 4  # Discretize to 0.25 increments
            if discretized > 0.1:  # Only include meaningful sensor values
                signature_parts.append(f"s{i}:{discretized}")
        
        return "|".join(signature_parts) if signature_parts else "empty_environment"
    
    def _calculate_motor_complexity(self, action: Dict[str, float]) -> float:
        """Calculate complexity/uniqueness of motor pattern."""
        if not action:
            return 0.0
        
        # Complex actions use multiple motors simultaneously
        active_motors = sum(1 for v in action.values() if abs(v) > 0.1)
        motor_variety = len(set(round(v * 4) for v in action.values() if abs(v) > 0.1))
        
        complexity = (active_motors + motor_variety) / 6.0  # Normalize
        return min(1.0, complexity)
    
    def learn_from_experience(self, experience, context: DriveContext):
        """Learn from experience to update familiarity tracking."""
        if not hasattr(experience, 'action_taken'):
            return
        
        # Update experience familiarity
        exp_signature = self._predict_experience_signature(experience.action_taken, context)
        self.experience_familiarity[exp_signature] = min(1.0, 
            self.experience_familiarity[exp_signature] + 0.1)
        
        # Update motor pattern familiarity
        motor_signature = self._create_motor_signature(experience.action_taken)
        self.motor_pattern_familiarity[motor_signature] = min(1.0,
            self.motor_pattern_familiarity[motor_signature] + 0.1)
        
        # Update sensor pattern familiarity
        if hasattr(experience, 'actual_sensory') and experience.actual_sensory:
            env_signature = self._create_environment_signature(experience.actual_sensory[:10])
            self.sensor_pattern_familiarity[env_signature] = min(1.0,
                self.sensor_pattern_familiarity[env_signature] + 0.1)
            
            # Update individual sensor familiarity
            for i in range(min(len(experience.actual_sensory), 24)):
                sensor_sig = f"sensor_{i}"
                self.sensor_pattern_familiarity[sensor_sig] = min(1.0,
                    self.sensor_pattern_familiarity[sensor_sig] + 0.05)
        
        # Gradual decay of all familiarity (allows re-exploration)
        self._decay_familiarity()
    
    def _decay_familiarity(self):
        """Gradually decay familiarity to allow re-exploration of old patterns."""
        # Decay experience familiarity
        for key in list(self.experience_familiarity.keys()):
            self.experience_familiarity[key] = max(0.0, 
                self.experience_familiarity[key] - self.familiarity_decay_rate)
            if self.experience_familiarity[key] <= 0.0:
                del self.experience_familiarity[key]
        
        # Decay motor pattern familiarity
        for key in list(self.motor_pattern_familiarity.keys()):
            self.motor_pattern_familiarity[key] = max(0.0,
                self.motor_pattern_familiarity[key] - self.familiarity_decay_rate)
            if self.motor_pattern_familiarity[key] <= 0.0:
                del self.motor_pattern_familiarity[key]
        
        # Decay sensor pattern familiarity
        for key in list(self.sensor_pattern_familiarity.keys()):
            self.sensor_pattern_familiarity[key] = max(0.0,
                self.sensor_pattern_familiarity[key] - self.familiarity_decay_rate)
            if self.sensor_pattern_familiarity[key] <= 0.0:
                del self.sensor_pattern_familiarity[key]
    
    def _generate_novelty_reasoning(self, novelty: float, self_discovery: float, 
                                  environment_novelty: float, sensor_exploration: float) -> str:
        """Generate human-readable reasoning for novelty-seeking."""
        scores = {
            'general_novelty': novelty,
            'self_discovery': self_discovery,
            'environment_novelty': environment_novelty,
            'sensor_exploration': sensor_exploration
        }
        
        dominant_type = max(scores.keys(), key=lambda k: scores[k])
        dominant_score = scores[dominant_type]
        
        if dominant_score > 0.7:
            if dominant_type == 'self_discovery':
                return f"High self-discovery potential ({dominant_score:.2f}) - exploring motor capabilities"
            elif dominant_type == 'environment_novelty':
                return f"High environmental novelty ({dominant_score:.2f}) - exploring world patterns"
            elif dominant_type == 'sensor_exploration':
                return f"High sensor exploration value ({dominant_score:.2f}) - learning about sensors"
            else:
                return f"High general novelty ({dominant_score:.2f}) - unfamiliar experience"
        elif dominant_score > 0.4:
            return f"Moderate {dominant_type.replace('_', ' ')} ({dominant_score:.2f})"
        else:
            return f"Low novelty ({dominant_score:.2f}) - familiar patterns"
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """
        Curiosity drive's information processing pleasure/pain evaluation.
        
        Your insight: curiosity pain comes from information pipeline imbalance:
        - Information starvation (boredom) = PAIN
        - Information overload (chaos) = PAIN  
        - Optimal learning (moderate challenge) = PLEASURE
        - Successful consolidation (pattern integration) = PLEASURE
        """
        prediction_error = experience.prediction_error
        
        # Information processing pain/pleasure based on prediction error patterns
        if 0.1 < prediction_error < 2.0:
            # Sweet spot for learning - moderate challenge creates pleasure
            learning_pleasure = min(1.0, prediction_error / 2.0)
            return learning_pleasure
            
        elif prediction_error > 2.0:
            # Information overload - too much chaos creates pain
            overload_pain = min(0.8, (prediction_error - 2.0) / 3.0)
            return -overload_pain
            
        else:  # prediction_error <= 0.1
            # Very low error - context dependent
            
            # Check if this is consolidation (good) or stagnation (bad)
            if self._is_experience_consolidation(experience, context):
                # Low error during consolidation = satisfaction
                return 0.2  # Mild pleasure from successful pattern integration
            else:
                # Low error from repetitive/boring experience = pain
                boredom_intensity = 0.1 / max(0.01, prediction_error)  # Inversely related
                return -min(0.6, boredom_intensity * 0.1)  # Boredom pain
    
    def _is_experience_consolidation(self, experience, context: DriveContext) -> bool:
        """
        Determine if this low-error experience represents consolidation vs stagnation.
        
        Consolidation indicators:
        - Low error but with meaningful action diversity
        - Following a period of higher learning
        """
        # Simple heuristic: if action shows exploration despite low error
        action = experience.action_taken
        movement = abs(action.get('forward_motor', 0)) + abs(action.get('turn_motor', 0))
        
        # If robot is moving despite predictable outcome, it might be consolidating learning
        if movement > 0.3:
            return True  # Active consolidation
        else:
            return False  # Likely stagnation
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Curiosity drive's contribution to robot mood."""
        # Satisfaction based on information processing state
        satisfaction = self._calculate_information_processing_satisfaction(context)
        
        # Urgency based on learning stagnation vs overload
        urgency = max(self.learning_stagnation_penalty, self._calculate_information_overload())
        
        # Confidence based on prediction improvement
        confidence = max(-0.5, min(0.5, self.prediction_improvement_rate))
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def _calculate_information_processing_satisfaction(self, context: DriveContext) -> float:
        """
        Calculate satisfaction based on information pipeline state.
        
        Your brilliant insight: curiosity can be satisfied by data consolidation,
        not just new experiences. Sometimes the brain needs to "digest" information.
        """
        # Check if recent learning rate is in the "sweet spot"
        if len(self.recent_prediction_errors) > 0:
            avg_error = sum(self.recent_prediction_errors[-5:]) / min(5, len(self.recent_prediction_errors))
            
            # Optimal learning zone (0.1-1.0 prediction error)
            if 0.1 < avg_error < 1.0:
                return 0.5  # Good learning happening
            elif avg_error < 0.05:
                # Very low error - could be:
                # 1. Boring stagnation (bad) OR
                # 2. Successful consolidation (good)
                
                # Check if we're in a consolidation phase
                if self._is_brain_consolidating(context):
                    return 0.3  # Satisfied by consolidation work
                else:
                    return -0.6  # Bored, need new stimulation
                    
            elif avg_error > 2.0:
                return -0.4  # Information overload, need consolidation time
            else:
                return -0.1  # Moderate challenge but not optimal
        
        return 0.0  # No data to evaluate
    
    def _calculate_information_overload(self) -> float:
        """
        Calculate if the brain is experiencing information overload.
        
        High variance in recent prediction errors = chaotic, need consolidation.
        """
        if len(self.recent_prediction_errors) < 5:
            return 0.0
        
        # High variance suggests chaotic, overwhelming input
        variance = self._calculate_variance(self.recent_prediction_errors[-5:])
        if variance > 1.0:  # High chaos
            return min(1.0, variance / 2.0)  # Need consolidation time
        
        return 0.0
    
    def _is_brain_consolidating(self, context: DriveContext) -> bool:
        """
        Detect if the brain is currently in a consolidation phase.
        
        Indicators:
        - Low prediction errors (patterns are being integrated)
        - Recent memory merges would be ideal but we don't have access here
        """
        # Simple heuristic: very low recent errors suggest consolidation
        if len(self.recent_prediction_errors) >= 3:
            recent_avg = sum(self.recent_prediction_errors[-3:]) / 3
            return recent_avg < 0.05  # Very predictable = possibly consolidating
        
        return False
    
    def get_curiosity_stats(self) -> Dict:
        """Get detailed curiosity drive statistics."""
        stats = self.get_drive_info()
        stats.update({
            'prediction_improvement_rate': self.prediction_improvement_rate,
            'learning_stagnation_penalty': self.learning_stagnation_penalty,
            'recent_prediction_errors_count': len(self.recent_prediction_errors),
            'average_recent_error': sum(self.recent_prediction_errors[-5:]) / min(5, len(self.recent_prediction_errors)) if self.recent_prediction_errors else 0.0,
            'experience_patterns_tracked': len(self.experience_familiarity),
            'motor_patterns_tracked': len(self.motor_pattern_familiarity),
            'sensor_patterns_tracked': len(self.sensor_pattern_familiarity),
            'familiarity_decay_rate': self.familiarity_decay_rate
        })
        return stats