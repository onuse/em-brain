"""
Curiosity Drive - Motivated by prediction uncertainty and learning opportunities.
The robot wants to understand and predict its world better.
"""

import math
from typing import Dict, List
from .base_drive import BaseDrive, DriveContext, ActionEvaluation


class CuriosityDrive(BaseDrive):
    """
    Drive to reduce prediction uncertainty and learn about the world.
    
    This drive motivates the robot to:
    - Seek out situations with high prediction error
    - Try actions that might lead to learning
    - Avoid repetitive, predictable situations
    """
    
    def __init__(self, base_weight: float = 0.4):
        super().__init__("Curiosity", base_weight)
        self.recent_prediction_errors = []
        self.prediction_improvement_rate = 0.0
        self.learning_stagnation_penalty = 0.0
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate action based on learning potential and prediction uncertainty."""
        self.total_evaluations += 1
        
        # Base curiosity score from prediction uncertainty
        uncertainty_score = self._calculate_uncertainty_score(context)
        
        # Novelty score based on action uniqueness
        novelty_score = self._calculate_action_novelty(action, context)
        
        # Learning opportunity score
        learning_score = self._calculate_learning_opportunity(action, context)
        
        # Combine scores (weighted average)
        curiosity_score = (
            uncertainty_score * 0.4 +
            novelty_score * 0.3 +
            learning_score * 0.3
        )
        
        # Confidence based on how much data we have
        confidence = min(1.0, len(context.prediction_errors) / 10.0)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(uncertainty_score, novelty_score, learning_score)
        
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
        """Calculate score based on prediction uncertainty."""
        if not context.prediction_errors:
            return 0.8  # High curiosity when no data
        
        # Recent prediction error indicates uncertainty
        recent_error = sum(context.prediction_errors[-3:]) / min(3, len(context.prediction_errors))
        
        # Normalize to 0-1 scale (assuming errors typically range 0-2)
        uncertainty_score = min(1.0, recent_error / 2.0)
        
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
        
        # Novelty metric - inversely related to recent prediction accuracy
        novelty_metric = 0.5  # Baseline novelty
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
        
        # Apply safety and threat suppression
        safety_factor = max(0.1, 1.0 - survival_urgency - threat_suppression)
        modulated_curiosity = curiosity_promotion * safety_factor
        
        # Apply drive strength curve - maps curiosity need to drive weight
        # When modulated_curiosity = 0 (no learning needed): drive_strength ≈ 0.15 (quiet baseline)
        # When modulated_curiosity = 0.5 (moderate): drive_strength ≈ 0.45 (active)
        # When modulated_curiosity = 1.0 (high need): drive_strength ≈ 0.75 (dominant, but not survival-level)
        
        base_strength = 0.15  # Baseline curiosity - always some interest in learning
        max_strength = 0.75   # Maximum curiosity strength (lower than survival max)
        
        # Sigmoid-like curve for smooth, natural response
        curiosity_response = 1.0 / (1.0 + math.exp(-6 * (modulated_curiosity - 0.5)))
        drive_strength = base_strength + (max_strength - base_strength) * curiosity_response
        
        return min(max_strength, max(base_strength, drive_strength))
    
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
            'average_recent_error': sum(self.recent_prediction_errors[-5:]) / min(5, len(self.recent_prediction_errors)) if self.recent_prediction_errors else 0.0
        })
        return stats