"""
Survival Drive - Motivated by self-preservation and maintaining operational status.
The robot wants to stay alive, healthy, and energized.
"""

import math
from typing import Dict, List
from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from collections import defaultdict


class SurvivalDrive(BaseDrive):
    """
    Drive to maintain health, energy, and avoid damage.
    
    This drive motivates the robot to:
    - Seek food/energy sources when energy is low
    - Avoid dangerous areas and situations
    - Maintain safe operational parameters
    - Minimize risky behaviors when health is low
    """
    
    def __init__(self, base_weight: float = 0.15):
        super().__init__("Survival", base_weight)
        self.energy_panic_threshold = 0.05  # Only panic at 5% energy (was 15%)
        self.health_panic_threshold = 0.02  # Only panic at 2% health (was 10%)
        self.recent_damage_memory = 5  # Steps to remember recent damage
        self.food_seeking_urgency = 0.0
        
        # Drive-specific pain/pleasure learning
        self.action_pain_associations = defaultdict(float)  # action_signature -> pain_level
        self.action_pleasure_associations = defaultdict(float)  # action_signature -> pleasure_level  
        self.learning_rate = 0.1
        
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Evaluate action based on survival implications."""
        self.total_evaluations += 1
        
        # Energy preservation score
        energy_score = self._calculate_energy_preservation_score(action, context)
        
        # Health protection score  
        health_score = self._calculate_health_protection_score(action, context)
        
        # Damage avoidance score
        safety_score = self._calculate_safety_score(action, context)
        
        # Resource seeking score (food, energy sources)
        resource_score = self._calculate_resource_seeking_score(action, context)
        
        # Combine scores based on current survival state
        if context.robot_energy < self.energy_panic_threshold:
            # Energy crisis - prioritize resource seeking but allow exploration
            survival_score = resource_score * 0.5 + safety_score * 0.2 + energy_score * 0.3
        elif context.robot_health < self.health_panic_threshold:
            # Health crisis - prioritize safety but not completely
            survival_score = safety_score * 0.5 + health_score * 0.2 + resource_score * 0.3
        else:
            # Normal operation - much more balanced, allow exploration
            survival_score = (energy_score * 0.2 + health_score * 0.2 + 
                            safety_score * 0.3 + resource_score * 0.3)
        
        # Confidence based on how critical survival needs are
        urgency_factor = self._calculate_survival_urgency(context)
        confidence = 0.5 + urgency_factor * 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(energy_score, health_score, safety_score, 
                                           resource_score, context)
        
        evaluation = ActionEvaluation(
            drive_name=self.name,
            action_score=survival_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency_factor
        )
        
        # Log high activation
        self.log_activation(survival_score, context.step_count)
        
        return evaluation
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update survival drive based on current robot state."""
        # Calculate survival urgency
        urgency = self._calculate_survival_urgency(context)
        
        # Update food seeking urgency
        if context.robot_energy < 0.5:
            self.food_seeking_urgency = min(1.0, self.food_seeking_urgency + 0.1)
        else:
            self.food_seeking_urgency = max(0.0, self.food_seeking_urgency - 0.05)
        
        # Fluid drive strength based on continuous survival metrics
        survival_strength = self._calculate_fluid_survival_strength(context)
        
        # Apply smooth curve: nearly silent when healthy, dominant when critical
        self.current_weight = survival_strength
        return self.current_weight
    
    def _calculate_energy_preservation_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well action preserves energy."""
        # Lower energy consumption is better for survival
        forward_cost = abs(action.get('forward_motor', 0.0)) * 0.01
        turn_cost = abs(action.get('turn_motor', 0.0)) * 0.005
        brake_cost = action.get('brake_motor', 0.0) * 0.002
        
        total_energy_cost = forward_cost + turn_cost + brake_cost
        
        # If energy is low, heavily penalize high-cost actions
        if context.robot_energy < self.energy_panic_threshold:
            energy_penalty_factor = 3.0
        elif context.robot_energy < 0.5:
            energy_penalty_factor = 1.5
        else:
            energy_penalty_factor = 1.0
        
        # Score inversely related to energy cost
        energy_score = max(0.0, 1.0 - (total_energy_cost * energy_penalty_factor * 20))
        
        # Bonus for brake usage when energy is critically low, but reduced
        if context.robot_energy < 0.05 and action.get('brake_motor', 0.0) > 0.5:
            energy_score = min(1.0, energy_score + 0.2)
        
        return energy_score
    
    def _calculate_health_protection_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well action protects health."""
        # Avoid risky maneuvers when health is low
        health_score = 1.0
        
        # Penalize aggressive actions when health is low
        if context.robot_health < self.health_panic_threshold:
            forward_magnitude = abs(action.get('forward_motor', 0.0))
            turn_magnitude = abs(action.get('turn_motor', 0.0))
            
            # Heavy penalty for aggressive movement when health is critical
            aggression_penalty = (forward_magnitude + turn_magnitude) * 0.5
            health_score -= aggression_penalty
            
            # Bonus for cautious behavior (braking)
            brake_bonus = action.get('brake_motor', 0.0) * 0.3
            health_score += brake_bonus
        
        # Recent damage makes us more cautious
        if context.time_since_last_damage < self.recent_damage_memory:
            damage_recency_factor = 1.0 - (context.time_since_last_damage / self.recent_damage_memory)
            caution_bonus = damage_recency_factor * action.get('brake_motor', 0.0) * 0.2
            health_score += caution_bonus
        
        return max(0.0, min(1.0, health_score))
    
    def _calculate_safety_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how safe this action is given current threat level."""
        # Base safety score
        safety_score = 0.7
        
        # Adjust based on threat level
        threat_penalties = {
            'normal': 0.0,
            'alert': 0.1,
            'danger': 0.3,
            'critical': 0.6
        }
        
        threat_penalty = threat_penalties.get(context.threat_level, 0.2)
        
        # In dangerous situations, prefer stopping/braking
        if context.threat_level in ['danger', 'critical']:
            brake_usage = action.get('brake_motor', 0.0)
            movement_magnitude = abs(action.get('forward_motor', 0.0)) + abs(action.get('turn_motor', 0.0))
            
            # High brake, low movement = safer
            safety_score = 0.3 + brake_usage * 0.5 - movement_magnitude * 0.3
        else:
            # Normal safety evaluation
            safety_score = 1.0 - threat_penalty
        
        return max(0.0, min(1.0, safety_score))
    
    def _calculate_resource_seeking_score(self, action: Dict[str, float], context: DriveContext) -> float:
        """Calculate how well action helps find resources (food/energy)."""
        # Start with baseline resource seeking
        resource_score = 0.5  # Neutral baseline
        
        # Use discovered smell sensors for food attraction (if available)
        smell_direction, smell_intensity = self._get_food_smell_data(context)
        
        # If we smell food (intensity > 0.1), create intrinsic attraction
        if smell_intensity > 0.1:
            # Reward forward movement when smell is detected
            forward_movement = action.get('forward_motor', 0.0)
            if forward_movement > 0.2:
                food_attraction_bonus = smell_intensity * 0.4  # Strong bonus for moving toward food
                resource_score += food_attraction_bonus
            
            # Reward turning toward smell direction (simplified - assume any turning helps exploration)
            turn_movement = abs(action.get('turn_motor', 0.0))
            if turn_movement > 0.3:
                exploration_bonus = smell_intensity * 0.2  # Moderate bonus for exploring when food nearby
                resource_score += exploration_bonus
        
        # Basic resource seeking when energy is low (but not panic level)
        if context.robot_energy < 0.5 and context.time_since_last_food > 30:
            # Encourage forward movement to find food
            forward_movement = action.get('forward_motor', 0.0)
            if forward_movement > 0.2:
                resource_score += 0.3
            
            # Encourage turning to explore
            turn_movement = abs(action.get('turn_motor', 0.0))
            if turn_movement > 0.3:
                resource_score += 0.2
            
            # Discourage excessive braking when seeking resources
            brake_penalty = action.get('brake_motor', 0.0) * 0.3
            resource_score -= brake_penalty
        
        # If energy is very low, this becomes more urgent
        if context.robot_energy < self.energy_panic_threshold:
            resource_score *= 1.3
        
        return max(0.0, min(1.0, resource_score))
    
    def _get_relevant_sensor_purposes(self):
        """Return sensor purposes relevant to survival drive."""
        from core.sensor_semantics import SensorPurpose
        return [
            SensorPurpose.SURVIVAL,
            SensorPurpose.FOOD_DETECTION,
            SensorPurpose.THREAT_DETECTION,
            SensorPurpose.ENERGY_MANAGEMENT,
            SensorPurpose.SELF_STATE
        ]
    
    def _get_food_smell_data(self, context: DriveContext) -> tuple[float, float]:
        """Extract food smell information using discovered sensors."""
        # Try to get chemical sensors for food detection
        chemical_sensors = self.extract_sensor_values(context, 'food_detection')
        
        if len(chemical_sensors) >= 2:
            # We have smell direction and intensity
            return chemical_sensors[0], chemical_sensors[1]  # direction, intensity
        else:
            # No chemical sensors discovered yet
            return 0.0, 0.0
    
    def _calculate_survival_urgency(self, context: DriveContext) -> float:
        """Calculate overall survival urgency [0.0-1.0]."""
        # Energy urgency
        energy_urgency = 0.0
        if context.robot_energy < self.energy_panic_threshold:
            energy_urgency = 1.0 - (context.robot_energy / self.energy_panic_threshold)
        elif context.robot_energy < 0.5:
            energy_urgency = (0.5 - context.robot_energy) * 0.4
        
        # Health urgency  
        health_urgency = 0.0
        if context.robot_health < self.health_panic_threshold:
            health_urgency = 1.0 - (context.robot_health / self.health_panic_threshold)
        elif context.robot_health < 0.6:
            health_urgency = (0.6 - context.robot_health) * 0.3
        
        # Threat urgency
        threat_urgency = {
            'normal': 0.0,
            'alert': 0.2,
            'danger': 0.6,
            'critical': 1.0
        }.get(context.threat_level, 0.0)
        
        # Recent damage urgency
        damage_urgency = 0.0
        if context.time_since_last_damage < self.recent_damage_memory:
            damage_urgency = 0.3 * (1.0 - context.time_since_last_damage / self.recent_damage_memory)
        
        # Take maximum of all urgencies
        overall_urgency = max(energy_urgency, health_urgency, threat_urgency, damage_urgency)
        return min(1.0, overall_urgency)
    
    def _calculate_fluid_survival_strength(self, context: DriveContext) -> float:
        """
        Calculate fluid, continuous drive strength based on watched metrics.
        
        This implements the user's vision of drives being "totally fluid" - 
        strength changes gradually and continuously with watched values.
        """
        import math
        
        # Core survival metrics (0.0 = critical, 1.0 = perfect)
        health_metric = context.robot_health
        energy_metric = context.robot_energy
        
        # Health component - exponential urgency as health drops
        # f(x) = (1-x)^2 gives smooth curve: 0.01 at 90% health, 0.36 at 40% health, 1.0 at 0% health
        health_urgency = (1.0 - health_metric) ** 2
        
        # Energy component - polynomial urgency as energy drops  
        # f(x) = (1-x)^1.5 gives moderate urgency curve
        energy_urgency = (1.0 - energy_metric) ** 1.5
        
        # Recent damage component - exponential decay over time
        damage_recency_factor = 0.0
        if context.time_since_last_damage < self.recent_damage_memory:
            # Recent damage creates temporary urgency boost
            time_factor = (self.recent_damage_memory - context.time_since_last_damage) / self.recent_damage_memory
            damage_recency_factor = 0.3 * math.exp(-time_factor * 2)  # Exponential decay
        
        # Food seeking component - grows with time since last food
        food_urgency = 0.0
        if context.time_since_last_food > 20:
            # Gradual increase in food seeking urgency
            food_urgency = min(0.4, (context.time_since_last_food - 20) / 100.0)
        
        # Threat level component - immediate but smooth response
        threat_multipliers = {
            'normal': 0.0,
            'alert': 0.1,
            'danger': 0.3,
            'critical': 0.6
        }
        threat_urgency = threat_multipliers.get(context.threat_level, 0.0)
        
        # Combine components with weighted importance
        # Health is most critical, then energy, then damage recency, food, and threats
        combined_urgency = (
            health_urgency * 0.4 +      # Health dominates when low
            energy_urgency * 0.25 +     # Energy important but secondary  
            damage_recency_factor * 0.15 + # Recent damage creates temporary urgency
            food_urgency * 0.1 +        # Food seeking grows gradually
            threat_urgency * 0.1        # Environmental threats
        )
        
        # Apply drive strength curve - maps urgency to drive weight
        # REMOVED artificial baseline - survival can be truly zero when safe
        # When urgency = 0 (all good): drive_strength = 0.0 (true safety)
        # When urgency = 0.5 (moderate): drive_strength ≈ 0.2 (mild concern)  
        # When urgency = 1.0 (critical): drive_strength ≈ 0.5 (important but not overwhelming)
        
        base_strength = 0.0   # NO artificial baseline - can be truly zero when safe
        max_strength = 0.5    # Much reduced maximum to prioritize exploration
        
        # Sigmoid-like curve for smooth, natural response
        urgency_response = 1.0 / (1.0 + math.exp(-8 * (combined_urgency - 0.5)))
        drive_strength = base_strength + (max_strength - base_strength) * urgency_response
        
        return min(max_strength, max(base_strength, drive_strength))
    
    def _generate_reasoning(self, energy_score: float, health_score: float, 
                          safety_score: float, resource_score: float, context: DriveContext) -> str:
        """Generate human-readable reasoning for survival evaluation."""
        urgency = self._calculate_survival_urgency(context)
        
        if urgency > 0.8:
            if context.robot_energy < self.energy_panic_threshold:
                return f"CRITICAL: Energy at {context.robot_energy:.1%} - must find food immediately"
            elif context.robot_health < self.health_panic_threshold:
                return f"CRITICAL: Health at {context.robot_health:.1%} - minimize movement"
            elif context.threat_level == 'critical':
                return "CRITICAL: Extreme danger detected - emergency stop"
            else:
                return "CRITICAL: Multiple survival threats - prioritize safety"
        elif urgency > 0.5:
            if energy_score > max(health_score, safety_score, resource_score):
                return f"Moderate energy conservation (energy: {context.robot_energy:.1%})"
            elif safety_score > max(health_score, resource_score):
                return f"Caution due to {context.threat_level} threat level"
            else:
                return "Balanced survival response"
        elif resource_score > 0.7:
            return f"Seeking resources (energy: {context.robot_energy:.1%}, last food: {context.time_since_last_food} steps ago)"
        else:
            return "Good survival status - normal operation"
    
    def evaluate_experience_significance(self, experience, context: DriveContext) -> float:
        """
        Survival drive evaluates experiences based on health/energy outcomes.
        
        This is where survival-specific logic belongs, not in brain_interface.
        """
        if not hasattr(experience, 'actual_sensory') or not hasattr(experience, 'predicted_sensory'):
            return 1.0
        
        actual = experience.actual_sensory
        predicted = experience.predicted_sensory
        
        if len(actual) < 21 or len(predicted) < 21:
            return 1.0
        
        # Calculate health and energy changes (survival outcomes)
        # Updated indices: health=19, energy=20 (indices 17-18 are smell sensors)
        health_change = actual[19] - predicted[19]  # Health change
        energy_change = actual[20] - predicted[20]  # Energy change
        
        # Calculate survival significance
        survival_significance = 1.0
        
        # Health changes are very important
        if health_change > 0.05:  # Significant health gain (food found)
            survival_significance = 3.0  # Strongly remember good outcomes
        elif health_change < -0.01:  # Any health loss (damage taken)
            survival_significance = 4.0  # Strongly remember dangerous situations
        
        # Energy changes are moderately important
        if energy_change > 0.1:  # Energy gain (food consumption)
            survival_significance = max(survival_significance, 2.0)
        elif energy_change < -0.05:  # Significant energy loss
            survival_significance = max(survival_significance, 1.5)
        
        # If robot is in critical state, ALL experiences are more important
        urgency = self._calculate_survival_urgency(context)
        if urgency > 0.7:
            survival_significance *= 1.5  # Crisis makes everything more memorable
        
        return survival_significance
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """
        Survival drive's pain/pleasure evaluation.
        
        Damage = PAIN (strong negative signal)
        Health/Energy gain = PLEASURE (positive signal)
        """
        if not hasattr(experience, 'actual_sensory') or not hasattr(experience, 'predicted_sensory'):
            return 0.0
        
        actual = experience.actual_sensory
        predicted = experience.predicted_sensory
        
        if len(actual) < 21 or len(predicted) < 21:
            return 0.0
        
        # Calculate changes in survival metrics
        # Updated indices: health=19, energy=20 (indices 17-18 are smell sensors)
        health_change = actual[19] - predicted[19]
        energy_change = actual[20] - predicted[20]
        
        # Pain signals for survival drive
        pain_score = 0.0
        
        # PAIN: Health loss is BAD (this should create avoidance)
        if health_change < -0.001:  # Any meaningful health loss
            # Scale pain by severity: small damage = mild pain, big damage = severe pain
            pain_intensity = abs(health_change) * 200  # 0.005 damage = 1.0 pain
            pain_score -= min(1.0, pain_intensity)
            
        # PLEASURE: Health gain is GOOD
        if health_change > 0.001:  # Healing
            pleasure_intensity = health_change * 100
            pain_score += min(1.0, pleasure_intensity)
        
        # PAIN: Energy loss when low is BAD
        if energy_change < -0.001 and context.robot_energy < 0.5:
            energy_pain = abs(energy_change) * 50
            pain_score -= min(0.5, energy_pain)
            
        # PLEASURE: Energy gain when low is GOOD
        if energy_change > 0.01:  # Food consumption
            energy_pleasure = energy_change * 20
            pain_score += min(1.0, energy_pleasure)
        
        return max(-1.0, min(1.0, pain_score))
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Survival drive's contribution to robot mood."""
        urgency = self._calculate_survival_urgency(context)
        
        # Satisfaction inversely related to urgency
        satisfaction = 1.0 - urgency
        
        # Confidence based on current survival state
        confidence = min(context.robot_health, context.robot_energy)
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def get_survival_stats(self) -> Dict:
        """Get detailed survival drive statistics."""
        stats = self.get_drive_info()
        stats.update({
            'food_seeking_urgency': self.food_seeking_urgency,
            'energy_panic_threshold': self.energy_panic_threshold,
            'health_panic_threshold': self.health_panic_threshold,
            'recent_damage_memory_steps': self.recent_damage_memory
        })
        return stats
    
    def evaluate_action_pain_pleasure(self, action: Dict[str, float], context: DriveContext) -> tuple[float, float]:
        """
        Evaluate expected pain/pleasure of an action for survival drive.
        
        Survival drive considers:
        - Pain: Actions that historically led to health/energy loss
        - Pleasure: Actions that historically led to health/energy gain
        """
        action_signature = self._create_action_signature(action)
        
        # Get learned associations
        expected_pain = self.action_pain_associations.get(action_signature, 0.0)
        expected_pleasure = self.action_pleasure_associations.get(action_signature, 0.0)
        
        # Add contextual pain/pleasure based on current state
        contextual_pain = 0.0
        contextual_pleasure = 0.0
        
        # High movement when energy is low = predicted pain
        if context.robot_energy < 0.3:
            movement_intensity = abs(action.get('forward_motor', 0.0)) + abs(action.get('turn_motor', 0.0))
            if movement_intensity > 0.5:
                contextual_pain = -0.3  # Moderate pain for high energy expenditure when low
        
        # Moving towards safety when health is low = predicted pleasure
        if context.robot_health < 0.5:
            # Assume braking or slower movement is safer
            if action.get('brake_motor', 0.0) > 0.3:
                contextual_pleasure = 0.2  # Mild pleasure for cautious behavior
        
        # Combine learned and contextual predictions
        total_pain = max(-1.0, expected_pain + contextual_pain)
        total_pleasure = min(1.0, expected_pleasure + contextual_pleasure)
        
        return total_pain, total_pleasure
    
    def learn_pain_pleasure_association(self, action: Dict[str, float], context: DriveContext, 
                                      outcome_pain: float, outcome_pleasure: float):
        """
        Learn survival-specific pain/pleasure associations from action outcomes.
        """
        action_signature = self._create_action_signature(action)
        
        # Update pain associations using exponential moving average
        current_pain = self.action_pain_associations[action_signature]
        self.action_pain_associations[action_signature] = (
            current_pain * (1 - self.learning_rate) + outcome_pain * self.learning_rate
        )
        
        # Update pleasure associations
        current_pleasure = self.action_pleasure_associations[action_signature]
        self.action_pleasure_associations[action_signature] = (
            current_pleasure * (1 - self.learning_rate) + outcome_pleasure * self.learning_rate
        )
    
    def _create_action_signature(self, action: Dict[str, float]) -> str:
        """Create a signature string for an action for association learning."""
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
    
    def get_pain_pleasure_statistics(self) -> Dict:
        """Get statistics about this drive's pain/pleasure learning."""
        total_pain_associations = len([p for p in self.action_pain_associations.values() if p < -0.1])
        total_pleasure_associations = len([p for p in self.action_pleasure_associations.values() if p > 0.1])
        
        return {
            "learning_implemented": True,
            "total_pain_associations": total_pain_associations,
            "total_pleasure_associations": total_pleasure_associations,
            "total_action_signatures": len(self.action_pain_associations),
            "strongest_pain_association": min(self.action_pain_associations.values()) if self.action_pain_associations else 0.0,
            "strongest_pleasure_association": max(self.action_pleasure_associations.values()) if self.action_pleasure_associations else 0.0
        }