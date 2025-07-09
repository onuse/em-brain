"""
Emergent Motivation System - Simplified, adaptive motivation without artificial thresholds.

This replaces the complex multi-drive system with a simpler approach where:
1. Learning drive handles curiosity and exploration emergently
2. Survival drive only activates when actually needed
3. All parameters auto-tune based on robot's actual performance
4. Homeostatic rest emerges naturally from low learning potential

Key Design Principles:
- No artificial floors or thresholds
- Behavior emerges from natural dynamics
- Parameters adapt to robot's capabilities
- True rest states are possible and desirable
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .unified_learning_drive import UnifiedLearningDrive
from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from .motivation_system import MotivationSystem, MotivationResult


class EmergentSurvivalDrive(BaseDrive):
    """
    Simplified survival drive that only activates when actually needed.
    No artificial anxiety - only responds to real threats and needs.
    """
    
    def __init__(self, base_weight: float = 0.3):
        super().__init__("EmergentSurvival", base_weight)
        
        # Health and energy thresholds that adapt to robot's actual needs
        self.critical_health_threshold = 0.2  # Only activate when truly critical
        self.critical_energy_threshold = 0.15  # Only activate when truly critical
        
        # Adaptive thresholds based on robot's survival history
        self.survival_history = []
        self.max_survival_history = 100
    
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """Only evaluate when survival is actually threatened."""
        self.total_evaluations += 1
        
        # Calculate actual survival need (no artificial anxiety)
        health_need = self._calculate_health_need(context.robot_health)
        energy_need = self._calculate_energy_need(context.robot_energy)
        threat_response_need = self._calculate_threat_response_need(context.threat_level)
        
        # Survival drive only activates when there's real need
        survival_urgency = max(health_need, energy_need, threat_response_need)
        
        if survival_urgency < 0.1:
            # No survival threat - drive is essentially off
            return ActionEvaluation(
                drive_name=self.name,
                action_score=0.0,
                confidence=1.0,  # Very confident there's no threat
                reasoning="No survival threats detected",
                urgency=0.0
            )
        
        # When survival is needed, evaluate actions for survival benefit
        survival_score = self._evaluate_survival_action(action, context, survival_urgency)
        
        confidence = min(1.0, survival_urgency + 0.3)  # More urgent = more confident in need
        reasoning = self._generate_survival_reasoning(health_need, energy_need, threat_response_need)
        
        evaluation = ActionEvaluation(
            drive_name=self.name,
            action_score=survival_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=survival_urgency
        )
        
        self.log_activation(survival_score, context.step_count)
        return evaluation
    
    def update_drive_state(self, context: DriveContext) -> float:
        """Update drive based on actual survival needs."""
        # Calculate current survival pressure
        health_pressure = max(0, self.critical_health_threshold - context.robot_health)
        energy_pressure = max(0, self.critical_energy_threshold - context.robot_energy)
        
        threat_pressure = 0.0
        if context.threat_level in ['danger', 'critical']:
            threat_pressure = 0.8 if context.threat_level == 'danger' else 1.0
        elif context.threat_level == 'alert':
            threat_pressure = 0.3
        
        # Drive weight emerges from actual survival needs - can be zero!
        survival_pressure = max(health_pressure, energy_pressure, threat_pressure)
        self.current_weight = self.base_weight * survival_pressure
        
        # Auto-tune thresholds based on survival history
        self._update_survival_thresholds(context)
        
        return self.current_weight
    
    def _calculate_health_need(self, health: float) -> float:
        """Calculate how much we need to focus on health."""
        if health > self.critical_health_threshold:
            return 0.0  # No health concern
        
        # Exponential urgency as health gets critical
        health_deficit = self.critical_health_threshold - health
        return min(1.0, (health_deficit / self.critical_health_threshold) ** 2)
    
    def _calculate_energy_need(self, energy: float) -> float:
        """Calculate how much we need to focus on energy."""
        if energy > self.critical_energy_threshold:
            return 0.0  # No energy concern
        
        # Exponential urgency as energy gets critical  
        energy_deficit = self.critical_energy_threshold - energy
        return min(1.0, (energy_deficit / self.critical_energy_threshold) ** 2)
    
    def _calculate_threat_response_need(self, threat_level: str) -> float:
        """Calculate need to respond to environmental threats."""
        threat_map = {
            'safe': 0.0,
            'normal': 0.0, 
            'alert': 0.2,
            'danger': 0.7,
            'critical': 1.0
        }
        return threat_map.get(threat_level, 0.0)
    
    def _evaluate_survival_action(self, action: Dict[str, float], context: DriveContext, urgency: float) -> float:
        """Evaluate how well this action serves survival needs."""
        # Simple survival action evaluation
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        brake_motor = action.get('brake_motor', 0.0)
        
        # Conservative actions are better when survival is threatened
        if urgency > 0.7:
            # High urgency - prefer safe, controlled actions
            if brake_motor > 0.5:
                return 0.8  # Stopping is safe
            elif abs(forward_motor) < 0.3 and abs(turn_motor) < 0.3:
                return 0.6  # Careful movement
            else:
                return 0.2  # Risky when survival threatened
        
        elif urgency > 0.3:
            # Moderate urgency - prefer moderate actions
            if 0.2 <= abs(forward_motor) <= 0.5:
                return 0.7  # Moderate movement
            else:
                return 0.4  # Too aggressive or too passive
        
        else:
            # Low urgency - most actions are fine
            return 0.5
    
    def _update_survival_thresholds(self, context: DriveContext):
        """Auto-tune survival thresholds based on robot's actual survival patterns."""
        # Record survival events
        survival_event = {
            'health': context.robot_health,
            'energy': context.robot_energy,
            'threat': context.threat_level,
            'step': context.step_count
        }
        
        self.survival_history.append(survival_event)
        if len(self.survival_history) > self.max_survival_history:
            self.survival_history = self.survival_history[-50:]
        
        # Adapt thresholds based on actual survival patterns
        if len(self.survival_history) >= 20:
            recent_min_health = min(event['health'] for event in self.survival_history[-20:])
            recent_min_energy = min(event['energy'] for event in self.survival_history[-20:])
            
            # Adjust thresholds to be just above recent minimums
            self.critical_health_threshold = max(0.1, recent_min_health + 0.05)
            self.critical_energy_threshold = max(0.1, recent_min_energy + 0.05)
    
    def _generate_survival_reasoning(self, health_need: float, energy_need: float, threat_need: float) -> str:
        """Generate reasoning for survival evaluation."""
        max_need = max(health_need, energy_need, threat_need)
        
        if health_need == max_need and health_need > 0.5:
            return f"Critical health concern ({health_need:.2f})"
        elif energy_need == max_need and energy_need > 0.5:
            return f"Critical energy concern ({energy_need:.2f})"
        elif threat_need == max_need and threat_need > 0.5:
            return f"Environmental threat response ({threat_need:.2f})"
        elif max_need > 0.1:
            return f"Moderate survival concern"
        else:
            return "No survival threats"
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """Survival drive's pain/pleasure evaluation."""
        # Health loss = significant pain
        # Energy loss = moderate pain  
        # Safety gain = pleasure
        # Health/energy gain = pleasure
        
        if hasattr(experience, 'actual_sensory') and len(experience.actual_sensory) > 18:
            # Compare health and energy changes
            current_health = context.robot_health
            current_energy = context.robot_energy
            
            # Health changes
            if current_health < 0.3:
                return -0.8  # Low health = significant pain
            elif current_health > 0.8:
                return 0.3   # Good health = mild pleasure
            
            # Energy changes  
            if current_energy < 0.2:
                return -0.5  # Low energy = moderate pain
            elif current_energy > 0.8:
                return 0.2   # Good energy = mild pleasure
        
        return 0.0  # Neutral when survival is fine
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """Survival drive's contribution to robot mood."""
        health_status = context.robot_health
        energy_status = context.robot_energy
        
        # Satisfaction based on survival security
        if health_status > 0.7 and energy_status > 0.7:
            satisfaction = 0.4  # Secure = good
        elif health_status < 0.3 or energy_status < 0.3:
            satisfaction = -0.6  # Threatened = bad
        else:
            satisfaction = 0.0  # Normal
        
        # Urgency based on survival needs
        survival_urgency = max(
            self._calculate_health_need(health_status),
            self._calculate_energy_need(energy_status),
            self._calculate_threat_response_need(context.threat_level)
        )
        
        # Confidence based on survival stability
        confidence = min(1.0, (health_status + energy_status) / 2.0)
        
        return {
            'satisfaction': satisfaction,
            'urgency': survival_urgency,
            'confidence': confidence
        }


class EmergentMotivationSystem:
    """
    Simplified motivation system with emergent behavior and no artificial thresholds.
    
    Uses only two drives:
    1. UnifiedLearningDrive - handles curiosity and exploration emergently
    2. EmergentSurvivalDrive - only activates when truly needed
    
    All complex behavior emerges from these simple, robust systems.
    """
    
    def __init__(self, world_graph=None):
        self.world_graph = world_graph
        
        # Simple two-drive system
        self.learning_drive = UnifiedLearningDrive(base_weight=0.7)
        self.survival_drive = EmergentSurvivalDrive(base_weight=0.3)
        
        # System adapts to robot's actual capabilities
        self.robot_capabilities = {
            'max_learning_rate': 1.0,  # Will be learned
            'typical_survival_needs': 0.1,  # Will be learned
            'natural_rest_threshold': 0.05  # Will be learned
        }
        
        # Performance tracking for auto-tuning
        self.system_performance = {
            'total_decisions': 0,
            'learning_satisfaction_events': 0,
            'survival_activation_events': 0,
            'natural_rest_events': 0,
            'recent_satisfaction_levels': []
        }
    
    def make_decision(self, context: DriveContext) -> MotivationResult:
        """Make a motivation-driven decision using emergent drive system."""
        self.system_performance['total_decisions'] += 1
        
        # Update both drives
        learning_weight = self.learning_drive.update_drive_state(context)
        survival_weight = self.survival_drive.update_drive_state(context)
        
        # Check for natural homeostatic rest
        if self._is_in_natural_rest_state(learning_weight, survival_weight, context):
            return self._generate_rest_response(context)
        
        # Generate action candidates
        action_candidates = self._generate_adaptive_action_candidates(context)
        
        # Evaluate actions with both drives
        learning_eval = self.learning_drive.evaluate_action(action_candidates[0], context)
        survival_eval = self.survival_drive.evaluate_action(action_candidates[0], context)
        
        # Simple, natural drive combination (no artificial complexity)
        total_learning_contribution = learning_eval.action_score * learning_weight
        total_survival_contribution = survival_eval.action_score * survival_weight
        
        total_weight = learning_weight + survival_weight
        if total_weight > 0:
            combined_score = (total_learning_contribution + total_survival_contribution) / total_weight
        else:
            combined_score = 0.0
        
        # Determine dominant drive naturally
        if learning_weight > survival_weight:
            dominant_drive = "UnifiedLearning"
            dominant_reasoning = learning_eval.reasoning
        else:
            dominant_drive = "EmergentSurvival"
            dominant_reasoning = survival_eval.reasoning
        
        # Create result
        result = MotivationResult(
            chosen_action=action_candidates[0],
            total_score=combined_score,
            drive_contributions={
                "UnifiedLearning": total_learning_contribution,
                "EmergentSurvival": total_survival_contribution
            },
            dominant_drive=dominant_drive,
            confidence=(learning_eval.confidence + survival_eval.confidence) / 2.0,
            reasoning=dominant_reasoning,
            urgency=max(learning_eval.urgency, survival_eval.urgency),
            alternative_actions=action_candidates[1:]
        )
        
        # Auto-tune system based on decision outcome
        self._update_system_performance(result, context)
        
        return result
    
    def _is_in_natural_rest_state(self, learning_weight: float, survival_weight: float, context: DriveContext) -> bool:
        """Check if robot has naturally achieved homeostatic rest."""
        # Natural rest occurs when:
        # 1. Learning potential is very low (mastery achieved)
        # 2. No survival needs
        # 3. No artificial floors preventing rest
        
        total_drive_pressure = learning_weight + survival_weight
        
        # True rest emerges when total drive pressure is naturally low
        rest_threshold = self.robot_capabilities['natural_rest_threshold']
        
        if total_drive_pressure < rest_threshold:
            self.system_performance['natural_rest_events'] += 1
            return True
        
        return False
    
    def _generate_rest_response(self, context: DriveContext) -> MotivationResult:
        """Generate a natural rest response when drives are satisfied."""
        # Minimal action during rest - just enough to maintain position
        rest_action = {
            'forward_motor': 0.0,
            'turn_motor': 0.0,
            'brake_motor': 0.1  # Light brake to maintain position
        }
        
        return MotivationResult(
            chosen_action=rest_action,
            total_score=0.1,  # Low score = rest state
            drive_contributions={"Rest": 0.1},
            dominant_drive="Rest",
            confidence=0.9,  # High confidence in rest decision
            reasoning="Natural homeostatic rest - drives satisfied",
            urgency=0.0,
            alternative_actions=[]
        )
    
    def _generate_adaptive_action_candidates(self, context: DriveContext) -> List[Dict[str, float]]:
        """Generate action candidates that adapt to robot's capabilities."""
        import random
        
        # Simple adaptive candidate generation
        # In a full implementation, this would be more sophisticated
        
        candidates = []
        
        # Conservative action
        candidates.append({
            'forward_motor': random.uniform(0.1, 0.3),
            'turn_motor': 0.0,
            'brake_motor': 0.0
        })
        
        # Exploratory action
        candidates.append({
            'forward_motor': random.uniform(0.3, 0.6),
            'turn_motor': random.choice([-0.5, 0.0, 0.5]),
            'brake_motor': 0.0
        })
        
        # Careful action
        candidates.append({
            'forward_motor': random.uniform(-0.2, 0.2),
            'turn_motor': random.uniform(-0.3, 0.3),
            'brake_motor': random.uniform(0.0, 0.3)
        })
        
        return candidates
    
    def _update_system_performance(self, result: MotivationResult, context: DriveContext):
        """Update system performance metrics for auto-tuning."""
        # Track satisfaction events
        if result.total_score > 0.7:
            self.system_performance['learning_satisfaction_events'] += 1
        
        if result.dominant_drive == "EmergentSurvival":
            self.system_performance['survival_activation_events'] += 1
        
        # Track recent satisfaction for auto-tuning
        satisfaction_level = result.confidence * result.total_score
        self.system_performance['recent_satisfaction_levels'].append(satisfaction_level)
        
        if len(self.system_performance['recent_satisfaction_levels']) > 50:
            self.system_performance['recent_satisfaction_levels'] = self.system_performance['recent_satisfaction_levels'][-25:]
        
        # Auto-tune rest threshold based on system performance
        if len(self.system_performance['recent_satisfaction_levels']) >= 20:
            avg_satisfaction = np.mean(self.system_performance['recent_satisfaction_levels'])
            
            # If average satisfaction is high, robot can rest more easily
            if avg_satisfaction > 0.6:
                self.robot_capabilities['natural_rest_threshold'] = min(0.1, self.robot_capabilities['natural_rest_threshold'] * 1.01)
            else:
                self.robot_capabilities['natural_rest_threshold'] = max(0.01, self.robot_capabilities['natural_rest_threshold'] * 0.99)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the emergent motivation system."""
        learning_stats = self.learning_drive.get_learning_statistics()
        
        total_decisions = self.system_performance['total_decisions']
        
        stats = {
            'emergent_system_stats': {
                'total_decisions': total_decisions,
                'natural_rest_events': self.system_performance['natural_rest_events'],
                'rest_frequency': self.system_performance['natural_rest_events'] / max(1, total_decisions),
                'survival_activation_rate': self.system_performance['survival_activation_events'] / max(1, total_decisions),
                'learning_satisfaction_rate': self.system_performance['learning_satisfaction_events'] / max(1, total_decisions)
            },
            'learning_drive_stats': learning_stats,
            'survival_drive_stats': {
                'current_weight': self.survival_drive.current_weight,
                'critical_health_threshold': self.survival_drive.critical_health_threshold,
                'critical_energy_threshold': self.survival_drive.critical_energy_threshold
            },
            'robot_capabilities': self.robot_capabilities.copy(),
            'current_drive_weights': {
                'learning': self.learning_drive.current_weight,
                'survival': self.survival_drive.current_weight
            }
        }
        
        if self.system_performance['recent_satisfaction_levels']:
            stats['emergent_system_stats']['recent_avg_satisfaction'] = np.mean(self.system_performance['recent_satisfaction_levels'])
        
        return stats
    
    def calculate_robot_mood(self, context: DriveContext) -> Dict[str, float]:
        """Calculate robot's overall emotional state from emergent drives."""
        learning_mood = self.learning_drive.get_current_mood_contribution(context)
        survival_mood = self.survival_drive.get_current_mood_contribution(context)
        
        # Natural mood combination (no artificial weighting)
        learning_weight = self.learning_drive.current_weight
        survival_weight = self.survival_drive.current_weight
        total_weight = learning_weight + survival_weight
        
        if total_weight > 0:
            combined_satisfaction = (
                learning_mood['satisfaction'] * learning_weight +
                survival_mood['satisfaction'] * survival_weight
            ) / total_weight
            
            combined_urgency = max(learning_mood['urgency'], survival_mood['urgency'])
            combined_confidence = (
                learning_mood['confidence'] * learning_weight +
                survival_mood['confidence'] * survival_weight  
            ) / total_weight
        else:
            # True rest state
            combined_satisfaction = 0.8  # Rest is satisfying
            combined_urgency = 0.0
            combined_confidence = 0.9
        
        # Natural mood descriptors
        if combined_urgency > 0.7:
            mood_descriptor = "urgent"
        elif combined_satisfaction > 0.5:
            mood_descriptor = "content"
        elif combined_satisfaction > 0.0:
            mood_descriptor = "calm"
        elif combined_satisfaction > -0.3:
            mood_descriptor = "restless"
        else:
            mood_descriptor = "distressed"
        
        return {
            'overall_satisfaction': combined_satisfaction,
            'overall_urgency': combined_urgency,
            'overall_confidence': combined_confidence,
            'mood_descriptor': mood_descriptor,
            'dominant_emotion': 'learning' if learning_weight > survival_weight else 'survival',
            'drive_moods': {
                'learning': learning_mood,
                'survival': survival_mood
            }
        }


def create_emergent_motivation_system(world_graph=None) -> EmergentMotivationSystem:
    """Create a simplified, emergent motivation system."""
    return EmergentMotivationSystem(world_graph=world_graph)