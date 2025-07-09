"""
Pain/Pleasure Learning System.

This system implements sophisticated pain avoidance and pleasure seeking behaviors
that influence action selection based on past emotional experiences.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from drives.base_drive import DriveContext


@dataclass
class PainPleasureAssociation:
    """Association between actions/contexts and pain/pleasure outcomes."""
    action_signature: str  # String representation of action
    context_signature: str  # String representation of context
    pain_level: float  # Negative for pain, positive for pleasure
    confidence: float  # How confident we are in this association
    sample_count: int  # Number of experiences contributing to this association
    last_updated: float  # When this association was last updated


class PainPleasureLearningSystem:
    """
    System for learning action-outcome associations and biasing future decisions
    based on pain/pleasure experiences.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Pain/pleasure associations
        self.action_associations: Dict[str, PainPleasureAssociation] = {}
        self.context_associations: Dict[str, PainPleasureAssociation] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.decay_rate = 0.95
        self.min_confidence = 0.1
        self.pain_amplification = 2.0  # Pain is more memorable than pleasure
        self.pleasure_amplification = 1.5
        
        # Avoidance parameters
        self.pain_avoidance_threshold = -0.3
        self.pleasure_seeking_threshold = 0.2
        self.avoidance_strength = 0.8  # How much to avoid painful actions
        
        # Statistics
        self.total_associations_learned = 0
        self.pain_avoidance_events = 0
        self.pleasure_seeking_events = 0
        
    def learn_from_experience(self, experience: ExperienceNode, context: DriveContext):
        """
        Learn pain/pleasure associations from a new experience.
        
        Args:
            experience: The experience node to learn from
            context: The context when the experience occurred
        """
        # Calculate pain/pleasure from experience
        pain_pleasure = self._calculate_pain_pleasure_from_experience(experience, context)
        
        if abs(pain_pleasure) < 0.1:  # Not significant enough to learn from
            return
        
        # Create action and context signatures
        action_signature = self._create_action_signature(experience.action_taken)
        context_signature = self._create_context_signature(experience.mental_context)
        
        # Update action association
        self._update_action_association(action_signature, pain_pleasure)
        
        # Update context association
        self._update_context_association(context_signature, pain_pleasure)
        
        self.total_associations_learned += 1
    
    def _calculate_pain_pleasure_from_experience(self, experience: ExperienceNode, 
                                               context: DriveContext) -> float:
        """Calculate pain/pleasure level from an experience."""
        total_pain_pleasure = 0.0
        
        # Check for direct pain/pleasure signals
        if hasattr(experience, 'pain_signal'):
            total_pain_pleasure += experience.pain_signal
        
        if hasattr(experience, 'pleasure_signal'):
            total_pain_pleasure += experience.pleasure_signal
        
        # Infer pain/pleasure from context changes
        if context:
            # Health decrease = pain
            if context.robot_health < 0.5:
                health_pain = (0.5 - context.robot_health) * 2.0
                total_pain_pleasure -= health_pain
            
            # Energy decrease = mild pain
            if context.robot_energy < 0.3:
                energy_pain = (0.3 - context.robot_energy) * 1.0
                total_pain_pleasure -= energy_pain
            
            # High prediction error = frustration (mild pain)
            if experience.prediction_error > 0.7:
                frustration = (experience.prediction_error - 0.7) * 0.5
                total_pain_pleasure -= frustration
            
            # Low prediction error = satisfaction (mild pleasure)
            if experience.prediction_error < 0.2:
                satisfaction = (0.2 - experience.prediction_error) * 0.3
                total_pain_pleasure += satisfaction
        
        return total_pain_pleasure
    
    def _create_action_signature(self, action: Dict[str, float]) -> str:
        """Create a signature string for an action."""
        if not action:
            return "no_action"
        
        # Discretize action values to create meaningful signatures
        signature_parts = []
        for key, value in sorted(action.items()):
            if abs(value) > 0.1:  # Only include significant action components
                discretized = round(value * 4) / 4  # Discretize to 0.25 increments
                signature_parts.append(f"{key}:{discretized}")
        
        return "|".join(signature_parts) if signature_parts else "no_action"
    
    def _create_context_signature(self, context: List[float]) -> str:
        """Create a signature string for a context."""
        if not context:
            return "no_context"
        
        # Use first few context elements as signature (environmental features)
        significant_elements = []
        for i, value in enumerate(context[:6]):  # Use first 6 elements
            if abs(value) > 0.1:
                discretized = round(value * 4) / 4
                significant_elements.append(f"{i}:{discretized}")
        
        return "|".join(significant_elements) if significant_elements else "no_context"
    
    def _update_action_association(self, action_signature: str, pain_pleasure: float):
        """Update the pain/pleasure association for an action."""
        if action_signature not in self.action_associations:
            self.action_associations[action_signature] = PainPleasureAssociation(
                action_signature=action_signature,
                context_signature="",
                pain_level=pain_pleasure,
                confidence=0.5,
                sample_count=1,
                last_updated=time.time()
            )
        else:
            assoc = self.action_associations[action_signature]
            
            # Update using exponential moving average
            assoc.pain_level = (assoc.pain_level * (1 - self.learning_rate) + 
                              pain_pleasure * self.learning_rate)
            
            # Update confidence (more samples = higher confidence)
            assoc.sample_count += 1
            assoc.confidence = min(1.0, assoc.confidence + 0.1)
            assoc.last_updated = time.time()
    
    def _update_context_association(self, context_signature: str, pain_pleasure: float):
        """Update the pain/pleasure association for a context."""
        if context_signature not in self.context_associations:
            self.context_associations[context_signature] = PainPleasureAssociation(
                action_signature="",
                context_signature=context_signature,
                pain_level=pain_pleasure,
                confidence=0.5,
                sample_count=1,
                last_updated=time.time()
            )
        else:
            assoc = self.context_associations[context_signature]
            
            # Update using exponential moving average
            assoc.pain_level = (assoc.pain_level * (1 - self.learning_rate) + 
                              pain_pleasure * self.learning_rate)
            
            # Update confidence
            assoc.sample_count += 1
            assoc.confidence = min(1.0, assoc.confidence + 0.1)
            assoc.last_updated = time.time()
    
    def evaluate_action_pain_pleasure(self, action: Dict[str, float], 
                                    context: DriveContext) -> Tuple[float, str]:
        """
        Evaluate the expected pain/pleasure of an action in a given context.
        
        Returns:
            Tuple of (pain_pleasure_score, reasoning)
        """
        action_signature = self._create_action_signature(action)
        context_signature = self._create_context_signature(context.current_sensory[:6])
        
        total_pain_pleasure = 0.0
        reasoning_parts = []
        
        # Check action association
        if action_signature in self.action_associations:
            assoc = self.action_associations[action_signature]
            weighted_pain_pleasure = assoc.pain_level * assoc.confidence
            total_pain_pleasure += weighted_pain_pleasure
            
            if assoc.pain_level < self.pain_avoidance_threshold:
                reasoning_parts.append(f"painful action (pain: {assoc.pain_level:.2f})")
                self.pain_avoidance_events += 1
            elif assoc.pain_level > self.pleasure_seeking_threshold:
                reasoning_parts.append(f"pleasurable action (pleasure: {assoc.pain_level:.2f})")
                self.pleasure_seeking_events += 1
        
        # Check context association
        if context_signature in self.context_associations:
            assoc = self.context_associations[context_signature]
            weighted_pain_pleasure = assoc.pain_level * assoc.confidence * 0.5  # Context has less weight
            total_pain_pleasure += weighted_pain_pleasure
            
            if assoc.pain_level < self.pain_avoidance_threshold:
                reasoning_parts.append(f"dangerous context (pain: {assoc.pain_level:.2f})")
            elif assoc.pain_level > self.pleasure_seeking_threshold:
                reasoning_parts.append(f"favorable context (pleasure: {assoc.pain_level:.2f})")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "no significant associations"
        
        return total_pain_pleasure, reasoning
    
    def get_action_bias(self, action: Dict[str, float], context: DriveContext) -> float:
        """
        Get the bias (positive or negative) for an action based on pain/pleasure learning.
        
        Returns:
            Bias value (-1.0 to 1.0) where negative values discourage the action
        """
        pain_pleasure, _ = self.evaluate_action_pain_pleasure(action, context)
        
        # Convert pain/pleasure to bias
        if pain_pleasure < self.pain_avoidance_threshold:
            # Strong negative bias for painful actions
            bias = -self.avoidance_strength * min(1.0, abs(pain_pleasure))
        elif pain_pleasure > self.pleasure_seeking_threshold:
            # Positive bias for pleasurable actions
            bias = 0.5 * min(1.0, pain_pleasure)
        else:
            # Neutral bias
            bias = 0.0
        
        return bias
    
    def decay_associations(self):
        """Decay old pain/pleasure associations over time."""
        current_time = time.time()
        
        # Decay action associations
        for signature, assoc in list(self.action_associations.items()):
            age = current_time - assoc.last_updated
            if age > 300:  # 5 minutes
                assoc.confidence *= self.decay_rate
                
                # Remove very weak associations
                if assoc.confidence < self.min_confidence:
                    del self.action_associations[signature]
        
        # Decay context associations
        for signature, assoc in list(self.context_associations.items()):
            age = current_time - assoc.last_updated
            if age > 300:  # 5 minutes
                assoc.confidence *= self.decay_rate
                
                # Remove very weak associations
                if assoc.confidence < self.min_confidence:
                    del self.context_associations[signature]
    
    def get_strongest_pain_associations(self, limit: int = 10) -> List[PainPleasureAssociation]:
        """Get the strongest pain associations (for debugging/analysis)."""
        pain_associations = [
            assoc for assoc in self.action_associations.values()
            if assoc.pain_level < self.pain_avoidance_threshold
        ]
        
        pain_associations.sort(key=lambda x: x.pain_level * x.confidence)
        return pain_associations[:limit]
    
    def get_strongest_pleasure_associations(self, limit: int = 10) -> List[PainPleasureAssociation]:
        """Get the strongest pleasure associations (for debugging/analysis)."""
        pleasure_associations = [
            assoc for assoc in self.action_associations.values()
            if assoc.pain_level > self.pleasure_seeking_threshold
        ]
        
        pleasure_associations.sort(key=lambda x: x.pain_level * x.confidence, reverse=True)
        return pleasure_associations[:limit]
    
    def get_pain_pleasure_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pain/pleasure learning system."""
        total_action_associations = len(self.action_associations)
        total_context_associations = len(self.context_associations)
        
        pain_actions = len([a for a in self.action_associations.values() 
                          if a.pain_level < self.pain_avoidance_threshold])
        pleasure_actions = len([a for a in self.action_associations.values() 
                             if a.pain_level > self.pleasure_seeking_threshold])
        
        return {
            "total_associations_learned": self.total_associations_learned,
            "total_action_associations": total_action_associations,
            "total_context_associations": total_context_associations,
            "pain_actions_learned": pain_actions,
            "pleasure_actions_learned": pleasure_actions,
            "pain_avoidance_events": self.pain_avoidance_events,
            "pleasure_seeking_events": self.pleasure_seeking_events,
            "avoidance_ratio": pain_actions / max(1, total_action_associations),
            "approach_ratio": pleasure_actions / max(1, total_action_associations)
        }
    
    def reset_learning(self):
        """Reset all learned associations."""
        self.action_associations.clear()
        self.context_associations.clear()
        self.total_associations_learned = 0
        self.pain_avoidance_events = 0
        self.pleasure_seeking_events = 0
        
        print("🧠 Pain/pleasure learning system reset")