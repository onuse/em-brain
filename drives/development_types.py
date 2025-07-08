"""
Shared types for developmental systems.

Contains enums and classes used by both action proficiency and cognitive development systems.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field


class DevelopmentalStage(Enum):
    """Stages of behavioral maturation."""
    INFANCY = "infancy"           # High exploration, random actions
    EARLY_LEARNING = "early_learning"     # Beginning to notice patterns
    ADOLESCENCE = "adolescence"   # Building skills, developing preferences
    MATURITY = "maturity"         # Stable competent behavior
    EXPERTISE = "expertise"       # Refined specialization


@dataclass
class ActionProficiency:
    """Tracks proficiency for a specific action type."""
    action_type: str
    total_attempts: int = 0
    successful_executions: int = 0  # Actions that matched predictions
    accuracy_history: List[float] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    
    # Skill development metrics
    consistency_score: float = 0.0  # How consistent the execution is
    improvement_rate: float = 0.0   # How quickly improving
    plateau_detected: bool = False  # Whether improvement has stalled
    
    # Maturation indicators
    mastery_level: float = 0.0      # Overall mastery (0.0 to 1.0)
    preference_strength: float = 0.0 # How much the system "likes" this action
    
    def update_proficiency(self, prediction_accuracy: float, execution_success: bool):
        """Update proficiency based on latest action outcome."""
        import statistics
        
        self.total_attempts += 1
        self.accuracy_history.append(prediction_accuracy)
        
        if execution_success:
            self.successful_executions += 1
        
        # Calculate recent consistency (lower variance = higher consistency)
        if len(self.accuracy_history) >= 5:
            recent_accuracies = self.accuracy_history[-10:]
            self.consistency_score = 1.0 - min(1.0, statistics.stdev(recent_accuracies))
        
        # Calculate improvement rate (trend of recent accuracies)
        if len(self.accuracy_history) >= 10:
            recent = self.accuracy_history[-10:]
            early = recent[:5]
            late = recent[5:]
            self.improvement_rate = sum(late) / len(late) - sum(early) / len(early)
        
        # Update mastery level
        self._calculate_mastery_level()
        
        # Update preference strength based on success and consistency
        self._calculate_preference_strength()
    
    def _calculate_mastery_level(self):
        """Calculate overall mastery level for this action."""
        if not self.accuracy_history:
            self.mastery_level = 0.0
            return
        
        # Base skill: average accuracy
        avg_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
        
        # Experience bonus: more attempts = higher potential mastery
        experience_factor = min(1.0, self.total_attempts / 200.0)
        
        # Consistency bonus: consistent performance indicates mastery
        consistency_bonus = self.consistency_score * 0.3
        
        # Recent performance weight: how we're doing lately
        recent_performance = sum(self.accuracy_history[-5:]) / min(5, len(self.accuracy_history))
        
        self.mastery_level = min(1.0, (
            avg_accuracy * 0.4 +
            recent_performance * 0.4 +
            consistency_bonus +
            experience_factor * 0.2
        ))
    
    def _calculate_preference_strength(self):
        """Calculate how much the system should prefer this action."""
        # High mastery → higher preference
        mastery_component = self.mastery_level * 0.5
        
        # Positive improvement → higher preference
        improvement_component = max(0.0, self.improvement_rate) * 0.3
        
        # Success rate component
        success_rate = self.successful_executions / max(1, self.total_attempts)
        success_component = success_rate * 0.2
        
        self.preference_strength = min(1.0, 
            mastery_component + improvement_component + success_component
        )
    
    def get_confidence_level(self) -> float:
        """Get current confidence level for this action."""
        if self.total_attempts < 5:
            return 0.2  # Low confidence with little experience
        
        return min(1.0, self.mastery_level * self.consistency_score)