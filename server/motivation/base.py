"""
Base Motivation System Components

Defines core interfaces for competitive action selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time


@dataclass
class ActionProposal:
    """
    Standardized action proposal from a motivation.
    
    Motivations compete by proposing actions with associated values.
    The MotivationSystem selects the highest-valued proposal.
    """
    action: Any                    # The proposed action
    predicted_outcome: Any         # Brain's prediction for this action
    motivation_value: float        # How much this motivation wants this (0-1)
    confidence: float              # Motivation's confidence in value (0-1) 
    motivation_name: str           # Which motivation proposed this
    reasoning: str                 # Why this motivation wants this action
    timestamp: float = None        # When proposal was created
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @property
    def combined_score(self) -> float:
        """Overall proposal strength (used for competition)."""
        return self.motivation_value * self.confidence
    
    def __str__(self) -> str:
        return f"{self.motivation_name}: {self.action} (value={self.motivation_value:.2f}, conf={self.confidence:.2f})"


class BaseMotivation(ABC):
    """
    Abstract base class for all motivations.
    
    Motivations use the 4-system brain to evaluate states and propose actions.
    They compete through proposal quality, not direct interaction.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize motivation.
        
        Args:
            name: Human-readable motivation name
            weight: Global weight for this motivation (affects competition)
        """
        self.name = name
        self.weight = weight
        self.brain = None  # Connected by MotivationSystem
        
        # Statistics for analysis
        self.proposals_made = 0
        self.proposals_won = 0
        self.total_value_generated = 0.0
        self.last_proposal = None
        
    def connect_to_brain(self, brain):
        """Connect this motivation to the 4-system brain."""
        self.brain = brain
        
    @abstractmethod
    def propose_action(self, current_state: Any) -> ActionProposal:
        """
        Generate action proposal for current state.
        
        Args:
            current_state: Current sensory/robot state
            
        Returns:
            ActionProposal with this motivation's preferred action
        """
        pass
    
    @abstractmethod
    def calculate_value(self, predicted_outcome: Any) -> float:
        """
        Calculate how much this motivation values a predicted outcome.
        
        Args:
            predicted_outcome: Brain's prediction of action result
            
        Returns:
            Value between 0.0 (undesirable) and 1.0 (highly desirable)
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get motivation performance statistics."""
        win_rate = self.proposals_won / max(1, self.proposals_made)
        avg_value = self.total_value_generated / max(1, self.proposals_made)
        
        return {
            'name': self.name,
            'weight': self.weight,
            'proposals_made': self.proposals_made,
            'proposals_won': self.proposals_won,
            'win_rate': win_rate,
            'average_value': avg_value,
            'last_proposal': str(self.last_proposal) if self.last_proposal else None
        }
    
    def _record_proposal(self, proposal: ActionProposal):
        """Record proposal for statistics tracking."""
        self.proposals_made += 1
        self.total_value_generated += proposal.motivation_value
        self.last_proposal = proposal
    
    def _record_win(self):
        """Record that this motivation's proposal won."""
        self.proposals_won += 1
    
    def __str__(self) -> str:
        return f"{self.name}(weight={self.weight:.2f})"