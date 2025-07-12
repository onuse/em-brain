"""
Experience Data Models

The fundamental unit of memory: a single sensory-motor moment.
This is the atomic building block from which all intelligence emerges.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import uuid


@dataclass
class Experience:
    """
    A single moment of embodied experience.
    
    This is the fundamental unit of memory - what I sensed, what I did,
    what happened, and how wrong my prediction was.
    
    Everything else emerges from collections of these simple records.
    """
    
    # Core experience data
    sensory_input: List[float]      # What I sensed at this moment
    action_taken: List[float]       # What I did in response
    outcome: List[float]            # What actually happened next
    prediction_error: float         # How wrong my prediction was (0.0-1.0)
    timestamp: float                # When this occurred (time.time())
    
    # Unique identifier
    experience_id: str = None
    
    # Activation state (for working memory)
    activation_level: float = 0.0   # Current activation (0.0-1.0)
    last_accessed: float = 0.0      # When last retrieved
    access_count: int = 0           # How many times retrieved
    
    # Connection weights to other experiences (learned similarities)
    similar_experiences: Dict[str, float] = None  # experience_id -> similarity
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.experience_id is None:
            self.experience_id = str(uuid.uuid4())
        
        if self.similar_experiences is None:
            self.similar_experiences = {}
        
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
    
    def activate(self, strength: float = 1.0):
        """
        Activate this experience (brings into working memory).
        
        Args:
            strength: Activation strength (0.0-1.0)
        """
        self.activation_level = min(1.0, self.activation_level + strength)
        self.last_accessed = time.time()
        self.access_count += 1
    
    def decay_activation(self, decay_rate: float = 0.01):
        """
        Natural decay of activation over time.
        
        Args:
            decay_rate: How fast activation decays (0.0-1.0)
        """
        self.activation_level = max(0.0, self.activation_level - decay_rate)
    
    def add_similarity(self, other_experience_id: str, similarity: float):
        """
        Record similarity to another experience.
        
        Args:
            other_experience_id: ID of similar experience
            similarity: Similarity strength (0.0-1.0)
        """
        self.similar_experiences[other_experience_id] = similarity
    
    def get_similarity(self, other_experience_id: str) -> float:
        """Get similarity to another experience (0.0 if not recorded)."""
        return self.similar_experiences.get(other_experience_id, 0.0)
    
    def get_context_vector(self) -> List[float]:
        """
        Get the 'context' of this experience for similarity matching.
        
        For now, this is just the sensory input, but could be more
        sophisticated (sensory + action, or learned features).
        """
        return self.sensory_input.copy()
    
    def get_action_vector(self) -> List[float]:
        """Get the action taken in this experience."""
        return self.action_taken.copy()
    
    def get_outcome_vector(self) -> List[float]:
        """Get the outcome that occurred in this experience."""
        return self.outcome.copy()
    
    def was_successful(self, error_threshold: float = 0.3) -> bool:
        """
        Was this experience successful (low prediction error)?
        
        Args:
            error_threshold: Error below this is considered successful
        """
        return self.prediction_error < error_threshold
    
    def get_memory_size(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate: each float = 8 bytes, plus overhead
        vector_memory = (len(self.sensory_input) + len(self.action_taken) + 
                        len(self.outcome)) * 8
        similarity_memory = len(self.similar_experiences) * 50  # UUID + float
        return vector_memory + similarity_memory + 200  # Fixed overhead
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experience_id': self.experience_id,
            'sensory_input': self.sensory_input,
            'action_taken': self.action_taken,
            'outcome': self.outcome,
            'prediction_error': self.prediction_error,
            'timestamp': self.timestamp,
            'activation_level': self.activation_level,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'similar_experiences': self.similar_experiences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create Experience from dictionary."""
        return cls(
            sensory_input=data['sensory_input'],
            action_taken=data['action_taken'],
            outcome=data['outcome'],
            prediction_error=data['prediction_error'],
            timestamp=data['timestamp'],
            experience_id=data.get('experience_id'),
            activation_level=data.get('activation_level', 0.0),
            last_accessed=data.get('last_accessed', data['timestamp']),
            access_count=data.get('access_count', 0),
            similar_experiences=data.get('similar_experiences', {})
        )
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"Experience({self.experience_id[:8]}... "
                f"sensors={len(self.sensory_input)}, "
                f"error={self.prediction_error:.3f}, "
                f"activation={self.activation_level:.3f})")
    
    def __repr__(self) -> str:
        return self.__str__()