"""
ExperienceNode - The fundamental unit of memory and learning.
Each node represents a single moment of experience: prediction + action + reality.
Enhanced with neural-like properties for emergent memory phenomena.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import time


@dataclass
class ExperienceNode:
    """
    A single moment of experience: prediction + action + reality.
    Enhanced with neural-like properties that create emergent memory phenomena:
    - Activation levels create "working memory" effects
    - Connection weights create associative networks
    - Natural decay creates forgetting
    - Recency effects prioritize recent experiences
    """
    # Core experience data
    mental_context: List[float]           # Current brain state when this experience occurred
    action_taken: Dict[str, float]        # Motor commands executed {actuator_id: value}
    predicted_sensory: List[float]        # What we expected to sense
    actual_sensory: List[float]           # What actually happened
    prediction_error: float               # Magnitude of prediction miss
    
    # Node metadata
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strength: float = 1.0                 # Usage-based reinforcement (legacy)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Neural-like properties for emergent memory
    activation_level: float = 1.0         # How "excited" this memory is (0.0-2.0)
    recency_bonus: float = 1.0            # Recent memories are more accessible (0.0-1.0)
    access_frequency: int = 0             # How often this node has been accessed
    connection_weights: Dict[str, float] = field(default_factory=dict)  # Weighted connections to other nodes
    consolidation_strength: float = 1.0   # How "permanent" this memory is (0.0-2.0)
    last_activation_time: float = field(default_factory=time.time)  # When last activated
    
    # Graph connections (enhanced)
    temporal_predecessor: Optional[str] = None     # Previous experience in time
    temporal_successor: Optional[str] = None       # Next experience in time
    prediction_sources: List[str] = field(default_factory=list)  # Nodes that predicted this
    similar_contexts: List[str] = field(default_factory=list)    # Similar experience nodes (legacy)
    
    # Learning tracking (legacy compatibility)
    times_accessed: int = 0               # Legacy - use access_frequency instead
    last_accessed: datetime = field(default_factory=datetime.now)  # Legacy - use last_activation_time instead
    merge_count: int = 0                  # How many nodes have been merged into this one
    
    def __hash__(self) -> int:
        return hash(self.node_id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ExperienceNode):
            return False
        return self.node_id == other.node_id
    
    def access_node(self):
        """Mark this node as being accessed for prediction/learning (legacy method)."""
        self.times_accessed += 1
        self.last_accessed = datetime.now()
        
        # Also activate the node using new neural-like system
        self.activate()
    
    def activate(self, strength: float = 1.0):
        """Activate this memory node - creates emergent 'working memory' effects."""
        current_time = time.time()
        
        # Increase activation level
        self.activation_level = min(2.0, self.activation_level + strength)
        
        # Reset recency bonus (fresh access)
        self.recency_bonus = 1.0
        
        # Update access tracking
        self.access_frequency += 1
        self.last_activation_time = current_time
        
        # Consolidation strengthening for frequently accessed memories
        if self.access_frequency > 5:
            consolidation_bonus = min(0.1, 1.0 / self.access_frequency)
            self.consolidation_strength = min(2.0, self.consolidation_strength + consolidation_bonus)
    
    def decay_over_time(self, time_step: float = 1.0):
        """Natural decay over time - creates emergent forgetting."""
        current_time = time.time()
        time_since_last_access = current_time - self.last_activation_time
        
        # Activation level decays (faster for unused memories)
        base_decay = 0.995 ** time_step
        recency_decay = 0.99 ** (time_since_last_access / 3600.0)  # Decay based on hours since access
        
        self.activation_level *= base_decay * recency_decay
        
        # Recency bonus decays faster
        self.recency_bonus *= (0.99 ** time_step)
        
        # Ensure minimums
        self.activation_level = max(0.0, self.activation_level)
        self.recency_bonus = max(0.0, self.recency_bonus)
    
    def get_accessibility(self) -> float:
        """Calculate how accessible this memory is - emergent 'working memory' phenomenon."""
        # Combine multiple factors that make memories more/less accessible
        base_accessibility = self.activation_level * self.recency_bonus
        
        # Frequently accessed memories are easier to access
        frequency_bonus = min(1.0, self.access_frequency / 10.0)
        
        # Consolidated memories are more stable
        consolidation_bonus = self.consolidation_strength * 0.5
        
        # Recent activations boost accessibility
        current_time = time.time()
        time_since_access = current_time - self.last_activation_time
        recency_factor = max(0.1, 1.0 - (time_since_access / 3600.0))  # Decay over hours
        
        total_accessibility = (base_accessibility + frequency_bonus + consolidation_bonus) * recency_factor
        
        return max(0.0, min(3.0, total_accessibility))
    
    def strengthen_connection(self, other_node_id: str, connection_strength: float):
        """Strengthen connection to another node - Hebbian learning."""
        current_strength = self.connection_weights.get(other_node_id, 0.0)
        
        # Hebbian rule: "neurons that fire together, wire together"
        new_strength = current_strength + (connection_strength * 0.1)
        self.connection_weights[other_node_id] = min(1.0, new_strength)
    
    def weaken_connection(self, other_node_id: str, decay_factor: float = 0.95):
        """Weaken connection to another node over time."""
        if other_node_id in self.connection_weights:
            self.connection_weights[other_node_id] *= decay_factor
            
            # Remove very weak connections
            if self.connection_weights[other_node_id] < 0.05:
                del self.connection_weights[other_node_id]
    
    def is_forgettable(self, forgetting_threshold: float = 0.1) -> bool:
        """Determine if this memory should be forgotten - emergent forgetting."""
        # Memories are forgettable if they are:
        # 1. Very low activation
        # 2. Rarely accessed
        # 3. Not well consolidated
        # 4. Haven't been accessed recently
        
        low_activation = self.activation_level < forgetting_threshold
        rarely_accessed = self.access_frequency < 2
        poorly_consolidated = self.consolidation_strength < 0.5
        
        current_time = time.time()
        old_memory = (current_time - self.last_activation_time) > 7200  # 2 hours
        
        return low_activation and rarely_accessed and poorly_consolidated and old_memory
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about this memory node."""
        current_time = time.time()
        return {
            'activation_level': self.activation_level,
            'recency_bonus': self.recency_bonus,
            'accessibility': self.get_accessibility(),
            'access_frequency': self.access_frequency,
            'consolidation_strength': self.consolidation_strength,
            'connection_count': len(self.connection_weights),
            'average_connection_strength': sum(self.connection_weights.values()) / max(1, len(self.connection_weights)),
            'time_since_last_access': current_time - self.last_activation_time,
            'prediction_error': self.prediction_error,
            'is_forgettable': self.is_forgettable()
        }
    
    def calculate_prediction_accuracy(self) -> float:
        """Calculate how accurate this node's prediction was (0.0 to 1.0)."""
        if not self.predicted_sensory or not self.actual_sensory:
            return 0.0
        
        if len(self.predicted_sensory) != len(self.actual_sensory):
            return 0.0
        
        # Calculate normalized accuracy based on prediction error
        max_possible_error = len(self.predicted_sensory) * 4.0  # Assuming sensor values range -2 to +2
        accuracy = max(0.0, 1.0 - (self.prediction_error / max_possible_error))
        return accuracy
    
    def is_similar_context(self, other_context: List[float], threshold: float = 0.7) -> bool:
        """Check if this node's context is similar to another context."""
        if len(self.mental_context) != len(other_context):
            return False
        
        # Handle empty contexts
        if len(self.mental_context) == 0:
            return True  # Empty contexts are considered similar
        
        # Calculate Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(self.mental_context, other_context)) ** 0.5
        max_possible_distance = (len(self.mental_context) * 4.0) ** 0.5
        
        # Convert to similarity
        similarity = max(0.0, 1.0 - (distance / max_possible_distance))
        return similarity >= threshold
    
    def merge_with(self, other_node: 'ExperienceNode', weight: float = 0.5) -> 'ExperienceNode':
        """
        Merge this node with another similar node using weighted averaging.
        Returns a new merged node.
        """
        # Weighted average of contexts
        new_context = []
        for i in range(len(self.mental_context)):
            new_val = (self.mental_context[i] * (1 - weight) + 
                      other_node.mental_context[i] * weight)
            new_context.append(new_val)
        
        # Weighted average of predictions
        new_predicted = []
        for i in range(len(self.predicted_sensory)):
            new_val = (self.predicted_sensory[i] * (1 - weight) + 
                      other_node.predicted_sensory[i] * weight)
            new_predicted.append(new_val)
        
        # Weighted average of actual sensory
        new_actual = []
        for i in range(len(self.actual_sensory)):
            new_val = (self.actual_sensory[i] * (1 - weight) + 
                      other_node.actual_sensory[i] * weight)
            new_actual.append(new_val)
        
        # Weighted average of actions
        new_action = {}
        for key in self.action_taken:
            if key in other_node.action_taken:
                new_action[key] = (self.action_taken[key] * (1 - weight) + 
                                 other_node.action_taken[key] * weight)
            else:
                new_action[key] = self.action_taken[key]
        
        # Add any actions only in other node
        for key in other_node.action_taken:
            if key not in new_action:
                new_action[key] = other_node.action_taken[key]
        
        # Create merged node
        merged_node = ExperienceNode(
            mental_context=new_context,
            action_taken=new_action,
            predicted_sensory=new_predicted,
            actual_sensory=new_actual,
            prediction_error=(self.prediction_error + other_node.prediction_error) / 2,
            strength=max(self.strength, other_node.strength) + 0.1,  # Merged nodes get slight boost
            times_accessed=self.times_accessed + other_node.times_accessed,
            merge_count=self.merge_count + other_node.merge_count + 1
        )
        
        # Combine similarity lists
        merged_node.similar_contexts = list(set(self.similar_contexts + other_node.similar_contexts))
        merged_node.prediction_sources = list(set(self.prediction_sources + other_node.prediction_sources))
        
        return merged_node