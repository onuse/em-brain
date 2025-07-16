"""
Working Memory Buffer

A non-overwriting queue of recent experiences that participates actively in 
brain processing. This represents the brain's working memory - experiences that
are immediately accessible for reasoning before consolidation to long-term storage.

Biological inspiration:
- Working memory holds ~7±2 items/chunks
- Immediately accessible for current thinking
- Higher activation than long-term memories
- Doesn't require consolidation to be useful
"""

import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import numpy as np


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    experience_id: str
    sensory_input: np.ndarray
    action_taken: np.ndarray
    outcome: Optional[np.ndarray]  # May not have outcome yet
    predicted_action: Optional[np.ndarray]
    timestamp: float
    activation_level: float = 1.0  # Starts high, decays over time
    access_count: int = 0  # Track how often accessed
    consolidation_eligible_time: Optional[float] = None  # When this can be consolidated
    
    def get_context_vector(self) -> np.ndarray:
        """Get combined context vector for similarity matching."""
        # Same format as long-term experiences
        return np.concatenate([self.sensory_input, self.action_taken])
    
    def decay_activation(self, decay_rate: float = 0.995):
        """Apply time-based decay to activation level."""
        self.activation_level *= decay_rate
    
    def boost_activation(self, boost: float = 0.1):
        """Boost activation when accessed."""
        self.activation_level = min(1.0, self.activation_level + boost)
        self.access_count += 1


class WorkingMemoryBuffer:
    """
    Working memory buffer for recent experiences.
    
    Key features:
    - Non-overwriting queue (preserves all recent experiences)
    - Participates in similarity search and prediction
    - Higher activation weights for recent experiences
    - Separate from long-term consolidation
    """
    
    def __init__(self, 
                 capacity: int = 50,  # Biological: ~7±2 chunks, but chunks can be complex
                 decay_rate: float = 0.995,
                 recency_boost: float = 0.2):
        """
        Initialize working memory buffer.
        
        Args:
            capacity: Maximum number of experiences to hold
            decay_rate: How fast activation decays (per cycle)
            recency_boost: Activation boost for recent items
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.recency_boost = recency_boost
        
        # Thread-safe queue of experiences
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.RLock()
        
        # Track statistics
        self.total_added = 0
        self.total_accessed = 0
        self.consolidation_queue = deque()  # Experiences ready for consolidation
        
        print(f"🧠 WorkingMemoryBuffer initialized")
        print(f"   Capacity: {capacity} experiences")
        print(f"   Decay rate: {decay_rate}")
        print(f"   Acts as active working memory for immediate reasoning")
    
    def add_experience(self, 
                      experience_id: str,
                      sensory_input: List[float],
                      action_taken: List[float],
                      outcome: Optional[List[float]] = None,
                      predicted_action: Optional[List[float]] = None) -> bool:
        """
        Add new experience to working memory.
        
        This makes the experience immediately available for reasoning,
        before it's consolidated to long-term storage.
        """
        with self.lock:
            # Convert to numpy arrays
            sensory_array = np.array(sensory_input, dtype=np.float32)
            action_array = np.array(action_taken, dtype=np.float32)
            outcome_array = np.array(outcome, dtype=np.float32) if outcome else None
            predicted_array = np.array(predicted_action, dtype=np.float32) if predicted_action else None
            
            # Create working memory item
            item = WorkingMemoryItem(
                experience_id=experience_id,
                sensory_input=sensory_array,
                action_taken=action_array,
                outcome=outcome_array,
                predicted_action=predicted_array,
                timestamp=time.time(),
                activation_level=1.0,  # Start with full activation
                consolidation_eligible_time=None  # Real experiences can consolidate immediately
            )
            
            # Add to buffer (automatically removes oldest if at capacity)
            self.buffer.append(item)
            self.total_added += 1
            
            # Apply decay to all items
            self._apply_activation_decay()
            
            return True
    
    def add_predicted_experience(self, 
                               predicted_experience: Dict[str, Any]) -> bool:
        """
        Add a predicted experience to working memory.
        
        This allows predictions to participate in reasoning alongside real experiences.
        Creates the emergent chaining effect where predictions naturally trigger more predictions.
        """
        with self.lock:
            # Convert to numpy arrays
            sensory_array = np.array(predicted_experience['sensory_input'], dtype=np.float32)
            action_array = np.array(predicted_experience['action_taken'], dtype=np.float32)
            outcome_array = np.array(predicted_experience['outcome'], dtype=np.float32) if predicted_experience['outcome'] else None
            predicted_array = np.array(predicted_experience['action_taken'], dtype=np.float32)
            
            # Create working memory item with individual consolidation timing
            item = WorkingMemoryItem(
                experience_id=predicted_experience['experience_id'],
                sensory_input=sensory_array,
                action_taken=action_array,
                outcome=outcome_array,
                predicted_action=predicted_array,
                timestamp=predicted_experience['prediction_time'],
                activation_level=predicted_experience.get('prediction_confidence', 1.0),
                consolidation_eligible_time=predicted_experience.get('consolidation_eligible_time')
            )
            
            # Add to buffer
            self.buffer.append(item)
            self.total_added += 1
            
            # Apply decay to all items
            self._apply_activation_decay()
            
            return True
    
    def get_experiences_for_matching(self, 
                                   n: Optional[int] = None,
                                   min_activation: float = 0.1) -> List[Tuple[WorkingMemoryItem, float]]:
        """
        Get working memory experiences for similarity matching.
        
        Returns experiences with their activation weights for use in
        similarity search alongside long-term memories.
        
        Returns:
            List of (experience, weight) tuples
        """
        with self.lock:
            active_items = []
            
            for item in self.buffer:
                if item.activation_level >= min_activation:
                    # Boost recent items
                    recency_factor = 1.0 + self.recency_boost * item.activation_level
                    weight = item.activation_level * recency_factor
                    
                    active_items.append((item, weight))
                    item.boost_activation(0.05)  # Small boost for being accessed
                    self.total_accessed += 1
            
            # Sort by weight (highest first)
            active_items.sort(key=lambda x: x[1], reverse=True)
            
            # Return top n if specified
            if n is not None:
                return active_items[:n]
            return active_items
    
    def get_recent_experiences(self, n: int = 10) -> List[WorkingMemoryItem]:
        """Get n most recent experiences regardless of activation."""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def mark_for_consolidation(self, experience_ids: List[str]):
        """Mark experiences as ready for consolidation to long-term storage."""
        with self.lock:
            for item in self.buffer:
                if item.experience_id in experience_ids:
                    self.consolidation_queue.append(item)
    
    def get_consolidation_batch(self, batch_size: int = 10) -> List[WorkingMemoryItem]:
        """Get batch of experiences ready for consolidation."""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.consolidation_queue))):
                if self.consolidation_queue:
                    batch.append(self.consolidation_queue.popleft())
            return batch
    
    def _apply_activation_decay(self):
        """Apply time-based decay to all items."""
        for item in self.buffer:
            item.decay_activation(self.decay_rate)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self.lock:
            if self.buffer:
                activations = [item.activation_level for item in self.buffer]
                avg_activation = sum(activations) / len(activations)
                access_counts = [item.access_count for item in self.buffer]
                avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
            else:
                avg_activation = 0.0
                avg_access_count = 0.0
            
            return {
                'capacity': self.capacity,
                'current_size': len(self.buffer),
                'total_added': self.total_added,
                'total_accessed': self.total_accessed,
                'avg_activation': avg_activation,
                'avg_access_count': avg_access_count,
                'consolidation_pending': len(self.consolidation_queue),
                'utilization': len(self.buffer) / self.capacity
            }
    
    def clear(self):
        """Clear working memory (e.g., during sleep/reset)."""
        with self.lock:
            # Move all to consolidation queue first
            self.consolidation_queue.extend(self.buffer)
            self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current number of items in working memory."""
        return len(self.buffer)
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"WorkingMemory({stats['current_size']}/{stats['capacity']} items, {stats['avg_activation']:.2f} avg activation)"


# Integration helper
def create_working_memory(capacity: int = 50) -> WorkingMemoryBuffer:
    """Create working memory buffer with standard configuration."""
    return WorkingMemoryBuffer(
        capacity=capacity,
        decay_rate=0.995,  # Gentle decay
        recency_boost=0.2  # 20% boost for recent access
    )