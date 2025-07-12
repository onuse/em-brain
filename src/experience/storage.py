"""
Experience Storage System

Stores and manages all experiences. This is the unified memory where
everything the brain learns is kept as raw sensory-motor records.

No categories, no organization, just the chronological stream of experience.
"""

from typing import List, Dict, Optional, Iterator, Tuple
import time
from collections import defaultdict

from .models import Experience


class ExperienceStorage:
    """
    Simple storage for all experiences.
    
    This is intentionally basic - just a list of experiences with
    fast lookup by ID. All intelligence emerges from similarity search
    and activation dynamics, not from sophisticated storage.
    """
    
    def __init__(self):
        # Core storage
        self._experiences: Dict[str, Experience] = {}  # experience_id -> Experience
        self._chronological_order: List[str] = []      # IDs in time order
        
        # Performance tracking
        self._total_added = 0
        self._total_memory_bytes = 0
        self._last_cleanup_time = time.time()
        
        # Activation tracking for working memory
        self._activation_decay_rate = 0.01  # Per second
        self._last_decay_time = time.time()
        
        print("💾 ExperienceStorage initialized")
    
    def add_experience(self, experience: Experience) -> str:
        """
        Add a new experience to storage.
        
        Args:
            experience: The experience to store
            
        Returns:
            The experience ID
        """
        experience_id = experience.experience_id
        
        # Store the experience
        self._experiences[experience_id] = experience
        self._chronological_order.append(experience_id)
        
        # Update tracking
        self._total_added += 1
        self._total_memory_bytes += experience.get_memory_size()
        
        # Natural decay of all activations when adding new experiences
        self._decay_activations()
        
        return experience_id
    
    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """
        Retrieve an experience by ID.
        
        Args:
            experience_id: The experience to retrieve
            
        Returns:
            The experience, or None if not found
        """
        experience = self._experiences.get(experience_id)
        if experience:
            # Activate when accessed (working memory effect)
            experience.activate(strength=0.1)
        return experience
    
    def get_recent_experiences(self, count: int = 10) -> List[Experience]:
        """
        Get the most recent experiences.
        
        Args:
            count: Number of recent experiences to return
            
        Returns:
            List of recent experiences (most recent first)
        """
        recent_ids = self._chronological_order[-count:]
        recent_ids.reverse()  # Most recent first
        
        experiences = []
        for exp_id in recent_ids:
            exp = self.get_experience(exp_id)
            if exp:
                experiences.append(exp)
        
        return experiences
    
    def get_all_experiences(self) -> Iterator[Experience]:
        """
        Iterate through all experiences.
        
        Yields:
            Each experience in chronological order
        """
        for exp_id in self._chronological_order:
            exp = self._experiences.get(exp_id)
            if exp:
                yield exp
    
    def get_activated_experiences(self, min_activation: float = 0.1) -> List[Experience]:
        """
        Get currently activated experiences (working memory).
        
        Args:
            min_activation: Minimum activation level to include
            
        Returns:
            List of activated experiences, sorted by activation level
        """
        activated = []
        for experience in self._experiences.values():
            if experience.activation_level >= min_activation:
                activated.append(experience)
        
        # Sort by activation level (highest first)
        activated.sort(key=lambda exp: exp.activation_level, reverse=True)
        return activated
    
    def get_experience_vectors(self, experience_ids: List[str]) -> List[List[float]]:
        """
        Get context vectors for a list of experience IDs.
        
        Args:
            experience_ids: List of experience IDs
            
        Returns:
            List of context vectors for similarity search
        """
        vectors = []
        for exp_id in experience_ids:
            exp = self.get_experience(exp_id)
            if exp:
                vectors.append(exp.get_context_vector())
        return vectors
    
    def find_experiences_by_time_range(self, start_time: float, end_time: float) -> List[Experience]:
        """
        Find experiences within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of experiences in time range
        """
        experiences = []
        for experience in self._experiences.values():
            if start_time <= experience.timestamp <= end_time:
                experiences.append(experience)
        
        # Sort by timestamp
        experiences.sort(key=lambda exp: exp.timestamp)
        return experiences
    
    def size(self) -> int:
        """Get total number of stored experiences."""
        return len(self._experiences)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return {
            'total_experiences': len(self._experiences),
            'estimated_bytes': self._total_memory_bytes,
            'estimated_mb': self._total_memory_bytes / (1024 * 1024),
            'activated_experiences': len(self.get_activated_experiences()),
            'avg_bytes_per_experience': self._total_memory_bytes / max(1, len(self._experiences))
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive storage statistics."""
        activated = self.get_activated_experiences()
        
        # Calculate prediction error statistics
        errors = [exp.prediction_error for exp in self._experiences.values()]
        avg_error = sum(errors) / len(errors) if errors else 0.0
        
        # Calculate access patterns
        access_counts = [exp.access_count for exp in self._experiences.values()]
        total_accesses = sum(access_counts)
        
        return {
            'storage': self.get_memory_usage(),
            'total_added': self._total_added,
            'working_memory_size': len(activated),
            'avg_prediction_error': avg_error,
            'total_accesses': total_accesses,
            'avg_accesses_per_experience': total_accesses / max(1, len(self._experiences)),
            'oldest_timestamp': min((exp.timestamp for exp in self._experiences.values()), default=0),
            'newest_timestamp': max((exp.timestamp for exp in self._experiences.values()), default=0)
        }
    
    def _decay_activations(self):
        """
        Natural decay of activation levels over time.
        
        This creates working memory effects - recently accessed experiences
        stay activated longer than old ones.
        """
        current_time = time.time()
        time_since_last_decay = current_time - self._last_decay_time
        
        if time_since_last_decay > 1.0:  # Decay every second
            decay_amount = self._activation_decay_rate * time_since_last_decay
            
            for experience in self._experiences.values():
                experience.decay_activation(decay_amount)
            
            self._last_decay_time = current_time
    
    def cleanup_weak_experiences(self, min_access_count: int = 1, max_age_hours: float = 24.0):
        """
        Optional cleanup of very old, unused experiences.
        
        This is only for memory management - the brain should work with
        unlimited experience storage in principle.
        
        Args:
            min_access_count: Minimum access count to keep
            max_age_hours: Maximum age in hours to keep
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for exp_id, experience in self._experiences.items():
            age = current_time - experience.timestamp
            if (experience.access_count < min_access_count and 
                age > max_age_seconds):
                to_remove.append(exp_id)
        
        # Remove old experiences
        for exp_id in to_remove:
            exp = self._experiences.pop(exp_id, None)
            if exp:
                self._total_memory_bytes -= exp.get_memory_size()
            
            # Remove from chronological order
            if exp_id in self._chronological_order:
                self._chronological_order.remove(exp_id)
        
        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} old experiences")
        
        self._last_cleanup_time = current_time
    
    def clear(self):
        """Clear all stored experiences (for testing)."""
        self._experiences.clear()
        self._chronological_order.clear()
        self._total_added = 0
        self._total_memory_bytes = 0
        print("🧹 ExperienceStorage cleared")
    
    def __len__(self) -> int:
        """Number of stored experiences."""
        return len(self._experiences)
    
    def __contains__(self, experience_id: str) -> bool:
        """Check if experience ID exists in storage."""
        return experience_id in self._experiences
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"ExperienceStorage({len(self._experiences)} experiences, "
                f"{self._total_memory_bytes / 1024:.1f}KB)")
    
    def __repr__(self) -> str:
        return self.__str__()