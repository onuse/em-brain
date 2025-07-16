"""
Stream to Experience Adapter

Bridges pure information streams with existing brain architecture.
Converts emergent patterns into experience-like structures that the
existing similarity, activation, and prediction systems can work with.

This allows Strategy 1 to coexist with the current system.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import uuid

from ..experience import Experience
from .pure_stream_storage import PureStreamStorage
from .pattern_discovery import PatternDiscovery


class StreamToExperienceAdapter:
    """
    Adapts emergent stream patterns to experience format.
    
    This is the bridge between:
    - Pure streams (no structure)
    - Emergent patterns (discovered structure)
    - Experience format (existing brain interface)
    """
    
    def __init__(self, stream_storage: PureStreamStorage, 
                 pattern_discovery: PatternDiscovery):
        """
        Initialize adapter.
        
        Args:
            stream_storage: The raw stream storage
            pattern_discovery: The pattern discovery system
        """
        self.stream_storage = stream_storage
        self.pattern_discovery = pattern_discovery
        
        # Cache of adapted experiences
        self.adapted_experiences = {}
        self.adaptation_count = 0
        
        print("StreamToExperienceAdapter initialized - bridging emergence with architecture")
    
    def get_emergent_experiences(self) -> Dict[str, Experience]:
        """
        Convert discovered patterns into experience format.
        
        This is where emergent structure meets existing architecture.
        
        Returns:
            Dictionary of experience_id -> Experience objects
        """
        # Get current emergent structure
        structure = self.pattern_discovery.get_emergent_structure()
        
        # Get the raw stream for conversion
        stream = self.stream_storage.get_stream_window()
        
        if not stream:
            return {}
        
        # Convert emergent experiences
        experiences = {}
        
        for emergent_exp in structure['emergent_experiences']:
            start_idx = emergent_exp['start']
            end_idx = emergent_exp['end']
            
            # Extract vectors for this emergent experience
            if 0 <= start_idx < len(stream) and start_idx < end_idx <= len(stream):
                exp_vectors = [stream[i]['vector'] for i in range(start_idx, end_idx)]
                
                # Create experience from emergent pattern
                experience = self._create_experience_from_pattern(
                    exp_vectors, start_idx, end_idx, stream
                )
                
                if experience:
                    experiences[experience.experience_id] = experience
                    self.adapted_experiences[experience.experience_id] = experience
                    self.adaptation_count += 1
        
        return experiences
    
    def _create_experience_from_pattern(self, vectors: List[np.ndarray],
                                      start_idx: int, end_idx: int,
                                      stream: List[Dict]) -> Optional[Experience]:
        """
        Create an experience from an emergent pattern.
        
        This interprets the pattern to extract:
        - Sensory input (beginning of pattern)
        - Action taken (middle transition)
        - Outcome (end of pattern)
        
        All emergent, not engineered!
        """
        if len(vectors) < 3:
            return None  # Need at least 3 vectors for input->action->outcome
        
        # Interpret pattern structure (this emerges from prediction boundaries)
        # Beginning vectors likely represent sensory state
        sensory_vectors = vectors[:len(vectors)//3]
        sensory_input = np.mean(sensory_vectors, axis=0) if sensory_vectors else vectors[0]
        
        # Middle vectors likely represent transition/action
        action_vectors = vectors[len(vectors)//3:2*len(vectors)//3]
        action_taken = np.mean(action_vectors, axis=0) if action_vectors else vectors[len(vectors)//2]
        
        # End vectors likely represent outcome
        outcome_vectors = vectors[2*len(vectors)//3:]
        outcome = np.mean(outcome_vectors, axis=0) if outcome_vectors else vectors[-1]
        
        # Compute emergent prediction error
        # (How predictable was the outcome from the input?)
        predicted_outcome = (sensory_input + action_taken) / 2  # Simple prediction
        prediction_error = np.linalg.norm(outcome - predicted_outcome) / (np.linalg.norm(outcome) + 0.001)
        
        # Get timestamp from stream
        timestamp = stream[start_idx]['timestamp']
        
        # Create experience with emergent structure
        experience = Experience(
            sensory_input=sensory_input.tolist(),
            action_taken=action_taken.tolist(),
            outcome=outcome.tolist(),
            prediction_error=float(prediction_error),
            timestamp=timestamp
        )
        
        # Mark as emergent
        experience.metadata = {
            'emergent': True,
            'stream_start': start_idx,
            'stream_end': end_idx,
            'pattern_length': len(vectors)
        }
        
        return experience
    
    def adapt_stream_window(self, window_size: int = 100) -> Dict[str, Experience]:
        """
        Adapt a recent window of the stream into experiences.
        
        Args:
            window_size: Size of stream window to analyze
            
        Returns:
            Dictionary of newly discovered experiences
        """
        # Get recent stream window
        recent_stream = self.stream_storage.get_stream_window(-window_size)
        if not recent_stream:
            return {}
        
        # Extract vectors for analysis
        vectors = [entry['vector'] for entry in recent_stream]
        
        # Run pattern discovery
        analysis = self.pattern_discovery.analyze_stream_segment(vectors)
        
        # Get emergent experiences
        return self.get_emergent_experiences()
    
    def stream_to_experience_vector(self, stream_position: int) -> Optional[List[float]]:
        """
        Convert a stream position to an experience-like context vector.
        
        This allows the existing similarity system to work with stream data.
        
        Args:
            stream_position: Position in the stream
            
        Returns:
            Context vector or None
        """
        # Get temporal context around this position
        context_size = 5
        start = max(0, stream_position - context_size)
        end = stream_position + context_size
        
        window = self.stream_storage.get_stream_window(start, end)
        if not window:
            return None
        
        # Create context vector from temporal window
        vectors = [entry['vector'] for entry in window]
        
        # Flatten temporal window into single context vector
        if len(vectors) >= 3:
            # Past, present, future components
            past = np.mean(vectors[:len(vectors)//3], axis=0)
            present = vectors[len(vectors)//2]
            future = np.mean(vectors[2*len(vectors)//3:], axis=0)
            
            # Concatenate temporal components
            context = np.concatenate([past, present, future])
            return context.tolist()
        elif vectors:
            # Just use available vectors
            return np.concatenate(vectors).tolist()
        
        return None
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about stream-to-experience adaptation."""
        
        stream_stats = self.stream_storage.compute_stream_statistics()
        pattern_stats = self.pattern_discovery.get_emergent_structure()
        
        return {
            'adaptation_count': self.adaptation_count,
            'cached_experiences': len(self.adapted_experiences),
            'stream_statistics': stream_stats,
            'pattern_statistics': pattern_stats['emergence_statistics'],
            'emergent_experiences': len(pattern_stats['emergent_experiences']),
            'discovered_motifs': len(pattern_stats['behavioral_motifs']),
            'adaptation_rate': self.adaptation_count / max(1, stream_stats['total_vectors'])
        }
    
    def reset_adapter(self):
        """Reset the adapter state."""
        self.adapted_experiences.clear()
        self.adaptation_count = 0
        print("Stream adapter reset")