"""
Pure Stream Storage

Raw temporal vector storage without any structure assumptions.
No concepts of "experience", "action", "outcome" - just continuous data flow.

The brain stores vectors with timestamps and lets patterns emerge through
prediction success rather than engineering structure.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import uuid


class PureStreamStorage:
    """
    Storage for raw temporal vector sequences.
    
    No structure imposed - just vectors flowing through time.
    Structure emerges from what predicts what, not from engineering.
    """
    
    def __init__(self, max_stream_length: int = 10000):
        """
        Initialize pure stream storage.
        
        Args:
            max_stream_length: Maximum vectors to keep in stream (memory limit)
        """
        self.max_stream_length = max_stream_length
        
        # The raw stream - just vectors and timestamps
        self.stream = deque(maxlen=max_stream_length)
        
        # Minimal metadata for analysis
        self.stream_start_time = time.time()
        self.total_vectors_seen = 0
        
        print("PureStreamStorage initialized - no structure, just flow")
    
    def append_vector(self, vector: List[float], timestamp: Optional[float] = None) -> str:
        """
        Append a raw vector to the stream.
        
        No assumptions about what this vector represents - could be sensory,
        could be motor, could be anything. Structure emerges later.
        
        Args:
            vector: Raw data vector
            timestamp: When this occurred (auto-generated if None)
            
        Returns:
            Unique ID for this stream position
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Generate unique ID for this position in stream
        stream_id = str(uuid.uuid4())[:8]
        
        # Store just the raw data
        stream_entry = {
            'id': stream_id,
            'vector': np.array(vector),
            'timestamp': timestamp,
            'index': self.total_vectors_seen
        }
        
        self.stream.append(stream_entry)
        self.total_vectors_seen += 1
        
        return stream_id
    
    def get_stream_window(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Dict]:
        """
        Get a window of the raw stream.
        
        Args:
            start_idx: Start index (negative for from-end indexing)
            end_idx: End index (None for current end)
            
        Returns:
            List of stream entries in temporal order
        """
        if end_idx is None:
            return list(self.stream)[start_idx:]
        return list(self.stream)[start_idx:end_idx]
    
    def get_recent_vectors(self, count: int = 10) -> List[np.ndarray]:
        """
        Get the most recent vectors from the stream.
        
        Args:
            count: Number of recent vectors to return
            
        Returns:
            List of recent vectors (newest last)
        """
        recent_entries = list(self.stream)[-count:]
        return [entry['vector'] for entry in recent_entries]
    
    def get_temporal_patterns(self, window_size: int = 5) -> List[List[np.ndarray]]:
        """
        Extract overlapping temporal windows for pattern analysis.
        
        This is how the system starts to discover structure - by looking
        at temporal patterns in the raw stream.
        
        Args:
            window_size: Size of temporal windows
            
        Returns:
            List of vector sequences
        """
        if len(self.stream) < window_size:
            return []
        
        patterns = []
        stream_list = list(self.stream)
        
        for i in range(len(stream_list) - window_size + 1):
            window = stream_list[i:i + window_size]
            pattern = [entry['vector'] for entry in window]
            patterns.append(pattern)
        
        return patterns
    
    def find_vector_by_id(self, stream_id: str) -> Optional[Dict]:
        """
        Find a specific vector by its stream ID.
        
        Args:
            stream_id: Unique ID of stream entry
            
        Returns:
            Stream entry or None if not found
        """
        for entry in self.stream:
            if entry['id'] == stream_id:
                return entry
        return None
    
    def compute_stream_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the raw stream.
        
        These stats help understand what patterns might be emerging.
        """
        if not self.stream:
            return {
                'total_vectors': 0,
                'stream_duration': 0.0,
                'vectors_per_second': 0.0,
                'vector_dimensions': 0
            }
        
        vectors = [entry['vector'] for entry in self.stream]
        timestamps = [entry['timestamp'] for entry in self.stream]
        
        # Basic statistics
        stream_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Vector statistics
        vector_dims = len(vectors[0]) if vectors else 0
        vector_array = np.array(vectors)
        
        stats = {
            'total_vectors': self.total_vectors_seen,
            'current_stream_length': len(self.stream),
            'stream_duration': stream_duration,
            'vectors_per_second': len(self.stream) / max(0.001, stream_duration),
            'vector_dimensions': vector_dims,
            'vector_stats': {
                'mean': np.mean(vector_array, axis=0).tolist() if len(vectors) > 0 else [],
                'std': np.std(vector_array, axis=0).tolist() if len(vectors) > 0 else [],
                'min': np.min(vector_array, axis=0).tolist() if len(vectors) > 0 else [],
                'max': np.max(vector_array, axis=0).tolist() if len(vectors) > 0 else []
            }
        }
        
        # Temporal statistics
        if len(timestamps) > 1:
            time_deltas = np.diff(timestamps)
            stats['temporal_stats'] = {
                'mean_delta': np.mean(time_deltas),
                'std_delta': np.std(time_deltas),
                'min_delta': np.min(time_deltas),
                'max_delta': np.max(time_deltas)
            }
        
        return stats
    
    def get_stream_segment(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Get stream segment by timestamp range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of stream entries in time range
        """
        segment = []
        for entry in self.stream:
            if start_time <= entry['timestamp'] <= end_time:
                segment.append(entry)
        return segment
    
    def clear_stream(self):
        """Clear the entire stream (use with caution)."""
        self.stream.clear()
        self.total_vectors_seen = 0
        self.stream_start_time = time.time()
        print("Stream cleared - starting fresh")
    
    def __len__(self) -> int:
        """Get current stream length."""
        return len(self.stream)
    
    def __str__(self) -> str:
        return f"PureStreamStorage({len(self.stream)} vectors, {self.total_vectors_seen} total seen)"