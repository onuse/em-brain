"""
Brain State Serializer

Handles conversion between brain objects and serializable formats.
Provides well-documented serialization for third-party analysis tools.
"""

import json
import gzip
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..experience.models import Experience
from .models import BrainCheckpoint


class BrainStateSerializer:
    """
    Serializes and deserializes brain state for persistence.
    
    Design principles:
    1. Human-readable JSON format for debugging and analysis
    2. Complete state capture for perfect restoration
    3. Efficient handling of NumPy arrays and complex objects
    4. Well-documented format for third-party tools
    """
    
    def __init__(self, use_compression: bool = True):
        """
        Initialize serializer.
        
        Args:
            use_compression: Whether to compress serialized data
        """
        self.use_compression = use_compression
    
    def serialize_experiences(self, experiences: Dict[str, Experience]) -> Dict[str, Dict[str, Any]]:
        """
        Serialize experience collection to dictionary format.
        
        Args:
            experiences: Dictionary of experience_id -> Experience objects
            
        Returns:
            Dictionary with experience data in serializable format
        """
        serialized = {}
        
        for exp_id, experience in experiences.items():
            # Use the existing to_dict method and enhance it
            exp_data = experience.to_dict()
            
            # Ensure all numpy arrays are converted to lists
            exp_data['sensory_input'] = self._serialize_vector(exp_data['sensory_input'])
            exp_data['action_taken'] = self._serialize_vector(exp_data['action_taken']) 
            exp_data['outcome'] = self._serialize_vector(exp_data['outcome'])
            
            # Add metadata for analysis
            exp_data['_metadata'] = {
                'serialized_at': experience.timestamp,
                'vector_dimensions': {
                    'sensory': len(exp_data['sensory_input']),
                    'action': len(exp_data['action_taken']),
                    'outcome': len(exp_data['outcome'])
                },
                'similarity_connections': len(exp_data.get('similar_experiences', {})),
                'is_active': exp_data.get('activation_level', 0) > 0
            }
            
            serialized[exp_id] = exp_data
        
        return serialized
    
    def deserialize_experiences(self, serialized_data: Dict[str, Dict[str, Any]]) -> Dict[str, Experience]:
        """
        Deserialize experiences from dictionary format.
        
        Args:
            serialized_data: Dictionary with serialized experience data
            
        Returns:
            Dictionary of experience_id -> Experience objects
        """
        experiences = {}
        
        for exp_id, exp_data in serialized_data.items():
            # Remove metadata before creating Experience
            clean_data = {k: v for k, v in exp_data.items() if not k.startswith('_')}
            
            # Convert lists back to numpy arrays if needed
            clean_data['sensory_input'] = self._deserialize_vector(clean_data['sensory_input'])
            clean_data['action_taken'] = self._deserialize_vector(clean_data['action_taken'])
            clean_data['outcome'] = self._deserialize_vector(clean_data['outcome'])
            
            # Create Experience object
            experience = Experience.from_dict(clean_data)
            experiences[exp_id] = experience
        
        return experiences
    
    def serialize_similarity_state(self, similarity_engine) -> Dict[str, Any]:
        """
        Serialize learnable similarity engine state.
        
        Args:
            similarity_engine: LearnableSimilarity instance
            
        Returns:
            Dictionary with serialized similarity state
        """
        if not hasattr(similarity_engine, 'feature_weights'):
            return {}
        
        state = {
            'feature_weights': self._serialize_vector(similarity_engine.feature_weights),
            'learning_rate': getattr(similarity_engine, 'learning_rate', 0.01),
            'adaptation_rate': getattr(similarity_engine, 'adaptation_rate', 0.001),
            'prediction_window_size': getattr(similarity_engine, 'prediction_window_size', 50),
            '_metadata': {
                'feature_count': len(similarity_engine.feature_weights) if similarity_engine.feature_weights is not None else 0,
                'has_interaction_matrix': hasattr(similarity_engine, 'interaction_matrix'),
                'total_adaptations': len(getattr(similarity_engine, 'adaptation_history', [])),
                'device': getattr(similarity_engine, 'device', 'cpu')
            }
        }
        
        # Include interaction matrix if it exists
        if hasattr(similarity_engine, 'interaction_matrix') and similarity_engine.interaction_matrix is not None:
            state['interaction_matrix'] = self._serialize_matrix(similarity_engine.interaction_matrix)
        
        # Include recent adaptation history
        if hasattr(similarity_engine, 'adaptation_history'):
            state['adaptation_history'] = list(similarity_engine.adaptation_history)[-50:]  # Last 50 adaptations
        
        return state
    
    def deserialize_similarity_state(self, state_data: Dict[str, Any], similarity_engine) -> None:
        """
        Restore similarity engine state from serialized data.
        
        Args:
            state_data: Serialized similarity state
            similarity_engine: LearnableSimilarity instance to restore
        """
        if not state_data:
            return
        
        # Restore feature weights
        if 'feature_weights' in state_data:
            similarity_engine.feature_weights = self._deserialize_vector(state_data['feature_weights'])
        
        # Restore parameters
        if 'learning_rate' in state_data:
            similarity_engine.learning_rate = state_data['learning_rate']
        if 'adaptation_rate' in state_data:
            similarity_engine.adaptation_rate = state_data['adaptation_rate']
        
        # Restore interaction matrix
        if 'interaction_matrix' in state_data:
            similarity_engine.interaction_matrix = self._deserialize_matrix(state_data['interaction_matrix'])
        
        # Restore adaptation history
        if 'adaptation_history' in state_data:
            similarity_engine.adaptation_history = list(state_data['adaptation_history'])
    
    def serialize_activation_state(self, activation_engine) -> Dict[str, Any]:
        """
        Serialize activation dynamics state.
        
        Args:
            activation_engine: ActivationDynamics instance
            
        Returns:
            Dictionary with activation state
        """
        state = {
            'base_decay_rate': getattr(activation_engine, 'base_decay_rate', 0.01),
            'spread_strength': getattr(activation_engine, 'spread_strength', 0.1),
            'min_activation_threshold': getattr(activation_engine, 'min_activation_threshold', 0.01),
            'adaptation_rate': getattr(activation_engine, 'adaptation_rate', 0.001),
            '_metadata': {
                'total_spreads': getattr(activation_engine, 'total_spreads', 0),
                'device': getattr(activation_engine, 'device', 'cpu'),
                'has_gpu_tensors': hasattr(activation_engine, '_gpu_activation_levels')
            }
        }
        
        # Include recent prediction errors if available
        if hasattr(activation_engine, 'recent_prediction_errors'):
            state['recent_prediction_errors'] = list(activation_engine.recent_prediction_errors)[-50:]
        
        return state
    
    def deserialize_activation_state(self, state_data: Dict[str, Any], activation_engine) -> None:
        """
        Restore activation engine state.
        
        Args:
            state_data: Serialized activation state
            activation_engine: ActivationDynamics instance to restore
        """
        if not state_data:
            return
        
        # Restore parameters
        for param in ['base_decay_rate', 'spread_strength', 'min_activation_threshold', 'adaptation_rate']:
            if param in state_data:
                setattr(activation_engine, param, state_data[param])
        
        # Restore recent errors
        if 'recent_prediction_errors' in state_data:
            activation_engine.recent_prediction_errors = list(state_data['recent_prediction_errors'])
    
    def serialize_prediction_state(self, prediction_engine) -> Dict[str, Any]:
        """
        Serialize prediction engine state.
        
        Args:
            prediction_engine: PredictionEngine instance
            
        Returns:
            Dictionary with prediction state
        """
        stats = getattr(prediction_engine, 'stats', {})
        
        state = {
            'performance_stats': dict(stats),
            'optimal_error_threshold': getattr(prediction_engine, 'optimal_error_threshold', 0.1),
            '_metadata': {
                'total_predictions': stats.get('total_predictions', 0),
                'accuracy': stats.get('accuracy', 0.0),
                'has_pattern_analyzer': hasattr(prediction_engine, 'pattern_analyzer')
            }
        }
        
        # Include pattern analyzer state if available
        if hasattr(prediction_engine, 'pattern_analyzer') and prediction_engine.pattern_analyzer:
            analyzer = prediction_engine.pattern_analyzer
            state['pattern_analyzer'] = {
                'device': getattr(analyzer, 'device', 'cpu'),
                'prediction_threshold': getattr(analyzer, 'prediction_threshold', 0.7),
                'pattern_count': len(getattr(analyzer, 'learned_patterns', []))
            }
        
        return state
    
    def deserialize_prediction_state(self, state_data: Dict[str, Any], prediction_engine) -> None:
        """
        Restore prediction engine state.
        
        Args:
            state_data: Serialized prediction state  
            prediction_engine: PredictionEngine instance to restore
        """
        if not state_data:
            return
        
        # Restore performance stats
        if 'performance_stats' in state_data:
            if not hasattr(prediction_engine, 'stats'):
                prediction_engine.stats = {}
            prediction_engine.stats.update(state_data['performance_stats'])
        
        # Restore optimal error threshold
        if 'optimal_error_threshold' in state_data:
            prediction_engine.optimal_error_threshold = state_data['optimal_error_threshold']
    
    def serialize_stream_state(self, stream_storage) -> Dict[str, Any]:
        """
        Serialize stream storage state.
        
        Args:
            stream_storage: StreamStorage instance
            
        Returns:
            Dictionary with stream state
        """
        if not hasattr(stream_storage, 'vectors'):
            return {}
        
        # Convert deque to list for serialization
        vectors_list = list(stream_storage.vectors)
        
        state = {
            'vectors': [
                {
                    'vector': self._serialize_vector(item['vector']),
                    'timestamp': item['timestamp'],
                    'experience_id': item.get('experience_id', '')
                }
                for item in vectors_list
            ],
            'max_size': getattr(stream_storage, 'max_size', 10000),
            '_metadata': {
                'current_size': len(vectors_list),
                'oldest_timestamp': vectors_list[0]['timestamp'] if vectors_list else None,
                'newest_timestamp': vectors_list[-1]['timestamp'] if vectors_list else None
            }
        }
        
        return state
    
    def deserialize_stream_state(self, state_data: Dict[str, Any], stream_storage) -> None:
        """
        Restore stream storage state.
        
        Args:
            state_data: Serialized stream state
            stream_storage: StreamStorage instance to restore
        """
        if not state_data or 'vectors' not in state_data:
            return
        
        # Clear existing data
        stream_storage.vectors.clear()
        
        # Restore vectors
        for item in state_data['vectors']:
            stream_storage.vectors.append({
                'vector': self._deserialize_vector(item['vector']),
                'timestamp': item['timestamp'],
                'experience_id': item.get('experience_id', '')
            })
    
    def save_checkpoint(self, checkpoint: BrainCheckpoint, file_path: Path) -> None:
        """
        Save checkpoint to file.
        
        Args:
            checkpoint: BrainCheckpoint to save
            file_path: Path to save checkpoint
        """
        # Convert to dictionary
        checkpoint_data = checkpoint.to_dict()
        
        # Serialize to JSON
        json_data = json.dumps(checkpoint_data, indent=2, default=self._json_serializer)
        
        # Save with optional compression
        if self.use_compression:
            with gzip.open(str(file_path) + '.gz', 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
    
    def load_checkpoint(self, file_path: Path) -> BrainCheckpoint:
        """
        Load checkpoint from file.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            Loaded BrainCheckpoint
        """
        # Check for compressed version first
        if (Path(str(file_path) + '.gz')).exists():
            with gzip.open(str(file_path) + '.gz', 'rt', encoding='utf-8') as f:
                json_data = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
        
        # Parse JSON
        checkpoint_data = json.loads(json_data)
        
        # Create checkpoint object
        return BrainCheckpoint.from_dict(checkpoint_data)
    
    def _serialize_vector(self, vector) -> List[float]:
        """Convert numpy array or list to list of floats."""
        if vector is None:
            return []
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return list(vector)
    
    def _deserialize_vector(self, vector_data: List[float]) -> np.ndarray:
        """Convert list of floats to numpy array."""
        return np.array(vector_data, dtype=np.float32)
    
    def _serialize_matrix(self, matrix) -> List[List[float]]:
        """Convert numpy matrix to nested list."""
        if matrix is None:
            return []
        if isinstance(matrix, np.ndarray):
            return matrix.tolist()
        return matrix
    
    def _deserialize_matrix(self, matrix_data: List[List[float]]) -> np.ndarray:
        """Convert nested list to numpy matrix."""
        return np.array(matrix_data, dtype=np.float32)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")