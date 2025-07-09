"""
Vectorized Backend - GPU-native storage and operations for WorldGraph.

This module contains all GPU programming complexity, providing a clean interface
that the rest of the system can use without knowing about tensors or CUDA.

Key Design Principles:
1. Hide all GPU complexity behind simple Python methods
2. Maintain exact compatibility with existing WorldGraph API
3. Provide fallback to CPU if GPU unavailable
4. Use lazy initialization to avoid startup overhead
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from collections import defaultdict

from core.experience_node import ExperienceNode


@dataclass
class VectorizedExperience:
    """
    Lightweight wrapper representing an experience in vectorized storage.
    
    This maintains the same interface as ExperienceNode while the actual
    data lives in GPU tensors for maximum performance.
    """
    index: int                    # Position in vectorized arrays
    backend: 'VectorizedBackend'  # Reference to the tensor storage
    
    @property
    def mental_context(self) -> List[float]:
        """Get mental context from vectorized storage."""
        return self.backend.get_mental_context(self.index)
    
    @property
    def action_taken(self) -> Dict[str, float]:
        """Get action from vectorized storage."""
        return self.backend.get_action_taken(self.index)
    
    @property
    def strength(self) -> float:
        """Get strength from vectorized storage."""
        return self.backend.get_strength(self.index)
    
    @property
    def actual_sensory(self) -> List[float]:
        """Get actual sensory from vectorized storage."""
        return self.backend.get_actual_sensory(self.index)
    
    @property
    def predicted_sensory(self) -> List[float]:
        """Get predicted sensory from vectorized storage."""
        return self.backend.get_predicted_sensory(self.index)
    
    @property
    def prediction_error(self) -> float:
        """Get prediction error from vectorized storage."""
        return self.backend.get_prediction_error(self.index)
    
    @property
    def connection_weights(self) -> Dict[str, float]:
        """Get connections from sparse matrix storage."""
        return self.backend.get_connection_weights(self.index)
    
    def to_experience_node(self) -> ExperienceNode:
        """Convert back to traditional ExperienceNode if needed."""
        return self.backend.get_experience_node(self.index)


class VectorizedBackend:
    """
    GPU-native storage backend for WorldGraph experiences.
    
    This class handles all the complex tensor operations and GPU memory management,
    providing a simple interface for the rest of the system.
    """
    
    def __init__(self, initial_capacity: int = 10000, device: str = 'auto'):
        """
        Initialize vectorized backend.
        
        Args:
            initial_capacity: Pre-allocate space for this many experiences
            device: 'auto', 'gpu', 'cpu', or specific device like 'cuda:0'
        """
        self.device = self._setup_device(device)
        self.capacity = initial_capacity
        self.size = 0  # Current number of experiences
        
        # Core tensor dimensions (will be set when first experience added)
        self.context_dim = None
        self.action_dim = None
        self.sensory_dim = None
        
        # Pre-allocated tensors (created lazily)
        self._mental_contexts = None      # [capacity, context_dim]
        self._action_vectors = None       # [capacity, action_dim]
        self._strength_values = None      # [capacity]
        self._actual_sensory = None       # [capacity, sensory_dim]
        self._predicted_sensory = None    # [capacity, sensory_dim]
        self._prediction_errors = None    # [capacity]
        
        # Sparse connection matrix
        self._connection_indices = None   # COO format indices
        self._connection_values = None    # COO format values
        self._connection_shape = None     # (capacity, capacity)
        
        # Index mapping for backward compatibility
        self._node_id_to_index = {}       # Map original node IDs to tensor indices
        self._index_to_node_id = {}       # Reverse mapping
        
        # Performance tracking
        self.stats = {
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Only print if using GPU (the interesting case)
        if str(self.device) != 'cpu':
            print(f"GPU Backend: {self.device} storage active")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device with automatic fallback."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        try:
            torch_device = torch.device(device)
            # Test device is actually available
            test_tensor = torch.zeros(1, device=torch_device)
            del test_tensor
            return torch_device
        except Exception as e:
            warnings.warn(f"Could not use device {device}: {e}. Falling back to CPU.")
            return torch.device('cpu')
    
    def _ensure_capacity(self, new_size: int):
        """Ensure tensors have capacity for new_size experiences."""
        if new_size <= self.capacity:
            return
        
        # Grow capacity by 50% or to new_size, whichever is larger
        new_capacity = max(int(self.capacity * 1.5), new_size)
        
        print(f"Expanding vectorized backend: {self.capacity} → {new_capacity}")
        
        # Resize all tensors
        if self._mental_contexts is not None:
            self._mental_contexts = self._resize_tensor(self._mental_contexts, new_capacity)
        if self._action_vectors is not None:
            self._action_vectors = self._resize_tensor(self._action_vectors, new_capacity)
        if self._strength_values is not None:
            self._strength_values = self._resize_tensor(self._strength_values, new_capacity)
        if self._actual_sensory is not None:
            self._actual_sensory = self._resize_tensor(self._actual_sensory, new_capacity)
        if self._predicted_sensory is not None:
            self._predicted_sensory = self._resize_tensor(self._predicted_sensory, new_capacity)
        if self._prediction_errors is not None:
            self._prediction_errors = self._resize_tensor(self._prediction_errors, new_capacity)
        
        self.capacity = new_capacity
    
    def _resize_tensor(self, tensor: torch.Tensor, new_capacity: int) -> torch.Tensor:
        """Resize a tensor while preserving existing data."""
        if len(tensor.shape) == 1:
            new_tensor = torch.zeros(new_capacity, dtype=tensor.dtype, device=self.device)
            new_tensor[:tensor.shape[0]] = tensor
        else:
            new_shape = (new_capacity,) + tensor.shape[1:]
            new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=self.device)
            new_tensor[:tensor.shape[0]] = tensor
        return new_tensor
    
    def add_experience(self, experience: ExperienceNode) -> int:
        """
        Add experience to vectorized storage.
        
        Returns:
            Index of the added experience in the vectorized arrays
        """
        # Initialize dimensions on first experience
        if self.context_dim is None:
            self._initialize_dimensions(experience)
        
        # Ensure capacity
        self._ensure_capacity(self.size + 1)
        
        # Convert experience to tensors and store
        index = self.size
        
        # Mental context
        context_tensor = torch.tensor(experience.mental_context, 
                                    dtype=torch.float32, device=self.device)
        self._mental_contexts[index, :len(experience.mental_context)] = context_tensor
        
        # Action (convert dict to vector)
        action_vector = self._dict_to_vector(experience.action_taken, self.action_dim)
        self._action_vectors[index] = action_vector
        
        # Scalar values
        self._strength_values[index] = experience.strength
        self._prediction_errors[index] = experience.prediction_error
        
        # Sensory data
        if experience.actual_sensory:
            actual_tensor = torch.tensor(experience.actual_sensory, 
                                       dtype=torch.float32, device=self.device)
            self._actual_sensory[index, :len(experience.actual_sensory)] = actual_tensor
        
        if experience.predicted_sensory:
            predicted_tensor = torch.tensor(experience.predicted_sensory, 
                                          dtype=torch.float32, device=self.device)
            self._predicted_sensory[index, :len(experience.predicted_sensory)] = predicted_tensor
        
        # Update mappings
        node_id = getattr(experience, 'node_id', f"node_{index}")
        self._node_id_to_index[node_id] = index
        self._index_to_node_id[index] = node_id
        
        self.size += 1
        return index
    
    def _initialize_dimensions(self, first_experience: ExperienceNode):
        """Initialize tensor dimensions based on first experience."""
        self.context_dim = len(first_experience.mental_context)
        self.action_dim = 8  # Standard action vector size (expandable)
        self.sensory_dim = len(first_experience.actual_sensory) if first_experience.actual_sensory else 8
        
        # Initialize tensors
        self._mental_contexts = torch.zeros(
            (self.capacity, self.context_dim), dtype=torch.float32, device=self.device
        )
        self._action_vectors = torch.zeros(
            (self.capacity, self.action_dim), dtype=torch.float32, device=self.device
        )
        self._strength_values = torch.zeros(
            self.capacity, dtype=torch.float32, device=self.device
        )
        self._actual_sensory = torch.zeros(
            (self.capacity, self.sensory_dim), dtype=torch.float32, device=self.device
        )
        self._predicted_sensory = torch.zeros(
            (self.capacity, self.sensory_dim), dtype=torch.float32, device=self.device
        )
        self._prediction_errors = torch.zeros(
            self.capacity, dtype=torch.float32, device=self.device
        )
        
        print(f"Initialized tensors: context_dim={self.context_dim}, "
              f"action_dim={self.action_dim}, sensory_dim={self.sensory_dim}")
    
    def _dict_to_vector(self, action_dict: Dict[str, float], vector_dim: int) -> torch.Tensor:
        """Convert action dictionary to standardized vector."""
        vector = torch.zeros(vector_dim, dtype=torch.float32, device=self.device)
        
        # Standard action mapping (expandable)
        standard_actions = {
            'forward_motor': 0,
            'turn_motor': 1, 
            'brake_motor': 2,
            'left_motor': 3,
            'right_motor': 4
        }
        
        for action_name, value in action_dict.items():
            if action_name in standard_actions:
                vector[standard_actions[action_name]] = value
        
        return vector
    
    def _vector_to_dict(self, vector: torch.Tensor) -> Dict[str, float]:
        """Convert action vector back to dictionary."""
        standard_actions = {
            0: 'forward_motor',
            1: 'turn_motor',
            2: 'brake_motor',
            3: 'left_motor',
            4: 'right_motor'
        }
        
        result = {}
        for i, value in enumerate(vector.cpu().numpy()):
            if i in standard_actions and abs(value) > 1e-6:
                result[standard_actions[i]] = float(value)
        
        return result
    
    def compute_similarities_vectorized(self, query_context: List[float], 
                                      top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarities to query context for all experiences simultaneously.
        
        This is the key GPU acceleration - one operation across all experiences.
        
        Returns:
            (similarities, indices) - Top K similar experiences
        """
        if self.size == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        self.stats['gpu_operations'] += 1
        
        # Convert query to tensor
        query_tensor = torch.tensor(query_context, dtype=torch.float32, device=self.device)
        
        # Pad query to match stored contexts
        if len(query_context) < self.context_dim:
            padded_query = torch.zeros(self.context_dim, device=self.device)
            padded_query[:len(query_context)] = query_tensor
            query_tensor = padded_query
        
        # Compute cosine similarities for all stored contexts simultaneously
        # This is the magic: one GPU operation across thousands of experiences
        stored_contexts = self._mental_contexts[:self.size]  # Only active experiences
        
        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query_tensor.unsqueeze(0), dim=1)
        stored_norm = torch.nn.functional.normalize(stored_contexts, dim=1)
        
        # Compute all similarities at once
        similarities = torch.mm(query_norm, stored_norm.t()).squeeze(0)
        
        # Get top K
        top_k = min(top_k, self.size)
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        return top_similarities, top_indices
    
    def get_experience_batch(self, indices: torch.Tensor) -> List[VectorizedExperience]:
        """Get multiple experiences by their indices."""
        return [VectorizedExperience(int(idx), self) for idx in indices]
    
    # Property getters for VectorizedExperience
    def get_mental_context(self, index: int) -> List[float]:
        """Get mental context for specific experience."""
        return self._mental_contexts[index].cpu().numpy().tolist()
    
    def get_action_taken(self, index: int) -> Dict[str, float]:
        """Get action for specific experience.""" 
        return self._vector_to_dict(self._action_vectors[index])
    
    def get_strength(self, index: int) -> float:
        """Get strength for specific experience."""
        return float(self._strength_values[index].cpu().item())
    
    def get_actual_sensory(self, index: int) -> List[float]:
        """Get actual sensory for specific experience."""
        return self._actual_sensory[index].cpu().numpy().tolist()
    
    def get_predicted_sensory(self, index: int) -> List[float]:
        """Get predicted sensory for specific experience."""
        return self._predicted_sensory[index].cpu().numpy().tolist()
    
    def get_prediction_error(self, index: int) -> float:
        """Get prediction error for specific experience."""
        return float(self._prediction_errors[index].cpu().item())
    
    def get_connection_weights(self, index: int) -> Dict[str, float]:
        """Get connection weights for specific experience (placeholder)."""
        # TODO: Implement sparse matrix connections
        return {}
    
    def get_experience_node(self, index: int) -> ExperienceNode:
        """Convert vectorized experience back to ExperienceNode."""
        return ExperienceNode(
            mental_context=self.get_mental_context(index),
            action_taken=self.get_action_taken(index),
            predicted_sensory=self.get_predicted_sensory(index),
            actual_sensory=self.get_actual_sensory(index),
            prediction_error=self.get_prediction_error(index),
            strength=self.get_strength(index)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        memory_usage = 0
        if self._mental_contexts is not None:
            memory_usage += self._mental_contexts.element_size() * self._mental_contexts.nelement()
            memory_usage += self._action_vectors.element_size() * self._action_vectors.nelement()
            memory_usage += self._strength_values.element_size() * self._strength_values.nelement()
        
        return {
            'device': str(self.device),
            'size': self.size,
            'capacity': self.capacity,
            'memory_usage_bytes': memory_usage,
            'memory_usage_mb': memory_usage / (1024 * 1024),
            **self.stats
        }
    
    def get_size(self) -> int:
        """Get current number of stored experiences."""
        return self.size
    
    def clear_cache(self):
        """Clear any cached computations."""
        # Future: Clear any cached similarity computations
        pass