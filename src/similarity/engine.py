"""
Core Similarity Search Engine

Fast similarity search through experience vectors using GPU acceleration when available.
The faster this is, the more intelligent the behavior becomes.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time
from .learnable_similarity import LearnableSimilarity

# Try GPU acceleration (PyTorch MPS preferred)
try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    
    # Test MPS functionality
    MPS_FUNCTIONAL = False
    if MPS_AVAILABLE:
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            _ = test_tensor + 1
            MPS_FUNCTIONAL = True
            print("🚀 GPU acceleration: PyTorch MPS available")
        except Exception:
            MPS_FUNCTIONAL = False
            
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    MPS_FUNCTIONAL = False


class SimilarityEngine:
    """
    Ultra-fast similarity search through experience vectors.
    
    This is the core intelligence engine - finding similar past experiences
    in milliseconds enables intelligent prediction and action selection.
    """
    
    def __init__(self, use_gpu: bool = True, use_learnable_similarity: bool = True):
        """
        Initialize similarity search engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            use_learnable_similarity: Whether to use adaptive similarity learning
        """
        self.use_gpu = use_gpu and MPS_FUNCTIONAL
        self.device = 'mps' if self.use_gpu else 'cpu'
        self.use_learnable_similarity = use_learnable_similarity
        
        # Learnable similarity function
        if use_learnable_similarity:
            self.learnable_similarity = LearnableSimilarity()
        else:
            self.learnable_similarity = None
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
        self.cache_hits = 0
        
        # Simple caching for repeated searches
        self._cache = {}
        self._max_cache_size = 1000
        
        similarity_type = "learnable adaptive" if use_learnable_similarity else "hardcoded cosine"
        if self.use_gpu:
            print(f"SimilarityEngine using {similarity_type} similarity with GPU acceleration")
            self._warmup_gpu()
        else:
            print(f"SimilarityEngine using {similarity_type} similarity with CPU")
    
    def find_similar_experiences(self, 
                               target_vector: List[float],
                               experience_vectors: List[List[float]],
                               experience_ids: List[str],
                               max_results: int = 10,
                               min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        """
        Find the most similar experience vectors to the target.
        
        Args:
            target_vector: The vector to find similarities for (context from current situation)
            experience_vectors: List of all experience context vectors to search through  
            experience_ids: Corresponding IDs for each experience vector
            max_results: Maximum number of similar experiences to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of (experience_id, similarity_score) sorted by similarity (highest first)
        """
        if not experience_vectors or not experience_ids:
            return []
        
        if len(experience_vectors) != len(experience_ids):
            raise ValueError("experience_vectors and experience_ids must have same length")
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_cache_key(target_vector, len(experience_vectors), max_results, min_similarity)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]
        
        # Compute similarities
        if self.use_gpu and len(experience_vectors) > 50:  # GPU worth it for larger datasets
            similarities = self._gpu_compute_similarities(target_vector, experience_vectors)
        else:
            similarities = self._cpu_compute_similarities(target_vector, experience_vectors)
        
        # Find best matches
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                results.append((experience_ids[i], float(similarity)))
        
        # Sort by similarity (highest first) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:max_results]
        
        # Update performance tracking
        search_time = time.time() - start_time
        self.total_searches += 1
        self.total_search_time += search_time
        
        # Cache result
        self._cache_result(cache_key, results)
        
        return results
    
    def _gpu_compute_similarities(self, target_vector: List[float], 
                                 experience_vectors: List[List[float]]) -> np.ndarray:
        """Compute similarities using GPU acceleration."""
        try:
            # Convert to GPU tensors
            target = torch.tensor(target_vector, dtype=torch.float32, device=self.device)
            experiences = torch.tensor(experience_vectors, dtype=torch.float32, device=self.device)
            
            # Vectorized cosine similarity computation
            # similarity = dot_product / (norm1 * norm2)
            target_norm = torch.norm(target)
            experience_norms = torch.norm(experiences, dim=1)
            
            if target_norm == 0 or torch.any(experience_norms == 0):
                # Handle zero vectors with Euclidean distance instead
                distances = torch.norm(experiences - target, dim=1)
                max_distance = torch.sqrt(torch.tensor(len(target_vector) * 4.0, device=self.device))
                similarities = torch.clamp(1.0 - (distances / max_distance), min=0.0)
            else:
                # Cosine similarity
                dot_products = torch.matmul(experiences, target)
                similarities = dot_products / (target_norm * experience_norms)
                # Convert from [-1, 1] to [0, 1] 
                similarities = (similarities + 1.0) / 2.0
            
            return similarities.cpu().numpy()
            
        except Exception as e:
            print(f"GPU similarity computation failed: {e}, falling back to CPU")
            return self._cpu_compute_similarities(target_vector, experience_vectors)
    
    def _cpu_compute_similarities(self, target_vector: List[float], 
                                 experience_vectors: List[List[float]]) -> np.ndarray:
        """Compute similarities using CPU (NumPy)."""
        
        if self.use_learnable_similarity and self.learnable_similarity is not None:
            # Use learned similarity function
            similarities = []
            for exp_vector in experience_vectors:
                sim = self.learnable_similarity.compute_similarity(target_vector, exp_vector)
                similarities.append(sim)
            return np.array(similarities)
        
        else:
            # Fallback to hardcoded cosine similarity
            target = np.array(target_vector)
            experiences = np.array(experience_vectors)
            
            # Vectorized cosine similarity
            target_norm = np.linalg.norm(target)
            experience_norms = np.linalg.norm(experiences, axis=1)
            
            if target_norm == 0 or np.any(experience_norms == 0):
                # Handle zero vectors with Euclidean distance
                distances = np.linalg.norm(experiences - target, axis=1)
                max_distance = np.sqrt(len(target_vector) * 4.0)
                similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
            else:
                # Cosine similarity
                dot_products = np.dot(experiences, target)
                similarities = dot_products / (target_norm * experience_norms)
                # Convert from [-1, 1] to [0, 1]
                similarities = (similarities + 1.0) / 2.0
            
            return similarities
    
    def get_most_similar(self, target_vector: List[float],
                        experience_vectors: List[List[float]],
                        experience_ids: List[str]) -> Tuple[Optional[str], float]:
        """
        Get the single most similar experience.
        
        Args:
            target_vector: The vector to find similarity for
            experience_vectors: List of experience vectors to search
            experience_ids: Corresponding experience IDs
            
        Returns:
            Tuple of (most_similar_id, similarity_score) or (None, 0.0) if no experiences
        """
        results = self.find_similar_experiences(
            target_vector, experience_vectors, experience_ids, 
            max_results=1, min_similarity=0.0
        )
        
        if results:
            return results[0][0], results[0][1]
        return None, 0.0
    
    def compute_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Compute similarity between two individual vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if len(vector1) != len(vector2):
            return 0.0
        
        if self.use_gpu:
            similarities = self._gpu_compute_similarities(vector1, [vector2])
            return float(similarities[0])
        else:
            similarities = self._cpu_compute_similarities(vector1, [vector2])
            return float(similarities[0])
    
    def _create_cache_key(self, target_vector: List[float], num_experiences: int, 
                         max_results: int, min_similarity: float) -> str:
        """Create a cache key for this search."""
        # Hash the target vector and parameters
        vector_hash = hash(tuple(target_vector))
        return f"{vector_hash}_{num_experiences}_{max_results}_{min_similarity}"
    
    def _cache_result(self, cache_key: str, results: List[Tuple[str, float]]):
        """Cache a search result."""
        if len(self._cache) >= self._max_cache_size:
            # Simple cache eviction - remove oldest entries
            oldest_keys = list(self._cache.keys())[:100]
            for key in oldest_keys:
                del self._cache[key]
        
        self._cache[cache_key] = results
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy computation."""
        try:
            dummy_target = [1.0, 2.0, 3.0, 4.0]
            dummy_experiences = [[0.5, 1.5, 2.5, 3.5], [2.0, 3.0, 4.0, 5.0]]
            _ = self._gpu_compute_similarities(dummy_target, dummy_experiences)
        except Exception as e:
            print(f"GPU warmup failed: {e}")
            self.use_gpu = False
            self.device = 'cpu'
    
    def record_prediction_outcome(self, 
                                query_vector: List[float],
                                similar_experience_id: str,
                                similar_vector: List[float],
                                prediction_success: float):
        """
        Record how well a similar experience helped with prediction.
        
        This is how the similarity function learns - by tracking whether
        experiences labeled as "similar" actually help predict each other.
        
        Args:
            query_vector: The experience we wanted to predict from
            similar_experience_id: ID of the experience used for prediction
            similar_vector: Vector of the experience used for prediction
            prediction_success: How well it worked (1.0 = perfect, 0.0 = useless)
        """
        if self.use_learnable_similarity and self.learnable_similarity is not None:
            self.learnable_similarity.record_prediction_outcome(
                query_vector, similar_vector, prediction_success
            )
    
    def adapt_similarity_function(self):
        """
        Adapt the similarity function based on prediction success patterns.
        
        Call this periodically to let the similarity function evolve.
        """
        if self.use_learnable_similarity and self.learnable_similarity is not None:
            self.learnable_similarity.adapt_similarity_function()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_search_time = self.total_search_time / max(1, self.total_searches)
        cache_hit_rate = self.cache_hits / max(1, self.total_searches)
        
        stats = {
            'total_searches': self.total_searches,
            'total_time': self.total_search_time,
            'avg_search_time': avg_search_time,
            'searches_per_second': 1.0 / max(0.000001, avg_search_time),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'gpu_enabled': self.use_gpu,
            'device': self.device,
            'similarity_type': 'learnable' if self.use_learnable_similarity else 'hardcoded_cosine'
        }
        
        # Add similarity learning statistics if available
        if self.use_learnable_similarity and self.learnable_similarity is not None:
            similarity_stats = self.learnable_similarity.get_similarity_statistics()
            stats['similarity_learning'] = similarity_stats
        
        return stats
    
    def clear_cache(self):
        """Clear the similarity cache."""
        self._cache.clear()
        print("🧹 Similarity cache cleared")
    
    def __str__(self) -> str:
        return f"SimilarityEngine(device={self.device}, searches={self.total_searches})"
    
    def __repr__(self) -> str:
        return self.__str__()