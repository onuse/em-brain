"""
Core Similarity Search Engine

Fast similarity search through experience vectors using GPU acceleration when available.
The faster this is, the more intelligent the behavior becomes.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time
from .learnable_similarity import LearnableSimilarity
from .adaptive_attention import AdaptiveAttentionScorer
from .hierarchical_index import HierarchicalExperienceIndex
from ..utils.cache_adapters import SimilarityEngineCacheAdapter

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
    
    def __init__(self, use_gpu: bool = True, use_learnable_similarity: bool = True, 
                 use_natural_attention: bool = True, use_hierarchical_indexing: bool = True):
        """
        Initialize similarity search engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            use_learnable_similarity: Whether to use adaptive similarity learning
            use_natural_attention: Whether to use natural attention weighting
            use_hierarchical_indexing: Whether to use hierarchical indexing for 10k+ experiences
        """
        self.use_gpu = use_gpu and MPS_FUNCTIONAL
        self.device = 'mps' if self.use_gpu else 'cpu'
        self.use_learnable_similarity = use_learnable_similarity
        self.use_natural_attention = use_natural_attention
        self.use_hierarchical_indexing = use_hierarchical_indexing
        
        # Learnable similarity function
        if use_learnable_similarity:
            # Experience vectors are sensory_input (4D) + action_taken (4D) = 8D
            self.learnable_similarity = LearnableSimilarity(vector_dimensions=8, use_gpu=use_gpu)
        else:
            self.learnable_similarity = None
        
        # Natural attention weighting (no explicit scoring, uses emergent properties)
        if use_natural_attention:
            from .adaptive_attention import NaturalAttentionSimilarity
            self.natural_attention_similarity = NaturalAttentionSimilarity(self)
        else:
            self.natural_attention_similarity = None
        
        # Hierarchical indexing for massive experience scaling
        if use_hierarchical_indexing:
            self.hierarchical_index = HierarchicalExperienceIndex(
                max_region_size=50,
                similarity_threshold=0.4,
                max_search_regions=3
            )
        else:
            self.hierarchical_index = None
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
        
        # Memory-managed caching for repeated searches
        self._cache = SimilarityEngineCacheAdapter(
            max_entries=1000,
            max_size_mb=50.0,  # Reasonable memory limit
            eviction_policy="hybrid"  # Use hybrid eviction strategy
        )
        
        similarity_type = "learnable adaptive" if use_learnable_similarity else "hardcoded cosine"
        attention_type = " + natural attention" if use_natural_attention else ""
        hierarchical_type = " + hierarchical indexing" if use_hierarchical_indexing else ""
        
        if self.use_gpu:
            print(f"SimilarityEngine using {similarity_type}{attention_type}{hierarchical_type} similarity with GPU acceleration")
            self._warmup_gpu()
        else:
            print(f"SimilarityEngine using {similarity_type}{attention_type}{hierarchical_type} similarity with CPU")
    
    def find_similar_experiences(self, 
                               target_vector: List[float],
                               experience_vectors: List[List[float]],
                               experience_ids: List[str],
                               max_results: int = 10,
                               min_similarity: float = 0.3,
                               use_raw_similarity: bool = False) -> List[Tuple[str, float]]:
        """
        Find the most similar experience vectors to the target.
        
        Args:
            target_vector: The vector to find similarities for (context from current situation)
            experience_vectors: List of all experience context vectors to search through  
            experience_ids: Corresponding IDs for each experience vector
            max_results: Maximum number of similar experiences to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            use_raw_similarity: Force raw similarity (bypass attention optimization)
            
        Returns:
            List of (experience_id, similarity_score) sorted by similarity (highest first)
        """
        if not experience_vectors or not experience_ids:
            return []
        
        if len(experience_vectors) != len(experience_ids):
            raise ValueError("experience_vectors and experience_ids must have same length")
        
        # Evolution 2.1: Intelligent Routing - DISABLED temporarily due to recursion
        # TODO: Fix recursion issue - natural attention calls back into this method
        # The attention system needs to use raw similarity internally
        pass
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_cache_key(target_vector, len(experience_vectors), max_results, min_similarity)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Decide whether to use hierarchical search
        use_hierarchical = (
            self.use_hierarchical_indexing and 
            self.hierarchical_index is not None and
            self.hierarchical_index.should_use_hierarchical_search(len(experience_vectors))
        )
        
        if use_hierarchical:
            # Use hierarchical search for massive datasets
            target_array = np.array(target_vector, dtype=np.float32)
            experience_arrays = [np.array(vec, dtype=np.float32) for vec in experience_vectors]
            
            results = self.hierarchical_index.find_similar_experiences(
                target_array, experience_arrays, experience_ids, max_results
            )
            
            # Filter by minimum similarity
            results = [(exp_id, sim) for exp_id, sim in results if sim >= min_similarity]
            
        else:
            # Use traditional similarity search
            # Compute similarities - use hardware adaptation to decide GPU usage
            should_use_gpu = False
            if self.use_gpu:
                try:
                    from ..utils.hardware_adaptation import should_use_gpu_for_similarity_search
                    should_use_gpu = should_use_gpu_for_similarity_search(len(experience_vectors))
                except ImportError:
                    # Fallback to hardcoded threshold
                    should_use_gpu = len(experience_vectors) > 50
            
            if should_use_gpu:
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
        
        # Cache result with utility score based on result quality
        if results:
            avg_similarity = sum(score for _, score in results) / len(results)
            result_quality = min(1.0, avg_similarity * len(results) / max_results)
        else:
            result_quality = 0.1
        
        self._cache.put(cache_key, results, result_quality)
        
        return results
    
    def find_similar_experiences_with_natural_attention(self,
                                                      target_vector: List[float],
                                                      experiences: List[Any],
                                                      max_results: int = 10,
                                                      min_similarity: float = 0.3,
                                                      retrieval_mode: str = 'normal') -> List[tuple]:
        """
        Find similar experiences using natural attention weighting.
        
        Uses emergent properties (utility, clustering, access patterns) instead of
        explicit attention scores to create biological memory suppression effects.
        
        Args:
            target_vector: Vector to find similarities for
            experiences: List of Experience objects with natural properties
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold  
            retrieval_mode: 'normal', 'deep', 'hybrid', or 'utility_focused'
            
        Returns:
            List of (experience, weighted_similarity, base_similarity, natural_attention) tuples
        """
        if not self.use_natural_attention or self.natural_attention_similarity is None:
            # Fallback to regular similarity search
            experience_vectors = []
            experience_ids = []
            
            for i, exp in enumerate(experiences):
                context_vector = list(exp.sensory_input) + list(exp.action_taken)
                experience_vectors.append(context_vector)
                experience_ids.append(f"exp_{i}")
            
            base_results = self.find_similar_experiences(
                target_vector, experience_vectors, experience_ids, max_results, min_similarity
            )
            
            # Convert to natural attention format (no attention weighting)
            results = []
            for exp_id, similarity in base_results:
                exp_index = int(exp_id.split('_')[1])
                experience = experiences[exp_index]
                natural_attention = experience.get_natural_attention_weight()
                results.append((experience, similarity, similarity, natural_attention))
            
            return results
        
        # Use natural attention-weighted similarity
        return self.natural_attention_similarity.find_similar_experiences_with_natural_attention(
            target_vector, experiences, max_results, min_similarity, retrieval_mode
        )
    
    def update_experience_utility(self, experience, prediction_success: float):
        """
        Update an experience's prediction utility based on how well it helped with prediction.
        
        Args:
            experience: Experience object to update
            prediction_success: How successful the prediction was (0.0-1.0)
        """
        experience.update_prediction_utility(prediction_success)
    
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
        """Cache a search result (legacy method for compatibility)."""
        # This method is now handled by the put() call in the main search method
        # Keeping for backward compatibility
        pass
    
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
    
    def add_experience_to_index(self, experience_id: str, experience_vector: List[float]):
        """
        Add a new experience to the hierarchical index.
        
        Call this whenever a new experience is stored to maintain the index.
        
        Args:
            experience_id: Unique identifier for the experience
            experience_vector: Vector representation of the experience
        """
        if self.use_hierarchical_indexing and self.hierarchical_index is not None:
            vector_array = np.array(experience_vector, dtype=np.float32)
            self.hierarchical_index.add_experience(experience_id, vector_array)
    
    def _vectors_to_experiences_bridge(self, experience_vectors: List[List[float]], experience_ids: List[str]):
        """
        Bridge function to convert vectors back to experience-like objects for attention system.
        
        This is a temporary bridge between the old vector-based interface and the new
        attention-aware interface. Eventually the brain should pass experience objects directly.
        """
        from ..experience import Experience
        import time
        
        experiences = []
        current_time = time.time()
        
        for i, (vector, exp_id) in enumerate(zip(experience_vectors, experience_ids)):
            # Create a minimal experience object for attention processing
            # Assumes vector format: [sensory_input..., action_taken...]
            vector_len = len(vector)
            
            # Rough heuristic: assume last 4 elements are action, rest are sensory
            if vector_len >= 4:
                sensory_len = vector_len - 4
                sensory_input = vector[:sensory_len] if sensory_len > 0 else vector
                action_taken = vector[-4:] if vector_len >= 4 else [0.0, 0.0, 0.0, 0.0]
            else:
                sensory_input = vector
                action_taken = [0.0, 0.0, 0.0, 0.0]
            
            # Create experience with reasonable defaults
            # The attention system mainly needs: id, prediction_utility, local_cluster_density
            experience = Experience(
                sensory_input=sensory_input,
                action_taken=action_taken,
                outcome=sensory_input,  # Dummy outcome
                prediction_error=0.3,  # Default moderate error
                timestamp=current_time - i  # Spread timestamps slightly
            )
            
            # Set experience ID to match
            experience.id = exp_id
            
            # Set reasonable defaults for attention calculation
            experience.prediction_utility = 0.5  # Default utility
            experience.local_cluster_density = 0.3  # Default density
            experience.access_count = 1  # Default access
            
            experiences.append(experience)
        
        return experiences
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_search_time = self.total_search_time / max(1, self.total_searches)
        cache_stats = self._cache.get_stats()
        
        stats = {
            'total_searches': self.total_searches,
            'total_time': self.total_search_time,
            'avg_search_time': avg_search_time,
            'searches_per_second': 1.0 / max(0.000001, avg_search_time),
            'cache_stats': cache_stats,
            'gpu_enabled': self.use_gpu,
            'device': self.device,
            'similarity_type': 'learnable' if self.use_learnable_similarity else 'hardcoded_cosine'
        }
        
        # Add similarity learning statistics if available
        if self.use_learnable_similarity and self.learnable_similarity is not None:
            similarity_stats = self.learnable_similarity.get_similarity_statistics()
            stats['similarity_learning'] = similarity_stats
        
        # Add natural attention statistics if available
        if self.use_natural_attention and self.natural_attention_similarity is not None:
            # We'll get these when needed from the brain state
            stats['natural_attention_available'] = True
        else:
            stats['natural_attention_available'] = False
        
        # Add hierarchical indexing statistics if available
        if self.use_hierarchical_indexing and self.hierarchical_index is not None:
            hierarchical_stats = self.hierarchical_index.get_performance_stats()
            stats['hierarchical_indexing'] = hierarchical_stats
        else:
            stats['hierarchical_indexing'] = None
        
        return stats
    
    def clear_cache(self):
        """Clear the similarity cache."""
        self._cache.clear()
        print("🧹 Similarity cache cleared")
    
    def __str__(self) -> str:
        return f"SimilarityEngine(device={self.device}, searches={self.total_searches})"
    
    def __repr__(self) -> str:
        return self.__str__()