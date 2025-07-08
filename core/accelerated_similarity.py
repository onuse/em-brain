"""
Hardware-agnostic accelerated similarity search for brain traversals.
Automatically detects and uses the best available acceleration (JAX GPU/CPU, NumPy).
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import time
from collections import defaultdict

# Try to import sklearn for spatial indexing
try:
    from sklearn.neighbors import NearestNeighbors, BallTree
    SKLEARN_AVAILABLE = True
    print("Spatial Indexing: scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Spatial Indexing: scikit-learn not available, using grid-based fallback")

# Try to import PyTorch with MPS support
try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    MPS_BUILT = torch.backends.mps.is_built()
    
    print(f"PyTorch Acceleration: PyTorch {torch.__version__} available")
    print(f"  MPS Built: {'Yes' if MPS_BUILT else 'No'}")
    print(f"  MPS Available: {'Yes' if MPS_AVAILABLE else 'No'}")
    
    # Test MPS functionality
    MPS_FUNCTIONAL = False
    if MPS_AVAILABLE:
        try:
            # Test basic MPS operation
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            _ = test_tensor + 1
            MPS_FUNCTIONAL = True
            print(f"  MPS Functional: Yes")
        except Exception as e:
            print(f"  MPS Functional: No ({e})")
            MPS_FUNCTIONAL = False
    else:
        print(f"  MPS Functional: No (not available)")
        
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    MPS_BUILT = False
    MPS_FUNCTIONAL = False
    print("PyTorch Acceleration: PyTorch not available")

# Try to import acceleration libraries - JAX Metal disabled to avoid warnings
try:
    # Disable JAX Metal support to avoid noisy warnings
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU only
    
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    GPU_AVAILABLE = False  # Disabled for now
    METAL_AVAILABLE = False  # Disabled for now
    METAL_FUNCTIONAL = False  # Disabled for now
    
    print(f"Accelerated Similarity: JAX {jax.__version__} available (CPU-only mode)")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Metal Support: Disabled (to avoid warnings)")
    print(f"  GPU/Acceleration: CPU Optimized")
    
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    METAL_AVAILABLE = False
    METAL_FUNCTIONAL = False
    print("Accelerated Similarity: JAX not available, using NumPy fallback")


class AcceleratedSimilarityEngine:
    """
    Hardware-agnostic similarity search engine.
    Automatically chooses the best available acceleration method.
    """
    
    def __init__(self, enable_pytorch: bool = True, enable_jax: bool = True, enable_caching: bool = True, 
                 enable_spatial_indexing: bool = True, spatial_index_threshold: int = 100):
        """
        Initialize the acceleration engine.
        
        Args:
            enable_pytorch: Whether to use PyTorch MPS acceleration if available (highest priority)
            enable_jax: Whether to use JAX acceleration if available
            enable_caching: Whether to enable context caching for repeated searches
            enable_spatial_indexing: Whether to use spatial indexing for O(log n) search
            spatial_index_threshold: Minimum number of contexts to trigger spatial indexing
        """
        # Determine best acceleration method (priority order)
        self.enable_pytorch = enable_pytorch and TORCH_AVAILABLE and MPS_FUNCTIONAL
        self.enable_jax = enable_jax and JAX_AVAILABLE and not self.enable_pytorch  # JAX fallback
        self.enable_caching = enable_caching
        self.enable_spatial_indexing = enable_spatial_indexing and SKLEARN_AVAILABLE
        self.spatial_index_threshold = spatial_index_threshold
        
        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.cache_hits = 0
        self.spatial_index_hits = 0
        
        # Context caching for performance
        self._context_cache = {} if enable_caching else None
        self._max_cache_size = 5000  # Increased cache size for better performance with large memories
        
        # Spatial indexing components
        self._spatial_index = None
        self._indexed_contexts = None
        self._index_dirty = False
        
        # Initialize acceleration method
        self._init_acceleration()
        
        # Determine acceleration method name
        if self.enable_pytorch:
            method_name = "PyTorch MPS"
        elif self.enable_jax:
            method_name = "JAX"
        else:
            method_name = "NumPy"
        
        print(f"AcceleratedSimilarityEngine initialized:")
        print(f"  Method: {method_name}")
        print(f"  Caching: {'Enabled' if enable_caching else 'Disabled'}")
        print(f"  Spatial Indexing: {'Enabled' if self.enable_spatial_indexing else 'Disabled'}")
        if self.enable_spatial_indexing:
            print(f"  Index Threshold: {spatial_index_threshold} contexts")
    
    def _init_acceleration(self):
        """Initialize the appropriate acceleration method"""
        # Always initialize NumPy acceleration as fallback
        self._init_numpy_acceleration()
        
        if self.enable_pytorch:
            self._init_pytorch_acceleration()
        elif self.enable_jax:
            self._init_jax_acceleration()
    
    def _init_pytorch_acceleration(self):
        """Initialize PyTorch MPS-based acceleration"""
        def _pytorch_similarity_search(target_context, all_contexts):
            # Convert to PyTorch tensors on MPS device
            target = torch.tensor(target_context, dtype=torch.float32).to('mps')
            contexts = torch.tensor(all_contexts, dtype=torch.float32).to('mps')
            
            # Vectorized Euclidean distance on GPU
            distances = torch.norm(contexts - target, dim=1)
            max_distance = torch.sqrt(torch.tensor(len(target_context) * 4.0)).to('mps')
            
            similarities = torch.clamp(1.0 - (distances / max_distance), min=0.0)
            return similarities.cpu().numpy()  # Return to CPU as numpy array
        
        def _pytorch_similarity_with_filtering(target_context, all_contexts,
                                             similarity_threshold, max_results):
            similarities = _pytorch_similarity_search(target_context, all_contexts)
            
            # Find indices that meet threshold
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            valid_similarities = similarities[valid_indices]
            
            # Sort by similarity (highest first)
            if len(valid_similarities) > 0:
                sorted_order = np.argsort(-valid_similarities)
                top_indices = valid_indices[sorted_order[:max_results]]
                top_similarities = valid_similarities[sorted_order[:max_results]]
            else:
                top_indices = np.array([], dtype=int)
                top_similarities = np.array([])
            
            return top_indices, top_similarities
        
        self._pytorch_similarity_search = _pytorch_similarity_search
        self._pytorch_similarity_with_filtering = _pytorch_similarity_with_filtering
        
        # Warmup PyTorch MPS with dummy data
        dummy_context = [0.0] * 8
        dummy_contexts = np.zeros((10, 8))
        
        try:
            _ = self._pytorch_similarity_search(dummy_context, dummy_contexts)
            _ = self._pytorch_similarity_with_filtering(dummy_context, dummy_contexts, 0.5, 5)
            print("  PyTorch MPS warmup: Complete")
        except Exception as e:
            print(f"  PyTorch MPS warmup warning: {e}")
            # Fall back to JAX/NumPy on MPS failure
            print("  Falling back to JAX/NumPy acceleration")
            self.enable_pytorch = False
            if JAX_AVAILABLE:
                self.enable_jax = True
                self._init_jax_acceleration()
            else:
                self._init_numpy_acceleration()
    
    def _init_jax_acceleration(self):
        """Initialize JAX-based acceleration"""
        @jax.jit
        def _jax_similarity_search(target_context, all_contexts):
            target = jnp.array(target_context)
            contexts = jnp.array(all_contexts)
            
            # Vectorized Euclidean distance
            distances = jnp.linalg.norm(contexts - target, axis=1)
            max_distance = jnp.sqrt(len(target_context) * 4.0)
            
            similarities = jnp.maximum(0.0, 1.0 - (distances / max_distance))
            return similarities
        
        def _jax_similarity_with_filtering(target_context, all_contexts, 
                                         similarity_threshold, max_results):
            similarities = _jax_similarity_search(target_context, all_contexts)
            
            # Sort all similarities in descending order
            sorted_indices = jnp.argsort(-similarities)
            sorted_similarities = similarities[sorted_indices]
            
            # Use lax.dynamic_slice for dynamic indexing
            from jax import lax
            top_indices = lax.dynamic_slice(sorted_indices, (0,), (max_results,))
            top_similarities = lax.dynamic_slice(sorted_similarities, (0,), (max_results,))
            
            # Create threshold mask
            threshold_mask = top_similarities >= similarity_threshold
            
            return top_indices, top_similarities, threshold_mask
        
        self._jax_similarity_search = _jax_similarity_search
        self._jax_similarity_with_filtering = _jax_similarity_with_filtering
        
        # Warmup JIT compilation with dummy data
        dummy_context = [0.0] * 8
        dummy_contexts = np.zeros((10, 8))
        
        try:
            _ = self._jax_similarity_search(dummy_context, dummy_contexts)
            _ = self._jax_similarity_with_filtering(dummy_context, dummy_contexts, 0.5, 5)
            print("  JAX JIT compilation: Complete")
        except Exception as e:
            print(f"  JAX JIT compilation warning: {e}")
            # Fall back to NumPy on JIT compilation failure
            print("  Falling back to NumPy acceleration")
            self.enable_jax = False
            self._init_numpy_acceleration()
    
    def _init_numpy_acceleration(self):
        """Initialize NumPy-based acceleration"""
        def _numpy_similarity_search(target_context, all_contexts):
            target = np.array(target_context)
            contexts = np.array(all_contexts)
            
            # Vectorized Euclidean distance
            distances = np.linalg.norm(contexts - target, axis=1)
            max_distance = np.sqrt(len(target_context) * 4.0)
            
            similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
            return similarities
        
        def _numpy_similarity_with_filtering(target_context, all_contexts,
                                           similarity_threshold, max_results):
            similarities = _numpy_similarity_search(target_context, all_contexts)
            
            # Find indices that meet threshold
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            valid_similarities = similarities[valid_indices]
            
            # Sort by similarity (highest first)
            if len(valid_similarities) > 0:
                sorted_order = np.argsort(-valid_similarities)
                top_indices = valid_indices[sorted_order[:max_results]]
                top_similarities = valid_similarities[sorted_order[:max_results]]
            else:
                top_indices = np.array([], dtype=int)
                top_similarities = np.array([])
            
            return top_indices, top_similarities
        
        self._numpy_similarity_search = _numpy_similarity_search
        self._numpy_similarity_with_filtering = _numpy_similarity_with_filtering
    
    def find_similar_contexts(self, target_context: List[float], 
                            all_contexts: List[List[float]],
                            similarity_threshold: float = 0.7,
                            max_results: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find contexts similar to the target context.
        
        Args:
            target_context: The context to find similarities for
            all_contexts: List of all available contexts
            similarity_threshold: Minimum similarity threshold (0.0 to 1.0)
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (indices, similarities) of matching contexts
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = None
        if self._context_cache is not None:
            cache_key = (tuple(target_context), len(all_contexts), similarity_threshold, max_results)
            if cache_key in self._context_cache:
                self.cache_hits += 1
                return self._context_cache[cache_key]
        
        # Convert to appropriate format
        all_contexts_array = np.array(all_contexts)
        
        # Choose search method based on data size and availability
        if (self.enable_spatial_indexing and 
            len(all_contexts) >= self.spatial_index_threshold):
            # Use spatial indexing for large datasets (O(log n))
            indices, similarities = self._spatial_find_similar(
                target_context, all_contexts_array, similarity_threshold, max_results
            )
            self.spatial_index_hits += 1
        else:
            # Use accelerated linear search for smaller datasets (O(n))
            if self.enable_pytorch:
                indices, similarities = self._pytorch_find_similar(
                    target_context, all_contexts_array, similarity_threshold, max_results
                )
            elif self.enable_jax:
                indices, similarities = self._jax_find_similar(
                    target_context, all_contexts_array, similarity_threshold, max_results
                )
            else:
                indices, similarities = self._numpy_find_similar(
                    target_context, all_contexts_array, similarity_threshold, max_results
                )
        
        # Update performance tracking
        search_time = time.time() - start_time
        self.search_count += 1
        self.total_search_time += search_time
        
        # Cache result
        if self._context_cache is not None and cache_key is not None:
            if len(self._context_cache) >= self._max_cache_size:
                # Simple cache eviction: remove oldest entries
                oldest_keys = list(self._context_cache.keys())[:100]
                for old_key in oldest_keys:
                    del self._context_cache[old_key]
            
            self._context_cache[cache_key] = (indices, similarities)
        
        return indices, similarities
    
    def _pytorch_find_similar(self, target_context, all_contexts_array,
                             similarity_threshold, max_results):
        """PyTorch MPS-accelerated similarity search"""
        indices, similarities = self._pytorch_similarity_with_filtering(
            target_context, all_contexts_array, similarity_threshold, max_results
        )
        
        return indices, similarities
    
    def _jax_find_similar(self, target_context, all_contexts_array, 
                         similarity_threshold, max_results):
        """JAX-accelerated similarity search"""
        indices, similarities, mask = self._jax_similarity_with_filtering(
            target_context, all_contexts_array, similarity_threshold, max_results
        )
        
        # Convert back to numpy and filter by mask
        indices_np = np.array(indices)
        similarities_np = np.array(similarities)
        mask_np = np.array(mask)
        
        # Return only valid results
        valid_indices = indices_np[mask_np]
        valid_similarities = similarities_np[mask_np]
        
        return valid_indices, valid_similarities
    
    def _numpy_find_similar(self, target_context, all_contexts_array,
                           similarity_threshold, max_results):
        """NumPy-accelerated similarity search"""
        indices, similarities = self._numpy_similarity_with_filtering(
            target_context, all_contexts_array, similarity_threshold, max_results
        )
        
        return indices, similarities
    
    def _spatial_find_similar(self, target_context, all_contexts_array,
                             similarity_threshold, max_results):
        """Spatial index-accelerated similarity search for O(log n) performance"""
        # Update spatial index if needed
        self._update_spatial_index(all_contexts_array)
        
        if self._spatial_index is None:
            # Fallback to regular search if indexing failed
            return self._numpy_find_similar(target_context, all_contexts_array, 
                                          similarity_threshold, max_results)
        
        # Convert similarity threshold to distance threshold
        # similarity = 1 - (distance / max_distance)
        # distance = (1 - similarity) * max_distance
        max_distance = np.sqrt(len(target_context) * 4.0)
        distance_threshold = (1.0 - similarity_threshold) * max_distance
        
        # Query spatial index for nearby contexts
        # Request more candidates than needed to account for filtering
        search_k = min(max_results * 5, len(all_contexts_array))
        
        try:
            distances, indices = self._spatial_index.query(
                [target_context], k=search_k, return_distance=True
            )
            
            # Convert distances to similarities
            distances = distances[0]  # Remove batch dimension
            indices = indices[0]
            
            similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
            
            # Filter by threshold and limit results
            valid_mask = similarities >= similarity_threshold
            valid_indices = indices[valid_mask]
            valid_similarities = similarities[valid_mask]
            
            # Sort by similarity (highest first) and limit
            if len(valid_similarities) > 0:
                sort_order = np.argsort(-valid_similarities)
                final_indices = valid_indices[sort_order[:max_results]]
                final_similarities = valid_similarities[sort_order[:max_results]]
            else:
                final_indices = np.array([], dtype=int)
                final_similarities = np.array([])
            
            return final_indices, final_similarities
            
        except Exception as e:
            print(f"Spatial index search failed: {e}, falling back to linear search")
            return self._numpy_find_similar(target_context, all_contexts_array,
                                          similarity_threshold, max_results)
    
    def _update_spatial_index(self, all_contexts_array):
        """Update the spatial index if contexts have changed"""
        # Check if we need to rebuild the index
        if (self._spatial_index is None or 
            self._indexed_contexts is None or
            not np.array_equal(self._indexed_contexts, all_contexts_array)):
            
            self._build_spatial_index(all_contexts_array)
    
    def _build_spatial_index(self, all_contexts_array):
        """Build spatial index for the given contexts"""
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, spatial indexing disabled")
            return
        
        try:
            # Use Ball Tree for efficient nearest neighbor search in high dimensions
            # Ball Tree works better than KD-Tree for dimensions > 10
            self._spatial_index = BallTree(all_contexts_array, metric='euclidean')
            self._indexed_contexts = all_contexts_array.copy()
            self._index_dirty = False
            
            print(f"Spatial index built: {len(all_contexts_array)} contexts indexed")
            
        except Exception as e:
            print(f"Failed to build spatial index: {e}")
            self._spatial_index = None
            self._indexed_contexts = None
    
    def compute_all_similarities(self, target_context: List[float],
                               all_contexts: List[List[float]]) -> np.ndarray:
        """
        Compute similarities to all contexts (for debugging/analysis).
        
        Args:
            target_context: The context to compute similarities for
            all_contexts: List of all available contexts
            
        Returns:
            Array of similarities for all contexts
        """
        all_contexts_array = np.array(all_contexts)
        
        if self.enable_pytorch:
            similarities = self._pytorch_similarity_search(target_context, all_contexts_array)
            return similarities
        elif self.enable_jax:
            similarities = self._jax_similarity_search(target_context, all_contexts_array)
            return np.array(similarities)
        else:
            similarities = self._numpy_similarity_search(target_context, all_contexts_array)
            return similarities
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        avg_search_time = self.total_search_time / max(1, self.search_count)
        cache_hit_rate = self.cache_hits / max(1, self.search_count)
        spatial_hit_rate = self.spatial_index_hits / max(1, self.search_count)
        
        return {
            'total_searches': self.search_count,
            'total_time': self.total_search_time,
            'avg_search_time': avg_search_time,
            'searches_per_second': 1.0 / max(0.000001, avg_search_time),
            'cache_enabled': self._context_cache is not None,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'spatial_indexing_enabled': self.enable_spatial_indexing,
            'spatial_index_hits': self.spatial_index_hits,
            'spatial_hit_rate': spatial_hit_rate,
            'spatial_index_active': self._spatial_index is not None,
            'acceleration_method': self._get_acceleration_method_name(),
            'gpu_available': GPU_AVAILABLE or MPS_AVAILABLE
        }
    
    def _get_acceleration_method_name(self) -> str:
        """Get the name of the current acceleration method"""
        if self.enable_pytorch:
            return "PyTorch MPS"
        elif self.enable_jax:
            return "JAX"
        else:
            return "NumPy"
    
    def clear_cache(self):
        """Clear the similarity cache"""
        if self._context_cache is not None:
            self._context_cache.clear()
            print("Similarity cache cleared")


# Global similarity engine instance
_global_engine: Optional[AcceleratedSimilarityEngine] = None

def get_similarity_engine() -> AcceleratedSimilarityEngine:
    """Get the global similarity engine instance"""
    global _global_engine
    if _global_engine is None:
        _global_engine = AcceleratedSimilarityEngine()
    return _global_engine

def benchmark_similarity_engine():
    """Benchmark the similarity engine performance"""
    engine = get_similarity_engine()
    
    # Create test data
    n_nodes = 2140
    context_dim = 8
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0
    
    print(f"\\nBenchmarking similarity engine with {n_nodes} nodes...")
    
    # Benchmark similarity search
    start_time = time.time()
    for i in range(10):
        indices, similarities = engine.find_similar_contexts(
            target_context, all_contexts.tolist(), similarity_threshold=0.7
        )
    benchmark_time = time.time() - start_time
    
    # Show results
    print(f"10 similarity searches: {benchmark_time:.4f}s ({benchmark_time/10:.6f}s each)")
    print(f"Found {len(indices)} similar contexts")
    
    # Performance stats
    stats = engine.get_performance_stats()
    print(f"Performance: {stats['searches_per_second']:.0f} searches/second")
    print(f"Method: {stats['acceleration_method']}")
    
    return benchmark_time / 10