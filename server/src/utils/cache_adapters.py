"""
Cache Adapters for Existing Brain Systems

Provides adapter classes to integrate existing cache implementations with the new unified memory manager.
This allows gradual migration while maintaining compatibility with existing code.

Key Features:
- Wraps existing cache implementations
- Provides memory bounds and intelligent eviction
- Maintains API compatibility
- Enables coordinated memory management
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import threading
from collections import defaultdict, OrderedDict

from .memory_manager import (
    IntelligentCache, CacheType, EvictionPolicy, get_memory_manager, create_managed_cache
)


class SimilarityEngineCacheAdapter:
    """
    Adapter for SimilarityEngine cache with memory bounds and intelligent eviction.
    
    Replaces the simple dictionary cache in SimilarityEngine with a memory-managed cache.
    """
    
    def __init__(self, 
                 max_entries: int = 1000,
                 max_size_mb: float = 50.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID):
        """Initialize similarity cache adapter."""
        self._cache = create_managed_cache(
            name="similarity_engine_cache",
            cache_type=CacheType.SIMILARITY_SEARCH,
            max_entries=max_entries,
            max_size_mb=max_size_mb,
            eviction_policy=eviction_policy,
            priority=2.0  # High priority - similarity search is critical
        )
        
        # Compatibility with existing SimilarityEngine interface
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, cache_key: str) -> Optional[List[Tuple[str, float]]]:
        """Get cached similarity results."""
        result = self._cache.get(cache_key)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result
    
    def put(self, cache_key: str, results: List[Tuple[str, float]], utility_score: float = 0.5) -> None:
        """Cache similarity results with utility tracking."""
        # Calculate utility based on result quality
        if results:
            # Higher similarity scores and more results indicate higher utility
            avg_similarity = sum(score for _, score in results) / len(results)
            result_count_factor = min(1.0, len(results) / 10.0)  # Normalize to 0-1
            computed_utility = (avg_similarity * 0.7) + (result_count_factor * 0.3)
        else:
            computed_utility = 0.1  # Low utility for empty results
        
        final_utility = max(utility_score, computed_utility)
        self._cache.put(cache_key, results, final_utility)
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __contains__(self, cache_key: str) -> bool:
        """Check if key is in cache."""
        return self._cache.get(cache_key) is not None
    
    def __getitem__(self, cache_key: str) -> List[Tuple[str, float]]:
        """Get item using dictionary-like interface."""
        result = self.get(cache_key)
        if result is None:
            raise KeyError(cache_key)
        return result
    
    def __setitem__(self, cache_key: str, value: List[Tuple[str, float]]) -> None:
        """Set item using dictionary-like interface."""
        self.put(cache_key, value)
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._cache._entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self._cache.get_stats()
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_entries': cache_stats.total_entries,
            'hit_rate': cache_stats.hit_rate,
            'memory_usage_mb': cache_stats.total_size_mb,
            'evictions': cache_stats.eviction_count
        }


class ActivationCacheAdapter:
    """
    Adapter for activation dynamics caches with memory bounds.
    
    Manages similarity caches and spreading activation data with memory limits.
    """
    
    def __init__(self, 
                 max_entries: int = 2000,
                 max_size_mb: float = 100.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.UTILITY_BASED):
        """Initialize activation cache adapter."""
        self._similarity_cache = create_managed_cache(
            name="activation_similarity_cache",
            cache_type=CacheType.ACTIVATION_CACHE,
            max_entries=max_entries,
            max_size_mb=max_size_mb,
            eviction_policy=eviction_policy,
            priority=1.5  # Medium-high priority
        )
        
        # GPU-specific cache for tensor operations
        self._gpu_cache = create_managed_cache(
            name="activation_gpu_cache",
            cache_type=CacheType.GPU_TENSOR_CACHE,
            max_entries=max_entries // 2,
            max_size_mb=max_size_mb * 2,  # GPU tensors can be larger
            eviction_policy=EvictionPolicy.LRU,  # GPU tensors benefit from LRU
            priority=1.8  # High priority for GPU operations
        )
        
        # Thread safety
        self._lock = threading.RLock()
    
    def cache_similar_experiences(self, 
                                 experience_id: str, 
                                 similar_experiences: Dict[str, float],
                                 utility_score: float = 0.5) -> None:
        """Cache similar experiences for an experience ID."""
        with self._lock:
            self._similarity_cache.put(
                f"similar_{experience_id}", 
                similar_experiences, 
                utility_score
            )
    
    def get_similar_experiences(self, experience_id: str) -> Optional[Dict[str, float]]:
        """Get cached similar experiences."""
        with self._lock:
            return self._similarity_cache.get(f"similar_{experience_id}")
    
    def cache_gpu_tensor(self, 
                        tensor_key: str, 
                        tensor_data: Any,
                        utility_score: float = 0.8) -> None:
        """Cache GPU tensor data."""
        with self._lock:
            # GPU tensors have high utility by default due to computation cost
            self._gpu_cache.put(tensor_key, tensor_data, utility_score)
    
    def get_gpu_tensor(self, tensor_key: str) -> Any:
        """Get cached GPU tensor."""
        with self._lock:
            return self._gpu_cache.get(tensor_key)
    
    def clear_similarity_cache(self) -> None:
        """Clear similarity cache."""
        self._similarity_cache.clear()
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache."""
        self._gpu_cache.clear()
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.clear_similarity_cache()
        self.clear_gpu_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        similarity_stats = self._similarity_cache.get_stats()
        gpu_stats = self._gpu_cache.get_stats()
        
        return {
            'similarity_cache': {
                'entries': similarity_stats.total_entries,
                'size_mb': similarity_stats.total_size_mb,
                'hit_rate': similarity_stats.hit_rate,
                'evictions': similarity_stats.eviction_count
            },
            'gpu_cache': {
                'entries': gpu_stats.total_entries,
                'size_mb': gpu_stats.total_size_mb,
                'hit_rate': gpu_stats.hit_rate,
                'evictions': gpu_stats.eviction_count
            }
        }


class PatternCacheAdapter:
    """
    Adapter for pattern analysis cache with time-based and utility-based eviction.
    
    Manages pattern predictions and analysis results with intelligent cleanup.
    """
    
    def __init__(self, 
                 max_entries: int = 500,
                 max_size_mb: float = 75.0,
                 max_age_seconds: float = 300.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID):
        """Initialize pattern cache adapter."""
        self._cache = create_managed_cache(
            name="pattern_analysis_cache",
            cache_type=CacheType.PATTERN_CACHE,
            max_entries=max_entries,
            max_size_mb=max_size_mb,
            eviction_policy=eviction_policy,
            priority=1.2  # Medium priority
        )
        
        self.max_age_seconds = max_age_seconds
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # Cleanup every minute
        
        # Compatibility with existing pattern cache interface
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, cache_key: str) -> Optional[Tuple[Any, float]]:
        """Get cached pattern with timestamp."""
        result = self._cache.get(cache_key)
        if result is not None:
            pattern_data, timestamp = result
            
            # Check if pattern is still valid (not too old)
            if time.time() - timestamp <= self.max_age_seconds:
                self.cache_hits += 1
                return pattern_data, timestamp
            else:
                # Pattern expired, remove it
                self._cache.evict(1)  # This will likely remove the old entry
                
        self.cache_misses += 1
        return None
    
    def put(self, cache_key: str, pattern_data: Any, confidence: float = 0.5) -> None:
        """Cache pattern with confidence as utility score."""
        current_time = time.time()
        
        # Utility based on confidence and freshness
        utility_score = confidence * 0.8 + 0.2  # Boost base utility
        
        self._cache.put(cache_key, (pattern_data, current_time), utility_score)
        
        # Periodic cleanup of old entries
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_entries()
            self._last_cleanup = current_time
    
    def _cleanup_old_entries(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache._entries.items():
            pattern_data, timestamp = entry.data
            if current_time - timestamp > self.max_age_seconds:
                expired_keys.append(key)
        
        if expired_keys:
            # Remove expired entries
            for key in expired_keys:
                if key in self._cache._entries:
                    del self._cache._entries[key]
                    if key in self._cache._access_order:
                        del self._cache._access_order[key]
                    if key in self._cache._frequency_counts:
                        del self._cache._frequency_counts[key]
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __contains__(self, cache_key: str) -> bool:
        """Check if key is in cache and not expired."""
        return self.get(cache_key) is not None
    
    def __getitem__(self, cache_key: str) -> Tuple[Any, float]:
        """Get item using dictionary-like interface."""
        result = self.get(cache_key)
        if result is None:
            raise KeyError(cache_key)
        return result
    
    def __setitem__(self, cache_key: str, value: Tuple[Any, float]) -> None:
        """Set item using dictionary-like interface."""
        pattern_data, confidence = value
        self.put(cache_key, pattern_data, confidence)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self._cache.get_stats()
        
        # Count expired entries
        current_time = time.time()
        expired_count = 0
        for entry in self._cache._entries.values():
            pattern_data, timestamp = entry.data
            if current_time - timestamp > self.max_age_seconds:
                expired_count += 1
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_entries': cache_stats.total_entries,
            'expired_entries': expired_count,
            'hit_rate': cache_stats.hit_rate,
            'memory_usage_mb': cache_stats.total_size_mb,
            'evictions': cache_stats.eviction_count,
            'max_age_seconds': self.max_age_seconds
        }


class StreamBufferAdapter:
    """
    Adapter for stream buffer with memory bounds and efficient circular buffering.
    
    Provides bounded stream storage with intelligent eviction based on utility and age.
    """
    
    def __init__(self, 
                 max_entries: int = 1000,
                 max_size_mb: float = 25.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        """Initialize stream buffer adapter."""
        self._cache = create_managed_cache(
            name="stream_buffer_cache",
            cache_type=CacheType.STREAM_BUFFER,
            max_entries=max_entries,
            max_size_mb=max_size_mb,
            eviction_policy=eviction_policy,
            priority=1.0  # Medium priority
        )
        
        self._sequence_counter = 0
        self._lock = threading.RLock()
    
    def append(self, stream_data: Any, utility_score: float = 0.5) -> str:
        """Append data to stream buffer."""
        with self._lock:
            # Generate unique key for stream entry
            self._sequence_counter += 1
            stream_key = f"stream_{self._sequence_counter}_{time.time():.3f}"
            
            self._cache.put(stream_key, stream_data, utility_score)
            return stream_key
    
    def get_recent(self, count: int = 10) -> List[Tuple[str, Any]]:
        """Get most recent stream entries."""
        with self._lock:
            # Get all entries sorted by access order (most recent first)
            # Create a copy to avoid mutation during iteration
            all_keys = list(self._cache._access_order.keys())
            recent_keys = list(reversed(all_keys))
            
            results = []
            for key in recent_keys[:count]:
                data = self._cache.get(key)
                if data is not None:
                    results.append((key, data))
            
            return results
    
    def get_all(self) -> List[Tuple[str, Any]]:
        """Get all stream entries."""
        with self._lock:
            results = []
            # Create a copy of keys to avoid mutation during iteration
            keys = list(self._cache._access_order.keys())
            for key in keys:
                data = self._cache.get(key)
                if data is not None:
                    results.append((key, data))
            return results
    
    def clear(self) -> None:
        """Clear stream buffer."""
        self._cache.clear()
        self._sequence_counter = 0
    
    def __len__(self) -> int:
        """Get stream buffer size."""
        return len(self._cache._entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream buffer statistics."""
        cache_stats = self._cache.get_stats()
        
        return {
            'total_entries': cache_stats.total_entries,
            'memory_usage_mb': cache_stats.total_size_mb,
            'sequence_counter': self._sequence_counter,
            'evictions': cache_stats.eviction_count,
            'max_entries': self._cache.max_entries
        }


def migrate_existing_cache(existing_cache: Dict[str, Any], 
                          adapter_class: type,
                          **adapter_kwargs) -> Any:
    """
    Migrate existing dictionary-based cache to managed cache adapter.
    
    Args:
        existing_cache: Existing cache dictionary
        adapter_class: Adapter class to migrate to
        **adapter_kwargs: Arguments for adapter initialization
        
    Returns:
        New adapter instance with migrated data
    """
    # Create new adapter
    adapter = adapter_class(**adapter_kwargs)
    
    # Migrate existing data
    migrated_count = 0
    for key, value in existing_cache.items():
        try:
            if hasattr(adapter, 'put'):
                adapter.put(key, value)
            else:
                adapter[key] = value
            migrated_count += 1
        except Exception as e:
            print(f"⚠️ Failed to migrate cache entry {key}: {e}")
    
    print(f"📦 Cache migration complete: {migrated_count} entries migrated to {adapter_class.__name__}")
    return adapter


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all managed caches."""
    memory_manager = get_memory_manager()
    return memory_manager.get_comprehensive_stats()