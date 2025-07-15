"""
Unified Memory Manager and Cache Coordination System

Implements memory bounds and intelligent cache management to prevent unbounded memory growth.
Coordinates multiple cache systems across the brain to ensure overall memory limits are respected.

Key Features:
- Global memory monitoring and bounds checking
- Intelligent cache eviction strategies (LRU, frequency-based, utility-based)
- Cache coordination to prevent memory overflow
- Adaptive memory pressure detection and response
- Performance monitoring and automatic adjustment

This system ensures the brain can run indefinitely without memory issues while maintaining performance.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple, Protocol, NamedTuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    UTILITY_BASED = "utility"      # Based on prediction utility
    HYBRID = "hybrid"              # Combination of multiple factors
    RANDOM = "random"              # Random eviction (baseline)


class CacheType(Enum):
    """Types of caches in the system."""
    SIMILARITY_SEARCH = "similarity_search"
    ACTIVATION_CACHE = "activation_cache"
    PATTERN_CACHE = "pattern_cache"
    GPU_TENSOR_CACHE = "gpu_tensor_cache"
    STREAM_BUFFER = "stream_buffer"
    EMBEDDING_CACHE = "embedding_cache"
    GENERAL = "general"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata for intelligent eviction."""
    key: str
    data: Any
    created_at: float
    last_accessed: float
    access_count: int
    utility_score: float = 0.0
    size_bytes: int = 0
    eviction_priority: float = 0.0  # Lower = higher eviction priority


@dataclass
class MemoryStats:
    """System memory statistics."""
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    memory_pressure: float  # 0.0-1.0
    brain_memory_usage_mb: float
    cache_memory_usage_mb: float
    gpu_memory_usage_mb: float = 0.0


@dataclass
class CacheStats:
    """Cache statistics and performance metrics."""
    cache_type: CacheType
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float
    eviction_count: int
    avg_access_time_ms: float
    memory_efficiency: float


class CacheProtocol(Protocol):
    """Protocol for cache implementations."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        ...
    
    def put(self, key: str, value: Any, utility_score: float = 0.0) -> None:
        """Put item in cache."""
        ...
    
    def evict(self, count: int) -> List[str]:
        """Evict specified number of items."""
        ...
    
    def clear(self) -> None:
        """Clear all cache entries."""
        ...
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...


class IntelligentCache:
    """
    Intelligent cache with multiple eviction strategies and utility tracking.
    """
    
    def __init__(self, 
                 cache_type: CacheType,
                 max_entries: int = 1000,
                 max_size_mb: float = 100.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
                 enable_size_tracking: bool = True):
        """
        Initialize intelligent cache.
        
        Args:
            cache_type: Type of cache for coordination
            max_entries: Maximum number of cache entries
            max_size_mb: Maximum cache size in megabytes
            eviction_policy: Eviction strategy to use
            enable_size_tracking: Whether to track memory usage of entries
        """
        self.cache_type = cache_type
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self.eviction_policy = eviction_policy
        self.enable_size_tracking = enable_size_tracking
        
        # Cache storage
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_counts = defaultdict(int)  # For LFU
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_accesses = 0
        self.total_access_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        print(f"🧠 IntelligentCache initialized: {cache_type.value}, max_entries={max_entries}, "
              f"max_size_mb={max_size_mb:.1f}, policy={eviction_policy}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        start_time = time.time()
        
        with self._lock:
            self.total_accesses += 1
            
            if key not in self._entries:
                self.misses += 1
                self.total_access_time += time.time() - start_time
                return None
            
            # Update access patterns
            entry = self._entries[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._frequency_counts[key] += 1
            
            # Update LRU order
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = True
            
            self.hits += 1
            self.total_access_time += time.time() - start_time
            return entry.data
    
    def put(self, key: str, value: Any, utility_score: float = 0.0) -> None:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # Calculate entry size if tracking is enabled
            size_bytes = 0
            if self.enable_size_tracking:
                size_bytes = self._estimate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                utility_score=utility_score,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict entries
            self._ensure_capacity(entry)
            
            # Store entry
            self._entries[key] = entry
            self._access_order[key] = True
            self._frequency_counts[key] = 1
    
    def evict(self, count: int) -> List[str]:
        """Evict specified number of items using configured policy."""
        with self._lock:
            if count <= 0 or not self._entries:
                return []
            
            # Select entries for eviction based on policy
            if self.eviction_policy == EvictionPolicy.LRU:
                candidates = self._get_lru_candidates(count)
            elif self.eviction_policy == EvictionPolicy.LFU:
                candidates = self._get_lfu_candidates(count)
            elif self.eviction_policy == EvictionPolicy.UTILITY_BASED:
                candidates = self._get_utility_candidates(count)
            elif self.eviction_policy == EvictionPolicy.HYBRID:
                candidates = self._get_hybrid_candidates(count)
            else:  # RANDOM
                candidates = list(self._entries.keys())[:count]
            
            # Evict selected entries
            evicted_keys = []
            for key in candidates[:count]:
                if key in self._entries:
                    del self._entries[key]
                    if key in self._access_order:
                        del self._access_order[key]
                    if key in self._frequency_counts:
                        del self._frequency_counts[key]
                    evicted_keys.append(key)
                    self.evictions += 1
            
            return evicted_keys
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            cleared_count = len(self._entries)
            self._entries.clear()
            self._access_order.clear()
            self._frequency_counts.clear()
            
            if cleared_count > 0:
                print(f"🧹 Cache cleared: {self.cache_type.value}, {cleared_count} entries removed")
    
    def get_current_size_mb(self) -> float:
        """Get current cache size in MB."""
        if not self.enable_size_tracking:
            return 0.0
        
        total_bytes = sum(entry.size_bytes for entry in self._entries.values())
        return total_bytes / (1024 * 1024)
    
    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            miss_rate = self.misses / max(1, total_requests)
            avg_access_time = (self.total_access_time / max(1, self.total_accesses)) * 1000  # ms
            
            # Calculate memory efficiency (hits per MB)
            current_size = self.get_current_size_mb()
            memory_efficiency = self.hits / max(1, current_size)
            
            return CacheStats(
                cache_type=self.cache_type,
                total_entries=len(self._entries),
                total_size_mb=current_size,
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                eviction_count=self.evictions,
                avg_access_time_ms=avg_access_time,
                memory_efficiency=memory_efficiency
            )
    
    def _ensure_capacity(self, new_entry: CacheEntry) -> None:
        """Ensure cache has capacity for new entry."""
        # Check entry count limit
        if len(self._entries) >= self.max_entries:
            evict_count = len(self._entries) - self.max_entries + 1
            self.evict(evict_count)
        
        # Check size limit
        if self.enable_size_tracking:
            current_size = self.get_current_size_mb()
            new_size = new_entry.size_bytes / (1024 * 1024)
            
            if current_size + new_size > self.max_size_mb:
                # Evict entries until we have enough space
                target_free_space = new_size * 1.2  # 20% buffer
                freed_space = 0.0
                
                while freed_space < target_free_space and self._entries:
                    evicted_keys = self.evict(max(1, len(self._entries) // 10))
                    if not evicted_keys:
                        break
                    
                    # Estimate freed space (simplified)
                    freed_space += len(evicted_keys) * (current_size / len(self._entries))
    
    def _get_lru_candidates(self, count: int) -> List[str]:
        """Get LRU eviction candidates."""
        # Get oldest entries from access order
        candidates = list(self._access_order.keys())
        return candidates[:count]
    
    def _get_lfu_candidates(self, count: int) -> List[str]:
        """Get LFU eviction candidates."""
        # Sort by frequency (ascending)
        sorted_by_freq = sorted(self._entries.keys(), 
                               key=lambda k: self._frequency_counts.get(k, 0))
        return sorted_by_freq[:count]
    
    def _get_utility_candidates(self, count: int) -> List[str]:
        """Get utility-based eviction candidates."""
        # Sort by utility score (ascending - lower utility evicted first)
        sorted_by_utility = sorted(self._entries.keys(),
                                  key=lambda k: self._entries[k].utility_score)
        return sorted_by_utility[:count]
    
    def _get_hybrid_candidates(self, count: int) -> List[str]:
        """Get hybrid eviction candidates combining multiple factors."""
        current_time = time.time()
        
        # Calculate composite eviction priority for each entry
        for key, entry in self._entries.items():
            age_score = current_time - entry.last_accessed  # Higher = older
            freq_score = 1.0 / max(1, entry.access_count)    # Higher = less used
            utility_score = 1.0 - entry.utility_score        # Higher = less useful
            
            # Weighted combination
            entry.eviction_priority = (
                age_score * 0.4 +      # Recent access matters most
                freq_score * 0.3 +     # Frequency of use
                utility_score * 0.3    # Prediction utility
            )
        
        # Sort by eviction priority (ascending - higher priority evicted first)
        sorted_by_priority = sorted(self._entries.keys(),
                                   key=lambda k: self._entries[k].eviction_priority,
                                   reverse=True)
        return sorted_by_priority[:count]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cache value."""
        try:
            if isinstance(value, (list, tuple)):
                return len(value) * 8  # Rough estimate for float lists
            elif isinstance(value, dict):
                return len(str(value))  # Rough estimate
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            else:
                return len(str(value))  # Fallback
        except:
            return 64  # Default estimate


class MemoryManager:
    """
    Global memory manager that coordinates all caches and monitors system memory.
    """
    
    def __init__(self, 
                 max_total_cache_memory_mb: float = 500.0,
                 memory_pressure_threshold: float = 0.8,
                 emergency_cleanup_threshold: float = 0.95,
                 monitoring_interval: float = 30.0):
        """
        Initialize memory manager.
        
        Args:
            max_total_cache_memory_mb: Maximum total cache memory across all caches
            memory_pressure_threshold: System memory pressure threshold (0.0-1.0)
            emergency_cleanup_threshold: Emergency cleanup threshold (0.0-1.0)
            monitoring_interval: Memory monitoring interval in seconds
        """
        self.max_total_cache_memory_mb = max_total_cache_memory_mb
        self.memory_pressure_threshold = memory_pressure_threshold
        self.emergency_cleanup_threshold = emergency_cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
        # Registered caches
        self._caches: Dict[str, IntelligentCache] = {}
        self._cache_priorities: Dict[str, float] = {}  # Higher = more important
        
        # Memory monitoring
        self._last_memory_check = time.time()
        self._memory_history = []
        self._pressure_events = 0
        self._cleanup_events = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"🧠 MemoryManager initialized: max_cache_memory={max_total_cache_memory_mb:.1f}MB, "
              f"pressure_threshold={memory_pressure_threshold:.2f}")
    
    def register_cache(self, 
                      name: str, 
                      cache: IntelligentCache, 
                      priority: float = 1.0) -> None:
        """
        Register a cache for coordinated management.
        
        Args:
            name: Unique cache name
            cache: IntelligentCache instance
            priority: Cache priority (higher = more important, less likely to be evicted)
        """
        with self._lock:
            self._caches[name] = cache
            self._cache_priorities[name] = priority
            
            print(f"📝 Cache registered: {name} ({cache.cache_type.value}), priority={priority:.1f}")
    
    def unregister_cache(self, name: str) -> None:
        """Unregister a cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                del self._cache_priorities[name]
                print(f"📝 Cache unregistered: {name}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current system memory statistics."""
        try:
            memory = psutil.virtual_memory()
            
            # Calculate cache memory usage
            cache_memory = 0.0
            for cache in self._caches.values():
                cache_memory += cache.get_current_size_mb()
            
            # Calculate GPU memory if available
            gpu_memory = 0.0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                except:
                    pass
            
            return MemoryStats(
                total_ram_gb=memory.total / (1024 ** 3),
                available_ram_gb=memory.available / (1024 ** 3),
                used_ram_gb=(memory.total - memory.available) / (1024 ** 3),
                memory_pressure=memory.percent / 100.0,
                brain_memory_usage_mb=cache_memory + gpu_memory,
                cache_memory_usage_mb=cache_memory,
                gpu_memory_usage_mb=gpu_memory
            )
            
        except Exception as e:
            print(f"⚠️ Memory stats error: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0)
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        
        # Check multiple pressure indicators
        system_pressure = stats.memory_pressure > self.memory_pressure_threshold
        cache_pressure = stats.cache_memory_usage_mb > self.max_total_cache_memory_mb
        
        return system_pressure or cache_pressure
    
    def cleanup_caches(self, 
                      target_reduction_mb: float = 0.0, 
                      force_cleanup: bool = False) -> Dict[str, int]:
        """
        Cleanup caches to reduce memory usage.
        
        Args:
            target_reduction_mb: Target memory reduction in MB (0 = automatic)
            force_cleanup: Force cleanup even without pressure
            
        Returns:
            Dictionary of cache_name -> entries_evicted
        """
        with self._lock:
            if not force_cleanup and not self.check_memory_pressure():
                return {}
            
            stats = self.get_memory_stats()
            
            # Calculate target reduction if not specified
            if target_reduction_mb <= 0:
                if stats.cache_memory_usage_mb > self.max_total_cache_memory_mb:
                    target_reduction_mb = stats.cache_memory_usage_mb - self.max_total_cache_memory_mb * 0.8
                else:
                    target_reduction_mb = stats.cache_memory_usage_mb * 0.2  # Reduce by 20%
            
            # Sort caches by priority (lowest priority cleaned first)
            sorted_caches = sorted(self._caches.items(), 
                                 key=lambda x: self._cache_priorities.get(x[0], 1.0))
            
            cleaned_caches = {}
            total_cleaned_mb = 0.0
            
            for cache_name, cache in sorted_caches:
                if total_cleaned_mb >= target_reduction_mb:
                    break
                
                cache_size_before = cache.get_current_size_mb()
                
                # Calculate how much to clean from this cache
                if cache_size_before > 0:
                    cache_reduction_ratio = min(0.5, target_reduction_mb / cache_size_before)
                else:
                    cache_reduction_ratio = 0.2  # Default 20% reduction
                
                entries_to_evict = max(1, int(len(cache._entries) * cache_reduction_ratio))
                
                # Evict entries
                evicted_keys = cache.evict(entries_to_evict)
                cleaned_caches[cache_name] = len(evicted_keys)
                
                cache_size_after = cache.get_current_size_mb()
                cleaned_mb = cache_size_before - cache_size_after
                total_cleaned_mb += cleaned_mb
                
                if evicted_keys:
                    print(f"🧹 Cache cleanup: {cache_name}, evicted {len(evicted_keys)} entries, "
                          f"freed {cleaned_mb:.1f}MB")
            
            self._cleanup_events += 1
            return cleaned_caches
    
    def emergency_cleanup(self) -> None:
        """Emergency memory cleanup when system is critically low on memory."""
        print("🚨 EMERGENCY MEMORY CLEANUP TRIGGERED")
        
        stats = self.get_memory_stats()
        print(f"   Memory pressure: {stats.memory_pressure:.1%}")
        print(f"   Cache memory: {stats.cache_memory_usage_mb:.1f}MB")
        
        # Aggressive cleanup - clear 50% of all caches
        target_reduction = stats.cache_memory_usage_mb * 0.5
        cleaned = self.cleanup_caches(target_reduction, force_cleanup=True)
        
        # If still under pressure, clear entire low-priority caches
        if self.check_memory_pressure():
            sorted_caches = sorted(self._caches.items(), 
                                 key=lambda x: self._cache_priorities.get(x[0], 1.0))
            
            for cache_name, cache in sorted_caches[:len(sorted_caches)//2]:
                cache.clear()
                print(f"🚨 Emergency: Cleared entire cache {cache_name}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and cache statistics."""
        memory_stats = self.get_memory_stats()
        
        cache_stats = {}
        for name, cache in self._caches.items():
            cache_stats[name] = {
                'stats': cache.get_stats(),
                'priority': self._cache_priorities.get(name, 1.0)
            }
        
        return {
            'memory_stats': memory_stats,
            'cache_stats': cache_stats,
            'management_stats': {
                'total_caches': len(self._caches),
                'pressure_events': self._pressure_events,
                'cleanup_events': self._cleanup_events,
                'monitoring_active': self._monitoring_active,
                'max_cache_memory_mb': self.max_total_cache_memory_mb
            }
        }
    
    def _memory_monitor_loop(self) -> None:
        """Background memory monitoring loop."""
        while self._monitoring_active:
            try:
                time.sleep(self.monitoring_interval)
                
                stats = self.get_memory_stats()
                self._memory_history.append((time.time(), stats))
                
                # Keep only recent history
                cutoff_time = time.time() - 3600  # 1 hour
                self._memory_history = [(t, s) for t, s in self._memory_history if t > cutoff_time]
                
                # Check for memory pressure
                if stats.memory_pressure > self.emergency_cleanup_threshold:
                    self.emergency_cleanup()
                elif self.check_memory_pressure():
                    self._pressure_events += 1
                    self.cleanup_caches()
                
            except Exception as e:
                print(f"⚠️ Memory monitoring error: {e}")
    
    def shutdown(self) -> None:
        """Shutdown memory manager."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        print("🛑 MemoryManager shutdown complete")


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    
    return _global_memory_manager


def create_managed_cache(name: str,
                        cache_type: CacheType,
                        max_entries: int = 1000,
                        max_size_mb: float = 50.0,
                        eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
                        priority: float = 1.0,
                        enable_size_tracking: bool = True) -> IntelligentCache:
    """
    Create and register a new managed cache.
    
    Args:
        name: Unique cache name
        cache_type: Type of cache
        max_entries: Maximum cache entries
        max_size_mb: Maximum cache size in MB
        eviction_policy: Eviction strategy
        priority: Cache priority (higher = more important)
        
    Returns:
        IntelligentCache instance
    """
    cache = IntelligentCache(
        cache_type=cache_type,
        max_entries=max_entries,
        max_size_mb=max_size_mb,
        eviction_policy=eviction_policy,
        enable_size_tracking=enable_size_tracking
    )
    
    memory_manager = get_memory_manager()
    memory_manager.register_cache(name, cache, priority)
    
    return cache


def cleanup_all_caches(target_reduction_mb: float = 0.0) -> Dict[str, int]:
    """Cleanup all registered caches."""
    memory_manager = get_memory_manager()
    return memory_manager.cleanup_caches(target_reduction_mb)


def get_system_memory_stats() -> MemoryStats:
    """Get current system memory statistics."""
    memory_manager = get_memory_manager()
    return memory_manager.get_memory_stats()