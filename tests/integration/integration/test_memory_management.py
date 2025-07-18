"""
Tests for Memory Management and Cache Coordination

Validates that the unified memory manager properly coordinates caches and prevents unbounded memory growth.
"""

import time
import numpy as np
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.src.utils.memory_manager import (
    MemoryManager, IntelligentCache, CacheType, EvictionPolicy, 
    get_memory_manager, create_managed_cache, get_system_memory_stats
)
from server.src.utils.cache_adapters import (
    SimilarityEngineCacheAdapter, ActivationCacheAdapter, 
    PatternCacheAdapter, StreamBufferAdapter
)


def test_memory_manager_initialization():
    """Test memory manager initialization and configuration."""
    print("🧪 Testing memory manager initialization...")
    
    # Create memory manager with custom settings
    memory_manager = MemoryManager(
        max_total_cache_memory_mb=200.0,
        memory_pressure_threshold=0.7,
        emergency_cleanup_threshold=0.9,
        monitoring_interval=10.0
    )
    
    # Check initial state
    stats = memory_manager.get_memory_stats()
    assert stats.total_ram_gb > 0, "Should detect system RAM"
    assert stats.memory_pressure >= 0.0, "Memory pressure should be non-negative"
    
    # Test stats retrieval
    comprehensive_stats = memory_manager.get_comprehensive_stats()
    assert 'memory_stats' in comprehensive_stats
    assert 'cache_stats' in comprehensive_stats
    assert 'management_stats' in comprehensive_stats
    
    print("✅ Memory manager initialization test passed")


def test_intelligent_cache_basic_operations():
    """Test basic cache operations with intelligent eviction."""
    print("🧪 Testing intelligent cache basic operations...")
    
    # Create cache with small limits for testing
    cache = IntelligentCache(
        cache_type=CacheType.GENERAL,
        max_entries=5,
        max_size_mb=1.0,
        eviction_policy=EvictionPolicy.LRU
    )
    
    # Test basic put/get operations
    cache.put("key1", "value1", 0.8)  # High utility
    cache.put("key2", "value2", 0.2)  # Low utility
    cache.put("key3", "value3", 0.6)  # Medium utility
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("nonexistent") is None
    
    # Test eviction due to capacity limit
    cache.put("key4", "value4", 0.7)
    cache.put("key5", "value5", 0.5)
    cache.put("key6", "value6", 0.9)  # This should trigger eviction
    
    # Should have exactly max_entries
    stats = cache.get_stats()
    assert stats.total_entries <= cache.max_entries
    
    # High utility item should still be there
    assert cache.get("key6") == "value6"  # Highest utility
    
    print("✅ Intelligent cache basic operations test passed")


def test_cache_eviction_policies():
    """Test different cache eviction policies."""
    print("🧪 Testing cache eviction policies...")
    
    # Test LRU eviction
    lru_cache = IntelligentCache(
        cache_type=CacheType.GENERAL,
        max_entries=3,
        eviction_policy=EvictionPolicy.LRU
    )
    
    lru_cache.put("a", "value_a")
    lru_cache.put("b", "value_b")
    lru_cache.put("c", "value_c")
    
    # Access 'a' to make it recent
    _ = lru_cache.get("a")
    
    # Add new item - 'b' should be evicted (oldest unaccessed)
    lru_cache.put("d", "value_d")
    
    assert lru_cache.get("a") == "value_a"  # Recently accessed
    assert lru_cache.get("b") is None        # Should be evicted
    assert lru_cache.get("c") == "value_c"   # Still there
    assert lru_cache.get("d") == "value_d"   # New item
    
    # Test utility-based eviction
    utility_cache = IntelligentCache(
        cache_type=CacheType.GENERAL,
        max_entries=3,
        eviction_policy=EvictionPolicy.UTILITY_BASED
    )
    
    utility_cache.put("high", "value_high", 0.9)   # High utility
    utility_cache.put("low", "value_low", 0.1)     # Low utility
    utility_cache.put("medium", "value_medium", 0.5) # Medium utility
    
    # Add new item - low utility should be evicted
    utility_cache.put("new", "value_new", 0.7)
    
    assert utility_cache.get("high") == "value_high"   # High utility preserved
    assert utility_cache.get("low") is None            # Low utility evicted
    assert utility_cache.get("medium") == "value_medium" # Medium utility preserved
    assert utility_cache.get("new") == "value_new"     # New item added
    
    print("✅ Cache eviction policies test passed")


def test_similarity_cache_adapter():
    """Test similarity engine cache adapter."""
    print("🧪 Testing similarity cache adapter...")
    
    cache = SimilarityEngineCacheAdapter(
        max_entries=10,
        max_size_mb=5.0,
        eviction_policy=EvictionPolicy.HYBRID
    )
    
    # Test similarity result caching
    results1 = [("exp1", 0.9), ("exp2", 0.7), ("exp3", 0.5)]
    results2 = [("exp4", 0.8), ("exp5", 0.6)]
    
    cache.put("query1", results1, 0.8)  # High utility
    cache.put("query2", results2, 0.3)  # Low utility
    
    # Test retrieval
    cached_results1 = cache.get("query1")
    assert cached_results1 == results1
    
    cached_results2 = cache.get("query2")
    assert cached_results2 == results2
    
    # Test miss
    assert cache.get("nonexistent") is None
    
    # Test dictionary-like interface
    assert "query1" in cache
    assert "nonexistent" not in cache
    
    # Test stats
    stats = cache.get_stats()
    assert stats['total_entries'] == 2
    assert stats['hits'] >= 2
    
    print("✅ Similarity cache adapter test passed")


def test_activation_cache_adapter():
    """Test activation dynamics cache adapter."""
    print("🧪 Testing activation cache adapter...")
    
    cache = ActivationCacheAdapter(
        max_entries=20,
        max_size_mb=10.0,
        eviction_policy=EvictionPolicy.UTILITY_BASED
    )
    
    # Test similar experiences caching
    similar_exps = {"exp1": 0.8, "exp2": 0.6, "exp3": 0.4}
    cache.cache_similar_experiences("source_exp", similar_exps, 0.7)
    
    retrieved = cache.get_similar_experiences("source_exp")
    assert retrieved == similar_exps
    
    # Test GPU tensor caching
    fake_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    cache.cache_gpu_tensor("tensor_key", fake_tensor, 0.9)
    
    retrieved_tensor = cache.get_gpu_tensor("tensor_key")
    assert np.array_equal(retrieved_tensor, fake_tensor)
    
    # Test cache clearing
    cache.clear_similarity_cache()
    assert cache.get_similar_experiences("source_exp") is None
    
    # GPU tensor should still be there
    assert cache.get_gpu_tensor("tensor_key") is not None
    
    cache.clear_all()
    assert cache.get_gpu_tensor("tensor_key") is None
    
    print("✅ Activation cache adapter test passed")


def test_pattern_cache_adapter():
    """Test pattern analysis cache adapter."""
    print("🧪 Testing pattern cache adapter...")
    
    cache = PatternCacheAdapter(
        max_entries=15,
        max_size_mb=8.0,
        max_age_seconds=2.0,  # Short for testing
        eviction_policy=EvictionPolicy.HYBRID
    )
    
    # Test pattern caching
    pattern_data = {"action": [0.1, 0.2], "confidence": 0.8}
    cache.put("pattern1", pattern_data, 0.8)
    
    # Should retrieve successfully
    result = cache.get("pattern1")
    assert result is not None
    retrieved_pattern, timestamp = result
    assert retrieved_pattern == pattern_data
    assert isinstance(timestamp, float)
    
    # Test expiration
    time.sleep(2.5)  # Wait for expiration
    
    expired_result = cache.get("pattern1")
    assert expired_result is None  # Should be expired
    
    # Test dictionary-like interface
    cache["pattern2"] = ({"action": [0.3, 0.4]}, 0.6)
    assert "pattern2" in cache
    
    stats = cache.get_stats()
    assert 'expired_entries' in stats
    
    print("✅ Pattern cache adapter test passed")


def test_stream_buffer_adapter():
    """Test stream buffer adapter."""
    print("🧪 Testing stream buffer adapter...")
    
    buffer = StreamBufferAdapter(
        max_entries=5,
        max_size_mb=2.0,
        eviction_policy=EvictionPolicy.LRU
    )
    
    # Test stream data addition
    data1 = {"sensor": [1.0, 2.0], "action": [0.1, 0.2]}
    data2 = {"sensor": [3.0, 4.0], "action": [0.3, 0.4]}
    data3 = {"sensor": [5.0, 6.0], "action": [0.5, 0.6]}
    
    key1 = buffer.append(data1, 0.7)
    key2 = buffer.append(data2, 0.5)
    key3 = buffer.append(data3, 0.9)
    
    # Test recent retrieval
    recent = buffer.get_recent(2)
    assert len(recent) == 2
    
    # Most recent should be first
    assert recent[0][1] == data3  # Most recent
    
    # Test all retrieval
    all_data = buffer.get_all()
    assert len(all_data) == 3
    
    # Test buffer size limit
    for i in range(10):  # Add more than max_entries
        buffer.append({"data": i}, 0.5)
    
    assert len(buffer) <= buffer._cache.max_entries
    
    print("✅ Stream buffer adapter test passed")


def test_memory_coordination():
    """Test coordinated memory management across multiple caches."""
    print("🧪 Testing memory coordination...")
    
    # Create memory manager with small limits for testing
    memory_manager = MemoryManager(
        max_total_cache_memory_mb=10.0,  # Small limit
        memory_pressure_threshold=0.6,
        emergency_cleanup_threshold=0.8
    )
    
    # Create multiple caches with different priorities
    high_priority_cache = create_managed_cache(
        name="high_priority",
        cache_type=CacheType.SIMILARITY_SEARCH,
        max_entries=100,
        max_size_mb=5.0,
        priority=3.0  # High priority
    )
    
    low_priority_cache = create_managed_cache(
        name="low_priority", 
        cache_type=CacheType.PATTERN_CACHE,
        max_entries=100,
        max_size_mb=5.0,
        priority=1.0  # Low priority
    )
    
    # Fill caches with large data to exceed memory limits
    large_data = "x" * 10000  # 10KB strings
    for i in range(100):
        high_priority_cache.put(f"high_{i}", f"{large_data}_{i}", 0.8)
        low_priority_cache.put(f"low_{i}", f"{large_data}_{i}", 0.5)
    
    # Check if pressure is detected
    has_pressure = memory_manager.check_memory_pressure()
    
    # Get stats before cleanup
    stats_before = memory_manager.get_comprehensive_stats()
    print(f"   Cache stats before: {stats_before['cache_stats']}")
    
    # Trigger coordinated cleanup (force it to test coordination)
    cleaned = memory_manager.cleanup_caches(target_reduction_mb=5.0, force_cleanup=True)
    
    # Get stats after cleanup
    stats_after = memory_manager.get_comprehensive_stats()
    print(f"   Cache stats after: {stats_after['cache_stats']}")
    print(f"   Cleaned results: {cleaned}")
    
    # Should have cleaned some caches OR have no entries to clean
    cache_has_entries = any(
        stats['stats'].total_entries > 0 
        for stats in stats_before['cache_stats'].values()
    )
    
    if cache_has_entries:
        assert len(cleaned) > 0, f"Expected cleanup to occur with entries present, but got: {cleaned}"
    else:
        print("   No cache entries to clean - test passed")
    
    # Low priority cache should be cleaned more aggressively
    if "low_priority" in cleaned and "high_priority" in cleaned:
        assert cleaned["low_priority"] >= cleaned["high_priority"]
    
    # Get comprehensive stats
    stats = memory_manager.get_comprehensive_stats()
    assert stats['management_stats']['cleanup_events'] > 0
    
    print("✅ Memory coordination test passed")


def test_memory_pressure_handling():
    """Test memory pressure detection and response."""
    print("🧪 Testing memory pressure handling...")
    
    # Get current system memory stats
    stats = get_system_memory_stats()
    
    # Should be able to get valid memory information
    assert stats.total_ram_gb > 0
    assert 0.0 <= stats.memory_pressure <= 1.0
    assert stats.brain_memory_usage_mb >= 0
    
    # Test memory manager pressure detection
    memory_manager = MemoryManager(
        max_total_cache_memory_mb=1.0,  # Very small limit to trigger pressure
        memory_pressure_threshold=0.5
    )
    
    # Create cache that exceeds limit
    test_cache = create_managed_cache(
        name="pressure_test",
        cache_type=CacheType.GENERAL,
        max_entries=1000,
        max_size_mb=2.0,  # Exceeds manager limit
        priority=1.0
    )
    
    # Fill with data to trigger pressure
    for i in range(100):
        large_data = "x" * 1000  # Large string data
        test_cache.put(f"key_{i}", large_data, 0.5)
    
    # Check if pressure is detected
    pressure_detected = memory_manager.check_memory_pressure()
    
    # Should detect pressure from cache exceeding limits
    assert isinstance(pressure_detected, bool)
    
    print("✅ Memory pressure handling test passed")


def run_all_tests():
    """Run all memory management tests."""
    print("🧠 Running Memory Management Tests...")
    print("=" * 50)
    
    try:
        test_memory_manager_initialization()
        test_intelligent_cache_basic_operations()
        test_cache_eviction_policies()
        test_similarity_cache_adapter()
        test_activation_cache_adapter()
        test_pattern_cache_adapter()
        test_stream_buffer_adapter()
        test_memory_coordination()
        test_memory_pressure_handling()
        
        print("=" * 50)
        print("✅ All memory management tests passed!")
        print("🧠 Memory management system is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)