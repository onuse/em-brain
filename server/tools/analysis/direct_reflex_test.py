#!/usr/bin/env python3
"""
Direct Reflex Test - Test the fast reflex mode directly

This test bypasses the full brain pipeline and tests the fast reflex mode
directly at the temporal hierarchy level.
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

from vector_stream.unified_cortical_storage import UnifiedCorticalStorage
from vector_stream.emergent_temporal_constraints import EmergentTemporalHierarchy
from vector_stream.sparse_representations import SparsePatternEncoder
import torch

def test_direct_reflex_mode():
    """Test fast reflex mode directly."""
    print("🏃 DIRECT REFLEX MODE TEST")
    print("=" * 40)
    
    # Create unified storage and temporal hierarchy
    unified_storage = UnifiedCorticalStorage(
        pattern_dim=16,
        max_patterns=1000,
        max_columns=100,
        quiet_mode=True
    )
    
    hierarchy = EmergentTemporalHierarchy(unified_storage, quiet_mode=True)
    
    # Create encoder for sparse patterns
    encoder = SparsePatternEncoder(16, sparsity=0.02, quiet_mode=True)
    
    # Create test pattern
    test_vector = torch.tensor([1.0, 2.0, 3.0, 4.0] + [0.0] * 12)
    test_pattern = encoder.encode_top_k(test_vector, "test_pattern")
    
    current_time = time.time()
    
    print("Testing reflex mode performance...")
    
    # Test multiple predictions to measure cache effectiveness
    times = []
    
    for i in range(10):
        start_time = time.time()
        
        # Test the specific budget selection
        budget = hierarchy._select_budget_adaptively(test_pattern, current_time)
        
        # Process with adaptive budget
        result = hierarchy.process_with_adaptive_budget(test_pattern, current_time)
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        times.append(elapsed)
        
        print(f"  Prediction {i+1}: {elapsed:.1f}ms, Budget: {budget}, Strategy: {result['prediction_info'].get('strategy', 'unknown')}")
    
    # Get hierarchy stats
    stats = hierarchy.get_hierarchy_stats()
    reflex_cache = stats.get('reflex_cache', {})
    
    print(f"\nReflex cache statistics:")
    print(f"  Cache size: {reflex_cache.get('cache_size', 0)}")
    print(f"  Cache hits: {reflex_cache.get('cache_hits', 0)}")
    print(f"  Cache misses: {reflex_cache.get('cache_misses', 0)}")
    print(f"  Hit rate: {reflex_cache.get('cache_hit_rate', 0):.2f}")
    
    print(f"\nTiming analysis:")
    print(f"  First prediction: {times[0]:.1f}ms")
    print(f"  Average time: {sum(times)/len(times):.1f}ms")
    print(f"  Fastest time: {min(times):.1f}ms")
    print(f"  Slowest time: {max(times):.1f}ms")
    
    # Test with different pattern to verify cache behavior
    print(f"\nTesting with different pattern...")
    
    different_vector = torch.tensor([5.0, 6.0, 7.0, 8.0] + [0.0] * 12)
    different_pattern = encoder.encode_top_k(different_vector, "different_pattern")
    
    different_times = []
    for i in range(5):
        start_time = time.time()
        result = hierarchy.process_with_adaptive_budget(different_pattern, current_time)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        different_times.append(elapsed)
        
        print(f"  Different pattern {i+1}: {elapsed:.1f}ms")
    
    # Final stats
    final_stats = hierarchy.get_hierarchy_stats()
    final_cache = final_stats.get('reflex_cache', {})
    
    print(f"\nFinal reflex cache statistics:")
    print(f"  Cache size: {final_cache.get('cache_size', 0)}")
    print(f"  Cache hits: {final_cache.get('cache_hits', 0)}")
    print(f"  Cache misses: {final_cache.get('cache_misses', 0)}")
    print(f"  Hit rate: {final_cache.get('cache_hit_rate', 0):.2f}")
    
    # Performance analysis
    fastest_overall = min(min(times), min(different_times))
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print(f"=" * 30)
    print(f"Fastest prediction: {fastest_overall:.1f}ms")
    print(f"Target: 10.0ms (true reflex speed)")
    
    if fastest_overall < 10.0:
        print(f"✅ TRUE REFLEX SPEED ACHIEVED!")
    elif fastest_overall < 100.0:
        print(f"✅ FAST REFLEX MODE WORKING")
    else:
        print(f"❌ REFLEX MODE NEEDS OPTIMIZATION")
    
    return fastest_overall

if __name__ == "__main__":
    fastest_time = test_direct_reflex_mode()
    print(f"\n🏁 DIRECT REFLEX TEST COMPLETE")
    print(f"=" * 35)
    print(f"Fastest prediction time: {fastest_time:.1f}ms")