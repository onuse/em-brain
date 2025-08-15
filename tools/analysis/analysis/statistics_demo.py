#!/usr/bin/env python3
"""
Statistics Control System Demo

This demonstrates how to use the statistics control system to avoid
performance degradation from expensive statistics collection.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
from src.statistics_control import (
    enable_production_mode, 
    enable_investigation_mode, 
    enable_performance_mode,
    print_current_config
)

def benchmark_brain_performance(mode_name: str, num_predictions: int = 10):
    """Benchmark brain performance with current statistics settings."""
    print(f"\n🔬 BENCHMARKING {mode_name.upper()} MODE")
    print("=" * (15 + len(mode_name)))
    
    # Create brain
    brain = MinimalBrain(quiet_mode=True)
    
    # Test input
    test_input = [1.0, 2.0, 3.0, 4.0]
    
    # Warm up (first prediction might be slower)
    brain.process_sensory_input(test_input, action_dimensions=2)
    
    # Benchmark predictions
    times = []
    for i in range(num_predictions):
        start_time = time.time()
        action, brain_state = brain.process_sensory_input(test_input, action_dimensions=2)
        elapsed = (time.time() - start_time) * 1000
        times.append(elapsed)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Predictions: {num_predictions}")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"Max time: {max_time:.2f}ms")
    
    # Show what statistics are available
    stats = brain.get_brain_stats()
    vector_stats = stats['vector_brain']
    
    print(f"\nAvailable statistics:")
    print(f"  Core stats: ✅ (total_cycles, architecture, etc.)")
    print(f"  Stream stats: {'✅' if 'streams' in vector_stats else '❌'}")
    print(f"  Coactivation stats: {'✅' if 'cross_stream' in vector_stats else '❌'}")
    print(f"  Hierarchy stats: {'✅' if 'temporal_hierarchy' in vector_stats else '❌'}")
    print(f"  Competition stats: {'✅' if 'competitive_dynamics' in vector_stats else '❌'}")
    
    return avg_time

def main():
    """Demonstrate statistics control system."""
    
    print("📊 STATISTICS CONTROL SYSTEM DEMO")
    print("=" * 40)
    
    # Test 1: Production mode (minimal statistics)
    print("\n🚀 PRODUCTION MODE - Maximum Performance")
    enable_production_mode()
    print_current_config()
    production_time = benchmark_brain_performance("production", 10)
    
    # Test 2: Performance investigation mode
    print("\n⚡ PERFORMANCE INVESTIGATION MODE")
    enable_performance_mode()
    print_current_config()
    perf_time = benchmark_brain_performance("performance", 10)
    
    # Test 3: Full investigation mode (all statistics)
    print("\n🔍 FULL INVESTIGATION MODE - All Statistics")
    enable_investigation_mode()
    print_current_config()
    investigation_time = benchmark_brain_performance("investigation", 10)
    
    # Summary
    print(f"\n📈 PERFORMANCE IMPACT SUMMARY")
    print("=" * 35)
    print(f"Production mode: {production_time:.2f}ms (baseline)")
    print(f"Performance mode: {perf_time:.2f}ms ({perf_time/production_time:.1f}x)")
    print(f"Investigation mode: {investigation_time:.2f}ms ({investigation_time/production_time:.1f}x)")
    
    if investigation_time > production_time * 2:
        print(f"\n⚠️  Investigation mode is {investigation_time/production_time:.1f}x slower!")
        print("   Only enable for debugging/investigation purposes")
    else:
        print(f"\n✅ Statistics collection impact is minimal")
    
    # Reset to production mode
    enable_production_mode()
    print(f"\n✅ Reset to production mode for normal operation")

if __name__ == "__main__":
    main()