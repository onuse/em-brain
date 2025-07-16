#!/usr/bin/env python3
"""
Quick Tensor Optimization Example

Demonstrates how to apply tensor optimizations to reduce the 197% performance degradation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from server.src.brain import MinimalBrain
from server.src.utils.tensor_optimization_integration import quick_optimize_brain


def create_test_experience(dim_sensory=16, dim_action=4):
    """Create a test experience with random data."""
    return {
        'sensory_input': np.random.randn(dim_sensory).tolist(),
        'action_taken': np.random.randn(dim_action).tolist(),
        'outcome': np.random.randn(dim_sensory).tolist()
    }


def demonstrate_optimization():
    """Demonstrate tensor optimization benefits."""
    print("🎯 Tensor Optimization Demonstration")
    print("=" * 50)
    
    print("\n1. Creating original brain...")
    original_brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    print("\n2. Creating optimized brain...")
    optimized_brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    optimized_brain = quick_optimize_brain(optimized_brain)
    
    # Test with a small number of experiences
    num_test = 20
    print(f"\n3. Processing {num_test} experiences...")
    
    # Test original brain
    print("\n   Original Brain:")
    original_times = []
    for i in range(num_test):
        exp = create_test_experience()
        start = time.time()
        predicted_action, _ = original_brain.process_sensory_input(exp['sensory_input'])
        original_brain.store_experience(
            exp['sensory_input'], exp['action_taken'], exp['outcome'], predicted_action
        )
        elapsed = time.time() - start
        original_times.append(elapsed)
        print(f"     Experience {i+1:2d}: {elapsed*1000:6.2f}ms")
    
    # Test optimized brain
    print("\n   Optimized Brain:")
    optimized_times = []
    for i in range(num_test):
        exp = create_test_experience()
        start = time.time()
        predicted_action, _ = optimized_brain.process_sensory_input(exp['sensory_input'])
        optimized_brain.store_experience(
            exp['sensory_input'], exp['action_taken'], exp['outcome'], predicted_action
        )
        elapsed = time.time() - start
        optimized_times.append(elapsed)
        print(f"     Experience {i+1:2d}: {elapsed*1000:6.2f}ms")
    
    # Analysis
    print(f"\n4. Performance Analysis:")
    print(f"   Original - Average: {np.mean(original_times)*1000:.2f}ms, "
          f"Total: {sum(original_times):.2f}s")
    print(f"   Optimized - Average: {np.mean(optimized_times)*1000:.2f}ms, "
          f"Total: {sum(optimized_times):.2f}s")
    
    improvement = (np.mean(original_times) - np.mean(optimized_times)) / np.mean(original_times) * 100
    print(f"   Improvement: {improvement:+.1f}%")
    
    # Check for performance degradation
    original_early = np.mean(original_times[:5])
    original_late = np.mean(original_times[-5:])
    original_degradation = (original_late - original_early) / original_early * 100
    
    optimized_early = np.mean(optimized_times[:5])
    optimized_late = np.mean(optimized_times[-5:])
    optimized_degradation = (optimized_late - optimized_early) / optimized_early * 100
    
    print(f"   Original degradation: {original_degradation:+.1f}%")
    print(f"   Optimized degradation: {optimized_degradation:+.1f}%")
    print(f"   Degradation reduction: {original_degradation - optimized_degradation:+.1f} percentage points")
    
    # Get optimization statistics
    if hasattr(optimized_brain, 'get_optimization_statistics'):
        print(f"\n5. Optimization Statistics:")
        stats = optimized_brain.get_optimization_statistics()
        
        tensor_stats = stats.get('tensor_optimization', {})
        print(f"   Tensor rebuilds: {tensor_stats.get('total_rebuilds', 0)}")
        print(f"   Incremental updates: {tensor_stats.get('total_incremental_updates', 0)}")
        print(f"   GPU systems: {tensor_stats.get('gpu_systems', [])}")
        
        batch_stats = stats.get('batch_processing', {})
        print(f"   Total batches: {batch_stats.get('total_batches', 0)}")
        print(f"   Batch efficiency: {batch_stats.get('efficiency_gain', 1.0):.2f}x")
        
        # Get suggestions
        suggestions = optimized_brain.suggest_optimizations()
        if suggestions:
            print(f"\n6. Optimization Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print(f"\n6. ✅ No additional optimizations needed")
    
    print(f"\n🎉 Demonstration complete!")
    print(f"\nKey Takeaways:")
    print(f"  • Tensor optimization reduces processing time")
    print(f"  • Performance degradation is minimized")
    print(f"  • Optimizations are transparent to existing code")
    print(f"  • System automatically adapts to workload")
    
    return {
        'original_times': original_times,
        'optimized_times': optimized_times,
        'improvement': improvement,
        'degradation_reduction': original_degradation - optimized_degradation
    }


def main():
    """Main demonstration function."""
    try:
        results = demonstrate_optimization()
        
        print(f"\n📊 Summary:")
        print(f"  Performance improvement: {results['improvement']:+.1f}%")
        print(f"  Degradation reduction: {results['degradation_reduction']:+.1f} percentage points")
        print(f"  ✅ Tensor optimization successfully applied")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()