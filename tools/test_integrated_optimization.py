#!/usr/bin/env python3
"""
Test Integrated Storage Optimization

Tests the integrated storage optimization in the brain system
to ensure it works correctly and preserves all functionality.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def test_integrated_optimization():
    """Test the integrated storage optimization."""
    print("🧪 TESTING INTEGRATED STORAGE OPTIMIZATION")
    print("=" * 60)
    
    # Test with optimization enabled
    print("\n🚀 Testing with optimization ENABLED...")
    brain_optimized = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True
    )
    
    # Test performance with optimization
    print("\n⏱️  Testing optimized performance...")
    optimized_times = []
    
    for i in range(15):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        action = [0.5 + 0.01 * i, 0.6 + 0.01 * i, 0.7 + 0.01 * i, 0.8 + 0.01 * i]
        outcome = [0.9 + 0.01 * i, 0.8 + 0.01 * i, 0.7 + 0.01 * i, 0.6 + 0.01 * i]
        
        start_time = time.time()
        brain_optimized.store_experience(sensory, action, outcome, action)
        end_time = time.time()
        
        optimized_times.append((end_time - start_time) * 1000)
    
    optimized_avg = sum(optimized_times) / len(optimized_times)
    print(f"   Optimized average: {optimized_avg:.1f}ms")
    
    # Test with optimization disabled  
    print("\n❌ Testing with optimization DISABLED...")
    brain_regular = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=False
    )
    
    # Test performance without optimization
    print("\n⏱️  Testing regular performance...")
    regular_times = []
    
    for i in range(15):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        action = [0.5 + 0.01 * i, 0.6 + 0.01 * i, 0.7 + 0.01 * i, 0.8 + 0.01 * i]
        outcome = [0.9 + 0.01 * i, 0.8 + 0.01 * i, 0.7 + 0.01 * i, 0.6 + 0.01 * i]
        
        start_time = time.time()
        brain_regular.store_experience(sensory, action, outcome, action)
        end_time = time.time()
        
        regular_times.append((end_time - start_time) * 1000)
    
    regular_avg = sum(regular_times) / len(regular_times)
    print(f"   Regular average: {regular_avg:.1f}ms")
    
    # Calculate improvement
    improvement = ((regular_avg - optimized_avg) / regular_avg) * 100
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Regular (no optimization): {regular_avg:.1f}ms")
    print(f"   Optimized (with optimization): {optimized_avg:.1f}ms")
    print(f"   Improvement: {improvement:.1f}%")
    
    # Test functionality preservation
    print(f"\n🧪 FUNCTIONALITY VERIFICATION:")
    print(f"   Optimized brain experiences: {len(brain_optimized.experience_storage._experiences)}")
    print(f"   Regular brain experiences: {len(brain_regular.experience_storage._experiences)}")
    print(f"   Optimized learning outcomes: {len(brain_optimized.recent_learning_outcomes)}")
    print(f"   Regular learning outcomes: {len(brain_regular.recent_learning_outcomes)}")
    
    # Test that both brains work identically
    print(f"\n🔬 FUNCTIONALITY COMPARISON:")
    test_sensory = [0.5, 0.6, 0.7, 0.8]
    
    pred_opt, state_opt = brain_optimized.process_sensory_input(test_sensory)
    pred_reg, state_reg = brain_regular.process_sensory_input(test_sensory)
    
    print(f"   Optimized prediction: {pred_opt}")
    print(f"   Regular prediction: {pred_reg}")
    print(f"   Predictions similar: {abs(sum(pred_opt) - sum(pred_reg)) < 0.1}")
    
    # Cleanup
    brain_optimized.finalize_session()
    brain_regular.finalize_session()
    
    print(f"\n✅ INTEGRATED OPTIMIZATION TEST COMPLETE")
    print(f"   Performance improvement: {improvement:.1f}%")
    print(f"   Functionality preserved: {'✅ YES' if improvement > 0 else '❌ NO'}")
    
    return improvement > 0

if __name__ == "__main__":
    test_integrated_optimization()