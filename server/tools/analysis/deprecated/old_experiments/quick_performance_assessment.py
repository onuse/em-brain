#!/usr/bin/env python3
"""
Quick Performance Assessment

Rapidly assess current performance state and identify next optimization targets.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def quick_performance_assessment():
    """Quick assessment of current performance state."""
    print("⚡ QUICK PERFORMANCE ASSESSMENT")
    print("=" * 50)
    
    # Create optimized brain
    print("🚀 Testing optimized brain...")
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True
    )
    
    # Add modest number of experiences
    print("📊 Adding 25 experiences...")
    for i in range(25):
        sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
        action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
        outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
        brain.store_experience(sensory, action, outcome, action)
    
    # Test current performance
    print("⏱️  Testing 5 complete brain cycles...")
    
    cycle_times = []
    component_times = {}
    
    for i in range(5):
        test_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        
        # Time full cycle
        cycle_start = time.time()
        
        # Time process_sensory_input
        process_start = time.time()
        predicted_action, brain_state = brain.process_sensory_input(test_sensory)
        process_time = (time.time() - process_start) * 1000
        
        # Time store_experience (optimized)
        store_start = time.time()
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(test_sensory, predicted_action, outcome, predicted_action)
        store_time = (time.time() - store_start) * 1000
        
        cycle_end = time.time()
        total_cycle_time = (cycle_end - cycle_start) * 1000
        
        cycle_times.append(total_cycle_time)
        
        if 'process_sensory_input' not in component_times:
            component_times['process_sensory_input'] = []
            component_times['store_experience'] = []
        
        component_times['process_sensory_input'].append(process_time)
        component_times['store_experience'].append(store_time)
        
        print(f"   Cycle {i+1}: {total_cycle_time:.1f}ms (process: {process_time:.1f}ms, store: {store_time:.1f}ms)")
    
    # Calculate averages
    avg_total = sum(cycle_times) / len(cycle_times)
    avg_process = sum(component_times['process_sensory_input']) / len(component_times['process_sensory_input'])
    avg_store = sum(component_times['store_experience']) / len(component_times['store_experience'])
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Average total cycle: {avg_total:.1f}ms")
    print(f"   Average process_sensory_input: {avg_process:.1f}ms")
    print(f"   Average store_experience: {avg_store:.1f}ms")
    print(f"   Real-time ready: {'✅ YES' if avg_total < 100 else '❌ NO'}")
    
    # Identify biggest bottleneck
    if avg_process > avg_store:
        print(f"\n🎯 BIGGEST BOTTLENECK: process_sensory_input ({avg_process:.1f}ms)")
        print(f"   This includes activation system, prediction engine, cognitive autopilot")
        print(f"   Next optimization target: Activation system (68.5ms in profiling)")
    else:
        print(f"\n🎯 BIGGEST BOTTLENECK: store_experience ({avg_store:.1f}ms)")
        print(f"   Storage optimization may need further tuning")
    
    # Quick calculation of remaining optimization potential
    original_estimated = 177  # From biological validation
    current_performance = avg_total
    improvement_achieved = ((original_estimated - current_performance) / original_estimated) * 100
    
    print(f"\n📈 OPTIMIZATION PROGRESS:")
    print(f"   Original performance: ~177ms (from biological validation)")
    print(f"   Current performance: {current_performance:.1f}ms")
    print(f"   Improvement achieved: {improvement_achieved:.1f}%")
    
    if current_performance < 100:
        print(f"   🎉 SUCCESS: Real-time performance achieved!")
    else:
        remaining_needed = ((current_performance - 100) / current_performance) * 100
        print(f"   🔧 Need {remaining_needed:.1f}% more improvement for real-time")
    
    brain.finalize_session()
    
    return avg_total < 100, avg_process, avg_store

def recommend_next_optimization(avg_process: float, avg_store: float):
    """Recommend the next optimization based on results."""
    print(f"\n🗺️  NEXT OPTIMIZATION RECOMMENDATION:")
    
    if avg_process > 30:  # If process_sensory_input is still slow
        print(f"🎯 PRIORITY 1: Optimize Activation System")
        print(f"   Current: {avg_process:.1f}ms in process_sensory_input")
        print(f"   Target: Reduce to <20ms")
        print(f"   Strategy: GPU batch operations, sparse activation updates")
        print(f"   Expected gain: 30-50% reduction")
        print(f"   Intelligence impact: LOW (computational optimization)")
        
        print(f"\n🔧 Activation System Optimization Opportunities:")
        print(f"   - Batch GPU operations for utility computation")
        print(f"   - Use sparse tensor operations")
        print(f"   - Cache activation computations")
        print(f"   - Optimize working memory retrieval")
        
    elif avg_store > 10:  # If storage still needs work
        print(f"🎯 PRIORITY 1: Further Storage Optimization")
        print(f"   Current: {avg_store:.1f}ms")
        print(f"   Target: Reduce to <5ms")
        print(f"   Strategy: Larger batches, better async handling")
        
    else:
        print(f"🎉 Both systems optimized! Look for other bottlenecks:")
        print(f"   - Cognitive autopilot optimization")
        print(f"   - Prediction engine fine-tuning")
        print(f"   - GPU memory management")

def main():
    """Run quick performance assessment."""
    real_time_ready, avg_process, avg_store = quick_performance_assessment()
    
    if not real_time_ready:
        recommend_next_optimization(avg_process, avg_store)
    else:
        print(f"\n🏆 ACHIEVEMENT UNLOCKED: Real-time brain performance!")

if __name__ == "__main__":
    main()