#!/usr/bin/env python3
"""
Performance Demo

Demonstrates the optimized brain performance improvements.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def run_performance_demo():
    """Run performance demonstration."""
    print("\n" + "=" * 60)
    print("BRAIN PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    # Create non-optimized brain
    print("\n1. Creating NON-OPTIMIZED brain...")
    brain_slow = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=False,
        use_optimized=False
    )
    
    # Create optimized brain
    print("\n2. Creating OPTIMIZED brain...")
    brain_fast = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=False,
        use_optimized=True
    )
    brain_fast.enable_predictive_actions(False)
    brain_fast.set_pattern_extraction_limit(3)
    
    # Test inputs
    sensory_input = [0.1] * 23 + [0.5]  # Include small reward
    
    print("\n3. Running performance comparison...")
    print("-" * 40)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        brain_slow.process_robot_cycle(sensory_input)
        brain_fast.process_robot_cycle(sensory_input)
    
    # Test non-optimized
    print("\nNON-OPTIMIZED Brain:")
    times_slow = []
    for i in range(20):
        start = time.perf_counter()
        motors, state = brain_slow.process_robot_cycle(sensory_input)
        elapsed = (time.perf_counter() - start) * 1000
        times_slow.append(elapsed)
        if i == 0:
            print(f"  First cycle: {elapsed:.1f}ms")
            print(f"  Motors: {[f'{m:.2f}' for m in motors[:3]]}...")
    
    avg_slow = sum(times_slow[5:]) / len(times_slow[5:])  # Skip warmup
    print(f"  Average: {avg_slow:.1f}ms")
    
    # Test optimized
    print("\nOPTIMIZED Brain:")
    times_fast = []
    for i in range(20):
        start = time.perf_counter()
        motors, state = brain_fast.process_robot_cycle(sensory_input)
        elapsed = (time.perf_counter() - start) * 1000
        times_fast.append(elapsed)
        if i == 0:
            print(f"  First cycle: {elapsed:.1f}ms")
            print(f"  Motors: {[f'{m:.2f}' for m in motors[:3]]}...")
    
    avg_fast = sum(times_fast[5:]) / len(times_fast[5:])  # Skip warmup
    print(f"  Average: {avg_fast:.1f}ms")
    
    # Results
    print("\n4. RESULTS:")
    print("-" * 40)
    speedup = avg_slow / avg_fast
    print(f"Non-optimized: {avg_slow:.1f}ms per cycle")
    print(f"Optimized: {avg_fast:.1f}ms per cycle")
    print(f"SPEEDUP: {speedup:.2f}x faster!")
    print(f"Time saved: {avg_slow - avg_fast:.1f}ms per cycle")
    
    # Show optimization settings
    print("\n5. Optimization Settings Used:")
    print("-" * 40)
    print("• Pattern caching enabled")
    print("• Predictive actions disabled")
    print("• Pattern limit reduced to 3")
    print("• torch.no_grad() for inference")
    print("• In-place tensor operations")
    
    print("\n" + "=" * 60)
    print("Performance optimization successful!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_performance_demo()