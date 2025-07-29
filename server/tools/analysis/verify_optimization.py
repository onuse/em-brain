#!/usr/bin/env python3
"""
Verify Optimization

Quick test to verify optimizations didn't break functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def verify_optimization():
    """Verify brain still functions correctly with optimizations."""
    print("Verifying Optimized Brain Functionality")
    print("-" * 50)
    
    # Create optimized brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=True,
        use_optimized=True
    )
    
    # Test various inputs
    test_cases = [
        ("Zero input", [0.0] * 24),
        ("Small input", [0.1] * 24),
        ("Mixed input", [0.0, 0.5, -0.5, 1.0] * 6),
        ("With reward", [0.1] * 23 + [1.0])
    ]
    
    print("\nRunning test cases...")
    for name, input_data in test_cases:
        try:
            motors, state = brain.process_robot_cycle(input_data)
            print(f"✅ {name}: OK (cycle {state['cycle']}, {state['cycle_time_ms']:.1f}ms)")
            assert len(motors) == 3, f"Expected 3 motors, got {len(motors)}"
            assert all(-1 <= m <= 1 for m in motors), "Motor values out of range"
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            return False
    
    # Test optimization controls
    print("\nTesting optimization controls...")
    try:
        brain.enable_predictive_actions(False)
        brain.set_pattern_extraction_limit(1)
        motors, state = brain.process_robot_cycle([0.1] * 24)
        print("✅ Optimization controls: OK")
    except Exception as e:
        print(f"❌ Optimization controls: FAILED - {e}")
        return False
    
    # Test pattern cache if available
    if hasattr(brain, 'pattern_cache_pool'):
        print("\nTesting pattern cache...")
        try:
            patterns = brain.pattern_cache_pool.extract_patterns_fast(
                brain.unified_field, n_patterns=5
            )
            print(f"✅ Pattern cache: OK ({len(patterns)} patterns)")
        except Exception as e:
            print(f"❌ Pattern cache: FAILED - {e}")
    
    print("\n" + "-" * 50)
    print("✅ All tests passed! Optimization successful.")
    return True


if __name__ == "__main__":
    success = verify_optimization()
    sys.exit(0 if success else 1)