#!/usr/bin/env python3
"""
Quick test of GPU optimization implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

try:
    # Test imports
    print("Testing GPU optimization imports...")
    
    from brains.field import create_brain, GPU_OPTIMIZATION_AVAILABLE
    print(f"✅ GPU optimization available: {GPU_OPTIMIZATION_AVAILABLE}")
    
    if GPU_OPTIMIZATION_AVAILABLE:
        from brains.field import create_optimized_brain, quick_performance_test
        print("✅ GPU components imported successfully")
    
    # Test brain creation
    print("\nTesting brain creation...")
    brain = create_brain(sensory_dim=8, motor_dim=3, quiet_mode=True)
    print(f"✅ Created brain: {brain.__class__.__name__} on {brain.device}")
    
    # Test basic cycle
    print("\nTesting processing cycle...")
    sensory_input = [0.1, 0.2, -0.1, 0.05, 0.0, 0.15, -0.05, 0.1, 0.0]  # 8 sensors + reward
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    print(f"✅ Cycle completed: {len(motor_output)} motor outputs")
    print(f"   Cycle time: {brain_state.get('cycle_time_ms', 0):.2f}ms")
    
    # Test performance if GPU available
    if GPU_OPTIMIZATION_AVAILABLE and hasattr(brain, 'get_performance_stats'):
        perf_stats = brain.get_performance_stats()
        print(f"✅ Performance stats: {perf_stats.get('optimization', 'standard')}")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)