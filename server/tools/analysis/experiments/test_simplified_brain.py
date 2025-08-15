#!/usr/bin/env python3
"""
Test Simplified Brain Performance

Compare the simplified 4D tensor brain with the complex 11D version.
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from src.core.simplified_brain_factory import SimplifiedBrainFactory
from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_brain_performance(brain_wrapper, brain_type: str, num_cycles: int = 10):
    """Test brain performance over multiple cycles."""
    print(f"\n🧪 Testing {brain_type}")
    print("-" * 50)
    
    # Warm up
    for _ in range(3):
        sensory_input = [0.5] * 24 + [0.0]  # 24 sensors + no reward
        brain_wrapper.brain.process_robot_cycle(sensory_input)
    
    # Test cycles
    cycle_times = []
    for i in range(num_cycles):
        # Varied sensory input
        sensory_input = [
            np.sin(i * 0.1),
            np.cos(i * 0.1),
            0.5
        ] + [0.2] * 21 + [0.0]
        
        start_time = time.perf_counter()
        action, state = brain_wrapper.brain.process_robot_cycle(sensory_input)
        cycle_time = (time.perf_counter() - start_time) * 1000
        cycle_times.append(cycle_time)
        
        if i == 0:
            print(f"   First action: {[f'{a:.3f}' for a in action[:4]]}")
            print(f"   Device: {state.get('device', 'unknown')}")
            print(f"   Tensor shape: {state.get('tensor_shape', 'unknown')}")
    
    # Calculate statistics
    avg_time = np.mean(cycle_times)
    min_time = np.min(cycle_times)
    max_time = np.max(cycle_times)
    
    print(f"\n   Performance over {num_cycles} cycles:")
    print(f"   Average: {avg_time:.1f}ms")
    print(f"   Min: {min_time:.1f}ms")
    print(f"   Max: {max_time:.1f}ms")
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'cycle_times': cycle_times
    }


def test_gpu_acceleration():
    """Test GPU acceleration with simplified brain."""
    print("🚀 GPU Acceleration Test")
    print("=" * 60)
    
    # Check available devices
    print("\n📱 Available devices:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Simplified Brain (4D tensor)
    print("\n1️⃣ SIMPLIFIED BRAIN (4D Tensor)")
    simplified_factory = SimplifiedBrainFactory({'quiet_mode': False})
    simplified_brain = simplified_factory.create(
        sensory_dim=24,
        motor_dim=5,
        spatial_resolution=32
    )
    
    simplified_results = test_brain_performance(simplified_brain, "Simplified Brain")
    
    # Test 2: Complex Brain (11D tensor)
    print("\n2️⃣ COMPLEX BRAIN (11D Tensor)")
    complex_factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'quiet_mode': False,
        'field_spatial_resolution': 4
    })
    complex_brain = complex_factory.create(
        field_dimensions=None,  # Let it calculate
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=5
    )
    
    complex_results = test_brain_performance(complex_brain, "Complex Brain", num_cycles=5)
    
    # Compare results
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    speedup = complex_results['avg_time'] / simplified_results['avg_time']
    print(f"   Simplified: {simplified_results['avg_time']:.1f}ms average")
    print(f"   Complex: {complex_results['avg_time']:.1f}ms average")
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup > 2.0:
        print("\n✅ Significant performance improvement!")
    elif speedup > 1.5:
        print("\n✅ Good performance improvement!")
    else:
        print("\n⚠️  Modest performance improvement")
    
    # Test reward processing
    print("\n3️⃣ REWARD PROCESSING TEST")
    print("-" * 50)
    
    # Give positive reward
    reward_input = [1.0, 0.0, 0.5] + [0.2] * 21 + [0.8]  # Strong positive reward
    
    print("   Testing reward processing on simplified brain...")
    action, state = simplified_brain.brain.process_robot_cycle(reward_input)
    
    topology_state = state.get('topology_shaping', {})
    print(f"   Active attractors: {topology_state.get('active_attractors', 0)}")
    print(f"   Energy state: {state.get('energy_state', {})}")
    
    print("\n✅ Test Complete!")
    
    return {
        'simplified': simplified_results,
        'complex': complex_results,
        'speedup': speedup
    }


if __name__ == "__main__":
    results = test_gpu_acceleration()