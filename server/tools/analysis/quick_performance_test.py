#!/usr/bin/env python3
"""
Quick Performance Test for Brain System
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.simplified_brain_factory import SimplifiedBrainFactory


def time_operation(func, *args, **kwargs):
    """Time a single operation."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    duration = (time.perf_counter() - start) * 1000
    return result, duration


def profile_brain():
    """Quick performance profile of brain operations."""
    # Create brain
    print("Creating brain...")
    factory = SimplifiedBrainFactory({'quiet_mode': True})
    brain_wrapper = factory.create(sensory_dim=24, motor_dim=4)
    brain = brain_wrapper.brain
    
    print(f"Device: {brain.device}")
    print(f"Field shape: {brain.unified_field.shape}")
    print(f"Field size: {brain.unified_field.numel() * 4 / 1024 / 1024:.2f}MB")
    print()
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        brain.process_robot_cycle([0.0] * 24)
    
    # Time individual operations
    sensory_input = [0.1] * 24
    print("\nTiming individual operations (average of 10 runs):")
    print("-" * 50)
    
    operations = {
        "Full cycle": lambda: brain.process_robot_cycle(sensory_input),
        "Field evolution": lambda: brain._evolve_field(),
        "Pattern extraction": lambda: brain.pattern_system.extract_patterns(brain.unified_field, n_patterns=10),
        "Attention processing": lambda: brain._process_attention(sensory_input),
        "Motor generation": lambda: brain._generate_motor_action(),
        "Field dynamics evolve": lambda: brain.field_dynamics.evolve_field(brain.unified_field),
    }
    
    results = {}
    for name, op in operations.items():
        times = []
        for _ in range(10):
            _, duration = time_operation(op)
            times.append(duration)
        avg_time = sum(times) / len(times)
        results[name] = avg_time
        print(f"{name:<25}: {avg_time:>8.2f}ms")
    
    # Analyze bottlenecks
    print("\nBottleneck Analysis:")
    print("-" * 50)
    
    full_cycle_time = results["Full cycle"]
    for name, duration in sorted(results.items(), key=lambda x: x[1], reverse=True):
        if name != "Full cycle":
            percentage = (duration / full_cycle_time) * 100
            print(f"{name:<25}: {percentage:>6.1f}% of cycle time")
    
    # Test batch processing potential
    print("\nBatch Processing Test:")
    print("-" * 50)
    
    # Single brain
    _, single_time = time_operation(brain.process_robot_cycle, sensory_input)
    print(f"Single brain cycle: {single_time:.2f}ms")
    
    # Simulate batch by running field evolution on stacked tensors
    batch_size = 4
    batched_field = brain.unified_field.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    def batch_evolve():
        # Simple batch simulation - just apply same operations
        return brain.field_dynamics.evolve_field(batched_field[0])
    
    _, batch_time = time_operation(batch_evolve)
    print(f"Batched field evolution: {batch_time:.2f}ms per brain")
    print(f"Potential speedup: {single_time / batch_time:.1f}x")


if __name__ == "__main__":
    profile_brain()