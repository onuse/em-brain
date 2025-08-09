#!/usr/bin/env python3
"""
Quick Real-Time Performance Test for PureFieldBrain
Focus on hardware_constrained config (6³×64) for robot control
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS


def test_realtime_performance():
    """Test if PureFieldBrain can achieve 30+ Hz for robot control"""
    
    print("=" * 60)
    print("Real-Time Performance Test for Robot Control")
    print("Target: 30+ Hz (< 33.33ms per cycle)")
    print("=" * 60)
    
    # Use hardware_constrained config optimized for real-time
    config = SCALE_CONFIGS['hardware_constrained']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration: hardware_constrained")
    print(f"Field size: {config.levels[0][0]}³ × {config.levels[0][1]} channels")
    print(f"Total parameters: {config.total_params:,}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Create brain
    brain = PureFieldBrain(
        input_dim=10,  # Typical sensor count
        output_dim=4,   # Motor outputs (forward, turn, etc.)
        scale_config=config,
        device=device,
        aggressive=True
    )
    
    # Warmup
    print("\nWarming up...")
    for _ in range(20):
        _ = brain(torch.randn(10, device=device))
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Test different batch sizes
    test_cycles = [100, 500, 1000]
    
    for num_cycles in test_cycles:
        print(f"\nTesting {num_cycles} cycles:")
        
        # Prepare inputs
        inputs = [torch.randn(10, device=device) for _ in range(num_cycles)]
        
        # Time the cycles
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for i in range(num_cycles):
            _ = brain(inputs[i])
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.perf_counter() - start_time
        
        # Calculate metrics
        ms_per_cycle = (elapsed_time / num_cycles) * 1000
        achieved_hz = 1000 / ms_per_cycle
        meets_requirement = ms_per_cycle < 33.33
        
        print(f"  Time per cycle: {ms_per_cycle:.2f}ms")
        print(f"  Achieved rate: {achieved_hz:.1f} Hz")
        print(f"  Meets 30Hz requirement: {'✅ YES' if meets_requirement else '❌ NO'}")
        
        if not meets_requirement:
            deficit = ms_per_cycle - 33.33
            print(f"  Deficit: {deficit:.1f}ms (need {deficit/ms_per_cycle*100:.1f}% improvement)")
    
    # Test with learning enabled
    print("\n" + "-" * 60)
    print("Testing with learning (prediction error updates):")
    
    learning_cycles = 100
    times_with_learning = []
    
    for i in range(learning_cycles):
        sensory = torch.randn(10, device=device)
        
        start = time.perf_counter()
        motor = brain(sensory)
        
        # Simulate prediction error learning every 10 cycles
        if i % 10 == 0:
            brain.learn_from_prediction_error(
                actual=sensory,
                predicted=torch.randn(10, device=device)
            )
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000
        times_with_learning.append(elapsed)
    
    mean_time = np.mean(times_with_learning)
    std_time = np.std(times_with_learning)
    p95_time = np.percentile(times_with_learning, 95)
    
    print(f"  Mean: {mean_time:.2f}ms ({1000/mean_time:.1f} Hz)")
    print(f"  Std dev: {std_time:.2f}ms")
    print(f"  95th percentile: {p95_time:.2f}ms ({1000/p95_time:.1f} Hz)")
    print(f"  Meets 30Hz at P95: {'✅ YES' if p95_time < 33.33 else '❌ NO'}")
    
    # Memory usage
    if device == 'cuda':
        print("\n" + "-" * 60)
        print("GPU Memory Usage:")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved: {reserved:.1f} MB")
    
    # Test CPU-GPU transfer overhead
    if device == 'cuda':
        print("\n" + "-" * 60)
        print("CPU-GPU Transfer Overhead:")
        
        # Test input transfer (CPU -> GPU)
        cpu_input = torch.randn(10)
        transfer_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            gpu_input = cpu_input.to(device)
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
        
        print(f"  Input transfer: {np.mean(transfer_times):.3f}ms")
        
        # Test output transfer (GPU -> CPU)
        gpu_output = torch.randn(4, device=device)
        transfer_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            cpu_output = gpu_output.cpu()
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
        
        print(f"  Output transfer: {np.mean(transfer_times):.3f}ms")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    if ms_per_cycle < 33.33:
        print(f"\n✅ SUCCESS: hardware_constrained config achieves {achieved_hz:.1f} Hz")
        print(f"   Safe for real-time robot control at 30+ Hz")
        margin = 33.33 - ms_per_cycle
        print(f"   Safety margin: {margin:.1f}ms ({margin/33.33*100:.1f}%)")
    else:
        print(f"\n⚠️ WARNING: Only achieving {achieved_hz:.1f} Hz")
        print(f"   May not be safe for real-time robot control")
        print(f"\n   Optimization suggestions:")
        print(f"   1. Ensure CUDA is available and being used")
        print(f"   2. Reduce field size further if needed")
        print(f"   3. Use torch.jit.script for optimization")
        print(f"   4. Consider mixed precision (fp16)")
        print(f"   5. Profile with torch.profiler to find bottlenecks")
    
    # Additional insights
    print(f"\n📊 Key Insights:")
    print(f"   - Field size: {config.levels[0][0]}³ = {config.levels[0][0]**3} spatial points")
    print(f"   - Channels: {config.levels[0][1]}")
    print(f"   - Total field elements: {config.levels[0][0]**3 * config.levels[0][1]:,}")
    print(f"   - Operations per cycle: ~{config.total_params * 2:,} (rough estimate)")
    
    if device == 'cuda':
        print(f"   - GPU provides {1000/ms_per_cycle:.1f}x realtime performance")
    else:
        print(f"   - CPU-only mode - consider GPU for better performance")
    
    return ms_per_cycle < 33.33, achieved_hz


if __name__ == "__main__":
    success, hz = test_realtime_performance()
    
    # Return exit code
    import sys
    sys.exit(0 if success else 1)