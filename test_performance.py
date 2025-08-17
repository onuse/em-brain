#!/usr/bin/env python3
"""
Performance test for optimized brain.
Tests the optimization fixes for the 96³×192 tensor field brain.
"""

import torch
import time
import cProfile
import pstats
import io
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

def test_brain_performance():
    """Test brain performance with the large configuration."""
    
    print("=" * 70)
    print("BRAIN PERFORMANCE TEST")
    print("=" * 70)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Create brain with original large size
    print("\nInitializing brain...")
    brain = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=96,  # 96³ spatial grid
        channels=192,     # 192 channels
        device=device,
        quiet_mode=False
    )
    
    # Calculate total parameters
    total_params = 96 * 96 * 96 * 192
    print(f"Total field parameters: {total_params:,}")
    print(f"Memory footprint: ~{total_params * 4 / (1024**3):.2f} GB")
    
    # Prepare test input
    sensors = [0.5] * 12
    
    # Warmup runs
    print("\n" + "=" * 50)
    print("WARMUP (3 cycles)")
    print("=" * 50)
    
    for i in range(3):
        start = time.perf_counter()
        motors, telemetry = brain.process(sensors)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Warmup {i+1}: {elapsed:.1f}ms")
    
    # Performance test
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST (10 cycles)")
    print("=" * 50)
    
    times = []
    for i in range(10):
        start = time.perf_counter()
        motors, telemetry = brain.process(sensors)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Cycle {i+1:2d}: {elapsed:6.1f}ms")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min:     {min_time:.1f}ms")
    print(f"  Max:     {max_time:.1f}ms")
    
    # Performance assessment
    print("\n" + "=" * 50)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 50)
    
    if avg_time < 100:
        print("✅ EXCELLENT: Brain running at expected speed (<100ms)")
    elif avg_time < 200:
        print("⚠️  GOOD: Brain running acceptably (100-200ms)")
    elif avg_time < 500:
        print("⚠️  SLOW: Brain running slowly (200-500ms)")
    else:
        print("❌ TOO SLOW: Brain running very slowly (>500ms)")
    
    # GPU utilization check
    if device.type == 'cuda':
        print("\n" + "=" * 50)
        print("GPU MEMORY USAGE")
        print("=" * 50)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    
    # Profile one cycle for detailed analysis
    print("\n" + "=" * 50)
    print("PROFILING ONE CYCLE")
    print("=" * 50)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.perf_counter()
    motors, telemetry = brain.process(sensors)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    profiler.disable()
    
    print(f"Profiled cycle time: {elapsed:.1f}ms")
    
    # Print top time consumers
    print("\nTop 15 time consumers:")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    profile_output = s.getvalue()
    
    # Filter and format the output
    for line in profile_output.split('\n'):
        if 'percall' in line or 'filename' in line or line.strip() and not line.startswith(' '):
            print(line)
    
    # Check for specific bottlenecks
    print("\n" + "=" * 50)
    print("BOTTLENECK ANALYSIS")
    print("=" * 50)
    
    found_issues = []
    
    # Check if 'builtins.max' or 'builtins.min' appear in profile
    if 'builtins.max' in profile_output or 'builtins.min' in profile_output:
        found_issues.append("❌ Python's builtin min/max still being used")
    else:
        print("✅ No Python min/max bottleneck detected")
    
    # Check for excessive .item() calls
    item_count = profile_output.count('.item')
    if item_count > 20:
        found_issues.append(f"⚠️  Excessive .item() calls detected ({item_count})")
    else:
        print(f"✅ Reasonable number of .item() calls ({item_count})")
    
    # Print any issues found
    if found_issues:
        print("\nIssues found:")
        for issue in found_issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major bottlenecks detected!")
    
    return avg_time


if __name__ == "__main__":
    avg_time = test_brain_performance()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if avg_time < 100:
        print(f"✅ SUCCESS: Brain optimized! Average time: {avg_time:.1f}ms")
        print("   Target: <100ms on RTX 3070 ✓")
    else:
        print(f"⚠️  Brain still needs optimization. Average time: {avg_time:.1f}ms")
        print("   Target: <100ms on RTX 3070")