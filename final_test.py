#!/usr/bin/env python3
"""Final comprehensive performance test."""

import torch
import time
import cProfile
import pstats
import io
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

def main():
    print("=" * 70)
    print("FINAL PERFORMANCE TEST - 96³×192 BRAIN")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("WARNING: Running on CPU")
    
    # Create brain
    print("\nInitializing brain...")
    brain = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=96,
        channels=192,
        device=device,
        quiet_mode=True
    )
    
    total_params = 96 * 96 * 96 * 192
    print(f"Total parameters: {total_params:,}")
    print(f"Memory: ~{total_params * 4 / (1024**3):.2f} GB")
    
    sensors = [0.5] * 12
    
    # Warmup - important for CUDA kernel compilation
    print("\n" + "-" * 50)
    print("WARMUP (5 cycles)")
    print("-" * 50)
    
    for i in range(5):
        start = time.perf_counter()
        motors, telemetry = brain.process(sensors)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Warmup {i+1}: {elapsed:7.1f}ms")
    
    # Performance test
    print("\n" + "-" * 50)
    print("PERFORMANCE TEST (20 cycles)")
    print("-" * 50)
    
    times = []
    for i in range(20):
        start = time.perf_counter()
        motors, telemetry = brain.process(sensors)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        if i < 10 or i % 5 == 0:  # Print first 10 and every 5th after
            print(f"  Cycle {i+1:3}: {elapsed:7.1f}ms")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Exclude first run (often slower due to lazy init)
    avg_after_first = sum(times[1:]) / len(times[1:]) if len(times) > 1 else avg_time
    
    print("\n" + "-" * 50)
    print("STATISTICS")
    print("-" * 50)
    print(f"  Average (all):        {avg_time:7.1f}ms")
    print(f"  Average (excl first): {avg_after_first:7.1f}ms")
    print(f"  Minimum:              {min_time:7.1f}ms")
    print(f"  Maximum:              {max_time:7.1f}ms")
    
    # Performance verdict
    print("\n" + "=" * 70)
    print("PERFORMANCE VERDICT")
    print("=" * 70)
    
    target = 100  # Target: <100ms on RTX 3070
    
    if avg_after_first < target:
        print(f"✅ SUCCESS! Brain is running at {avg_after_first:.1f}ms")
        print(f"   This is {target/avg_after_first:.1f}x the target speed!")
        print(f"   Target: <{target}ms on RTX 3070 ✓")
    elif avg_after_first < target * 2:
        print(f"⚠️  ACCEPTABLE: Brain is running at {avg_after_first:.1f}ms")
        print(f"   This is {avg_after_first/target:.1f}x slower than target")
        print(f"   Target: <{target}ms on RTX 3070")
    else:
        print(f"❌ TOO SLOW: Brain is running at {avg_after_first:.1f}ms")
        print(f"   This is {avg_after_first/target:.1f}x slower than target")
        print(f"   Target: <{target}ms on RTX 3070")
    
    # Show what improved
    print("\n" + "-" * 50)
    print("OPTIMIZATIONS APPLIED")
    print("-" * 50)
    print("✅ Eliminated Python min/max in hot paths")
    print("✅ Replaced scalar operations with tensor ops")
    print("✅ Minimized CPU-GPU transfers (.item() calls)")
    print("✅ Used ultra-fast motor extraction for large fields")
    print("✅ Pre-allocated work tensors")
    print("✅ Batched GPU operations")
    
    return avg_after_first

if __name__ == "__main__":
    try:
        avg_time = main()
        print(f"\nFinal result: {avg_time:.1f}ms average processing time")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()