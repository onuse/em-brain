#!/usr/bin/env python3
"""
Test the GPU performance fix for 96³×192 brain
Should run in <200ms instead of 9+ seconds
"""

import torch
import time
import sys
import os

# Add server to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

from brains.field.truly_minimal_brain import TrulyMinimalBrain
from brains.field.gpu_fixed_brain import GPUFixedBrain

def benchmark_brain(brain_class, name, spatial_size=96, channels=192, iterations=10):
    """Benchmark a brain implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Size: {spatial_size}³×{channels} = {spatial_size**3 * channels:,} parameters")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create brain
    brain = brain_class(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=spatial_size,
        channels=channels,
        device=device,
        quiet_mode=True
    )
    
    sensory_input = [0.5] * 16
    
    # Warmup
    print("Warming up...")
    for i in range(3):
        brain.process(sensory_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        motors, telemetry = brain.process(sensory_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Theoretical Hz: {1000/avg_time:.1f}")
    
    # Cleanup
    del brain
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_time

def main():
    print("\n" + "="*60)
    print("GPU PERFORMANCE FIX TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test different sizes
    test_configs = [
        (64, 128, "Small (33M params)"),
        (96, 192, "Large (170M params)"),
    ]
    
    for spatial_size, channels, desc in test_configs:
        print(f"\n\n{'#'*60}")
        print(f"TESTING: {desc}")
        print(f"{'#'*60}")
        
        # Test original (with timeout protection)
        print("\n1. Original Implementation")
        print("-"*40)
        try:
            # Set a timeout using threading
            import threading
            result = [None]
            
            def run_test():
                result[0] = benchmark_brain(TrulyMinimalBrain, "Original Brain", 
                                           spatial_size, channels, iterations=3)
            
            thread = threading.Thread(target=run_test)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if thread.is_alive():
                print("TIMEOUT: Original implementation took too long (>30s)")
                original_time = None
            else:
                original_time = result[0]
        except Exception as e:
            print(f"ERROR: {e}")
            original_time = None
        
        # Test optimized
        print("\n2. GPU-Optimized Implementation")
        print("-"*40)
        try:
            optimized_time = benchmark_brain(GPUFixedBrain, "GPU-Optimized Brain", 
                                            spatial_size, channels, iterations=10)
        except Exception as e:
            print(f"ERROR: {e}")
            optimized_time = None
        
        # Compare
        if original_time and optimized_time:
            speedup = original_time / optimized_time
            print(f"\n{'='*60}")
            print(f"SPEEDUP: {speedup:.1f}x faster!")
            print(f"Original: {original_time:.2f} ms")
            print(f"Optimized: {optimized_time:.2f} ms")
            print(f"{'='*60}")

if __name__ == "__main__":
    main()