#!/usr/bin/env python3
"""
Minimal test to verify GPU optimization
"""

import torch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

def test_vectorized_injection():
    """Test the vectorized sensory injection optimization."""
    print("\nTesting Vectorized Sensory Injection")
    print("-"*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test sizes
    for spatial_size, channels in [(64, 128), (96, 192)]:
        print(f"\nSize: {spatial_size}³×{channels}")
        
        # Create field
        field = torch.randn(spatial_size, spatial_size, spatial_size, channels, device=device) * 0.01
        sensors = torch.randn(16, device=device)
        sensor_spots = torch.randint(0, spatial_size, (16, 3), device=device)
        
        # Method 1: Loop (original - BAD)
        field_loop = field.clone()
        start = time.perf_counter()
        for i in range(16):
            x, y, z = sensor_spots[i]
            field_loop[x, y, z, i % 8] += sensors[i] * 0.3
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loop_time = (time.perf_counter() - start) * 1000
        print(f"  Loop method: {loop_time:.2f} ms")
        
        # Method 2: Vectorized (optimized - GOOD)
        field_vec = field.clone()
        start = time.perf_counter()
        
        # Prepare indices
        sensor_x = sensor_spots[:, 0]
        sensor_y = sensor_spots[:, 1]
        sensor_z = sensor_spots[:, 2]
        sensor_c = torch.arange(16, device=device) % 8
        
        # Convert to flat indices
        flat_idx = (
            sensor_x * (spatial_size * spatial_size * channels) +
            sensor_y * (spatial_size * channels) +
            sensor_z * channels +
            sensor_c
        )
        
        # Vectorized injection
        field_flat = field_vec.view(-1)
        injection_values = sensors * 0.3
        field_flat.scatter_add_(0, flat_idx, injection_values)
        field_vec = field_flat.view(spatial_size, spatial_size, spatial_size, channels)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vec_time = (time.perf_counter() - start) * 1000
        print(f"  Vectorized method: {vec_time:.2f} ms")
        print(f"  Speedup: {loop_time/vec_time:.1f}x")

def test_full_brain():
    """Test the full optimized brain."""
    print("\n\nTesting Full Optimized Brain")
    print("-"*40)
    
    from brains.field.gpu_fixed_brain import GPUFixedBrain
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for spatial_size, channels in [(64, 128), (96, 192)]:
        print(f"\nSize: {spatial_size}³×{channels}")
        
        brain = GPUFixedBrain(
            sensory_dim=16,
            motor_dim=5,
            spatial_size=spatial_size,
            channels=channels,
            device=device,
            quiet_mode=True
        )
        
        sensory_input = [0.5] * 16
        
        # Warmup
        for _ in range(3):
            brain.process(sensory_input)
        
        # Time
        times = []
        for i in range(5):
            start = time.perf_counter()
            brain.process(sensory_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"  Average cycle time: {avg_time:.2f} ms")
        print(f"  Theoretical Hz: {1000/avg_time:.1f}")

if __name__ == "__main__":
    print("="*60)
    print("GPU OPTIMIZATION TEST")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    test_vectorized_injection()
    test_full_brain()