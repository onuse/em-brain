#!/usr/bin/env python3
"""
Test GPU optimizations to ensure they produce identical results to original implementations.
"""

import torch
import numpy as np
import time
from server.src.brains.field.simple_field_dynamics import SimpleFieldDynamics
from server.src.brains.field.simple_motor import SimpleMotorExtraction
from server.src.brains.field.intrinsic_tensions import IntrinsicTensions


def test_diffusion_optimization():
    """Test that GPU-optimized diffusion produces same results as original."""
    print("\n=== Testing Diffusion Optimization ===")
    
    # Create test field
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test multiple field sizes
    for size in [16, 32, 96]:
        print(f"\nTesting field size: {size}³")
        
        field = torch.randn(size, size, size, 32, device=device) * 0.1
        dynamics = SimpleFieldDynamics()
        
        # Test original implementation
        field_copy1 = field.clone()
        start = time.time()
        result_original = dynamics._apply_diffusion_original(field_copy1)
        time_original = (time.time() - start) * 1000
        
        # Test GPU-optimized implementation
        field_copy2 = field.clone()
        start = time.time()
        result_optimized = dynamics._apply_diffusion_gpu_optimized(field_copy2)
        time_optimized = (time.time() - start) * 1000
        
        # Compare results
        diff = torch.abs(result_original - result_optimized).max().item()
        speedup = time_original / time_optimized if time_optimized > 0 else 0
        
        print(f"  Original time: {time_original:.2f}ms")
        print(f"  Optimized time: {time_optimized:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Max difference: {diff:.8f}")
        print(f"  Results match: {diff < 1e-5}")
        
        if diff > 1e-5:
            print(f"  WARNING: Results differ by more than tolerance!")


def test_motor_extraction_optimization():
    """Test that GPU-optimized motor extraction produces same results as original."""
    print("\n=== Testing Motor Extraction Optimization ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test multiple field sizes
    for size in [16, 32, 64]:
        print(f"\nTesting field size: {size}³")
        
        field = torch.randn(size, size, size, 32, device=device) * 0.1
        motor = SimpleMotorExtraction(motor_dim=6, device=device, field_size=size)
        
        # Test original implementation
        start = time.time()
        result_original = motor._extract_motors_original(field)
        time_original = (time.time() - start) * 1000
        
        # Test GPU-optimized implementation
        start = time.time()
        result_optimized = motor._extract_motors_gpu_optimized(field)
        time_optimized = (time.time() - start) * 1000
        
        # Compare results
        diff = max(abs(a - b) for a, b in zip(result_original, result_optimized))
        speedup = time_original / time_optimized if time_optimized > 0 else 0
        
        print(f"  Original time: {time_original:.2f}ms")
        print(f"  Optimized time: {time_optimized:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Max difference: {diff:.8f}")
        print(f"  Results match: {diff < 1e-4}")
        
        if diff > 1e-4:
            print(f"  WARNING: Results differ by more than tolerance!")
            print(f"  Original: {result_original}")
            print(f"  Optimized: {result_optimized}")


def test_metrics_batching():
    """Test that batched metrics computation produces same results."""
    print("\n=== Testing Metrics Batching ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create test field and tensions
    field_shape = (32, 32, 32, 64)
    field = torch.randn(*field_shape, device=device) * 0.1
    tensions = IntrinsicTensions(field_shape, device)
    
    # Measure performance
    start = time.time()
    for _ in range(10):
        metrics = tensions.get_comfort_metrics(field)
    time_batched = (time.time() - start) * 100  # Average per call
    
    print(f"  Batched metrics time: {time_batched:.2f}ms")
    print(f"  Metrics computed: {list(metrics.keys())}")
    print(f"  Overall comfort: {metrics['overall_comfort']:.3f}")
    
    # Verify metrics are reasonable
    assert 0 <= metrics['overall_comfort'] <= 1, "Comfort should be in [0, 1]"
    assert metrics['field_mean'] == field.mean().item(), "Field mean should match"
    print("  All metrics valid!")


def benchmark_full_system():
    """Benchmark the full system with optimizations."""
    print("\n=== Full System Benchmark ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Create components
    size = 64
    channels = 128
    field = torch.randn(size, size, size, channels, device=device) * 0.01
    
    dynamics = SimpleFieldDynamics()
    motor = SimpleMotorExtraction(motor_dim=6, device=device, field_size=size)
    tensions = IntrinsicTensions(field.shape, device)
    
    # Run one full cycle
    print(f"\nField size: {size}³×{channels} = {size**3 * channels:,} parameters")
    
    total_start = time.time()
    
    # Apply tensions
    start = time.time()
    field = tensions.apply_tensions(field, prediction_error=0.1)
    tension_time = (time.time() - start) * 1000
    
    # Evolve dynamics
    start = time.time()
    field = dynamics.evolve(field)
    dynamics_time = (time.time() - start) * 1000
    
    # Extract motors
    start = time.time()
    motors = motor.extract_motors(field)
    motor_time = (time.time() - start) * 1000
    
    # Get metrics
    start = time.time()
    metrics = tensions.get_comfort_metrics(field)
    metrics_time = (time.time() - start) * 1000
    
    total_time = (time.time() - total_start) * 1000
    
    print(f"\nTiming breakdown:")
    print(f"  Tensions: {tension_time:.2f}ms")
    print(f"  Dynamics: {dynamics_time:.2f}ms")
    print(f"  Motors: {motor_time:.2f}ms")
    print(f"  Metrics: {metrics_time:.2f}ms")
    print(f"  Total: {total_time:.2f}ms")
    print(f"  Frequency: {1000/total_time:.1f}Hz")
    
    # Check if we meet real-time requirements
    target_hz = 20
    if 1000/total_time >= target_hz:
        print(f"  ✓ Meets {target_hz}Hz real-time requirement")
    else:
        print(f"  ✗ Below {target_hz}Hz target")


if __name__ == "__main__":
    print("GPU Optimization Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_diffusion_optimization()
    test_motor_extraction_optimization()
    test_metrics_batching()
    benchmark_full_system()
    
    print("\n" + "=" * 60)
    print("All tests completed!")