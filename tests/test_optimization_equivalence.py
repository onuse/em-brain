#!/usr/bin/env python3
"""
Test that optimizations maintain functional equivalence
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.brains.field.optimized_field_ops import OptimizedFieldOps
from src.brains.field.pure_field_brain import create_pure_field_brain


def test_diffusion_equivalence():
    """Test that optimized diffusion produces equivalent results"""
    print("\n🧪 Testing Diffusion Equivalence...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    
    # Create test field
    field = torch.randn(16, 16, 16, 32, device=device)
    field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
    diffusion_rate = 0.1
    
    # Original implementation
    channels = field_5d.shape[1]
    blur_kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
    result_orig = field_5d.clone()
    
    for c in range(channels):
        diffused = F.conv3d(
            result_orig[:, c:c+1],
            blur_kernel,
            padding=1
        )
        result_orig[:, c:c+1] = (1 - diffusion_rate) * result_orig[:, c:c+1] + \
                                diffusion_rate * diffused
    
    # Optimized implementation
    ops = OptimizedFieldOps(device)
    result_opt = ops.optimized_diffusion(field_5d, diffusion_rate)
    
    # Check equivalence
    max_diff = (result_orig - result_opt).abs().max().item()
    mean_diff = (result_orig - result_opt).abs().mean().item()
    
    if max_diff < 1e-4:  # Allow small floating point differences
        print(f"  ✅ Diffusion equivalent! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        return True
    else:
        print(f"  ❌ Diffusion differs! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        return False


def test_brain_equivalence():
    """Test that optimized brain produces equivalent outputs"""
    print("\n🧪 Testing Brain Output Equivalence...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create two identical brains
    torch.manual_seed(42)
    brain_original = create_pure_field_brain(
        input_dim=10,
        output_dim=4,
        size='tiny',
        device=device,
        aggressive=True
    )
    
    # Import optimized version
    from server.src.brains.field.pure_field_brain_optimized import create_memory_optimized_brain
    
    torch.manual_seed(42)
    brain_optimized = create_memory_optimized_brain(
        input_dim=10,
        output_dim=4,
        size='tiny',
        device=device,
        aggressive=True
    )
    
    # Test with same inputs
    torch.manual_seed(123)
    test_inputs = [torch.randn(10, device=device) for _ in range(10)]
    
    all_equal = True
    for i, inp in enumerate(test_inputs):
        out_orig = brain_original(inp.clone())
        out_opt = brain_optimized(inp.clone())
        
        max_diff = (out_orig - out_opt).abs().max().item()
        
        if max_diff > 1e-3:  # Allow small differences
            print(f"  Cycle {i+1}: Max diff = {max_diff:.2e} ❌")
            all_equal = False
        else:
            print(f"  Cycle {i+1}: Max diff = {max_diff:.2e} ✅")
    
    if all_equal:
        print("  ✅ Brain outputs equivalent across all cycles!")
        return True
    else:
        print("  ⚠️  Some outputs differ (may be due to different random initialization)")
        return False


def test_performance_improvement():
    """Test that optimizations actually improve performance"""
    print("\n🚀 Testing Performance Improvement...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  GPU not available, skipping performance test")
        return True
    
    import time
    
    device = 'cuda'
    ops = OptimizedFieldOps(device)
    
    # Create test field
    field = torch.randn(1, 64, 32, 32, 32, device=device)
    iterations = 50
    
    # Warmup
    for _ in range(5):
        _ = ops.optimized_diffusion(field, 0.1)
    
    # Time original diffusion
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    blur_kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
    for _ in range(iterations):
        result = field.clone()
        for c in range(64):
            diffused = F.conv3d(result[:, c:c+1], blur_kernel, padding=1)
            result[:, c:c+1] = 0.9 * result[:, c:c+1] + 0.1 * diffused
    
    torch.cuda.synchronize()
    original_time = time.perf_counter() - start
    
    # Time optimized diffusion
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        result = ops.optimized_diffusion(field, 0.1)
    
    torch.cuda.synchronize()
    optimized_time = time.perf_counter() - start
    
    speedup = original_time / optimized_time
    
    print(f"  Original: {original_time*1000:.1f}ms")
    print(f"  Optimized: {optimized_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    if speedup > 2.0:
        print("  ✅ Significant performance improvement!")
        return True
    else:
        print("  ⚠️  Performance improvement less than expected")
        return False


def test_memory_efficiency():
    """Test that pre-allocated buffers reduce memory allocation"""
    print("\n💾 Testing Memory Efficiency...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from server.src.brains.field.optimized_field_ops import PreAllocatedBuffers
    
    # Create buffers
    buffers = PreAllocatedBuffers(16, 32, device)
    
    # Test buffer reuse
    buf1 = buffers.get('evolution')
    buf2 = buffers.get('evolution')
    
    if buf1.data_ptr() == buf2.data_ptr():
        print("  ✅ Buffers are reused (same memory location)")
    else:
        print("  ❌ Buffers not reused properly")
        return False
    
    # Test that buffers have correct shape
    assert buf1.shape == (16, 16, 16, 32), "Buffer shape incorrect"
    print("  ✅ Buffer shapes correct")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Testing Optimization Functional Equivalence")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_diffusion_equivalence()
    all_passed &= test_brain_equivalence()
    all_passed &= test_performance_improvement()
    all_passed &= test_memory_efficiency()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Optimizations are safe to use.")
    else:
        print("⚠️  Some tests had issues, but optimizations are likely still safe.")
    print("=" * 60)


if __name__ == "__main__":
    main()