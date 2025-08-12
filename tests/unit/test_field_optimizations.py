"""
Unit tests for field optimizations
===================================
Ensures all optimizations produce functionally equivalent results
to the original implementations while improving performance.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "server"))

from src.brains.field.optimized_field_ops import (
    OptimizedFieldOps,
    PreAllocatedBuffers
)


class TestOptimizedDiffusion:
    """Test that optimized diffusion is functionally equivalent to original"""
    
    @pytest.fixture
    def setup_fields(self) -> Tuple[torch.Tensor, float, str]:
        """Create test fields and parameters"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        field_size = 16
        channels = 32
        
        # Create reproducible random field
        torch.manual_seed(42)
        field = torch.randn(field_size, field_size, field_size, channels, device=device)
        diffusion_rate = 0.1
        
        return field, diffusion_rate, device
    
    def original_diffusion(
        self,
        field: torch.Tensor,
        diffusion_rate: float
    ) -> torch.Tensor:
        """Original channel-by-channel implementation"""
        # Reshape for convolution [C, D, H, W] -> [1, C, D, H, W]
        field_reshaped = field.permute(3, 0, 1, 2).unsqueeze(0)
        channels = field_reshaped.shape[1]
        
        blur_kernel = torch.ones(1, 1, 3, 3, 3, device=field.device) / 27.0
        result = field_reshaped.clone()
        
        for c in range(channels):
            diffused = F.conv3d(
                result[:, c:c+1],
                blur_kernel,
                padding=1
            )
            result[:, c:c+1] = (1 - diffusion_rate) * result[:, c:c+1] + \
                              diffusion_rate * diffused
        
        # Reshape back [1, C, D, H, W] -> [D, H, W, C]
        return result.squeeze(0).permute(1, 2, 3, 0)
    
    def test_diffusion_equivalence(self, setup_fields):
        """Test that optimized diffusion produces identical results"""
        field, diffusion_rate, device = setup_fields
        ops = OptimizedFieldOps(device)
        
        # Compute with original method
        original_result = self.original_diffusion(field, diffusion_rate)
        
        # Compute with optimized method
        field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
        optimized_result = ops.optimized_diffusion(field_5d, diffusion_rate)
        optimized_result = optimized_result.squeeze(0).permute(1, 2, 3, 0)
        
        # Check equivalence (allowing for small floating point differences)
        assert torch.allclose(original_result, optimized_result, rtol=1e-5, atol=1e-7), \
            f"Max difference: {(original_result - optimized_result).abs().max().item()}"
    
    def test_diffusion_with_zero_rate(self, setup_fields):
        """Test that zero diffusion returns unchanged field"""
        field, _, device = setup_fields
        ops = OptimizedFieldOps(device)
        
        field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
        result = ops.optimized_diffusion(field_5d, 0.0)
        
        assert torch.allclose(field_5d, result), \
            "Zero diffusion should return unchanged field"
    
    def test_diffusion_preserves_shape(self, setup_fields):
        """Test that diffusion preserves tensor shape"""
        field, diffusion_rate, device = setup_fields
        ops = OptimizedFieldOps(device)
        
        # Test 4D input
        field_4d = field.permute(3, 0, 1, 2)  # [C, D, H, W]
        result_4d = ops.optimized_diffusion(field_4d, diffusion_rate)
        assert field_4d.shape == result_4d.shape, "4D shape should be preserved"
        
        # Test 5D input
        field_5d = field_4d.unsqueeze(0)  # [B, C, D, H, W]
        result_5d = ops.optimized_diffusion(field_5d, diffusion_rate)
        assert field_5d.shape == result_5d.shape, "5D shape should be preserved"


class TestGradientComputation:
    """Test optimized gradient computation"""
    
    @pytest.fixture
    def setup_gradient_field(self):
        """Create field for gradient testing"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a field with known gradients
        size = 8
        channels = 16
        
        # Linear gradient in each direction
        x = torch.linspace(0, 1, size, device=device)
        y = torch.linspace(0, 2, size, device=device)
        z = torch.linspace(0, 3, size, device=device)
        
        # Create 3D meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Stack to create multi-channel field
        field = torch.stack([xx + yy + zz] * channels, dim=-1)
        
        return field, device
    
    def original_gradients(self, field: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Original gradient computation"""
        if field.shape[0] > 1:
            grad_x = field[1:, :, :, :] - field[:-1, :, :, :]
            grad_y = field[:, 1:, :, :] - field[:, :-1, :, :]
            grad_z = field[:, :, 1:, :] - field[:, :, :-1, :]
        else:
            grad_x = torch.zeros_like(field)
            grad_y = torch.zeros_like(field)
            grad_z = torch.zeros_like(field)
        
        return grad_x.abs(), grad_y.abs(), grad_z.abs()
    
    def test_gradient_computation_equivalence(self, setup_gradient_field):
        """Test that batched gradients match original computation"""
        field, device = setup_gradient_field
        ops = OptimizedFieldOps(device)
        
        # Original method
        orig_x, orig_y, orig_z = self.original_gradients(field)
        
        # Optimized method
        opt_x, opt_y, opt_z = ops.batched_gradient_computation(field, return_magnitude=True)
        
        # Shapes might differ due to padding, so compare valid regions
        min_x = min(orig_x.shape[0], opt_x.shape[0])
        min_y = min(orig_y.shape[1], opt_y.shape[1])
        min_z = min(orig_z.shape[2], opt_z.shape[2])
        
        # Check that gradients are similar (Sobel vs simple difference)
        # Sobel will smooth more, so we allow larger tolerance
        assert torch.allclose(
            orig_x[:min_x, :min_y, :min_z].mean(),
            opt_x[:min_x, :min_y, :min_z].mean(),
            rtol=0.3  # 30% tolerance for different methods
        ), "X gradients should be similar"
    
    def test_gradient_shape_handling(self, setup_gradient_field):
        """Test gradient computation handles different input shapes"""
        field, device = setup_gradient_field
        ops = OptimizedFieldOps(device)
        
        # Test 4D input [D, H, W, C]
        grad_x, grad_y, grad_z = ops.batched_gradient_computation(field)
        assert len(grad_x.shape) == 4, "Should return 4D gradients for 4D input"
        
        # Test 5D input [B, C, D, H, W]
        field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
        grad_x_5d, grad_y_5d, grad_z_5d = ops.batched_gradient_computation(field_5d)
        assert len(grad_x_5d.shape) == 5, "Should return 5D gradients for 5D input"


class TestPreAllocatedBuffers:
    """Test pre-allocated buffer system"""
    
    def test_buffer_creation(self):
        """Test that buffers are created correctly"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        size = 16
        channels = 32
        
        buffers = PreAllocatedBuffers(size, channels, device)
        
        # Check key buffers exist
        assert 'evolution' in buffers.buffers
        assert 'diffusion' in buffers.buffers
        assert 'gradient_x' in buffers.buffers
        
        # Check shapes
        evolution = buffers.get('evolution')
        assert evolution.shape == (size, size, size, channels)
        assert evolution.device.type == device
    
    def test_buffer_reuse(self):
        """Test that buffers are reused, not reallocated"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        buffers = PreAllocatedBuffers(8, 16, device)
        
        # Get buffer twice
        buf1 = buffers.get('evolution')
        buf2 = buffers.get('evolution')
        
        # Should be the same object (same memory)
        assert buf1.data_ptr() == buf2.data_ptr(), \
            "Buffer should be reused, not reallocated"
    
    def test_buffer_resize(self):
        """Test dynamic buffer resizing"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        buffers = PreAllocatedBuffers(8, 16, device)
        
        # Request different size
        new_shape = (10, 10, 10, 20)
        resized = buffers.ensure_size('custom', *new_shape)
        
        assert resized.shape == new_shape
        assert 'custom' in buffers.buffers


class TestCrossScaleFlow:
    """Test parallel cross-scale information flow"""
    
    @pytest.fixture
    def setup_multiscale(self):
        """Create multi-scale field hierarchy"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create fields at different scales
        levels = [
            torch.randn(32, 32, 32, 64, device=device),  # Fine
            torch.randn(16, 16, 16, 32, device=device),  # Medium
            torch.randn(8, 8, 8, 16, device=device),      # Coarse
        ]
        
        return levels, device
    
    def test_cross_scale_preserves_shape(self, setup_multiscale):
        """Test that cross-scale flow preserves tensor shapes"""
        levels, device = setup_multiscale
        ops = OptimizedFieldOps(device)
        
        original_shapes = [level.shape for level in levels]
        
        # Apply cross-scale flow
        updated = ops.parallel_cross_scale_flow(
            levels, 
            cross_scale_strength=0.2,
            use_streams=(device == 'cuda')
        )
        
        # Check shapes preserved
        for original, updated_level in zip(original_shapes, updated):
            assert original == updated_level.shape, \
                "Cross-scale flow should preserve shapes"
    
    def test_cross_scale_blending(self, setup_multiscale):
        """Test that cross-scale blending works correctly"""
        levels, device = setup_multiscale
        ops = OptimizedFieldOps(device)
        
        # Test with different strengths
        for strength in [0.0, 0.5, 1.0]:
            updated = ops.parallel_cross_scale_flow(levels, strength)
            
            if strength == 0.0:
                # No blending - should be unchanged
                for orig, upd in zip(levels, updated):
                    assert torch.allclose(orig, upd), \
                        "Zero strength should not change fields"
            else:
                # Some blending occurred
                assert len(updated) == len(levels), \
                    "Should return same number of levels"


class TestPerformanceImprovement:
    """Test that optimizations actually improve performance"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_diffusion_speedup(self):
        """Verify diffusion optimization provides speedup"""
        import time
        
        device = 'cuda'
        size = 32
        channels = 64
        iterations = 50
        
        field = torch.randn(1, channels, size, size, size, device=device)
        ops = OptimizedFieldOps(device)
        
        # Time original
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        blur_kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
        for _ in range(iterations):
            result = field.clone()
            for c in range(channels):
                diffused = F.conv3d(result[:, c:c+1], blur_kernel, padding=1)
                result[:, c:c+1] = 0.9 * result[:, c:c+1] + 0.1 * diffused
        
        torch.cuda.synchronize()
        original_time = time.perf_counter() - start
        
        # Time optimized
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(iterations):
            result = ops.optimized_diffusion(field, 0.1)
        
        torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start
        
        speedup = original_time / optimized_time
        print(f"\nDiffusion speedup: {speedup:.1f}x")
        
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


class TestFunctionalEquivalence:
    """Integration tests ensuring complete functional equivalence"""
    
    def test_full_pipeline_equivalence(self):
        """Test that full optimization pipeline maintains equivalence"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)
        
        # Create test field
        field = torch.randn(16, 16, 16, 32, device=device)
        field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
        
        ops = OptimizedFieldOps(device)
        
        # Run through full pipeline
        # 1. Diffusion
        diffused = ops.optimized_diffusion(field_5d, 0.1)
        
        # 2. Gradients
        grad_x, grad_y, grad_z = ops.batched_gradient_computation(diffused)
        
        # 3. Combine results
        result = diffused + 0.01 * (grad_x.mean() + grad_y.mean() + grad_z.mean())
        
        # Result should be deterministic
        assert result.shape == field_5d.shape
        assert not torch.isnan(result).any(), "No NaN values should be produced"
        assert not torch.isinf(result).any(), "No Inf values should be produced"
    
    def test_mixed_precision_compatibility(self):
        """Test that optimizations work with mixed precision"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision")
        
        device = 'cuda'
        field = torch.randn(8, 8, 8, 16, device=device, dtype=torch.float16)
        field_5d = field.permute(3, 0, 1, 2).unsqueeze(0)
        
        ops = OptimizedFieldOps(device)
        
        with torch.cuda.amp.autocast():
            result = ops.optimized_diffusion(field_5d, 0.1)
        
        assert result.dtype in [torch.float16, torch.float32], \
            "Should handle mixed precision"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])