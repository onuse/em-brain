#!/usr/bin/env python3
"""
Continuous Integration Tests for PureFieldBrain
================================================
Lightweight tests that run on every commit to catch regressions early.
Designed to run in <1 minute on CI servers.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from src.brains.field.pure_field_brain import create_pure_field_brain, SCALE_CONFIGS
from src.brains.field.optimized_field_ops import OptimizedFieldOps, PreAllocatedBuffers


class TestBrainSafety(unittest.TestCase):
    """Critical safety tests - MUST pass"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = create_pure_field_brain(size='tiny', device=self.device)
    
    def test_motor_bounds(self):
        """Motor outputs must be in [-1, 1]"""
        # Test with extreme inputs
        test_inputs = [
            torch.zeros(10, device=self.device),
            torch.ones(10, device=self.device) * 1000,
            torch.randn(10, device=self.device) * 100,
        ]
        
        for inp in test_inputs:
            output = self.brain(inp)
            self.assertTrue(
                output.abs().max().item() <= 1.0,
                f"Motor output exceeded bounds: {output.abs().max().item()}"
            )
    
    def test_nan_inf_handling(self):
        """NaN/Inf inputs must not propagate"""
        bad_input = torch.randn(10, device=self.device)
        bad_input[2] = float('nan')
        bad_input[5] = float('inf')
        
        output = self.brain(bad_input)
        
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_dimension_mismatch(self):
        """Brain should handle variable input dimensions"""
        # Test with different input sizes
        for input_dim in [5, 10, 16, 20]:
            input_data = torch.randn(input_dim, device=self.device)
            output = self.brain(input_data)
            self.assertEqual(output.shape[0], self.brain.output_dim)


class TestFieldDynamics(unittest.TestCase):
    """Test field evolution and dynamics"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_field_evolution(self):
        """Field should evolve over time"""
        brain = create_pure_field_brain(size='tiny', device=self.device)
        
        initial_field = brain.field.clone()
        
        # Run several cycles
        for _ in range(10):
            brain(torch.randn(10, device=self.device))
        
        field_change = (brain.field - initial_field).abs().mean().item()
        
        # Field should change
        self.assertGreater(field_change, 0.001, "Field didn't evolve")
        # But not explode
        self.assertLess(field_change, 100.0, "Field changed too much")
    
    def test_deterministic_behavior(self):
        """Same input + same seed = same output"""
        torch.manual_seed(42)
        brain1 = create_pure_field_brain(size='tiny', device=self.device)
        
        torch.manual_seed(42)
        brain2 = create_pure_field_brain(size='tiny', device=self.device)
        
        test_input = torch.randn(10, device=self.device)
        
        out1 = brain1(test_input.clone())
        out2 = brain2(test_input.clone())
        
        self.assertTrue(torch.allclose(out1, out2, rtol=1e-5))
    
    def test_scale_configs(self):
        """All scale configurations should work"""
        for scale_name in ['hardware_constrained', 'tiny', 'small']:
            with self.subTest(scale=scale_name):
                brain = create_pure_field_brain(
                    size=scale_name,
                    device=self.device
                )
                
                # Should process input without error
                output = brain(torch.randn(10, device=self.device))
                self.assertEqual(output.shape[0], 4)


class TestOptimizations(unittest.TestCase):
    """Test optimization correctness"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ops = OptimizedFieldOps(self.device)
    
    def test_diffusion_equivalence(self):
        """Optimized diffusion should match original"""
        torch.manual_seed(42)
        field = torch.randn(1, 32, 8, 8, 8, device=self.device)
        
        # Original
        channels = 32
        blur_kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27.0
        orig = field.clone()
        for c in range(channels):
            diffused = torch.nn.functional.conv3d(
                orig[:, c:c+1], blur_kernel, padding=1
            )
            orig[:, c:c+1] = 0.9 * orig[:, c:c+1] + 0.1 * diffused
        
        # Optimized
        opt = self.ops.optimized_diffusion(field, 0.1)
        
        # Should be very close
        max_diff = (orig - opt).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Diffusion differs by {max_diff}")
    
    def test_buffer_reuse(self):
        """Buffers should be reused, not reallocated"""
        buffers = PreAllocatedBuffers(8, 16, self.device)
        
        buf1 = buffers.get('evolution')
        buf2 = buffers.get('evolution')
        
        # Should be same memory location
        self.assertEqual(buf1.data_ptr(), buf2.data_ptr())
    
    def test_grouped_convolution_speedup(self):
        """Grouped convolution should be faster than loop"""
        if self.device != 'cuda':
            self.skipTest("GPU required for performance test")
        
        import time
        
        field = torch.randn(1, 64, 16, 16, 16, device='cuda')
        
        # Time loop version (simplified)
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        blur = torch.ones(1, 1, 3, 3, 3, device='cuda') / 27
        result = field.clone()
        for c in range(64):
            conv = torch.nn.functional.conv3d(result[:, c:c+1], blur, padding=1)
            result[:, c:c+1] = result[:, c:c+1] * 0.9 + conv * 0.1
        
        torch.cuda.synchronize()
        loop_time = time.perf_counter() - start
        
        # Time optimized version
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = self.ops.optimized_diffusion(field, 0.1)
        
        torch.cuda.synchronize()
        opt_time = time.perf_counter() - start
        
        speedup = loop_time / opt_time
        self.assertGreater(speedup, 2.0, f"Only {speedup:.1f}x speedup")


class TestEmergence(unittest.TestCase):
    """Test for emergent behaviors (not specific behaviors, just that they emerge)"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_exploration_emerges(self):
        """Brain should show exploratory behavior"""
        brain = create_pure_field_brain(
            size='tiny',
            device=self.device,
            aggressive=True  # Aggressive params for faster emergence
        )
        
        # Constant input
        constant_input = torch.ones(10, device=self.device) * 0.1
        
        outputs = []
        for _ in range(30):
            output = brain(constant_input)
            outputs.append(output.cpu().numpy())
        
        # Calculate variance (exploration indicator)
        variance = np.var(outputs, axis=0).mean()
        
        # Should show some variation
        self.assertGreater(variance, 0.0001, "No exploration emerged")
    
    def test_learning_changes_behavior(self):
        """Behavior should change with learning"""
        brain = create_pure_field_brain(size='tiny', device=self.device)
        
        # Capture early behavior
        early_outputs = []
        for _ in range(10):
            out = brain(torch.randn(10, device=self.device))
            early_outputs.append(out.mean().item())
        
        # Apply learning
        for _ in range(50):
            brain(torch.randn(10, device=self.device))
            brain.learn_from_prediction_error(
                torch.randn(10, device=self.device),
                torch.randn(10, device=self.device)
            )
        
        # Capture late behavior
        late_outputs = []
        for _ in range(10):
            out = brain(torch.randn(10, device=self.device))
            late_outputs.append(out.mean().item())
        
        # Behaviors should differ
        early_mean = np.mean(early_outputs)
        late_mean = np.mean(late_outputs)
        
        # We don't care which direction, just that it changed
        change = abs(late_mean - early_mean)
        self.assertGreater(change, 0.0001, "No behavioral change from learning")


class TestPersistence(unittest.TestCase):
    """Test state persistence"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_state_save_load(self):
        """Brain state should persist correctly"""
        brain1 = create_pure_field_brain(size='tiny', device=self.device)
        
        # Run some cycles
        for _ in range(10):
            brain1(torch.randn(10, device=self.device))
        
        # Save state
        state = brain1.get_state_dict()
        
        # Create new brain and load state
        brain2 = create_pure_field_brain(size='tiny', device=self.device)
        brain2.load_state_dict(state)
        
        # Should produce same output
        test_input = torch.randn(10, device=self.device)
        out1 = brain1(test_input.clone())
        out2 = brain2(test_input.clone())
        
        self.assertTrue(torch.allclose(out1, out2, rtol=1e-4))


def run_quick_ci_tests():
    """Run only the most critical tests for CI"""
    suite = unittest.TestSuite()
    
    # Add only critical tests
    suite.addTest(TestBrainSafety('test_motor_bounds'))
    suite.addTest(TestBrainSafety('test_nan_inf_handling'))
    suite.addTest(TestFieldDynamics('test_field_evolution'))
    suite.addTest(TestOptimizations('test_diffusion_equivalence'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run full test suite
    unittest.main(verbosity=2)