#!/usr/bin/env python3
"""
Test GPU Memory Manager Integration

Verifies that the enhanced GPU memory manager properly handles
device selection and MPS limitations.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import unittest
import torch
from src.config.enhanced_gpu_memory_manager import (
    EnhancedGPUMemoryManager,
    configure_gpu_memory,
    get_device_for_tensor,
    create_managed_tensor,
    get_gpu_memory_stats
)
from src.core.dynamic_brain_factory import DynamicBrainFactory


class TestGPUMemoryIntegration(unittest.TestCase):
    """Test GPU memory manager integration with MPS awareness."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'system': {
                'device_type': 'auto',
                'gpu_memory_limit_mb': 512
            }
        }
        configure_gpu_memory(self.config)
    
    def test_mps_dimension_limitations(self):
        """Test that MPS dimension limitations are respected."""
        manager = EnhancedGPUMemoryManager()
        
        # Test various tensor dimensions
        test_cases = [
            ((2, 3, 4), 'should use GPU/MPS for 3D'),
            ((2,) * 9, 'should use GPU/MPS for 9D'),
            ((2,) * 10, 'should use GPU/MPS for 10D'),
            ((2,) * 11, 'should use CPU for 11D on MPS'),
            ((2,) * 16, 'should use CPU for 16D on MPS'),
            ((2,) * 17, 'should use CPU for 17D (exceeds MPS limit)'),
        ]
        
        for shape, description in test_cases:
            device = get_device_for_tensor(shape)
            dims = len(shape)
            
            # Check behavior based on available hardware
            if torch.backends.mps.is_available():
                if dims > manager.MPS_MAX_DIMENSIONS:
                    self.assertEqual(device.type, 'cpu', 
                                   f"{description}: {dims}D exceeds MPS limit")
                elif dims > manager.MPS_PERFORMANCE_THRESHOLD:
                    self.assertEqual(device.type, 'cpu',
                                   f"{description}: {dims}D has MPS performance issues")
                else:
                    self.assertIn(device.type, ['mps', 'cuda'],
                                f"{description}: Should use GPU")
            elif torch.cuda.is_available():
                # CUDA has no dimension limitations
                self.assertEqual(device.type, 'cuda',
                               f"{description}: CUDA supports all dimensions")
            else:
                self.assertEqual(device.type, 'cpu',
                               f"{description}: No GPU available")
    
    def test_float64_handling(self):
        """Test that float64 is converted to float32 on MPS."""
        # Create a float64 tensor
        data = [1.0, 2.0, 3.0]
        
        # Try to create with float64
        tensor = create_managed_tensor(data, dtype=torch.float64)
        
        # Check dtype based on device
        if tensor.device.type == 'mps':
            self.assertEqual(tensor.dtype, torch.float32,
                           "MPS should force float32")
        else:
            self.assertEqual(tensor.dtype, torch.float64,
                           "Non-MPS should preserve float64")
    
    def test_brain_creation_with_memory_manager(self):
        """Test that brain creation uses the GPU memory manager."""
        factory = DynamicBrainFactory({'quiet_mode': True})
        
        # Create brain with 11D tensor (should trigger MPS fallback)
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Check device assignment
        if torch.backends.mps.is_available():
            # With 11D tensor, should fallback to CPU on MPS
            tensor_dims = len(brain.unified_field.shape)
            if tensor_dims > 10:
                self.assertEqual(brain.device.type, 'cpu',
                               f"11D tensor should use CPU on MPS")
        
        # Verify brain is functional
        sensory_input = [0.5] * 17
        motor_output, state = brain.process_robot_cycle(sensory_input)
        self.assertIsNotNone(motor_output)
        self.assertIsNotNone(state)
    
    def test_memory_stats_collection(self):
        """Test memory statistics collection."""
        # Create some tensors
        tensors = []
        for i in range(5):
            shape = (10, 20, 30)
            tensor = create_managed_tensor(torch.randn(shape))
            tensors.append(tensor)
        
        # Get memory stats
        stats = get_gpu_memory_stats()
        
        # Check basic stats
        self.assertIn('device', stats)
        self.assertIn('tracked_allocations', stats)
        self.assertIn('tracked_memory_mb', stats)
        
        # Check device-specific stats
        if stats['device'].startswith('cuda'):
            self.assertIn('cuda_allocated_mb', stats)
            self.assertIn('cuda_total_mb', stats)
        elif stats['device'] == 'mps':
            self.assertIn('mps_limitations', stats)
            self.assertEqual(stats['mps_limitations']['max_dimensions'], 16)
    
    def test_memory_limit_configuration(self):
        """Test memory limit configuration."""
        stats = get_gpu_memory_stats()
        
        # Check configured limit
        self.assertEqual(stats['memory_limit_mb'], 512)
        
        # For CUDA, verify the limit is enforced
        if torch.cuda.is_available():
            # This would be set via torch.cuda.set_per_process_memory_fraction
            # We can't easily test the actual enforcement without potentially
            # causing OOM, so we just verify the configuration
            pass


if __name__ == '__main__':
    unittest.main()