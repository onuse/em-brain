#!/usr/bin/env python3
"""
Enhanced GPU Memory Management System

Enforces GPU memory limits and handles device selection with full awareness
of MPS limitations and workarounds.

MPS Known Issues:
1. Limited to 16 dimensions (hardcoded PyTorch limitation)
2. Float64 performance issues (force float32)
3. Performance degradation with 11D+ tensors
"""

import torch
from typing import Optional, Dict, Any, Tuple
import psutil
import warnings


class EnhancedGPUMemoryManager:
    """
    Centralized GPU memory management with MPS-aware device selection.
    
    Handles:
    - Device selection with MPS dimension limitations
    - Memory limit enforcement (CUDA)
    - Memory monitoring and pressure detection
    - MPS-specific workarounds
    """
    
    _instance = None
    _device = None
    _memory_limit_mb = None
    _monitoring_enabled = True
    
    # MPS limitations
    MPS_MAX_DIMENSIONS = 16
    MPS_PERFORMANCE_THRESHOLD = 10  # Use CPU for >10D tensors on MPS
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._current_allocations = {}
            self._allocation_id_counter = 0
            self._mps_warnings_shown = set()
    
    def configure(self, config: Dict[str, Any], quiet: bool = False):
        """Configure GPU memory management from system config."""
        system_config = config.get('system', {})
        
        # Get memory limit from config
        self._memory_limit_mb = system_config.get('gpu_memory_limit_mb', 0)
        
        # Device selection based on config or auto-detection
        device_type = system_config.get('device_type', 'auto')
        
        if device_type == 'auto':
            self._device = self._detect_best_device()
        elif device_type == 'cuda' and torch.cuda.is_available():
            self._device = torch.device('cuda')
        elif device_type == 'mps' and torch.backends.mps.is_available():
            self._device = torch.device('mps')
        else:
            self._device = torch.device('cpu')
        
        # Set PyTorch memory fraction if on CUDA and limit is specified
        if self._device.type == 'cuda' and self._memory_limit_mb > 0:
            try:
                # Convert MB to bytes and set memory fraction
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_fraction = (self._memory_limit_mb * 1024 * 1024) / total_memory
                memory_fraction = min(memory_fraction, 1.0)  # Cap at 100%
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                if not quiet:
                    print(f"🎯 CUDA memory limit: {self._memory_limit_mb}MB ({memory_fraction*100:.1f}%)")
            except Exception as e:
                if not quiet:
                    print(f"⚠️  Could not set CUDA memory fraction: {e}")
        
        # Quiet initialization - device info shown at server ready
    
    def _detect_best_device(self, tensor_dimensions: Optional[int] = None) -> torch.device:
        """
        Detect the best available device, considering tensor dimensions for MPS.
        
        Args:
            tensor_dimensions: Number of tensor dimensions (for MPS compatibility check)
            
        Returns:
            Best available device
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            # Check MPS compatibility if dimensions provided
            if tensor_dimensions is not None:
                if tensor_dimensions > self.MPS_MAX_DIMENSIONS:
                    self._show_mps_warning('dimension_limit', tensor_dimensions)
                    return torch.device('cpu')
                elif tensor_dimensions > self.MPS_PERFORMANCE_THRESHOLD:
                    self._show_mps_warning('performance', tensor_dimensions)
                    return torch.device('cpu')
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device_for_tensor(self, tensor_shape: Tuple[int, ...]) -> torch.device:
        """
        Get the optimal device for a tensor with specific shape.
        
        This is MPS-aware and will fallback to CPU for problematic dimensions.
        
        Args:
            tensor_shape: Shape of the tensor to create
            
        Returns:
            Optimal device for this tensor
        """
        tensor_dimensions = len(tensor_shape)
        
        # If we have a configured device
        if self._device is not None:
            # Special handling for MPS
            if self._device.type == 'mps':
                if tensor_dimensions > self.MPS_MAX_DIMENSIONS:
                    self._show_mps_warning('dimension_limit', tensor_dimensions)
                    return torch.device('cpu')
                elif tensor_dimensions > self.MPS_PERFORMANCE_THRESHOLD:
                    self._show_mps_warning('performance', tensor_dimensions)
                    return torch.device('cpu')
            return self._device
        
        # Auto-detect with dimension awareness
        return self._detect_best_device(tensor_dimensions)
    
    def _show_mps_warning(self, warning_type: str, dimensions: int):
        """Show MPS-related warnings only once per type."""
        if warning_type not in self._mps_warnings_shown:
            self._mps_warnings_shown.add(warning_type)
            if warning_type == 'dimension_limit':
                print(f"💻 Using CPU (MPS limited to {self.MPS_MAX_DIMENSIONS}D, tensor needs {dimensions}D)")
            elif warning_type == 'performance':
                print(f"💻 Using CPU (MPS performance degrades for {dimensions}D tensors)")
    
    def get_device(self) -> torch.device:
        """
        Get the configured device (without dimension checks).
        
        For dimension-aware device selection, use get_device_for_tensor().
        """
        if self._device is None:
            # Fallback if not configured
            self._device = self._detect_best_device()
        return self._device
    
    def create_tensor(self, *args, dtype=None, **kwargs) -> torch.Tensor:
        """
        Create tensor on managed device with memory tracking and MPS fixes.
        
        Usage: tensor = gpu_manager.create_tensor(data, dtype=torch.float32)
        """
        # Force float32 for MPS to avoid float64 issues
        if dtype is None or (self.get_device().type == 'mps' and dtype == torch.float64):
            dtype = torch.float32
        
        # Determine shape for device selection
        if args and hasattr(args[0], 'shape'):
            shape = args[0].shape
        elif 'size' in kwargs:
            shape = kwargs['size']
        elif args and hasattr(args[0], '__len__'):
            # Infer shape from data
            shape = torch.tensor(args[0]).shape
        else:
            shape = ()  # Scalar
        
        # Get optimal device for this tensor
        kwargs['device'] = self.get_device_for_tensor(shape)
        kwargs['dtype'] = dtype
        
        # Create tensor
        tensor = torch.tensor(*args, **kwargs)
        
        # Track allocation if monitoring enabled
        if self._monitoring_enabled:
            self._track_allocation(tensor)
        
        return tensor
    
    def to_device(self, tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """
        Move tensor to optimal device with MPS awareness.
        
        Args:
            tensor: Tensor to move
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Tensor on optimal device
        """
        optimal_device = self.get_device_for_tensor(tensor.shape)
        
        # Force float32 for MPS
        if optimal_device.type == 'mps' and tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        
        device_tensor = tensor.to(optimal_device, non_blocking=non_blocking)
        
        # Track allocation if monitoring enabled
        if self._monitoring_enabled:
            self._track_allocation(device_tensor)
        
        return device_tensor
    
    def _track_allocation(self, tensor: torch.Tensor):
        """Track tensor allocation for monitoring."""
        allocation_id = self._allocation_id_counter
        self._allocation_id_counter += 1
        
        # Calculate tensor size in MB
        size_bytes = tensor.element_size() * tensor.nelement()
        size_mb = size_bytes / (1024 * 1024)
        
        self._current_allocations[allocation_id] = {
            'size_mb': size_mb,
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'device': str(tensor.device)
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            'device': str(self._device),
            'memory_limit_mb': self._memory_limit_mb,
            'tracked_allocations': len(self._current_allocations),
            'tracked_memory_mb': sum(alloc['size_mb'] for alloc in self._current_allocations.values())
        }
        
        # Get device-specific memory info
        if self._device and self._device.type == 'cuda':
            try:
                stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                stats['cuda_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except Exception:
                pass
        elif self._device and self._device.type == 'mps':
            # MPS doesn't expose detailed memory info, use system memory
            memory = psutil.virtual_memory()
            stats['system_memory_used_mb'] = (memory.total - memory.available) / (1024 * 1024)
            stats['system_memory_total_mb'] = memory.total / (1024 * 1024)
            stats['mps_limitations'] = {
                'max_dimensions': self.MPS_MAX_DIMENSIONS,
                'performance_threshold': self.MPS_PERFORMANCE_THRESHOLD
            }
        
        return stats
    
    def check_memory_pressure(self) -> bool:
        """Check if we're approaching memory limits."""
        if self._memory_limit_mb <= 0:
            return False  # No limit configured
        
        if self._device and self._device.type == 'cuda':
            try:
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                return allocated_mb > (self._memory_limit_mb * 0.9)  # 90% threshold
            except Exception:
                return False
        
        # For other devices, use tracked allocations as approximation
        tracked_mb = sum(alloc['size_mb'] for alloc in self._current_allocations.values())
        return tracked_mb > (self._memory_limit_mb * 0.9)
    
    def cleanup_allocations(self):
        """Clean up tracked allocations and free memory."""
        self._current_allocations.clear()
        
        if self._device and self._device.type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 CUDA memory cache cleared")
        elif self._device and self._device.type == 'mps':
            # MPS doesn't have explicit cache clearing, but we can suggest GC
            import gc
            gc.collect()
            print("🧹 Triggered garbage collection for MPS")
    
    def print_memory_report(self):
        """Print detailed memory usage report."""
        stats = self.get_memory_stats()
        
        print(f"\n📊 GPU Memory Report:")
        print(f"   Device: {stats['device']}")
        print(f"   Memory limit: {stats['memory_limit_mb']}MB" if stats['memory_limit_mb'] > 0 else "   Memory limit: unlimited")
        print(f"   Tracked allocations: {stats['tracked_allocations']}")
        print(f"   Tracked memory: {stats['tracked_memory_mb']:.1f}MB")
        
        if 'cuda_allocated_mb' in stats:
            print(f"   CUDA allocated: {stats['cuda_allocated_mb']:.1f}MB")
            print(f"   CUDA reserved: {stats['cuda_reserved_mb']:.1f}MB")
            print(f"   CUDA total: {stats['cuda_total_mb']:.1f}MB")
            utilization = (stats['cuda_allocated_mb'] / stats['cuda_total_mb']) * 100
            print(f"   GPU utilization: {utilization:.1f}%")
        elif 'system_memory_used_mb' in stats:
            print(f"   System memory used: {stats['system_memory_used_mb']:.1f}MB")
            print(f"   System memory total: {stats['system_memory_total_mb']:.1f}MB")
            if 'mps_limitations' in stats:
                print(f"   MPS max dimensions: {stats['mps_limitations']['max_dimensions']}")
                print(f"   MPS performance threshold: {stats['mps_limitations']['performance_threshold']}D")


# Global instance
_gpu_manager = EnhancedGPUMemoryManager()


def configure_gpu_memory(config: Dict[str, Any], quiet: bool = False):
    """Configure the global GPU memory manager."""
    _gpu_manager.configure(config, quiet=quiet)


def get_managed_device() -> torch.device:
    """Get the managed GPU device (without dimension checks)."""
    return _gpu_manager.get_device()


def get_device_for_tensor(tensor_shape: Tuple[int, ...]) -> torch.device:
    """Get optimal device for a tensor with specific shape (MPS-aware)."""
    return _gpu_manager.get_device_for_tensor(tensor_shape)


def create_managed_tensor(*args, **kwargs) -> torch.Tensor:
    """Create tensor on managed device with MPS fixes."""
    return _gpu_manager.create_tensor(*args, **kwargs)


def to_managed_device(tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """Move tensor to optimal managed device."""
    return _gpu_manager.to_device(tensor, non_blocking=non_blocking)


def get_gpu_memory_stats() -> Dict[str, Any]:
    """Get GPU memory statistics."""
    return _gpu_manager.get_memory_stats()


def check_gpu_memory_pressure() -> bool:
    """Check if GPU memory is under pressure."""
    return _gpu_manager.check_memory_pressure()


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    _gpu_manager.cleanup_allocations()


def print_gpu_memory_report():
    """Print GPU memory usage report."""
    _gpu_manager.print_memory_report()


if __name__ == "__main__":
    # Test the enhanced GPU memory manager
    config = {
        'system': {
            'device_type': 'auto',
            'gpu_memory_limit_mb': 1024
        }
    }
    
    configure_gpu_memory(config)
    
    # Test tensor creation with different dimensions
    print("\n🧪 Testing tensor creation:")
    
    # Small tensor - should use GPU/MPS
    small_tensor = create_managed_tensor([1.0, 2.0, 3.0])
    print(f"3D tensor device: {small_tensor.device}")
    
    # 11D tensor - should fallback to CPU on MPS
    shape_11d = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    tensor_11d = create_managed_tensor(torch.randn(shape_11d))
    print(f"11D tensor device: {tensor_11d.device}")
    
    # Print memory report
    print_gpu_memory_report()