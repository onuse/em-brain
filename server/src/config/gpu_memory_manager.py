#!/usr/bin/env python3
"""
GPU Memory Management System

Enforces GPU memory limits configured through the adaptive configuration system.
Provides centralized device selection and memory monitoring.
"""

import torch
from typing import Optional, Dict, Any
import psutil

class GPUMemoryManager:
    """
    Centralized GPU memory management with configurable limits.
    """
    
    _instance = None
    _device = None
    _memory_limit_mb = None
    _monitoring_enabled = True
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._current_allocations = {}
            self._allocation_id_counter = 0
    
    def configure(self, config: Dict[str, Any]):
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
                torch.cuda.set_memory_fraction(memory_fraction)
                print(f"🎯 CUDA memory limit: {self._memory_limit_mb}MB ({memory_fraction*100:.1f}%)")
            except Exception as e:
                print(f"⚠️  Could not set CUDA memory fraction: {e}")
        
        print(f"🖥️  GPU Memory Manager configured:")
        print(f"   Device: {self._device}")
        print(f"   Memory limit: {self._memory_limit_mb}MB" if self._memory_limit_mb > 0 else "   Memory limit: unlimited")
    
    def _detect_best_device(self) -> torch.device:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device(self) -> torch.device:
        """Get the configured device."""
        if self._device is None:
            # Fallback if not configured
            self._device = self._detect_best_device()
        return self._device
    
    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """
        Create tensor on managed device with memory tracking.
        
        Usage: tensor = gpu_manager.create_tensor(data, dtype=torch.float32)
        """
        # Force device to managed device
        kwargs['device'] = self.get_device()
        
        # Create tensor
        tensor = torch.tensor(*args, **kwargs)
        
        # Track allocation if monitoring enabled
        if self._monitoring_enabled:
            self._track_allocation(tensor)
        
        return tensor
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device."""
        device_tensor = tensor.to(self.get_device())
        
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
            'dtype': tensor.dtype
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
        if self._device.type == 'cuda':
            try:
                stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                stats['cuda_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except Exception:
                pass
        elif self._device.type == 'mps':
            # MPS doesn't expose detailed memory info, use system memory
            memory = psutil.virtual_memory()
            stats['system_memory_used_mb'] = (memory.total - memory.available) / (1024 * 1024)
            stats['system_memory_total_mb'] = memory.total / (1024 * 1024)
        
        return stats
    
    def check_memory_pressure(self) -> bool:
        """Check if we're approaching memory limits."""
        if self._memory_limit_mb <= 0:
            return False  # No limit configured
        
        if self._device.type == 'cuda':
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
        
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
    
    def print_memory_report(self):
        """Print detailed memory usage report."""
        stats = self.get_memory_stats()
        
        print(f"\\n📊 GPU Memory Report:")
        print(f"   Device: {stats['device']}")
        print(f"   Memory limit: {stats['memory_limit_mb']}MB" if stats['memory_limit_mb'] > 0 else "   Memory limit: unlimited")
        print(f"   Tracked allocations: {stats['tracked_allocations']}")
        print(f"   Tracked memory: {stats['tracked_memory_mb']:.1f}MB")
        
        if 'cuda_allocated_mb' in stats:
            print(f"   CUDA allocated: {stats['cuda_allocated_mb']:.1f}MB")
            print(f"   CUDA reserved: {stats['cuda_reserved_mb']:.1f}MB")
            print(f"   CUDA total: {stats['cuda_total_mb']:.1f}MB")
        elif 'system_memory_used_mb' in stats:
            print(f"   System memory used: {stats['system_memory_used_mb']:.1f}MB")
            print(f"   System memory total: {stats['system_memory_total_mb']:.1f}MB")


# Global instance
_gpu_manager = GPUMemoryManager()

def configure_gpu_memory(config: Dict[str, Any]):
    """Configure the global GPU memory manager."""
    _gpu_manager.configure(config)

def get_managed_device() -> torch.device:
    """Get the managed GPU device."""
    return _gpu_manager.get_device()

def create_managed_tensor(*args, **kwargs) -> torch.Tensor:
    """Create tensor on managed device."""
    return _gpu_manager.create_tensor(*args, **kwargs)

def to_managed_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to managed device."""
    return _gpu_manager.to_device(tensor)

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
    # Test the GPU memory manager
    config = {
        'system': {
            'device_type': 'auto',
            'gpu_memory_limit_mb': 1024
        }
    }
    
    configure_gpu_memory(config)
    
    # Test tensor creation
    test_tensor = create_managed_tensor([1.0, 2.0, 3.0])
    print(f"Created tensor: {test_tensor}")
    
    # Print memory report
    print_gpu_memory_report()