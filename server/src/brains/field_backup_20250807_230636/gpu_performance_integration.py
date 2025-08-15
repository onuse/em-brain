"""
GPU Performance Integration - Week 1 Implementation
Provides factory methods and utilities to enable GPU optimizations
"""

import torch
from typing import Optional, Type, Any, Dict
import logging

from .unified_field_brain import UnifiedFieldBrain
from .optimized_unified_field_brain import OptimizedUnifiedFieldBrain
from .gpu_optimizations import GPUProfiler

logger = logging.getLogger(__name__)


class GPUBrainFactory:
    """Factory for creating GPU-optimized brain instances"""
    
    @staticmethod
    def create_optimized_brain(
        sensory_dim: int = 16,
        motor_dim: int = 5,
        spatial_resolution: int = 32,
        device: Optional[torch.device] = None,
        force_gpu: bool = False,
        quiet_mode: bool = False
    ) -> UnifiedFieldBrain:
        """
        Create GPU-optimized brain if possible, fallback to standard brain
        
        Args:
            sensory_dim: Number of sensors
            motor_dim: Number of motors  
            spatial_resolution: Spatial resolution (32 recommended)
            device: Specific device to use
            force_gpu: Force GPU optimization (will fail if GPU unavailable)
            quiet_mode: Suppress output
        
        Returns:
            Optimized brain instance
        """
        # Device selection logic
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        # Check if GPU optimization is viable
        can_optimize = (device.type in ['cuda', 'mps'] and 
                       torch.cuda.is_available() if device.type == 'cuda' else True)
        
        if can_optimize or force_gpu:
            try:
                brain = OptimizedUnifiedFieldBrain(
                    sensory_dim=sensory_dim,
                    motor_dim=motor_dim,
                    spatial_resolution=spatial_resolution,
                    device=device,
                    quiet_mode=quiet_mode
                )
                
                if not quiet_mode:
                    print(f"✅ Created GPU-optimized brain on {device}")
                    memory_stats = brain.get_performance_stats()
                    print(f"   GPU Memory: {memory_stats.get('gpu_memory_allocated_mb', 0):.1f}MB allocated")
                
                return brain
                
            except Exception as e:
                if force_gpu:
                    raise RuntimeError(f"Failed to create GPU-optimized brain: {e}")
                
                logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
        
        # Fallback to standard brain
        brain = UnifiedFieldBrain(
            sensory_dim=sensory_dim,
            motor_dim=motor_dim,
            spatial_resolution=spatial_resolution,
            device=torch.device('cpu'),
            quiet_mode=quiet_mode
        )
        
        if not quiet_mode:
            print(f"⚠️ Using standard brain on CPU (GPU optimization unavailable)")
            
        return brain


class PerformanceBenchmark:
    """Benchmark GPU vs CPU performance"""
    
    @staticmethod
    def run_cycle_benchmark(
        brain: UnifiedFieldBrain,
        n_cycles: int = 100,
        sensory_dim: int = 16
    ) -> Dict[str, float]:
        """
        Benchmark brain processing cycles
        
        Args:
            brain: Brain instance to benchmark
            n_cycles: Number of cycles to run
            sensory_dim: Number of sensory inputs
        
        Returns:
            Performance metrics
        """
        import time
        import numpy as np
        
        # Warm up
        dummy_sensory = [0.1] * (sensory_dim + 1)  # +1 for reward
        for _ in range(5):
            brain.process_robot_cycle(dummy_sensory)
        
        # Benchmark
        cycle_times = []
        
        start_time = time.perf_counter()
        
        for cycle in range(n_cycles):
            # Vary sensory input to avoid caching effects
            sensory_input = [np.sin(cycle * 0.1 + i) * 0.5 for i in range(sensory_dim)]
            sensory_input.append(0.0)  # Reward
            
            cycle_start = time.perf_counter()
            motor_output, brain_state = brain.process_robot_cycle(sensory_input)
            cycle_end = time.perf_counter()
            
            cycle_times.append((cycle_end - cycle_start) * 1000)  # ms
        
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        cycle_times = np.array(cycle_times)
        
        results = {
            'total_time_ms': total_time * 1000,
            'avg_cycle_time_ms': np.mean(cycle_times),
            'std_cycle_time_ms': np.std(cycle_times),
            'min_cycle_time_ms': np.min(cycle_times),
            'max_cycle_time_ms': np.max(cycle_times),
            'cycles_per_second': n_cycles / total_time,
            'device': str(brain.device),
            'optimization_enabled': isinstance(brain, OptimizedUnifiedFieldBrain)
        }
        
        return results
    
    @staticmethod
    def compare_cpu_gpu_performance(
        sensory_dim: int = 16,
        motor_dim: int = 5,
        n_cycles: int = 50
    ) -> Dict[str, Any]:
        """
        Compare CPU vs GPU brain performance
        
        Returns:
            Comparison results
        """
        results = {}
        
        # Test CPU brain
        try:
            cpu_brain = UnifiedFieldBrain(
                sensory_dim=sensory_dim,
                motor_dim=motor_dim,
                device=torch.device('cpu'),
                quiet_mode=True
            )
            
            results['cpu'] = PerformanceBenchmark.run_cycle_benchmark(
                cpu_brain, n_cycles, sensory_dim
            )
            
        except Exception as e:
            results['cpu'] = {'error': str(e)}
        
        # Test GPU brain if available
        if torch.cuda.is_available():
            try:
                gpu_brain = GPUBrainFactory.create_optimized_brain(
                    sensory_dim=sensory_dim,
                    motor_dim=motor_dim,
                    quiet_mode=True
                )
                
                results['gpu'] = PerformanceBenchmark.run_cycle_benchmark(
                    gpu_brain, n_cycles, sensory_dim
                )
                
                # Calculate speedup
                if 'cpu' in results and 'avg_cycle_time_ms' in results['cpu']:
                    speedup = results['cpu']['avg_cycle_time_ms'] / results['gpu']['avg_cycle_time_ms']
                    results['speedup'] = f"{speedup:.1f}x"
                    
            except Exception as e:
                results['gpu'] = {'error': str(e)}
        else:
            results['gpu'] = {'error': 'CUDA not available'}
        
        return results


class OptimizationChecker:
    """Check which optimizations are available and working"""
    
    @staticmethod
    def check_environment() -> Dict[str, Any]:
        """Check optimization environment"""
        return {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'mps_available': torch.backends.mps.is_available(),
            'compilation_available': hasattr(torch, 'compile'),
            'mixed_precision_available': hasattr(torch.cuda.amp, 'autocast')
        }
    
    @staticmethod
    def verify_optimizations(brain: UnifiedFieldBrain) -> Dict[str, bool]:
        """Verify that optimizations are working"""
        checks = {}
        
        # Check if using optimized brain
        checks['optimized_brain_class'] = isinstance(brain, OptimizedUnifiedFieldBrain)
        
        # Check device
        checks['gpu_device'] = brain.device.type in ['cuda', 'mps']
        
        # Check GPU memory pools
        if hasattr(brain, 'memory_pool'):
            checks['memory_pools'] = True
        else:
            checks['memory_pools'] = False
        
        # Check for key optimized methods
        optimized_methods = [
            '_fused_field_evolution_kernel',
            '_extract_motor_tendencies_parallel',
            '_process_sensory_gpu'
        ]
        
        for method in optimized_methods:
            checks[f'method_{method}'] = hasattr(brain, method)
        
        return checks


# Convenience functions for quick setup
def create_optimized_brain(**kwargs) -> UnifiedFieldBrain:
    """Create optimized brain with default settings"""
    return GPUBrainFactory.create_optimized_brain(**kwargs)


def quick_performance_test():
    """Run a quick performance test"""
    print("🚀 GPU Optimization Performance Test")
    print("=" * 50)
    
    # Check environment
    env = OptimizationChecker.check_environment()
    print(f"PyTorch: {env['pytorch_version']}")
    print(f"CUDA Available: {env['cuda_available']}")
    if env['cuda_available']:
        print(f"CUDA Device: {env['cuda_device_name']}")
    print()
    
    # Create and test brain
    brain = create_optimized_brain(quiet_mode=False)
    print()
    
    # Verify optimizations
    optimizations = OptimizationChecker.verify_optimizations(brain)
    print("Optimization Status:")
    for check, status in optimizations.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check.replace('_', ' ').title()}")
    print()
    
    # Quick benchmark
    print("Running 20-cycle benchmark...")
    results = PerformanceBenchmark.run_cycle_benchmark(brain, n_cycles=20)
    
    print(f"Average cycle time: {results['avg_cycle_time_ms']:.2f}ms")
    print(f"Cycles per second: {results['cycles_per_second']:.1f}")
    print(f"Device: {results['device']}")
    
    if hasattr(brain, 'get_performance_stats'):
        perf_stats = brain.get_performance_stats()
        if 'gpu_memory_allocated_mb' in perf_stats:
            print(f"GPU Memory: {perf_stats['gpu_memory_allocated_mb']:.1f}MB")
    
    return results


if __name__ == "__main__":
    quick_performance_test()