#!/usr/bin/env python3
"""
Comprehensive performance profiling of UnifiedFieldBrain.
Identifies where the actual computational work is being done.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import time
import cProfile
import pstats
from io import StringIO
import torch
import numpy as np
from brains.field.core_brain import UnifiedFieldBrain
import psutil
import gc

class BrainPerformanceProfiler:
    def __init__(self, spatial_resolution=8):
        self.spatial_resolution = spatial_resolution
        self.brain = None
        self.process = psutil.Process()
        
    def profile_initialization(self):
        """Profile brain initialization."""
        print("\n=== BRAIN INITIALIZATION PROFILING ===")
        
        # Memory before
        gc.collect()
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        start = time.perf_counter()
        self.brain = UnifiedFieldBrain(
            spatial_resolution=self.spatial_resolution, 
            quiet_mode=True
        )
        init_time = (time.perf_counter() - start) * 1000
        
        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initialization time: {init_time:.1f}ms")
        print(f"Memory allocated: {mem_after - mem_before:.1f}MB")
        print(f"Field shape: {self.brain.unified_field.shape}")
        print(f"Field elements: {self.brain.unified_field.numel():,}")
        print(f"Field memory: {self.brain.unified_field.numel() * 4 / 1024 / 1024:.1f}MB")
        
    def profile_single_cycle(self):
        """Profile a single brain cycle in detail."""
        print("\n=== SINGLE CYCLE DETAILED TIMING ===")
        
        sensory_input = [0.5] * 24
        
        # Warm up
        self.brain.process_robot_cycle(sensory_input)
        
        # Detailed timing
        timings = {}
        
        # Override methods to add timing
        original_methods = {}
        methods_to_time = [
            '_robot_sensors_to_field_experience',
            '_apply_field_experience', 
            '_evolve_unified_field',
            '_field_gradients_to_robot_action',
            '_calculate_gradient_flows',
            '_apply_spatial_diffusion',
            '_apply_constraint_guided_evolution',
            '_update_topology_regions'
        ]
        
        for method_name in methods_to_time:
            if hasattr(self.brain, method_name):
                original_methods[method_name] = getattr(self.brain, method_name)
                timings[method_name] = []
                
                def create_timed_method(name, original):
                    def timed_method(*args, **kwargs):
                        start = time.perf_counter()
                        result = original(*args, **kwargs)
                        timings[name].append((time.perf_counter() - start) * 1000)
                        return result
                    return timed_method
                
                setattr(self.brain, method_name, create_timed_method(method_name, original_methods[method_name]))
        
        # Run cycle
        total_start = time.perf_counter()
        action, state = self.brain.process_robot_cycle(sensory_input)
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Restore original methods
        for method_name, original in original_methods.items():
            setattr(self.brain, method_name, original)
        
        # Print timings
        print(f"\nTotal cycle time: {total_time:.1f}ms")
        print("\nMethod timings:")
        for method, times in sorted(timings.items(), key=lambda x: sum(x[1]), reverse=True):
            if times:
                total = sum(times)
                avg = total / len(times)
                pct = (total / total_time) * 100
                print(f"  {method:<40} {avg:>8.2f}ms ({pct:>5.1f}%)")
        
        return total_time
        
    def profile_with_cprofile(self):
        """Use cProfile for detailed profiling."""
        print("\n=== CPROFILE ANALYSIS (5 cycles) ===")
        
        sensory_input = [0.5] * 24
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run 5 cycles
        for _ in range(5):
            self.brain.process_robot_cycle(sensory_input)
            
        profiler.disable()
        
        # Analyze results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        print(s.getvalue())
        
    def profile_tensor_operations(self):
        """Profile tensor operations specifically."""
        print("\n=== TENSOR OPERATION PROFILING ===")
        
        # Test key tensor operations
        field = self.brain.unified_field
        
        operations = {
            'Field decay (multiply)': lambda: field * 0.999,
            'Field baseline (maximum)': lambda: torch.maximum(field, torch.tensor(0.01)),
            'Gradient computation': lambda: torch.gradient(field[:,:,:,0,0,0,0,0,0,0,0]),
            'Mean over 3x3x3': lambda: torch.mean(field[2:5, 2:5, 2:5]),
            'Field imprint': lambda: field.__setitem__((3,3,3,5,7,1,1,1,0,1,0), field[3,3,3,5,7,1,1,1,0,1,0] + 0.5),
            'Norm calculation': lambda: torch.norm(field[3,3,3]),
        }
        
        for name, op in operations.items():
            # Warm up
            op()
            
            # Time
            times = []
            for _ in range(10):
                torch.cuda.synchronize() if field.is_cuda else None
                start = time.perf_counter()
                op()
                torch.cuda.synchronize() if field.is_cuda else None
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            print(f"{name:<30} {avg_time:>8.2f}ms")
            
    def profile_scaling(self):
        """Profile how performance scales with resolution."""
        print("\n=== PERFORMANCE SCALING ANALYSIS ===")
        
        resolutions = [3, 5, 8, 10]
        results = []
        
        for res in resolutions:
            # Create new brain
            brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
            
            # Time cycles
            sensory_input = [0.5] * 24
            
            # Warm up
            brain.process_robot_cycle(sensory_input)
            
            # Time 10 cycles
            times = []
            for _ in range(10):
                start = time.perf_counter()
                brain.process_robot_cycle(sensory_input)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            field_size = brain.unified_field.numel()
            
            results.append({
                'resolution': res,
                'field_size': field_size,
                'cycle_time': avg_time,
                'throughput': 1000 / avg_time  # Hz
            })
            
            print(f"Resolution {res}³: {avg_time:>8.1f}ms/cycle, {field_size:>10,} elements, {1000/avg_time:>5.1f} Hz")
            
            # Clean up
            del brain
            gc.collect()
            
        return results
        
    def analyze_bottlenecks(self):
        """Analyze and summarize bottlenecks."""
        print("\n=== BOTTLENECK ANALYSIS ===")
        
        # Check if using GPU
        print(f"Device: {self.brain.device}")
        print(f"Field dtype: {self.brain.unified_field.dtype}")
        
        # Check cache performance
        if hasattr(self.brain, 'gradient_calculator'):
            stats = self.brain.gradient_calculator.get_cache_stats()
            print(f"\nGradient cache stats:")
            print(f"  Hit rate: {stats['hit_rate']:.1%}")
            print(f"  Cache size: {stats['cache_size']}")
        
        # Memory bandwidth estimate
        field_bytes = self.brain.unified_field.numel() * 4  # float32
        # Assume we touch the field ~10 times per cycle
        bandwidth_per_cycle = field_bytes * 10 / 1024 / 1024  # MB
        print(f"\nEstimated memory bandwidth per cycle: {bandwidth_per_cycle:.1f}MB")
        
        # Computational complexity
        print(f"\nComputational complexity:")
        print(f"  Field dimensions: {self.brain.unified_field.shape}")
        print(f"  Total elements: {self.brain.unified_field.numel():,}")
        print(f"  Gradient dims computed: 5 (spatial + scale + time)")
        
    def run_full_analysis(self):
        """Run complete performance analysis."""
        print("="*60)
        print("UNIFIED FIELD BRAIN PERFORMANCE ANALYSIS")
        print("="*60)
        
        self.profile_initialization()
        cycle_time = self.profile_single_cycle()
        self.profile_tensor_operations()
        self.profile_with_cprofile()
        scaling_results = self.profile_scaling()
        self.analyze_bottlenecks()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Current configuration cycle time: {cycle_time:.1f}ms ({1000/cycle_time:.1f} Hz)")
        print(f"Real-time robot control needs: 25-40ms (25-40 Hz)")
        print(f"Performance gap: {cycle_time/40:.1f}x too slow" if cycle_time > 40 else "✅ Meets real-time requirements")


if __name__ == "__main__":
    profiler = BrainPerformanceProfiler(spatial_resolution=8)
    profiler.run_full_analysis()