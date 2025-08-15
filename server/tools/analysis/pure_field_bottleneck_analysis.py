#!/usr/bin/env python3
"""
Deep Bottleneck Analysis for PureFieldBrain
Identifies specific performance bottlenecks and optimization opportunities
"""

import torch
import torch.profiler
import time
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS


class BottleneckAnalyzer:
    """Analyzes PureFieldBrain for performance bottlenecks"""
    
    def __init__(self, config_name: str = 'hardware_constrained'):
        self.config_name = config_name
        self.config = SCALE_CONFIGS[config_name]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create brain
        self.brain = PureFieldBrain(
            input_dim=10,
            output_dim=4,
            scale_config=self.config,
            device=self.device,
            aggressive=True
        )
        
        # Warmup
        for _ in range(10):
            self.brain(torch.randn(10, device=self.device))
    
    def analyze_memory_patterns(self) -> Dict:
        """Analyze memory access patterns and efficiency"""
        
        print("\n🔍 Analyzing Memory Patterns...")
        print("-" * 50)
        
        results = {}
        
        # Check field tensor properties
        field = self.brain.field
        results['field_shape'] = list(field.shape)
        results['field_dtype'] = str(field.dtype)
        results['field_device'] = str(field.device)
        results['field_contiguous'] = field.is_contiguous()
        results['field_size_mb'] = field.numel() * field.element_size() / (1024**2)
        
        print(f"Field shape: {results['field_shape']}")
        print(f"Field size: {results['field_size_mb']:.2f} MB")
        print(f"Contiguous: {results['field_contiguous']}")
        
        # Check for memory alignment
        if self.device == 'cuda':
            # Check if tensor is aligned for optimal GPU access
            stride = field.stride()
            results['strides'] = stride
            # Optimal if innermost dimension is contiguous
            results['optimal_layout'] = stride[-1] == 1
            print(f"Memory layout optimal: {results['optimal_layout']}")
        
        # Test memory bandwidth
        print("\nTesting memory bandwidth...")
        test_tensor = torch.randn_like(field)
        
        # Read bandwidth
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = field.sum()
            if self.device == 'cuda':
                torch.cuda.synchronize()
        read_time = time.perf_counter() - start
        read_bandwidth_gb = (field.numel() * 4 * iterations) / (read_time * 1e9)
        results['read_bandwidth_gb'] = read_bandwidth_gb
        print(f"Read bandwidth: {read_bandwidth_gb:.1f} GB/s")
        
        # Write bandwidth
        start = time.perf_counter()
        for _ in range(iterations):
            field.copy_(test_tensor)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        write_time = time.perf_counter() - start
        write_bandwidth_gb = (field.numel() * 4 * iterations) / (write_time * 1e9)
        results['write_bandwidth_gb'] = write_bandwidth_gb
        print(f"Write bandwidth: {write_bandwidth_gb:.1f} GB/s")
        
        return results
    
    def analyze_computation_patterns(self) -> Dict:
        """Analyze computational patterns and bottlenecks"""
        
        print("\n🔧 Analyzing Computation Patterns...")
        print("-" * 50)
        
        results = {}
        
        # Test individual components
        input_tensor = torch.randn(10, device=self.device)
        
        # 1. Sensory injection
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.brain._inject_sensory_hierarchical(input_tensor)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        sensory_time = (time.perf_counter() - start) / iterations * 1000
        results['sensory_injection_ms'] = sensory_time
        print(f"Sensory injection: {sensory_time:.3f}ms")
        
        # 2. Field evolution
        start = time.perf_counter()
        for _ in range(iterations):
            self.brain._evolve_hierarchical(0.0)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        evolution_time = (time.perf_counter() - start) / iterations * 1000
        results['field_evolution_ms'] = evolution_time
        print(f"Field evolution: {evolution_time:.3f}ms")
        
        # 3. Motor extraction
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.brain._extract_motor_hierarchical()
            if self.device == 'cuda':
                torch.cuda.synchronize()
        motor_time = (time.perf_counter() - start) / iterations * 1000
        results['motor_extraction_ms'] = motor_time
        print(f"Motor extraction: {motor_time:.3f}ms")
        
        # Analyze evolution kernel operations
        level = self.brain.levels[0]
        kernel = level.evolution_kernel
        field = level.field
        
        print(f"\nEvolution kernel shape: {list(kernel.shape)}")
        print(f"Kernel parameters: {kernel.numel():,}")
        
        # Test convolution performance
        field_reshaped = field.permute(3, 0, 1, 2).unsqueeze(0)
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.nn.functional.conv3d(
                field_reshaped,
                kernel,
                padding=1,
                groups=1
            )
            if self.device == 'cuda':
                torch.cuda.synchronize()
        conv_time = (time.perf_counter() - start) / iterations * 1000
        results['convolution_ms'] = conv_time
        print(f"3D convolution time: {conv_time:.3f}ms")
        
        # Calculate theoretical FLOPS
        conv_flops = 2 * kernel.numel() * field.shape[0] * field.shape[1] * field.shape[2]
        results['conv_gflops'] = conv_flops / 1e9
        results['achieved_gflops'] = (conv_flops / 1e9) / (conv_time / 1000)
        print(f"Theoretical GFLOPs per conv: {results['conv_gflops']:.2f}")
        print(f"Achieved GFLOPs: {results['achieved_gflops']:.1f}")
        
        return results
    
    def analyze_bottlenecks(self) -> Dict:
        """Identify specific bottlenecks"""
        
        print("\n⚠️ Bottleneck Analysis...")
        print("-" * 50)
        
        bottlenecks = []
        
        # Run full forward pass timing
        input_tensor = torch.randn(10, device=self.device)
        
        # Time 100 forward passes
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = self.brain(input_tensor)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Forward pass: {mean_time:.2f}±{std_time:.2f}ms")
        
        # Check specific bottlenecks
        
        # 1. Memory bottleneck
        field_mb = self.brain.field.numel() * 4 / (1024**2)
        if field_mb > 10:
            bottlenecks.append(f"Large field size ({field_mb:.1f}MB) may cause cache misses")
        
        # 2. Computation bottleneck
        if mean_time > 10:
            bottlenecks.append(f"High computation time ({mean_time:.1f}ms)")
        
        # 3. Tensor reshaping overhead
        # Check how many reshapes happen in forward pass
        original_permute = torch.Tensor.permute
        reshape_count = [0]
        
        def counting_permute(self, *args, **kwargs):
            reshape_count[0] += 1
            return original_permute(self, *args, **kwargs)
        
        torch.Tensor.permute = counting_permute
        _ = self.brain(input_tensor)
        torch.Tensor.permute = original_permute
        
        if reshape_count[0] > 5:
            bottlenecks.append(f"Excessive tensor reshaping ({reshape_count[0]} permutes per forward)")
        
        print(f"Tensor permutes per forward: {reshape_count[0]}")
        
        # 4. Check for unnecessary copies
        if not self.brain.field.is_contiguous():
            bottlenecks.append("Non-contiguous field tensor may cause extra copies")
        
        # 5. Check for autocast overhead
        if self.device == 'cuda':
            # Test with and without autocast
            with torch.autocast('cuda', enabled=False):
                start = time.perf_counter()
                for _ in range(50):
                    _ = self.brain(input_tensor)
                    torch.cuda.synchronize()
                time_without = time.perf_counter() - start
            
            with torch.autocast('cuda', enabled=True):
                start = time.perf_counter()
                for _ in range(50):
                    _ = self.brain(input_tensor)
                    torch.cuda.synchronize()
                time_with = time.perf_counter() - start
            
            autocast_overhead = (time_with - time_without) / time_without * 100
            if autocast_overhead > 10:
                bottlenecks.append(f"Autocast overhead: {autocast_overhead:.1f}%")
            print(f"Autocast overhead: {autocast_overhead:.1f}%")
        
        return {
            'mean_forward_ms': mean_time,
            'std_forward_ms': std_time,
            'bottlenecks': bottlenecks
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest specific optimizations"""
        
        print("\n💡 Optimization Suggestions...")
        print("-" * 50)
        
        suggestions = []
        
        # Analyze current state
        memory_results = self.analyze_memory_patterns()
        compute_results = self.analyze_computation_patterns()
        bottleneck_results = self.analyze_bottlenecks()
        
        # Memory optimizations
        if not memory_results['field_contiguous']:
            suggestions.append("Make field tensor contiguous: field = field.contiguous()")
        
        if memory_results['field_size_mb'] > 5:
            suggestions.append("Consider using float16 for reduced memory and faster computation")
        
        # Computation optimizations
        if compute_results['convolution_ms'] > 2:
            suggestions.append("Consider using depthwise separable convolutions")
            suggestions.append("Try torch.jit.script for the evolution kernel")
        
        if compute_results['achieved_gflops'] < 100 and self.device == 'cuda':
            suggestions.append("GPU underutilized - increase batch size or use streams")
        
        # General optimizations
        if bottleneck_results['mean_forward_ms'] > 10:
            suggestions.append("Consider reducing field size or channel count")
            suggestions.append("Use torch.compile() for 2x speedup (PyTorch 2.0+)")
        
        if self.device == 'cpu':
            suggestions.append("Use GPU for 10-100x speedup")
            suggestions.append("Enable MKL-DNN for CPU optimization")
            suggestions.append("Set torch.set_num_threads() for optimal CPU usage")
        
        # Print suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        return suggestions


def main():
    """Run comprehensive bottleneck analysis"""
    
    print("=" * 70)
    print("PureFieldBrain Bottleneck Analysis")
    print("=" * 70)
    
    # Test hardware_constrained config
    analyzer = BottleneckAnalyzer('hardware_constrained')
    
    # Run analyses
    memory_results = analyzer.analyze_memory_patterns()
    compute_results = analyzer.analyze_computation_patterns()
    bottleneck_results = analyzer.analyze_bottlenecks()
    suggestions = analyzer.suggest_optimizations()
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    total_time = (compute_results['sensory_injection_ms'] + 
                  compute_results['field_evolution_ms'] + 
                  compute_results['motor_extraction_ms'])
    
    print(f"\nTotal component time: {total_time:.2f}ms")
    print(f"Actual forward time: {bottleneck_results['mean_forward_ms']:.2f}ms")
    overhead = bottleneck_results['mean_forward_ms'] - total_time
    print(f"Framework overhead: {overhead:.2f}ms ({overhead/bottleneck_results['mean_forward_ms']*100:.1f}%)")
    
    print(f"\nAchieved rate: {1000/bottleneck_results['mean_forward_ms']:.1f} Hz")
    print(f"Meets 30Hz requirement: {'✅ YES' if bottleneck_results['mean_forward_ms'] < 33.33 else '❌ NO'}")
    
    if bottleneck_results['bottlenecks']:
        print("\n⚠️ Identified Bottlenecks:")
        for bottleneck in bottleneck_results['bottlenecks']:
            print(f"  - {bottleneck}")
    else:
        print("\n✅ No major bottlenecks identified")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION POTENTIAL")
    print("=" * 70)
    
    # Estimate potential improvements
    print("\nEstimated improvements with optimizations:")
    
    current_ms = bottleneck_results['mean_forward_ms']
    
    # Estimate improvements
    improvements = []
    
    if analyzer.device == 'cpu':
        improvements.append(("Use GPU", current_ms * 0.1))  # 10x speedup typical
    
    if memory_results['field_dtype'] == 'torch.float32':
        improvements.append(("Use float16", current_ms * 0.7))  # 30% speedup
    
    if 'torch.compile' not in str(analyzer.brain.forward):
        improvements.append(("Use torch.compile", current_ms * 0.5))  # 2x speedup
    
    for name, estimated_ms in improvements:
        estimated_hz = 1000 / estimated_ms
        print(f"  {name}: ~{estimated_ms:.1f}ms ({estimated_hz:.0f} Hz)")
    
    # Best case scenario
    if improvements:
        best_case = min(imp[1] for imp in improvements)
        best_hz = 1000 / best_case
        print(f"\nBest case with all optimizations: ~{best_case:.1f}ms ({best_hz:.0f} Hz)")


if __name__ == "__main__":
    main()