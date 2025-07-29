#!/usr/bin/env python3
"""
Performance Profiler for Brain System

Profiles GPU/CPU usage, memory consumption, and identifies bottlenecks.
"""

import torch
import time
import psutil
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation: str
    duration_ms: float
    gpu_memory_mb: float
    cpu_percent: float
    calls: int = 1


class PerformanceProfiler:
    """Profile brain system performance."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = defaultdict(lambda: PerformanceMetrics("", 0, 0, 0, 0))
        self.start_times = {}
        self.start_memory = {}
        
    def start_operation(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.start_memory[name] = torch.cuda.memory_allocated() / 1024 / 1024
        elif self.device.type == 'mps':
            # MPS doesn't have direct memory query yet
            self.start_memory[name] = 0
            
    def end_operation(self, name: str):
        """End timing an operation and record metrics."""
        if name not in self.start_times:
            return
            
        # Get timing
        if self.device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
        duration = (time.perf_counter() - self.start_times[name]) * 1000
        
        # Get memory usage
        memory_mb = 0
        if self.device.type == 'cuda':
            memory_mb = (torch.cuda.memory_allocated() / 1024 / 1024) - self.start_memory[name]
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0)
        
        # Update metrics
        if name in self.metrics:
            m = self.metrics[name]
            m.duration_ms += duration
            m.gpu_memory_mb += memory_mb
            m.cpu_percent = max(m.cpu_percent, cpu_percent)
            m.calls += 1
        else:
            self.metrics[name] = PerformanceMetrics(name, duration, memory_mb, cpu_percent, 1)
        
        # Cleanup
        del self.start_times[name]
        if name in self.start_memory:
            del self.start_memory[name]
    
    def profile_function(self, func: Callable, name: str, *args, **kwargs):
        """Profile a single function call."""
        self.start_operation(name)
        result = func(*args, **kwargs)
        self.end_operation(name)
        return result
    
    def get_report(self) -> str:
        """Generate performance report."""
        report = ["Performance Profile Report", "=" * 50, ""]
        
        # Sort by total time
        sorted_metrics = sorted(
            self.metrics.values(), 
            key=lambda m: m.duration_ms, 
            reverse=True
        )
        
        # Header
        report.append(f"{'Operation':<30} {'Calls':>6} {'Total(ms)':>10} {'Avg(ms)':>10} {'GPU(MB)':>10} {'CPU%':>6}")
        report.append("-" * 80)
        
        # Metrics
        total_time = sum(m.duration_ms for m in sorted_metrics)
        for m in sorted_metrics:
            avg_time = m.duration_ms / m.calls if m.calls > 0 else 0
            pct = (m.duration_ms / total_time * 100) if total_time > 0 else 0
            report.append(
                f"{m.operation:<30} {m.calls:>6} {m.duration_ms:>10.2f} "
                f"{avg_time:>10.2f} {m.gpu_memory_mb:>10.2f} {m.cpu_percent:>6.1f} ({pct:>5.1f}%)"
            )
        
        # Summary
        report.append("")
        report.append(f"Total time: {total_time:.2f}ms")
        if self.device.type == 'cuda':
            report.append(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
        
        return "\n".join(report)
    
    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_times.clear()
        self.start_memory.clear()


def profile_brain_cycle(brain, sensory_input: List[float], profiler: PerformanceProfiler):
    """Profile a single brain cycle with detailed breakdown."""
    
    # Overall cycle
    profiler.start_operation("full_cycle")
    
    # Profile each major step
    with torch.no_grad():  # Disable gradient computation for inference
        
        # Field evolution
        profiler.start_operation("field_evolution")
        brain._evolve_field()
        profiler.end_operation("field_evolution")
        
        # Pattern extraction
        profiler.start_operation("pattern_extraction")
        patterns = brain.pattern_system.extract_patterns(brain.unified_field, n_patterns=10)
        profiler.end_operation("pattern_extraction")
        
        # Attention processing
        profiler.start_operation("attention_processing")
        attention_state = brain._process_attention(sensory_input)
        profiler.end_operation("attention_processing")
        
        # Motor generation
        profiler.start_operation("motor_generation")
        motor_output = brain._generate_motor_action()
        profiler.end_operation("motor_generation")
    
    profiler.end_operation("full_cycle")
    
    return motor_output


def analyze_memory_usage(brain):
    """Analyze memory usage of brain components."""
    report = ["Memory Usage Analysis", "=" * 50, ""]
    
    def get_tensor_size_mb(tensor):
        return tensor.element_size() * tensor.nelement() / 1024 / 1024
    
    # Main field
    field_size = get_tensor_size_mb(brain.unified_field)
    report.append(f"Unified field: {field_size:.2f}MB")
    report.append(f"  Shape: {brain.unified_field.shape}")
    report.append(f"  Device: {brain.unified_field.device}")
    
    # Pattern memory
    if hasattr(brain.pattern_system, 'pattern_memory'):
        pattern_count = len(brain.pattern_system.pattern_memory)
        report.append(f"\nPattern memory: {pattern_count} patterns")
    
    # Field dynamics memory
    if hasattr(brain.field_dynamics, 'pattern_memory'):
        dynamics_patterns = len(brain.field_dynamics.pattern_memory)
        report.append(f"Field dynamics patterns: {dynamics_patterns}")
    
    # Topology regions
    if hasattr(brain, 'topology_regions'):
        report.append(f"\nTopology regions: {len(brain.topology_regions)}")
    
    # Total estimated
    total_mb = field_size  # Add other components as needed
    report.append(f"\nTotal estimated: {total_mb:.2f}MB")
    
    return "\n".join(report)


def identify_bottlenecks(profiler: PerformanceProfiler) -> List[str]:
    """Identify performance bottlenecks."""
    bottlenecks = []
    
    # Find operations taking >20% of total time
    total_time = sum(m.duration_ms for m in profiler.metrics.values())
    for name, metrics in profiler.metrics.items():
        pct = (metrics.duration_ms / total_time * 100) if total_time > 0 else 0
        if pct > 20:
            bottlenecks.append(f"{name}: {pct:.1f}% of total time")
    
    # Find memory-heavy operations
    for name, metrics in profiler.metrics.items():
        if metrics.gpu_memory_mb > 50:  # More than 50MB
            bottlenecks.append(f"{name}: {metrics.gpu_memory_mb:.1f}MB GPU memory")
    
    return bottlenecks


if __name__ == "__main__":
    # Test profiling
    import sys
    sys.path.append('/Users/jkarlsson/Documents/Projects/robot-project/brain/server')
    
    from src.core.simplified_brain_factory import SimplifiedBrainFactory
    
    # Create brain
    factory = SimplifiedBrainFactory({'quiet_mode': True})
    brain_wrapper = factory.create(sensory_dim=24, motor_dim=4)
    brain = brain_wrapper.brain
    
    # Create profiler
    profiler = PerformanceProfiler(brain.device)
    
    print("Profiling brain performance...")
    print(f"Device: {brain.device}")
    print(f"Field shape: {brain.unified_field.shape}")
    print()
    
    # Warmup
    for _ in range(10):
        brain.process_robot_cycle([0.0] * 24)
    
    # Profile
    sensory_input = [0.1] * 24
    for i in range(100):
        profile_brain_cycle(brain, sensory_input, profiler)
    
    # Report
    print(profiler.get_report())
    print()
    print(analyze_memory_usage(brain))
    
    # Bottlenecks
    bottlenecks = identify_bottlenecks(profiler)
    if bottlenecks:
        print("\nBottlenecks identified:")
        for b in bottlenecks:
            print(f"  - {b}")