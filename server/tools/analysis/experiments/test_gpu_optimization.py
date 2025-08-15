#!/usr/bin/env python3
"""
Test GPU Memory Optimization

Tests memory optimization and batch processing capabilities.
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.brains.field.gpu_memory_optimizer import GPUMemoryOptimizer, BatchedBrainProcessor
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_memory_optimization():
    """Test GPU memory optimization techniques."""
    print("Testing GPU Memory Optimization")
    print("-" * 50)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Create optimizer
    optimizer = GPUMemoryOptimizer(device)
    
    # Test tensor pooling
    print("\nTensor Pooling Test:")
    shape = (32, 32, 32, 64)
    
    # Allocate multiple tensors
    tensors = []
    for i in range(5):
        t = optimizer.get_pooled_tensor(shape, key=f"test_{i % 2}")
        tensors.append(t)
    
    print(f"Pool size: {len(optimizer.memory_pool)}")
    print(f"Expected: 2 (reusing tensors)")
    
    # Test fused operations
    print("\nFused Operations Test:")
    field = torch.randn(shape, device=device)
    
    # Time separate operations
    start = time.perf_counter()
    for _ in range(10):
        temp = field.clone()
        temp *= 0.99  # decay
        # Simple diffusion simulation
        temp = temp + torch.randn_like(temp) * 0.001
    separate_time = (time.perf_counter() - start) * 100
    
    # Time fused operations
    start = time.perf_counter()
    for _ in range(10):
        field = optimizer.fused_decay_diffusion(field.clone(), 0.99, 0.001)
    fused_time = (time.perf_counter() - start) * 100
    
    print(f"Separate operations: {separate_time:.2f}ms")
    print(f"Fused operations: {fused_time:.2f}ms")
    print(f"Speedup: {separate_time / fused_time:.2f}x")
    
    # Memory stats
    stats = optimizer.get_memory_stats()
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")


def test_batch_processing():
    """Test batch processing of multiple brains."""
    print("\n\nTesting Batch Processing")
    print("-" * 50)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    batch_processor = BatchedBrainProcessor(device)
    
    # Create multiple brain fields
    print("\nCreating test fields...")
    n_brains = 4
    fields = [torch.randn(32, 32, 32, 64, device=device) for _ in range(n_brains)]
    
    # Test individual processing
    print("\nIndividual Processing:")
    start = time.perf_counter()
    for _ in range(10):
        results_individual = []
        for field in fields:
            evolved = field * 0.99  # Simple evolution
            results_individual.append(evolved)
    individual_time = (time.perf_counter() - start) * 100
    print(f"Time: {individual_time:.2f}ms")
    
    # Test batch processing
    print("\nBatch Processing:")
    start = time.perf_counter()
    for _ in range(10):
        results_batch = batch_processor.batch_process_fields(fields, 0.99, 0.0)
    batch_time = (time.perf_counter() - start) * 100
    print(f"Time: {batch_time:.2f}ms")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    
    # Test batch pattern extraction
    print("\nBatch Pattern Extraction:")
    start = time.perf_counter()
    patterns = batch_processor.batch_extract_patterns(fields, n_patterns=5)
    pattern_time = (time.perf_counter() - start) * 1000
    print(f"Time for {n_brains} brains: {pattern_time:.2f}ms")
    print(f"Time per brain: {pattern_time / n_brains:.2f}ms")
    print(f"Patterns extracted: {len(patterns)} brains × {len(patterns[0])} patterns")


def test_real_brain_optimization():
    """Test optimization with real brain instances."""
    print("\n\nTesting Real Brain Optimization")
    print("-" * 50)
    
    # Create optimized brain with memory optimization
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=True
    )
    
    # Add memory optimizer
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    mem_optimizer = GPUMemoryOptimizer(device)
    
    # Override field evolution with optimized version
    original_evolve = brain.field_dynamics.evolve_field
    
    def optimized_evolve(field):
        return mem_optimizer.optimize_field_evolution(
            field, 
            brain.field_decay_rate,
            brain.field_diffusion_rate,
            brain.modulation.get('spontaneous_weight', 0.5)
        )
    
    brain.field_dynamics.evolve_field = optimized_evolve
    
    # Benchmark
    sensory_input = [0.1] * 24
    
    print("\nMemory-Optimized Brain:")
    times = []
    for i in range(20):
        start = time.perf_counter()
        _, state = brain.process_robot_cycle(sensory_input)
        times.append((time.perf_counter() - start) * 1000)
        
        if i == 0:
            print(f"First cycle: {times[0]:.2f}ms")
    
    avg_time = sum(times[5:]) / len(times[5:])  # Skip warmup
    print(f"Average cycle: {avg_time:.2f}ms")
    print(f"Min/Max: {min(times):.2f}ms / {max(times):.2f}ms")
    
    # Memory stats
    if device.type == 'cuda':
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    test_memory_optimization()
    test_batch_processing()
    test_real_brain_optimization()