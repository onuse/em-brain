#!/usr/bin/env python3
"""
Test JAX GPU functionality on M1 Mac
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

def test_jax_devices():
    """Test available JAX devices"""
    print("JAX Configuration:")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.devices()[0]}")
    print()
    
    # Check if GPU is available
    gpu_available = any(device.device_kind == 'gpu' for device in jax.devices())
    print(f"GPU available: {gpu_available}")
    
    return gpu_available

def test_similarity_computation():
    """Test similarity computation performance on available devices"""
    print("Testing similarity computation performance...")
    
    # Create test data matching our brain
    n_nodes = 2140
    context_dim = 8
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # Generate test data  
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0
    
    # JAX similarity function
    @jax.jit
    def jax_similarity_search(target_context, all_contexts):
        target = jnp.array(target_context)
        contexts = jnp.array(all_contexts)
        
        # Vectorized Euclidean distance
        distances = jnp.linalg.norm(contexts - target, axis=1)
        max_distance = jnp.sqrt(len(target_context) * 4.0)
        
        similarities = jnp.maximum(0.0, 1.0 - (distances / max_distance))
        return similarities
    
    # Warmup (important for JIT compilation)
    print("Warming up JIT compilation...")
    _ = jax_similarity_search(target_context, all_contexts[:10])
    
    # Benchmark JAX performance
    print("Benchmarking JAX performance...")
    start_time = time.time()
    for i in range(10):  # 10 iterations
        jax_results = jax_similarity_search(target_context, all_contexts)
        jax_results.block_until_ready()  # Ensure computation completes
    jax_time = (time.time() - start_time) / 10
    
    # Compare with NumPy baseline
    def numpy_similarity_search(target_context, all_contexts):
        target = np.array(target_context)
        contexts = np.array(all_contexts)
        
        distances = np.linalg.norm(contexts - target, axis=1)
        max_distance = np.sqrt(len(target_context) * 4.0)
        
        similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
        return similarities
    
    print("Benchmarking NumPy baseline...")
    start_time = time.time()
    for i in range(10):
        numpy_results = numpy_similarity_search(target_context, all_contexts)
    numpy_time = (time.time() - start_time) / 10
    
    # Results
    print(f"\\nPerformance Results:")
    print(f"NumPy (CPU): {numpy_time:.6f}s per similarity search")
    print(f"JAX (device): {jax_time:.6f}s per similarity search")
    print(f"Speedup: {numpy_time / jax_time:.1f}x")
    
    # Verify correctness
    max_diff = np.max(np.abs(np.array(jax_results) - numpy_results))
    print(f"Max difference: {max_diff:.8f}")
    
    return jax_time, numpy_time

def test_brain_integration():
    """Test how this would integrate with brain operations"""
    print("\\nBrain Integration Test:")
    
    # Simulate brain prediction loop
    n_nodes = 2140
    context_dim = 8
    
    # Generate realistic brain data
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0
    
    @jax.jit
    def brain_similarity_search(target_context, all_contexts, similarity_threshold=0.7):
        target = jnp.array(target_context)
        contexts = jnp.array(all_contexts)
        
        # Calculate similarities
        distances = jnp.linalg.norm(contexts - target, axis=1)
        max_distance = jnp.sqrt(len(target_context) * 4.0)
        similarities = jnp.maximum(0.0, 1.0 - (distances / max_distance))
        
        # Find nodes above threshold
        valid_mask = similarities >= similarity_threshold
        valid_indices = jnp.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity (highest first)
        sorted_indices = jnp.argsort(-valid_similarities)
        
        return valid_indices[sorted_indices], valid_similarities[sorted_indices]
    
    # Test with typical brain traversal
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # Warmup
    _ = brain_similarity_search(target_context, all_contexts[:10])
    
    # Benchmark
    start_time = time.time()
    for i in range(100):  # 100 similarity searches (typical for multiple traversals)
        indices, similarities = brain_similarity_search(target_context, all_contexts)
        indices.block_until_ready()
    total_time = time.time() - start_time
    
    print(f"100 brain similarity searches: {total_time:.4f}s")
    print(f"Average per search: {total_time/100:.6f}s")
    print(f"Theoretical max FPS: {1/(total_time/100):.0f}")
    
    # Show example results
    final_indices, final_similarities = brain_similarity_search(target_context, all_contexts)
    print(f"\\nExample results:")
    print(f"Found {len(final_indices)} similar nodes")
    print(f"Top 5 similarities: {final_similarities[:5]}")

def main():
    print("=== JAX GPU Test for Brain Implementation ===")
    print()
    
    # Test 1: Device availability
    gpu_available = test_jax_devices()
    
    # Test 2: Basic similarity computation
    jax_time, numpy_time = test_similarity_computation()
    
    # Test 3: Brain integration simulation
    test_brain_integration()
    
    print()
    print("=== Summary ===")
    print(f"JAX is {'GPU' if gpu_available else 'CPU'} accelerated")
    print(f"Performance improvement: {numpy_time/jax_time:.1f}x over NumPy")
    print("Ready for brain integration!" if jax_time < 0.001 else "Performance needs optimization")

if __name__ == "__main__":
    main()