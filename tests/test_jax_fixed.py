#!/usr/bin/env python3
"""
Fixed JAX test with proper GPU detection and JIT-compatible code
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

def test_jax_devices():
    """Test available JAX devices and configuration"""
    print("JAX Configuration:")
    print(f"JAX version: {jax.__version__}")
    print(f"JAXlib version: {jax.lib.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Check for GPU/Metal support
    gpu_available = any(device.device_kind in ['gpu', 'metal'] for device in jax.devices())
    
    print()
    print("Device Analysis:")
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device.device_kind} - {device}")
    
    print(f"\\nGPU/Metal available: {gpu_available}")
    
    # Try to detect Metal backend specifically
    try:
        # Force Metal backend if available
        import os
        os.environ.setdefault('JAX_PLATFORM_NAME', 'metal')
        print(f"Platform name setting: {os.environ.get('JAX_PLATFORM_NAME', 'not set')}")
    except:
        pass
    
    return gpu_available

def test_basic_operations():
    """Test basic JAX operations and performance"""
    print("\\nTesting basic JAX operations...")
    
    # Simple test data
    n = 1000
    a = jnp.ones((n, n))
    b = jnp.ones((n, n))
    
    # Test matrix multiplication
    @jax.jit
    def matmul_test(a, b):
        return jnp.dot(a, b)
    
    # Warmup
    _ = matmul_test(a[:10, :10], b[:10, :10])
    
    # Benchmark
    start_time = time.time()
    result = matmul_test(a, b)
    result.block_until_ready()
    jax_time = time.time() - start_time
    
    # NumPy comparison
    a_np = np.ones((n, n))
    b_np = np.ones((n, n))
    
    start_time = time.time()
    result_np = np.dot(a_np, b_np)
    numpy_time = time.time() - start_time
    
    print(f"Matrix multiplication ({n}x{n}):")
    print(f"  JAX: {jax_time:.4f}s")
    print(f"  NumPy: {numpy_time:.4f}s")
    print(f"  Speedup: {numpy_time/jax_time:.1f}x")

def test_similarity_fixed():
    """Test similarity computation with JAX-compatible code"""
    print("\\nTesting fixed similarity computation...")
    
    # Brain-realistic data
    n_nodes = 2140
    context_dim = 8
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0
    
    # JAX similarity function (JIT-compatible)
    @jax.jit
    def jax_similarity_search(target_context, all_contexts):
        target = jnp.array(target_context)
        contexts = jnp.array(all_contexts)
        
        # Vectorized Euclidean distance
        distances = jnp.linalg.norm(contexts - target, axis=1)
        max_distance = jnp.sqrt(len(target_context) * 4.0)
        
        similarities = jnp.maximum(0.0, 1.0 - (distances / max_distance))
        return similarities
    
    # JAX similarity with filtering (fixed for JIT)
    @jax.jit
    def jax_similarity_with_threshold(target_context, all_contexts, similarity_threshold=0.7, max_results=10):
        similarities = jax_similarity_search(target_context, all_contexts)
        
        # Sort all similarities in descending order
        sorted_indices = jnp.argsort(-similarities)
        sorted_similarities = similarities[sorted_indices]
        
        # Take top results that meet threshold
        # Use fixed size to avoid dynamic shapes
        top_indices = sorted_indices[:max_results]
        top_similarities = sorted_similarities[:max_results]
        
        # Create mask for threshold filtering
        threshold_mask = top_similarities >= similarity_threshold
        
        return top_indices, top_similarities, threshold_mask
    
    # Warmup
    print("Warming up JIT compilation...")
    _ = jax_similarity_search(target_context, all_contexts[:10])
    _ = jax_similarity_with_threshold(target_context, all_contexts[:100])
    
    # Benchmark basic similarity
    print("Benchmarking similarity search...")
    start_time = time.time()
    for i in range(10):
        similarities = jax_similarity_search(target_context, all_contexts)
        similarities.block_until_ready()
    jax_time = (time.time() - start_time) / 10
    
    # NumPy baseline
    def numpy_similarity_search(target_context, all_contexts):
        target = np.array(target_context)
        contexts = np.array(all_contexts)
        
        distances = np.linalg.norm(contexts - target, axis=1)
        max_distance = np.sqrt(len(target_context) * 4.0)
        
        similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
        return similarities
    
    start_time = time.time()
    for i in range(10):
        numpy_results = numpy_similarity_search(target_context, all_contexts)
    numpy_time = (time.time() - start_time) / 10
    
    print(f"\\nSimilarity Search Results:")
    print(f"  JAX: {jax_time:.6f}s per search")
    print(f"  NumPy: {numpy_time:.6f}s per search")
    print(f"  Speedup: {numpy_time/jax_time:.1f}x")
    
    # Test with threshold filtering
    print("\\nTesting threshold filtering...")
    start_time = time.time()
    for i in range(10):
        indices, similarities, mask = jax_similarity_with_threshold(target_context, all_contexts)
        indices.block_until_ready()
    filtered_time = (time.time() - start_time) / 10
    
    print(f"  JAX with filtering: {filtered_time:.6f}s per search")
    
    # Show example results
    final_indices, final_similarities, final_mask = jax_similarity_with_threshold(target_context, all_contexts)
    valid_count = jnp.sum(final_mask)
    
    print(f"\\nExample Results:")
    print(f"  Top similarities: {final_similarities[:5]}")
    print(f"  Valid results (≥0.7): {valid_count}")
    
    return jax_time, numpy_time

def test_brain_performance():
    """Test realistic brain performance scenarios"""
    print("\\nTesting realistic brain performance...")
    
    # Realistic brain parameters
    n_nodes = 2140
    context_dim = 8
    n_traversals = 5  # Typical number of traversals per prediction
    
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0
    
    @jax.jit
    def brain_prediction_cycle(target_context, all_contexts):
        """Simulate a complete brain prediction cycle"""
        # Find similar nodes for traversal start
        similarities = jax_similarity_search(target_context, all_contexts)
        
        # Sort and get top candidates
        sorted_indices = jnp.argsort(-similarities)
        top_similarities = similarities[sorted_indices[:10]]
        
        return sorted_indices[:10], top_similarities
    
    # Warmup
    _ = brain_prediction_cycle([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], all_contexts[:100])
    
    # Benchmark full prediction cycle
    print("Benchmarking complete prediction cycles...")
    start_time = time.time()
    
    for cycle in range(10):  # 10 prediction cycles
        for traversal in range(n_traversals):  # 5 traversals per cycle
            context = [cycle % 10, traversal % 10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            indices, similarities = brain_prediction_cycle(context, all_contexts)
            indices.block_until_ready()
    
    total_time = time.time() - start_time
    cycles_per_second = 10 / total_time
    
    print(f"Performance Summary:")
    print(f"  10 cycles x 5 traversals: {total_time:.4f}s")
    print(f"  Cycles per second: {cycles_per_second:.1f}")
    print(f"  Theoretical demo FPS: {cycles_per_second:.1f}")
    print(f"  Time per similarity search: {total_time/50:.6f}s")

def main():
    print("=== Fixed JAX GPU Test for Brain Implementation ===")
    print()
    
    # Test 1: Device detection
    gpu_available = test_jax_devices()
    
    # Test 2: Basic operations
    test_basic_operations()
    
    # Test 3: Fixed similarity computation
    jax_time, numpy_time = test_similarity_fixed()
    
    # Test 4: Brain performance simulation
    test_brain_performance()
    
    print()
    print("=== Summary ===")
    print(f"Platform: {'GPU/Metal' if gpu_available else 'CPU'}")
    print(f"JAX Performance: {numpy_time/jax_time:.1f}x vs NumPy")
    
    if jax_time < 0.001:
        print("✅ Excellent performance - ready for brain integration!")
    elif jax_time < 0.01:
        print("✅ Good performance - should work well for brain")
    else:
        print("⚠️  Performance needs optimization")

if __name__ == "__main__":
    main()