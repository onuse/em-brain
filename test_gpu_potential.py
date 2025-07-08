#!/usr/bin/env python3
"""
Test GPU vectorization potential for brain similarity calculations
"""

import time
import numpy as np

def current_similarity_cpu(context1, context2):
    """Current implementation from world_graph.py"""
    if len(context1) != len(context2):
        return 0.0
    
    # Euclidean distance
    distance = sum((a - b) ** 2 for a, b in zip(context1, context2)) ** 0.5
    max_possible_distance = (len(context1) * 4.0) ** 0.5
    
    if max_possible_distance == 0:
        return 1.0 if distance == 0 else 0.0
    
    similarity = max(0.0, 1.0 - (distance / max_possible_distance))
    return similarity

def vectorized_similarity_cpu(target_context, all_contexts):
    """Vectorized version using NumPy - this is what GPU would do"""
    target = np.array(target_context)
    contexts = np.array(all_contexts)
    
    # Vectorized Euclidean distance
    distances = np.linalg.norm(contexts - target, axis=1)
    max_distance = np.sqrt(len(target_context) * 4.0)
    
    similarities = np.maximum(0.0, 1.0 - (distances / max_distance))
    return similarities

def main():
    # Test with realistic data
    n_nodes = 2140  # Current brain size
    context_dim = 8  # Context vector dimension
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # Generate test data
    np.random.seed(42)
    all_contexts = np.random.randn(n_nodes, context_dim) * 2.0

    print(f"Testing with {n_nodes} nodes, {context_dim}D context vectors")
    print()

    # Test current sequential approach
    start_time = time.time()
    sequential_results = []
    for i in range(n_nodes):
        similarity = current_similarity_cpu(target_context, all_contexts[i].tolist())
        sequential_results.append(similarity)
    sequential_time = time.time() - start_time

    # Test vectorized approach (simulates GPU)
    start_time = time.time()
    vectorized_results = vectorized_similarity_cpu(target_context, all_contexts)
    vectorized_time = time.time() - start_time

    print(f"Sequential (current): {sequential_time:.3f}s")
    print(f"Vectorized (GPU-like): {vectorized_time:.3f}s")
    print(f"Speedup: {sequential_time / vectorized_time:.1f}x")
    print()

    # Verify correctness
    max_diff = max(abs(a - b) for a, b in zip(sequential_results, vectorized_results))
    print(f"Max difference between methods: {max_diff:.6f}")
    
    # Show what this means for the demo
    print()
    print("Impact on demo_ultimate_2d_brain.py:")
    print(f"- Current similarity search: {sequential_time:.3f}s per traversal")
    print(f"- GPU similarity search: {vectorized_time:.3f}s per traversal")
    print(f"- With 3-10 traversals per prediction: {3*sequential_time:.3f}s → {3*vectorized_time:.3f}s")
    print(f"- Potential FPS improvement: {1/(3*sequential_time):.1f} → {1/(3*vectorized_time):.1f}")

if __name__ == "__main__":
    main()