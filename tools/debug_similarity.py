#!/usr/bin/env python3
"""
Debug Similarity Computation

Focused debugging to find why similarity computation returns 1.0 for all vectors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.similarity.engine import SimilarityEngine
from server.src.similarity.learnable_similarity import LearnableSimilarity
import numpy as np


def debug_hardcoded_similarity():
    """Debug the hardcoded cosine similarity computation."""
    print("=== Debugging Hardcoded Similarity ===")
    
    # Create engine with hardcoded similarity (no learnable)
    engine = SimilarityEngine(use_gpu=False, use_learnable_similarity=False)
    
    # Test vectors
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]
    vec_c = [10.0, 20.0, 30.0, 40.0]
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    
    # Manual cosine similarity calculation
    target = np.array(vec_a)
    experiences = np.array([vec_b, vec_c])
    
    target_norm = np.linalg.norm(target)
    experience_norms = np.linalg.norm(experiences, axis=1)
    
    print(f"\nManual calculation:")
    print(f"Target norm: {target_norm}")
    print(f"Experience norms: {experience_norms}")
    
    if target_norm != 0 and not np.any(experience_norms == 0):
        dot_products = np.dot(experiences, target)
        similarities = dot_products / (target_norm * experience_norms)
        similarities = (similarities + 1.0) / 2.0
        
        print(f"Dot products: {dot_products}")
        print(f"Raw cosine similarities: {similarities}")
    
    # Test engine computation
    sim_ab = engine.compute_similarity(vec_a, vec_b)
    sim_ac = engine.compute_similarity(vec_a, vec_c)
    
    print(f"\nEngine computation:")
    print(f"A-B similarity: {sim_ab}")
    print(f"A-C similarity: {sim_ac}")
    
    return sim_ab, sim_ac


def debug_learnable_similarity():
    """Debug the learnable similarity computation."""
    print("\n=== Debugging Learnable Similarity ===")
    
    # Create learnable similarity directly
    learnable = LearnableSimilarity(use_gpu=False, learning_rate=0.1)
    
    # Test vectors
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]
    vec_c = [10.0, 20.0, 30.0, 40.0]
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    
    # Test similarities
    sim_ab = learnable.compute_similarity(vec_a, vec_b)
    sim_ac = learnable.compute_similarity(vec_a, vec_c)
    
    print(f"\nLearnable similarity computation:")
    print(f"A-B similarity: {sim_ab}")
    print(f"A-C similarity: {sim_ac}")
    
    # Check internal state
    print(f"\nInternal state:")
    print(f"Feature weights: {learnable.feature_weights}")
    print(f"Interaction matrix shape: {learnable.interaction_matrix.shape if learnable.interaction_matrix is not None else 'None'}")
    
    return sim_ab, sim_ac


def debug_cpu_computation():
    """Debug the CPU computation path step by step."""
    print("\n=== Debugging CPU Computation Step by Step ===")
    
    # Create engine with hardcoded similarity
    engine = SimilarityEngine(use_gpu=False, use_learnable_similarity=False)
    
    # Test with the _cpu_compute_similarities method directly
    target_vector = [1.0, 2.0, 3.0, 4.0]
    experience_vectors = [
        [1.1, 2.1, 3.1, 4.1],  # Similar
        [10.0, 20.0, 30.0, 40.0]  # Different
    ]
    
    print(f"Target: {target_vector}")
    print(f"Experiences: {experience_vectors}")
    
    # Call the method directly
    similarities = engine._cpu_compute_similarities(target_vector, experience_vectors)
    
    print(f"Computed similarities: {similarities}")
    
    # Step through the computation manually
    target = np.array(target_vector)
    experiences = np.array(experience_vectors)
    
    print(f"\nStep-by-step computation:")
    print(f"Target array: {target}")
    print(f"Experiences array: {experiences}")
    
    target_norm = np.linalg.norm(target)
    experience_norms = np.linalg.norm(experiences, axis=1)
    
    print(f"Target norm: {target_norm}")
    print(f"Experience norms: {experience_norms}")
    
    if target_norm != 0 and not np.any(experience_norms == 0):
        dot_products = np.dot(experiences, target)
        raw_similarities = dot_products / (target_norm * experience_norms)
        normalized_similarities = (raw_similarities + 1.0) / 2.0
        
        print(f"Dot products: {dot_products}")
        print(f"Raw cosine similarities: {raw_similarities}")
        print(f"Normalized similarities: {normalized_similarities}")
    
    return similarities


def debug_engine_with_learnable():
    """Debug engine with learnable similarity enabled."""
    print("\n=== Debugging Engine with Learnable Similarity ===")
    
    # Create engine with learnable similarity
    engine = SimilarityEngine(use_gpu=False, use_learnable_similarity=True)
    
    # Test vectors
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]
    vec_c = [10.0, 20.0, 30.0, 40.0]
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    
    # Test similarities
    sim_ab = engine.compute_similarity(vec_a, vec_b)
    sim_ac = engine.compute_similarity(vec_a, vec_c)
    
    print(f"\nEngine with learnable similarity:")
    print(f"A-B similarity: {sim_ab}")
    print(f"A-C similarity: {sim_ac}")
    
    # Check learnable similarity state
    if engine.learnable_similarity:
        print(f"\nLearnable similarity state:")
        print(f"Feature weights: {engine.learnable_similarity.feature_weights}")
        print(f"Vector dimensions: {engine.learnable_similarity.vector_dimensions}")
    
    return sim_ab, sim_ac


def main():
    """Run all debugging tests."""
    print("Debugging Similarity Computation Issues")
    print("=" * 50)
    
    # Test 1: Hardcoded similarity
    hardcoded_ab, hardcoded_ac = debug_hardcoded_similarity()
    
    # Test 2: Learnable similarity
    learnable_ab, learnable_ac = debug_learnable_similarity()
    
    # Test 3: CPU computation step by step
    cpu_similarities = debug_cpu_computation()
    
    # Test 4: Engine with learnable similarity
    engine_ab, engine_ac = debug_engine_with_learnable()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print(f"Hardcoded similarity: A-B={hardcoded_ab:.3f}, A-C={hardcoded_ac:.3f}")
    print(f"Learnable similarity: A-B={learnable_ab:.3f}, A-C={learnable_ac:.3f}")
    print(f"CPU computation: {cpu_similarities}")
    print(f"Engine with learnable: A-B={engine_ab:.3f}, A-C={engine_ac:.3f}")
    
    # Analyze results
    if hardcoded_ab == 1.0 and hardcoded_ac == 1.0:
        print("\n❌ ISSUE: Hardcoded similarity computation is broken")
    elif learnable_ab == 1.0 and learnable_ac == 1.0:
        print("\n❌ ISSUE: Learnable similarity computation is broken")
    else:
        print("\n✅ Similarity computation appears to be working")


if __name__ == "__main__":
    main()