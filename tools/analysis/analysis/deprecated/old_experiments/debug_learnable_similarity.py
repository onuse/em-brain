#!/usr/bin/env python3
"""
Debug Learnable Similarity Computation

Focus on the specific issue in learnable similarity calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.similarity.learnable_similarity import LearnableSimilarity
import numpy as np


def debug_learnable_similarity_step_by_step():
    """Debug learnable similarity computation step by step."""
    print("=== Debugging Learnable Similarity Step by Step ===")
    
    # Create learnable similarity
    learnable = LearnableSimilarity(use_gpu=False, learning_rate=0.1)
    
    # Test vectors
    vec_a = np.array([1.0, 2.0, 3.0, 4.0])
    vec_b = np.array([1.1, 2.1, 3.1, 4.1])
    vec_c = np.array([10.0, 20.0, 30.0, 40.0])
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    
    # Initialize parameters
    learnable._initialize_parameters(len(vec_a))
    
    print(f"\nInitialized parameters:")
    print(f"Feature weights: {learnable.feature_weights}")
    print(f"Interaction matrix:\n{learnable.interaction_matrix}")
    
    # Step through computation for A-B
    print(f"\n=== Computing A-B Similarity ===")
    
    # Step 1: Apply feature weighting
    weighted_a = vec_a * learnable.feature_weights
    weighted_b = vec_b * learnable.feature_weights
    
    print(f"Weighted A: {weighted_a}")
    print(f"Weighted B: {weighted_b}")
    
    # Step 2: Apply feature interactions
    interaction_a = np.dot(learnable.interaction_matrix, vec_a)
    interaction_b = np.dot(learnable.interaction_matrix, vec_b)
    
    print(f"Interaction A: {interaction_a}")
    print(f"Interaction B: {interaction_b}")
    
    # Step 3: Combine weighted and interaction terms
    transformed_a = weighted_a + interaction_a
    transformed_b = weighted_b + interaction_b
    
    print(f"Transformed A: {transformed_a}")
    print(f"Transformed B: {transformed_b}")
    
    # Step 4: Compute norms
    norm_a = np.linalg.norm(transformed_a)
    norm_b = np.linalg.norm(transformed_b)
    
    print(f"Norm A: {norm_a}")
    print(f"Norm B: {norm_b}")
    
    # Step 5: Compute cosine similarity
    if norm_a != 0 and norm_b != 0:
        dot_product = np.dot(transformed_a, transformed_b)
        cosine_sim = dot_product / (norm_a * norm_b)
        similarity = (cosine_sim + 1.0) / 2.0
        
        print(f"Dot product: {dot_product}")
        print(f"Cosine similarity: {cosine_sim}")
        print(f"Normalized similarity: {similarity}")
    
    # Compare with actual method
    actual_sim_ab = learnable.compute_similarity(vec_a.tolist(), vec_b.tolist())
    actual_sim_ac = learnable.compute_similarity(vec_a.tolist(), vec_c.tolist())
    
    print(f"\nActual method results:")
    print(f"A-B similarity: {actual_sim_ab}")
    print(f"A-C similarity: {actual_sim_ac}")
    
    # Test with simple vectors to understand the issue
    print(f"\n=== Testing with Simple Vectors ===")
    
    simple_a = np.array([1.0, 0.0, 0.0, 0.0])
    simple_b = np.array([0.0, 1.0, 0.0, 0.0])  # Orthogonal
    
    print(f"Simple A: {simple_a}")
    print(f"Simple B: {simple_b}")
    
    # Manual cosine similarity (should be 0.0)
    manual_cosine = np.dot(simple_a, simple_b) / (np.linalg.norm(simple_a) * np.linalg.norm(simple_b))
    print(f"Manual cosine similarity: {manual_cosine}")
    
    # Learnable similarity
    learnable_sim = learnable.compute_similarity(simple_a.tolist(), simple_b.tolist())
    print(f"Learnable similarity: {learnable_sim}")
    
    # Check if interaction matrix is the issue
    print(f"\n=== Analyzing Interaction Matrix Impact ===")
    
    # Create a simpler test with identity interaction matrix
    learnable_simple = LearnableSimilarity(use_gpu=False, learning_rate=0.1)
    learnable_simple._initialize_parameters(4)
    
    # Set interaction matrix to zero (no interactions)
    learnable_simple.interaction_matrix = np.zeros((4, 4))
    learnable_simple.feature_weights = np.ones(4)  # Unit weights
    
    print(f"Zero interaction matrix:")
    print(f"Feature weights: {learnable_simple.feature_weights}")
    print(f"Interaction matrix:\n{learnable_simple.interaction_matrix}")
    
    # Test with zero interactions
    zero_sim_ab = learnable_simple.compute_similarity(vec_a.tolist(), vec_b.tolist())
    zero_sim_ac = learnable_simple.compute_similarity(vec_a.tolist(), vec_c.tolist())
    
    print(f"\nWith zero interactions:")
    print(f"A-B similarity: {zero_sim_ab}")
    print(f"A-C similarity: {zero_sim_ac}")


def main():
    """Run debugging."""
    debug_learnable_similarity_step_by_step()


if __name__ == "__main__":
    main()