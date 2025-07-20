#!/usr/bin/env python3
"""
Simple Similarity Search Test

A focused test to identify issues with similarity search in the brain.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.brain_factory import MinimalBrain
from server.src.similarity.engine import SimilarityEngine
import numpy as np


def test_similarity_search_basics():
    """Test basic similarity search functionality and identify issues."""
    print("=== Similarity Search Issues Investigation ===\n")
    
    # Test 1: Basic similarity computation
    print("1. Testing basic similarity computation...")
    engine = SimilarityEngine(use_gpu=False, use_learnable_similarity=False)
    
    # Create test vectors
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]  # Very similar
    vec_c = [10.0, 20.0, 30.0, 40.0]  # Very different
    
    sim_ab = engine.compute_similarity(vec_a, vec_b)
    sim_ac = engine.compute_similarity(vec_a, vec_c)
    
    print(f"   Vector A: {vec_a}")
    print(f"   Vector B: {vec_b}")
    print(f"   Vector C: {vec_c}")
    print(f"   Similarity A-B: {sim_ab:.3f}")
    print(f"   Similarity A-C: {sim_ac:.3f}")
    
    # ISSUE 1: Check if similarities are all 1.0 (broken)
    if sim_ab == 1.0 and sim_ac == 1.0:
        print("   ❌ ISSUE: All similarities are 1.0 - similarity computation is broken!")
        return False
    elif sim_ab > sim_ac:
        print("   ✅ Similarity computation working correctly")
    else:
        print("   ❌ ISSUE: Similar vectors have lower similarity than dissimilar ones")
        return False
    
    # Test 2: Similarity search with multiple vectors
    print("\n2. Testing similarity search with multiple vectors...")
    
    vectors = [vec_a, vec_b, vec_c]
    ids = ['a', 'b', 'c']
    target = [1.05, 2.05, 3.05, 4.05]  # Should be most similar to A
    
    results = engine.find_similar_experiences(target, vectors, ids, max_results=3, min_similarity=0.0)
    
    print(f"   Target: {target}")
    print(f"   Results: {results}")
    
    if not results:
        print("   ❌ ISSUE: No similar experiences found")
        return False
    
    # Check if results are ordered correctly
    if len(results) >= 2:
        if results[0][1] < results[1][1]:
            print("   ❌ ISSUE: Results not ordered by similarity (descending)")
            return False
        else:
            print("   ✅ Results ordered correctly by similarity")
    
    # Test 3: Test with brain integration
    print("\n3. Testing brain integration...")
    
    brain = MinimalBrain()
    
    # Add some experiences
    brain.store_experience([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [1.1, 2.1, 3.1, 4.1])
    brain.store_experience([1.1, 2.1, 3.1, 4.1], [0.1, 0.2, 0.3, 0.4], [1.2, 2.2, 3.2, 4.2])
    brain.store_experience([5.0, 6.0, 7.0, 8.0], [0.5, 0.6, 0.7, 0.8], [5.1, 6.1, 7.1, 8.1])
    
    # Test prediction (uses similarity search internally)
    test_input = [1.02, 2.02, 3.02, 4.02]
    predicted_action, brain_state = brain.process_sensory_input(test_input)
    
    prediction_details = brain_state.get('prediction_details', {})
    method = prediction_details.get('method', 'unknown')
    num_similar = prediction_details.get('num_similar', 0)
    
    print(f"   Test input: {test_input}")
    print(f"   Prediction method: {method}")
    print(f"   Number of similar experiences found: {num_similar}")
    
    # Check if prediction is using similarity search effectively
    if method == 'bootstrap_random':
        print("   ❌ ISSUE: Prediction falling back to random - similarity search not finding experiences")
        return False
    elif method == 'consensus' and num_similar > 0:
        print("   ✅ Prediction using consensus from similar experiences")
    else:
        print(f"   ⚠️  Prediction method: {method} with {num_similar} similar experiences")
    
    # Test 4: Check similarity thresholds
    print("\n4. Testing similarity thresholds...")
    
    # Get all experiences and compute similarity distribution
    all_experiences = brain.experience_storage.get_all_experiences()
    if len(all_experiences) < 2:
        print("   ⚠️  Not enough experiences to test thresholds")
        return True
    
    # Test the actual similarity search used by prediction engine
    exp_vectors = []
    exp_ids = []
    for exp_id, exp in all_experiences.items():
        exp_vectors.append(exp.get_context_vector())
        exp_ids.append(exp_id)
    
    # Test with different similarity thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        similar_results = brain.similarity_engine.find_similar_experiences(
            test_input, exp_vectors, exp_ids, max_results=10, min_similarity=threshold
        )
        print(f"   Threshold {threshold}: {len(similar_results)} experiences found")
    
    # Test 5: Check if learnable similarity is adapting
    print("\n5. Testing learnable similarity adaptation...")
    
    sim_stats = brain.similarity_engine.get_performance_stats()
    if 'similarity_learning' in sim_stats:
        learning_stats = sim_stats['similarity_learning']
        adaptations = learning_stats.get('adaptations_performed', 0)
        correlation = learning_stats.get('similarity_success_correlation', 0.0)
        
        print(f"   Adaptations performed: {adaptations}")
        print(f"   Similarity-success correlation: {correlation:.3f}")
        
        if adaptations == 0:
            print("   ⚠️  Learnable similarity not adapting yet (needs more data)")
        else:
            print("   ✅ Learnable similarity is adapting")
    else:
        print("   ⚠️  Learnable similarity not available")
    
    print("\n=== Summary ===")
    print("✅ Similarity search investigation completed")
    return True


if __name__ == "__main__":
    success = test_similarity_search_basics()
    if success:
        print("\n🎉 No critical issues found with similarity search")
    else:
        print("\n❌ Critical issues found - similarity search needs fixing")