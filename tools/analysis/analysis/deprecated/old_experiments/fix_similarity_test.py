#!/usr/bin/env python3
"""
Test similarity search after understanding the root cause.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.similarity.engine import SimilarityEngine
import numpy as np


def test_similarity_engine_issue():
    """Test the specific issue with similarity engine."""
    print("=== Testing Similarity Engine Issue ===")
    
    # Create engine with hardcoded similarity (should work)
    print("1. Testing with hardcoded similarity (use_learnable_similarity=False):")
    engine_hardcoded = SimilarityEngine(use_gpu=False, use_learnable_similarity=False)
    
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]
    vec_c = [10.0, 20.0, 30.0, 40.0]
    
    sim_ab_hardcoded = engine_hardcoded.compute_similarity(vec_a, vec_b)
    sim_ac_hardcoded = engine_hardcoded.compute_similarity(vec_a, vec_c)
    
    print(f"   A-B similarity: {sim_ab_hardcoded:.6f}")
    print(f"   A-C similarity: {sim_ac_hardcoded:.6f}")
    print(f"   Difference: {abs(sim_ab_hardcoded - sim_ac_hardcoded):.6f}")
    
    # Create engine with learnable similarity (problematic)
    print("\n2. Testing with learnable similarity (use_learnable_similarity=True):")
    engine_learnable = SimilarityEngine(use_gpu=False, use_learnable_similarity=True)
    
    sim_ab_learnable = engine_learnable.compute_similarity(vec_a, vec_b)
    sim_ac_learnable = engine_learnable.compute_similarity(vec_a, vec_c)
    
    print(f"   A-B similarity: {sim_ab_learnable:.6f}")
    print(f"   A-C similarity: {sim_ac_learnable:.6f}")
    print(f"   Difference: {abs(sim_ab_learnable - sim_ac_learnable):.6f}")
    
    # Test find_similar_experiences with both engines
    print("\n3. Testing find_similar_experiences with hardcoded similarity:")
    vectors = [vec_b, vec_c]
    ids = ['b', 'c']
    
    results_hardcoded = engine_hardcoded.find_similar_experiences(
        vec_a, vectors, ids, max_results=2, min_similarity=0.0
    )
    print(f"   Results: {results_hardcoded}")
    
    print("\n4. Testing find_similar_experiences with learnable similarity:")
    results_learnable = engine_learnable.find_similar_experiences(
        vec_a, vectors, ids, max_results=2, min_similarity=0.0
    )
    print(f"   Results: {results_learnable}")
    
    # The issue: learnable similarity makes everything look similar
    print(f"\n=== Analysis ===")
    if abs(sim_ab_learnable - sim_ac_learnable) < 0.001:
        print("❌ CONFIRMED: Learnable similarity makes all vectors appear equally similar")
        print("   This breaks the prediction engine's ability to find relevant experiences")
    else:
        print("✅ Learnable similarity is working correctly")
    
    return {
        'hardcoded_ab': sim_ab_hardcoded,
        'hardcoded_ac': sim_ac_hardcoded,
        'learnable_ab': sim_ab_learnable,
        'learnable_ac': sim_ac_learnable
    }


def test_with_brain():
    """Test how this affects the brain's prediction capability."""
    print("\n=== Testing Brain Impact ===")
    
    from server.src.brain_factory import MinimalBrain
    
    # Create brain (uses learnable similarity by default)
    brain = MinimalBrain()
    
    # Add diverse experiences
    brain.store_experience([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [1.1, 2.1, 3.1, 4.1])
    brain.store_experience([1.1, 2.1, 3.1, 4.1], [0.1, 0.2, 0.3, 0.4], [1.2, 2.2, 3.2, 4.2])
    brain.store_experience([10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0], [11.0, 21.0, 31.0, 41.0])
    
    # Test prediction
    test_input = [1.05, 2.05, 3.05, 4.05]  # Should be most similar to first two experiences
    predicted_action, brain_state = brain.process_sensory_input(test_input)
    
    prediction_details = brain_state.get('prediction_details', {})
    method = prediction_details.get('method', 'unknown')
    num_similar = prediction_details.get('num_similar', 0)
    confidence = brain_state.get('prediction_confidence', 0.0)
    
    print(f"Test input: {test_input}")
    print(f"Prediction method: {method}")
    print(f"Number of similar experiences: {num_similar}")
    print(f"Confidence: {confidence:.3f}")
    
    if method == 'bootstrap_random' or num_similar == 0:
        print("❌ CONFIRMED: Brain cannot find similar experiences due to broken similarity search")
    elif method == 'consensus' and num_similar > 0:
        print("✅ Brain is finding similar experiences and using them for prediction")
    
    # Test direct similarity search
    all_experiences = brain.experience_storage.get_all_experiences()
    exp_vectors = []
    exp_ids = []
    for exp_id, exp in all_experiences.items():
        exp_vectors.append(exp.get_context_vector())
        exp_ids.append(exp_id)
    
    similar_results = brain.similarity_engine.find_similar_experiences(
        test_input, exp_vectors, exp_ids, max_results=3, min_similarity=0.4
    )
    
    print(f"Direct similarity search results (threshold 0.4): {len(similar_results)} experiences")
    for exp_id, similarity in similar_results:
        exp = all_experiences[exp_id]
        print(f"   {exp_id[:8]}...: similarity={similarity:.3f}, context={exp.get_context_vector()}")
    
    return brain_state


def main():
    """Run the investigation."""
    print("Similarity Search Root Cause Analysis")
    print("=" * 50)
    
    results = test_similarity_engine_issue()
    brain_state = test_with_brain()
    
    print("\n" + "=" * 50)
    print("FINDINGS:")
    print("=" * 50)
    
    hardcoded_diff = abs(results['hardcoded_ab'] - results['hardcoded_ac'])
    learnable_diff = abs(results['learnable_ab'] - results['learnable_ac'])
    
    print(f"1. Hardcoded similarity discriminates well (difference: {hardcoded_diff:.6f})")
    print(f"2. Learnable similarity fails to discriminate (difference: {learnable_diff:.6f})")
    
    if learnable_diff < 0.001:
        print("\n❌ ROOT CAUSE IDENTIFIED:")
        print("   The learnable similarity initialization makes all vectors appear equally similar.")
        print("   This prevents the brain from finding relevant past experiences for prediction.")
        print("\n💡 SOLUTION:")
        print("   - Fix the interaction matrix initialization in LearnableSimilarity")
        print("   - Or temporarily disable learnable similarity until it's fixed")
        print("   - The interaction matrix should start at zero, not identity * 0.1")
    
    return results


if __name__ == "__main__":
    main()