#!/usr/bin/env python3
"""
Similarity Search Investigation Tool

Investigates how similarity search is implemented and identifies potential issues
that might prevent the brain from finding relevant past experiences for prediction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.brain import MinimalBrain
from server.src.similarity.engine import SimilarityEngine
from server.src.similarity.learnable_similarity import LearnableSimilarity
from server.src.experience.models import Experience
import numpy as np
import time


def test_basic_similarity_search():
    """Test basic similarity search functionality."""
    print("=== Basic Similarity Search Test ===")
    
    # Create similarity engine
    engine = SimilarityEngine(use_gpu=False, use_learnable_similarity=False)
    
    # Create test experiences
    experiences = [
        Experience(
            sensory_input=[1.0, 2.0, 3.0, 4.0],
            action_taken=[0.1, 0.2, 0.3, 0.4],
            outcome=[1.1, 2.1, 3.1, 4.1],
            prediction_error=0.2,
            timestamp=time.time()
        ),
        Experience(
            sensory_input=[1.1, 2.1, 3.1, 4.1],  # Similar to first
            action_taken=[0.1, 0.2, 0.3, 0.4],
            outcome=[1.2, 2.2, 3.2, 4.2],
            prediction_error=0.3,
            timestamp=time.time()
        ),
        Experience(
            sensory_input=[10.0, 20.0, 30.0, 40.0],  # Very different
            action_taken=[1.0, 2.0, 3.0, 4.0],
            outcome=[11.0, 21.0, 31.0, 41.0],
            prediction_error=0.1,
            timestamp=time.time()
        )
    ]
    
    # Extract vectors and IDs
    vectors = []
    ids = []
    for i, exp in enumerate(experiences):
        vectors.append(exp.get_context_vector())
        ids.append(f"exp_{i}")
    
    # Test similarity search
    target_vector = [1.05, 2.05, 3.05, 4.05]  # Should be most similar to first experience
    
    print(f"Target vector: {target_vector}")
    print(f"Experience 0 context: {vectors[0]}")
    print(f"Experience 1 context: {vectors[1]}")
    print(f"Experience 2 context: {vectors[2]}")
    
    # Find similar experiences
    similar_experiences = engine.find_similar_experiences(
        target_vector, vectors, ids, max_results=3, min_similarity=0.0
    )
    
    print(f"\nSimilar experiences found: {len(similar_experiences)}")
    for exp_id, similarity in similar_experiences:
        print(f"  {exp_id}: similarity = {similarity:.3f}")
    
    # Test individual similarity computation
    print(f"\nIndividual similarity tests:")
    for i, vector in enumerate(vectors):
        sim = engine.compute_similarity(target_vector, vector)
        print(f"  Target vs Experience {i}: {sim:.3f}")
    
    return similar_experiences


def test_learnable_similarity():
    """Test learnable similarity functionality."""
    print("\n=== Learnable Similarity Test ===")
    
    # Create learnable similarity engine
    learnable_sim = LearnableSimilarity(use_gpu=False, learning_rate=0.1)
    
    # Create test vectors
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [1.1, 2.1, 3.1, 4.1]  # Similar to A
    vec_c = [10.0, 20.0, 30.0, 40.0]  # Very different
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    
    # Test initial similarities
    print(f"\nInitial similarities:")
    sim_ab = learnable_sim.compute_similarity(vec_a, vec_b)
    sim_ac = learnable_sim.compute_similarity(vec_a, vec_c)
    sim_bc = learnable_sim.compute_similarity(vec_b, vec_c)
    
    print(f"  A-B: {sim_ab:.3f}")
    print(f"  A-C: {sim_ac:.3f}")
    print(f"  B-C: {sim_bc:.3f}")
    
    # Simulate learning scenarios
    print(f"\nSimulating learning scenarios...")
    
    # Scenario 1: A-B similarity led to good prediction
    learnable_sim.record_prediction_outcome(vec_a, vec_b, 0.8)
    print(f"  Recorded: A-B similarity led to good prediction (0.8)")
    
    # Scenario 2: A-C similarity led to bad prediction
    learnable_sim.record_prediction_outcome(vec_a, vec_c, 0.2)
    print(f"  Recorded: A-C similarity led to bad prediction (0.2)")
    
    # Test if similarity function adapts
    if len(learnable_sim.prediction_outcomes) >= 2:
        print(f"  Adapting similarity function...")
        learnable_sim.adapt_similarity_function()
        
        # Test updated similarities
        print(f"\nUpdated similarities:")
        sim_ab_new = learnable_sim.compute_similarity(vec_a, vec_b)
        sim_ac_new = learnable_sim.compute_similarity(vec_a, vec_c)
        
        print(f"  A-B: {sim_ab:.3f} -> {sim_ab_new:.3f} (change: {sim_ab_new-sim_ab:+.3f})")
        print(f"  A-C: {sim_ac:.3f} -> {sim_ac_new:.3f} (change: {sim_ac_new-sim_ac:+.3f})")
    
    return learnable_sim


def test_brain_similarity_integration():
    """Test how similarity search integrates with the brain."""
    print("\n=== Brain Similarity Integration Test ===")
    
    # Create minimal brain
    brain = MinimalBrain()
    
    # Add some experiences
    experiences_data = [
        ([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]),
        ([1.1, 2.1, 3.1, 4.1], [0.1, 0.2, 0.3, 0.4]),  # Similar context
        ([5.0, 6.0, 7.0, 8.0], [0.5, 0.6, 0.7, 0.8]),  # Different context
        ([1.05, 2.05, 3.05, 4.05], [0.1, 0.2, 0.3, 0.4])  # Very similar to first
    ]
    
    print(f"Adding {len(experiences_data)} experiences to brain...")
    for i, (sensory, action) in enumerate(experiences_data):
        outcome = [s + 0.1 for s in sensory]  # Simple outcome
        brain.store_experience(sensory, action, outcome)
        print(f"  Experience {i}: sensors={sensory}, action={action}")
    
    # Test prediction (which uses similarity search)
    test_context = [1.02, 2.02, 3.02, 4.02]  # Should be similar to first experiences
    print(f"\nTesting prediction for context: {test_context}")
    
    predicted_action, brain_state = brain.process_sensory_input(test_context)
    confidence = brain_state.get('prediction_confidence', 0.0)
    details = brain_state.get('prediction_details', {})
    print(f"Predicted action: {predicted_action}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Prediction details: {details}")
    
    # Check similarity engine statistics
    print(f"\nSimilarity engine statistics:")
    sim_stats = brain.similarity_engine.get_performance_stats()
    for key, value in sim_stats.items():
        print(f"  {key}: {value}")
    
    return brain


def test_experience_storage_and_retrieval():
    """Test how experiences are stored and retrieved."""
    print("\n=== Experience Storage and Retrieval Test ===")
    
    brain = MinimalBrain()
    
    # Add experiences with known patterns
    pattern_a_experiences = []
    pattern_b_experiences = []
    
    # Pattern A: gradual increase
    for i in range(5):
        base = i * 0.1
        sensory = [base + 1.0, base + 2.0, base + 3.0, base + 4.0]
        action = [0.1, 0.2, 0.3, 0.4]
        outcome = [s + 0.1 for s in sensory]
        
        brain.store_experience(sensory, action, outcome)
        pattern_a_experiences.append((sensory, action, outcome))
    
    # Pattern B: different structure
    for i in range(5):
        base = i * 0.1
        sensory = [5.0 - base, 4.0 - base, 3.0 - base, 2.0 - base]
        action = [0.5, 0.6, 0.7, 0.8]
        outcome = [s + 0.1 for s in sensory]
        
        brain.store_experience(sensory, action, outcome)
        pattern_b_experiences.append((sensory, action, outcome))
    
    print(f"Added {len(pattern_a_experiences)} Pattern A experiences")
    print(f"Added {len(pattern_b_experiences)} Pattern B experiences")
    
    # Test retrieval for Pattern A-like input
    test_input_a = [1.25, 2.25, 3.25, 4.25]  # Should match Pattern A
    predicted_action_a, brain_state_a = brain.process_sensory_input(test_input_a)
    confidence_a = brain_state_a.get('prediction_confidence', 0.0)
    details_a = brain_state_a.get('prediction_details', {})
    
    print(f"\nPattern A test input: {test_input_a}")
    print(f"Predicted action: {predicted_action_a}")
    print(f"Confidence: {confidence_a:.3f}")
    print(f"Method: {details_a.get('method', 'unknown')}")
    print(f"Num similar: {details_a.get('num_similar', 0)}")
    
    # Test retrieval for Pattern B-like input
    test_input_b = [3.75, 2.75, 1.75, 0.75]  # Should match Pattern B
    predicted_action_b, brain_state_b = brain.process_sensory_input(test_input_b)
    confidence_b = brain_state_b.get('prediction_confidence', 0.0)
    details_b = brain_state_b.get('prediction_details', {})
    
    print(f"\nPattern B test input: {test_input_b}")
    print(f"Predicted action: {predicted_action_b}")
    print(f"Confidence: {confidence_b:.3f}")
    print(f"Method: {details_b.get('method', 'unknown')}")
    print(f"Num similar: {details_b.get('num_similar', 0)}")
    
    # Analyze what similar experiences were found
    all_experiences = brain.experience_storage.get_all_experiences()
    print(f"\nTotal experiences stored: {len(all_experiences)}")
    
    # Test direct similarity search
    exp_vectors = []
    exp_ids = []
    for exp_id, exp in all_experiences.items():
        exp_vectors.append(exp.get_context_vector())
        exp_ids.append(exp_id)
    
    similar_to_a = brain.similarity_engine.find_similar_experiences(
        test_input_a, exp_vectors, exp_ids, max_results=5, min_similarity=0.3
    )
    
    print(f"\nSimilar experiences to Pattern A input:")
    for exp_id, similarity in similar_to_a:
        exp = all_experiences[exp_id]
        print(f"  {exp_id[:8]}...: similarity={similarity:.3f}, context={exp.get_context_vector()}")
    
    return brain


def identify_potential_issues():
    """Identify potential issues with similarity search."""
    print("\n=== Potential Issues Analysis ===")
    
    issues = []
    
    # Issue 1: Check if similarity thresholds are appropriate
    print("1. Analyzing similarity thresholds...")
    brain = MinimalBrain()
    
    # Add diverse experiences
    for i in range(10):
        sensory = [np.random.normal(0, 1) for _ in range(4)]
        action = [np.random.normal(0, 0.5) for _ in range(4)]
        outcome = [s + np.random.normal(0, 0.1) for s in sensory]
        brain.store_experience(sensory, action, outcome)
    
    # Test similarity distribution
    all_experiences = brain.experience_storage.get_all_experiences()
    exp_vectors = [exp.get_context_vector() for exp in all_experiences.values()]
    
    if len(exp_vectors) > 1:
        # Compute all pairwise similarities
        similarities = []
        for i in range(len(exp_vectors)):
            for j in range(i+1, len(exp_vectors)):
                sim = brain.similarity_engine.compute_similarity(exp_vectors[i], exp_vectors[j])
                similarities.append(sim)
        
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            
            print(f"   Similarity distribution: mean={mean_sim:.3f}, std={std_sim:.3f}")
            print(f"   Range: [{min_sim:.3f}, {max_sim:.3f}]")
            
            # Check if default threshold (0.4) is appropriate
            if mean_sim < 0.2:
                issues.append("Similarity threshold (0.4) may be too high for typical similarities")
            elif mean_sim > 0.8:
                issues.append("Similarity threshold (0.4) may be too low - not discriminative enough")
    
    # Issue 2: Check if context vectors are meaningful
    print("\n2. Analyzing context vector construction...")
    
    # Create two very similar experiences
    exp1 = Experience(
        sensory_input=[1.0, 2.0, 3.0, 4.0],
        action_taken=[0.1, 0.2, 0.3, 0.4],
        outcome=[1.1, 2.1, 3.1, 4.1],
        prediction_error=0.2,
        timestamp=time.time()
    )
    
    exp2 = Experience(
        sensory_input=[1.001, 2.001, 3.001, 4.001],  # Almost identical
        action_taken=[0.1, 0.2, 0.3, 0.4],
        outcome=[1.1, 2.1, 3.1, 4.1],
        prediction_error=0.2,
        timestamp=time.time()
    )
    
    context1 = exp1.get_context_vector()
    context2 = exp2.get_context_vector()
    
    print(f"   Context vector 1: {context1}")
    print(f"   Context vector 2: {context2}")
    
    # Note: Context vector currently only uses sensory_input, not action_taken
    if context1 == exp1.sensory_input:
        issues.append("Context vectors only use sensory input - might miss action context")
    
    # Issue 3: Check prediction engine's use of similarity search
    print("\n3. Analyzing prediction engine integration...")
    
    # The prediction engine uses min_similarity=0.4 by default
    # Let's see if this is finding enough similar experiences
    test_context = [1.5, 2.5, 3.5, 4.5]
    predicted_action, brain_state = brain.process_sensory_input(test_context)
    confidence = brain_state.get('prediction_confidence', 0.0)
    details = brain_state.get('prediction_details', {})
    
    num_similar = details.get('num_similar', 0)
    method = details.get('method', 'unknown')
    
    print(f"   Prediction method: {method}")
    print(f"   Similar experiences found: {num_similar}")
    
    if num_similar == 0:
        issues.append("Prediction engine finding no similar experiences - threshold may be too strict")
    elif method == 'bootstrap_random':
        issues.append("Prediction engine falling back to random actions - similarity search not effective")
    
    # Issue 4: Check if learnable similarity is being used
    print("\n4. Analyzing learnable similarity usage...")
    
    sim_stats = brain.similarity_engine.get_performance_stats()
    similarity_type = sim_stats.get('similarity_type', 'unknown')
    
    print(f"   Similarity type: {similarity_type}")
    
    if similarity_type == 'hardcoded_cosine':
        issues.append("Using hardcoded cosine similarity - not adapting to data")
    elif 'similarity_learning' in sim_stats:
        learning_stats = sim_stats['similarity_learning']
        correlation = learning_stats.get('similarity_success_correlation', 0.0)
        adaptations = learning_stats.get('adaptations_performed', 0)
        
        print(f"   Learning correlation: {correlation:.3f}")
        print(f"   Adaptations performed: {adaptations}")
        
        if correlation < 0.2:
            issues.append("Learnable similarity showing poor correlation with prediction success")
        if adaptations == 0:
            issues.append("Learnable similarity not adapting - may need more training data")
    
    # Summary
    print(f"\n=== Issues Identified ===")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No obvious issues detected.")
    
    return issues


def main():
    """Run all similarity search investigations."""
    print("Similarity Search Investigation")
    print("=" * 50)
    
    # Run all tests
    test_basic_similarity_search()
    test_learnable_similarity()
    test_brain_similarity_integration()
    test_experience_storage_and_retrieval()
    issues = identify_potential_issues()
    
    print(f"\n" + "=" * 50)
    print("INVESTIGATION COMPLETE")
    print(f"Found {len(issues)} potential issues")
    
    if issues:
        print("\nRECOMMENDATIONS:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\nSimilarity search system appears to be working correctly.")


if __name__ == "__main__":
    main()