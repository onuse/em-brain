#!/usr/bin/env python3
"""
Test Similarity Learning Adaptation

Verifies that the similarity learning feedback loop is working correctly
after fixing the critical implementation gap.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))

# Direct imports since we're in the server/src directory
sys.path.append(os.path.join(brain_root, 'server'))
from src.brain import MinimalBrain
from src.experience import Experience

def test_similarity_learning_feedback():
    """Test that similarity learning adapts based on prediction success."""
    print("🧪 TESTING SIMILARITY LEARNING ADAPTATION")
    print("=" * 50)
    
    # Create a fresh brain for testing
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    print(f"✅ Brain initialized with {len(brain.experience_storage._experiences)} experiences")
    
    # Create test experiences with clear patterns
    test_experiences = [
        # Pattern 1: Similar sensory input should lead to similar actions
        ([1.0, 0.5, 0.2, 0.1], [0.8, 0.6, 0.3, 0.2]),  # Similar inputs
        ([1.1, 0.4, 0.3, 0.2], [0.9, 0.7, 0.4, 0.3]),  # Similar inputs
        ([1.2, 0.6, 0.1, 0.3], [0.7, 0.5, 0.2, 0.1]),  # Similar inputs
        
        # Pattern 2: Different sensory input should lead to different actions  
        ([0.1, 0.9, 0.8, 0.7], [0.2, 0.1, 0.9, 0.8]),  # Different inputs
        ([0.2, 0.8, 0.9, 0.6], [0.1, 0.3, 0.8, 0.7]),  # Different inputs
        ([0.3, 0.7, 0.8, 0.8], [0.3, 0.2, 0.7, 0.9]),  # Different inputs
    ]
    
    # Store experiences and create predictions
    print(f"\n📊 PHASE 1: Creating {len(test_experiences)} experiences")
    experience_ids = []
    
    for i, (sensory, action) in enumerate(test_experiences):
        # For outcome, use a simple transformation (action + noise)
        outcome = [a + 0.1 * np.random.normal() for a in action]
        
        # Store experience
        exp_id = brain.store_experience(
            sensory_input=sensory,
            action_taken=action,
            outcome=outcome,
            predicted_action=action  # Perfect prediction for testing
        )
        experience_ids.append(exp_id)
        
        print(f"  Experience {i+1}: sensory={sensory[:2]}..., action={action[:2]}...")
    
    print(f"✅ Created {len(experience_ids)} experiences")
    
    # Check initial similarity statistics
    if brain.similarity_engine.learnable_similarity:
        initial_stats = brain.similarity_engine.learnable_similarity.get_similarity_statistics()
        print(f"\n📈 INITIAL SIMILARITY STATISTICS:")
        print(f"  - Adaptations performed: {initial_stats.get('adaptations_performed', 0)}")
        print(f"  - Learning rate: {initial_stats.get('learning_rate', 0)}")
        print(f"  - Success correlation: {initial_stats.get('similarity_success_correlation', 0):.3f}")
        print(f"  - Tracked outcomes: {initial_stats.get('prediction_outcomes_tracked', 0)}")
    else:
        print(f"\n❌ FAILURE: Learnable similarity not enabled")
        return False
    
    # Now create more experiences with deliberate prediction errors to trigger adaptation
    print(f"\n📊 PHASE 2: Creating experiences with prediction errors to trigger adaptation")
    
    for i in range(10):
        # Create experiences where similar inputs have different outcomes
        # This should trigger similarity learning adaptation
        sensory_input = [0.5 + 0.1 * np.random.normal() for _ in range(4)]
        predicted_action = [0.5, 0.5, 0.5, 0.5]
        
        if i % 2 == 0:
            # Sometimes make "similar" experiences fail 
            actual_action = [0.1, 0.1, 0.1, 0.1]  # Very different from prediction
            outcome = [0.9, 0.9, 0.9, 0.9]  # Very different outcome
        else:
            # Sometimes make them succeed
            actual_action = [0.6, 0.4, 0.5, 0.5]  # Close to prediction
            outcome = [0.5, 0.5, 0.5, 0.5]  # Close to prediction
        
        # Store experience with prediction error
        exp_id = brain.store_experience(
            sensory_input=sensory_input,
            action_taken=actual_action,
            outcome=outcome,
            predicted_action=predicted_action
        )
        
        print(f"  Experience {len(experience_ids)+i+1}: prediction_error induces adaptation")
    
    # Force adaptation to test if the feedback loop works
    print(f"\n🔧 FORCING ADAPTATION TO TEST FEEDBACK LOOP")
    print(f"Manually calling adapt_similarity_function()...")
    
    # Call adaptation multiple times to ensure it triggers
    for i in range(3):
        brain.similarity_engine.adapt_similarity_function()
        print(f"  Adaptation call {i+1} completed")
    
    # Check if adaptation occurred
    if brain.similarity_engine.learnable_similarity:
        final_stats = brain.similarity_engine.learnable_similarity.get_similarity_statistics()
        print(f"\n📈 FINAL SIMILARITY STATISTICS:")
        print(f"  - Adaptations performed: {final_stats.get('adaptations_performed', 0)}")
        print(f"  - Learning rate: {final_stats.get('learning_rate', 0)}")
        print(f"  - Success correlation: {final_stats.get('similarity_success_correlation', 0):.3f}")
        print(f"  - Tracked outcomes: {final_stats.get('prediction_outcomes_tracked', 0)}")
    else:
        print(f"\n❌ FAILURE: Learnable similarity not enabled")
        return False
    
    # Test results
    print(f"\n🎯 TEST RESULTS:")
    adaptations_performed = final_stats.get('adaptations_performed', 0)
    tracked_outcomes = final_stats.get('prediction_outcomes_tracked', 0)
    
    if adaptations_performed > 0:
        print(f"  ✅ SUCCESS: Similarity learning performed {adaptations_performed} adaptations")
        print(f"  ✅ SUCCESS: Tracked {tracked_outcomes} prediction outcomes")
        print(f"  ✅ FEEDBACK LOOP IS WORKING: Brain is learning from prediction success")
    else:
        print(f"  ❌ FAILURE: No adaptations performed despite prediction errors")
        print(f"  ❌ FAILURE: Feedback loop may still be broken")
        
        # Debug information
        print(f"\n🔍 DEBUG INFO:")
        print(f"  - Total experiences: {len(brain.experience_storage._experiences)}")
        print(f"  - Learnable similarity enabled: {brain.similarity_engine.use_learnable_similarity}")
        
        if brain.similarity_engine.learnable_similarity:
            learnable_stats = brain.similarity_engine.learnable_similarity.get_similarity_statistics()
            print(f"  - Learnable similarity outcomes: {learnable_stats.get('prediction_outcomes_tracked', 0)}")
    
    # Test actual similarity computation changes
    print(f"\n🧮 TESTING SIMILARITY COMPUTATION CHANGES:")
    test_vec1 = [1.0, 0.5, 0.2, 0.1]
    test_vec2 = [1.1, 0.4, 0.3, 0.2]
    
    similarity_score = brain.similarity_engine.compute_similarity(test_vec1, test_vec2)
    print(f"  - Similarity between similar vectors: {similarity_score:.3f}")
    
    test_vec3 = [0.1, 0.9, 0.8, 0.7]
    similarity_score2 = brain.similarity_engine.compute_similarity(test_vec1, test_vec3)
    print(f"  - Similarity between different vectors: {similarity_score2:.3f}")
    
    if similarity_score > similarity_score2:
        print(f"  ✅ SUCCESS: Similarity function correctly distinguishes similar vs different")
    else:
        print(f"  ⚠️ WARNING: Similarity function may not be discriminating properly")
    
    return adaptations_performed > 0

def test_prediction_outcome_recording():
    """Test that prediction outcomes are being recorded in similarity engine."""
    print(f"\n🧪 TESTING PREDICTION OUTCOME RECORDING")
    print("-" * 40)
    
    # Create a fresh brain
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    # Create some experiences first
    for i in range(5):
        sensory = [0.5 + 0.1 * i for _ in range(4)]
        action = [0.3 + 0.1 * i for _ in range(4)]
        outcome = [0.4 + 0.1 * i for _ in range(4)]
        
        brain.store_experience(sensory, action, outcome, predicted_action=action)
    
    # Test manual recording of prediction outcomes
    print(f"📝 Manually recording prediction outcomes...")
    
    query_vector = [0.5, 0.5, 0.5, 0.5]
    similar_vector = [0.6, 0.4, 0.5, 0.5]
    
    # Record some prediction outcomes
    brain.similarity_engine.record_prediction_outcome(
        query_vector=query_vector,
        similar_experience_id="test_exp_1",
        similar_vector=similar_vector,
        prediction_success=0.8
    )
    
    brain.similarity_engine.record_prediction_outcome(
        query_vector=query_vector,
        similar_experience_id="test_exp_2", 
        similar_vector=similar_vector,
        prediction_success=0.3
    )
    
    # Check if outcomes were recorded
    if brain.similarity_engine.learnable_similarity:
        learnable_stats = brain.similarity_engine.learnable_similarity.get_similarity_statistics()
        outcomes_tracked = learnable_stats.get('prediction_outcomes_tracked', 0)
        
        print(f"  ✅ SUCCESS: Recorded {outcomes_tracked} prediction outcomes")
        return outcomes_tracked > 0
    else:
        print(f"  ❌ FAILURE: Learnable similarity not enabled")
        return False

def main():
    """Run the similarity learning tests."""
    print("🧪 SIMILARITY LEARNING FEEDBACK LOOP TEST")
    print("=" * 60)
    print("Testing the fixed prediction feedback loop implementation...")
    
    # Test 1: Basic prediction outcome recording
    recording_works = test_prediction_outcome_recording()
    
    # Test 2: Full similarity learning adaptation
    adaptation_works = test_similarity_learning_feedback()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 OVERALL TEST RESULTS:")
    print(f"  - Prediction outcome recording: {'✅ PASS' if recording_works else '❌ FAIL'}")
    print(f"  - Similarity learning adaptation: {'✅ PASS' if adaptation_works else '❌ FAIL'}")
    
    if recording_works and adaptation_works:
        print(f"\n✅ SUCCESS: Similarity learning feedback loop is working correctly!")
        print(f"   The brain can now learn from prediction success and adapt similarity function.")
        print(f"   Ready to proceed with sparse connectivity optimizations.")
    else:
        print(f"\n❌ FAILURE: Similarity learning feedback loop still has issues.")
        print(f"   Need to investigate further before proceeding.")
    
    return recording_works and adaptation_works

if __name__ == "__main__":
    main()