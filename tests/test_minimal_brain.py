#!/usr/bin/env python3
"""
Quick test of the minimal brain implementation.

Test that all 4 systems work together to create basic intelligent behavior.
"""

import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.brain import MinimalBrain
import numpy as np


def test_basic_brain_functionality():
    """Test basic brain functionality - the complete learning loop."""
    
    print("🧪 Testing MinimalBrain basic functionality")
    
    # Initialize brain
    brain = MinimalBrain()
    
    # Test 1: Initial prediction (should be random)
    print("\n📍 Test 1: Initial prediction (no experiences)")
    sensory_input = [1.0, 2.0, 3.0, 4.0]  # Simple 4D sensory input
    action, brain_state = brain.process_sensory_input(sensory_input, action_dimensions=2)
    
    print(f"   Sensory input: {sensory_input}")
    print(f"   Predicted action: {action}")
    print(f"   Prediction method: {brain_state['prediction_method']}")
    print(f"   Confidence: {brain_state['prediction_confidence']:.3f}")
    
    assert len(action) == 2, "Action should have 2 dimensions"
    assert brain_state['prediction_method'] in ['bootstrap_random', 'bootstrap_from_similar'], "Should be bootstrap for first prediction"
    
    # Test 2: Store first experience
    print("\n📝 Test 2: Storing first experience")
    actual_outcome = [1.5, 2.5, 3.5, 4.5]  # What actually happened (same dims as sensory input)
    exp_id = brain.store_experience(sensory_input, action, actual_outcome, action)
    
    print(f"   Stored experience ID: {exp_id[:8]}...")
    print(f"   Total experiences: {brain.total_experiences}")
    
    assert brain.total_experiences == 1, "Should have 1 experience"
    
    # Test 3: Learn from multiple experiences
    print("\n🎓 Test 3: Learning from multiple experiences")
    
    for i in range(10):
        # Create slightly different sensory inputs
        sensors = [1.0 + i*0.1, 2.0 + i*0.1, 3.0, 4.0]
        predicted_action, state = brain.process_sensory_input(sensors, action_dimensions=2)
        
        # Simulate outcome (with some pattern)
        outcome = [sensors[0] + 0.5, sensors[1] + 0.5, sensors[2], sensors[3]]
        
        brain.store_experience(sensors, predicted_action, outcome, predicted_action)
        
        if i % 3 == 0:
            print(f"   Experience {i+2}: {state['prediction_method']}, confidence: {state['prediction_confidence']:.3f}")
    
    print(f"   Total experiences: {brain.total_experiences}")
    
    # Test 4: Test pattern recognition
    print("\n🔍 Test 4: Testing pattern recognition")
    
    # Try a sensory input similar to what we've seen before
    familiar_input = [1.05, 2.05, 3.0, 4.0]
    predicted_action, state = brain.process_sensory_input(familiar_input, action_dimensions=2)
    
    print(f"   Familiar input: {familiar_input}")
    print(f"   Predicted action: {predicted_action}")
    print(f"   Prediction method: {state['prediction_method']}")
    print(f"   Confidence: {state['prediction_confidence']:.3f}")
    print(f"   Similar experiences found: {state['num_similar_experiences']}")
    print(f"   Working memory size: {state['working_memory_size']}")
    
    # Should now use consensus prediction with reasonable confidence
    assert state['num_similar_experiences'] > 0, "Should find similar experiences"
    
    # Test 5: Get comprehensive brain stats
    print("\n📊 Test 5: Brain performance statistics")
    stats = brain.get_brain_stats()
    
    print(f"   Brain summary: {stats['brain_summary']}")
    print(f"   Similarity engine: {stats['similarity_engine']['device']}")
    # Handle different activation system stats structures
    activation_stats = stats['activation_dynamics']
    if 'working_memory_size' in activation_stats:
        working_memory_size = activation_stats['working_memory_size']
    elif 'current_working_memory_size' in activation_stats:
        working_memory_size = activation_stats['current_working_memory_size']
    else:
        working_memory_size = "unknown"
    
    print(f"   Activation dynamics: {working_memory_size} in working memory")
    print(f"   Prediction engine: {stats['prediction_engine']['consensus_rate']:.2f} consensus rate")
    
    # Verify stats make sense
    assert stats['brain_summary']['total_experiences'] == brain.total_experiences
    assert stats['brain_summary']['total_predictions'] == brain.total_predictions
    
    print("\n✅ All tests passed! Minimal brain is working correctly.")
    return True


def test_similarity_performance():
    """Test similarity search performance with larger dataset."""
    
    print("\n🚀 Testing similarity search performance")
    
    brain = MinimalBrain()
    
    # Add many experiences
    print("   Adding 100 experiences...")
    for i in range(100):
        sensors = np.random.normal(0, 1, 8).tolist()  # 8D sensory input
        action = np.random.normal(0, 0.5, 4).tolist()  # 4D action
        outcome = (np.array(sensors) + np.random.normal(0, 0.1, 8)).tolist()
        
        brain.store_experience(sensors, action, outcome)
    
    # Test search performance
    import time
    test_input = np.random.normal(0, 1, 8).tolist()
    
    start_time = time.time()
    for _ in range(10):
        predicted_action, state = brain.process_sensory_input(test_input, action_dimensions=4)
    search_time = (time.time() - start_time) / 10
    
    print(f"   100 experiences stored")
    print(f"   Average prediction time: {search_time*1000:.2f}ms")
    print(f"   Similarity engine: {brain.similarity_engine.get_performance_stats()['device']}")
    
    assert search_time < 0.1, f"Prediction should be fast (<100ms), got {search_time*1000:.2f}ms"
    
    print("✅ Performance test passed!")


def test_working_memory_effects():
    """Test that activation dynamics create working memory effects."""
    
    print("\n🧠 Testing working memory effects")
    
    brain = MinimalBrain()
    
    # Add some experiences
    experiences = []
    for i in range(20):
        sensors = [float(i), float(i+1), 0.0, 0.0]
        action = [0.5, 0.5]
        outcome = [float(i+0.5), float(i+1.5), 0.0, 0.0]
        
        exp_id = brain.store_experience(sensors, action, outcome)
        experiences.append(exp_id)
    
    # Process a sensory input that should activate specific experiences
    target_input = [5.0, 6.0, 0.0, 0.0]  # Similar to experience #5
    predicted_action, state = brain.process_sensory_input(target_input)
    
    # Check working memory (handle different activation systems)
    working_memory_size = state['working_memory_size']
    
    # Get activated experiences based on activation system type
    if hasattr(brain.activation_dynamics, 'get_activated_experiences'):
        # Traditional activation system
        activated_experiences = brain.activation_dynamics.get_activated_experiences(
            brain.experience_storage._experiences, min_activation=0.1
        )
    elif hasattr(brain.activation_dynamics, 'get_working_memory_experiences'):
        # Utility-based activation system
        working_memory_list = brain.activation_dynamics.get_working_memory_experiences(min_activation=0.1)
        activated_experiences = [brain.experience_storage._experiences[exp_id] for exp_id, _ in working_memory_list]
    else:
        activated_experiences = []
    
    print(f"   Total experiences: {len(brain.experience_storage._experiences)}")
    print(f"   Working memory size: {working_memory_size}")
    print(f"   Activated experiences: {len(activated_experiences)}")
    
    if activated_experiences:
        print(f"   Most activated: {activated_experiences[0].experience_id[:8]}... "
              f"(activation: {activated_experiences[0].activation_level:.3f})")
    
    assert working_memory_size > 0, "Should have some experiences in working memory"
    # Working memory might include all experiences if they're all recently activated
    assert working_memory_size <= len(brain.experience_storage._experiences), "Working memory should not exceed total"
    
    print("✅ Working memory test passed!")


def main():
    """Main test function."""
    try:
        test_basic_brain_functionality()
        test_similarity_performance() 
        test_working_memory_effects()
        
        print("\n🎉 All minimal brain tests passed successfully!")
        print("The 4-system minimal brain is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()