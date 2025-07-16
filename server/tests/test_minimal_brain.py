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
    stats = brain.get_brain_stats()
    total_cycles = stats['brain_summary']['total_cycles']
    print(f"   Total cycles: {total_cycles}")
    
    assert total_cycles >= 1, "Should have at least 1 cycle"
    
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
    
    final_stats = brain.get_brain_stats()
    final_cycles = final_stats['brain_summary']['total_cycles']
    print(f"   Total cycles: {final_cycles}")
    print(f"   Vector patterns formed: {final_stats['brain_summary']['streams']['sensory_patterns'] + final_stats['brain_summary']['streams']['motor_patterns']}")
    
    # Test 4: Test pattern recognition
    print("\n🔍 Test 4: Testing pattern recognition")
    
    # Try a sensory input similar to what we've seen before
    familiar_input = [1.05, 2.05, 3.0, 4.0]
    predicted_action, state = brain.process_sensory_input(familiar_input, action_dimensions=2)
    
    print(f"   Familiar input: {familiar_input}")
    print(f"   Predicted action: {predicted_action}")
    print(f"   Prediction method: {state['prediction_method']}")
    print(f"   Confidence: {state['prediction_confidence']:.3f}")
    print(f"   Architecture: {state['architecture']}")
    
    # Should now use vector stream prediction  
    assert state['prediction_method'] == 'vector_stream_continuous', "Should use vector stream prediction"
    
    # Test 5: Get comprehensive brain stats
    print("\n📊 Test 5: Brain performance statistics")
    stats = brain.get_brain_stats()
    
    print(f"   Brain summary: {stats['brain_summary']}")
    print(f"   Architecture: {stats['brain_summary']['architecture']}")
    print(f"   Vector streams:")
    print(f"     - Sensory patterns: {stats['brain_summary']['streams']['sensory_patterns']}")
    print(f"     - Motor patterns: {stats['brain_summary']['streams']['motor_patterns']}")
    print(f"     - Temporal patterns: {stats['brain_summary']['streams']['temporal_patterns']}")
    print(f"   Vector brain: {stats['vector_brain']['prediction_confidence']:.3f} confidence")
    
    # Verify vector stream stats make sense
    assert stats['brain_summary']['total_cycles'] > 0, "Should have processed cycles"
    assert stats['brain_summary']['architecture'] == 'vector_stream', "Should be vector stream architecture"
    assert stats['vector_brain']['prediction_confidence'] >= 0.0, "Should have valid confidence"
    
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
    
    print(f"   100 vector stream updates stored")
    print(f"   Average prediction time: {search_time*1000:.2f}ms")
    print(f"   Vector stream architecture: continuous processing")
    
    assert search_time < 0.1, f"Vector stream prediction should be fast (<100ms), got {search_time*1000:.2f}ms"
    
    print("✅ Performance test passed!")


def test_working_memory_effects():
    """Test that vector streams create learning patterns."""
    
    print("\n🧠 Testing vector stream learning")
    
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
    
    # Check vector stream patterns formed
    stats = brain.get_brain_stats()
    total_patterns = (stats['brain_summary']['streams']['sensory_patterns'] + 
                     stats['brain_summary']['streams']['motor_patterns'] +
                     stats['brain_summary']['streams']['temporal_patterns'])
    
    print(f"   Total cycles: {stats['brain_summary']['total_cycles']}")
    print(f"   Stream patterns formed: {total_patterns}")
    print(f"   Prediction confidence: {state['prediction_confidence']:.3f}")
    print(f"   Architecture: {state['architecture']}")
    
    assert total_patterns > 0, "Should have formed stream patterns"
    assert state['architecture'] == 'vector_stream', "Should use vector stream architecture"
    assert stats['brain_summary']['total_cycles'] >= 1, "Should have processed cycles"
    
    print("✅ Vector stream learning test passed!")


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