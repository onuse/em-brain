#!/usr/bin/env python3
"""
Concrete Brain Learning Tests

Tests whether the brain can actually learn recognizable patterns
and make intelligent decisions. Replaces abstract field energy metrics
with concrete, understandable capabilities.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_factory import BrainFactory

def test_pattern_recognition():
    """Test if brain can learn to recognize and respond to specific patterns."""
    print("🧠 Test 1: Pattern Recognition Learning")
    
    # Clear memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    # Define distinct patterns
    pattern_A = [1.0, 0.0, 1.0, 0.0] * 4  # Alternating high-low
    pattern_B = [0.3, 0.7, 0.3, 0.7] * 4  # Alternating medium values
    pattern_C = [1.0, 1.0, 0.0, 0.0] * 4  # Block pattern
    
    print(f"   Teaching 3 distinct patterns...")
    
    # Track brain responses to each pattern
    responses_A = []
    responses_B = []  
    responses_C = []
    
    # Training phase: 30 cycles of each pattern
    for cycle in range(30):
        # Present each pattern and record response
        action_A, _ = brain.process_sensory_input(pattern_A)
        action_B, _ = brain.process_sensory_input(pattern_B)
        action_C, _ = brain.process_sensory_input(pattern_C)
        
        responses_A.append(action_A[:2])  # First 2 action values
        responses_B.append(action_B[:2])
        responses_C.append(action_C[:2])
    
    # Analysis: Check if brain developed distinct responses
    mean_response_A = np.mean(responses_A[-10:], axis=0)  # Last 10 responses
    mean_response_B = np.mean(responses_B[-10:], axis=0)
    mean_response_C = np.mean(responses_C[-10:], axis=0)
    
    # Calculate separation between responses
    sep_AB = np.linalg.norm(mean_response_A - mean_response_B)
    sep_AC = np.linalg.norm(mean_response_A - mean_response_C)
    sep_BC = np.linalg.norm(mean_response_B - mean_response_C)
    
    avg_separation = (sep_AB + sep_AC + sep_BC) / 3
    
    print(f"   Pattern A response: [{mean_response_A[0]:.3f}, {mean_response_A[1]:.3f}]")
    print(f"   Pattern B response: [{mean_response_B[0]:.3f}, {mean_response_B[1]:.3f}]")
    print(f"   Pattern C response: [{mean_response_C[0]:.3f}, {mean_response_C[1]:.3f}]")
    print(f"   Average separation: {avg_separation:.3f}")
    
    # Success criteria: distinct responses (separation > 0.1)
    success = avg_separation > 0.1
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'} - {'Learned distinct responses' if success else 'Responses too similar'}")
    
    brain.finalize_session()
    return success, avg_separation

def test_sequence_prediction():
    """Test if brain can learn to predict sequence patterns."""
    print("\\n🧠 Test 2: Sequence Prediction")
    
    # Clear memory
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    # Simple repeating sequence: A -> B -> C -> A -> B -> C...
    sequence = [
        [1.0, 0.0, 0.0, 0.0] * 4,  # A
        [0.0, 1.0, 0.0, 0.0] * 4,  # B  
        [0.0, 0.0, 1.0, 0.0] * 4   # C
    ]
    
    print(f"   Teaching sequence: A → B → C → A → B → C...")
    
    prediction_errors = []
    
    # Training: 20 full sequences (60 steps)
    for cycle in range(60):
        current_pattern = sequence[cycle % 3]
        next_pattern = sequence[(cycle + 1) % 3]
        
        # Get brain's response to current pattern
        action, brain_state = brain.process_sensory_input(current_pattern)
        
        # After 30 cycles, start measuring prediction accuracy
        if cycle >= 30:
            # Check if brain's action correlates with next pattern
            predicted_values = action[:4]  # First 4 action values
            actual_next = next_pattern[:4]  # First 4 values of next pattern
            
            # Calculate prediction error
            error = np.mean(np.abs(np.array(predicted_values) - np.array(actual_next)))
            prediction_errors.append(error)
    
    avg_prediction_error = np.mean(prediction_errors)
    improvement = prediction_errors[0] - prediction_errors[-1] if len(prediction_errors) > 1 else 0
    
    print(f"   Initial prediction error: {prediction_errors[0]:.3f}")
    print(f"   Final prediction error: {prediction_errors[-1]:.3f}")
    print(f"   Improvement: {improvement:+.3f}")
    print(f"   Average error: {avg_prediction_error:.3f}")
    
    # Success criteria: error < 0.3 and improvement > 0.05
    success = avg_prediction_error < 0.3 and improvement > 0.05
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'} - {'Learned to predict' if success else 'Poor prediction accuracy'}")
    
    brain.finalize_session()
    return success, improvement

def test_adaptation_speed():
    """Test how quickly brain adapts to pattern changes."""
    print("\\n🧠 Test 3: Adaptation Speed")
    
    # Clear memory
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    # Start with pattern X
    pattern_X = [0.8, 0.2, 0.8, 0.2] * 4
    pattern_Y = [0.2, 0.8, 0.2, 0.8] * 4  # Opposite pattern
    
    print(f"   Phase 1: Learning pattern X (20 cycles)")
    print(f"   Phase 2: Switch to pattern Y - measuring adaptation speed")
    
    responses = []
    
    # Phase 1: Learn pattern X (20 cycles)
    for cycle in range(20):
        action, _ = brain.process_sensory_input(pattern_X)
        responses.append(np.mean(action))
    
    phase1_response = np.mean(responses[-5:])  # Stable response to X
    
    # Phase 2: Switch to pattern Y and measure adaptation
    adaptation_responses = []
    for cycle in range(15):
        action, _ = brain.process_sensory_input(pattern_Y)
        adaptation_responses.append(np.mean(action))
        responses.append(np.mean(action))
    
    # Measure how quickly response changed
    initial_Y_response = adaptation_responses[0]
    final_Y_response = np.mean(adaptation_responses[-3:])
    
    adaptation_magnitude = abs(final_Y_response - initial_Y_response)
    total_change = abs(phase1_response - final_Y_response)
    
    print(f"   Pattern X response: {phase1_response:.3f}")
    print(f"   Initial Y response: {initial_Y_response:.3f}")
    print(f"   Final Y response: {final_Y_response:.3f}")
    print(f"   Adaptation magnitude: {adaptation_magnitude:.3f}")
    print(f"   Total change: {total_change:.3f}")
    
    # Success criteria: significant adaptation (change > 0.1)
    success = total_change > 0.1
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'} - {'Adapted to new pattern' if success else 'Failed to adapt'}")
    
    brain.finalize_session()
    return success, adaptation_magnitude

def test_memory_persistence():
    """Test if brain remembers learned patterns after restart."""
    print("\\n🧠 Test 4: Memory Persistence")
    
    # Clear memory
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    # Phase 1: Train brain
    print(f"   Phase 1: Training brain on specific pattern")
    brain1 = BrainFactory(quiet_mode=True)
    
    training_pattern = [0.9, 0.1, 0.5, 0.3] * 4
    
    for cycle in range(25):
        action, _ = brain1.process_sensory_input(training_pattern)
    
    # Get final response
    final_action, _ = brain1.process_sensory_input(training_pattern)
    trained_response = np.mean(final_action[:2])
    
    brain1.finalize_session()  # Save memory
    print(f"   Trained response: {trained_response:.3f}")
    
    # Phase 2: Create new brain and test memory
    print(f"   Phase 2: Creating new brain - testing memory recall")
    brain2 = BrainFactory(quiet_mode=True)
    
    # Test immediate response (should recall from memory)
    immediate_action, _ = brain2.process_sensory_input(training_pattern)
    recalled_response = np.mean(immediate_action[:2])
    
    memory_similarity = 1.0 - abs(trained_response - recalled_response)
    
    print(f"   Recalled response: {recalled_response:.3f}")
    print(f"   Memory similarity: {memory_similarity:.3f}")
    
    # Success criteria: similar response (similarity > 0.8)
    success = memory_similarity > 0.8
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'} - {'Memory preserved' if success else 'Memory lost'}")
    
    brain2.finalize_session()
    return success, memory_similarity

def run_all_brain_tests():
    """Run comprehensive brain capability tests."""
    print("🧪 CONCRETE BRAIN LEARNING TESTS")
    print("="*50)
    print("Testing actual learning capabilities instead of abstract metrics\\n")
    
    results = []
    
    # Run all tests
    test1_success, pattern_sep = test_pattern_recognition()
    test2_success, prediction_improvement = test_sequence_prediction()
    test3_success, adaptation_speed = test_adaptation_speed()
    test4_success, memory_quality = test_memory_persistence()
    
    results = [test1_success, test2_success, test3_success, test4_success]
    
    # Summary
    print("\\n" + "="*50)
    print("🏆 BRAIN CAPABILITY SUMMARY")
    print("="*50)
    print(f"Pattern Recognition: {'✅' if test1_success else '❌'} (separation: {pattern_sep:.3f})")
    print(f"Sequence Prediction: {'✅' if test2_success else '❌'} (improvement: {prediction_improvement:+.3f})")
    print(f"Adaptation Speed:    {'✅' if test3_success else '❌'} (magnitude: {adaptation_speed:.3f})")
    print(f"Memory Persistence:  {'✅' if test4_success else '❌'} (quality: {memory_quality:.3f})")
    
    success_rate = sum(results) / len(results) * 100
    print(f"\\nOverall Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)} tests passed)")
    
    if success_rate >= 75:
        print("\\n🎉 BRAIN IS WORKING WELL! Clear evidence of learning and intelligence.")
    elif success_rate >= 50:
        print("\\n⚠️  BRAIN PARTIALLY WORKING. Some capabilities present but needs improvement.")
    else:
        print("\\n❌ BRAIN NOT WORKING. Minimal evidence of learning or intelligence.")
    
    print("\\nThese concrete tests provide clear evidence of brain functionality,")
    print("unlike abstract field energy metrics that are hard to interpret.")
    
    return success_rate

if __name__ == "__main__":
    run_all_brain_tests()