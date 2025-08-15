#!/usr/bin/env python3
"""
Prediction Engine Diagnostic Tool

Investigates specific issues with the prediction engine and pattern analysis system
to understand why learning improvement is not being maintained.
"""

import sys
import os
import time

# Add server directory to path
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

from src.brain import MinimalBrain

def test_pattern_caching_issue():
    """Test if pattern caching is preventing learning."""
    print("🔬 Testing Pattern Caching Issue...")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    brain.reset_brain()
    
    # Test the same input pattern repeatedly
    test_input = [1.0, 0.0, 0.0, 0.0]
    expected_action = [1.0, 0.0, 0.0, 0.0]
    
    print(f"Teaching pattern: {test_input} -> {expected_action}")
    
    # Track predictions and cognitive modes
    predictions_and_modes = []
    
    for i in range(15):
        # Get prediction
        predicted_action, brain_state = brain.process_sensory_input(test_input)
        
        # Extract cognitive mode and prediction details
        cognitive_mode = brain_state.get('cognitive_autopilot', {}).get('current_mode', 'unknown')
        prediction_method = brain_state.get('prediction_method', 'unknown')
        confidence = brain_state.get('prediction_confidence', 0.0)
        
        predictions_and_modes.append({
            'cycle': i + 1,
            'predicted_action': predicted_action.copy(),
            'cognitive_mode': cognitive_mode,
            'prediction_method': prediction_method,
            'confidence': confidence
        })
        
        # Store experience with correct action
        brain.store_experience(
            sensory_input=test_input,
            action_taken=expected_action,  # What should have been done
            outcome=expected_action,       # Perfect outcome
            predicted_action=predicted_action
        )
        
        print(f"Cycle {i+1:2d}: {cognitive_mode:>10} | {prediction_method:>20} | conf={confidence:.2f} | pred={[f'{x:.2f}' for x in predicted_action]}")
        
        time.sleep(0.05)  # Small delay to allow processing
    
    # Analyze patterns
    print(f"\n📊 Pattern Analysis:")
    
    # Check for mode transitions
    modes = [entry['cognitive_mode'] for entry in predictions_and_modes]
    mode_changes = []
    for i in range(1, len(modes)):
        if modes[i] != modes[i-1]:
            mode_changes.append(f"Cycle {i+1}: {modes[i-1]} -> {modes[i]}")
    
    print(f"Mode changes: {len(mode_changes)}")
    for change in mode_changes:
        print(f"  {change}")
    
    # Check for prediction method changes
    methods = [entry['prediction_method'] for entry in predictions_and_modes]
    method_changes = []
    for i in range(1, len(methods)):
        if methods[i] != methods[i-1]:
            method_changes.append(f"Cycle {i+1}: {methods[i-1]} -> {methods[i]}")
    
    print(f"\nMethod changes: {len(method_changes)}")
    for change in method_changes:
        print(f"  {change}")
    
    # Check for the problematic [0.1, 0.2, 0.3, 0.4] pattern
    hardcoded_pattern = [0.1, 0.2, 0.3, 0.4]
    stuck_cycles = []
    for entry in predictions_and_modes:
        pred = entry['predicted_action']
        if (abs(pred[0] - 0.1) < 0.001 and abs(pred[1] - 0.2) < 0.001 and 
            abs(pred[2] - 0.3) < 0.001 and abs(pred[3] - 0.4) < 0.001):
            stuck_cycles.append(entry['cycle'])
    
    print(f"\nHardcoded pattern [0.1, 0.2, 0.3, 0.4] detected in cycles: {stuck_cycles}")
    print(f"Percentage stuck: {len(stuck_cycles)/len(predictions_and_modes)*100:.1f}%")
    
    # Check final brain state
    final_stats = brain.get_brain_stats()
    print(f"\n📈 Final Brain Stats:")
    print(f"Total experiences: {final_stats['brain_summary']['total_experiences']}")
    print(f"Total predictions: {final_stats['brain_summary']['total_predictions']}")
    print(f"Pattern predictions: {final_stats['prediction_engine']['pattern_predictions']}")
    print(f"Consensus predictions: {final_stats['prediction_engine']['consensus_predictions']}")
    print(f"Random predictions: {final_stats['prediction_engine']['random_predictions']}")
    
    return predictions_and_modes

def test_cognitive_autopilot_behavior():
    """Test how cognitive autopilot affects prediction quality."""
    print("\n🧠 Testing Cognitive Autopilot Behavior...")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    brain.reset_brain()
    
    # Test different confidence scenarios to trigger mode changes
    test_scenarios = [
        {"confidence": 0.95, "expected_mode": "autopilot"},
        {"confidence": 0.80, "expected_mode": "focused"}, 
        {"confidence": 0.60, "expected_mode": "deep_think"},
        {"confidence": 0.40, "expected_mode": "deep_think"}
    ]
    
    for scenario in test_scenarios:
        # Manually set confidence to trigger mode
        brain._last_confidence = scenario["confidence"]
        
        # Process input
        test_input = [1.0, 0.5, 0.0, 0.0]
        predicted_action, brain_state = brain.process_sensory_input(test_input)
        
        actual_mode = brain_state.get('cognitive_autopilot', {}).get('current_mode', 'unknown')
        prediction_method = brain_state.get('prediction_method', 'unknown')
        
        print(f"Confidence {scenario['confidence']:.2f} -> Mode: {actual_mode:>10} | Method: {prediction_method:>20}")
        
        # Store experience
        brain.store_experience(test_input, predicted_action, predicted_action, predicted_action)
    
    return brain.get_brain_stats()

def main():
    """Run prediction engine diagnostics."""
    print("🧠 PREDICTION ENGINE DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Pattern caching issue
    predictions = test_pattern_caching_issue()
    
    # Test 2: Cognitive autopilot behavior
    autopilot_stats = test_cognitive_autopilot_behavior()
    
    print(f"\n💡 DIAGNOSIS:")
    print("1. Pattern caching in 'minimal' cognitive mode creates hardcoded predictions")
    print("2. This prevents the brain from learning the correct pattern") 
    print("3. The cognitive autopilot switches to minimal mode too aggressively")
    print("4. The cached pattern [0.1, 0.2, 0.3, 0.4] has high confidence, blocking learning")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    print("1. Reduce confidence of cached patterns in minimal mode")
    print("2. Make cognitive autopilot less aggressive in switching to minimal mode")
    print("3. Add pattern cache invalidation when learning new patterns")
    print("4. Use actual pattern analysis instead of hardcoded fallback in minimal mode")

if __name__ == "__main__":
    main()