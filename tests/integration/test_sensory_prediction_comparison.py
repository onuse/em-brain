#!/usr/bin/env python3
"""
Sensory Prediction A/B Test
Compare prediction learning with and without proper sensory prediction.
"""

import sys
import time
import json
sys.path.append('server/tools/testing')

def run_prediction_comparison():
    """Run side-by-side comparison of prediction methods."""
    print("🧪 Sensory Prediction A/B Test")
    print("=" * 60)
    
    from behavioral_test_framework import BehavioralTestFramework
    
    # Test configuration
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 16,
            'motor_dim': 4,
            'spatial_resolution': 6,  # Small for speed
        },
        'memory': {'enable_persistence': False}
    }
    
    results = {}
    
    # TEST A: Current approach (confidence-based prediction error)
    print("\n🅰️  TEST A: Current Confidence-Based Prediction")
    print("-" * 50)
    
    framework_a = BehavioralTestFramework(quiet_mode=True)
    brain_a = framework_a.create_brain(config)
    
    # Ensure we're using current (disabled) prediction error approach
    original_process = brain_a.process_sensory_input
    
    start_time = time.time()
    score_a = framework_a.test_prediction_learning(brain_a, cycles=100)
    time_a = time.time() - start_time
    
    results['test_a'] = {
        'approach': 'confidence_based',
        'score': score_a,
        'time': time_a,
        'description': 'Current approach: prediction_error = 1.0 - confidence'
    }
    
    print(f"  Score: {score_a:.3f}")
    print(f"  Time: {time_a:.1f}s")
    
    # TEST B: With proper sensory prediction
    print("\n🅱️  TEST B: Proper Sensory Prediction")
    print("-" * 50)
    
    framework_b = BehavioralTestFramework(quiet_mode=True)
    brain_b = framework_b.create_brain(config)
    
    # Patch the brain to use proper sensory prediction
    brain_b._last_sensory_input = None
    brain_b._predicted_sensory = None
    
    def enhanced_process_sensory_input(sensory_input, action_dimensions=4):
        """Enhanced processing with sensory prediction."""
        
        # Store prediction from last cycle for error calculation
        if brain_b._predicted_sensory is not None and brain_b._last_sensory_input is not None:
            # Calculate actual prediction error (sensory prediction vs reality)
            import numpy as np
            predicted = np.array(brain_b._predicted_sensory[:len(sensory_input)])
            actual = np.array(sensory_input[:len(brain_b._predicted_sensory)])
            
            if len(predicted) > 0 and len(actual) > 0:
                prediction_error = np.linalg.norm(predicted - actual) / max(1.0, np.linalg.norm(actual))
                prediction_error = min(1.0, prediction_error)  # Clamp to [0,1]
            else:
                prediction_error = 0.5
        else:
            prediction_error = 0.5  # Initial cycle
            
        # Call original processing
        predicted_action, brain_state = original_process(sensory_input, action_dimensions)
        
        # Generate prediction for next sensory state using simple field momentum
        if hasattr(brain_b.unified_brain, '_last_unified_field'):
            # Use field momentum to predict next sensory state
            field_momentum = brain_b.unified_brain.unified_field - brain_b.unified_brain._last_unified_field
            # Extract sensory prediction from field momentum (first 16 values)
            momentum_sum = field_momentum.sum(dim=(0,1,2,3,4)).cpu().numpy()
            
            # Simple prediction: current sensors + momentum-based change
            next_sensory_prediction = np.array(sensory_input) + momentum_sum[:len(sensory_input)] * 0.1
            # Clamp to reasonable sensor range
            next_sensory_prediction = np.clip(next_sensory_prediction, 0.0, 1.0)
            brain_b._predicted_sensory = next_sensory_prediction.tolist()
        else:
            # First cycle - just predict current state
            brain_b._predicted_sensory = sensory_input[:]
            
        # Store field state for next momentum calculation
        brain_b.unified_brain._last_unified_field = brain_b.unified_brain.unified_field.clone()
        
        # Update brain state with proper prediction error
        brain_state['prediction_error'] = prediction_error
        brain_state['prediction_confidence'] = max(0.0, 1.0 - prediction_error)
        
        # Store for next cycle
        brain_b._last_sensory_input = sensory_input[:]
        
        return predicted_action, brain_state
    
    # Patch the brain's processing method
    brain_b.process_sensory_input = enhanced_process_sensory_input
    
    start_time = time.time()
    score_b = framework_b.test_prediction_learning(brain_b, cycles=100)
    time_b = time.time() - start_time
    
    results['test_b'] = {
        'approach': 'sensory_prediction',
        'score': score_b,
        'time': time_b,
        'description': 'Enhanced: actual sensory prediction vs reality comparison'
    }
    
    print(f"  Score: {score_b:.3f}")
    print(f"  Time: {time_b:.1f}s")
    
    # COMPARISON ANALYSIS
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Current Approach (A):     {score_a:.3f} in {time_a:.1f}s")
    print(f"Sensory Prediction (B):   {score_b:.3f} in {time_b:.1f}s")
    
    improvement = score_b - score_a
    improvement_pct = (improvement / max(0.001, score_a)) * 100
    
    print(f"\nImprovement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    # DECISION CRITERIA
    if improvement > 0.05:  # More than 5% absolute improvement
        print("✅ SIGNIFICANT IMPROVEMENT: Worth implementing!")
        recommendation = "implement"
    elif improvement > 0.01:  # 1-5% improvement
        print("⚠️  MODEST IMPROVEMENT: Consider implementing")
        recommendation = "consider"
    elif improvement > -0.01:  # Within 1%
        print("➖ NO SIGNIFICANT CHANGE: Current approach is fine")
        recommendation = "no_change"
    else:
        print("❌ PERFORMANCE REGRESSION: Don't implement")
        recommendation = "reject"
    
    results['comparison'] = {
        'improvement_absolute': improvement,
        'improvement_percent': improvement_pct,
        'recommendation': recommendation
    }
    
    # Save results to file
    results_file = f"sensory_prediction_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = run_prediction_comparison()
    
    print(f"\n🎯 FINAL RECOMMENDATION: {results['comparison']['recommendation'].upper()}")
    
    if results['comparison']['recommendation'] == 'implement':
        print("   → Implement proper sensory prediction in UnifiedFieldBrain")
    elif results['comparison']['recommendation'] == 'consider':
        print("   → Consider implementing if you have time")
    else:
        print("   → Stick with current confidence-based approach")