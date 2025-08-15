#!/usr/bin/env python3
"""
Debug prediction learning test.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np


def test_prediction_learning():
    """Debug prediction learning test."""
    
    print("🔍 Testing Prediction Learning")
    print("=" * 50)
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    # Clear history
    if hasattr(brain, '_prediction_confidence_history'):
        brain._prediction_confidence_history.clear()
        brain._improvement_rate_history.clear()
    
    # Test pattern from behavioral framework
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2 + [0.5] * 8
    
    print(f"Pattern length: {len(pattern)}")
    print("Running prediction learning test...")
    
    prediction_errors = []
    prediction_confidences = []
    
    for i in range(100):
        motor_output, brain_state = brain.process_robot_cycle(pattern)
        
        prediction_confidence = brain_state.get('prediction_confidence', 0.5)
        prediction_errors.append(1.0 - prediction_confidence)
        prediction_confidences.append(prediction_confidence)
        
        if i % 20 == 0:
            print(f"\nCycle {i}:")
            print(f"  Prediction confidence: {prediction_confidence:.3f}")
            print(f"  Motor output: {[f'{m:.4f}' for m in motor_output]}")
            print(f"  Field energy: {brain_state['field_energy']:.6f}")
    
    # Check learning
    print("\n📊 Learning analysis:")
    
    quarter_size = 25
    first_quarter = np.mean(prediction_errors[:quarter_size])
    last_quarter = np.mean(prediction_errors[-quarter_size:])
    
    print(f"  First quarter avg error: {first_quarter:.3f}")
    print(f"  Last quarter avg error: {last_quarter:.3f}")
    print(f"  Improvement: {first_quarter - last_quarter:.3f}")
    
    # Check confidence history
    print(f"\n  Confidence history length: {len(brain._prediction_confidence_history)}")
    if len(brain._prediction_confidence_history) > 10:
        recent = list(brain._prediction_confidence_history)[-10:]
        print(f"  Recent confidences: {[f'{c:.3f}' for c in recent]}")
    
    # Check if field is changing
    print(f"\n  Brain cycles: {brain.brain_cycles}")
    print(f"  Field evolution cycles: {brain.field_evolution_cycles}")
    print(f"  Predicted field exists: {brain._predicted_field is not None}")


if __name__ == "__main__":
    test_prediction_learning()