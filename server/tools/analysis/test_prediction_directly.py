#!/usr/bin/env python3
"""
Test prediction learning by directly accessing the brain

This bypasses the connection/adapter layers to see the actual brain state
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.brains.field.dynamic_unified_brain import DynamicUnifiedFieldBrain

def test_prediction_directly():
    """Test prediction by directly accessing brain internals"""
    print("🔍 Direct Prediction Test")
    print("=" * 60)
    
    # Create brain directly
    factory = DynamicBrainFactory(config={'quiet_mode': False})
    
    # Create a brain with known dimensions
    brain_wrapper = factory.create(
        field_dimensions=26,
        spatial_resolution=8,
        sensory_dim=16,
        motor_dim=4
    )
    
    # Get the actual brain
    brain = brain_wrapper.brain
    print(f"\nBrain type: {type(brain).__name__}")
    print(f"Field dimensions: {brain.total_dimensions}D")
    print(f"Initial prediction confidence: {brain._current_prediction_confidence:.3f}")
    
    # Test with repeating pattern
    print("\n📊 Testing with repeating sine pattern...")
    print("-" * 60)
    
    pattern = [0.5 + 0.3 * np.sin(i * 0.5) for i in range(16)]
    
    for cycle in range(30):
        # Process directly
        action, brain_state = brain.process_robot_cycle(pattern)
        
        # Print detailed state every 5 cycles
        if cycle % 5 == 0:
            print(f"\nCycle {cycle}:")
            print(f"  Prediction confidence: {brain_state['prediction_confidence']:.3f}")
            print(f"  Field energy: {brain_state['field_energy']:.6f}")
            print(f"  Active constraints: {brain_state['active_constraints']}")
            print(f"  Cognitive mode: {brain_state['cognitive_mode']}")
            
            # Check internal state
            if hasattr(brain, '_predicted_field') and brain._predicted_field is not None:
                print(f"  Predicted field exists: Yes")
                print(f"  Last prediction error: {getattr(brain, '_last_prediction_error', 'N/A')}")
            else:
                print(f"  Predicted field exists: No")
            
            # Check prediction history
            if hasattr(brain, '_prediction_confidence_history'):
                history = list(brain._prediction_confidence_history)
                if len(history) > 0:
                    print(f"  Confidence history (last 5): {[f'{c:.3f}' for c in history[-5:]]}")
    
    # Final analysis
    print("\n" + "=" * 60)
    print("📊 FINAL ANALYSIS")
    print("=" * 60)
    
    print(f"\nFinal prediction confidence: {brain._current_prediction_confidence:.3f}")
    print(f"Total brain cycles: {brain.brain_cycles}")
    
    # Check if prediction improved
    if hasattr(brain, '_prediction_confidence_history') and len(brain._prediction_confidence_history) > 10:
        history = list(brain._prediction_confidence_history)
        early_avg = np.mean(history[:10])
        late_avg = np.mean(history[-10:])
        improvement = late_avg - early_avg
        
        print(f"\nConfidence improvement: {improvement:.3f}")
        print(f"Early average: {early_avg:.3f}")
        print(f"Late average: {late_avg:.3f}")
        
        if improvement > 0.05:
            print("✅ Prediction learning is working!")
        else:
            print("❌ No significant prediction improvement")
    
    # Test with changing pattern
    print("\n📊 Testing with changing pattern...")
    print("-" * 60)
    
    for cycle in range(10):
        # Change pattern each time
        pattern = [np.random.rand() for _ in range(16)]
        action, brain_state = brain.process_robot_cycle(pattern)
        print(f"Cycle {cycle}: confidence = {brain_state['prediction_confidence']:.3f}")
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_prediction_directly()