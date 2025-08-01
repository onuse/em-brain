#!/usr/bin/env python3
"""
Debug confidence calculation issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'server'))

import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory

def test_confidence_directly():
    """Test confidence calculation directly on the brain."""
    print("\n=== Testing Confidence Directly ===\n")
    
    # Create brain with debug output
    factory = SimplifiedBrainFactory({'quiet_mode': False})
    wrapper = factory.create(sensory_dim=16, motor_dim=5)
    brain = wrapper.brain  # Get the actual brain instance
    
    print(f"\nBrain type: {type(brain)}")
    print(f"Initial confidence: {brain._current_prediction_confidence}")
    print(f"Predictive phases status (check methods exist):")
    print(f"  Has enable_hierarchical_prediction: {hasattr(brain, 'enable_hierarchical_prediction')}")
    print(f"  Has enable_action_prediction: {hasattr(brain, 'enable_action_prediction')}")
    print(f"  Has enable_active_vision: {hasattr(brain, 'enable_active_vision')}")
    
    # Process a few cycles to see confidence updates
    print("\n=== Processing cycles ===")
    for i in range(5):
        # Varying sensory input
        sensory = [0.5 + 0.1 * torch.sin(torch.tensor(i * 0.5)).item() for _ in range(16)]
        
        # Call process_robot_cycle directly
        action, brain_state = brain.process_robot_cycle(sensory)
        
        print(f"\nCycle {i+1}:")
        print(f"  Current confidence: {brain._current_prediction_confidence:.3f}")
        print(f"  Last prediction error: {getattr(brain, '_last_prediction_error', 'N/A')}")
        print(f"  Predicted field exists: {brain._predicted_field is not None}")
        print(f"  Predicted sensory exists: {brain._predicted_sensory is not None}")
        print(f"  Brain state confidence: {brain_state.get('prediction_confidence', 'N/A')}")
        
        # Check field dynamics confidence
        if hasattr(brain.field_dynamics, 'smoothed_confidence'):
            print(f"  Field dynamics confidence: {brain.field_dynamics.smoothed_confidence:.3f}")
    
    # Force enable debug mode and run more cycles
    print("\n=== Forcing debug output (every 10 cycles) ===")
    old_quiet = brain.quiet_mode
    brain.quiet_mode = False
    
    for i in range(20):
        sensory = [0.5 + 0.2 * torch.sin(torch.tensor(i * 0.1)).item() for _ in range(16)]
        action, brain_state = brain.process_robot_cycle(sensory)
        
        if i % 5 == 0:
            print(f"Cycle {brain.brain_cycles}: confidence={brain._current_prediction_confidence:.3f}")
    
    brain.quiet_mode = old_quiet

if __name__ == "__main__":
    test_confidence_directly()