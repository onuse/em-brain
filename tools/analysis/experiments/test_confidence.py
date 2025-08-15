#!/usr/bin/env python3
"""
Test to verify brain confidence is being updated properly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'server'))

import time
import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory
from src.parameters.cognitive_config import get_cognitive_config

def test_brain_confidence():
    """Test that brain confidence is being updated properly."""
    print("\n=== Testing Brain Confidence Update ===\n")
    
    # Create brain
    factory = SimplifiedBrainFactory({'quiet_mode': True})
    brain = factory.create(sensory_dim=16, motor_dim=5)
    
    # Check initial state
    initial_state = brain._create_brain_state()
    print(f"Initial confidence: {initial_state['prediction_confidence']:.3f}")
    print(f"Predictive phases enabled: {initial_state.get('predictive_phases', {})}")
    
    # Process some cycles with varying sensory input
    for i in range(10):
        # Create sensory input with some variation
        sensory_input = [0.5 + 0.1 * torch.sin(torch.tensor(i * 0.5)).item() for _ in range(16)]
        
        # Process cycle
        motor_output = brain.process_field_dynamics(sensory_input)
        
        # Get state
        state = brain._create_brain_state()
        
        # Print available keys first
        if i == 0:
            print(f"  Available state keys: {list(state.keys())}")
        
        confidence = state.get('prediction_confidence', 0.0)
        energy = state.get('field_energy', 0.0)
        
        print(f"Cycle {i+1}: confidence={confidence:.3f}, energy={energy:.3f}")
        
        # Check internal brain state
        if hasattr(brain, 'brain') and hasattr(brain.brain, '_current_prediction_confidence'):
            internal_conf = brain.brain._current_prediction_confidence
            print(f"  Internal confidence: {internal_conf:.3f}")
            
            # Check if predictive processing is enabled
            if hasattr(brain.brain, 'enable_phase_3'):
                print(f"  Phase 3 enabled: {brain.brain.enable_phase_3}")
                print(f"  Phase 4 enabled: {brain.brain.enable_phase_4}")
                print(f"  Phase 5 enabled: {brain.brain.enable_phase_5}")
    
    # Final check
    final_state = brain._create_brain_state()
    print(f"\nFinal confidence: {final_state['prediction_confidence']:.3f}")
    
    # Test with more cycles to see if confidence changes
    print("\n=== Running 100 cycles ===")
    for i in range(100):
        sensory_input = [0.5 + 0.2 * torch.sin(torch.tensor(i * 0.1)).item() for _ in range(16)]
        motor_output = brain.process_field_dynamics(sensory_input)
        
        if i % 20 == 0:
            state = brain._create_brain_state()
            print(f"Cycle {i}: confidence={state['prediction_confidence']:.3f}")
    
    return brain

if __name__ == "__main__":
    brain = test_brain_confidence()
    print("\nTest complete!")