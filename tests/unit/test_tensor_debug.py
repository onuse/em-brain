#!/usr/bin/env python3
"""Debug tensor size mismatch error."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

import torch
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Debugging tensor size mismatch...\n")

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=23,  # 23 sensors (reward will be added as 24th element)
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=False  # Enable debug output
)

# Run cycles with detailed logging
for i in range(5):
    print(f"\n{'='*50}")
    print(f"CYCLE {i}")
    print('='*50)
    
    sensory_input = [0.5] * 24  # 24 total values
    print(f"Input size: {len(sensory_input)}")
    print(f"Brain sensory_dim: {brain.sensory_dim}")
    
    try:
        # Add debug hook
        original_process_errors = brain.field_dynamics.process_prediction_errors
        def debug_process_errors(prediction_errors, topology_regions, current_field):
            print(f"\n[DEBUG] process_prediction_errors called:")
            print(f"  - prediction_errors type: {type(prediction_errors)}")
            if torch.is_tensor(prediction_errors):
                print(f"  - prediction_errors shape: {prediction_errors.shape}")
                print(f"  - prediction_errors device: {prediction_errors.device}")
            else:
                print(f"  - prediction_errors value: {prediction_errors}")
            return original_process_errors(prediction_errors, topology_regions, current_field)
        
        brain.field_dynamics.process_prediction_errors = debug_process_errors
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        print(f"✅ Success! Motor output: {len(motor_output)} values")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Print tensor sizes at error point
        if hasattr(brain, '_predicted_sensory') and brain._predicted_sensory is not None:
            print(f"\n_predicted_sensory shape: {brain._predicted_sensory.shape}")
        if hasattr(brain, '_last_prediction_error') and brain._last_prediction_error is not None:
            if torch.is_tensor(brain._last_prediction_error):
                print(f"_last_prediction_error shape: {brain._last_prediction_error.shape}")
        
        break