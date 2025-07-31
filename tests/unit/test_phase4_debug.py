#!/usr/bin/env python3
"""
Debug Phase 4 to understand what's happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

# Import brain components
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def debug_action_generation():
    """Debug action generation process."""
    print("\n=== Debugging Action Generation ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=2,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("Running 10 cycles with debug info...\n")
    
    for cycle in range(10):
        sensory_input = [0.5, -0.5, 0.0]  # reward = 0
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        if cycle % 3 == 0:
            print(f"Cycle {cycle}:")
            print(f"  Motor output: {motor_output}")
            
            # Check action prediction stats
            if 'action_prediction' in brain_state:
                stats = brain_state['action_prediction']
                print(f"  Total actions: {stats['total_actions']}")
                print(f"  Action types: {stats.get('action_types', {})}")
                if 'prediction_accuracy' in stats:
                    print(f"  Prediction accuracy: {stats['prediction_accuracy']}")
            
            # Check if last action was stored
            if hasattr(brain, '_last_predicted_action') and brain._last_predicted_action:
                action = brain._last_predicted_action
                print(f"  Last action type: {action.action_type}")
                print(f"  Uncertainty reduction: {action.uncertainty_reduction:.3f}")
                print(f"  Immediate confidence: {action.immediate_confidence:.3f}")


if __name__ == "__main__":
    debug_action_generation()