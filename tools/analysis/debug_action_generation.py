#!/usr/bin/env python3
"""
Debug action generation to understand why the brain appears lobotomized.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import torch
import numpy as np
import time

from brain_factory import BrainFactory


def main():
    print("🔍 Debugging Action Generation")
    print("=" * 50)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150,
            'field_evolution_rate': 0.1,  # Default value
            'gradient_following_strength': 0.2  # Default value
        }
    }
    
    # Create two brains with different settings to compare
    print("\n📊 Testing gradient following strength impact...")
    
    # Test 1: Default settings
    factory1 = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Give different sensory inputs
    actions1 = []
    for i in range(5):
        sensory_input = [1.0, 0.0, 0.0] + [0.0] * 13  # Strong X signal
        action, _ = factory1.process_sensory_input(sensory_input)
        actions1.append(action[:4])
    
    print(f"\nWith X=1.0 input:")
    for i, action in enumerate(actions1):
        print(f"   Action {i}: {action}")
    
    # Test with different input
    actions2 = []
    for i in range(5):
        sensory_input = [0.0, 1.0, 0.0] + [0.0] * 13  # Strong Y signal
        action, _ = factory1.process_sensory_input(sensory_input)
        actions2.append(action[:4])
    
    print(f"\nWith Y=1.0 input:")
    for i, action in enumerate(actions2):
        print(f"   Action {i}: {action}")
    
    # Check gradient following parameters
    print(f"\n📊 Brain Parameters:")
    print(f"   Evolution rate: {factory1.brain.field_evolution_rate}")
    print(f"   Gradient following: {factory1.brain.gradient_following_strength}")
    print(f"   Field decay: {factory1.brain.field_decay_rate}")
    print(f"   Field diffusion: {factory1.brain.field_diffusion_rate}")
    
    # Check action confidence
    print(f"\n📊 Action Generation Details:")
    # Run one more cycle and inspect
    sensory_input = [0.5, 0.5, 0.0] + [0.0] * 13
    action, state = factory1.process_sensory_input(sensory_input)
    
    print(f"   Last action: {action[:4]}")
    print(f"   Confidence: {state.get('last_action_confidence', 0):.4f}")
    print(f"   Brain cycles: {state.get('brain_cycles', 0)}")
    
    # Check if gradients are being used properly
    if hasattr(factory1.brain, 'gradient_flows'):
        print(f"\n📊 Gradient Utilization:")
        for name, grad in factory1.brain.gradient_flows.items():
            if grad is not None:
                # Get gradient values at robot position
                center = factory1.brain.spatial_resolution // 2
                local_grad = grad[center-1:center+2, center-1:center+2, center-1:center+2]
                max_grad = torch.max(torch.abs(local_grad)).item()
                mean_grad = torch.mean(torch.abs(local_grad)).item()
                print(f"   {name} at robot: max={max_grad:.6f}, mean={mean_grad:.6f}")
    
    factory1.shutdown()


if __name__ == "__main__":
    main()