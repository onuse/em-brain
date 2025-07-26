#!/usr/bin/env python3
"""
Debug the brain state to understand why it's not working properly.
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
    print("🔍 Debugging Brain State")
    print("=" * 50)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=False)
    
    print("\n📊 Initial Brain State:")
    print(f"   Spatial resolution: {factory.brain.spatial_resolution}")
    print(f"   Field shape: {factory.brain.unified_field.shape}")
    print(f"   Field min/max: {factory.brain.unified_field.min():.4f} / {factory.brain.unified_field.max():.4f}")
    print(f"   Field mean/std: {factory.brain.unified_field.mean():.4f} / {factory.brain.unified_field.std():.4f}")
    
    # Run a few cycles with input
    print("\n🔄 Running 10 cycles with sensory input...")
    for i in range(10):
        sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
        action, state = factory.process_sensory_input(sensory_input)
        print(f"   Cycle {i}: action={action[:4]}, confidence={state.get('last_action_confidence', 0):.4f}")
    
    print("\n📊 After 10 cycles:")
    print(f"   Field min/max: {factory.brain.unified_field.min():.4f} / {factory.brain.unified_field.max():.4f}")
    print(f"   Field mean/std: {factory.brain.unified_field.mean():.4f} / {factory.brain.unified_field.std():.4f}")
    print(f"   Non-zero elements: {(factory.brain.unified_field != 0).sum().item()} / {factory.brain.unified_field.numel()}")
    
    # Check gradient flows
    print("\n🌊 Gradient Flows:")
    if hasattr(factory.brain, 'gradient_flows'):
        for name, grad in factory.brain.gradient_flows.items():
            if grad is not None:
                print(f"   {name}: shape={grad.shape}, non-zero={(grad != 0).sum().item()}, max={torch.max(torch.abs(grad)).item():.6f}")
    
    # Check constraints
    print("\n🔗 Constraint System:")
    stats = factory.brain.constraint_field.get_constraint_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Check topology regions
    print(f"\n🗺️ Topology Regions: {len(factory.brain.topology_regions)}")
    for i, (key, region) in enumerate(list(factory.brain.topology_regions.items())[:3]):
        print(f"   Region {i}: activation={region['activation']:.4f}, position={region['field_indices'][:3]}")
    
    factory.shutdown()


if __name__ == "__main__":
    main()