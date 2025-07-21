#!/usr/bin/env python3
"""
Debug Gradient Generation

Debug why field gradients are not generating meaningful actions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter

def debug_gradient_generation():
    """
    Debug field gradient generation step by step.
    """
    print("🔍 Field Gradient Generation Debug")
    print("=" * 40)
    
    # Create TCP adapter with debugging
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=8,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=False
    )
    
    # Access the field brain internals
    field_brain = adapter.field_brain
    field_impl = field_brain.field_impl
    
    print(f"\n🧠 Field Implementation: {type(field_impl).__name__}")
    print(f"   Field device: {field_impl.field_device}")
    print(f"   Field tensors: {len(field_impl.field_tensors)} tensors")
    
    for name, tensor in field_impl.field_tensors.items():
        print(f"   - {name}: {tensor.shape} on {tensor.device}")
    
    # Process one input and check gradients
    sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\n📊 Processing sensory input: {sensory_input}")
    
    # Process through adapter
    action, brain_state = adapter.process_sensory_input(sensory_input)
    
    print(f"   Resulting action: {[f'{x:.6f}' for x in action]}")
    print(f"   Action magnitude: {sum(abs(x) for x in action):.6f}")
    
    # Check gradient flows
    print(f"\n🌊 Gradient flows:")
    if hasattr(field_impl, 'gradient_flows') and field_impl.gradient_flows:
        for name, grad_tensor in field_impl.gradient_flows.items():
            if isinstance(grad_tensor, dict):
                print(f"   {name}: {type(grad_tensor)} (nested)")
            else:
                magnitude = float(grad_tensor.abs().max()) if hasattr(grad_tensor, 'abs') else 0.0
                mean_val = float(grad_tensor.mean()) if hasattr(grad_tensor, 'mean') else 0.0
                print(f"   {name}: shape={grad_tensor.shape}, max_mag={magnitude:.6f}, mean={mean_val:.6f}")
    else:
        print("   ❌ No gradient flows found!")
    
    # Check field statistics
    print(f"\n📈 Field statistics:")
    field_stats = field_impl.get_field_statistics()
    for key, value in field_stats.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Try to manually call output generation
    print(f"\n🎯 Manual output generation test:")
    try:
        # Create dummy field coordinates
        import torch
        dummy_coords = torch.zeros(4, device=field_impl.field_device)
        manual_output = field_impl.generate_field_output(dummy_coords)
        print(f"   Manual output: {[f'{x:.6f}' for x in manual_output.tolist()]}")
        print(f"   Manual magnitude: {manual_output.abs().sum():.6f}")
    except Exception as e:
        print(f"   ❌ Manual output generation failed: {e}")
    
    return action

if __name__ == "__main__":
    debug_gradient_generation()