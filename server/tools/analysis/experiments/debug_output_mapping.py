#!/usr/bin/env python3
"""
Debug Output Mapping

Investigate the output mapping matrix multiplication that's zeroing out 
the amplified gradient actions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory
import torch

def debug_output_mapping():
    """
    Debug the output mapping matrix multiplication that's killing actions.
    """
    print("🔍 Output Mapping Debug")
    print("=" * 40)
    
    # Create field brain through BrainFactory
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4,
            'enable_enhanced_dynamics': True,
            'enable_attention_guidance': True,
            'enable_hierarchical_processing': True
        }
    }
    
    brain_factory = BrainFactory(config=config, quiet_mode=True)
    field_brain = brain_factory.vector_brain.field_brain
    field_impl = field_brain.field_impl
    
    # Process one input to get gradients
    sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    brain_factory.process_sensory_input(sensory_input)
    
    print(f"\n📊 Field Brain Configuration:")
    print(f"   Total dimensions: {field_brain.total_dimensions}")
    print(f"   Output mapping shape: {field_brain.output_mapping.shape}")
    print(f"   Stream capabilities: {field_brain.stream_capabilities.output_dimensions}")
    
    # Test manual output generation
    print(f"\n🎯 Manual Output Generation:")
    center_coords = torch.zeros(field_brain.total_dimensions, device=field_brain.field_device)
    raw_output = field_impl.generate_field_output(center_coords)
    
    print(f"   Raw output tensor: {[f'{x:.6f}' for x in raw_output.tolist()]}")
    print(f"   Raw output length: {len(raw_output)}")
    print(f"   Raw magnitude: {raw_output.abs().sum():.6f}")
    
    # Test the output mapping transformation
    print(f"\n🔬 Output Mapping Analysis:")
    output_mapping = field_brain.output_mapping
    print(f"   Output mapping shape: {output_mapping.shape}")
    print(f"   Raw output length: {len(raw_output)}")
    
    # Check which part of the mapping is used
    used_mapping = output_mapping[:len(raw_output), :]
    print(f"   Used mapping shape: {used_mapping.shape}")
    
    # Show some mapping values
    print(f"   Mapping matrix (first 4 rows):")
    for i in range(min(4, used_mapping.shape[0])):
        row = used_mapping[i, :].tolist()
        print(f"      Row {i}: {[f'{x:.4f}' for x in row]}")
    
    # Show mapping statistics
    mapping_magnitude = used_mapping.abs().sum()
    mapping_max = used_mapping.abs().max()
    mapping_mean = used_mapping.abs().mean()
    print(f"   Mapping stats: sum={mapping_magnitude:.4f}, max={mapping_max:.4f}, mean={mapping_mean:.4f}")
    
    # Perform the matrix multiplication step by step
    print(f"\n🧮 Matrix Multiplication Step-by-Step:")
    print(f"   Input vector: {[f'{x:.6f}' for x in raw_output.tolist()]}")
    
    mapped_output = torch.matmul(raw_output, used_mapping)
    print(f"   Mapped output: {[f'{x:.6f}' for x in mapped_output.tolist()]}")
    print(f"   Mapped magnitude: {mapped_output.abs().sum():.6f}")
    
    # Compare individual contributions
    print(f"\n🔍 Individual Contribution Analysis:")
    for i, input_val in enumerate(raw_output.tolist()):
        if abs(input_val) > 0.001:  # Only show significant inputs
            contribution = input_val * used_mapping[i, :]
            print(f"   Input[{i}] = {input_val:.6f} → {[f'{x:.6f}' for x in contribution.tolist()]}")
    
    # Check if the issue is size mismatch
    print(f"\n⚠️ Potential Issues:")
    if len(raw_output) != used_mapping.shape[0]:
        print(f"   ❌ Size mismatch: raw_output={len(raw_output)}, mapping_rows={used_mapping.shape[0]}")
    
    if mapping_magnitude < 0.1:
        print(f"   ❌ Mapping matrix has very small values (sum={mapping_magnitude:.6f})")
    
    if mapped_output.abs().sum() < 0.001:
        print(f"   ❌ Matrix multiplication produces near-zero output")
    
    # Test bypassing the mapping
    print(f"\n🚨 Bypass Test:")
    print(f"   Raw output (no mapping): {[f'{x:.6f}' for x in raw_output.tolist()]}")
    
    if len(raw_output) == 4:
        print(f"   ✅ Raw output already has correct dimensions for 4D motor!")
        print(f"   💡 SOLUTION: Could use raw output directly instead of mapping")
    
    return raw_output, mapped_output

if __name__ == "__main__":
    raw, mapped = debug_output_mapping()