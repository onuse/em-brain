#!/usr/bin/env python3
"""
Debug script to trace tensor dimension issues
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
import torch

def debug_tensor_dimensions():
    """Debug the tensor dimension flow."""
    print("🔍 DEBUGGING TENSOR DIMENSIONS")
    print("=" * 40)
    
    # Create brain
    brain = MinimalBrain(quiet_mode=True)
    print(f"Brain sensory_dim: {brain.sensory_dim}")
    print(f"Brain motor_dim: {brain.motor_dim}")
    print(f"Brain temporal_dim: {brain.temporal_dim}")
    
    # Test input
    test_input = [1.0, 2.0, 3.0, 4.0]
    print(f"\nOriginal input: {test_input} (length: {len(test_input)})")
    
    # Check what happens in process_sensory_input
    if len(test_input) > brain.sensory_dim:
        processed_input = test_input[:brain.sensory_dim]
        print(f"Truncated to: {processed_input} (length: {len(processed_input)})")
    elif len(test_input) < brain.sensory_dim:
        processed_input = test_input + [0.0] * (brain.sensory_dim - len(test_input))
        print(f"Padded to: {processed_input} (length: {len(processed_input)})")
    else:
        processed_input = test_input
        print(f"Same size: {processed_input} (length: {len(processed_input)})")
    
    # Check vector brain configuration
    vector_brain = brain.vector_brain
    print(f"\nVector brain sensory config dim: {vector_brain.sensory_config.dim}")
    print(f"Vector brain motor config dim: {vector_brain.motor_config.dim}")
    print(f"Vector brain temporal config dim: {vector_brain.temporal_config.dim}")
    
    # Check sensory stream configuration
    sensory_stream = vector_brain.sensory_stream
    print(f"\nSensory stream config dim: {sensory_stream.config.dim}")
    print(f"Sensory stream buffer shape: {sensory_stream.activation_buffer.shape}")
    
    # Test tensor conversion
    sensory_tensor = torch.tensor(processed_input, dtype=torch.float32)
    print(f"\nSensory tensor shape: {sensory_tensor.shape}")
    print(f"Sensory tensor: {sensory_tensor}")
    
    # Check if tensor fits in buffer
    if sensory_tensor.shape[0] == sensory_stream.activation_buffer.shape[1]:
        print("✅ Tensor dimensions match buffer")
    else:
        print(f"❌ Tensor dimensions mismatch!")
        print(f"   Tensor shape: {sensory_tensor.shape}")
        print(f"   Buffer shape: {sensory_stream.activation_buffer.shape}")
        print(f"   Expected: tensor should be ({sensory_stream.activation_buffer.shape[1]},)")
    
    # Try to reproduce the error
    try:
        print(f"\nTrying to call sensory_stream.update...")
        import time
        result = sensory_stream.update(sensory_tensor, time.time())
        print(f"✅ Success: {result.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   This is the dimension mismatch we're investigating")

if __name__ == "__main__":
    debug_tensor_dimensions()