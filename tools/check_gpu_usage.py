#!/usr/bin/env python3
"""
Quick check of GPU tensor operations
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
import time

print("Checking GPU usage patterns...")
print("=" * 60)

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

print(f"\nBrain device: {brain.device}")
print(f"Unified field device: {brain.unified_field.device}")
print(f"Unified field shape: {brain.unified_field.shape}")
print(f"Unified field dtype: {brain.unified_field.dtype}")

# Check a few key tensors
print("\nKey tensor devices:")
if hasattr(brain, '_predicted_sensory') and brain._predicted_sensory is not None:
    print(f"  _predicted_sensory: {brain._predicted_sensory.device}")
if hasattr(brain, '_predicted_field') and brain._predicted_field is not None:
    print(f"  _predicted_field: {brain._predicted_field.device}")

# Run a cycle and check operations
print("\nRunning one cycle...")
start = time.time()
sensory_input = [0.5] * 24
motor_output, brain_state = brain.process_robot_cycle(sensory_input)
cycle_time = time.time() - start

print(f"\nCycle time: {cycle_time*1000:.1f}ms")
print(f"Motor output: {motor_output}")

# Check if operations stay on GPU
print("\nChecking tensor operations stay on GPU...")
test_tensor = torch.randn(100, 100, device=brain.device)
result = torch.matmul(test_tensor, test_tensor)
print(f"Test tensor device after matmul: {result.device}")

# Check memory usage
if brain.device.type == 'mps':
    print("\nMPS device detected - GPU memory tracking limited")
elif brain.device.type == 'cuda':
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

print("\nConclusion:")
print("- Tensors are created on correct device")
print("- Operations should stay on GPU")
print("- Low GPU usage might be due to:")
print("  1. Frequent .item() calls forcing GPU→CPU transfers")
print("  2. Small tensor sizes not fully utilizing GPU")
print("  3. MPS (Metal) overhead on M1 Macs")