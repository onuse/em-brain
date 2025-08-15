#!/usr/bin/env python3
"""
Profile GPU->CPU transfers in brain cycle
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

# Monkey patch torch operations to count transfers
import torch
original_item = torch.Tensor.item
original_cpu = torch.Tensor.cpu
original_numpy = torch.Tensor.numpy
original_detach = torch.Tensor.detach

transfer_count = {'item': 0, 'cpu': 0, 'numpy': 0, 'detach': 0}

def tracked_item(self):
    transfer_count['item'] += 1
    return original_item(self)

def tracked_cpu(self):
    transfer_count['cpu'] += 1
    return original_cpu(self)

def tracked_numpy(self):
    transfer_count['numpy'] += 1
    return original_numpy(self)

def tracked_detach(self):
    transfer_count['detach'] += 1
    return original_detach(self)

torch.Tensor.item = tracked_item
torch.Tensor.cpu = tracked_cpu
torch.Tensor.numpy = tracked_numpy
torch.Tensor.detach = tracked_detach

from server.src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Profiling GPU->CPU transfers per brain cycle...")
print("=" * 60)

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Run a few cycles
sensory_input = [0.5] * 24
for i in range(5):
    transfer_count = {'item': 0, 'cpu': 0, 'numpy': 0, 'detach': 0}
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    print(f"\nCycle {i+1} transfers:")
    print(f"  .item() calls: {transfer_count['item']}")
    print(f"  .cpu() calls: {transfer_count['cpu']}")
    print(f"  .numpy() calls: {transfer_count['numpy']}")
    print(f"  .detach() calls: {transfer_count['detach']}")
    total = sum(transfer_count.values())
    print(f"  TOTAL: {total} GPU->CPU operations")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("Each brain cycle forces multiple GPU->CPU synchronizations")
print("This prevents the GPU from running efficiently in parallel")
print("For deployment on faster hardware, consider batching operations")