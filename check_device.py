#!/usr/bin/env python3
"""Check if brain is actually using GPU."""

import torch
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

# Create brain
brain = TrulyMinimalBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_size=96,
    channels=192,
    device=torch.device('cuda'),
    quiet_mode=False
)

print(f"\nChecking tensor devices:")
print(f"  field device: {brain.field.device}")
print(f"  field_momentum device: {brain.field_momentum.device}")

if hasattr(brain.dynamics, 'noise_scale'):
    print(f"  dynamics device: CPU operations")
    
# Check if operations stay on GPU
sensors = [0.5] * 24
print(f"\nSensor input type: {type(sensors)}")

# Check what happens in process
import time
start = time.time()

# Convert sensors to tensor
sensors_tensor = torch.tensor(sensors[:brain.sensory_dim], 
                              dtype=torch.float32, device=brain.device)
print(f"  sensors_tensor device: {sensors_tensor.device}")

# Do one operation
field_mean = brain.field.mean()
print(f"  field.mean() device: {field_mean.device}")

# Check if .item() is being called a lot
print(f"\nChecking .item() calls...")
start = time.time()
for i in range(100):
    val = field_mean.item()  # Forces GPU sync
elapsed = time.time() - start
print(f"  100x .item() calls: {elapsed*1000:.1f}ms")

# This could be the issue!
print(f"\nChecking tensor creation in loops...")
start = time.time()
for i in range(100):
    # This happens in the brain code
    x = torch.tensor(0.5, device=brain.device)
elapsed = time.time() - start
print(f"  100x tensor creation: {elapsed*1000:.1f}ms")