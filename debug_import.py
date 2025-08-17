#!/usr/bin/env python3
"""Debug which version is being used."""

import torch
import time

# First check the imports
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain
from server.src.brains.field import simple_motor

# Check if extract_motors has been updated
import inspect
source = inspect.getsource(simple_motor.SimpleMotorExtraction.extract_motors)
print("Checking motor extraction implementation...")
if "torch.clamp" in source and "motor_commands.cpu().tolist()" in source:
    print("✅ Using optimized motor extraction")
else:
    print("❌ Still using old motor extraction")
    
# Now test actual performance
print("\nTesting actual motor extraction performance...")
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=96,
    channels=192,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    quiet_mode=True
)

# Get a field tensor
field = brain.field

# Time motor extraction directly
print(f"Field shape: {field.shape}")
print(f"Device: {field.device}")

for i in range(3):
    start = time.perf_counter()
    motors = brain.motor.extract_motors(field)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Motor extraction {i+1}: {elapsed:.1f}ms")

print("\nDone!")