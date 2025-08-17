#!/usr/bin/env python3
"""Debug the process method to find where it hangs."""

import sys
import os
import torch
import time

sys.path.append(os.path.join(os.getcwd(), 'server', 'src'))

# Monkey-patch the process method to add debugging
from brains.field.gpu_fixed_brain import GPUFixedBrain

original_process = GPUFixedBrain.process

def debug_process(self, sensory_input):
    print("DEBUG: Starting process method")
    start_time = time.perf_counter()
    self.cycle += 1
    
    print("DEBUG: Converting input...")
    sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                          dtype=torch.float32, device=self.device)
    
    print("DEBUG: Starting sensory injection...")
    # Vectorized injection
    injection_values = sensors * 0.3
    
    print("DEBUG: Flattening field...")
    field_flat = self.field.view(-1)
    
    print("DEBUG: Scatter add operation...")
    print(f"  Field flat shape: {field_flat.shape}")
    print(f"  Sensor flat idx shape: {self.sensor_flat_idx.shape}")
    print(f"  Injection values shape: {injection_values.shape}")
    print(f"  Max index: {self.sensor_flat_idx.max().item()}")
    print(f"  Field flat numel: {field_flat.numel()}")
    
    # This might be the problem line
    field_flat.scatter_add_(0, self.sensor_flat_idx[:len(sensors)], injection_values)
    
    print("DEBUG: Scatter complete, reshaping...")
    self.field = field_flat.view(self.spatial_size, self.spatial_size, 
                                 self.spatial_size, self.channels)
    
    print("DEBUG: Sensory injection complete!")
    
    # Return dummy values for now
    return [0.0] * self.motor_dim, {'time_ms': 0}

GPUFixedBrain.process = debug_process

# Now test
device = torch.device('cuda')
brain = GPUFixedBrain(16, 5, 96, 192, device, quiet_mode=True)

print('Testing modified process method...')
sensory_input = [0.5] * 16

motors, telemetry = brain.process(sensory_input)
print('Success!')