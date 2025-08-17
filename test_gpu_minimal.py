#!/usr/bin/env python3
"""Test a minimal version of GPUFixedBrain process."""

import torch
import time
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'server', 'src'))

class MinimalGPUBrain:
    def __init__(self, spatial_size=96, channels=192):
        self.device = torch.device('cuda')
        self.spatial_size = spatial_size
        self.channels = channels
        self.sensory_dim = 16
        self.motor_dim = 5
        
        # Field
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        self.field_momentum = torch.zeros_like(self.field)
        
        # Pre-compute sensor injection
        self.sensor_spots = torch.randint(0, spatial_size, (16, 3), device=self.device)
        self.sensor_x = self.sensor_spots[:, 0]
        self.sensor_y = self.sensor_spots[:, 1]
        self.sensor_z = self.sensor_spots[:, 2]
        self.sensor_c = torch.arange(16, device=self.device) % 8
        
        self.sensor_flat_idx = (
            self.sensor_x * (spatial_size * spatial_size * channels) +
            self.sensor_y * (spatial_size * channels) +
            self.sensor_z * channels +
            self.sensor_c
        )
        
        self.cycle = 0
        
    def process(self, sensory_input):
        start = time.perf_counter()
        self.cycle += 1
        
        # Convert input
        sensors = torch.tensor(sensory_input[:16], dtype=torch.float32, device=self.device)
        
        # Sensory injection (vectorized)
        injection_values = sensors * 0.3
        field_flat = self.field.view(-1)
        field_flat.scatter_add_(0, self.sensor_flat_idx, injection_values)
        self.field = field_flat.view(self.spatial_size, self.spatial_size, 
                                     self.spatial_size, self.channels)
        
        # Simple dynamics
        self.field *= 0.995  # Decay
        self.field += torch.randn_like(self.field) * 0.001  # Noise
        
        # Momentum
        self.field_momentum = 0.9 * self.field_momentum + 0.1 * self.field
        self.field = self.field + self.field_momentum * 0.05
        
        # Simple motor extraction (no loop)
        motors = self.field.mean(dim=(0,1,2))[:self.motor_dim].cpu().tolist()
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return motors, {'time_ms': elapsed}

# Test
print("Creating minimal GPU brain...")
brain = MinimalGPUBrain(96, 192)

print("Testing process cycles...")
sensory_input = [0.5] * 16

times = []
for i in range(5):
    motors, telemetry = brain.process(sensory_input)
    torch.cuda.synchronize()
    times.append(telemetry['time_ms'])
    print(f"  Cycle {i+1}: {telemetry['time_ms']:.2f} ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.2f} ms")
print(f"Theoretical Hz: {1000/avg:.1f}")