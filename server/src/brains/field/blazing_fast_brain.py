"""
Blazing Fast Field Brain

The ultimate optimization: We recognize that for a 96³×192 tensor (169M parameters),
certain operations are fundamentally expensive. The solution is to be smarter about
WHEN and HOW we apply them.

Key insights:
1. torch.randn_like() on 169M parameters is expensive - use sparse noise
2. Many operations can be done less frequently
3. Some computations can be approximated
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class BlazingFastBrain:
    """
    Ultra-high-performance brain optimized for massive tensor fields.
    
    This version makes intelligent tradeoffs:
    - Sparse operations instead of dense
    - Temporal batching of expensive ops
    - Approximations that preserve behavior
    """
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_size: int = 16,
                 channels: int = 32,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """Initialize blazing fast brain."""
        
        self.quiet_mode = quiet_mode
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.spatial_size = spatial_size
        self.channels = channels
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if not quiet_mode:
            params = spatial_size ** 3 * channels
            print(f"⚡ Blazing Fast Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
        
        # Initialize field with small values
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        
        # Simplified momentum (smaller tensor)
        self.global_momentum = torch.zeros(channels, device=self.device)
        
        # Pre-compute random injection points for sensors
        self.sensor_spots = torch.randint(0, spatial_size, (sensory_dim, 3), device=self.device)
        
        # Pre-compute motor extraction points
        self.motor_spots = torch.randint(0, spatial_size, (motor_dim, 3), device=self.device)
        
        # Constants
        self.decay_rate = 0.995
        self.diffusion_rate = 0.1
        
        # Cycle counter for temporal operations
        self.cycle = 0
        
        if not quiet_mode:
            print("✅ Blazing fast brain initialized")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Ultra-fast processing cycle.
        
        Key optimizations:
        1. Sparse noise injection
        2. Simplified diffusion
        3. Direct motor extraction
        4. Temporal batching
        """
        start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if start:
            start.record()
        
        self.cycle += 1
        
        # Convert input (keep on GPU)
        sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                              dtype=torch.float32, device=self.device)
        
        # 1. SPARSE SENSORY INJECTION (fast)
        for i in range(min(len(sensors), self.sensory_dim)):
            x, y, z = self.sensor_spots[i]
            self.field[x, y, z, i % self.channels] += sensors[i] * 0.3
        
        # 2. SIMPLIFIED PHYSICS (every cycle)
        # Decay
        self.field *= self.decay_rate
        
        # Simple diffusion (only every 5 cycles to save computation)
        if self.cycle % 5 == 0:
            self.field = self._fast_diffusion(self.field)
        
        # 3. SPARSE NOISE (instead of dense noise)
        # Only add noise to ~1% of the field
        if self.cycle % 3 == 0:
            noise_mask = torch.rand_like(self.field[:, :, :, 0]) < 0.01
            noise_mask = noise_mask.unsqueeze(-1).expand_as(self.field)
            sparse_noise = torch.randn_like(self.field) * 0.01
            self.field = torch.where(noise_mask, self.field + sparse_noise, self.field)
        
        # 4. GLOBAL MOMENTUM (simplified)
        channel_means = self.field.mean(dim=(0, 1, 2))
        self.global_momentum = 0.9 * self.global_momentum + 0.1 * channel_means
        
        # Add momentum influence back (broadcasted efficiently)
        momentum_influence = self.global_momentum.view(1, 1, 1, -1) * 0.05
        self.field = self.field + momentum_influence
        
        # 5. ACTIVITY MAINTENANCE (simplified)
        if self.cycle % 10 == 0:
            activity = torch.abs(self.field).mean()
            if activity < 0.05:
                # Sparse energy injection
                energy_mask = torch.rand_like(self.field[:, :, :, 0]) < 0.05
                energy_mask = energy_mask.unsqueeze(-1).expand_as(self.field)
                energy = torch.randn_like(self.field) * 0.05
                self.field = torch.where(energy_mask, self.field + energy, self.field)
        
        # 6. ULTRA-FAST MOTOR EXTRACTION
        motors = self._extract_motors_blazing()
        
        # 7. Clamp field to prevent explosion
        self.field = torch.clamp(self.field, -10, 10)
        
        # Timing
        if end:
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
        else:
            time_ms = 0
        
        # Simple telemetry (avoid expensive computations)
        telemetry = {
            'cycle': self.cycle,
            'time_ms': time_ms,
            'energy': float(torch.abs(self.field).mean()),
            'motors': motors
        }
        
        return motors, telemetry
    
    def _fast_diffusion(self, field: torch.Tensor) -> torch.Tensor:
        """
        Simplified diffusion using average pooling.
        Much faster than computing full laplacian.
        """
        # Use average pooling as approximation of diffusion
        # This is MUCH faster than explicit neighbor computation
        kernel_size = 3
        padding = 1
        
        # Reshape for 3D pooling (batch, channel, depth, height, width)
        field_reshaped = field.permute(3, 0, 1, 2).unsqueeze(0)
        
        # Apply 3D average pooling
        pooled = torch.nn.functional.avg_pool3d(
            field_reshaped, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Reshape back
        pooled = pooled.squeeze(0).permute(1, 2, 3, 0)
        
        # Blend with original (diffusion effect)
        field = field * (1 - self.diffusion_rate) + pooled * self.diffusion_rate
        
        return field
    
    def _extract_motors_blazing(self) -> List[float]:
        """
        Blazing fast motor extraction.
        Just sample field values at motor points.
        """
        motors = []
        
        for i in range(self.motor_dim):
            x, y, z = self.motor_spots[i]
            
            # Just take the mean of channels at this point
            motor_val = self.field[x, y, z].mean()
            
            # Apply tanh to bound
            motor_cmd = torch.tanh(motor_val * 5).item()
            motors.append(motor_cmd)
        
        return motors
    
    def reset(self):
        """Reset brain state."""
        self.field = torch.randn(self.spatial_size, self.spatial_size, 
                                self.spatial_size, self.channels, device=self.device) * 0.01
        self.global_momentum.zero_()
        self.cycle = 0


# For compatibility
TrulyMinimalBrain = BlazingFastBrain
MinimalUnifiedBrain = BlazingFastBrain
UnifiedFieldBrain = BlazingFastBrain