#!/usr/bin/env python3
"""
Memory-Optimized PureFieldBrain

Key optimizations:
1. Pre-allocated tensors for all intermediate computations
2. In-place operations where possible
3. Reuse of tensor buffers
4. No functionality reduction - all emergence preserved!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging

# Import the original to extend it
from .pure_field_brain import (
    PureFieldBrain, HierarchicalField, ScaleConfig, SCALE_CONFIGS
)


class MemoryOptimizedField(HierarchicalField):
    """
    Memory-efficient version of HierarchicalField with pre-allocated buffers.
    """
    
    def __init__(self, field_size: int, channels: int, meta_channels: int, device: str = 'cuda'):
        super().__init__(field_size, channels, meta_channels, device)
        
        # Pre-allocate working buffers
        self.field_buffer = torch.zeros_like(self.field)
        self.grad_buffer = torch.zeros_like(self.field)
        
        # Pre-allocate convolution buffers
        self.conv_input_buffer = torch.zeros(
            1, channels, field_size, field_size, field_size,
            device=device, dtype=torch.float32
        )
        self.conv_output_buffer = torch.zeros_like(self.conv_input_buffer)
        
        # Pre-allocate diffusion kernel (reusable)
        self.blur_kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
    
    def evolve(self, diffusion_rate: float, decay_rate: float, 
               noise_scale: float, meta_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient evolution using pre-allocated buffers."""
        
        # Use pre-allocated buffer for reshaping
        self.conv_input_buffer.zero_()  # Clear buffer
        self.conv_input_buffer[0] = self.field.permute(3, 0, 1, 2)
        
        # Apply evolution kernel (in-place when possible)
        kernel = self.evolution_kernel
        if meta_modulation is not None and self.meta_kernel is not None:
            # Modulate kernel in-place
            kernel = kernel * (1.0 + meta_modulation.mean() * 0.1)
        
        # Convolution into output buffer
        torch.nn.functional.conv3d(
            self.conv_input_buffer,
            kernel,
            padding=1,
            groups=1,
            out=self.conv_output_buffer
        )
        
        # Apply diffusion in-place
        if diffusion_rate > 0:
            for c in range(self.channels):
                # Reuse buffer for each channel
                diffused = F.conv3d(
                    self.conv_output_buffer[:, c:c+1],
                    self.blur_kernel,
                    padding=1
                )
                # In-place blend
                self.conv_output_buffer[:, c:c+1].mul_(1 - diffusion_rate).add_(
                    diffused, alpha=diffusion_rate
                )
        
        # Apply decay in-place
        self.conv_output_buffer.mul_(decay_rate)
        
        # Add noise in-place if needed
        if noise_scale > 0:
            if self.meta_channels > 0:
                meta_state = self.field[:, :, :, -self.meta_channels:].mean()
                noise_scale = noise_scale * (1.0 + meta_state * 0.5)
            # Add noise in-place
            self.conv_output_buffer.add_(
                torch.randn_like(self.conv_output_buffer), alpha=noise_scale
            )
        
        # Copy result back to field buffer and reshape
        self.field_buffer.copy_(self.conv_output_buffer.squeeze(0).permute(1, 2, 3, 0))
        return self.field_buffer


class MemoryOptimizedPureFieldBrain(PureFieldBrain):
    """
    Memory-optimized version of PureFieldBrain.
    
    All functionality preserved, but with:
    - Pre-allocated tensors
    - In-place operations
    - Buffer reuse
    - Reduced memory fragmentation
    """
    
    def __init__(self, input_dim: int = 10, output_dim: int = 4,
                 scale_config: Optional[ScaleConfig] = None,
                 device: str = None, aggressive: bool = True):
        
        # Initialize parent (but we'll replace the levels)
        super().__init__(input_dim, output_dim, scale_config, device, aggressive)
        
        # Replace levels with memory-optimized versions
        self.levels = nn.ModuleList()
        for field_size, channels in self.scale_config.levels:
            level = MemoryOptimizedField(
                field_size=field_size,
                channels=channels,
                meta_channels=self.scale_config.meta_channels,
                device=self.device
            )
            self.levels.append(level)
        
        # Pre-allocate common buffers
        self._init_buffers()
        
        # Log optimization
        self.logger.info("🚀 Memory-optimized PureFieldBrain initialized")
    
    def _init_buffers(self):
        """Pre-allocate all reusable buffers."""
        
        # Sensory injection buffers
        first_level = self.levels[0]
        self.sensory_buffer = torch.zeros(
            first_level.channels, device=self.device
        )
        self.field_perturbation_buffer = torch.zeros_like(first_level.field)
        
        # Motor extraction buffers
        max_channels = max(level.channels for level in self.levels)
        self.gradient_buffer = torch.zeros(max_channels, device=self.device)
        
        # Cross-scale buffers (if multi-level)
        if len(self.levels) > 1:
            # Pre-allocate interpolation buffers for each level pair
            self.upsample_buffers = []
            self.downsample_buffers = []
            
            for i in range(len(self.levels) - 1):
                size1 = self.scale_config.levels[i][0]
                size2 = self.scale_config.levels[i+1][0]
                channels1 = self.scale_config.levels[i][1]
                channels2 = self.scale_config.levels[i+1][1]
                
                # Upsample buffer (coarse to fine)
                if size2 < size1:
                    self.upsample_buffers.append(
                        torch.zeros(1, channels2, size1, size1, size1, device=self.device)
                    )
                else:
                    self.upsample_buffers.append(None)
                
                # Downsample buffer (fine to coarse)
                if size1 > size2:
                    self.downsample_buffers.append(
                        torch.zeros(1, channels1, size2, size2, size2, device=self.device)
                    )
                else:
                    self.downsample_buffers.append(None)
    
    def forward(self, sensory_input: torch.Tensor, reward: float = 0.0) -> torch.Tensor:
        """
        Memory-efficient forward pass.
        
        Functionality identical to original, but uses pre-allocated buffers.
        """
        self.cycle_count += 1
        self.brain_cycles += 1
        self.experience_count += 1
        
        # Ensure input is on device and correct shape
        if not torch.is_tensor(sensory_input):
            sensory_input = torch.tensor(sensory_input, device=self.device, dtype=torch.float32)
        else:
            sensory_input = sensory_input.to(self.device).float()
        
        if sensory_input.dim() == 0:
            sensory_input = sensory_input.unsqueeze(0)
        
        # SAFETY: Sanitize input in-place
        if torch.isnan(sensory_input).any() or torch.isinf(sensory_input).any():
            self.logger.warning("NaN/Inf detected in sensory input, replacing with zeros")
            torch.nan_to_num_(sensory_input, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp inputs in-place
        sensory_input.clamp_(min=-10.0, max=10.0)
        
        # Handle variable input dimensions
        actual_input_dim = sensory_input.shape[0]
        if actual_input_dim != self.input_dim:
            first_channels = self.scale_config.levels[0][1]
            if not hasattr(self, '_dynamic_input_projection') or self._last_input_dim != actual_input_dim:
                self._dynamic_input_projection = nn.Linear(actual_input_dim, first_channels, device=self.device, bias=False)
                self._last_input_dim = actual_input_dim
        
        # Use autocast for mixed precision
        with torch.amp.autocast(device_type=self.device, enabled=(self.device == 'cuda')):
            
            # 1. Inject sensory (uses pre-allocated buffers internally)
            sensory_field = self._inject_sensory_hierarchical(sensory_input)
            
            # 2. Hierarchical evolution (memory-efficient)
            self._evolve_hierarchical(reward)
            
            # 3. Add sensory influence in-place
            self.levels[0].field.add_(sensory_field, alpha=self.learning_rate)
            
            # 4. Apply nonlinearity in-place
            for level in self.levels:
                level.field.tanh_()  # In-place tanh
            
            # Update compatibility references
            self.field = self.levels[0].field
            self.unified_field = self.field
            
            # 5. Extract motor
            motor_output = self._extract_motor_hierarchical()
            
            # 6. Update metrics (only periodically)
            if self.cycle_count % 10 == 0:
                self._update_emergence_metrics()
                self._update_practical_metrics(sensory_input, motor_output)
        
        return motor_output
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2 if self.device == 'cuda' else 0,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2 if self.device == 'cuda' else 0,
            'num_levels': len(self.levels),
            'total_parameters': self.scale_config.total_params,
            'buffer_count': len(self.upsample_buffers) + len(self.downsample_buffers) if hasattr(self, 'upsample_buffers') else 0
        }
        return stats


def create_memory_optimized_brain(
    input_dim: int = 10,
    output_dim: int = 4,
    size: str = 'medium',
    aggressive: bool = True,
    device: str = None
) -> MemoryOptimizedPureFieldBrain:
    """Create a memory-optimized brain."""
    
    if size not in SCALE_CONFIGS:
        raise ValueError(f"Size must be one of {list(SCALE_CONFIGS.keys())}")
    
    scale_config = SCALE_CONFIGS[size]
    
    return MemoryOptimizedPureFieldBrain(
        input_dim=input_dim,
        output_dim=output_dim,
        scale_config=scale_config,
        device=device,
        aggressive=aggressive
    )


if __name__ == "__main__":
    print("🧠 Testing Memory-Optimized PureFieldBrain")
    print("=" * 60)
    
    # Create optimized brain
    brain = create_memory_optimized_brain(
        input_dim=16,  # Robot sensors
        output_dim=4,   # Motor channels
        size='medium',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nBrain created: {brain}")
    
    # Test memory usage
    initial_stats = brain.get_memory_stats()
    print(f"\nInitial memory stats:")
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")
    
    # Run some cycles
    print(f"\nRunning 100 cycles...")
    for i in range(100):
        sensory = torch.randn(16, device=brain.device)
        motor = brain(sensory)
    
    # Check memory again
    final_stats = brain.get_memory_stats()
    print(f"\nFinal memory stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Check for memory growth
    if brain.device == 'cuda':
        memory_growth = final_stats['allocated_mb'] - initial_stats['allocated_mb']
        print(f"\nMemory growth: {memory_growth:.2f} MB")
        if memory_growth < 1.0:
            print("✅ Excellent! Minimal memory growth")
        else:
            print("⚠️  Some memory growth detected")
    
    print("\n✅ Memory optimization test complete!")