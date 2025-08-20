"""
Simple Field Dynamics

Just the physics - diffusion, decay, and bounded noise.
No patterns, no waves, no complex features. Just physics.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class SimpleFieldDynamics:
    """
    Minimal field evolution using simple physics.
    
    Three basic operations:
    1. Decay - field gradually returns to rest
    2. Diffusion - information spreads locally
    3. Noise - prevents complete stasis
    """
    
    def __init__(self, decay_rate: float = 0.995, diffusion_rate: float = 0.1, noise_scale: float = 0.001):
        """
        Initialize dynamics parameters.
        
        Args:
            decay_rate: How fast field decays (0.99 = slow, 0.9 = fast)
            diffusion_rate: How much information spreads (0.1 = moderate)
            noise_scale: Background noise level (0.001 = subtle)
        """
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.noise_scale = noise_scale
        self._laplacian_kernel = None  # Will be created on first use
        
    def evolve(self, field: torch.Tensor, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolve field one step using simple physics.
        
        Args:
            field: Current field state [D, H, W, C]
            external_input: Optional external influence
            
        Returns:
            Updated field
        """
        # 1. DECAY - everything trends toward zero
        field = field * self.decay_rate
        
        # 2. DIFFUSION - local spreading (simple 3D convolution)
        if self.diffusion_rate > 0:
            field = self._apply_diffusion(field)
        
        # 3. NOISE - prevent complete silence
        noise = torch.randn_like(field) * self.noise_scale
        field = field + noise
        
        # 4. EXTERNAL INPUT (if any)
        if external_input is not None:
            field = field + external_input
        
        # 5. BOUND - prevent explosion
        field = torch.clamp(field, -10, 10)
        
        return field
    
    def _apply_diffusion(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply simple diffusion using nearest-neighbor averaging.
        GPU-optimized using conv3d for better performance.
        """
        # Try GPU-optimized version first
        if hasattr(F, 'conv3d'):
            return self._apply_diffusion_gpu_optimized(field)
        else:
            # Fallback to original implementation
            return self._apply_diffusion_original(field)
    
    def _apply_diffusion_gpu_optimized(self, field: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized diffusion using 3D convolution.
        Produces identical results to the original but much faster.
        """
        # Create Laplacian kernel on first use (cached)
        if self._laplacian_kernel is None or self._laplacian_kernel.device != field.device:
            # 3D Laplacian kernel for discrete approximation
            kernel = torch.zeros(1, 1, 3, 3, 3, device=field.device, dtype=field.dtype)
            kernel[0, 0, 1, 1, 1] = -6.0  # center
            kernel[0, 0, 0, 1, 1] = 1.0   # -x
            kernel[0, 0, 2, 1, 1] = 1.0   # +x
            kernel[0, 0, 1, 0, 1] = 1.0   # -y
            kernel[0, 0, 1, 2, 1] = 1.0   # +y
            kernel[0, 0, 1, 1, 0] = 1.0   # -z
            kernel[0, 0, 1, 1, 2] = 1.0   # +z
            self._laplacian_kernel = kernel
        
        # For very large fields, use the same subsampling strategy
        if field.shape[0] > 64:
            # Subsample for diffusion calculation (4x faster)
            stride = 2
            subsampled = field[::stride, ::stride, ::stride, :]
            
            # Reshape for conv3d: [D,H,W,C] -> [1,C,D,H,W]
            C = subsampled.shape[3]
            field_conv = subsampled.permute(3, 0, 1, 2).unsqueeze(0)
            
            # Expand kernel for all channels (depthwise convolution)
            kernel_expanded = self._laplacian_kernel.expand(C, 1, -1, -1, -1)
            
            # Apply 3D convolution with proper padding
            laplacian_sub = F.conv3d(field_conv, kernel_expanded, padding=1, groups=C)
            
            # Reshape back: [1,C,D,H,W] -> [D,H,W,C]
            laplacian_sub = laplacian_sub.squeeze(0).permute(1, 2, 3, 0)
            
            # Upsample back using nearest neighbor interpolation
            laplacian_full = F.interpolate(
                laplacian_sub.permute(3, 0, 1, 2).unsqueeze(0),
                size=field.shape[:3],
                mode='nearest'
            ).squeeze(0).permute(1, 2, 3, 0)
            
            # Apply diffusion with slightly reduced rate to compensate
            field = field + self.diffusion_rate * laplacian_full * 0.8
        else:
            # For smaller fields, apply conv3d directly
            # Reshape for conv3d: [D,H,W,C] -> [1,C,D,H,W]
            C = field.shape[3]
            field_conv = field.permute(3, 0, 1, 2).unsqueeze(0)
            
            # Expand kernel for all channels (depthwise convolution)
            kernel_expanded = self._laplacian_kernel.expand(C, 1, -1, -1, -1)
            
            # Apply 3D convolution with proper padding
            laplacian = F.conv3d(field_conv, kernel_expanded, padding=1, groups=C)
            
            # Reshape back: [1,C,D,H,W] -> [D,H,W,C]
            laplacian = laplacian.squeeze(0).permute(1, 2, 3, 0)
            
            field = field + self.diffusion_rate * laplacian
        
        return field
    
    def _apply_diffusion_original(self, field: torch.Tensor) -> torch.Tensor:
        """
        Original diffusion implementation for reference and fallback.
        """
        # For very large fields, use strided diffusion for efficiency
        if field.shape[0] > 64:
            # Subsample for diffusion calculation (4x faster)
            stride = 2
            subsampled = field[::stride, ::stride, ::stride, :]
            
            # Apply diffusion on smaller field
            padded = torch.nn.functional.pad(subsampled, (0, 0, 1, 1, 1, 1, 1, 1), mode='constant', value=0)
            
            neighbors = (
                padded[:-2, 1:-1, 1:-1, :] +  # -x
                padded[2:, 1:-1, 1:-1, :] +   # +x
                padded[1:-1, :-2, 1:-1, :] +  # -y
                padded[1:-1, 2:, 1:-1, :] +   # +y
                padded[1:-1, 1:-1, :-2, :] +  # -z
                padded[1:-1, 1:-1, 2:, :]     # +z
            )
            
            laplacian_sub = neighbors - 6 * subsampled
            
            # Upsample back using nearest neighbor interpolation
            laplacian_full = torch.nn.functional.interpolate(
                laplacian_sub.permute(3, 0, 1, 2).unsqueeze(0),
                size=field.shape[:3],
                mode='nearest'
            ).squeeze(0).permute(1, 2, 3, 0)
            
            # Apply diffusion with slightly reduced rate to compensate
            field = field + self.diffusion_rate * laplacian_full * 0.8
        else:
            # Original implementation for smaller fields
            padded = torch.nn.functional.pad(field, (0, 0, 1, 1, 1, 1, 1, 1), mode='constant', value=0)
            
            neighbors = (
                padded[:-2, 1:-1, 1:-1, :] +  # -x
                padded[2:, 1:-1, 1:-1, :] +   # +x
                padded[1:-1, :-2, 1:-1, :] +  # -y
                padded[1:-1, 2:, 1:-1, :] +   # +y
                padded[1:-1, 1:-1, :-2, :] +  # -z
                padded[1:-1, 1:-1, 2:, :]     # +z
            )
            
            laplacian = neighbors - 6 * field
            field = field + self.diffusion_rate * laplacian
        
        return field
    
    def get_energy(self, field: torch.Tensor) -> float:
        """Get total field energy (activity level)."""
        return torch.abs(field).mean().item()
    
    def get_variance(self, field: torch.Tensor) -> float:
        """Get field variance (diversity of states)."""
        return field.var().item()
    
    def get_gradients(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradients (for motor extraction).
        
        Returns:
            Gradient magnitude at each point
        """
        # Compute differences in each dimension
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        
        # Magnitude of gradient vector
        gradient_mag = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        return gradient_mag