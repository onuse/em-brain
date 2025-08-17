"""
Simple Field Dynamics

Just the physics - diffusion, decay, and bounded noise.
No patterns, no waves, no complex features. Just physics.
"""

import torch
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
        Optimized for large tensors.
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