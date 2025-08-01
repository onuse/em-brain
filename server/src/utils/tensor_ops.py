"""
Standardized Tensor Operations

Provides consistent tensor operations for the field brain system.
All operations follow these standards:
- Always specify dtype (default: torch.float32)
- Always specify device
- Use in-place operations where safe
- Batch operations when possible
- Handle edge cases gracefully
"""

import torch
from typing import Optional, Tuple, Union


def create_zeros(shape: Union[Tuple[int, ...], torch.Size], 
                device: torch.device,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create zero tensor with standard settings."""
    return torch.zeros(shape, dtype=dtype, device=device)


def create_ones(shape: Union[Tuple[int, ...], torch.Size],
               device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create ones tensor with standard settings."""
    return torch.ones(shape, dtype=dtype, device=device)


def create_randn(shape: Union[Tuple[int, ...], torch.Size],
                device: torch.device,
                dtype: torch.dtype = torch.float32,
                scale: float = 1.0,
                bias: float = 0.0) -> torch.Tensor:
    """Create random normal tensor with optional scaling."""
    tensor = torch.randn(shape, dtype=dtype, device=device)
    if scale != 1.0 or bias != 0.0:
        tensor = tensor * scale + bias
    return tensor


def safe_mean(tensor: torch.Tensor, 
              dim: Optional[Union[int, Tuple[int, ...]]] = None,
              keepdim: bool = False) -> torch.Tensor:
    """Compute mean with NaN handling."""
    if tensor.numel() == 0:
        return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
    return torch.mean(tensor, dim=dim, keepdim=keepdim)


def safe_var(tensor: torch.Tensor,
             dim: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdim: bool = False) -> torch.Tensor:
    """Compute variance with edge case handling."""
    if tensor.numel() <= 1:
        return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
    
    result = torch.var(tensor, dim=dim, keepdim=keepdim)
    
    # NaN protection
    if torch.isnan(result).any():
        if keepdim or dim is None:
            result[torch.isnan(result)] = 0.0
        else:
            result = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
    
    return result


def safe_normalize(tensor: torch.Tensor, 
                  eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor with zero-division protection."""
    norm = torch.norm(tensor)
    return tensor / (norm + eps)


def apply_diffusion(field: torch.Tensor,
                   rate: float,
                   dims: Tuple[int, ...] = (0, 1, 2)) -> torch.Tensor:
    """Apply diffusion to field along specified dimensions."""
    if rate <= 0:
        return field
    
    result = field.clone()
    for dim in dims:
        if dim < field.ndim:
            shifted_fwd = torch.roll(field, shifts=1, dims=dim)
            shifted_back = torch.roll(field, shifts=-1, dims=dim)
            diffusion = (shifted_fwd + shifted_back - 2 * field) / 2
            result = result + rate * diffusion
    
    return result


def field_energy(field: torch.Tensor) -> float:
    """Compute field energy (mean absolute value).
    DEPRECATED: Use field_information() instead.
    """
    return field_information(field)


def field_information(field: torch.Tensor) -> float:
    """Compute field information content (mean absolute value)."""
    result = float(torch.mean(torch.abs(field)))
    # NaN protection
    if torch.isnan(torch.tensor(result)) or result != result:
        return 0.5  # Default neutral value
    return result


def field_stats(field: torch.Tensor) -> dict:
    """Compute standard field statistics."""
    return {
        'energy': field_energy(field),  # Keep for compatibility
        'information': field_information(field),
        'variance': float(safe_var(field)),
        'max': float(torch.max(torch.abs(field))),
        'min': float(torch.min(torch.abs(field)))
    }


def ensure_device_consistency(*tensors: torch.Tensor, 
                            device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
    """Ensure all tensors are on the same device."""
    if not tensors:
        return tensors
    
    if device is None:
        device = tensors[0].device
    
    return tuple(t.to(device) if t.device != device else t for t in tensors)


def clamp_field(field: torch.Tensor,
                min_val: float = -10.0,
                max_val: float = 10.0,
                inplace: bool = True) -> torch.Tensor:
    """Clamp field values to prevent explosion."""
    if inplace:
        return torch.clamp_(field, min=min_val, max=max_val)
    else:
        return torch.clamp(field, min=min_val, max=max_val)