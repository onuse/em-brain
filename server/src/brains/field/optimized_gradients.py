#!/usr/bin/env python3
"""
Optimized gradient calculation for UnifiedFieldBrain.

Key optimizations:
1. Compute gradients only where needed (sparse approach)
2. Use in-place operations to reduce memory allocation
3. Cache gradients when field hasn't changed significantly
4. Only compute gradients for meaningful dimensions
"""

import torch
from typing import Dict, Tuple, Optional, List
import time


class OptimizedGradientCalculator:
    """Efficient gradient calculation for high-dimensional fields."""
    
    def __init__(self, 
                 activation_threshold: float = 0.01,
                 cache_enabled: bool = True,
                 sparse_computation: bool = True):
        
        self.activation_threshold = activation_threshold
        self.cache_enabled = cache_enabled
        self.sparse_computation = sparse_computation
        
        # Cache management
        self.cached_gradients: Optional[Dict[str, torch.Tensor]] = None
        self.cache_field_hash: Optional[int] = None
        self.cache_hits = 0
        self.cache_misses = 0
        
    def calculate_gradients(self, field: torch.Tensor, 
                          dimensions_to_compute: Optional[List[int]] = None,
                          local_region_only: bool = False,
                          region_center: Optional[Tuple[int, int, int]] = None,
                          region_size: int = 3) -> Dict[str, torch.Tensor]:
        """
        Calculate gradients efficiently.
        
        Args:
            field: The multi-dimensional field tensor
            dimensions_to_compute: Which dimensions to compute gradients for (default: first 5)
            local_region_only: If True, only compute gradients in a local region
            region_center: Center of the local region (default: field center)
            region_size: Size of the local region (default: 3x3x3)
            
        Returns:
            Dictionary of gradient tensors
        """
        # Check cache first
        if self.cache_enabled:
            field_hash = self._compute_field_hash(field)
            if self.cached_gradients is not None and field_hash == self.cache_field_hash:
                self.cache_hits += 1
                return self.cached_gradients
            self.cache_misses += 1
        
        # Default to first 5 dimensions (spatial + scale + time)
        if dimensions_to_compute is None:
            dimensions_to_compute = list(range(min(5, field.ndim)))
        
        gradients = {}
        
        # Local region optimization - massive speedup for robot control
        if local_region_only and field.ndim >= 3:
            # Default to center if not specified
            if region_center is None:
                region_center = tuple(field.shape[i]//2 for i in range(3))
            
            # Extract local region for first 3 spatial dimensions
            half_size = region_size // 2
            x_start = max(0, region_center[0] - half_size)
            x_end = min(field.shape[0], region_center[0] + half_size + 1)
            y_start = max(0, region_center[1] - half_size)
            y_end = min(field.shape[1], region_center[1] + half_size + 1)
            z_start = max(0, region_center[2] - half_size)
            z_end = min(field.shape[2], region_center[2] + half_size + 1)
            
            # For higher dimensions, take a representative slice
            # This preserves the ability to have distributed actuators
            other_indices = []
            for dim in range(3, field.ndim):
                # Take middle slice for non-spatial dimensions
                other_indices.append(field.shape[dim]//2)
            
            # Extract local field slice
            # Build index tuple for higher dimensions
            index_tuple = [slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)]
            for dim in range(3, field.ndim):
                index_tuple.append(slice(None))  # Take all values for non-spatial dims
            local_field = field[tuple(index_tuple)]
            
            # Compute gradients on small local field (much faster!)
            for dim_idx in dimensions_to_compute:
                if dim_idx < 3 and local_field.shape[dim_idx] > 1:
                    # Compute gradient for spatial dimension
                    local_grad = self._compute_dense_gradient(local_field, dim_idx)
                    
                    # Create full-size gradient tensor with zeros
                    full_grad = torch.zeros_like(field)
                    
                    # Place local gradient in correct position
                    grad_index = [slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)]
                    grad_index.extend([slice(None)] * (field.ndim - 3))
                    full_grad[tuple(grad_index)] = local_grad
                    
                    gradients[f'gradient_dim_{dim_idx}'] = full_grad
                elif dim_idx >= 3:
                    # For non-spatial dimensions, compute gradient at center only
                    # This allows for future distributed actuators
                    center_slices = [slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)]
                    center_slices.extend([slice(None)] * (field.ndim - 3))
                    
                    # Extract relevant slice and compute gradient
                    field_slice = field[tuple(center_slices)]
                    grad = self._compute_dense_gradient(field_slice, dim_idx)
                    
                    # Create full gradient tensor
                    full_grad = torch.zeros_like(field)
                    full_grad[tuple(center_slices)] = grad
                    
                    gradients[f'gradient_dim_{dim_idx}'] = full_grad
            
            return gradients
        
        if self.sparse_computation:
            # Find active regions
            active_mask = torch.abs(field) > self.activation_threshold
            active_regions = self._find_active_regions(active_mask)
            
            # Compute gradients only in active regions
            for dim_idx in dimensions_to_compute:
                if field.shape[dim_idx] > 1:
                    grad = self._compute_sparse_gradient(field, dim_idx, active_regions)
                    gradients[f'gradient_dim_{dim_idx}'] = grad
        else:
            # Standard dense computation
            for dim_idx in dimensions_to_compute:
                if field.shape[dim_idx] > 1:
                    grad = self._compute_dense_gradient(field, dim_idx)
                    gradients[f'gradient_dim_{dim_idx}'] = grad
        
        # Update cache
        if self.cache_enabled:
            self.cached_gradients = gradients
            self.cache_field_hash = field_hash
        
        return gradients
    
    def _compute_dense_gradient(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute gradient using optimized dense operations."""
        # Use torch.gradient for better performance
        if hasattr(torch, 'gradient') and field.shape[dim] > 2:
            # torch.gradient is available and dimension is large enough
            grads = torch.gradient(field, dim=dim)
            return grads[0] if isinstance(grads, tuple) else grads
        else:
            # Fallback to finite differences
            grad = torch.zeros_like(field)
            
            # Create slice objects for efficient indexing
            slices_left = [slice(None)] * field.ndim
            slices_right = [slice(None)] * field.ndim
            slices_center = [slice(None)] * field.ndim
            
            slices_left[dim] = slice(None, -1)
            slices_right[dim] = slice(1, None)
            slices_center[dim] = slice(None, -1)
            
            # Forward difference (more memory efficient than creating new tensor)
            grad[tuple(slices_center)] = field[tuple(slices_right)] - field[tuple(slices_left)]
            
            return grad
    
    def _compute_sparse_gradient(self, field: torch.Tensor, dim: int, 
                               active_regions: List[Tuple[int, ...]]) -> torch.Tensor:
        """Compute gradient only in active regions."""
        grad = torch.zeros_like(field)
        
        # Early return if no active regions
        if not active_regions:
            return grad
        
        # Compute gradients only around active points
        for region_center in active_regions:
            # Define region bounds (with padding for gradient calculation)
            region_slices = []
            for i, center in enumerate(region_center):
                if i == dim and field.shape[i] > 2:
                    # Extend region in gradient dimension
                    start = max(0, center - 2)
                    end = min(field.shape[i], center + 3)
                else:
                    # Keep region tight in other dimensions
                    start = max(0, center - 1)
                    end = min(field.shape[i], center + 2)
                region_slices.append(slice(start, end))
            
            # Compute local gradient
            region_field = field[tuple(region_slices)]
            if region_field.shape[dim] > 1:
                # Local gradient computation
                local_grad = self._compute_dense_gradient(region_field, dim)
                
                # Place back into full gradient tensor
                grad[tuple(region_slices)] = local_grad
        
        return grad
    
    def _find_active_regions(self, active_mask: torch.Tensor, 
                           min_distance: int = 5) -> List[Tuple[int, ...]]:
        """Find centers of active regions for sparse computation."""
        # Get all active points
        active_points = torch.nonzero(active_mask)
        
        if len(active_points) == 0:
            return []
        
        # Cluster nearby points to avoid redundant computation
        regions = []
        used = torch.zeros(len(active_points), dtype=torch.bool)
        
        for i in range(len(active_points)):
            if used[i]:
                continue
                
            # Start new region
            center = active_points[i]
            regions.append(tuple(center.tolist()))
            used[i] = True
            
            # Mark nearby points as used
            for j in range(i + 1, len(active_points)):
                if used[j]:
                    continue
                    
                # Check distance (only first 3 dimensions for efficiency)
                dist = torch.sum(torch.abs(active_points[j][:3] - center[:3]))
                if dist < min_distance:
                    used[j] = True
        
        return regions
    
    def _compute_field_hash(self, field: torch.Tensor) -> int:
        """Compute a fast hash of field content for caching."""
        # Sample field values for quick comparison
        sample_indices = [
            tuple(torch.randint(0, s, (1,)).item() for s in field.shape)
            for _ in range(min(20, field.numel() // 1000))
        ]
        
        sample_sum = sum(field[idx].item() for idx in sample_indices)
        field_stats = (
            field.shape,
            torch.sum(torch.abs(field) > self.activation_threshold).item(),
            int(sample_sum * 1000000)
        )
        
        return hash(field_stats)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    def clear_cache(self):
        """Clear the gradient cache."""
        self.cached_gradients = None
        self.cache_field_hash = None


def create_optimized_gradient_calculator(**kwargs) -> OptimizedGradientCalculator:
    """Factory function for creating optimized gradient calculator."""
    return OptimizedGradientCalculator(**kwargs)