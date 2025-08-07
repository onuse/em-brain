"""
GPU Memory Optimizer

Optimizes tensor operations for reduced memory usage and better GPU utilization.
"""

import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMemoryOptimizer:
    """
    Optimizes GPU memory usage through:
    1. In-place operations where possible
    2. Gradient checkpointing for large operations
    3. Memory pooling and reuse
    4. Tensor operation fusion
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_pool = {}
        self.operation_count = 0
        
    def get_pooled_tensor(self, shape: Tuple[int, ...], 
                         dtype: torch.dtype = torch.float32,
                         key: Optional[str] = None) -> torch.Tensor:
        """Get a tensor from the memory pool or create new."""
        if key is None:
            key = f"tensor_{shape}_{dtype}"
            
        if key in self.memory_pool:
            tensor = self.memory_pool[key]
            if tensor.shape == shape and tensor.dtype == dtype:
                return tensor.zero_()  # Clear and return
        
        # Create new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        self.memory_pool[key] = tensor
        return tensor
    
    @staticmethod
    def fused_decay_diffusion(field: torch.Tensor, 
                             decay_rate: float,
                             diffusion_rate: float) -> torch.Tensor:
        """Fused decay and diffusion operation."""
        # Apply decay in-place
        field.mul_(decay_rate)
        
        # Apply simplified diffusion in-place
        if diffusion_rate > 0:
            # Simple neighbor averaging for diffusion
            # Much faster than convolution for our use case
            with torch.no_grad():
                # Create shifted versions
                diff = torch.zeros_like(field)
                
                # Add contributions from neighbors (6-connectivity)
                diff[1:, :, :, :] += field[:-1, :, :, :]  # left
                diff[:-1, :, :, :] += field[1:, :, :, :]  # right
                diff[:, 1:, :, :] += field[:, :-1, :, :]  # front
                diff[:, :-1, :, :] += field[:, 1:, :, :]  # back
                diff[:, :, 1:, :] += field[:, :, :-1, :]  # bottom
                diff[:, :, :-1, :] += field[:, :, 1:, :]  # top
                
                # Average and apply diffusion
                field.add_(diff.mul_(diffusion_rate / 6.0))
            
        return field
    
    @staticmethod
    def fused_pattern_features(field_region: torch.Tensor) -> Dict[str, float]:
        """Compute multiple pattern features in one pass."""
        # Flatten for efficiency
        flat = field_region.flatten()
        
        if len(flat) < 2:
            return {
                'energy': 0.0,
                'variance': 0.0,
                'coherence': 0.0,
                'salience': 0.0
            }
        
        # Compute all statistics in one pass
        abs_flat = torch.abs(flat)
        energy = torch.mean(abs_flat)
        variance = torch.var(flat)
        
        # Avoid item() calls until the end
        energy_val = float(energy)
        variance_val = float(variance)
        coherence = variance_val / (energy_val + 1e-8)
        
        return {
            'energy': energy_val,
            'variance': variance_val,
            'coherence': coherence,
            'salience': energy_val * (1 + variance_val)
        }
    
    def optimize_field_evolution(self, field: torch.Tensor,
                               decay_rate: float,
                               diffusion_rate: float,
                               spontaneous_weight: float) -> torch.Tensor:
        """Optimized field evolution with minimal memory allocation."""
        # Use in-place operations
        with torch.no_grad():
            # Fused decay and diffusion
            field = self.fused_decay_diffusion(field, decay_rate, diffusion_rate)
            
            # Add spontaneous activity in-place
            if spontaneous_weight > 0:
                # Generate noise directly on field
                noise = torch.randn_like(field) * spontaneous_weight * 0.01
                field.add_(noise)
            
        return field
    
    def clear_pool(self):
        """Clear memory pool."""
        self.memory_pool.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            'pool_size': len(self.memory_pool),
            'pool_memory_mb': sum(
                t.element_size() * t.nelement() / 1024 / 1024 
                for t in self.memory_pool.values()
            )
        }
        
        if self.device.type == 'cuda':
            stats.update({
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            })
        
        return stats


class BatchedBrainProcessor:
    """
    Process multiple brain instances in batches for better GPU utilization.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_optimizer = GPUMemoryOptimizer(device)
        
    def batch_process_fields(self, fields: List[torch.Tensor],
                           decay_rate: float,
                           diffusion_rate: float) -> List[torch.Tensor]:
        """Process multiple fields in batch."""
        if not fields:
            return []
        
        # Stack fields for batch processing
        batch_shape = [len(fields)] + list(fields[0].shape)
        batched = torch.stack(fields)
        
        # Apply operations in batch
        with torch.no_grad():
            # Decay all at once
            batched.mul_(decay_rate)
            
            # Batch diffusion if needed
            if diffusion_rate > 0:
                # Apply diffusion to each field
                # This could be further optimized with custom CUDA kernels
                for i in range(len(fields)):
                    batched[i] = self.memory_optimizer.fused_decay_diffusion(
                        batched[i], 1.0, diffusion_rate
                    )
        
        # Return as list
        return [batched[i] for i in range(len(fields))]
    
    def batch_extract_patterns(self, fields: List[torch.Tensor],
                             n_patterns: int = 5) -> List[List[Dict[str, float]]]:
        """Extract patterns from multiple fields in batch."""
        results = []
        
        # Process in chunks to avoid memory overflow
        chunk_size = 4
        for i in range(0, len(fields), chunk_size):
            chunk = fields[i:i+chunk_size]
            
            # Extract patterns for each field
            for field in chunk:
                patterns = []
                
                # Global pattern
                global_features = self.memory_optimizer.fused_pattern_features(field)
                patterns.append(global_features)
                
                # Local patterns (simplified)
                if n_patterns > 1:
                    # Sample regions
                    for j in range(min(n_patterns - 1, 4)):
                        # Random region
                        x = torch.randint(0, field.shape[0] - 4, (1,)).item()
                        y = torch.randint(0, field.shape[1] - 4, (1,)).item()
                        z = torch.randint(0, field.shape[2] - 4, (1,)).item()
                        
                        region = field[x:x+4, y:y+4, z:z+4]
                        features = self.memory_optimizer.fused_pattern_features(region)
                        patterns.append(features)
                
                results.append(patterns)
        
        return results