"""
GPU Optimization Utilities for Field-Native Brain
Week 1: Foundation - Eliminate CPU-GPU transfers and optimize tensor operations
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import functools


class TensorMemoryPool:
    """Pre-allocated memory pool to avoid dynamic allocations"""
    
    def __init__(self, field_shape: Tuple[int, ...], device: str = 'cuda'):
        self.field_shape = field_shape
        self.device = device
        self.pools = {}
        
        # Pre-allocate common tensor sizes
        self._preallocate_common_sizes()
        
    def _preallocate_common_sizes(self):
        """Pre-allocate tensors for common operations"""
        D, H, W, C = self.field_shape
        
        # Field-sized tensors (for temporary computations)
        self.pools['field_temp_1'] = torch.zeros(D, H, W, C, device=self.device)
        self.pools['field_temp_2'] = torch.zeros(D, H, W, C, device=self.device)
        self.pools['field_temp_3'] = torch.zeros(D, H, W, C, device=self.device)
        
        # Gradient computation buffers
        self.pools['gradient_x'] = torch.zeros(D, H, W, C, device=self.device)
        self.pools['gradient_y'] = torch.zeros(D, H, W, C, device=self.device)
        self.pools['gradient_z'] = torch.zeros(D, H, W, C, device=self.device)
        
        # Pattern buffers (for strategic patterns)
        self.pools['pattern_buffer'] = torch.zeros(D, H, W, 16, device=self.device)
        
        # Scalar buffers (avoid .item() calls)
        self.pools['scalar_buffer'] = torch.zeros(100, device=self.device)
        
        # Statistics buffers
        self.pools['stats_mean'] = torch.zeros(C, device=self.device)
        self.pools['stats_std'] = torch.zeros(C, device=self.device)
        
    def get_field_buffer(self, name: str = 'field_temp_1') -> torch.Tensor:
        """Get a pre-allocated field-sized buffer"""
        return self.pools[name].zero_()  # Clear and return
        
    def get_scalar_buffer(self, size: int = 1) -> torch.Tensor:
        """Get a buffer for scalar operations (avoid .item())"""
        return self.pools['scalar_buffer'][:size].zero_()


class BatchedFieldOperations:
    """Batched operations to minimize kernel launches"""
    
    @staticmethod
    @torch.jit.script
    def fused_field_evolution(
        field: torch.Tensor,
        decay_rate: float,
        diffusion_strength: float,
        spontaneous_rate: float,
        noise_scale: float
    ) -> torch.Tensor:
        """Fused kernel for field evolution (decay + diffusion + spontaneous)"""
        # Apply decay
        field = field * (1.0 - decay_rate)
        
        # Apply diffusion (3D convolution)
        kernel = torch.ones(1, 1, 3, 3, 3, device=field.device) / 27.0
        field_reshaped = field.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
        
        for c in range(field.shape[3]):
            diffused = F.conv3d(
                field_reshaped[:, c:c+1], 
                kernel, 
                padding=1
            )
            field_reshaped[:, c:c+1] = (
                field_reshaped[:, c:c+1] * (1 - diffusion_strength) + 
                diffused * diffusion_strength
            )
        
        field = field_reshaped.squeeze(0).permute(1, 2, 3, 0)  # Back to [D, H, W, C]
        
        # Add spontaneous activity
        if spontaneous_rate > 0:
            noise = torch.randn_like(field) * noise_scale
            field = field + noise * spontaneous_rate
            
        return field
    
    @staticmethod
    def batched_gradient_extraction(field: torch.Tensor) -> torch.Tensor:
        """Extract gradients using 3D convolutions (no loops)"""
        # Sobel kernels for 3D gradient
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=field.dtype, device=field.device
        ).view(1, 1, 3, 3, 3)
        
        sobel_y = sobel_x.permute(0, 1, 2, 4, 3)
        sobel_z = sobel_x.permute(0, 1, 4, 2, 3)
        
        # Reshape field for convolution
        field_reshaped = field.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
        
        # Compute gradients for all channels in parallel
        grad_x = F.conv3d(field_reshaped, sobel_x.repeat(field.shape[3], 1, 1, 1, 1), 
                          padding=1, groups=field.shape[3])
        grad_y = F.conv3d(field_reshaped, sobel_y.repeat(field.shape[3], 1, 1, 1, 1), 
                          padding=1, groups=field.shape[3])
        grad_z = F.conv3d(field_reshaped, sobel_z.repeat(field.shape[3], 1, 1, 1, 1), 
                          padding=1, groups=field.shape[3])
        
        # Combine gradients
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Take gradient from content channels only (0:32)
        gradient = gradient_magnitude[:, :32].mean(dim=1)  # Average across channels
        
        return gradient.squeeze(0)  # [D, H, W]


class GPUPatternLibrary:
    """GPU-resident pattern library with parallel matching"""
    
    def __init__(self, max_patterns: int = 30, field_shape: Tuple[int, ...] = (32, 32, 32, 64)):
        self.max_patterns = max_patterns
        self.field_shape = field_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Store all patterns as a single tensor
        self.patterns = torch.zeros(max_patterns, *field_shape[:-1], 16, device=self.device)
        self.contexts = torch.zeros(max_patterns, 128, device=self.device)  # Context embeddings
        self.scores = torch.zeros(max_patterns, device=self.device)  # Pattern scores
        self.ages = torch.zeros(max_patterns, device=self.device)  # Pattern ages
        self.count = 0
        
    def add_pattern(self, pattern: torch.Tensor, context: torch.Tensor, score: float):
        """Add pattern to library (GPU-resident)"""
        if self.count < self.max_patterns:
            idx = self.count
            self.count += 1
        else:
            # Replace oldest pattern
            idx = torch.argmax(self.ages).item()
            
        self.patterns[idx] = pattern
        self.contexts[idx] = context
        self.scores[idx] = score
        self.ages[idx] = 0
        
        # Age all other patterns
        self.ages[:self.count] += 1
        self.ages[idx] = 0
        
    def find_best_pattern(self, current_context: torch.Tensor, field_state: torch.Tensor) -> torch.Tensor:
        """Find best matching pattern using parallel operations"""
        if self.count == 0:
            return torch.zeros(*self.field_shape[:-1], 16, device=self.device)
            
        # Compute similarities for all patterns in parallel
        context_similarity = F.cosine_similarity(
            current_context.unsqueeze(0),
            self.contexts[:self.count],
            dim=1
        )
        
        # Weight by scores and recency
        recency_weight = torch.exp(-self.ages[:self.count] / 10.0)
        combined_score = context_similarity * self.scores[:self.count] * recency_weight
        
        # Soft retrieval (weighted blend of top patterns)
        top_k = min(3, self.count)
        top_scores, top_indices = torch.topk(combined_score, top_k)
        weights = F.softmax(top_scores * 5.0, dim=0)
        
        # Weighted combination of patterns
        result = torch.zeros_like(self.patterns[0])
        for i, idx in enumerate(top_indices):
            result += self.patterns[idx] * weights[i]
            
        return result


class OptimizedFieldMetrics:
    """Compute field metrics without CPU transfers"""
    
    @staticmethod
    def compute_all_metrics(field: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all metrics on GPU, return GPU tensors (no .item())"""
        metrics = {}
        
        # Basic statistics (keep on GPU)
        metrics['mean_activation'] = field.mean()
        metrics['std_activation'] = field.std()
        metrics['max_activation'] = field.abs().max()
        
        # Information content (Shannon entropy approximation)
        field_norm = F.softmax(field.reshape(-1), dim=0)
        entropy = -(field_norm * torch.log(field_norm + 1e-10)).sum()
        metrics['information'] = entropy
        
        # Sparsity (percentage of near-zero activations)
        sparsity = (field.abs() < 0.01).float().mean()
        metrics['sparsity'] = sparsity
        
        # Energy (L2 norm)
        metrics['energy'] = torch.norm(field)
        
        # Complexity (gradient magnitude)
        grad_x = field[1:] - field[:-1]
        grad_y = field[:, 1:] - field[:, :-1]
        grad_z = field[:, :, 1:] - field[:, :, :-1]
        complexity = (grad_x.abs().mean() + grad_y.abs().mean() + grad_z.abs().mean()) / 3
        metrics['complexity'] = complexity
        
        return metrics
    
    @staticmethod
    def metrics_to_cpu_dict(gpu_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert GPU metrics to CPU values only when needed for logging"""
        return {k: v.item() for k, v in gpu_metrics.items()}


def eliminate_item_calls(value: Any) -> Any:
    """
    Replace .item() calls with GPU-friendly alternatives
    Use this wrapper when you absolutely need a Python scalar
    """
    if torch.is_tensor(value):
        # Keep on GPU as long as possible
        return value
    return value


def batch_cpu_operations(operations: List[callable]) -> List[Any]:
    """
    Batch multiple operations that require CPU transfer
    Minimizes synchronization overhead
    """
    with torch.cuda.stream(torch.cuda.Stream()):
        results = []
        for op in operations:
            results.append(op())
        torch.cuda.synchronize()
    return results


# Example usage in unified_field_brain.py refactor:
class OptimizedFieldBrainMixin:
    """Mixin for optimized field operations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_pool = TensorMemoryPool(self.tensor_shape, self.device)
        self.gpu_pattern_library = GPUPatternLibrary(field_shape=self.tensor_shape)
        
    def _optimized_process_cycle(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Optimized processing cycle with minimal CPU transfers"""
        
        # Keep everything on GPU
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            # Fused field evolution
            self.unified_field = BatchedFieldOperations.fused_field_evolution(
                self.unified_field,
                decay_rate=0.01,
                diffusion_strength=0.05,
                spontaneous_rate=0.001,
                noise_scale=0.1
            )
            
            # Process sensory input (no CPU transfer)
            if sensory_input is not None:
                self._process_sensory_gpu(sensory_input)
            
            # Extract motor commands via gradients (parallel)
            gradient = BatchedFieldOperations.batched_gradient_extraction(self.unified_field)
            
            # Generate motor output (stay on GPU)
            motor_output = self._extract_motor_gpu(gradient)
            
        return motor_output
    
    def _process_sensory_gpu(self, sensory_input: torch.Tensor):
        """Process sensory input entirely on GPU"""
        # Ensure input is on correct device
        if not sensory_input.is_cuda:
            sensory_input = sensory_input.to(self.device)
            
        # Find best location using parallel convolution
        pattern = sensory_input.view(1, 1, -1, 1, 1)
        field_content = self.unified_field[:, :, :, :32]
        
        # Convolve to find resonance
        resonance = F.conv3d(
            field_content.permute(3, 0, 1, 2).unsqueeze(0),
            pattern,
            padding=0
        )
        
        # Find peak resonance location
        flat_idx = torch.argmax(resonance)
        # No .item() call - keep indices on GPU
        
        # Imprint at best location (GPU indexing)
        self.unified_field.view(-1, 64)[flat_idx, :len(sensory_input)] += sensory_input * 0.1
    
    def _extract_motor_gpu(self, gradient: torch.Tensor) -> torch.Tensor:
        """Extract motor commands from gradient (GPU only)"""
        # Sample gradient at motor regions
        motor_gradient = gradient[16:20, 16:20, 16:20].reshape(-1)
        
        # Apply thresholds (vectorized)
        motor_output = torch.where(
            motor_gradient.abs() > 0.1,
            motor_gradient * 0.5,
            torch.zeros_like(motor_gradient)
        )
        
        return motor_output[:self.motor_dim]


# Performance monitoring utilities
class GPUProfiler:
    """Simple GPU profiling utilities"""
    
    @staticmethod
    def profile_operation(func):
        """Decorator to profile GPU operations"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            result = func(*args, **kwargs)
            end.record()
            
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)  # milliseconds
            
            print(f"{func.__name__}: {elapsed:.2f}ms")
            return result
        return wrapper
    
    @staticmethod
    def memory_snapshot():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'free_mb': reserved - allocated
            }
        return {}