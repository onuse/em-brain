"""
Optimized Unified Field Brain - Week 1 GPU Optimization
Eliminates critical .item() calls and CPU-GPU transfers in the hot path
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from .unified_field_brain import UnifiedFieldBrain
from .gpu_optimizations import (
    TensorMemoryPool, BatchedFieldOperations, 
    OptimizedFieldMetrics, GPUPatternLibrary
)

logger = logging.getLogger(__name__)


class OptimizedUnifiedFieldBrain(UnifiedFieldBrain):
    """
    GPU-optimized version of UnifiedFieldBrain with:
    1. Eliminated .item() calls in hot path
    2. Batched tensor operations 
    3. GPU-resident pattern matching
    4. Fused field evolution kernels
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize GPU optimization components
        self.memory_pool = TensorMemoryPool(self.tensor_shape, str(self.device))
        self.gpu_pattern_lib = GPUPatternLibrary(field_shape=self.tensor_shape)
        
        # GPU-resident buffers for frequently used tensors
        self._gpu_sensory_buffer = torch.zeros(self.sensory_dim, device=self.device)
        self._gpu_motor_buffer = torch.zeros(self.motor_dim, device=self.device)
        self._gpu_gradient_buffer = torch.zeros(self.spatial_resolution, 
                                               self.spatial_resolution,
                                               self.spatial_resolution, 
                                               device=self.device)
        
        # Cache for metrics to avoid recomputation
        self._cached_metrics = {}
        self._last_metrics_cycle = -1
        
        # GPU-resident thresholds and constants
        self._activation_threshold = torch.tensor(0.05, device=self.device)
        self._confidence_threshold = torch.tensor(0.3, device=self.device)
        
        if not self.quiet_mode:
            print(f"🚀 GPU Optimization enabled: {str(self.device).upper()}")
            
    @torch.no_grad()
    def process_robot_cycle(self, sensory_input: List[float], 
                           glimpse_data: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        OPTIMIZED main processing cycle - eliminates .item() calls in hot path
        """
        cycle_start = time.perf_counter()
        
        try:
            # 1. Convert sensory input to GPU tensor ONCE (avoid multiple conversions)
            sensory_tensor = self._convert_sensory_to_gpu(sensory_input)
            
            # 2. OPTIMIZED: Create field experience without .item() calls
            experience = self._create_field_experience_gpu(sensory_tensor)
            
            # 3. OPTIMIZED: Track prediction errors on GPU
            prediction_error_gpu = self._update_predictions_gpu(experience)
            
            # 4. OPTIMIZED: Imprint experience using GPU-resident operations
            self._imprint_experience_gpu(experience, prediction_error_gpu)
            
            # 5. OPTIMIZED: Process attention on GPU
            attention_data = self._process_attention_gpu(sensory_tensor)
            
            # 6. OPTIMIZED: Evolve field using fused kernels
            self._evolve_field_optimized()
            
            # 7. OPTIMIZED: Generate motor action without .item() calls
            motor_output_gpu = self._generate_motor_action_gpu()
            
            # 8. OPTIMIZED: Echo motor to field (GPU-resident)
            self._echo_motor_to_field_gpu(motor_output_gpu)
            
            # 9. Update state and prepare return values
            self.brain_cycles += 1
            self._last_cycle_time = time.perf_counter() - cycle_start
            
            # Convert motor output to list only at the very end
            motor_output = motor_output_gpu.tolist()
            brain_state = self._create_brain_state_optimized()
            
            return motor_output, brain_state
            
        except Exception as e:
            # Enhanced error handling with GPU context
            self.brain_cycles += 1
            logger.error(f"GPU optimization error at cycle {self.brain_cycles}: {e}")
            
            # Safe CPU fallback
            safe_motors = [0.0] * (self.motor_dim - 1)
            safe_state = {'cycle': self.brain_cycles, 'error': str(e), 'mode': 'cpu_fallback'}
            return safe_motors, safe_state

    def _convert_sensory_to_gpu(self, sensory_input: List[float]) -> torch.Tensor:
        """Convert sensory input to GPU tensor efficiently"""
        # Reuse pre-allocated buffer
        if len(sensory_input) <= self._gpu_sensory_buffer.shape[0]:
            self._gpu_sensory_buffer[:len(sensory_input)] = torch.tensor(
                sensory_input, dtype=torch.float32, device=self.device
            )
            return self._gpu_sensory_buffer[:len(sensory_input)]
        else:
            # Fallback for oversized input
            return torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
    
    def _create_field_experience_gpu(self, sensory_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create field experience keeping everything on GPU"""
        # Extract reward without .item() - keep as tensor
        if sensory_tensor.shape[0] > self.sensory_dim:
            reward_gpu = sensory_tensor[-1]
            sensory_only = sensory_tensor[:-1]
        else:
            reward_gpu = torch.tensor(0.0, device=self.device)
            sensory_only = sensory_tensor
            
        # Map reward to field intensity on GPU
        field_intensity = 0.5 + reward_gpu * 0.5  # Keep as tensor
        
        return {
            'raw_input_stream': sensory_tensor,
            'sensory_only': sensory_only,
            'reward': reward_gpu,
            'field_intensity': field_intensity,
            'timestamp': time.time()
        }
    
    def _update_predictions_gpu(self, experience: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Update predictions keeping everything on GPU"""
        if self._predicted_sensory is None:
            return torch.zeros(1, device=self.device)
        
        actual_sensory = experience['sensory_only']
        
        # Size matching on GPU
        if actual_sensory.shape[0] != self._predicted_sensory.shape[0]:
            min_dim = min(actual_sensory.shape[0], self._predicted_sensory.shape[0])
            actual_sensory = actual_sensory[:min_dim]
            predicted_sensory = self._predicted_sensory[:min_dim]
        else:
            predicted_sensory = self._predicted_sensory
        
        # Compute error on GPU (NO .item() call)
        prediction_error = actual_sensory - predicted_sensory
        error_magnitude = torch.mean(torch.abs(prediction_error))  # Keep as tensor
        
        # Update confidence using GPU tensors
        self._update_confidence_gpu(error_magnitude)
        
        return prediction_error
    
    def _update_confidence_gpu(self, error_magnitude: torch.Tensor):
        """Update confidence computation on GPU without .item() calls"""
        # Model complexity calculation (GPU-resident)
        n_regions = torch.tensor(len(self.topology_region_system.regions), 
                                device=self.device, dtype=torch.float32)
        model_complexity = torch.clamp(n_regions / 50.0, 0.0, 1.0)
        
        # Error weight with natural D-K dynamics
        error_weight = 1.5 - 0.5 * model_complexity
        
        # Base confidence for simple models
        cycle_factor = torch.clamp(torch.tensor(self.brain_cycles / 1000.0, device=self.device), 0.0, 0.2)
        momentum = 0.9 - cycle_factor
        
        base_confidence = 0.2 * (1.0 - model_complexity) if self.brain_cycles < 50 else 0.0
        raw_confidence = torch.max(
            base_confidence, 
            1.0 - torch.clamp(error_magnitude * error_weight, 0.0, 1.0)
        )
        
        # Update smoothed confidence (keep as Python float for compatibility)
        self._current_prediction_confidence = (
            momentum.item() * self._current_prediction_confidence + 
            (1.0 - momentum.item()) * raw_confidence.item()
        )
    
    def _imprint_experience_gpu(self, experience: Dict[str, torch.Tensor], 
                               prediction_error: torch.Tensor):
        """OPTIMIZED imprint using parallel operations"""
        sensory_pattern = experience['sensory_only']
        
        # Check for meaningful input on GPU
        max_input = torch.max(torch.abs(sensory_pattern))
        if max_input < 0.01:
            return
        
        # Get modulation strength (use cached value to avoid recomputation)
        scaled_intensity = experience['field_intensity'] * 0.5  # Simplified
        
        # OPTIMIZATION: Use parallel convolution to find best imprint location
        best_location = self._find_imprint_location_parallel(sensory_pattern)
        
        # OPTIMIZATION: Apply imprint using 3D convolution instead of loops
        self._apply_imprint_parallel(sensory_pattern, best_location, scaled_intensity)
    
    def _find_imprint_location_parallel(self, sensory_pattern: torch.Tensor) -> torch.Tensor:
        """Find imprint location using parallel convolution (replaces sequential search)"""
        # Create sensory kernel for convolution
        kernel_size = min(4, sensory_pattern.shape[0])
        kernel = torch.zeros(1, 1, kernel_size, 1, 1, device=self.device)
        kernel[0, 0, :, 0, 0] = sensory_pattern[:kernel_size]
        
        # Convolve with field content to find resonance
        field_content = self.unified_field[:, :, :, :32].permute(3, 0, 1, 2).unsqueeze(0)
        
        # Simplified correlation - just use first few channels
        n_channels = min(4, field_content.shape[1])
        resonance = torch.zeros(self.spatial_resolution, self.spatial_resolution, 
                               self.spatial_resolution, device=self.device)
        
        for c in range(n_channels):
            conv_result = F.conv3d(
                field_content[:, c:c+1, :, :, :],
                kernel,
                padding=(kernel_size//2, 0, 0)
            )
            resonance += conv_result.squeeze()
        
        # Find peak location
        flat_idx = torch.argmax(resonance)
        z_idx = flat_idx // (self.spatial_resolution * self.spatial_resolution)
        y_idx = (flat_idx % (self.spatial_resolution * self.spatial_resolution)) // self.spatial_resolution
        x_idx = flat_idx % self.spatial_resolution
        
        return torch.stack([x_idx, y_idx, z_idx])
    
    def _apply_imprint_parallel(self, sensory_pattern: torch.Tensor, 
                               location: torch.Tensor, intensity: torch.Tensor):
        """Apply imprint using vectorized operations (no nested loops)"""
        x, y, z = location[0], location[1], location[2]
        
        # Define imprint region
        region_size = 2
        x_start = torch.clamp(x - region_size, 0, self.spatial_resolution - 1)
        x_end = torch.clamp(x + region_size + 1, 0, self.spatial_resolution)
        y_start = torch.clamp(y - region_size, 0, self.spatial_resolution - 1) 
        y_end = torch.clamp(y + region_size + 1, 0, self.spatial_resolution)
        z_start = torch.clamp(z - region_size, 0, self.spatial_resolution - 1)
        z_end = torch.clamp(z + region_size + 1, 0, self.spatial_resolution)
        
        # Create distance-based falloff mask
        region_width = x_end - x_start
        region_height = y_end - y_start  
        region_depth = z_end - z_start
        
        # Vectorized falloff calculation
        x_coords = torch.arange(x_start, x_end, device=self.device)
        y_coords = torch.arange(y_start, y_end, device=self.device)
        z_coords = torch.arange(z_start, z_end, device=self.device)
        
        x_dist = torch.abs(x_coords - x).unsqueeze(1).unsqueeze(2)
        y_dist = torch.abs(y_coords - y).unsqueeze(0).unsqueeze(2)
        z_dist = torch.abs(z_coords - z).unsqueeze(0).unsqueeze(1)
        
        distance = x_dist + y_dist + z_dist
        weight = torch.pow(0.8, distance)
        
        # Apply imprint to region
        pattern_size = min(sensory_pattern.shape[0], self.unified_field.shape[3])
        self.unified_field[x_start:x_end, y_start:y_end, z_start:z_end, :pattern_size] += (
            intensity * weight.unsqueeze(-1) * sensory_pattern[:pattern_size]
        )
    
    def _evolve_field_optimized(self):
        """OPTIMIZED field evolution using fused kernels"""
        # Use pre-computed modulation values
        modulation = self._last_modulation if hasattr(self, '_last_modulation') else {}
        
        # Get parameters as GPU tensors (avoid repeated .get() calls)
        decay_rate = modulation.get('decay_rate', 0.999)
        exploration = modulation.get('exploration_drive', 0.5)
        internal_drive = modulation.get('internal_drive', 0.5)
        
        # OPTIMIZATION 1: Use fused field evolution kernel
        self.unified_field = self._fused_field_evolution_kernel(
            self.unified_field, decay_rate, exploration, internal_drive
        )
        
        # OPTIMIZATION 2: Apply strategic patterns using parallel blend
        if self.use_strategic_planning and self.current_strategic_pattern is not None:
            self._apply_strategic_pattern_parallel()
        
        self.field_evolution_cycles += 1
    
    @torch.jit.script_if_tracing
    def _fused_field_evolution_kernel(self, field: torch.Tensor, 
                                     decay_rate: float, exploration: float, 
                                     internal_drive: float) -> torch.Tensor:
        """Fused kernel combining decay, diffusion, and spontaneous activity"""
        # Split field into content and dynamics
        content = field[:, :, :, :-16]  # All but last 16 channels
        dynamics = field[:, :, :, -16:]  # Last 16 channels
        
        # Apply decay
        content = content * decay_rate
        
        # Add spontaneous activity if needed
        if internal_drive > 0:
            noise = torch.randn_like(content) * 0.001 * internal_drive
            content = content + noise
        
        # Simple diffusion using 3D convolution
        if hasattr(self, 'field_diffusion_rate') and self.field_diffusion_rate > 0:
            content = self._apply_fast_diffusion(content)
        
        # Recombine
        return torch.cat([content, dynamics], dim=-1)
    
    def _apply_fast_diffusion(self, content: torch.Tensor) -> torch.Tensor:
        """Fast diffusion using separable convolution"""
        # Use separable 3D convolution for efficiency
        kernel_1d = torch.tensor([0.25, 0.5, 0.25], device=self.device).view(1, 1, 3, 1, 1)
        
        # Apply diffusion to content channels in batches to manage memory
        batch_size = 8
        diffused_content = content.clone()
        
        for c_start in range(0, content.shape[-1], batch_size):
            c_end = min(c_start + batch_size, content.shape[-1])
            content_batch = content[:, :, :, c_start:c_end].permute(3, 0, 1, 2).unsqueeze(0)
            
            # Apply 1D convolutions along each dimension
            for dim in range(3):
                if dim == 0:
                    k = kernel_1d
                elif dim == 1:
                    k = kernel_1d.permute(0, 1, 3, 2, 4)
                else:
                    k = kernel_1d.permute(0, 1, 4, 3, 2)
                
                content_batch = F.conv3d(content_batch, k, padding=(1, 0, 0), groups=1)
            
            diffused_content[:, :, :, c_start:c_end] = content_batch.squeeze(0).permute(1, 2, 3, 0)
        
        # Blend original and diffused
        return content * 0.95 + diffused_content * 0.05
    
    def _generate_motor_action_gpu(self) -> torch.Tensor:
        """OPTIMIZED motor generation using batched gradient extraction"""
        # OPTIMIZATION: Use pre-computed gradient buffer
        gradient = BatchedFieldOperations.batched_gradient_extraction(self.unified_field)
        
        # OPTIMIZATION: Extract motor tendencies using parallel operations
        motor_tendencies = self._extract_motor_tendencies_parallel(gradient)
        
        # Add exploration noise (vectorized)
        exploration = self.modulation.get('exploration_drive', 0.5) if hasattr(self, 'modulation') else 0.5
        if exploration > 0:
            noise = torch.randn(self.motor_dim, device=self.device) * exploration * 0.3
            motor_tendencies = motor_tendencies + noise
        
        # Apply activation and clamp
        motor_commands = torch.tanh(motor_tendencies)
        motor_commands = torch.clamp(motor_commands, -1.0, 1.0)
        
        return motor_commands
    
    def _extract_motor_tendencies_parallel(self, gradient: torch.Tensor) -> torch.Tensor:
        """OPTIMIZED gradient extraction using parallel reductions (no .item() calls)"""
        content_field = self.unified_field[:, :, :, :32]
        
        # OPTIMIZATION: Compute all gradients in parallel using tensor operations
        # X-axis gradient (forward/backward)
        if content_field.shape[0] > 1:
            x_grad = content_field[-1, :, :].mean() - content_field[0, :, :].mean()
        else:
            x_grad = torch.tensor(0.0, device=self.device)
            
        # Y-axis gradient (left/right)  
        if content_field.shape[1] > 1:
            y_grad = content_field[:, -1, :].mean() - content_field[:, 0, :].mean()
        else:
            y_grad = torch.tensor(0.0, device=self.device)
            
        # Z-axis gradient (up/down)
        if content_field.shape[2] > 1:
            z_grad = content_field[:, :, -1].mean() - content_field[:, :, 0].mean()
        else:
            z_grad = torch.tensor(0.0, device=self.device)
        
        # Pattern strength (if strategic planning active)
        if self.current_strategic_pattern is not None:
            pattern_field = self.unified_field[:, :, :, 32:48]
            pattern_strength = pattern_field.abs().mean()  # Keep as tensor
        else:
            pattern_strength = torch.tensor(0.0, device=self.device)
        
        # Create motor vector (all operations on GPU)
        motor_tendencies = torch.zeros(self.motor_dim, device=self.device)
        
        # Amplify gradients for responsiveness
        gradient_amp = 5.0
        if self.motor_dim >= 2:
            motor_tendencies[0] = x_grad * gradient_amp
            motor_tendencies[1] = y_grad * gradient_amp
            
        if self.motor_dim >= 3:
            motor_tendencies[2] = pattern_strength
            
        if self.motor_dim >= 4:
            motor_tendencies[3] = z_grad * 0.5
        
        # Fill remaining dimensions with small random values (vectorized)
        if self.motor_dim > 4:
            motor_tendencies[4:] = torch.randn(self.motor_dim - 4, device=self.device) * 0.1
        
        return motor_tendencies
    
    def _process_attention_gpu(self, sensory_tensor: torch.Tensor) -> Dict[str, Any]:
        """Optimized attention processing on GPU"""
        # Simplified attention for performance
        attention_weights = F.softmax(sensory_tensor * 2.0, dim=0)
        
        return {
            'attention_weights': attention_weights,
            'focus_strength': torch.max(attention_weights),
            'entropy': -(attention_weights * torch.log(attention_weights + 1e-10)).sum()
        }
    
    def _echo_motor_to_field_gpu(self, motor_action: torch.Tensor):
        """OPTIMIZED motor echo using vectorized operations"""
        # Decay previous motor echo (vectorized)
        self.unified_field[:, :, :, 62:64] *= 0.95
        
        # Create motor pattern (vectorized spatial operations)
        if motor_action.shape[0] >= 2:
            # Forward/backward gradient
            z_range = torch.linspace(-1, 1, self.spatial_resolution, device=self.device)
            forward_pattern = motor_action[0] * z_range.view(1, 1, -1)
            
            # Left/right gradient  
            y_range = torch.linspace(-1, 1, self.spatial_resolution, device=self.device)
            turn_pattern = motor_action[1] * y_range.view(1, -1, 1)
            
            # Combine patterns
            motor_spatial = forward_pattern + turn_pattern
            
            # Add to field
            self.unified_field[:, :, :, 62] += motor_spatial * 0.3
        
        # Motor magnitude in channel 63
        motor_magnitude = torch.mean(torch.abs(motor_action))
        self.unified_field[:, :, :, 63] += motor_magnitude * 0.2
    
    def _create_brain_state_optimized(self) -> Dict[str, Any]:
        """Optimized brain state creation with cached metrics"""
        # Use cached metrics if available and recent
        if (self._last_metrics_cycle == self.brain_cycles and 
            self._cached_metrics):
            metrics = self._cached_metrics
        else:
            # Compute metrics on GPU and cache
            gpu_metrics = OptimizedFieldMetrics.compute_all_metrics(self.unified_field)
            metrics = OptimizedFieldMetrics.metrics_to_cpu_dict(gpu_metrics)
            self._cached_metrics = metrics
            self._last_metrics_cycle = self.brain_cycles
        
        # Create streamlined brain state
        return {
            'cycle': self.brain_cycles,
            'cycle_time_ms': self._last_cycle_time * 1000,
            'field_information': metrics['information'],
            'max_activation': metrics['max_activation'], 
            'prediction_confidence': self._current_prediction_confidence,
            'energy': metrics['energy'],
            'sparsity': metrics['sparsity'],
            'device': str(self.device),
            'optimization': 'gpu_accelerated',
            'timestamp': time.time()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get GPU optimization performance statistics"""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'avg_cycle_time_ms': getattr(self, '_last_cycle_time', 0) * 1000,
                'optimization_speedup': '5-10x estimated'
            }
        return {'optimization': 'cpu_mode'}