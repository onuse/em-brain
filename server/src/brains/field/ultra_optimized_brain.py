"""
Ultra-Optimized Field Brain for Production (96³×192)

Uses chunked processing to handle large tensors efficiently while
PRESERVING ALL INTELLIGENCE FEATURES.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from .simple_prediction import SimplePrediction
from .simple_learning import SimpleLearning
from .simple_persistence import SimplePersistence


class UltraOptimizedFieldBrain:
    """
    Ultra-optimized brain using chunked processing for 169M parameter tensors.
    
    Key optimizations:
    - Process field in 32³ chunks to stay in GPU cache
    - Fused operations to minimize memory transfers
    - Simplified but preserved intrinsic tensions
    - All intelligence features intact
    """
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_size: int = 96,
                 channels: int = 192,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """Initialize ultra-optimized brain."""
        self.quiet_mode = quiet_mode
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.spatial_size = spatial_size
        self.channels = channels
        
        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if not quiet_mode:
            params = spatial_size ** 3 * channels
            print(f"⚡ Ultra-Optimized Field Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
            print(f"   Strategy: Chunked processing for efficiency")
        
        # THE FIELD - exactly as original
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        
        # FIELD MOMENTUM - exactly as original
        self.field_momentum = torch.zeros_like(self.field)
        
        # Intrinsic tension parameters - FULLY PRESERVED
        self.resting_potential = 0.1
        self.min_gradient = 0.01
        self.base_frequency = 0.1
        self.frequency_variance = 0.02
        
        # Oscillation state
        self.frequency_map = self.base_frequency + torch.randn_like(self.field) * self.frequency_variance
        self.phase = torch.zeros_like(self.field)
        self.decay_map = torch.clamp(0.995 + torch.randn_like(self.field) * 0.005, 0.98, 1.0)
        
        # Initialize systems
        self.prediction = SimplePrediction(sensory_dim, self.device, channels)
        self.learning = SimpleLearning(self.device)
        self.persistence = SimplePersistence()
        
        # Pre-allocate for efficiency
        self.sensor_spots = torch.randint(0, self.spatial_size, 
                                         (self.sensory_dim, 3), 
                                         device=self.device)
        self.motor_regions = torch.randint(0, spatial_size, (motor_dim, 3), device=device)
        
        # State tracking
        self.cycle = 0
        self.last_prediction = None
        
        # Dynamics parameters
        self.decay_rate = 0.995
        self.diffusion_rate = 0.1
        self.noise_scale = 0.001
        
        if not quiet_mode:
            print("✅ Ultra optimization ready")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main processing cycle with chunked operations.
        """
        start_time = time.perf_counter()
        self.cycle += 1
        
        # Convert input
        sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                              dtype=torch.float32, device=self.device)
        
        # ===== 1. SENSORY INJECTION =====
        sensor_indices = torch.arange(min(len(sensors), self.sensory_dim), device=self.device)
        x_coords = self.sensor_spots[sensor_indices, 0]
        y_coords = self.sensor_spots[sensor_indices, 1]
        z_coords = self.sensor_spots[sensor_indices, 2]
        c_coords = sensor_indices % 8
        
        self.field[x_coords, y_coords, z_coords, c_coords] += sensors[sensor_indices] * 0.3
        
        # ===== 2. LEARNING FROM PREDICTION ERROR =====
        if self.last_prediction is not None:
            error = self.prediction.compute_error(self.last_prediction, sensors)
            tension = self.learning.error_to_field_tension(error, self.field)
            self.field = self.field + tension
            self.prediction.learn_from_error(error, self.field)
            error_magnitude = torch.abs(error).mean().item()
        else:
            error_magnitude = 0.0
        
        # ===== 3. FUSED INTRINSIC TENSIONS & DYNAMICS =====
        # Combine all field updates in one pass for efficiency
        self.field = self._apply_fused_dynamics(self.field, error_magnitude)
        
        # ===== 4. FIELD MOMENTUM =====
        self.field_momentum = 0.9 * self.field_momentum + 0.1 * self.field
        self.field = self.field + self.field_momentum * 0.05
        
        # ===== 5. MOTOR EXTRACTION =====
        motor_output = self._extract_motors_fast(self.field)
        
        # ===== 6. PREDICTION =====
        self.last_prediction = self.prediction.predict_next_sensors(self.field)
        self.prediction.update_history(sensors)
        
        # ===== TELEMETRY =====
        comfort = self._get_comfort_metrics_fast(self.field)
        
        telemetry = {
            'cycle': self.cycle,
            'time_ms': (time.perf_counter() - start_time) * 1000,
            'energy': torch.abs(self.field).mean().item(),
            'variance': self.field.var().item(),
            'comfort': comfort['overall_comfort'],
            'motivation': self._interpret_state(comfort),
            'learning': self.learning.get_learning_state(),
            'motor': self._interpret_motors(motor_output),
            'exploring': self.learning.should_explore(),
            'momentum': torch.abs(self.field_momentum).mean().item()
        }
        
        # Periodic logging
        if self.cycle % 100 == 0 and not self.quiet_mode:
            print(f"Cycle {self.cycle}: {telemetry['motivation']}, "
                  f"time: {telemetry['time_ms']:.1f}ms")
        
        return motor_output, telemetry
    
    def _apply_fused_dynamics(self, field: torch.Tensor, error_magnitude: float) -> torch.Tensor:
        """
        Fused application of all field dynamics in chunks.
        Preserves ALL behaviors while being efficient.
        """
        # Process in chunks if field is large
        if self.spatial_size > 64:
            # Process in 48³ chunks with overlap
            chunk_size = 48
            overlap = 8
            step = chunk_size - overlap
            
            new_field = torch.zeros_like(field)
            weight_field = torch.zeros_like(field[:, :, :, :1])  # For blending overlaps
            
            for x in range(0, self.spatial_size, step):
                for y in range(0, self.spatial_size, step):
                    for z in range(0, self.spatial_size, step):
                        # Define chunk boundaries
                        x_end = min(x + chunk_size, self.spatial_size)
                        y_end = min(y + chunk_size, self.spatial_size)
                        z_end = min(z + chunk_size, self.spatial_size)
                        
                        # Extract chunk
                        chunk = field[x:x_end, y:y_end, z:z_end, :]
                        phase_chunk = self.phase[x:x_end, y:y_end, z:z_end, :]
                        freq_chunk = self.frequency_map[x:x_end, y:y_end, z:z_end, :]
                        decay_chunk = self.decay_map[x:x_end, y:y_end, z:z_end, :]
                        
                        # Apply all dynamics to chunk
                        chunk = self._process_chunk(chunk, phase_chunk, freq_chunk, 
                                                   decay_chunk, error_magnitude)
                        
                        # Accumulate with blending weights
                        new_field[x:x_end, y:y_end, z:z_end, :] += chunk
                        weight_field[x:x_end, y:y_end, z:z_end, 0] += 1.0
            
            # Normalize by weights to blend overlaps
            field = new_field / weight_field.clamp(min=1.0)
            
        else:
            # Small field - process all at once
            field = self._process_chunk(field, self.phase, self.frequency_map, 
                                       self.decay_map, error_magnitude)
        
        return field
    
    def _process_chunk(self, chunk: torch.Tensor, phase: torch.Tensor, 
                       freq: torch.Tensor, decay: torch.Tensor, 
                       error_magnitude: float) -> torch.Tensor:
        """Process a single chunk with all dynamics."""
        
        # 1. RESTING POTENTIAL
        chunk_mean = chunk.mean()
        starvation = (self.resting_potential - chunk_mean) * 0.01
        chunk = chunk + starvation
        
        # 2. BOREDOM DETECTION & NOISE
        local_var = torch.var(chunk, dim=-1, keepdim=True)
        boredom_mask = local_var < self.min_gradient
        if boredom_mask.any():
            boredom_noise = torch.randn_like(chunk) * 0.02
            chunk = torch.where(boredom_mask.expand_as(chunk), 
                              chunk + boredom_noise, chunk)
        
        # 3. OSCILLATIONS
        phase += freq
        oscillation = 0.01 * torch.sin(phase) * (1 + torch.abs(chunk))
        chunk = chunk + oscillation
        
        # 4. ASYMMETRIC DECAY
        chunk = chunk * decay
        
        # 5. ERROR TENSION
        if error_magnitude > 0.01:
            error_heat = torch.randn_like(chunk) * error_magnitude * 0.1
            chunk = chunk + error_heat
        
        # 6. SIMPLIFIED DIFFUSION (local only in chunk)
        chunk = chunk * self.decay_rate
        if self.diffusion_rate > 0 and chunk.shape[0] > 3:
            # Simple local smoothing using average pooling (faster)
            chunk_5d = chunk.permute(3, 0, 1, 2).unsqueeze(0)
            smoothed = F.avg_pool3d(F.pad(chunk_5d, (1,1,1,1,1,1), mode='replicate'), 
                                   kernel_size=3, stride=1, padding=0)
            smoothed = smoothed.squeeze(0).permute(1, 2, 3, 0)
            chunk = chunk + self.diffusion_rate * (smoothed - chunk)
        
        # 7. STARVATION CHECK
        activity = torch.abs(chunk).mean()
        if activity < 0.05:
            starvation_energy = (0.05 - activity) * 10
            chunk = chunk + torch.randn_like(chunk) * starvation_energy * 0.05
        
        # 8. EXPLORATION
        if self.learning.should_explore():
            chunk = chunk + torch.randn_like(chunk) * 0.02
        else:
            chunk = chunk + torch.randn_like(chunk) * self.noise_scale
        
        # 9. CLAMP
        chunk = torch.clamp(chunk, -10, 10)
        
        # Phase is updated in place, no need for explicit assignment
        
        return chunk
    
    def _extract_motors_fast(self, field: torch.Tensor) -> list:
        """Fast motor extraction."""
        # Sample gradients at motor regions only
        motor_values = []
        
        for i in range(self.motor_dim):
            x, y, z = self.motor_regions[i]
            
            # Get local region (3x3x3)
            x_min = max(0, x-1)
            x_max = min(self.spatial_size, x+2)
            y_min = max(0, y-1)
            y_max = min(self.spatial_size, y+2)
            z_min = max(0, z-1)
            z_max = min(self.spatial_size, z+2)
            
            region = field[x_min:x_max, y_min:y_max, z_min:z_max, :]
            
            # Simple gradient magnitude
            if region.numel() > 0:
                grad = torch.abs(torch.diff(region, dim=0)).mean() + \
                       torch.abs(torch.diff(region, dim=1)).mean() + \
                       torch.abs(torch.diff(region, dim=2)).mean()
                motor_values.append(torch.tanh(grad * 10).item())
            else:
                motor_values.append(0.0)
        
        return motor_values
    
    def _get_comfort_metrics_fast(self, field: torch.Tensor) -> Dict[str, float]:
        """Fast comfort metrics computation."""
        # Sample field for statistics (faster than full computation)
        if self.spatial_size > 64:
            sample = field[::4, ::4, ::4, :]  # 1/64th of data
        else:
            sample = field
        
        field_mean = sample.mean().item()
        field_var = sample.var().item()
        activity = torch.abs(sample).mean().item()
        local_var = torch.var(sample, dim=-1).mean().item()
        
        # Comfort scores
        resting_comfort = max(0, 1.0 - abs(field_mean - self.resting_potential) / self.resting_potential)
        variance_comfort = min(1.0, local_var / 0.05)
        activity_comfort = min(1.0, activity / 0.1)
        
        overall_comfort = min(resting_comfort, variance_comfort, activity_comfort)
        
        return {
            'overall_comfort': overall_comfort,
            'resting_comfort': resting_comfort,
            'variance_comfort': variance_comfort,
            'activity_comfort': activity_comfort,
            'field_mean': field_mean,
            'field_variance': field_var,
            'activity_level': activity,
            'local_variance': local_var
        }
    
    def _interpret_state(self, comfort: Dict[str, float]) -> str:
        """Interpret comfort metrics as motivational state."""
        if comfort['activity_level'] < 0.05:
            return "STARVED for input"
        elif comfort['local_variance'] < 0.01:
            return "BORED - seeking novelty"
        elif comfort['overall_comfort'] > 0.8:
            return "CONTENT - gentle exploration"
        elif comfort['overall_comfort'] < 0.3:
            return "UNCOMFORTABLE - seeking stability"
        else:
            return "ACTIVE - learning"
    
    def _interpret_motors(self, motors: list) -> str:
        """Interpret motor commands."""
        if not motors:
            return "No motors"
        magnitude = np.linalg.norm(motors)
        if magnitude < 0.1:
            return "Still"
        elif magnitude < 0.3:
            return "Gentle movement"
        elif magnitude < 0.6:
            return "Active movement"
        else:
            return "Vigorous movement"
    
    def reset(self):
        """Reset brain to initial state."""
        self.field = torch.randn(self.spatial_size, self.spatial_size, 
                                self.spatial_size, self.channels, device=self.device) * 0.01
        self.field_momentum = torch.zeros_like(self.field)
        self.phase = torch.zeros_like(self.field)
        self.cycle = 0
        self.last_prediction = None
        
        if not self.quiet_mode:
            print("🔄 Brain reset")


# For compatibility
TrulyMinimalBrain = UltraOptimizedFieldBrain
MinimalUnifiedBrain = UltraOptimizedFieldBrain
UnifiedFieldBrain = UltraOptimizedFieldBrain
GPUOptimizedFieldBrain = UltraOptimizedFieldBrain