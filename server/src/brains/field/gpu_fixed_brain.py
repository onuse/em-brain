"""
GPU-Optimized Brain with Performance Fixes

Fixes the critical performance issues:
1. Vectorized sensory injection (no loops)
2. Vectorized motor extraction (no loops)
3. Proper tensor operations without CPU roundtrips
"""

import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

from .simple_field_dynamics import SimpleFieldDynamics
from .simple_prediction import SimplePrediction
from .simple_learning import SimpleLearning
from .simple_persistence import SimplePersistence


class GPUOptimizedIntrinsicTensions:
    """GPU-optimized version of IntrinsicTensions."""
    
    def __init__(self, field_shape: Tuple[int, int, int, int], device: torch.device):
        self.field_shape = field_shape
        self.device = device
        
        # Comfort parameters
        self.resting_potential = 0.1
        self.min_gradient = 0.01
        self.max_flatness = 0.95
        self.comfort_variance = 0.05
        
        # Oscillation parameters
        self.base_frequency = 0.1
        self.frequency_variance = 0.02
        
        # Create frequency map
        self.frequency_map = self.base_frequency + torch.randn(field_shape, device=device) * self.frequency_variance
        
        # Phase accumulator
        self.phase = torch.zeros(field_shape, device=device)
        
        # Asymmetric decay map
        self.decay_map = 0.995 + torch.randn(field_shape, device=device) * 0.005
        self.decay_map = torch.clamp(self.decay_map, 0.98, 1.0)
        
        self.cycle = 0
    
    def apply_tensions(self, field: torch.Tensor, prediction_error: float = 0.0) -> torch.Tensor:
        self.cycle += 1
        
        # 1. Resting potential
        field_mean = field.mean()
        starvation = (self.resting_potential - field_mean) * 0.01
        field = field + starvation
        
        # 2. Gradient hunger (optimized)
        var_per_channel = torch.var(field, dim=-1, keepdim=True)
        local_variance = var_per_channel.expand_as(field)
        boredom_mask = local_variance < self.min_gradient
        boredom_noise = torch.randn_like(field) * 0.02
        field = torch.where(boredom_mask, field + boredom_noise, field)
        
        # 3. Oscillatory drive
        self.phase += self.frequency_map
        oscillation = 0.01 * torch.sin(self.phase) * (1 + torch.abs(field))
        field = field + oscillation
        
        # 4. Asymmetric decay
        field = field * self.decay_map
        
        # 5. Prediction error tension
        if prediction_error > 0.01:
            error_heat = torch.randn_like(field) * prediction_error * 0.1
            field = field + error_heat
            self.phase += torch.randn_like(self.phase) * prediction_error
        
        # 6. Edge detection (optimized gradient computation)
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        gradient_magnitude = torch.sqrt(dx**2 + dy**2 + dz**2)
        edge_enhancement = gradient_magnitude * 0.01
        field = field + edge_enhancement
        
        # 7. Information starvation
        activity_level = torch.abs(field).mean()
        if activity_level < 0.05:
            starvation_energy = (0.05 - activity_level) * 10
            field = field + torch.randn_like(field) * starvation_energy * 0.05
        
        return field
    
    def get_comfort_metrics(self, field: torch.Tensor) -> Dict[str, float]:
        """Optimized comfort metrics computation."""
        field_mean = field.mean().item()
        field_var = field.var().item()
        activity = torch.abs(field).mean().item()
        
        # Compute local variance
        var_per_channel = torch.var(field, dim=-1, keepdim=True)
        local_var = var_per_channel.mean().item()
        
        # Compute comfort scores efficiently
        resting_comfort = 1.0 - abs(field_mean - self.resting_potential) / self.resting_potential
        variance_comfort = min(local_var / self.comfort_variance, 1.0)
        activity_comfort = min(activity / 0.1, 1.0)
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
    
    def reset(self):
        self.phase = torch.zeros(self.field_shape, device=self.device)
        self.cycle = 0


class GPUOptimizedMotorExtraction:
    """GPU-optimized motor extraction without loops."""
    
    def __init__(self, motor_dim: int, device: torch.device, field_size: int = 16):
        self.motor_dim = motor_dim
        self.device = device
        self.field_size = field_size
        
        # Random motor regions
        self.motor_regions = torch.randint(0, field_size, (motor_dim, 3), device=device)
        
        # Pre-compute region bounds (vectorized)
        self.x_coords = self.motor_regions[:, 0]
        self.y_coords = self.motor_regions[:, 1]
        self.z_coords = self.motor_regions[:, 2]
        
        # Pre-allocate tensors for efficiency
        self.motor_values = torch.zeros(motor_dim, device=device)
        self.motor_directions = torch.zeros(motor_dim, device=device)
    
    def extract_motors(self, field: torch.Tensor) -> list:
        """Optimized motor extraction."""
        # Compute gradients once
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        
        # Gradient magnitude
        gradient_mag = torch.sqrt(dx**2 + dy**2 + dz**2).mean(dim=3)
        
        # VECTORIZED: Extract values at motor regions using advanced indexing
        # This replaces the loop with a single GPU operation
        x_idx = torch.clamp(self.x_coords, 0, gradient_mag.shape[0] - 1)
        y_idx = torch.clamp(self.y_coords, 0, gradient_mag.shape[1] - 1)
        z_idx = torch.clamp(self.z_coords, 0, gradient_mag.shape[2] - 1)
        
        # Get gradient magnitudes at motor points (single GPU op)
        self.motor_values = gradient_mag[x_idx, y_idx, z_idx]
        
        # Get directional components (vectorized)
        # Create masks for motor types
        forward_mask = torch.zeros(self.motor_dim, device=self.device, dtype=torch.bool)
        lateral_mask = torch.zeros(self.motor_dim, device=self.device, dtype=torch.bool)
        other_mask = torch.zeros(self.motor_dim, device=self.device, dtype=torch.bool)
        
        if self.motor_dim > 0:
            forward_mask[0] = True
        if self.motor_dim > 1:
            lateral_mask[1] = True
        if self.motor_dim > 2:
            other_mask[2:] = True
        
        # Get directions using masks (avoids if/else in loop)
        dx_vals = dx[x_idx, y_idx, z_idx].mean(dim=-1) if dx.dim() > 3 else dx[x_idx, y_idx, z_idx]
        dy_vals = dy[x_idx, y_idx, z_idx].mean(dim=-1) if dy.dim() > 3 else dy[x_idx, y_idx, z_idx]
        dz_vals = dz[x_idx, y_idx, z_idx].mean(dim=-1) if dz.dim() > 3 else dz[x_idx, y_idx, z_idx]
        
        self.motor_directions = torch.where(forward_mask, dx_vals,
                                           torch.where(lateral_mask, dy_vals, dz_vals))
        
        # Combine and apply tanh
        motor_commands = torch.tanh(self.motor_directions * self.motor_values * 10)
        
        # Single CPU transfer at the end
        return motor_commands.cpu().tolist()
    
    def get_motor_state(self, motors: list) -> str:
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


class GPUFixedBrain:
    """GPU-optimized brain with all performance fixes."""
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_size: int = 96,
                 channels: int = 192,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        
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
            print(f"GPU-Optimized Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
        
        # Initialize field
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        self.field_momentum = torch.zeros_like(self.field)
        
        # Initialize optimized components
        self.dynamics = SimpleFieldDynamics()
        self.tensions = GPUOptimizedIntrinsicTensions(self.field.shape, self.device)
        self.prediction = SimplePrediction(sensory_dim, self.device, channels)
        self.learning = SimpleLearning(self.device)
        self.motor = GPUOptimizedMotorExtraction(motor_dim, self.device, spatial_size)
        self.persistence = SimplePersistence()
        
        # CRITICAL OPTIMIZATION: Pre-compute sensor injection indices
        self.sensor_spots = torch.randint(0, spatial_size, (sensory_dim, 3), device=self.device)
        
        # VECTORIZED INJECTION: Create index tensors for scatter_add
        # This replaces the loop with a single GPU operation
        self.sensor_x = self.sensor_spots[:, 0]
        self.sensor_y = self.sensor_spots[:, 1]
        self.sensor_z = self.sensor_spots[:, 2]
        self.sensor_c = torch.arange(sensory_dim, device=self.device) % 8
        
        # Create flattened indices for 1D scatter operation
        # This converts 4D indices to 1D for maximum efficiency
        self.sensor_flat_idx = (
            self.sensor_x * (spatial_size * spatial_size * channels) +
            self.sensor_y * (spatial_size * channels) +
            self.sensor_z * channels +
            self.sensor_c
        )
        
        # State tracking
        self.cycle = 0
        self.last_prediction = None
        
        if not quiet_mode:
            print("GPU optimizations enabled with vectorized operations")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Optimized processing cycle."""
        start_time = time.perf_counter()
        self.cycle += 1
        
        # Convert input
        sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                              dtype=torch.float32, device=self.device)
        
        # ===== 1. VECTORIZED SENSORY INJECTION =====
        # This is the key optimization - no Python loop!
        # Use scatter_add for vectorized injection
        injection_values = sensors * 0.3
        
        # Flatten field for efficient scatter
        field_flat = self.field.view(-1)
        
        # Inject all sensors at once using scatter_add
        field_flat.scatter_add_(0, self.sensor_flat_idx[:len(sensors)], injection_values)
        
        # Reshape back (view doesn't copy, just changes shape)
        self.field = field_flat.view(self.spatial_size, self.spatial_size, 
                                     self.spatial_size, self.channels)
        
        # ===== 2. LEARNING FROM PREDICTION ERROR =====
        if self.last_prediction is not None:
            error = self.prediction.compute_error(self.last_prediction, sensors)
            tension = self.learning.error_to_field_tension(error, self.field)
            self.field = self.field + tension
            self.prediction.learn_from_error(error, self.field)
            error_magnitude = torch.abs(error).mean().item()
        else:
            error_magnitude = 0.0
        
        # ===== 3. INTRINSIC TENSIONS =====
        self.field = self.tensions.apply_tensions(self.field, error_magnitude)
        
        # ===== 3.5. FIELD MOMENTUM =====
        self.field_momentum = 0.9 * self.field_momentum + 0.1 * self.field
        self.field = self.field + self.field_momentum * 0.05
        
        # ===== 4. FIELD EVOLUTION =====
        exploration = self.learning.should_explore()
        if exploration:
            noise = torch.randn_like(self.field) * 0.02
        else:
            noise = None
        
        self.field = self.dynamics.evolve(self.field, noise)
        
        # ===== 5. MOTOR EXTRACTION (optimized) =====
        motor_output = self.motor.extract_motors(self.field)
        
        # ===== 6. PREDICTION =====
        self.last_prediction = self.prediction.predict_next_sensors(self.field)
        self.prediction.update_history(sensors)
        
        # ===== TELEMETRY =====
        comfort = self.tensions.get_comfort_metrics(self.field)
        
        telemetry = {
            'cycle': self.cycle,
            'time_ms': (time.perf_counter() - start_time) * 1000,
            'energy': self.dynamics.get_energy(self.field),
            'variance': self.dynamics.get_variance(self.field),
            'comfort': comfort['overall_comfort'],
            'motivation': self._interpret_state(comfort),
            'learning': self.learning.get_learning_state(),
            'motor': self.motor.get_motor_state(motor_output),
            'exploring': exploration,
            'momentum': torch.abs(self.field_momentum).mean().item()
        }
        
        # Periodic logging
        if self.cycle % 100 == 0 and not self.quiet_mode:
            print(f"Cycle {self.cycle}: {telemetry['motivation']}, "
                  f"{telemetry['motor']}, {telemetry['learning']}")
        
        # Auto-save
        self.persistence.auto_save(self, interval=1000)
        
        return motor_output, telemetry
    
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
    
    def save(self, name: Optional[str] = None) -> str:
        return self.persistence.save(self, name)
    
    def load(self, name: str) -> bool:
        return self.persistence.load(self, name)
    
    def reset(self):
        self.field = torch.randn(self.spatial_size, self.spatial_size, 
                                self.spatial_size, self.channels, device=self.device) * 0.01
        self.field_momentum = torch.zeros_like(self.field)
        self.cycle = 0
        self.last_prediction = None
        self.tensions.reset()
        
        if not self.quiet_mode:
            print("Brain reset")