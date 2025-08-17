"""
Large Field Implementation

Optimized implementation for large tensor fields (>64³) that:
1. Computes expensive operations on downsampled tensors
2. Temporally interleaves costly operations
3. Uses vectorized sensor/motor operations
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

from .simple_field_dynamics import SimpleFieldDynamics
from .simple_prediction import SimplePrediction
from .simple_learning import SimpleLearning
from .simple_persistence import SimplePersistence


class LargeFieldImplementation:
    """Optimized implementation for large field tensors."""
    
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
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if not quiet_mode:
            params = spatial_size ** 3 * channels
            print(f"Ultra-Fast GPU Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
            print(f"   Target: <200ms per cycle")
        
        # Initialize field
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        self.field_momentum = torch.zeros_like(self.field)
        
        # Intrinsic parameters
        self.resting_potential = 0.1
        self.min_gradient = 0.01
        self.decay_rate = 0.995
        
        # For large tensors, use lower-res maps
        if spatial_size > 64:
            # 2x downsampled for efficiency
            ds_size = spatial_size // 2
            self.frequency_map = 0.1 + torch.randn(ds_size, ds_size, ds_size, channels, 
                                                  device=self.device) * 0.02
            self.phase = torch.zeros_like(self.frequency_map)
            self.decay_map = torch.clamp(0.995 + torch.randn_like(self.frequency_map) * 0.005, 
                                        0.98, 1.0)
            self.use_downsampled = True
        else:
            self.frequency_map = 0.1 + torch.randn_like(self.field) * 0.02
            self.phase = torch.zeros_like(self.field)
            self.decay_map = torch.clamp(0.995 + torch.randn_like(self.field) * 0.005, 
                                        0.98, 1.0)
            self.use_downsampled = False
        
        # Systems
        self.dynamics = SimpleFieldDynamics()
        self.prediction = SimplePrediction(sensory_dim, self.device, channels)
        self.learning = SimpleLearning(self.device)
        self.persistence = SimplePersistence()
        
        # VECTORIZED SENSOR INJECTION
        self.sensor_spots = torch.randint(0, spatial_size, (sensory_dim, 3), device=self.device)
        self.sensor_x = self.sensor_spots[:, 0]
        self.sensor_y = self.sensor_spots[:, 1]
        self.sensor_z = self.sensor_spots[:, 2]
        self.sensor_c = torch.arange(sensory_dim, device=self.device) % 8
        
        # Flattened indices for scatter operation
        self.sensor_flat_idx = (
            self.sensor_x * (spatial_size * spatial_size * channels) +
            self.sensor_y * (spatial_size * channels) +
            self.sensor_z * channels +
            self.sensor_c
        )
        
        # VECTORIZED MOTOR EXTRACTION
        self.motor_regions = torch.randint(0, spatial_size, (motor_dim, 3), device=self.device)
        self.motor_x = self.motor_regions[:, 0]
        self.motor_y = self.motor_regions[:, 1]
        self.motor_z = self.motor_regions[:, 2]
        
        # State tracking
        self.cycle = 0
        self.last_prediction = None
        
        # Temporal interleaving counters
        self.gradient_cycle = 0  # Compute gradients every N cycles
        self.variance_cycle = 0  # Compute variance every N cycles
        
        if not quiet_mode:
            print("All optimizations enabled")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Ultra-fast processing cycle."""
        start_time = time.perf_counter()
        self.cycle += 1
        
        # Convert input
        sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                              dtype=torch.float32, device=self.device)
        
        # ===== 1. VECTORIZED SENSORY INJECTION (fast) =====
        injection_values = sensors * 0.3
        field_flat = self.field.view(-1)
        field_flat.scatter_add_(0, self.sensor_flat_idx[:len(sensors)], injection_values)
        self.field = field_flat.view(self.spatial_size, self.spatial_size, 
                                     self.spatial_size, self.channels)
        
        # ===== 2. LEARNING (preserved) =====
        error_magnitude = 0.0
        if self.last_prediction is not None:
            error = self.prediction.compute_error(self.last_prediction, sensors)
            error_magnitude = torch.abs(error).mean().item()
            if error_magnitude > 0.01:
                # Simple error injection
                self.field += torch.randn_like(self.field) * error_magnitude * 0.15
            self.prediction.learn_from_error(error, self.field)
        
        # ===== 3. INTERLEAVED DYNAMICS =====
        # Spread expensive operations across cycles
        
        # Always do basic dynamics
        self._apply_basic_dynamics(error_magnitude)
        
        # Expensive operations on rotation
        if self.cycle % 5 == 0:
            # Gradients (most expensive) - do rarely and on downsampled
            self._apply_gradient_enhancement()
        elif self.cycle % 5 == 1:
            # Variance/boredom check
            self._apply_boredom_injection()
        elif self.cycle % 5 == 2:
            # Oscillations
            self._apply_oscillations()
        
        # ===== 4. MOMENTUM (always) =====
        self.field_momentum = 0.9 * self.field_momentum + 0.1 * self.field
        self.field = self.field + self.field_momentum * 0.05
        
        # ===== 5. VECTORIZED MOTOR EXTRACTION (fast) =====
        motor_output = self._extract_motors_vectorized()
        
        # ===== 6. PREDICTION (preserved) =====
        self.last_prediction = self.prediction.predict_next_sensors(self.field)
        self.prediction.update_history(sensors)
        
        # ===== TELEMETRY =====
        # Fast approximation most cycles
        if self.cycle % 10 == 0:
            comfort = self._get_full_comfort_metrics()
        else:
            comfort = self._get_fast_comfort_metrics()
        
        telemetry = {
            'cycle': self.cycle,
            'time_ms': (time.perf_counter() - start_time) * 1000,
            'energy': torch.abs(self.field).mean().item(),
            'variance': self.field.var().item(),
            'comfort': comfort,
            'motivation': self._get_motivation(comfort, error_magnitude),
            'learning': 'Active' if error_magnitude > 0.1 else 'Stable',
            'motor': 'Moving' if np.linalg.norm(motor_output) > 0.1 else 'Still',
            'exploring': error_magnitude > 0.3,
            'momentum': torch.abs(self.field_momentum).mean().item()
        }
        
        # Logging
        if self.cycle % 100 == 0 and not self.quiet_mode:
            print(f"Cycle {self.cycle}: time={telemetry['time_ms']:.1f}ms")
        
        # Auto-save
        self.persistence.auto_save(self, interval=1000)
        
        return motor_output, telemetry
    
    def _apply_basic_dynamics(self, error_magnitude: float):
        """Basic dynamics - fast every-cycle operations."""
        # Decay
        self.field *= self.decay_rate
        
        # Resting potential
        field_mean = self.field.mean()
        self.field += (self.resting_potential - field_mean) * 0.01
        
        # Basic noise
        if error_magnitude > 0.1 or torch.abs(self.field).mean() < 0.05:
            self.field += torch.randn_like(self.field) * 0.02
        else:
            self.field += torch.randn_like(self.field) * 0.001
        
        # Clamp
        self.field = torch.clamp(self.field, -10, 10)
    
    def _apply_gradient_enhancement(self):
        """Apply gradient enhancement (expensive, do rarely)."""
        if self.spatial_size > 64:
            # Downsample for gradient computation
            downsampled = F.avg_pool3d(
                self.field.permute(3, 0, 1, 2).unsqueeze(0),
                kernel_size=2, stride=2
            ).squeeze(0).permute(1, 2, 3, 0)
            
            # Compute gradients on smaller tensor (8x faster)
            dx = torch.diff(downsampled, dim=0, prepend=downsampled[:1])
            dy = torch.diff(downsampled, dim=1, prepend=downsampled[:, :1])
            dz = torch.diff(downsampled, dim=2, prepend=downsampled[:, :, :1])
            
            # Simple gradient magnitude (avoid sqrt for speed)
            gradient_mag = torch.abs(dx) + torch.abs(dy) + torch.abs(dz)
            
            # Upsample gradient
            gradient_up = F.interpolate(
                gradient_mag.permute(3, 0, 1, 2).unsqueeze(0),
                size=(self.spatial_size, self.spatial_size, self.spatial_size),
                mode='trilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 3, 0)
            
            # Apply enhancement
            self.field += gradient_up * 0.005
        else:
            # Direct computation for smaller fields
            dx = torch.diff(self.field, dim=0, prepend=self.field[:1])
            dy = torch.diff(self.field, dim=1, prepend=self.field[:, :1])
            dz = torch.diff(self.field, dim=2, prepend=self.field[:, :, :1])
            gradient_mag = torch.abs(dx) + torch.abs(dy) + torch.abs(dz)
            self.field += gradient_mag * 0.005
    
    def _apply_boredom_injection(self):
        """Check for boredom and inject noise if needed."""
        # Sample-based variance check for efficiency
        if self.spatial_size > 64:
            sample = self.field[::4, ::4, ::4, :]
            local_var = sample.var()
        else:
            local_var = self.field.var()
        
        if local_var < self.min_gradient:
            # BORED - inject noise
            self.field += torch.randn_like(self.field) * 0.03
    
    def _apply_oscillations(self):
        """Apply oscillatory dynamics."""
        if self.use_downsampled:
            # Update low-res phase
            self.phase += self.frequency_map
            
            # Upsample phase
            phase_up = F.interpolate(
                self.phase.permute(3, 0, 1, 2).unsqueeze(0),
                size=(self.spatial_size, self.spatial_size, self.spatial_size),
                mode='trilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 3, 0)
            
            # Apply oscillation (simplified - no abs for speed)
            oscillation = 0.01 * torch.sin(phase_up)
            self.field += oscillation
            
            # Apply decay map
            decay_up = F.interpolate(
                self.decay_map.permute(3, 0, 1, 2).unsqueeze(0),
                size=(self.spatial_size, self.spatial_size, self.spatial_size),
                mode='nearest'
            ).squeeze(0).permute(1, 2, 3, 0)
            self.field *= decay_up
        else:
            # Direct application
            self.phase += self.frequency_map
            oscillation = 0.01 * torch.sin(self.phase)
            self.field += oscillation
            self.field *= self.decay_map
    
    def _extract_motors_vectorized(self) -> list:
        """Vectorized motor extraction (no loops)."""
        # Sample field values at motor regions
        motor_values = self.field[self.motor_x, self.motor_y, self.motor_z].mean(dim=1)
        
        # Simple tanh squashing
        motor_commands = torch.tanh(motor_values[:self.motor_dim] * 10)
        
        return motor_commands.cpu().tolist()
    
    def _get_fast_comfort_metrics(self) -> float:
        """Fast comfort approximation."""
        # Just return a simple metric
        activity = torch.abs(self.field).mean().item()
        comfort = min(1.0, activity / 0.1)  # Want ~0.1 activity
        return comfort
    
    def _get_full_comfort_metrics(self) -> float:
        """Full comfort calculation (do rarely)."""
        field_mean = self.field.mean().item()
        activity = torch.abs(self.field).mean().item()
        
        # Sample-based variance
        if self.spatial_size > 64:
            sample = self.field[::4, ::4, ::4, :]
            local_var = sample.var().item()
        else:
            local_var = self.field.var().item()
        
        # Comfort scores
        resting_comfort = 1.0 - abs(field_mean - self.resting_potential) / self.resting_potential
        variance_comfort = min(1.0, local_var / 0.05)
        activity_comfort = min(1.0, activity / 0.1)
        
        return min(resting_comfort, variance_comfort, activity_comfort)
    
    def _get_motivation(self, comfort: float, error: float) -> str:
        """Get motivation string."""
        if comfort < 0.3:
            return "UNCOMFORTABLE - seeking stability"
        elif error > 0.3:
            return "SURPRISED - adapting"
        elif comfort > 0.8:
            return "CONTENT - gentle exploration"
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
        self.phase = torch.zeros_like(self.frequency_map)
        
        if not self.quiet_mode:
            print("Brain reset")