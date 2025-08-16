"""
Truly Minimal Field Brain

Absolutely minimal implementation using only simple, understandable components.
Total: ~250 lines for a complete emergent intelligence system.
"""

import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

# Import our simple components
from .simple_field_dynamics import SimpleFieldDynamics
from .simple_prediction import SimplePrediction
from .simple_learning import SimpleLearning
from .simple_motor import SimpleMotorExtraction
from .intrinsic_tensions import IntrinsicTensions
from .simple_persistence import SimplePersistence


class TrulyMinimalBrain:
    """
    The absolute minimal brain for emergent intelligence.
    
    Just 5 simple systems:
    1. Field dynamics (physics)
    2. Intrinsic tensions (motivation) 
    3. Prediction (next state)
    4. Learning (error → tension)
    5. Motor extraction (gradients → action)
    
    Everything else emerges.
    """
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_size: int = 16,  # Smaller default for simplicity
                 channels: int = 32,  # Number of field channels
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """
        Initialize minimal brain.
        
        Args:
            sensory_dim: Number of sensors
            motor_dim: Number of motors
            spatial_size: Spatial dimensions (16 → 16³×32 field)
            device: Computation device
            quiet_mode: Suppress output
        """
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
            print(f"🧠 Truly Minimal Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
        
        # THE FIELD - a 4D tensor where everything happens
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        
        # Initialize simple systems
        self.dynamics = SimpleFieldDynamics()
        self.tensions = IntrinsicTensions(self.field.shape, self.device)
        self.prediction = SimplePrediction(sensory_dim, self.device, channels)
        self.learning = SimpleLearning(self.device)
        self.motor = SimpleMotorExtraction(motor_dim, self.device, spatial_size)
        self.persistence = SimplePersistence()
        
        # State tracking
        self.cycle = 0
        self.last_prediction = None
        
        if not quiet_mode:
            print("✅ Brain initialized with intrinsic motivation")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main processing cycle.
        
        1. Inject sensors
        2. Apply tensions (motivation)
        3. Evolve field (physics)
        4. Extract motors
        5. Predict next
        6. Learn from error
        
        Args:
            sensory_input: Sensor values
            
        Returns:
            motor_output: Motor commands
            telemetry: Brain state info
        """
        start_time = time.perf_counter()
        self.cycle += 1
        
        # Convert input
        sensors = torch.tensor(sensory_input[:self.sensory_dim], 
                              dtype=torch.float32, device=self.device)
        
        # ===== 1. SENSORY INJECTION =====
        # Simple: each sensor adds energy at a random but fixed location
        if not hasattr(self, 'sensor_spots'):
            # Initialize injection spots on first use
            self.sensor_spots = torch.randint(0, self.spatial_size, 
                                             (self.sensory_dim, 3), 
                                             device=self.device)
        
        for i, value in enumerate(sensors):
            if i >= self.sensory_dim:
                break
            x, y, z = self.sensor_spots[i]
            # Inject into first few channels
            self.field[x, y, z, i % 8] += value * 0.3
        
        # ===== 2. LEARNING FROM PREDICTION ERROR =====
        if self.last_prediction is not None:
            # Compute error between prediction and reality
            error = self.prediction.compute_error(self.last_prediction, sensors)
            
            # Error creates field tension (discomfort)
            tension = self.learning.error_to_field_tension(error, self.field)
            self.field = self.field + tension
            
            # Update prediction system
            self.prediction.learn_from_error(error, self.field)
            
            error_magnitude = torch.abs(error).mean().item()
        else:
            error_magnitude = 0.0
        
        # ===== 3. INTRINSIC TENSIONS (MOTIVATION) =====
        self.field = self.tensions.apply_tensions(self.field, error_magnitude)
        
        # ===== 4. FIELD EVOLUTION (PHYSICS) =====
        # Add exploration noise if we're not learning well
        exploration = self.learning.should_explore()
        if exploration:
            noise = torch.randn_like(self.field) * 0.02
        else:
            noise = None
        
        self.field = self.dynamics.evolve(self.field, noise)
        
        # ===== 5. MOTOR EXTRACTION =====
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
            'exploring': exploration
        }
        
        # Periodic logging
        if self.cycle % 100 == 0 and not self.quiet_mode:
            print(f"Cycle {self.cycle}: {telemetry['motivation']}, "
                  f"{telemetry['motor']}, {telemetry['learning']}")
        
        # Auto-save every 1000 cycles
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
        """Save brain state."""
        return self.persistence.save(self, name)
    
    def load(self, name: str) -> bool:
        """Load brain state."""
        return self.persistence.load(self, name)
    
    def reset(self):
        """Reset brain to initial state."""
        self.field = torch.randn(self.spatial_size, self.spatial_size, 
                                self.spatial_size, self.channels, device=self.device) * 0.01
        self.cycle = 0
        self.last_prediction = None
        self.tensions.reset()
        
        if not self.quiet_mode:
            print("🔄 Brain reset")


# For compatibility
MinimalUnifiedBrain = TrulyMinimalBrain
UnifiedFieldBrain = TrulyMinimalBrain