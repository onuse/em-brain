"""
Unified Field Brain

Single brain implementation that automatically selects the appropriate
algorithm based on tensor size for optimal performance.
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

# Import the three critical additions for true learning
from .selective_persistence import SelectivePersistence
from .spatial_sensory_encoding import SpatialSensoryEncoding
from .prediction_gated_learning import PredictionGatedLearning



class UnifiedFieldBrain:
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
    
    def __new__(cls, *args, **kwargs):
        """Use optimized version for large brains."""
        spatial_size = kwargs.get('spatial_size', 16)
        if len(args) >= 3:
            spatial_size = args[2]
            
        if spatial_size > 64:
            from .large_field_implementation import LargeFieldImplementation
            return LargeFieldImplementation(*args, **kwargs)
        return object.__new__(cls)
    
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
            print(f"🧠 Unified Field Brain")
            print(f"   Size: {spatial_size}³×{channels} = {params:,} parameters")
            print(f"   Device: {self.device}")
        
        # THE FIELD - a 4D tensor where everything happens
        self.field = torch.randn(spatial_size, spatial_size, spatial_size, channels, 
                                device=self.device) * 0.01
        
        # FIELD MOMENTUM - creates recurrence and self-organization
        # This tracks where the field has been and influences where it's going
        self.field_momentum = torch.zeros_like(self.field)
        
        # Initialize simple systems
        self.dynamics = SimpleFieldDynamics()
        self.tensions = IntrinsicTensions(self.field.shape, self.device)
        self.prediction = SimplePrediction(sensory_dim, self.device, channels)
        self.learning = SimpleLearning(self.device)
        self.motor = SimpleMotorExtraction(motor_dim, self.device, spatial_size)
        self.persistence = SimplePersistence()
        
        # Initialize the three critical additions for true learning
        self.selective_persistence = SelectivePersistence(self.field.shape, self.device)
        self.spatial_encoder = SpatialSensoryEncoding(self.field.shape, self.device)
        self.gated_learning = PredictionGatedLearning(self.field.shape, self.device)
        
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
        
        # ===== 1. SENSORY INJECTION (Now with spatial structure!) =====
        # Use spatial encoding instead of random injection
        # This preserves spatial relationships in sensory data
        
        # For now, inject sensors as before but with spatial encoding
        # In future, visual input can be added here too
        sensor_list = sensors.cpu().tolist() if isinstance(sensors, torch.Tensor) else sensors
        self.field = self.spatial_encoder.encode_sensor_array(sensor_list, self.field)
        
        # ===== 2. LEARNING FROM PREDICTION ERROR (Now gated!) =====
        if self.last_prediction is not None:
            # Compute error between prediction and reality
            error = self.prediction.compute_error(self.last_prediction, sensors)
            error_magnitude = torch.abs(error).mean().item()
            
            # Use prediction-gated learning
            # Only learn when surprised, not from every tiny error
            if self.gated_learning.should_learn(error_magnitude):
                # Error creates field tension (discomfort)
                tension = self.learning.error_to_field_tension(error, self.field)
                
                # Gate the learning update based on confidence
                gated_tension = self.gated_learning.gate_learning(tension, error_magnitude, self.field)
                self.field = self.field + gated_tension
                
                # Update prediction system
                self.prediction.learn_from_error(error, self.field)
            
            # Update selective persistence based on prediction success
            self.selective_persistence.update_stability(self.field, error_magnitude)
        else:
            error_magnitude = 0.0
        
        # ===== 3. INTRINSIC TENSIONS (MOTIVATION) =====
        self.field = self.tensions.apply_tensions(self.field, error_magnitude)
        
        # ===== 3.5. FIELD MOMENTUM (RECURRENCE & MEMORY) =====
        # This is the key to everything - the field influences its own evolution
        # Creating cycles, attractors, and self-organization
        
        # Update momentum with current field activity
        # 0.9 = how much history to keep, 0.1 = how much new to add
        self.field_momentum = 0.9 * self.field_momentum + 0.1 * self.field
        
        # Apply momentum back to field
        # This creates recurrence - patterns that can sustain themselves
        self.field = self.field + self.field_momentum * 0.05
        
        # Momentum also creates natural oscillations and prevents getting stuck
        # Strong activity builds momentum that overshoots and reverses
        # This should naturally break out of static patterns!
        
        # ===== 4. FIELD EVOLUTION (PHYSICS with selective persistence!) =====
        # Apply selective decay - successful patterns persist
        self.field = self.selective_persistence.apply_selective_decay(self.field)
        
        # Add exploration noise if we're not learning well
        exploration = self.learning.should_explore()
        if exploration:
            noise = torch.randn_like(self.field) * 0.02
        else:
            noise = None
        
        # Continue with normal dynamics (but decay is now handled above)
        # Temporarily override decay in dynamics to avoid double-decay
        original_decay = self.dynamics.decay_rate
        self.dynamics.decay_rate = 1.0  # No decay (already handled)
        self.field = self.dynamics.evolve(self.field, noise)
        self.dynamics.decay_rate = original_decay  # Restore
        
        # ===== 5. MOTOR EXTRACTION =====
        motor_output = self.motor.extract_motors(self.field)
        
        # ===== 6. PREDICTION =====
        self.last_prediction = self.prediction.predict_next_sensors(self.field)
        self.prediction.update_history(sensors)
        
        # ===== TELEMETRY =====
        comfort = self.tensions.get_comfort_metrics(self.field)
        learning_stats = self.gated_learning.get_learning_stats()
        
        telemetry = {
            'cycle': self.cycle,
            'time_ms': (time.perf_counter() - start_time) * 1000,
            'energy': self.dynamics.get_energy(self.field),
            'variance': self.dynamics.get_variance(self.field),
            'comfort': comfort['overall_comfort'],
            'memory_utilization': self.selective_persistence.get_memory_utilization(),
            'learning_active': learning_stats['should_learn'],
            'prediction_confidence': learning_stats['confidence'],
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


# Compatibility aliases for existing code
MinimalUnifiedBrain = UnifiedFieldBrain
TrulyMinimalBrain = UnifiedFieldBrain