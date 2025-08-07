"""
Minimal Field Brain - The Brutal Simplification

Philosophy: What's the MINIMAL viable brain that still shows intelligence?
Answer: A single 4D field with just 3 core operations:
1. Imprint (sensory → field)
2. Evolve (field dynamics)
3. Extract (field → motor)

Everything else emerges. No subsystems. No adapters. Just field dynamics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class MinimalFieldBrain:
    """
    The absolute minimum viable brain that still exhibits intelligence.
    
    Core insight: Intelligence doesn't need 15 subsystems. It needs:
    - A field that changes based on input
    - Dynamics that create patterns
    - A way to extract action from patterns
    
    That's it. Everything else is over-engineering.
    """
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_resolution: int = 32,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """
        Initialize minimal brain.
        
        Args:
            sensory_dim: Number of sensors
            motor_dim: Number of motors
            spatial_resolution: Field resolution (32 is plenty)
            device: Computation device
            quiet_mode: Suppress output
        """
        self.quiet_mode = quiet_mode
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
        # Fixed 4D tensor shape - no flexibility needed
        self.field_shape = (spatial_resolution, spatial_resolution, spatial_resolution, 64)
        
        # Device selection - prefer GPU but don't overthink it
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        if not quiet_mode:
            memory_mb = np.prod(self.field_shape) * 4 / (1024 * 1024)
            print(f"🧠 Minimal Field Brain")
            print(f"   Shape: {self.field_shape}")
            print(f"   Device: {self.device}")
            print(f"   Memory: {memory_mb:.1f}MB")
        
        # THE FIELD - This is the entire brain
        # Start with small random values for natural dynamics
        self.field = torch.randn(self.field_shape, device=self.device) * 0.1
        
        # AGGRESSIVE PARAMETERS - 10x more than conservative defaults
        self.learning_rate = 0.2  # Was 0.01, now actually learns
        self.decay_rate = 0.98    # Was 0.999, now actually forgets
        self.diffusion_rate = 0.1 # Was 0.01, now patterns actually spread
        self.noise_scale = 0.05   # Was 0.001, now has real spontaneous activity
        
        # Minimal state tracking
        self.cycle = 0
        self.confidence = 0.5
        self.last_sensory = None
        self.last_motor = None
        
        # Simple prediction tracking for learning
        self.predicted_sensory = None
        self.prediction_error = 0.5
        
        # Pattern memory - just keep recent field states
        self.pattern_memory = deque(maxlen=10)
        
        # Precompute motor extraction weights (random projection)
        # This is our "motor cortex" - just a linear projection
        self.motor_weights = torch.randn(64, motor_dim, device=self.device) * 0.5
        
        # Precompute sensory imprint weights
        # This is our "sensory cortex" - just spreads input across field  
        self.sensory_weights = torch.randn(sensory_dim, 64, device=self.device) * 0.5
        
        if not quiet_mode:
            print(f"✅ Minimal brain ready - 3 operations, {3} parameters")
    
    @torch.no_grad()
    def process_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process one cycle - the absolute minimum:
        1. Imprint sensory input
        2. Evolve field
        3. Extract motor output
        
        That's literally it. No 15 subsystems.
        """
        cycle_start = time.perf_counter()
        
        # Convert input to tensor
        sensory = torch.tensor(sensory_input[:self.sensory_dim], 
                               dtype=torch.float32, device=self.device)
        
        # Extract reward if present
        reward = sensory_input[-1] if len(sensory_input) > self.sensory_dim else 0.0
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # =============================================================
        # STEP 1: IMPRINT - Sensory input modifies field
        # =============================================================
        
        # Simple but effective: sensory input creates field perturbations
        # No complex mapping needed - let the field figure it out
        sensory_influence = torch.einsum('s,sc->c', sensory, self.sensory_weights)
        
        # Add sensory influence to center of field (or spread it)
        # This creates a "sensory echo" in the field
        center = self.field_shape[0] // 2
        influence_region = self.field[center-2:center+3, center-2:center+3, center-2:center+3, :]
        influence_region += sensory_influence.unsqueeze(0).unsqueeze(0).unsqueeze(0) * self.learning_rate
        
        # Reward modulates entire field (positive reward amplifies, negative suppresses)
        if torch.abs(reward) > 0.01:
            # Reward creates global field modulation
            self.field *= (1.0 + reward * 0.1)  # 10% modulation is plenty
            
            # Store this pattern if rewarding
            if reward > 0.1:
                self.pattern_memory.append(self.field.clone())
        
        # =============================================================
        # STEP 2: EVOLVE - Field dynamics create intelligence
        # =============================================================
        
        # A. Decay - patterns fade over time (forgetting)
        self.field *= self.decay_rate
        
        # B. Diffusion - patterns spread spatially (integration)
        # Use 3D convolution for spatial spreading
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27
        for c in range(0, self.field_shape[-1], 8):  # Process in chunks for memory
            chunk = self.field[:, :, :, c:c+8].permute(3, 0, 1, 2).unsqueeze(0)
            diffused = F.conv3d(chunk, kernel, padding=1)
            self.field[:, :, :, c:c+8] = diffused.squeeze(0).permute(1, 2, 3, 0) * (1 - self.diffusion_rate) + \
                                         self.field[:, :, :, c:c+8] * self.diffusion_rate
        
        # C. Nonlinearity - creates complex dynamics
        # Simple tanh keeps values bounded while allowing rich dynamics
        self.field = torch.tanh(self.field * 1.5)
        
        # D. Spontaneous activity - prevents dead zones
        noise = torch.randn_like(self.field) * self.noise_scale
        self.field += noise
        
        # E. Pattern echo - if we have good patterns, blend them in
        if len(self.pattern_memory) > 0 and np.random.rand() < 0.1:  # 10% chance
            # Occasionally recall a successful pattern
            past_pattern = self.pattern_memory[np.random.randint(len(self.pattern_memory))]
            self.field = self.field * 0.9 + past_pattern * 0.1
        
        # =============================================================
        # STEP 3: EXTRACT - Get motor output from field
        # =============================================================
        
        # Simple approach: pool field activity and project to motors
        # Take the mean across spatial dimensions, keep features
        field_features = self.field.mean(dim=(0, 1, 2))  # Shape: [64]
        
        # Project to motor space
        motor_raw = torch.einsum('c,cm->m', field_features, self.motor_weights)
        
        # Add some field gradient information for richer motor output
        # Gradient in each spatial dimension indicates "desire to move"
        grad_x = self.field[1:, :, :, :].mean() - self.field[:-1, :, :, :].mean()
        grad_y = self.field[:, 1:, :, :].mean() - self.field[:, :-1, :, :].mean()
        grad_z = self.field[:, :, 1:, :].mean() - self.field[:, :, :-1, :].mean()
        
        # Combine raw motor output with gradient information
        if self.motor_dim >= 3:
            motor_raw[0] += grad_x * 0.5
            motor_raw[1] += grad_y * 0.5
            motor_raw[2] += grad_z * 0.5
        
        # Squash to reasonable range
        motor_output = torch.tanh(motor_raw)
        
        # =============================================================
        # STEP 4: LEARN - Update predictions and confidence
        # =============================================================
        
        # Super simple prediction: current field state predicts next sensory
        if self.predicted_sensory is not None:
            # Calculate prediction error
            self.prediction_error = torch.mean(torch.abs(sensory - self.predicted_sensory)).item()
            
            # Update confidence (high error = low confidence)
            self.confidence = self.confidence * 0.9 + (1.0 - min(1.0, self.prediction_error)) * 0.1
            
            # Adjust learning rate based on prediction error
            # High error = learn faster (up to 2x)
            self.learning_rate = 0.2 * (1.0 + min(1.0, self.prediction_error))
        
        # Make prediction for next cycle (simple linear projection from field)
        field_summary = field_features[:self.sensory_dim]  # Take first N features
        self.predicted_sensory = torch.tanh(field_summary)
        
        # Update state
        self.cycle += 1
        self.last_sensory = sensory
        self.last_motor = motor_output
        
        # Prepare output
        motor_list = motor_output.cpu().tolist()
        
        # Add confidence as last motor dimension if room
        if len(motor_list) < self.motor_dim:
            motor_list.append(self.confidence)
        
        # Create minimal brain state
        brain_state = {
            'cycle': self.cycle,
            'confidence': self.confidence,
            'prediction_error': self.prediction_error,
            'field_energy': float(torch.abs(self.field).mean()),
            'field_info': float(torch.std(self.field)),
            'learning_rate': self.learning_rate,
            'time_ms': (time.perf_counter() - cycle_start) * 1000
        }
        
        return motor_list, brain_state
    
    def save_state(self) -> Dict[str, Any]:
        """Save brain state for persistence."""
        return {
            'field': self.field.cpu().numpy(),
            'motor_weights': self.motor_weights.cpu().numpy(),
            'sensory_weights': self.sensory_weights.cpu().numpy(),
            'cycle': self.cycle,
            'confidence': self.confidence,
            'learning_rate': self.learning_rate,
            'pattern_memory': [p.cpu().numpy() for p in self.pattern_memory]
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load brain state from persistence."""
        self.field = torch.tensor(state['field'], device=self.device)
        self.motor_weights = torch.tensor(state['motor_weights'], device=self.device)
        self.sensory_weights = torch.tensor(state['sensory_weights'], device=self.device)
        self.cycle = state['cycle']
        self.confidence = state['confidence']
        self.learning_rate = state['learning_rate']
        self.pattern_memory = deque(
            [torch.tensor(p, device=self.device) for p in state['pattern_memory']], 
            maxlen=10
        )