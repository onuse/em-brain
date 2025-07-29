"""
Simplified Unified Field Brain

Radical simplification with 4D tensor architecture for GPU optimization.
All cognitive properties emerge from unified field dynamics.
"""

import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Core imports
from .field_types import UnifiedFieldExperience
from .spontaneous_dynamics import SpontaneousDynamics
from ...parameters.cognitive_config import get_cognitive_config
from .unified_pattern_system import UnifiedPatternSystem
from .pattern_motor_adapter import PatternMotorAdapter
from .pattern_attention_adapter import PatternAttentionAdapter
from .motor_cortex import MotorCortex
from .field_constants import TOPOLOGY_REGIONS_MAX
from .unified_field_dynamics import UnifiedFieldDynamics
from .predictive_action_system import PredictiveActionSystem
from .reward_topology_shaping import RewardTopologyShaper


class SimplifiedUnifiedBrain:
    """
    Simplified brain with 4D tensor architecture.
    
    Major simplifications:
    - Fixed 4D tensor shape for GPU optimization
    - No complex dimension mappings
    - All properties emerge from field dynamics
    - Direct tensor operations without abstraction layers
    """
    
    def __init__(self,
                 sensory_dim: int = 16,
                 motor_dim: int = 5,
                 spatial_resolution: int = 32,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """
        Initialize simplified brain.
        
        Args:
            sensory_dim: Number of sensors (for compatibility)
            motor_dim: Number of motors
            spatial_resolution: Spatial resolution (32 recommended)
            device: Computation device
            quiet_mode: Suppress output
        """
        # Configuration
        self.cognitive_config = get_cognitive_config()
        brain_config = self.cognitive_config.brain_config
        self.quiet_mode = quiet_mode
        
        # Fixed 4D tensor shape for GPU optimization
        self.tensor_shape = [spatial_resolution, spatial_resolution, spatial_resolution, 64]
        self.spatial_resolution = spatial_resolution
        
        # Device selection - prefer GPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        if not quiet_mode:
            print(f"🧠 Simplified Unified Brain")
            print(f"   Tensor shape: {self.tensor_shape} (4D)")
            print(f"   Device: {self.device}")
            print(f"   Memory: {self._calculate_memory_usage():.1f}MB")
        
        # Initialize unified field with moderate random values
        self.unified_field = torch.randn(self.tensor_shape, dtype=torch.float32, device=self.device) * 0.3 + 0.1
        
        # Core parameters
        self.field_evolution_rate = brain_config.field_evolution_rate
        self.field_decay_rate = brain_config.field_decay_rate
        self.field_diffusion_rate = brain_config.field_diffusion_rate
        self.spontaneous_rate = brain_config.spontaneous_rate
        
        # Initialize core systems
        self._initialize_core_systems(motor_dim)
        
        # State tracking
        self.brain_cycles = 0
        self.field_evolution_cycles = 0
        self._last_cycle_time = 0
        self._current_prediction_confidence = brain_config.default_prediction_confidence
        self._predicted_field = None
        self._last_prediction_error = brain_config.optimal_prediction_error
        self._last_spontaneous_magnitude = 0.0
        self._last_spontaneous_variance = 0.0
        self._last_imprint_strength = 0.0
        self.modulation = {}  # Will be filled by unified field dynamics
        
        # Memory systems
        self.topology_regions = []
        self.working_memory = deque(maxlen=brain_config.working_memory_limit)
        self.temporal_experiences = deque(maxlen=100)
        self.field_experiences = deque(maxlen=1000)
        
        if not quiet_mode:
            print(f"✅ Brain initialized successfully on {self.device}")
            
    def _initialize_core_systems(self, motor_dim: int):
        """Initialize all core brain systems."""
        # Unified field dynamics (replaces energy + blended reality)
        self.field_dynamics = UnifiedFieldDynamics(
            pattern_memory_size=100,
            confidence_window=50,
            device=self.device
        )
        
        # Predictive action system
        self.predictive_actions = PredictiveActionSystem(
            field_shape=self.unified_field.shape,
            motor_dim=motor_dim,
            device=self.device
        )
        
        # Reward topology shaping
        self.topology_shaper = RewardTopologyShaper(
            field_shape=self.unified_field.shape,
            device=self.device,
            persistence_factor=0.95,
            max_attractors=20
        )
        
        # Blended reality now part of unified field dynamics
        
        # Spontaneous dynamics
        self.spontaneous = SpontaneousDynamics(
            field_shape=self.unified_field.shape,
            resting_potential=self.cognitive_config.brain_config.resting_potential,
            spontaneous_rate=self.spontaneous_rate,
            coherence_scale=3.0,
            device=self.device
        )
        
        # UNIFIED PATTERN SYSTEM - Shared by both motor and attention
        self.pattern_system = UnifiedPatternSystem(
            field_shape=self.tensor_shape,
            device=self.device,
            max_patterns=50,
            history_size=100
        )
        
        # Pattern-based motor using unified system
        self.motor_cortex = MotorCortex(
            motor_dim=motor_dim,
            device=self.device,
            activation_threshold=0.1
        )
        
        self.pattern_motor = PatternMotorAdapter(
            pattern_system=self.pattern_system,
            motor_dim=motor_dim - 1,  # Reserve last for confidence
            motor_cortex=self.motor_cortex,
            device=self.device
        )
        
        # Pattern-based attention using unified system
        self.pattern_attention = PatternAttentionAdapter(
            pattern_system=self.pattern_system,
            attention_capacity=5,
            device=self.device
        )
        
    def process_robot_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main processing cycle - simplified version.
        """
        cycle_start = time.perf_counter()
        
        # 1. Create field experience from sensors
        experience = self._create_field_experience(sensory_input)
        
        # 2. Update prediction tracking
        if self._predicted_field is not None:
            prediction_error = torch.mean(torch.abs(self.unified_field - self._predicted_field)).item()
            self._last_prediction_error = prediction_error
            self._current_prediction_confidence = 1.0 - min(1.0, prediction_error * 2.0)
        
        # 3. Imprint sensory experience
        self._imprint_experience(experience)
        
        # 4. Process attention
        attention_data = self._process_attention(sensory_input)
        
        # 5. Update unified field dynamics
        reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
        
        # Compute field state (energy, novelty, etc.)
        field_state = self.field_dynamics.compute_field_state(self.unified_field)
        novelty = self.field_dynamics.compute_novelty(self.unified_field)
        
        # Update confidence from prediction error
        self.field_dynamics.update_confidence(self._last_prediction_error)
        
        # Get unified modulation parameters
        has_input = len(sensory_input) > 0 and any(abs(v) > 0.01 for v in sensory_input[:-1])
        self.modulation = self.field_dynamics.compute_field_modulation(
            field_state, has_sensory_input=has_input
        )
        
        # 6. Process reward topology
        if abs(reward) > 0.1:
            self.topology_shaper.process_reward(
                current_field=self.unified_field,
                reward=reward,
                threshold=0.1
            )
        
        # 7. Evolve field
        self._evolve_field()
        
        # 8. Generate motor action
        motor_output = self._generate_motor_action()
        
        # 9. Update state
        self.brain_cycles += 1
        self._last_cycle_time = time.perf_counter() - cycle_start
        
        # Return motor output and state
        brain_state = self._create_brain_state()
        return motor_output, brain_state
    
    def _create_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """Create field experience from sensory input."""
        # Simple mapping - just convert to tensor
        raw_input = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        # Extract reward if present
        reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
        field_intensity = 0.5 + reward * 0.5  # Map [-1,1] to [0,1]
        
        # For 4D tensor, we don't need complex coordinate mapping
        # Just create a simple position in the center
        field_coords = torch.zeros(4, device=self.device)
        field_coords[:3] = 0.0  # Center position
        
        return UnifiedFieldExperience(
            timestamp=time.time(),
            raw_input_stream=raw_input,
            field_coordinates=field_coords,
            field_intensity=field_intensity,
            dynamics_family_activations={}
        )
    
    def _imprint_experience(self, experience: UnifiedFieldExperience):
        """Imprint experience into field - simplified version."""
        # Confidence is now updated in unified field dynamics
        
        # Check for meaningful input
        has_input = (experience.raw_input_stream is not None and
                    torch.max(torch.abs(experience.raw_input_stream)) > 0.01)
        
        if has_input:
            # Get modulated intensity from unified dynamics
            scaled_intensity = experience.field_intensity * self.modulation.get('imprint_strength', 0.5)
            scaled_intensity *= self.modulation.get('sensory_amplification', 1.0)
            
            # Simple imprint - add to center region
            cx, cy, cz = self.spatial_resolution // 2, self.spatial_resolution // 2, self.spatial_resolution // 2
            region_size = 3
            
            self.unified_field[
                cx-region_size:cx+region_size+1,
                cy-region_size:cy+region_size+1,
                cz-region_size:cz+region_size+1,
                :
            ] += scaled_intensity
            
            self._last_imprint_strength = scaled_intensity
        else:
            # No sensory input - just track for statistics
            pass
    
    def _evolve_field(self):
        """Evolve field - simplified version."""
        # 1. Energy-based modulation
        if hasattr(self, 'energy_recommendations'):
            self.unified_field = self.energy_system.modulate_field(
                self.unified_field,
                self.energy_recommendations
            )
        
        # 2. Simple diffusion
        if self.field_diffusion_rate > 0:
            for dim in range(3):  # Only spatial dimensions
                shifted_fwd = torch.roll(self.unified_field, shifts=1, dims=dim)
                shifted_back = torch.roll(self.unified_field, shifts=-1, dims=dim)
                diffusion = (shifted_fwd + shifted_back - 2 * self.unified_field) / 2
                self.unified_field += self.field_diffusion_rate * diffusion
        
        # 3. Spontaneous dynamics
        spontaneous_weight = self.modulation.get('spontaneous_weight', 0.5)
        sensory_gating = 1.0 - spontaneous_weight
        
        spontaneous_activity = self.spontaneous.generate_spontaneous_activity(
            self.unified_field,
            sensory_gating=sensory_gating
        )
        self.unified_field += spontaneous_activity
        
        self._last_spontaneous_magnitude = torch.mean(torch.abs(spontaneous_activity)).item()
        self._last_spontaneous_variance = torch.var(spontaneous_activity).item()
        
        # 4. Reward topology influence
        topology_influence = self.topology_shaper.apply_topology_influence(self.unified_field)
        self.unified_field += topology_influence
        
        # 5. Log blended reality state
        if self.brain_cycles % 100 == 0 and not self.quiet_mode:
            state_desc = self.field_dynamics.get_state_description()
            print(state_desc)
        
        self.field_evolution_cycles += 1
    
    def _process_attention(self, sensory_input: List[float]) -> Optional[Dict[str, Any]]:
        """Process attention - simplified with unified pattern system."""
        sensory_patterns = {
            'primary': torch.tensor(sensory_input[:-1], dtype=torch.float32, device=self.device)
        }
        attention_state = self.pattern_attention.process_field_patterns(
            field=self.unified_field,
            sensory_patterns=sensory_patterns
        )
        # Store for motor generation
        self._last_attention_state = attention_state
        return attention_state
    
    def _generate_motor_action(self) -> List[float]:
        """Generate motor action - simplified with unified pattern system."""
        # Generate motor action using unified pattern system
        exploration_params = {
            'exploration_drive': self.modulation.get('exploration_drive', 0.5),
            'motor_noise': 0.2
        }
        
        # Get spontaneous activity for this cycle
        spontaneous_activity = self.spontaneous.generate_spontaneous_activity(
            self.unified_field,
            sensory_gating=0.5  # Temporary gating
        )
        
        # Get attention state from current attention processing
        attention_state = getattr(self, '_last_attention_state', None)
        
        # Generate motor commands through unified pattern adapter
        motor_commands = self.pattern_motor.generate_motor_action(
            field=self.unified_field,
            spontaneous_activity=spontaneous_activity,
            attention_state=attention_state,
            exploration_params=exploration_params
        )
        
        # Predictive action selection (still uses pattern features)
        # Extract patterns for predictive system
        patterns = self.pattern_system.extract_patterns(self.unified_field, n_patterns=5)
        pattern_features = {}
        if patterns:
            # Convert top pattern to features dict
            top_pattern = patterns[0]
            pattern_features = top_pattern.to_dict()
        
        candidates = self.predictive_actions.generate_action_candidates(
            current_field=self.unified_field,
            current_patterns=pattern_features,
            n_candidates=3  # Fewer candidates for speed
        )
        
        # Preview outcomes
        for candidate in candidates:
            self.predictive_actions.preview_action_outcome(
                current_field=self.unified_field,
                action=candidate,
                evolution_steps=3  # Fewer steps for speed
            )
        
        # Select best action
        exploration_drive = self.modulation.get('exploration_drive', 0.5)
        selected_action = self.predictive_actions.select_action(
            candidates=candidates,
            exploration_drive=exploration_drive
        )
        
        # Blend pattern-based and predictive actions
        # Handle potential dimension mismatch
        if len(motor_commands) != len(selected_action.motor_pattern):
            # Just use motor commands if dimensions don't match
            final_commands = motor_commands
        else:
            # Use pattern-based as baseline, modulate with predictive
            final_commands = motor_commands * 0.7 + selected_action.motor_pattern * 0.3
        
        # Ensure commands are in valid range
        final_commands = torch.clamp(final_commands, -1.0, 1.0)
        
        return final_commands.tolist()
    
    def _create_brain_state(self) -> Dict[str, Any]:
        """Create brain state for telemetry."""
        return {
            'cycle': self.brain_cycles,
            'cycle_time_ms': self._last_cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.unified_field))),
            'max_activation': float(torch.max(torch.abs(self.unified_field))),
            'prediction_confidence': self._current_prediction_confidence,
            'memory_saturation': len(self.topology_regions) / TOPOLOGY_REGIONS_MAX,
            'energy_state': {
                'energy': self.modulation.get('energy', 0.5),
                'novelty': self.modulation.get('novelty', 0.0),
                'exploration_drive': self.modulation.get('exploration_drive', 0.5)
            },
            'topology_shaping': self.topology_shaper.get_topology_state(),
            'tensor_shape': self.tensor_shape,
            'device': str(self.device)
        }
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        elements = 1
        for dim in self.tensor_shape:
            elements *= dim
        return (elements * 4) / (1024 * 1024)