"""
Simplified Unified Field Brain

Radical simplification with 4D tensor architecture for GPU optimization.
All cognitive properties emerge from unified field dynamics.
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

# Core imports
from .field_types import UnifiedFieldExperience
from ...parameters.cognitive_config import get_cognitive_config
from .unified_pattern_system import UnifiedPatternSystem
from .pattern_motor_adapter import PatternMotorAdapter
from .pattern_attention_adapter import PatternAttentionAdapter
from .motor_cortex import MotorCortex
from .adaptive_motor_cortex import AdaptiveMotorCortex
from .field_constants import TOPOLOGY_REGIONS_MAX
from .evolved_field_dynamics import EvolvedFieldDynamics
from .emergent_sensory_mapping import EmergentSensoryMapping
from .predictive_action_system import PredictiveActionSystem
from .reward_topology_shaping import RewardTopologyShaper
from .consolidation_system import ConsolidationSystem
from .topology_region_system import TopologyRegionSystem
from .predictive_field_system import PredictiveFieldSystem
from ...utils.tensor_ops import create_randn, field_information, field_stats, apply_diffusion
from ...utils.error_handling import (
    validate_list_input, validate_tensor_shape, ErrorContext,
    BrainError, safe_tensor_op
)
from .pattern_cache_pool import PatternCachePool


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
        
        # Initialize unified field with moderate random values and baseline activity
        # Bias provides metabolic baseline to prevent complete silence
        self.unified_field = create_randn(self.tensor_shape, device=self.device, scale=0.2, bias=0.05)
        
        # Core parameters
        self.field_evolution_rate = brain_config.field_evolution_rate
        self.field_decay_rate = brain_config.field_decay_rate
        self.field_diffusion_rate = brain_config.field_diffusion_rate
        self.spontaneous_rate = brain_config.spontaneous_rate
        self.sensory_dim = sensory_dim
        
        # Initialize core systems
        self._initialize_core_systems(motor_dim)
        
        # Create pattern cache pool for performance
        self.pattern_cache_pool = PatternCachePool(
            field_shape=self.tensor_shape,
            max_patterns=50,
            device=self.device
        )
        
        # State tracking
        self.brain_cycles = 0
        self.field_evolution_cycles = 0
        self._last_cycle_time = 0
        self._current_prediction_confidence = brain_config.default_prediction_confidence
        self._predicted_field = None
        self._predicted_sensory = None  # Track what we expect to sense
        self._last_prediction_error = brain_config.optimal_prediction_error
        self._last_imprint_strength = 0.0
        self._last_activated_regions = []
        self.modulation = {}  # Will be filled by unified field dynamics
        
        # Memory systems
        self.working_memory = deque(maxlen=brain_config.working_memory_limit)
        self.temporal_experiences = deque(maxlen=100)
        self.field_experiences = deque(maxlen=1000)
        self.recent_sensory = deque(maxlen=10)  # For momentum-based prediction
        
        if not quiet_mode:
            print(f"✅ Brain initialized successfully on {self.device}")
            
    def _initialize_core_systems(self, motor_dim: int):
        """Initialize all core brain systems."""
        # Evolved field dynamics - THE core system
        self.field_dynamics = EvolvedFieldDynamics(
            field_shape=self.unified_field.shape,
            pattern_memory_size=100,
            confidence_window=50,
            initial_spontaneous_rate=self.spontaneous_rate,
            initial_resting_potential=self.cognitive_config.brain_config.resting_potential,
            temporal_features=16,  # For working memory
            dynamics_features=16,  # For self-modifying dynamics
            device=self.device
        )
        
        # Initialize the dynamics features in the field
        self.field_dynamics.initialize_field_dynamics(self.unified_field)
        
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
        
        # Blended reality and spontaneous dynamics now part of unified field dynamics
        
        # UNIFIED PATTERN SYSTEM - Shared by both motor and attention
        self.pattern_system = UnifiedPatternSystem(
            field_shape=self.tensor_shape,
            device=self.device,
            max_patterns=50,
            history_size=100
        )
        
        # Pattern-based motor using unified system
        # Use adaptive motor cortex for more nuanced outputs
        self.motor_cortex = AdaptiveMotorCortex(
            motor_dim=motor_dim,
            device=self.device,
            base_sensitivity=0.1,
            adaptation_rate=0.01,
            quiet_mode=self.quiet_mode
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
        
        # Consolidation system for advanced learning
        self.consolidation_system = ConsolidationSystem(
            field_shape=self.tensor_shape,
            device=self.device
        )
        
        # Topology region system for abstraction and causal understanding
        self.topology_region_system = TopologyRegionSystem(
            field_shape=self.tensor_shape,
            device=self.device,
            stability_threshold=0.05,  # Lower threshold for better detection
            max_regions=200
        )
        
        # Emergent sensory mapping - patterns find their place
        self.sensory_mapping = EmergentSensoryMapping(
            field_shape=self.tensor_shape,
            device=self.device,
            resonance_threshold=0.3,
            spatial_decay=0.95
        )
        
        # Predictive field system - the brain IS prediction
        self.predictive_field = PredictiveFieldSystem(
            field_shape=self.tensor_shape,
            sensory_dim=self.sensory_dim,
            device=self.device
        )
        
    @torch.no_grad()  # Disable gradient computation for performance
    def process_robot_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main processing cycle - simplified version.
        """
        cycle_start = time.perf_counter()
        
        try:
            # Validate input
            with ErrorContext("validating sensory input"):
                # Allow variable length for different robot configurations
                validate_list_input(sensory_input, len(sensory_input), "sensory_input", -10.0, 10.0)
            
            # 1. Create field experience from sensors
            with ErrorContext("creating field experience"):
                experience = self._create_field_experience(sensory_input)
            
            # 2. Update prediction tracking
            if self._predicted_sensory is not None and experience.raw_input_stream is not None:
                # Compare predicted vs actual sensory input
                actual_sensory = experience.raw_input_stream[:len(self._predicted_sensory)]
                sensory_error = torch.mean(torch.abs(actual_sensory - self._predicted_sensory)).item()
                self._last_prediction_error = sensory_error
                self._current_prediction_confidence = 1.0 - min(1.0, sensory_error * 2.0)
            elif self._predicted_field is not None:
                # Fallback to field comparison
                prediction_error = torch.mean(torch.abs(self.unified_field - self._predicted_field)).item()
                self._last_prediction_error = prediction_error
                self._current_prediction_confidence = 1.0 - min(1.0, prediction_error * 2.0)
            else:
                # First cycle - no prediction yet, so low confidence
                self._current_prediction_confidence = 0.5
            
            # 3. Imprint sensory experience
            self._imprint_experience(experience)
            
            # 4. Process attention
            attention_data = self._process_attention(sensory_input)
            
            # 5. Update unified field dynamics
            reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
            
            # Compute field state (information, novelty, etc.)
            field_state = self.field_dynamics.compute_field_state(self.unified_field)
            novelty = self.field_dynamics.compute_novelty(self.unified_field)
            
            # Update confidence from prediction error
            self.field_dynamics.update_confidence(self._last_prediction_error)
            
            # Debug confidence values
            if self.brain_cycles % 100 == 0 and not self.quiet_mode:
                print(f"[DEBUG] Confidence: current={self._current_prediction_confidence:.3f}, " +
                      f"smoothed={self.field_dynamics.smoothed_confidence:.3f}, " +
                      f"error={self._last_prediction_error:.3f}")
            
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
                
                # Also give feedback to adaptive motor cortex
                if hasattr(self.motor_cortex, 'inject_reward_feedback'):
                    self.motor_cortex.inject_reward_feedback(reward)
            
            # 7. Evolve field
            self._evolve_field()
            
            # Generate prediction for next cycle
            # Simple approach: current field + expected evolution
            self._predicted_field = self.unified_field.clone()
            # Apply expected decay and diffusion
            self._predicted_field *= self.modulation.get('decay_rate', 0.995)
            
            # Predict sensory input using the predictive field system
            # This is where the brain reveals what it expects to sense
            topology_regions = getattr(self, '_last_activated_regions', [])
            prediction = self.predictive_field.generate_sensory_prediction(
                field=self.unified_field,
                topology_regions=topology_regions,
                recent_sensory=self.recent_sensory
            )
            
            # Store predictions for next cycle
            self._predicted_sensory = prediction.values
            self._prediction_confidence_per_sensor = prediction.confidence
            
            # Update recent sensory history for momentum prediction
            self.recent_sensory.append(sensory_input)
            
            # 8. Detect and update topology regions
            # Only run every 5 cycles for performance
            if self.brain_cycles % 5 == 0:
                activated_regions = self.topology_region_system.detect_topology_regions(
                    self.unified_field,
                    current_patterns=self.pattern_system.extract_patterns(self.unified_field, n_patterns=5)
                )
            else:
                activated_regions = self._last_activated_regions
            
            # 9. Generate motor action
            motor_output = self._generate_motor_action()
            
            # 10. Update state
            self.brain_cycles += 1
            self._last_cycle_time = time.perf_counter() - cycle_start
            self._last_activated_regions = activated_regions
            
            # Return motor output and state
            brain_state = self._create_brain_state()
            return motor_output, brain_state
            
        except BrainError:
            # Re-raise brain errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            self.brain_cycles += 1  # Still increment to avoid getting stuck
            logger.error(f"Unexpected error in brain cycle {self.brain_cycles}: {e}")
            # Return safe defaults
            safe_motors = [0.0] * (self.motor_cortex.motor_dim - 1)
            safe_state = {'cycle': self.brain_cycles, 'error': str(e)}
            return safe_motors, safe_state
    
    def _create_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """Create field experience from sensory input."""
        # Simple mapping - just convert to tensor
        raw_input = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        # Extract reward if present
        reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
        field_intensity = 0.5 + reward * 0.5  # Map [-1,1] to [0,1]
        
        return UnifiedFieldExperience(
            timestamp=time.time(),
            raw_input_stream=raw_input,
            field_intensity=field_intensity,
            dynamics_family_activations={}
        )
    
    def _imprint_experience(self, experience: UnifiedFieldExperience):
        """Imprint experience into field through emergent mapping."""
        # Check for meaningful input
        has_input = (experience.raw_input_stream is not None and
                    torch.max(torch.abs(experience.raw_input_stream)) > 0.01)
        
        if has_input:
            # Get modulated intensity from unified dynamics
            scaled_intensity = experience.field_intensity * self.modulation.get('imprint_strength', 0.5)
            scaled_intensity *= self.modulation.get('sensory_amplification', 1.0)
            
            # Predictive sensory gating: suppress well-predicted inputs
            # High prediction error → high surprise → strong imprinting
            # Low prediction error → low surprise → weak imprinting
            prediction_confidence = self._current_prediction_confidence
            surprise_factor = 1.0 - prediction_confidence  # 0 = perfectly predicted, 1 = totally surprising
            
            # Add baseline to ensure some learning even with good predictions
            # But heavily modulate based on surprise
            min_imprint = 0.1  # Always learn a little
            scaled_intensity *= (min_imprint + (1.0 - min_imprint) * surprise_factor)
            
            # Find emergent location for this sensory pattern
            reward = experience.raw_input_stream[-1].item() if len(experience.raw_input_stream) > 24 else 0.0
            x, y, z = self.sensory_mapping.find_imprint_location(
                sensory_pattern=experience.raw_input_stream,
                field_state=self.unified_field,
                reward=reward
            )
            
            # Imprint at discovered location with spatial spread
            region_size = 2  # Slightly smaller since location is more precise
            
            # Ensure bounds
            x_start = max(0, x - region_size)
            x_end = min(self.spatial_resolution, x + region_size + 1)
            y_start = max(0, y - region_size)
            y_end = min(self.spatial_resolution, y + region_size + 1)
            z_start = max(0, z - region_size)
            z_end = min(self.spatial_resolution, z + region_size + 1)
            
            # Apply with distance-based falloff
            for dx in range(x_start - x, x_end - x):
                for dy in range(y_start - y, y_end - y):
                    for dz in range(z_start - z, z_end - z):
                        distance = abs(dx) + abs(dy) + abs(dz)
                        weight = 0.8 ** distance  # Exponential falloff
                        
                        self.unified_field[
                            x + dx, y + dy, z + dz, :
                        ] += scaled_intensity * weight
            
            self._last_imprint_strength = scaled_intensity
        else:
            # No sensory input - just track for statistics
            pass
    
    @safe_tensor_op
    def _evolve_field(self):
        """Evolve field - unified version with integrated spontaneous dynamics."""
        # 1. Apply unified field dynamics (includes spontaneous)
        self.unified_field = self.field_dynamics.evolve_field(self.unified_field)
        
        # 2. Apply diffusion
        if self.field_diffusion_rate > 0:
            self.unified_field = apply_diffusion(self.unified_field, self.field_diffusion_rate, dims=(0, 1, 2))
        
        # 3. Reward topology influence
        topology_influence = self.topology_shaper.apply_topology_influence(self.unified_field)
        self.unified_field += topology_influence
        
        # 4. Log state
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
    
    @safe_tensor_op
    def _generate_motor_action(self) -> List[float]:
        """Generate motor action - simplified with unified pattern system."""
        # Generate motor action using unified pattern system
        exploration_params = {
            'exploration_drive': self.modulation.get('exploration_drive', 0.5),
            'motor_noise': self.modulation.get('motor_noise', 0.2)
        }
        
        # Get attention state from current attention processing
        attention_state = getattr(self, '_last_attention_state', None)
        
        # Spontaneous activity is now integrated into field evolution
        # Just pass the current field state
        motor_commands = self.pattern_motor.generate_motor_action(
            field=self.unified_field,
            spontaneous_activity=None,  # No longer needed separately
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
        stats = field_stats(self.unified_field)
        topology_stats = self.topology_region_system.get_statistics()
        
        # Get evolution state from field dynamics
        evolution_props = self.field_dynamics.get_emergent_properties()
        working_memory = self.field_dynamics.get_working_memory_state(self.unified_field)
        
        # Determine cognitive mode based on information and confidence
        information = self.modulation.get('information', 0.5)
        confidence = evolution_props['smoothed_confidence']
        exploration = self.modulation.get('exploration_drive', 0.5)
        
        if information < 0.3:
            cognitive_mode = "exploring"
        elif information > 0.7 and self.field_dynamics.cycles_without_input > 50:
            cognitive_mode = "dreaming"
        elif confidence > 0.6:
            cognitive_mode = "exploiting"
        else:
            cognitive_mode = "balanced"
        
        return {
            'cycle': self.brain_cycles,
            'cycle_time_ms': self._last_cycle_time * 1000,
            'field_information': stats['information'],
            'max_activation': stats['max'],
            'prediction_confidence': self._current_prediction_confidence,
            'memory_saturation': topology_stats['total_regions'] / self.topology_region_system.max_regions,
            'cognitive_mode': cognitive_mode,
            'information_state': {
                'information': information,
                'novelty': self.modulation.get('novelty', 0.0),
                'exploration_drive': exploration
            },
            'evolution_state': {
                'self_modification_strength': evolution_props['self_modification_strength'],
                'evolution_cycles': evolution_props['evolution_cycles'],
                'smoothed_information': evolution_props.get('smoothed_information', evolution_props.get('smoothed_energy', 0.5)),
                'smoothed_confidence': evolution_props['smoothed_confidence'],
                'cycles_without_input': evolution_props['cycles_without_input'],
                'working_memory': working_memory
            },
            'topology_shaping': self.topology_shaper.get_topology_state(),
            'topology_regions': {
                'total': topology_stats['total_regions'],
                'active': topology_stats['active_regions'],
                'abstract': topology_stats['abstract_regions'],
                'causal_links': topology_stats['causal_links'],
                'activated_now': len(getattr(self, '_last_activated_regions', []))
            },
            'tensor_shape': self.tensor_shape,
            'device': str(self.device),
            'sensory_organization': self.sensory_mapping.get_statistics(),
            'timestamp': time.time()
        }
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        elements = 1
        for dim in self.tensor_shape:
            elements *= dim
        return (elements * 4) / (1024 * 1024)
    
    def perform_maintenance(self):
        """Perform maintenance including memory consolidation."""
        # Start consolidation
        self.consolidation_system.start_consolidation(self)
        
        # Run consolidation for a short period (5 seconds)
        metrics = self.consolidation_system.consolidate_memories(self, duration_seconds=5.0)
        
        # Also consolidate topology regions
        self.topology_region_system.consolidate_regions(self)
        
        # Reorganize sensory mappings for better topological organization
        self.sensory_mapping.reorganize_mappings(self.unified_field)
        
        if not self.quiet_mode:
            topology_stats = self.topology_region_system.get_statistics()
            print(f"🧠 Consolidation complete: {metrics.patterns_strengthened} patterns strengthened, "
                  f"{metrics.dream_sequences} dreams, benefit={metrics.consolidation_benefit:.3f}")
            print(f"🏔️ Topology: {topology_stats['total_regions']} regions, "
                  f"{topology_stats['abstract_regions']} abstractions, "
                  f"{topology_stats['causal_links']} causal links")
    
    def start_idle_consolidation(self, duration_seconds: float = 60.0):
        """Start extended consolidation during idle period."""
        if not self.quiet_mode:
            print(f"😴 Starting {duration_seconds}s consolidation phase...")
        
        self.consolidation_system.start_consolidation(self)
        metrics = self.consolidation_system.consolidate_memories(self, duration_seconds)
        
        if not self.quiet_mode:
            print(f"✅ Consolidation complete: benefit={metrics.consolidation_benefit:.3f}")
        
        return metrics
    
    def get_evolution_state(self) -> Dict[str, Any]:
        """Get current state of field evolution."""
        props = self.field_dynamics.get_emergent_properties()
        
        return {
            'evolution_cycles': props['evolution_cycles'],
            'self_modification_strength': props['self_modification_strength'],
            'smoothed_information': props.get('smoothed_information', props.get('smoothed_energy', 0.5)),
            'smoothed_confidence': props['smoothed_confidence'],
            'working_memory': self.field_dynamics.get_working_memory_state(self.unified_field)
        }