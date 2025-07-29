"""
Dynamic Unified Field Brain - Full Implementation

This is the complete implementation of UnifiedFieldBrain with dynamic dimensions,
including all advanced features: constraints, temporal dynamics, prediction, etc.
"""

import torch
import numpy as np
import time
import threading
import queue
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque, defaultdict
import math

from .field_types import (
    FieldDimension, FieldDynamicsFamily, UnifiedFieldExperience,
    FieldNativeAction
)

# TemporalExperience import removed - temporal_field_dynamics.py was archived
from .optimized_gradients import create_optimized_gradient_calculator
from ...core.dynamic_dimension_calculator import DynamicDimensionCalculator
from .spontaneous_dynamics import SpontaneousDynamics
from ...utils.cognitive_autopilot import CognitiveAutopilot
from .blended_reality import integrate_blended_reality
from ...parameters.cognitive_config import get_cognitive_config
from ...config.enhanced_gpu_memory_manager import get_device_for_tensor
# IntegratedAttention removed - using PatternBasedAttention only
from .enhanced_dynamics import EnhancedFieldDynamics, PhaseTransitionConfig, AttractorConfig
from .brain_field_adapter import BrainFieldAdapter
# Emergent spatial removed - pattern-based motor provides coordinate-free movement
from .pattern_based_motor import PatternBasedMotorGenerator
from .pattern_based_attention_fast import FastPatternBasedAttention
from .motor_cortex import MotorCortex
from .field_constants import TOPOLOGY_REGIONS_MAX
from .organic_energy_system import OrganicEnergySystem
from .predictive_action_system import PredictiveActionSystem
from .reward_topology_shaping import RewardTopologyShaper


class DynamicUnifiedFieldBrain:
    """
    Complete Unified Field Brain with Dynamic Dimensions
    
    This brain includes all features from the original UnifiedFieldBrain
    but accepts dynamic conceptual dimensions based on robot complexity.
    """
    
    def __init__(self,
                 field_dimensions: List[FieldDimension],
                 tensor_shape: List[int],
                 dimension_mapping: Dict[str, Any],
                 spatial_resolution: int = 4,
                 temporal_window: float = 10.0,
                 field_evolution_rate: Optional[float] = None,
                 constraint_discovery_rate: Optional[float] = None,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False,
                 enable_attention: Optional[bool] = None,
                 pattern_attention: Optional[bool] = None):
        """
        Initialize brain with dynamic dimensions and full feature set.
        """
        # Load cognitive configuration
        self.cognitive_config = get_cognitive_config()
        brain_config = self.cognitive_config.brain_config
        
        self.field_dimensions = field_dimensions
        self._dimension_definitions = field_dimensions  # Store for emergent interface
        self.tensor_shape = tensor_shape
        self.dimension_mapping = dimension_mapping
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        
        # Use cognitive constants with optional overrides
        self.field_evolution_rate = field_evolution_rate or brain_config.field_evolution_rate
        self.constraint_discovery_rate = constraint_discovery_rate or brain_config.constraint_discovery_rate
        self.quiet_mode = quiet_mode
        
        # Total conceptual dimensions
        self.total_dimensions = len(field_dimensions)
        
        # Set device using enhanced GPU memory manager
        if device is None:
            # Use GPU memory manager for MPS-aware device selection
            self.device = get_device_for_tensor(tensor_shape)
            if not quiet_mode and self.device.type == 'cpu' and torch.backends.mps.is_available():
                print(f"   Using CPU due to tensor dimensions ({len(tensor_shape)}D)")
        else:
            self.device = device
            
        # Create unified field with dynamic tensor shape
        # Initialize with moderate random values for pattern motor to work with
        # Higher initial energy enables better pattern formation and motor generation
        self.unified_field = torch.randn(tensor_shape, dtype=torch.float32, device=self.device) * 0.3 + 0.1
        
        # Log initial field energy
        initial_energy = torch.mean(torch.abs(self.unified_field)).item()
        if not quiet_mode:
            print(f"   Initial field energy: {initial_energy:.3f}")
        

        
        # Initialize optimized gradient calculator with cognitive constants
        self.gradient_calculator = create_optimized_gradient_calculator(
            activation_threshold=brain_config.activation_threshold,
            cache_enabled=True,
            sparse_computation=True
        )
        
        # Field evolution parameters from cognitive constants
        self.field_decay_rate = brain_config.field_decay_rate
        self.field_diffusion_rate = brain_config.field_diffusion_rate
        self.topology_stability_threshold = brain_config.topology_stability_threshold
        self.field_energy_dissipation_rate = brain_config.field_energy_dissipation_rate
        
        # Field state tracking
        self.field_experiences: deque = deque(maxlen=1000)
        self.field_actions: deque = deque(maxlen=1000)
        self.topology_regions: Dict[str, Dict] = {}
        self.gradient_flows: Dict[str, torch.Tensor] = {}
        
        # Temporal dynamics with cognitive capacity limits
        self.temporal_experiences: deque = deque(maxlen=int(temporal_window * 10))
        self.temporal_imprints: List[Any] = []
        self.working_memory: deque = deque(maxlen=brain_config.working_memory_limit)
        
        # Motor smoothing
        self.previous_motor_commands = None
        self.motor_smoothing_factor = brain_config.optimal_prediction_error  # 0.3 = smoothing matches optimal error
        
        # Robot interface dimensions
        self.expected_sensory_dim = None
        self.expected_motor_dim = None
        
        # Performance tracking
        self.brain_cycles = 0
        self.field_evolution_cycles = 0
        self.topology_discoveries = 0
        self.gradient_actions = 0
        
        # Developmental confidence (naive exploration)
        self.enable_developmental_confidence = True
        
        # Position tracking
        self._last_imprint_indices = None
        
        # Maintenance tracking removed - using unified energy system
        
        # Prediction system
        self._prediction_confidence_history = deque(maxlen=100)
        self._improvement_rate_history = deque(maxlen=50)
        self._current_prediction_confidence = brain_config.default_prediction_confidence
        self._predicted_field = None
        
        # Initialize organic energy system
        self.energy_system = OrganicEnergySystem(pattern_memory_size=100, device=self.device)
        if not quiet_mode:
            print(f"   Organic energy system: ENABLED (no modes, smooth transitions)")
            
        # Initialize predictive action system
        self.predictive_actions = PredictiveActionSystem(
            field_shape=self.unified_field.shape,
            motor_dim=self.expected_motor_dim or 4,
            device=self.device
        )
        if not quiet_mode:
            print(f"   Predictive action system: ENABLED (imagine before acting)")
            
        # Initialize reward topology shaper
        self.topology_shaper = RewardTopologyShaper(
            field_shape=self.unified_field.shape,
            device=self.device,
            persistence_factor=0.95,
            max_attractors=20
        )
        if not quiet_mode:
            print(f"   Reward topology shaping: ENABLED (emergent goal-seeking)")
        
        # Maintenance thread disabled - using unified energy system instead
        # self._maintenance_queue = queue.Queue(maxsize=1)
        # self._maintenance_thread = None
        # self._shutdown_maintenance = threading.Event()
        # self._start_maintenance_thread()
        
        # Spontaneous dynamics system with cognitive constants
        self.spontaneous_enabled = True
        self.spontaneous = SpontaneousDynamics(
            field_shape=self.unified_field.shape,
            resting_potential=brain_config.resting_potential,
            spontaneous_rate=brain_config.spontaneous_rate,
            coherence_scale=3.0,
            device=self.device
        )
        self._last_imprint_strength = 0.0
        self._last_prediction_error = brain_config.optimal_prediction_error  # Start at optimal
        self._last_spontaneous_magnitude = 0.0
        self._last_spontaneous_variance = 0.0
        
        # Cognitive autopilot with cognitive constants
        self.cognitive_autopilot = CognitiveAutopilot(
            autopilot_confidence_threshold=brain_config.autopilot_confidence_threshold,
            focused_confidence_threshold=brain_config.focused_confidence_threshold,
            stability_window=10
        )
        
        # Enable blended reality (confidence-based blending)
        self.blended_reality_enabled = True
        
        # Initialize integrated attention system
        # Use parameter if provided, otherwise check config, default to True
        if enable_attention is not None:
            self.attention_enabled = enable_attention
        else:
            self.attention_enabled = self.cognitive_config.brain_config.__dict__.get('enable_attention', True)
        
        # IntegratedAttention removed - only PatternBasedAttention remains
        
        # Initialize pattern-based attention (optional alternative)
        if pattern_attention is not None:
            self.pattern_attention_enabled = pattern_attention
        else:
            self.pattern_attention_enabled = self.cognitive_config.brain_config.__dict__.get('pattern_attention', False)
        
        if self.pattern_attention_enabled:
            self.pattern_attention = FastPatternBasedAttention(
                field_shape=self.unified_field.shape,
                attention_capacity=5,  # Limited attention slots
                device=self.device,
                quiet_mode=quiet_mode
            )
            if not quiet_mode:
                print(f"   Pattern-based attention: ENABLED (fast version)")
        
        # Initialize enhanced dynamics system
        self.enhanced_dynamics_enabled = self.cognitive_config.brain_config.__dict__.get('enhanced_dynamics', True)
        if self.enhanced_dynamics_enabled:
            # Configure phase transitions
            phase_config = PhaseTransitionConfig(
                energy_threshold=0.7,
                stability_threshold=0.3,
                transition_strength=brain_config.novelty_boost,  # Use novelty boost for transitions
                coherence_factor=0.6,
                decay_rate=brain_config.field_decay_rate
            )
            
            # Configure attractors
            attractor_config = AttractorConfig(
                attractor_strength=brain_config.constraint_enforcement_strength,
                repulsor_strength=0.2,
                spatial_spread=min(3.0, spatial_resolution / 2.0),
                temporal_persistence=temporal_window / 2.0,
                auto_discovery=True
            )
            
            # Create adapter for enhanced dynamics
            self.field_adapter = BrainFieldAdapter(self)
            self.enhanced_dynamics = EnhancedFieldDynamics(
                field_impl=self.field_adapter,
                phase_config=phase_config,
                attractor_config=attractor_config,
                quiet_mode=quiet_mode
            )
        
        # Emergent navigation removed - pattern-based motor provides coordinate-free movement
        self.emergent_navigation_enabled = False
        
        # Initialize pattern-based motor generation
        # Always use pattern-based motor (coordinate-free)
        self.pattern_motor_generator = PatternBasedMotorGenerator(
            field_shape=self.unified_field.shape,
            motor_dim=self.expected_motor_dim or 4,
            device=self.device,
            quiet_mode=quiet_mode
        )
        if not quiet_mode:
            print(f"   Pattern-based motor: ENABLED (coordinate-free)")
        
        # Motor cortex for intention-to-action translation
        self.motor_cortex = MotorCortex(
            motor_dim=self.expected_motor_dim or 4,
            device=self.device,
            quiet_mode=quiet_mode
        )
        
        # Emergent navigation removed - using pattern-based motor instead
        
        if not quiet_mode:
            print(f"🧠 Dynamic Unified Field Brain initialized (Full Features)")
            print(f"   Conceptual dimensions: {len(field_dimensions)}D")
            print(f"   Tensor shape: {tensor_shape} ({len(tensor_shape)}D)")
            print(f"   Memory usage: {self._calculate_memory_usage():.1f}MB")
            print(f"   Device: {self.device}")
            print(f"   Spontaneous dynamics: ENABLED")
            print(f"   Blended reality: ENABLED")
            print(f"   Integrated attention: {'ENABLED' if self.attention_enabled else 'DISABLED'}")
            print(f"   Enhanced dynamics: {'ENABLED' if self.enhanced_dynamics_enabled else 'DISABLED'}")            # Emergent navigation removed
            self._print_dimension_summary()
    
    def update_motor_dimensions(self, motor_dim: int):
        """Update motor dimensions after initialization."""
        # Always update to ensure motor generators match
        self.expected_motor_dim = motor_dim
        
        # Recreate motor generators with correct dimensions
        self.pattern_motor_generator = PatternBasedMotorGenerator(
            field_shape=self.unified_field.shape,
            motor_dim=motor_dim,
            device=self.device,
            quiet_mode=True
        )
        
        self.motor_cortex = MotorCortex(
            motor_dim=motor_dim,
            device=self.device,
            activation_threshold=0.1,
            confidence_threshold=0.05,
            max_amplification=3.0,
            quiet_mode=True
        )
        
        # Update predictive action system
        self.predictive_actions = PredictiveActionSystem(
            field_shape=self.unified_field.shape,
            motor_dim=motor_dim,
            device=self.device
        )

    @property
    def developmental_confidence(self) -> float:
        """
        Calculate developmental confidence factor (0.0 to 1.0).
        
        This implements a biologically-inspired confidence boost for naive brains,
        similar to how young animals exhibit fearless exploration before experience
        teaches caution. This is NOT a hack but a principled approach to the
        bootstrap problem: how does a brain with no experience begin learning?
        
        In nature, evolution provides innate behaviors and reflexes. Here, we
        provide elevated baseline confidence that naturally diminishes as the
        brain accumulates memories and learns from experience.
        
        Returns:
            float: Confidence boost factor (1.0 = maximum naive confidence, 0.0 = experienced)
        """
        if not self.enable_developmental_confidence:
            return 0.0
            
        # Memory saturation: how "full" is our long-term memory
        memory_saturation = min(1.0, len(self.topology_regions) / TOPOLOGY_REGIONS_MAX)
        
        # Developmental confidence is inverse of memory saturation
        # Empty brain = high confidence, Full brain = no boost
        return 1.0 - memory_saturation
    
    def _calculate_memory_usage(self) -> float:
        """Calculate field memory usage in MB."""
        elements = 1
        for dim in self.tensor_shape:
            elements *= dim
        return (elements * 4) / (1024 * 1024)
    
    def _print_dimension_summary(self):
        """Print summary of dimension organization."""
        from collections import defaultdict
        family_counts = defaultdict(int)
        for dim in self.field_dimensions:
            family_counts[dim.family] += 1
        
        print("   Conceptual dimension families:")
        for family, count in family_counts.items():
            tensor_range = self.dimension_mapping['family_tensor_ranges'].get(family, (0, 0))
            print(f"      {family.value}: {count}D conceptual → tensor indices {tensor_range}")
    
    def process_robot_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Complete robot brain cycle with all advanced features.
        """
        cycle_start = time.perf_counter()
        
        # 1. Convert robot sensory input to unified field experience
        field_experience = self._robot_sensors_to_field_experience(sensory_input)
        
        # 2. Prediction and confidence tracking
        if self._predicted_field is not None:
            self._update_prediction_confidence(field_experience)
            
            # Learn from prediction errors if we have previous action
            if hasattr(self, '_last_action_taken'):
                self.predictive_actions.update_predictions(
                    action_taken=self._last_action_taken,
                    actual_outcome=self.unified_field,
                    actual_sensory=torch.tensor(sensory_input, device=self.device)
                )
            
        # Apply developmental confidence boost
        if self.enable_developmental_confidence:
            dev_confidence = self.developmental_confidence
            if dev_confidence > 0:
                # Boost prediction confidence for naive brains
                # Max boost of 0.3 when completely naive
                confidence_boost = dev_confidence * 0.3
                self._current_prediction_confidence = min(1.0, 
                    self._current_prediction_confidence + confidence_boost)
        
        # Update cognitive autopilot state
        prediction_error = 0.0
        if hasattr(self, '_last_prediction_error'):
            prediction_error = self._last_prediction_error
        
        # Adjust prediction confidence for cognitive autopilot
        # Naive brains think they know more than they do
        effective_prediction_confidence = self._current_prediction_confidence
        if self.enable_developmental_confidence:
            dev_confidence = self.developmental_confidence
            # Inflate confidence for cognitive mode decision
            # This makes naive brains more likely to use autopilot
            confidence_inflation = dev_confidence * 0.2  # Max 20% inflation
            effective_prediction_confidence = min(1.0, 
                self._current_prediction_confidence + confidence_inflation)
            
        autopilot_state = self.cognitive_autopilot.update_cognitive_state(
            prediction_confidence=effective_prediction_confidence,
            prediction_error=prediction_error,
            brain_state={'field_energy': float(torch.mean(torch.abs(self.unified_field)))}
        )
        
        # 3. Imprint experience in unified field
        self._imprint_unified_experience(field_experience)
        
        # 4. Add to temporal experiences
        self.temporal_experiences.append(field_experience)
        self.field_experiences.append(field_experience)
        
        # 5. Process attention if enabled
        attention_data = None
        if self.pattern_attention_enabled and hasattr(self, 'pattern_attention'):
            # Use pattern-based attention (coordinate-free)
            sensory_patterns = {
                'primary': torch.tensor(sensory_input[:-1], device=self.device) if len(sensory_input) > 1 else torch.tensor(sensory_input, device=self.device)
            }
            attention_data = self.pattern_attention.process_field_patterns(
                field=self.unified_field,
                sensory_patterns=sensory_patterns
            )
        # IntegratedAttention removed - only pattern-based attention available
        
        # 6. Process emergent navigation if enabled
        navigation_state = None
        if self.emergent_navigation_enabled and hasattr(self, 'emergent_spatial'):
            # Process spatial experience
            navigation_state = self.emergent_spatial.process_spatial_experience(
                current_field=self.unified_field,
                sensory_input=sensory_input,
                reward=sensory_input[-1] if len(sensory_input) > 24 else 0.0
            )
        
        # 7. Update energy system (replaces old energy management)
        sensory_pattern = torch.tensor(sensory_input[:-1], device=self.device) if len(sensory_input) > 1 else torch.tensor(sensory_input, device=self.device)
        reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
        
        # Process field dynamics for energy-based behavior
        energy_dynamics = self.energy_system.process_field_dynamics(
            field=self.unified_field,
            sensory_pattern=sensory_pattern,
            prediction_error=self._last_prediction_error if hasattr(self, '_last_prediction_error') else 0.5,
            reward=reward
        )
        
        # Get smooth behavioral influence (no modes or thresholds)
        energy_behavior = energy_dynamics['behavior']
        self.energy_recommendations = energy_behavior  # Store for field evolution
        
        # 7b. Process reward for topology shaping
        # This creates lasting impressions in the field landscape
        if abs(reward) > 0.1:  # Only significant rewards shape topology
            deformation = self.topology_shaper.process_reward(
                current_field=self.unified_field,
                reward=reward,
                threshold=0.1
            )
            if not self.quiet_mode and deformation is not None:
                print(f"🎯 Reward {reward:+.2f} creating topology deformation")
        
        # 8. Evolve unified field with all dynamics
        self._evolve_unified_field()
        
        # 8. Apply attention modulation to field
        if attention_data:
            if self.pattern_attention_enabled:
                # Pattern-based attention modulation
                self.unified_field = self.pattern_attention.modulate_field_with_attention(
                    self.unified_field,
                    attention_data
                )
            # IntegratedAttention field modulation removed
        
        # 9. Update topology regions (memory formation)
        if self.brain_cycles % 10 == 0:
            self._update_topology_regions()
            
        # 10. Consolidate pattern memory if needed (organic forgetting)
        if self.brain_cycles % 100 == 0 and self.energy_system.should_consolidate_memory():
            self.energy_system.consolidate_patterns()
        
        # 10. Generate motor commands with predictive selection
        
        # Get pattern statistics first
        pattern_stats = self.pattern_motor_generator.get_pattern_stats()
        pattern_features = pattern_stats.get('pattern_features', {})
        
        # First, use pattern motor generator to get baseline action
        improvement_rate = 0.0
        if len(self._improvement_rate_history) > 0:
            improvement_rate = np.mean(list(self._improvement_rate_history))
        
        spontaneous_info = {
            'magnitude': self._last_spontaneous_magnitude,
            'variance': self._last_spontaneous_variance,
            'weight': self.blended_reality.calculate_spontaneous_weight() if hasattr(self, 'blended_reality') else 0.5
        }
        
        baseline_action = self.pattern_motor_generator.generate_motor_action(
            current_field=self.unified_field,
            experience=field_experience,
            improvement_rate=improvement_rate,
            spontaneous_info=spontaneous_info
        )
        
        # Now use predictive system to imagine outcomes
        exploration_drive = self.energy_recommendations.get('exploration_drive', 0.5) if hasattr(self, 'energy_recommendations') else 0.5
        
        # Full predictive selection - we have 750ms to work with!
        # Generate multiple action candidates
        candidates = self.predictive_actions.generate_action_candidates(
            current_field=self.unified_field,
            current_patterns=pattern_features,
            n_candidates=5  # Full set of candidates
        )
        
        # Add baseline action as first candidate
        candidates[0].motor_pattern = baseline_action.output_stream
        
        # Preview outcomes with full lookahead
        for candidate in candidates:
            self.predictive_actions.preview_action_outcome(
                current_field=self.unified_field,
                action=candidate,
                evolution_steps=5  # Full preview depth
            )
            
        # Select best action based on predicted outcomes
        selected_action = self.predictive_actions.select_action(
            candidates=candidates,
            exploration_drive=exploration_drive
        )
        
        # Use selected action
        action = baseline_action  # Keep metadata
        action.output_stream = selected_action.motor_pattern
        
        # Log when exploration drives novel actions
        if not self.quiet_mode and self.brain_cycles % 100 == 0:
            if selected_action.uncertainty > 0.7:
                print(f"🎲 Exploratory action selected (uncertainty: {selected_action.uncertainty:.2f})")
            elif selected_action.anticipated_value > 0.8:
                print(f"💎 High-value action selected (value: {selected_action.anticipated_value:.2f})")
        # Set timestamp
        action.timestamp = time.time()
        
        # Apply developmental confidence to action confidence
        effective_action_confidence = action.confidence
        if self.enable_developmental_confidence:
            dev_confidence = self.developmental_confidence
            # Boost action confidence for naive brains
            confidence_boost = dev_confidence * 0.4  # Stronger boost for actions
            effective_action_confidence = min(1.0, action.confidence + confidence_boost)
        
        motor_commands, motor_feedback = self.motor_cortex.process_intentions(
            intentions=action.output_stream,
            confidence=effective_action_confidence,
            pattern_features=pattern_features
        )
        
        # Apply energy-based motor noise for exploration
        if hasattr(self, 'energy_recommendations'):
            motor_noise = self.energy_recommendations.get('motor_noise', 0.0)
            if motor_noise > 0:
                # Add exploration noise when hungry
                noise = torch.randn_like(motor_commands) * motor_noise
                motor_commands = motor_commands + noise
        
        # Update action with motor cortex output
        action.output_stream = motor_commands
        self.field_actions.append(action)
        
        # Store action for learning from outcomes
        self._last_action_taken = motor_commands
        
        # 11. Create prediction for next cycle
        self._predicted_field = self._predict_next_field_state()
        
        # 12. Maintenance removed - energy system handles everything
        
        # Update tracking
        self.brain_cycles += 1
        cycle_time = time.perf_counter() - cycle_start
        self._last_cycle_time = cycle_time
        
        # Create comprehensive brain state
        brain_state = {
            'cycle': self.brain_cycles,
            'cycle_time_ms': cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.unified_field))),
            'max_activation': float(torch.max(torch.abs(self.unified_field))),

            'topology_regions': len(self.topology_regions),
            'prediction_confidence': self._current_prediction_confidence,
            'working_memory_size': len(self.working_memory),
            'conceptual_dims': self.total_dimensions,
            'tensor_dims': len(self.tensor_shape),
            'spontaneous_enabled': self.spontaneous_enabled,
            'cognitive_mode': self.cognitive_autopilot.current_mode.value,
            'cognitive_recommendations': self.cognitive_autopilot._generate_system_recommendations({}),
            'pattern_motor': True,  # Always pattern-based (coordinate-free)
            'pattern_attention_enabled': self.pattern_attention_enabled,
            'developmental_confidence': self.developmental_confidence,
            'memory_saturation': len(self.topology_regions) / TOPOLOGY_REGIONS_MAX,
            'energy_state': {
                'energy': energy_dynamics['smoothed_energy'] if 'energy_dynamics' in locals() else 0.5,
                'novelty': energy_dynamics['novelty'] if 'energy_dynamics' in locals() else 0.0,
                'exploration_drive': energy_behavior['exploration_drive'] if 'energy_behavior' in locals() else 0.5
            },
            'topology_shaping': self.topology_shaper.get_topology_state()
        }
        
        # Add attention state if enabled
        if self.pattern_attention_enabled and hasattr(self, 'pattern_attention') and attention_data:
            brain_state['attention'] = self.pattern_attention.get_attention_metrics()
            brain_state['attention']['type'] = 'pattern-based'
        # IntegratedAttention state removed
        # Emergent navigation removed
        
        # Add motor cortex statistics
        brain_state['motor_cortex'] = self.motor_cortex.get_statistics()
        
        # Handle different action types
        if hasattr(action, 'output_stream'):
            motor_output = action.output_stream.tolist()
        elif hasattr(action, 'motor_commands'):
            motor_output = action.motor_commands.tolist()
        else:
            motor_output = [0.0] * (self.expected_motor_dim or 4)
        
        return motor_output, brain_state
    
    def _robot_sensors_to_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """
        Map robot sensors to unified field coordinates using dynamic mapping.
        """
        # Create field coordinates tensor
        field_coords = torch.zeros(self.total_dimensions, device=self.device)
        
        # Extract reward if present (last sensor by convention)
        reward = sensory_input[-1] if len(sensory_input) > 0 else 0.0
        field_intensity = self.cognitive_config.brain_config.default_prediction_confidence + reward * self.cognitive_config.brain_config.default_prediction_confidence  # Map [-1,1] to [0,1]
        
        # Map sensors to conceptual dimensions
        # This is a simplified mapping - in production, use more sophisticated mapping
        
        # 1. Map position-related sensors to spatial dimensions
        spatial_dims = [d for d in self.field_dimensions if d.family == FieldDynamicsFamily.SPATIAL]
        
        # Map first 3 sensors to first 3 spatial dimensions
        for i in range(min(3, len(spatial_dims), len(sensory_input))):
            dim_idx = self.field_dimensions.index(spatial_dims[i])
            field_coords[dim_idx] = sensory_input[i] * 2.0 - 1.0  # Map [0,1] to [-1,1]
        
        # Map remaining spatial dimensions from other sensors
        for i in range(3, len(spatial_dims)):
            if i < len(sensory_input):
                dim_idx = self.field_dimensions.index(spatial_dims[i])
                field_coords[dim_idx] = sensory_input[i] * 2.0 - 1.0
        
        # 2. Map other sensors across remaining dimensions
        sensor_idx = 3
        for dim in self.field_dimensions:
            if dim.family != FieldDynamicsFamily.SPATIAL:
                dim_idx = self.field_dimensions.index(dim)
                if sensor_idx < len(sensory_input):
                    field_coords[dim_idx] = sensory_input[sensor_idx] * 2.0 - 1.0
                    sensor_idx += 1
        
        # Create experience
        experience = UnifiedFieldExperience(
            timestamp=time.time(),
            raw_input_stream=torch.tensor(sensory_input, dtype=torch.float32, device=self.device),
            field_coordinates=field_coords,
            field_intensity=field_intensity,
            dynamics_family_activations={}  # Empty for now
        )
        
        return experience
    
    def _imprint_unified_experience(self, experience: UnifiedFieldExperience):
        """
        Imprint experience in the unified field using dynamic tensor mapping.
        """
        # Map conceptual coordinates to tensor indices
        tensor_indices = self._conceptual_to_tensor_indices(experience.field_coordinates)
        
        # Create local region for imprinting (3x3x3 in spatial dims)
        region_slices = []
        for i, idx in enumerate(tensor_indices):
            if i < 3:  # Spatial dimensions get regions
                start = max(0, idx - 1)
                end = min(self.tensor_shape[i], idx + 2)
                region_slices.append(slice(start, end))
            else:  # Other dimensions get single points
                region_slices.append(idx)
        
        # Imprint with intensity based on reward and energy state
        # Use base imprint strength from blended reality config
        blended_config = self.cognitive_config.blended_reality_config
        base_imprint_strength = blended_config.base_imprint_strength * experience.field_intensity
        
        # Apply energy-based sensory amplification
        if hasattr(self, 'energy_recommendations'):
            sensory_amplification = self.energy_recommendations.get('sensory_amplification', 1.0)
            imprint_strength = base_imprint_strength * sensory_amplification
        else:
            imprint_strength = base_imprint_strength
            
        self.unified_field[tuple(region_slices)] += imprint_strength
        
        # Store position for tracking
        self._last_imprint_indices = tensor_indices
        # Track imprint strength for spontaneous gating
        self._last_imprint_strength = imprint_strength
    
    def _conceptual_to_tensor_indices(self, field_coords: torch.Tensor) -> List[int]:
        """
        Convert conceptual field coordinates to tensor indices.
        """
        tensor_indices = []
        
        # For each tensor dimension, find which conceptual dimensions map to it
        for tensor_idx in range(len(self.tensor_shape)):
            # Find conceptual dimensions that map to this tensor position
            conceptual_dims = []
            for dim_name, mapped_idx in self.dimension_mapping['conceptual_to_tensor'].items():
                if mapped_idx == tensor_idx:
                    # Find the dimension object
                    for dim in self.field_dimensions:
                        if dim.name == dim_name:
                            conceptual_dims.append(dim)
                            break
            
            if conceptual_dims:
                # Average the coordinates of conceptual dimensions mapping to this tensor position
                dim_indices = [self.field_dimensions.index(d) for d in conceptual_dims]
                avg_coord = torch.mean(field_coords[dim_indices])
                
                # Convert to tensor index
                tensor_size = self.tensor_shape[tensor_idx]
                normalized = (avg_coord + 1.0) / 2.0  # Map [-1,1] to [0,1]
                idx = int(torch.clamp(normalized * tensor_size, 0, tensor_size - 1))
                tensor_indices.append(idx)
            else:
                # Default to center if no mapping
                tensor_indices.append(self.tensor_shape[tensor_idx] // 2)
        
        return tensor_indices
    
    def _evolve_unified_field(self):
        """
        Evolve the unified field with all dynamics.
        """
        # 1. Apply energy-based modulation (organic, no pruning needed)
        if hasattr(self, 'energy_recommendations'):
            self.unified_field = self.energy_system.modulate_field(
                self.unified_field,
                self.energy_recommendations
            )
        
        # 2. Diffusion (spreading activation)
        if self.field_diffusion_rate > 0:
            # Apply diffusion in spatial dimensions
            spatial_range = self.dimension_mapping['family_tensor_ranges'].get(
                FieldDynamicsFamily.SPATIAL, (0, min(3, len(self.tensor_shape)))
            )
            
            for dim in range(spatial_range[0], spatial_range[1]):
                # Simple diffusion using shifts
                shifted_forward = torch.roll(self.unified_field, shifts=1, dims=dim)
                shifted_backward = torch.roll(self.unified_field, shifts=-1, dims=dim)
                
                diffusion = (shifted_forward + shifted_backward - 2 * self.unified_field) / 2
                self.unified_field += self.field_diffusion_rate * diffusion
        
        # 3. Spontaneous dynamics
        if self.spontaneous_enabled:
            # Get spontaneous weight from energy behavior
            if hasattr(self, 'energy_recommendations'):
                spontaneous_weight = self.energy_recommendations.get('spontaneous_weight', 0.5)
            else:
                spontaneous_weight = 0.5
            
            # sensory_gating: 0 = pure spontaneous, 1 = suppress spontaneous
            # So we invert our spontaneous weight
            sensory_gating = 1.0 - spontaneous_weight
            
            # Add spontaneous activity
            spontaneous_activity = self.spontaneous.generate_spontaneous_activity(
                self.unified_field,
                sensory_gating=sensory_gating
            )
            self.unified_field += spontaneous_activity
            
            # Track spontaneous magnitude for exploration
            self._last_spontaneous_magnitude = torch.mean(torch.abs(spontaneous_activity)).item()
            
            # Track spontaneous pattern richness (variance indicates complexity)
            self._last_spontaneous_variance = torch.var(spontaneous_activity).item()
        
        # 4. Apply enhanced dynamics if enabled
        if self.enhanced_dynamics_enabled and hasattr(self, 'enhanced_dynamics'):
            # Enhanced dynamics handles phase transitions, attractors, and energy redistribution
            # It will call back to our adapter which uses _imprint_unified_field
            self.enhanced_dynamics.evolve_with_enhancements(
                dt=0.1,  # Standard evolution timestep
                current_input_stream=None  # Already handled by brain
            )
        
        # 5. Apply reward topology influence
        topology_influence = self.topology_shaper.apply_topology_influence(self.unified_field)
        self.unified_field += topology_influence
        
        # 6. Baseline prevention handled by energy system
        
        self.field_evolution_cycles += 1
    

    def _update_topology_regions(self):
        """
        Update memory regions based on field topology.
        """
        # Find high-activation regions
        # Use topology stability threshold for high activation detection
        high_activation = self.unified_field > self.cognitive_config.brain_config.topology_stability_threshold
        
        # Find stable regions (low variance)
        field_std = torch.std(self.unified_field)
        stable_regions = self.unified_field > (field_std * self.cognitive_config.brain_config.default_prediction_confidence)
        
        # Combine criteria
        memory_regions = high_activation & stable_regions
        
        # Extract region centers
        if torch.any(memory_regions):
            # Find connected components (simplified)
            region_indices = torch.nonzero(memory_regions, as_tuple=False)
            
            # Create/update topology regions
            for idx in region_indices[:10]:  # Limit to top 10 regions
                region_id = f"region_{self.brain_cycles}_{len(self.topology_regions)}"
                self.topology_regions[region_id] = {
                    'center': idx.cpu().numpy(),
                    'activation': float(self.unified_field[tuple(idx)]),
                    'creation_cycle': self.brain_cycles,
                    'last_activation': self.brain_cycles
                }
            
            # Prune old regions
            if len(self.topology_regions) > TOPOLOGY_REGIONS_MAX:
                # Remove oldest regions
                sorted_regions = sorted(self.topology_regions.items(), 
                                      key=lambda x: x[1]['last_activation'])
                for region_id, _ in sorted_regions[:20]:
                    del self.topology_regions[region_id]
    
    
    def _update_prediction_confidence(self, experience: UnifiedFieldExperience):
        """
        Update prediction confidence based on actual vs predicted field.
        """
        if self._predicted_field is None:
            return
        
        # Calculate prediction error in local region
        tensor_indices = self._conceptual_to_tensor_indices(experience.field_coordinates)
        
        # Get 3x3x3 region around position
        region_slices = []
        for i, idx in enumerate(tensor_indices[:3]):
            start = max(0, idx - 1)
            end = min(self.tensor_shape[i], idx + 2)
            region_slices.append(slice(start, end))
        
        # Add remaining dims as single points
        for i in range(3, len(tensor_indices)):
            region_slices.append(tensor_indices[i])
        
        # Compare predicted vs actual
        predicted_region = self._predicted_field[tuple(region_slices)]
        actual_region = self.unified_field[tuple(region_slices)]
        
        # Calculate error
        prediction_error = torch.mean(torch.abs(predicted_region - actual_region))
        
        # Also consider the magnitude of activation
        # Low activation with low error shouldn't mean high confidence
        activation_magnitude = torch.mean(torch.abs(actual_region))
        
        # Confidence should be high when:
        # 1. Prediction error is low AND
        # 2. We're actually predicting something meaningful (not just baseline)
        
        # More nuanced baseline handling
        baseline_factor = 1.0
        if activation_magnitude < self.cognitive_config.brain_config.resting_potential * 5:  # More lenient threshold
            # Near baseline - reduce confidence but don't fix it at minimum
            # Use a smooth transition instead of hard cutoff
            baseline_factor = 0.3 + 0.7 * (activation_magnitude / (self.cognitive_config.brain_config.resting_potential * 5))
        
        # Calculate base confidence from prediction error
        if activation_magnitude > 1e-6:  # Avoid division by zero
            # Scale error by activation magnitude for normalization
            # Reduced multiplier (1.0 instead of 2.0) for less sensitivity
            normalized_error = prediction_error / (activation_magnitude + self.cognitive_config.brain_config.attention_threshold)
            base_confidence = float(1.0 / (1.0 + normalized_error * 1.0))
        else:
            # Very low activation - moderate confidence
            base_confidence = 0.5
        
        # Apply baseline factor
        raw_confidence = base_confidence * baseline_factor
        
        # Add small random variation to prevent getting stuck
        # This represents inherent uncertainty in predictions
        confidence_noise = np.random.normal(0, 0.05)
        raw_confidence = raw_confidence + confidence_noise
        
        # Ensure reasonable bounds with softer limits
        # Allow more variation instead of hard clamping at 0.21
        self._current_prediction_confidence = float(np.clip(raw_confidence, 0.1, 0.95))
        
        self._prediction_confidence_history.append(self._current_prediction_confidence)
        self._last_prediction_error = float(prediction_error)
        
        # Track improvement rate
        if len(self._prediction_confidence_history) > 10:
            recent_confidence = np.mean(list(self._prediction_confidence_history)[-10:])
            older_confidence = np.mean(list(self._prediction_confidence_history)[-20:-10])
            improvement = recent_confidence - older_confidence
            self._improvement_rate_history.append(improvement)
            
            # Periodic confidence "reset" if stuck at extremes
            if len(self._prediction_confidence_history) > 50:
                last_50_confidence = list(self._prediction_confidence_history)[-50:]
                confidence_variance = np.var(last_50_confidence)
                
                # If confidence has been too stable (low variance), add a reset
                if confidence_variance < 0.01:
                    avg_confidence = np.mean(last_50_confidence)
                    if avg_confidence < 0.3 or avg_confidence > 0.8:
                        # Stuck at extreme - nudge toward center
                        reset_target = 0.5 + np.random.normal(0, 0.1)
                        self._current_prediction_confidence = float(np.clip(reset_target, 0.3, 0.7))
                        if not self.quiet_mode:
                            print(f"🔄 Confidence reset: {avg_confidence:.2f} → {self._current_prediction_confidence:.2f}")
    
    def _predict_next_field_state(self) -> torch.Tensor:
        """
        Predict the next field state for confidence tracking.
        """
        # Simple prediction: current field evolved one step
        predicted = self.unified_field.clone()
        
        # Apply decay
        predicted *= self.field_decay_rate
        
        # Apply simple diffusion prediction
        if self.field_diffusion_rate > 0:
            for dim in range(min(3, len(self.tensor_shape))):
                shifted_forward = torch.roll(predicted, shifts=1, dims=dim)
                shifted_backward = torch.roll(predicted, shifts=-1, dims=dim)
                predicted += self.field_diffusion_rate * (
                    shifted_forward + shifted_backward - 2 * predicted
                ) / 2
        
        return predicted
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state for telemetry"""
        return {
            'cycle': self.brain_cycles,
            'cognitive_mode': self.cognitive_autopilot.current_mode.value if self.cognitive_autopilot else 'unknown',

            'cycle_time_ms': getattr(self, '_last_cycle_time', 0) * 1000
        }
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get field statistics for telemetry"""
        field_energy = float(torch.mean(torch.abs(self.unified_field)))
        max_activation = float(torch.max(torch.abs(self.unified_field)))
        
        # Calculate stability index
        field_std = float(torch.std(self.unified_field))
        stability_index = 1.0 / (1.0 + field_std) if field_std > 0 else 1.0
        
        return {
            'field_energy': field_energy,
            'max_activation': max_activation,
            'stability_index': stability_index,
            'total_activation': float(torch.sum(torch.abs(self.unified_field)))
        }
    
    # Maintenance methods removed - using unified energy system
    
    # _start_maintenance_thread removed - using unified energy system
    
    def __del__(self):
        """Cleanup when brain is destroyed."""
        pass  # Organic system has no threads to clean up