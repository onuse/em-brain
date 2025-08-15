"""
Unified Field Brain

4D tensor architecture where all cognitive properties emerge from field dynamics.
Strategic patterns shape behavior through field gradients.
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
from .reward_topology_shaping import RewardTopologyShaper
from .consolidation_system import ConsolidationSystem
from .topology_region_system import TopologyRegionSystem
from .predictive_field_system import PredictiveFieldSystem
from .active_vision_system import ActiveVisionSystem
from .field_strategic_planner import FieldStrategicPlanner, StrategicPattern
from .active_audio_system import ActiveAudioSystem
from .active_tactile_system import ActiveTactileSystem
from ...utils.tensor_ops import create_randn, field_information, field_stats, apply_diffusion
from ...utils.error_handling import (
    validate_list_input, validate_tensor_shape, ErrorContext,
    BrainError, safe_tensor_op
)
from .pattern_cache_pool import PatternCachePool


class UnifiedFieldBrain:
    """
    Unified field brain with 4D tensor architecture.
    
    Core principles:
    - Single 4D tensor field [D, H, W, C]
    - All cognition emerges from field dynamics
    - Strategic patterns shape behavior through gradients
    - No symbolic representations or explicit plans
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
        
        # Store dimensions
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
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
            print(f"🧠 Unified Field Brain")
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
        # Get decay multiplier from cognitive constants
        from ...parameters.cognitive_constants import StabilityConstants
        decay_multiplier = StabilityConstants.FIELD_DECAY_MULTIPLIER
        
        self.field_dynamics = EvolvedFieldDynamics(
            field_shape=self.unified_field.shape,
            pattern_memory_size=100,
            confidence_window=50,
            initial_spontaneous_rate=self.spontaneous_rate,
            initial_resting_potential=self.cognitive_config.brain_config.resting_potential,
            temporal_features=16,  # For working memory
            dynamics_features=16,  # For self-modifying dynamics
            decay_multiplier=decay_multiplier,
            device=self.device
        )
        
        # Initialize the dynamics features in the field
        self.field_dynamics.initialize_field_dynamics(self.unified_field)
        
        
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
        
        
        # Active sensing systems (Phase 5)
        # Vision is the primary implementation
        self.active_vision = ActiveVisionSystem(
            field_shape=self.tensor_shape,
            motor_dim=motor_dim - 1,  # Exclude confidence dimension
            device=self.device
        )
        self.use_active_vision = False  # Will be enabled when ready
        self._glimpse_adapter = None  # Set by enable_active_vision
        
        # Stubs for future modalities
        self.active_audio = None  # Created when needed
        self.active_tactile = None  # Created when needed
        
        # Strategic Planning: Field patterns that shape behavior
        self.strategic_planner = None
        self.use_strategic_planning = False
        self.current_strategic_pattern = None
        self._last_reward_signal = 0.0
        
    def enable_hierarchical_prediction(self, enable: bool = True):
        """
        Enable Phase 3: Hierarchical prediction at multiple timescales.
        
        When enabled, the brain predicts at immediate, short-term, long-term,
        and abstract timescales simultaneously.
        """
        self.predictive_field.enable_hierarchical_prediction(enable)
        if not self.quiet_mode:
            status = "enabled" if enable else "disabled"
            print(f"🧠 Hierarchical prediction {status}")
    
    def enable_strategic_planning(self, enable: bool = True):
        """
        Enable field-native strategic planning through pattern discovery.
        
        Instead of simulating action sequences, discovers field patterns that
        create beneficial behavioral attractors through natural dynamics.
        
        Args:
            enable: Whether to enable strategic planning
        """
        self.use_strategic_planning = enable
        
        if enable and self.strategic_planner is None:
            # Create the strategic planner
            self.strategic_planner = FieldStrategicPlanner(
                field_shape=self.tensor_shape,
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_cortex.motor_dim - 1,
                device=self.device
            )
                
            if not self.quiet_mode:
                print(f"🧠 Strategic planning enabled (field-native patterns)")
        elif not enable:
            self.use_strategic_planning = False
            if not self.quiet_mode:
                print("🧠 Strategic planning disabled")
    
    def enable_active_vision(self, enable: bool = True, glimpse_adapter=None):
        """
        Enable Phase 5: Active vision through predictive sampling.
        
        When enabled, the brain directs attention to uncertain areas,
        creating natural eye movements like saccades and smooth pursuit.
        
        Args:
            enable: Whether to enable active vision
            glimpse_adapter: Optional GlimpseSensoryAdapter instance
        """
        self.use_active_vision = enable
        self._glimpse_adapter = glimpse_adapter
        
        # Active vision requires both hierarchical and action prediction
        if enable:
            if not self.predictive_field.use_hierarchical:
                self.enable_hierarchical_prediction(True)
        
        if not self.quiet_mode:
            status = "enabled" if enable else "disabled"
            print(f"🧠 Active vision {status}")
    
    @torch.no_grad()  # Disable gradient computation for performance
    def process_robot_cycle(self, sensory_input: List[float], glimpse_data: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main processing cycle - simplified version.
        
        Args:
            sensory_input: Regular sensor values
            glimpse_data: Optional high-resolution glimpse data from active vision
        """
        cycle_start = time.perf_counter()
        
        try:
            # Validate input
            with ErrorContext("validating sensory input"):
                # Debug: Check if sensory_input contains tensors
                if self.brain_cycles == 6 and not self.quiet_mode:
                    # print(f"[DEBUG Cycle 6] sensory_input types: {[type(v).__name__ for v in sensory_input[:5]]}")
                    # print(f"[DEBUG Cycle 6] sensory_input values: {sensory_input[:5]}")
                    pass
                
                # Allow variable length for different robot configurations
                validate_list_input(sensory_input, len(sensory_input), "sensory_input", -10.0, 10.0)
            
            # 1. Create field experience from sensors
            with ErrorContext("creating field experience"):
                experience = self._create_field_experience(sensory_input)
            
            # Track uncertainty before any glimpse processing
            uncertainty_before = self._compute_current_uncertainty() if self.use_active_vision else 0.5
            
            # Phase 5: Process glimpse data if active vision is enabled
            if self.use_active_vision and glimpse_data is not None and self._glimpse_adapter is not None:
                
                # Integrate glimpse data with regular sensory input
                glimpse_field = self._glimpse_adapter.to_field_space_with_glimpses(
                    sensory=sensory_input,
                    glimpse_data=glimpse_data
                )
                
                # Blend glimpse information with high priority
                # Glimpses are focused attention, so they get higher weight
                glimpse_influence = self.sensory_mapping.process_patterns(
                    patterns=[glimpse_field],
                    reward=experience.field_intensity,
                    exploration_weight=0.8  # High weight for focused attention
                )
                
                # Apply glimpse influence to field
                self.unified_field += glimpse_influence * 1.5  # Boost glimpse importance
            
            # 2. Update prediction tracking
            if self.brain_cycles % 50 == 0 and not self.quiet_mode:
                # print(f"[DEBUG Cycle {self.brain_cycles}] _predicted_sensory is None: {self._predicted_sensory is None}")
                if self._predicted_sensory is not None:
                    # print(f"[DEBUG Cycle {self.brain_cycles}] _predicted_sensory shape: {self._predicted_sensory.shape}")
                    pass
            
            if self._predicted_sensory is not None and experience.raw_input_stream is not None:
                # Compare predicted vs actual sensory input
                # Check if input includes a reward (more elements than sensory_dim)
                if experience.raw_input_stream.shape[0] > self.sensory_dim:
                    # Input has reward as last element, exclude it
                    actual_sensory = experience.raw_input_stream[:self.sensory_dim]
                else:
                    # Input is just sensors, use as-is
                    actual_sensory = experience.raw_input_stream
                
                # Debug logging for size mismatch
                if actual_sensory.shape[0] != self._predicted_sensory.shape[0]:
                    if not self.quiet_mode and self.brain_cycles % 10 == 0:
                        # print(f"[DEBUG] Size mismatch: actual_sensory={actual_sensory.shape[0]}, predicted={self._predicted_sensory.shape[0]}")
                        # print(f"[DEBUG] raw_input_stream={experience.raw_input_stream.shape[0]}, sensory_dim={self.sensory_dim}")
                        pass
                
                if actual_sensory.shape[0] == self._predicted_sensory.shape[0]:
                    sensory_error = torch.mean(torch.abs(actual_sensory - self._predicted_sensory)).item()
                    self._last_prediction_error = actual_sensory - self._predicted_sensory  # Store tensor error for Phase 2
                    
                    # Debug: Show prediction quality
                    if self.brain_cycles % 50 == 0 and not self.quiet_mode:
                        # print(f"[DEBUG Cycle {self.brain_cycles}] Sensory prediction error: {sensory_error:.3f}")
                        # print(f"[DEBUG Cycle {self.brain_cycles}] Actual sensory (first 5): {actual_sensory[:5].tolist()}")
                        # print(f"[DEBUG Cycle {self.brain_cycles}] Predicted sensory (first 5): {self._predicted_sensory[:5].tolist()}")
                        pass
                else:
                    # Fallback if dimensions still don't match
                    min_dim = min(actual_sensory.shape[0], self._predicted_sensory.shape[0])
                    sensory_error = torch.mean(torch.abs(actual_sensory[:min_dim] - self._predicted_sensory[:min_dim])).item()
                    self._last_prediction_error = actual_sensory[:min_dim] - self._predicted_sensory[:min_dim]  # Store tensor error
                
                # Natural confidence dynamics through simple formula
                # Model complexity (0 = simple, 1 = complex)
                model_complexity = min(1.0, len(self.topology_region_system.regions) / 50.0)
                
                # Error weight decreases as model develops (natural D-K effect)
                error_weight = 1.5 - 0.5 * model_complexity  # 1.5 → 1.0
                
                # Base confidence higher for simple models (doesn't know what it doesn't know)
                base_confidence = 0.2 * (1.0 - model_complexity) if self.brain_cycles < 50 else 0.0
                
                # Calculate confidence with natural dynamics
                raw_confidence = max(base_confidence, 1.0 - min(1.0, sensory_error * error_weight))
                
                # Momentum decreases over time (early optimism, later realism)
                momentum = 0.9 - min(0.2, self.brain_cycles / 1000.0)
                self._current_prediction_confidence = (
                    momentum * self._current_prediction_confidence + 
                    (1.0 - momentum) * raw_confidence
                )
                
                
                # Process prediction error to improve region-sensor associations
                if hasattr(self, '_last_activated_regions') and self._last_activated_regions:
                    error_stats = self.predictive_field.process_prediction_error(
                        predicted=self._predicted_sensory,
                        actual=actual_sensory,
                        topology_regions=self._last_activated_regions,
                        current_field=self.unified_field
                    )
                    
                    # Phase 2: Send prediction errors to field dynamics for learning
                    if hasattr(self, '_last_prediction_error') and self._last_prediction_error is not None:
                        self.field_dynamics.process_prediction_errors(
                            prediction_errors=self._last_prediction_error,
                            topology_regions=self._last_activated_regions,
                            current_field=self.unified_field
                        )
            elif self._predicted_field is not None:
                # Fallback to field comparison
                prediction_error = torch.mean(torch.abs(self.unified_field - self._predicted_field)).item()
                self._last_prediction_error = prediction_error
                # Natural confidence dynamics through simple formula  
                # Model complexity (0 = simple, 1 = complex)
                model_complexity = min(1.0, len(self.topology_region_system.regions) / 50.0)
                
                # Error weight decreases as model develops (natural D-K effect)
                error_weight = 1.5 - 0.5 * model_complexity  # 1.5 → 1.0
                
                # Base confidence higher for simple models
                base_confidence = 0.2 * (1.0 - model_complexity) if self.brain_cycles < 50 else 0.0
                
                # Calculate confidence with natural dynamics
                raw_confidence = max(base_confidence, 1.0 - min(1.0, prediction_error * error_weight))
                
                # Momentum decreases over time
                momentum = 0.9 - min(0.2, self.brain_cycles / 1000.0)
                self._current_prediction_confidence = (
                    momentum * self._current_prediction_confidence + 
                    (1.0 - momentum) * raw_confidence
                )
            else:
                # First cycle - no prediction yet, so low confidence
                self._current_prediction_confidence = 0.5
            
            # 3. Imprint sensory experience
            self._imprint_experience(experience)
            
            # 4. Process attention
            attention_data = self._process_attention(sensory_input)
            
            # 5. Update unified field dynamics
            reward = sensory_input[-1] if len(sensory_input) > self.sensory_dim else 0.0
            
            # Debug cycle 17
            if self.brain_cycles == 17 and not self.quiet_mode:
                # print(f"[DEBUG Cycle 17] reward value: {reward}, type: {type(reward).__name__}")
                # print(f"[DEBUG Cycle 17] sensory_input length: {len(sensory_input)}, sensory_dim: {self.sensory_dim}")
                if len(sensory_input) > 0:
                    # print(f"[DEBUG Cycle 17] last sensory value: {sensory_input[-1]}, type: {type(sensory_input[-1]).__name__}")
                    pass
            
            # Compute field state (information, novelty, etc.)
            field_state = self.field_dynamics.compute_field_state(self.unified_field)
            novelty = self.field_dynamics.compute_novelty(self.unified_field)
            
            # Update confidence from prediction error
            # Always convert to scalar for confidence update
            if torch.is_tensor(self._last_prediction_error):
                error_scalar = torch.mean(torch.abs(self._last_prediction_error)).detach().item()
            else:
                # This shouldn't happen anymore, but handle it just in case
                error_scalar = abs(float(self._last_prediction_error))
            self.field_dynamics.update_confidence(error_scalar)
            
            # Debug confidence values
            if self.brain_cycles % 100 == 0 and not self.quiet_mode:
                # print(f"[DEBUG] Confidence: current={self._current_prediction_confidence:.3f}, " +
                #       f"smoothed={self.field_dynamics.smoothed_confidence:.3f}, " +
                #       f"error={error_scalar:.3f}")
                pass
            
            # Get unified modulation parameters
            try:
                # Simpler approach - sensory_input should always be a list of floats
                if len(sensory_input) > 0:
                    # Check if any sensor (except reward) has significant input
                    has_input = any(abs(float(v)) > 0.01 for v in sensory_input[:-1])
                else:
                    has_input = False
            except Exception as e:
                print(f"[ERROR] has_input calculation failed: {e}")
                print(f"[ERROR] sensory_input types: {[type(v).__name__ for v in sensory_input[:5]]}")
                print(f"[ERROR] sensory_input values: {sensory_input[:5]}")
                raise
            self.modulation = self.field_dynamics.compute_field_modulation(
                field_state, has_sensory_input=has_input
            )
            
            # 6. Process reward topology
            if abs(reward.item() if torch.is_tensor(reward) else reward) > 0.1:
                self.topology_shaper.process_reward(
                    current_field=self.unified_field,
                    reward=reward,
                    threshold=0.1
                )
                
                # Also give feedback to adaptive motor cortex
                if hasattr(self.motor_cortex, 'inject_reward_feedback'):
                    self.motor_cortex.inject_reward_feedback(reward)
            
            # Store reward signal for strategic planning
            self._last_reward_signal = reward.item() if torch.is_tensor(reward) else reward
            
            # 7. Strategic pattern discovery (background)
            if self.use_strategic_planning and self.strategic_planner is not None:
                # Measure current field tensions
                current_tensions = self.strategic_planner._measure_field_tensions(self.unified_field)
                total_tension = current_tensions['total']
                
                # Trigger discovery when:
                # - High tension (> 0.6) indicates unmet drives
                # - Every 20 cycles as baseline
                # - No pattern exists yet
                # - Low confidence (< 0.3) creates urgency
                should_discover = (
                    total_tension > 0.6 or
                    self.brain_cycles % 20 == 0 or
                    self.current_strategic_pattern is None or
                    self._current_prediction_confidence < 0.3
                )
                
                if should_discover:
                    # Start background pattern discovery
                    def on_pattern_discovered(pattern):
                        if pattern and pattern.score > 0:
                            self.current_strategic_pattern = pattern
                            if not self.quiet_mode:
                                print(f"🧠 New strategic pattern discovered (score: {pattern.score:.2f})")
                                print(f"   Tensions: {', '.join(f'{k}={v:.2f}' for k,v in current_tensions.items() if k != 'total')}")
                                print(f"   Pattern persistence: {pattern.persistence:.0f} cycles")
                    
                    self.strategic_planner.discover_async(
                        self.unified_field,
                        self._last_reward_signal,
                        callback=on_pattern_discovered
                    )
            
            # 8. Evolve field
            self._evolve_field()
            
            # Generate prediction for next cycle
            # Simple approach: current field + expected evolution
            self._predicted_field = self.unified_field.clone()
            # Apply expected decay and diffusion
            self._predicted_field *= self.modulation.get('decay_rate', 0.995)
            
            # Update topology regions for sensory prediction
            self.topology_region_system.update_sensory_predictions(
                sensory_dim=len(sensory_input) - 1,  # Exclude reward
                recent_sensory=self.recent_sensory
            )
            
            # Get predictive regions
            topology_regions = self.topology_region_system.get_predictive_regions()
            
            # Predict sensory input using the predictive field system
            # This is where the brain reveals what it expects to sense
            prediction = self.predictive_field.generate_sensory_prediction(
                field=self.unified_field,
                topology_regions=topology_regions,
                recent_sensory=self.recent_sensory
            )
            
            # Store predictions for next cycle
            self._predicted_sensory = prediction.values
            self._prediction_confidence_per_sensor = prediction.confidence
            self._temporal_basis = prediction.temporal_basis
            
            # Update topology regions with prediction results (will happen next cycle)
            self._last_activated_regions = topology_regions
            
            # Update recent sensory history for momentum prediction (exclude reward if present)
            if len(sensory_input) > self.sensory_dim:
                # Has reward, exclude it
                self.recent_sensory.append(sensory_input[:-1])
            else:
                # No reward, use as-is
                self.recent_sensory.append(sensory_input)
            
            # Phase 5: Update active vision learning if glimpse was processed
            if self.use_active_vision and glimpse_data is not None:
                # Compute uncertainty after glimpse processing
                uncertainty_after = self._compute_current_uncertainty()
                
                # Learn the value of this glimpse
                self.active_vision.process_attention_return(
                    attention_data=glimpse_data,
                    uncertainty_before=uncertainty_before,
                    uncertainty_after=uncertainty_after
                )
            
            # 8. Generate motor action (moved before topology detection to pass it along)
            motor_output = self._generate_motor_action()
            
            # 9. Detect and update topology regions
            # Only run every 5 cycles for performance
            if self.brain_cycles % 5 == 0:
                activated_regions = self.topology_region_system.detect_topology_regions(
                    self.unified_field,
                    current_patterns=self.pattern_system.extract_patterns(self.unified_field, n_patterns=5),
                    motor_action=motor_output  # Pass motor action for behavioral tracking
                )
            else:
                activated_regions = self._last_activated_regions
            
            # 10. Echo motor actions back into field (behavioral self-awareness)
            self._echo_motor_to_field(motor_output)
            
            # 11. Update state
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
            # Enhanced error logging for all errors
            error_msg = str(e)
            import traceback
            print(f"\n{'='*60}")
            print(f"ERROR at brain cycle {self.brain_cycles}")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            print("\nFull stack trace:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            logger.error(f"Unexpected error in brain cycle {self.brain_cycles}: {e}")
            # Return safe defaults
            safe_motors = [0.0] * (self.motor_cortex.motor_dim - 1)
            safe_state = {'cycle': self.brain_cycles, 'error': str(e)}
            return safe_motors, safe_state
    
    def _create_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """Create field experience from sensory input."""
        # Simple mapping - just convert to tensor
        raw_input = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        # Extract reward if present (only if input has more elements than sensory_dim)
        # This allows robots to either:
        # 1. Send sensory_dim values (no reward)
        # 2. Send sensory_dim + 1 values (with reward as last element)
        reward = sensory_input[-1] if len(sensory_input) > self.sensory_dim else 0.0
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
            
            # Phase 2 enhancement: Use error-driven learning rate if available
            if hasattr(self.field_dynamics, '_error_modulation') and self.field_dynamics._error_modulation:
                # Prediction errors directly modulate learning rate
                error_learning_rate = self.field_dynamics._error_modulation.get('learning_rate', 0.1)
                scaled_intensity *= error_learning_rate / 0.1  # Normalize by base rate
            else:
                # Fallback to surprise-based modulation
                min_imprint = 0.1  # Always learn a little
                scaled_intensity *= (min_imprint + (1.0 - min_imprint) * surprise_factor)
            
            # Find emergent location for this sensory pattern
            reward = experience.raw_input_stream[-1].item() if experience.raw_input_stream.shape[0] > self.sensory_dim else 0.0
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
        
        # 4. Apply hierarchical prediction updates if available
        hierarchical_update = self.predictive_field.get_pending_hierarchical_update()
        if hierarchical_update is not None:
            self.unified_field += hierarchical_update
        
        # 5. Apply strategic pattern if present
        if self.use_strategic_planning and self.current_strategic_pattern is not None:
            # Install pattern in memory channels (32-47) with gentle influence
            pattern = self.current_strategic_pattern.pattern
            current_pattern = self.unified_field[:, :, :, 32:48]
            
            # Blend pattern with existing field (persistent but not overwhelming)
            # Increased refresh from 0.05 to 0.15 for more active patterns
            self.unified_field[:, :, :, 32:48] = (
                current_pattern * 0.7 +  # Still persistent but more dynamic
                pattern * 0.3  # Stronger active refresh
            )
            
            # Pattern creates gradients that influence other channels
            # Increased influence from 0.02 to 0.1 for stronger behavioral effect
            pattern_energy = self.unified_field[:, :, :, 32:48].mean(dim=-1, keepdim=True)
            gradient_influence = torch.tanh(pattern_energy) * 0.2
            self.unified_field[:, :, :, :32] += gradient_influence.expand(-1, -1, -1, 32)
        
        # 6. Log state
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
        """Generate motor action from field gradients and strategic patterns."""
        exploration_drive = self.modulation.get('exploration_drive', 0.5)
        
        # Extract motor tendencies from field gradients
        # Strategic patterns in channels 32-47 create gradients that influence behavior
        motor_tendencies = self._extract_motor_tendencies_from_field()
        
        # Add exploration noise based on natural drive
        if exploration_drive > 0:
            noise = torch.randn(self.motor_dim, device=self.device) * exploration_drive * 0.3
            motor_tendencies = motor_tendencies + noise
        
        # Apply activation function and scale
        motor_commands = torch.tanh(motor_tendencies)
        
        # Store for learning
        self._last_action = motor_commands
        
        return motor_commands.tolist()
    
    def _motor_to_spatial_pattern(self, motor_action: List[float]) -> torch.Tensor:
        """Convert motor action to spatial pattern for behavioral self-awareness."""
        # Create a spatial pattern that encodes the motor action
        pattern = torch.zeros(self.spatial_resolution, self.spatial_resolution, 
                            self.spatial_resolution, device=self.device)
        
        if len(motor_action) >= 2:
            # Encode forward/backward in depth gradient
            forward = motor_action[0]
            pattern[:, :, :] += forward * torch.linspace(-1, 1, self.spatial_resolution, 
                                                        device=self.device).unsqueeze(0).unsqueeze(0)
            
            # Encode left/right in width gradient
            turn = motor_action[1] if len(motor_action) > 1 else 0
            pattern += turn * torch.linspace(-1, 1, self.spatial_resolution, 
                                           device=self.device).unsqueeze(0).unsqueeze(2)
        
        return pattern
    
    def _echo_motor_to_field(self, motor_action: List[float]):
        """Echo motor actions into field channels 62-63 for behavioral awareness."""
        # Decay previous motor echo
        self.unified_field[:, :, :, 62] *= 0.95
        self.unified_field[:, :, :, 63] *= 0.95
        
        # Create spatial pattern from motor action
        motor_pattern = self._motor_to_spatial_pattern(motor_action)
        
        # Add current motor pattern to channel 62
        self.unified_field[:, :, :, 62] += motor_pattern * 0.3
        
        # Store motor magnitude in channel 63 for monotony detection
        motor_magnitude = sum(abs(m) for m in motor_action) / len(motor_action)
        self.unified_field[:, :, :, 63] += motor_magnitude * 0.2
    
    def _extract_motor_tendencies_from_field(self) -> torch.Tensor:
        """Extract motor tendencies from field gradients created by strategic patterns."""
        # Get field activation in content channels
        content_field = self.unified_field[:, :, :, :32]
        
        # Compute spatial gradients (movement tendencies)
        gradients = []
        
        # X-axis gradient (forward/backward)
        if content_field.shape[0] > 1:
            # Positive gradient means higher activation ahead, so move forward
            x_grad = (content_field[-1, :, :].mean() - content_field[0, :, :].mean()).item()
        else:
            x_grad = 0.0
            
        # Y-axis gradient (left/right)
        if content_field.shape[1] > 1:
            # Positive gradient means higher activation to the right
            y_grad = (content_field[:, -1, :].mean() - content_field[:, 0, :].mean()).item()
        else:
            y_grad = 0.0
            
        # Z-axis gradient (up/down - less important for ground robots)
        if content_field.shape[2] > 1:
            z_grad = (content_field[:, :, -1].mean() - content_field[:, :, 0].mean()).item()
        else:
            z_grad = 0.0
        
        # Pattern influence strength
        if self.current_strategic_pattern is not None:
            pattern_field = self.unified_field[:, :, :, 32:48]
            pattern_strength = pattern_field.abs().mean().item()
        else:
            pattern_strength = 0.0
        
        # Create motor vector based on field gradients
        motor_tendencies = torch.zeros(self.motor_dim, device=self.device)
        
        # Map gradients to motor dimensions (robot-specific)
        # Amplify gradients for better responsiveness
        gradient_amplification = 5.0  # Increased from 2.0
        if self.motor_dim >= 2:
            motor_tendencies[0] = x_grad * gradient_amplification  # Forward/backward
            motor_tendencies[1] = y_grad * gradient_amplification  # Turn left/right
            
        if self.motor_dim >= 3:
            motor_tendencies[2] = pattern_strength  # Speed modulation
            
        if self.motor_dim >= 4:
            motor_tendencies[3] = z_grad * 0.5  # Vertical component (if applicable)
        
        # Additional motor dimensions get small random activations
        for i in range(4, self.motor_dim):
            motor_tendencies[i] = torch.randn(1, device=self.device).item() * 0.1
        
        return motor_tendencies
    
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
            'predictive_phases': {
                'phase_3_hierarchical': hasattr(self.predictive_field, 'use_hierarchical_timescales') and self.predictive_field.use_hierarchical_timescales,
                'strategic_planning': self.use_strategic_planning,
                'phase_5_active_vision': self.use_active_vision,
                'enabled_count': sum([
                    hasattr(self.predictive_field, 'use_hierarchical_timescales') and self.predictive_field.use_hierarchical_timescales,
                    self.use_strategic_planning,
                    self.use_active_vision
                ])
            },
            'sensory_organization': self.sensory_mapping.get_statistics(),
            'temporal_basis': getattr(self, '_temporal_basis', 'immediate'),
            'timestamp': time.time()
        }
        
        
        # Add active sensing statistics if enabled
        if self.use_active_vision:
            brain_state['active_vision'] = self.active_vision.get_attention_statistics()
        
        
        return brain_state
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        elements = 1
        for dim in self.tensor_shape:
            elements *= dim
        return (elements * 4) / (1024 * 1024)
    
    def _compute_current_uncertainty(self) -> float:
        """Compute current overall uncertainty from topology regions."""
        if not hasattr(self, '_last_activated_regions') or not self._last_activated_regions:
            return 0.5  # Default uncertainty
        
        # Average uncertainty across all active regions
        uncertainties = []
        for region in self._last_activated_regions:
            if hasattr(region, 'prediction_confidence'):
                uncertainties.append(1.0 - region.prediction_confidence)
        
        if uncertainties:
            return sum(uncertainties) / len(uncertainties)
        else:
            return 0.5
    
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