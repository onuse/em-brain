#!/usr/bin/env python3
"""
Field-Native Brain: Complete Unified Multi-Dimensional Field Intelligence

This is the revolutionary field-native brain implementation that abandons discrete
streams in favor of unified multi-dimensional field dynamics. All intelligence
emerges from field topology optimization and constraint satisfaction.

Architecture:
- Unified 37D field organized by dynamics families, not sensory modalities
- All concepts emerge from field topology (no discrete pattern storage)
- All actions emerge from field gradients (no discrete action generation)
- All memory emerges from field persistence (no discrete pattern recall)
- All learning emerges from field evolution (no discrete updates)

This represents a fundamental paradigm shift from discrete AI to continuous
field intelligence - potentially the most honest implementation of how
biological intelligence actually works.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque, defaultdict
import threading
import queue

# Import our field dynamics foundation
try:
    from .dynamics.constraint_field_nd import ConstraintFieldND, FieldConstraintND
    from .dynamics.constraint_field_dynamics import FieldConstraintType
    from .dynamics.temporal_field_dynamics import TemporalExperience, TemporalImprint
    from .optimized_gradients import create_optimized_gradient_calculator
except ImportError:
    from brains.field.dynamics.constraint_field_nd import ConstraintFieldND, FieldConstraintND
    from brains.field.dynamics.constraint_field_dynamics import FieldConstraintType
    from brains.field.dynamics.temporal_field_dynamics import TemporalExperience, TemporalImprint
    from brains.field.optimized_gradients import create_optimized_gradient_calculator


class FieldDynamicsFamily(Enum):
    """Families of field dynamics that organize dimensions."""
    SPATIAL = "spatial"              # Position, orientation, scale
    OSCILLATORY = "oscillatory"     # Frequencies, rhythms, periods
    FLOW = "flow"                   # Gradients, momentum, direction
    TOPOLOGY = "topology"           # Stable configurations, boundaries
    ENERGY = "energy"               # Intensity, activation, depletion
    COUPLING = "coupling"           # Relationships, correlations, binding
    EMERGENCE = "emergence"         # Novelty, creativity, phase transitions


@dataclass
class FieldDimension:
    """Definition of a single field dimension."""
    name: str
    family: FieldDynamicsFamily
    index: int  # Position in unified field
    range_min: float = -1.0
    range_max: float = 1.0
    default_value: float = 0.0
    description: str = ""


@dataclass
class UnifiedFieldExperience:
    """A complete experience in unified field space."""
    timestamp: float
    raw_sensory_input: torch.Tensor  # Original robot sensors
    field_coordinates: torch.Tensor  # Mapped to 37D field space
    field_intensity: float = 1.0
    experience_id: Optional[str] = None


@dataclass
class FieldNativeAction:
    """An action that emerges from field gradient following."""
    timestamp: float
    field_gradients: torch.Tensor  # Gradients in field space
    motor_commands: torch.Tensor   # Translated to robot actuators
    action_confidence: float = 1.0
    gradient_strength: float = 0.0


class UnifiedFieldBrain:
    """
    The Complete Field-Native Brain
    
    This implements intelligence as unified multi-dimensional field dynamics
    where all cognitive processes emerge from field topology optimization.
    
    Key Principles:
    1. No discrete patterns - only stable field topology regions
    2. No separate streams - unified field with dimension families
    3. No explicit memory - field imprint persistence
    4. No action generation - field gradient following
    5. No learning algorithms - field evolution dynamics
    """
    
    def __init__(self, 
                 spatial_resolution: Optional[int] = None,
                 temporal_window: float = 10.0,
                 field_evolution_rate: float = 0.1,
                 constraint_discovery_rate: float = 0.15,
                 quiet_mode: bool = False):
        
        # If resolution not specified, we'll use hardware-adapted value
        self._requested_resolution = spatial_resolution
        self.temporal_window = temporal_window
        self.field_evolution_rate = field_evolution_rate
        self.constraint_discovery_rate = constraint_discovery_rate
        self.quiet_mode = quiet_mode
        
        # Field evolution parameters
        self.field_decay_rate = 0.995  # Slight decay per cycle
        self.field_diffusion_rate = 0.05  # Diffusion strength
        self.topology_stability_threshold = 0.01  # Topology region threshold
        
        # DEVICE DETECTION: Use adaptive configuration system
        try:
            from ...adaptive_configuration import get_device
            hardware_device = get_device()
        except ImportError:
            # Fallback for direct import
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
            from adaptive_configuration import get_device
            hardware_device = get_device()
        
        # Calculate field dimensions early for MPS compatibility check
        field_dimensions = self._initialize_field_dimensions()
        total_dimensions = len(field_dimensions)
        
        # Get hardware profile for adaptive field sizing
        hw_profile = None
        self.hw_profile = None
        try:
            # Try relative import first
            from ...adaptive_configuration import get_configuration
            config = get_configuration()
            hw_profile = config
            self.hw_profile = config
        except:
            try:
                # Fallback to direct import
                from adaptive_configuration import get_configuration
                config = get_configuration()
                hw_profile = config
                self.hw_profile = config
            except Exception as e:
                if not quiet_mode:
                    print(f"⚠️  Adaptive configuration not available: {e}")
        
        # Use hardware-recommended resolution if not explicitly set
        if self._requested_resolution is None and hw_profile is not None:
            # Use spatial resolution from configuration
            self.spatial_resolution = hw_profile.spatial_resolution or 4
            if not quiet_mode:
                print(f"📐 Using configured resolution: {self.spatial_resolution}³")
        else:
            self.spatial_resolution = self._requested_resolution or 4  # Default to 4³
            
        # Handle MPS tensor dimension limitation  
        if hardware_device.type == 'mps' and total_dimensions > 16:
            self.device = torch.device('cpu')
            if not quiet_mode:
                print(f"💻 Using CPU processing (MPS limited to 16D, field needs {total_dimensions}D)")
        else:
            self.device = hardware_device
            if not quiet_mode and hardware_device.type != 'cpu':
                print(f"🚀 Using {hardware_device.type.upper()} acceleration")
        
        # Initialize field dimension architecture (already calculated above)
        self.field_dimensions = field_dimensions
        self.total_dimensions = total_dimensions
        
        # Create unified multi-dimensional field - ADAPTIVE based on hardware
        field_shape = self._calculate_adaptive_field_shape(
            self.spatial_resolution, hw_profile, quiet_mode
        )
        
        # Core unified field - this replaces ALL discrete structures
        # PERFORMANCE FIX: Explicit device specification for optimal performance
        self.unified_field = torch.zeros(field_shape, dtype=torch.float32, device=self.device)
        # Initialize with baseline activation
        self.unified_field.fill_(0.01)
        
        # Field dynamics systems - now using N-dimensional constraints
        self.constraint_field = ConstraintFieldND(
            field_shape=field_shape,
            dimension_names=[dim.name for dim in self.field_dimensions],
            constraint_discovery_rate=constraint_discovery_rate * 3,  # Increase discovery rate for better constraint learning
            constraint_enforcement_strength=0.3,
            max_constraints=50,
            device=self.device,
            quiet_mode=quiet_mode
        )
        
        # Optimized gradient calculator
        self.gradient_calculator = create_optimized_gradient_calculator(
            activation_threshold=0.01,
            cache_enabled=True,
            sparse_computation=True
        )
        
        # Field evolution parameters - FIXED: Improved for stronger gradients
        self.field_decay_rate = 0.999        # Much slower decay (was 0.995 - too aggressive)
        self.field_diffusion_rate = 0.05     # More diffusion for gradient propagation (was 0.02)
        self.gradient_following_strength = 0.5  # Increased for better exploration/exploitation balance
        self.topology_stability_threshold = 0.02  # Lower threshold for better memory formation
        
        # Field state tracking
        self.field_experiences: List[UnifiedFieldExperience] = []
        self.field_actions: List[FieldNativeAction] = []
        self.topology_regions: Dict[str, Dict] = {}
        self.gradient_flows: Dict[str, torch.Tensor] = {}
        
        # Motor smoothing for field testing
        self.previous_motor_commands = torch.zeros(4, device=self.device)
        self.motor_smoothing_factor = 0.3  # 0 = no smoothing, 1 = full smoothing
        
        # Robot interface
        self.expected_sensory_dim = 24  # From robot sensors (24 base dimensions)
        self.expected_motor_dim = 4     # To robot actuators
        
        # Performance tracking
        self.brain_cycles = 0
        self.field_evolution_cycles = 0
        self.topology_discoveries = 0
        self.gradient_actions = 0
        
        # Initialize robot position tracking
        self._last_imprint_indices = None
        
        # Maintenance tracking
        self.last_maintenance_cycle = 0
        self.maintenance_interval = 100  # Run maintenance every 100 cycles
        self.field_energy_dissipation_rate = 0.90  # Dampen field energy by 10% during maintenance
        
        # PREDICTION IMPROVEMENT ADDICTION: Track confidence for intrinsic reward
        self._prediction_confidence_history = []
        self._improvement_rate_history = []
        self._current_prediction_confidence = 0.5
        
        # Maintenance thread setup
        self._maintenance_queue = queue.Queue(maxsize=1)  # Only need latest signal
        self._maintenance_thread = None
        self._shutdown_maintenance = threading.Event()
        self._start_maintenance_thread()
        
        if not quiet_mode:
            print(f"🌊 UnifiedFieldBrain initialized")
            print(f"   Field dimensions: {self.total_dimensions}D unified space")
            print(f"   Spatial resolution: {spatial_resolution}³")
            print(f"   Temporal window: {temporal_window}s")
            print(f"   Field families: {len(FieldDynamicsFamily)} dynamics types")
            self._print_dimension_summary()
    
    def _calculate_adaptive_field_shape(self, spatial_resolution: int, 
                                      hw_profile: Optional[Any], 
                                      quiet_mode: bool) -> List[int]:
        """
        Calculate field dimensions adapted to hardware capabilities.
        
        Strategy:
        - Low-end hardware: Minimal dimensions for basic intelligence
        - Mid-range: Balanced dimensions  
        - High-end: Rich dimensions for complex intelligence
        """
        # Default balanced shape
        default_shape = [
            spatial_resolution,      # X position
            spatial_resolution,      # Y position  
            spatial_resolution,      # Z position
            10,                     # Scale levels
            15,                     # Time steps
            3,                      # Oscillatory (6D → 3)
            3,                      # Flow (8D → 3)
            2,                      # Topology (6D → 2)
            2,                      # Energy (4D → 2)
            2,                      # Coupling (5D → 2)
            2,                      # Emergence (3D → 2)
        ]
        
        if hw_profile is None:
            if not quiet_mode:
                print("🔧 No hardware profile, using balanced field dimensions")
            return default_shape
            
        # Determine hardware tier based on capabilities
        cpu_cores = hw_profile.cpu_cores
        memory_gb = hw_profile.system_memory_gb
        gpu_available = hw_profile.device_type in ['cuda', 'mps']
        gpu_memory_gb = hw_profile.gpu_memory_gb or 0
        
        # Calculate performance score
        cpu_score = min(cpu_cores / 8.0, 2.0)  # Normalize to 8 cores = 1.0
        memory_score = min(memory_gb / 16.0, 2.0)  # Normalize to 16GB = 1.0
        gpu_score = 2.0 if gpu_available else 1.0
        
        if gpu_memory_gb > 8:
            gpu_score = 3.0  # High-end GPU
        elif gpu_memory_gb > 4:
            gpu_score = 2.5  # Mid-range GPU
            
        total_score = cpu_score * memory_score * gpu_score
        
        # Define dimension configurations based on hardware tier
        if total_score >= 4.0:  # High-end hardware
            field_shape = [
                spatial_resolution,      # X position
                spatial_resolution,      # Y position  
                spatial_resolution,      # Z position
                10,                     # Scale levels
                15,                     # Time steps
                6,                      # Oscillatory (rich)
                6,                      # Flow (rich)
                4,                      # Topology (good)
                3,                      # Energy (good)
                4,                      # Coupling (rich)
                3,                      # Emergence (good)
            ]
            tier = "high-end"
        elif total_score >= 1.5:  # Mid-range hardware
            field_shape = [
                spatial_resolution,      # X position
                spatial_resolution,      # Y position  
                spatial_resolution,      # Z position
                10,                     # Scale levels
                15,                     # Time steps
                4,                      # Oscillatory (balanced)
                4,                      # Flow (balanced)
                3,                      # Topology (adequate)
                2,                      # Energy (basic)
                3,                      # Coupling (adequate)
                2,                      # Emergence (basic)
            ]
            tier = "mid-range"
        else:  # Low-end hardware
            field_shape = [
                min(spatial_resolution, 12),  # Smaller spatial
                min(spatial_resolution, 12),
                min(spatial_resolution, 12),
                8,                      # Fewer scale levels
                10,                     # Fewer time steps
                2,                      # Oscillatory (minimal)
                2,                      # Flow (minimal)
                2,                      # Topology (minimal)
                1,                      # Energy (binary)
                2,                      # Coupling (minimal)
                1,                      # Emergence (binary)
            ]
            tier = "low-end"
            
        # Calculate memory usage
        total_elements = 1
        for dim in field_shape:
            total_elements *= dim
        memory_mb = (total_elements * 4) / (1024 * 1024)
        
        if not quiet_mode:
            print(f"🔧 Hardware tier: {tier} (score: {total_score:.1f})")
            print(f"   CPU: {cpu_cores} cores, RAM: {memory_gb:.1f}GB")
            if gpu_available:
                print(f"   GPU: {hw_profile.gpu_memory_gb:.1f}GB")
            print(f"   Field dimensions: {field_shape}")
            print(f"   Field memory: {memory_mb:.1f}MB ({total_elements:,} elements)")
            
        # Safety check - ensure memory usage is reasonable
        max_allowed_mb = memory_gb * 1024 * 0.1  # Use max 10% of system RAM
        if memory_mb > max_allowed_mb:
            if not quiet_mode:
                print(f"⚠️  Field too large ({memory_mb:.1f}MB), reducing dimensions...")
            # Reduce non-spatial dimensions
            for i in range(5, len(field_shape)):
                field_shape[i] = max(1, field_shape[i] - 1)
            
            # Recalculate memory usage
            new_total_elements = 1
            for dim in field_shape:
                new_total_elements *= dim
            new_memory_mb = (new_total_elements * 4) / (1024 * 1024)
            
            if not quiet_mode:
                print(f"   Reduced to: {field_shape}")
                print(f"   New memory: {new_memory_mb:.1f}MB")
            
        return field_shape
    
    def _initialize_field_dimensions(self) -> List[FieldDimension]:
        """Initialize the unified field dimension architecture."""
        dimensions = []
        index = 0
        
        # SPATIAL DIMENSIONS (3D + scale + time)
        spatial_dims = [
            ("spatial_x", "Robot X position"),
            ("spatial_y", "Robot Y position"),  
            ("spatial_z", "Robot Z position/height"),
            ("scale", "Abstraction level (micro → macro)"),
            ("time", "Temporal position")
        ]
        for name, desc in spatial_dims:
            family = FieldDynamicsFamily.SPATIAL
            dimensions.append(FieldDimension(name, family, index, -1.0, 1.0, 0.0, desc))
            index += 1
        
        # OSCILLATORY DIMENSIONS (6D)
        oscillatory_dims = [
            ("sound_frequency", "Audio frequency patterns"),
            ("light_frequency", "Visual frequency patterns"),
            ("movement_rhythm", "Motor rhythm patterns"),
            ("neural_gamma", "40Hz gamma oscillations"),
            ("neural_theta", "6Hz theta oscillations"),
            ("environmental_rhythm", "Day/night and environmental cycles")
        ]
        for name, desc in oscillatory_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.OSCILLATORY, index, -1.0, 1.0, 0.0, desc))
            index += 1
        
        # FLOW DIMENSIONS (8D)
        flow_dims = [
            ("motion_x", "X-direction motion gradient"),
            ("motion_y", "Y-direction motion gradient"),
            ("attention_gradient", "Attention flow direction"),
            ("curiosity_gradient", "Curiosity exploration gradient"),
            ("temperature_gradient", "Thermal gradient following"),
            ("social_gradient", "Interpersonal flow dynamics"),
            ("exploration_flow", "Exploration momentum"),
            ("pressure_flow", "Physical pressure gradients")
        ]
        for name, desc in flow_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.FLOW, index, -1.0, 1.0, 0.0, desc))
            index += 1
        
        # TOPOLOGY DIMENSIONS (6D)
        topology_dims = [
            ("object_stability", "Stable object recognition"),
            ("concept_stability", "Stable concept topology"),
            ("habit_stability", "Stable behavioral patterns"),
            ("boundary_strength", "Field boundary definition"),
            ("memory_persistence", "Topology persistence strength"),
            ("identity_coherence", "Self-topology coherence")
        ]
        for name, desc in topology_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.TOPOLOGY, index, 0.0, 1.0, 0.0, desc))
            index += 1
        
        # ENERGY DIMENSIONS (4D)
        energy_dims = [
            ("motor_energy", "Available motor energy"),
            ("cognitive_energy", "Available cognitive resources"),
            ("sensory_intensity", "Overall sensory activation"),
            ("emotional_intensity", "Emotional activation level")
        ]
        for name, desc in energy_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.ENERGY, index, 0.0, 1.0, 0.5, desc))
            index += 1
        
        # COUPLING DIMENSIONS (5D)
        coupling_dims = [
            ("sensorimotor_coupling", "Sensor-motor correlation"),
            ("learning_correlation", "Learning association strength"),
            ("recognition_binding", "Pattern recognition binding"),
            ("social_coupling", "Social interaction coupling"),
            ("analogical_coupling", "Analogical reasoning strength")
        ]
        for name, desc in coupling_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.COUPLING, index, -1.0, 1.0, 0.0, desc))
            index += 1
        
        # EMERGENCE DIMENSIONS (3D)
        emergence_dims = [
            ("novelty_potential", "Potential for novel patterns"),
            ("creativity_space", "Creative combination space"),
            ("problem_solving_phase", "Problem-solving phase state")
        ]
        for name, desc in emergence_dims:
            dimensions.append(FieldDimension(name, FieldDynamicsFamily.EMERGENCE, index, 0.0, 1.0, 0.1, desc))
            index += 1
        
        return dimensions
    
    def _print_dimension_summary(self):
        """Print summary of field dimension organization."""
        family_counts = defaultdict(int)
        for dim in self.field_dimensions:
            family_counts[dim.family] += 1
        
        print(f"   Field dimension families:")
        for family, count in family_counts.items():
            print(f"      {family.value}: {count}D")
    
    def process_robot_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Complete robot brain cycle using unified field dynamics.
        
        This is the core interface: robot sensors → field dynamics → robot actions
        """
        cycle_start = time.perf_counter()
        
        # 1. Convert robot sensory input to unified field experience
        field_experience = self._robot_sensors_to_field_experience(sensory_input)
        
        # 2. PREDICTION: Compare predicted field with actual field BEFORE sensory input
        if hasattr(self, '_predicted_field'):
            # Focus on local spatial region around robot position for prediction accuracy
            center = self.spatial_resolution // 2
            region_slice = slice(center-1, center+2)  # 3x3x3 region
            
            # Extract local regions from predicted vs actual (before sensory input)
            predicted_region = self._predicted_field[region_slice, region_slice, region_slice]
            actual_region = self.unified_field[region_slice, region_slice, region_slice]
            
            # Calculate prediction error
            prediction_error = torch.mean(torch.abs(predicted_region - actual_region)).item()
            
            # Convert error to confidence (lower error = higher confidence)
            # Reduced sensitivity for more meaningful confidence range
            self._current_prediction_confidence = 1.0 / (1.0 + prediction_error * 100.0)
        
        # 3. Apply experience to unified field (this replaces ALL discrete processing)
        self._apply_field_experience(field_experience)
        
        # 4. Save current field state for prediction
        pre_evolution_field = self.unified_field.clone()
        
        # 5. Evolve unified field dynamics (this creates our prediction of next state)
        self._evolve_unified_field()
        
        # 6. Store predicted field state (what we think next sensory input will create)
        self._predicted_field = self.unified_field.clone()
        
        # 6. Generate robot action from field gradients
        field_action = self._field_gradients_to_robot_action()
        
        # 7. Update prediction confidence tracking for addiction system
        self._prediction_confidence_history.append(self._current_prediction_confidence)
        if len(self._prediction_confidence_history) > 20:
            self._prediction_confidence_history = self._prediction_confidence_history[-20:]
        
        # 6. Update brain state
        self.brain_cycles += 1
        cycle_time = (time.perf_counter() - cycle_start) * 1000
        
        # Performance warning for slow cycles
        if cycle_time > 750:
            print(f"⚠️  PERFORMANCE WARNING: Cycle {self.brain_cycles} took {cycle_time:.1f}ms (>750ms threshold)")
            print(f"   Consider reducing spatial resolution or disabling logging/persistence")
        
        # Record performance for adaptive configuration system
        # Note: Performance recording will be handled by the adaptive configuration manager
        
        # 7. Trigger maintenance operations periodically (async, not on hot path)
        if self.brain_cycles % self.maintenance_interval == 0:
            self._trigger_maintenance()
        
        # 8. Extract brain state info
        brain_state = self._get_field_brain_state(cycle_time)
        
        if not self.quiet_mode and self.brain_cycles % 100 == 0:
            print(f"🌊 Field Brain Cycle {self.brain_cycles}: "
                  f"{cycle_time:.2f}ms, "
                  f"topology_regions={len(self.topology_regions)}, "
                  f"field_max={torch.max(self.unified_field).item():.3f}")
        
        return field_action.motor_commands.tolist(), brain_state
    
    def _robot_sensors_to_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """
        Convert robot sensor data to unified field experience.
        
        This is the critical translation from discrete sensors to continuous field space.
        """
        # Ensure sensory input is correct size
        if len(sensory_input) != self.expected_sensory_dim:
            # Pad or truncate to expected size
            padded_input = sensory_input[:self.expected_sensory_dim]
            while len(padded_input) < self.expected_sensory_dim:
                padded_input.append(0.0)
            sensory_input = padded_input
        
        # Validate and sanitize input - replace NaN/inf with safe defaults
        sanitized_input = []
        for i, value in enumerate(sensory_input):
            if math.isnan(value) or math.isinf(value):
                # Use safe default based on sensor type
                if i < 3:  # Distance sensors - use far distance
                    sanitized_input.append(0.1)
                else:  # Other sensors - use neutral
                    sanitized_input.append(0.0)
            else:
                # Clamp to reasonable range
                sanitized_input.append(max(-10.0, min(10.0, value)))
        
        sensory_input = sanitized_input
        raw_input = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        # Map robot sensors to field dimensions - ENHANCED to use all dimensions
        field_coords = torch.zeros(self.total_dimensions, device=self.device)
        
        # Convert to tensor for easier manipulation
        sensor_tensor = torch.tensor(sensory_input, device=self.device)
        
        # SPATIAL MAPPING (0-2) - TUNED: Enhanced obstacle detection
        # Nonlinear response for better obstacle avoidance
        # INVERTED: High sensor values (close obstacles) map to NEGATIVE field coords
        for i in range(3):
            raw = sensory_input[i]
            if raw > 0.7:  # Close obstacle - strong repulsion
                field_coords[i] = -(raw - 0.1) * 2  # Strong negative response
            elif raw > 0.5:  # Medium distance - moderate repulsion
                field_coords[i] = -(raw - 0.1)      # Moderate negative response
            else:  # Far distance - neutral to positive
                field_coords[i] = (0.5 - raw) * 0.5  # Slight positive for free space
        
        # SCALE MAPPING (3) - derive from sensor variance
        sensor_variance = torch.var(sensor_tensor[:6])
        field_coords[3] = torch.tanh(sensor_variance * 2)
        
        # TIME MAPPING (4) - temporal dynamics
        field_coords[4] = torch.sin(sensor_tensor[0] * math.pi)
        
        # OSCILLATORY MAPPING (5-10) - frequencies, rhythms
        for i in range(6):
            sensor_idx = 3 + i  # Start from 4th sensor
            if sensor_idx < len(sensory_input):
                # Create rich oscillatory patterns
                freq = (i + 1) * 0.5
                phase = sensory_input[sensor_idx] * 2 * math.pi
                field_coords[5 + i] = math.sin(phase + freq * self.brain_cycles * 0.01)
            else:
                # Generate from available sensors
                field_coords[5 + i] = math.cos(sensor_tensor[i % len(sensory_input)] * math.pi * (i + 1))
        
        # FLOW MAPPING (11-18) - gradients, motion
        for i in range(8):
            sensor_idx = 9 + i
            if sensor_idx < len(sensory_input) and sensor_idx > 0:
                # Rich gradient calculation
                gradient = sensory_input[sensor_idx] - sensory_input[sensor_idx - 1]
                field_coords[11 + i] = gradient * 5 + 0.1 * math.sin(self.brain_cycles * 0.1 + i)
            else:
                # Derive from spatial changes
                field_coords[11 + i] = (sensor_tensor[(i + 1) % len(sensory_input)] - 
                                       sensor_tensor[i % len(sensory_input)])
        
        # TOPOLOGY MAPPING (19-24) - stable patterns
        for i in range(6):
            sensor_idx = 17 + i
            if sensor_idx < len(sensory_input):
                # Threshold-based topology with hysteresis
                threshold = 0.5 + 0.1 * math.sin(i)
                field_coords[19 + i] = 1.0 if sensory_input[sensor_idx] > threshold else -1.0
            else:
                # Pattern from sensor combinations
                field_coords[19 + i] = torch.sign(sensor_tensor[i % len(sensory_input)] - 0.5)
        
        # ENERGY MAPPING (25-28) - intensity levels
        for i in range(4):
            if i < len(sensory_input):
                # Energy as transformed magnitude
                energy = abs(sensory_input[i] - 0.5) * 2
                field_coords[25 + i] = energy * (1 + 0.2 * math.sin(self.brain_cycles * 0.05))
            else:
                # Derive from sensor statistics
                std_val = torch.std(sensor_tensor).item()
                # Handle single value or constant array (std returns nan)
                if torch.isnan(torch.tensor(std_val)) or sensor_tensor.numel() <= 1:
                    std_val = 0.1  # Default variance
                field_coords[25 + i] = std_val * (i + 1)
        
        # COUPLING MAPPING (29-33) - correlations
        for i in range(5):
            if i + 1 < len(sensory_input):
                # Richer correlations
                corr = sensory_input[i] * sensory_input[i + 1]
                field_coords[29 + i] = torch.tanh(torch.tensor(corr * 2 - 1.0))
            else:
                # Cross-sensor coupling
                idx1, idx2 = i % len(sensory_input), (i + 3) % len(sensory_input)
                field_coords[29 + i] = sensor_tensor[idx1] * sensor_tensor[idx2] * 2 - 1
        
        # EMERGENCE MAPPING (34-36) - creative combinations
        # Nonlinear combinations of sensors
        # Only set these if we have enough dimensions
        if self.total_dimensions > 34:
            field_coords[34] = torch.tanh(torch.sum(sensor_tensor[:3]) - 1.5)
        if self.total_dimensions > 35:
            field_coords[35] = torch.tanh(torch.prod(sensor_tensor[:3]) * 10)
        if self.total_dimensions > 36:
            field_coords[36] = torch.tanh(torch.max(sensor_tensor) - torch.min(sensor_tensor))
        
        # Calculate novelty from field distance to previous experiences
        novelty = self._calculate_experience_novelty(field_coords)
        # Novelty modulates the last emergence dimension
        if self.total_dimensions > 36:
            field_coords[36] = field_coords[36] * 0.7 + novelty * 0.3
        
        # REWARD INTEGRATION - Process 25th dimension as reward signal
        field_intensity = 0.3 + 0.2 * torch.norm(field_coords).item() / math.sqrt(self.total_dimensions)  # Default
        
        if len(sensory_input) >= 25:
            reward = sensory_input[24]  # 25th dimension (0-indexed)
            if reward != 0:
                # Map reward to multiple field dimensions for value learning
                # 1. Strengthen energy dimension (emotional_intensity index 28)
                if self.total_dimensions > 28:
                    field_coords[28] = torch.clamp(field_coords[28] + reward * 0.5, -2.0, 2.0)
                # 2. Modulate coupling for reward-state binding (index 30)
                if self.total_dimensions > 30:
                    field_coords[30] = torch.clamp(field_coords[30] + reward * 0.3, -2.0, 2.0)
                # 3. Increase field intensity for stronger memory formation
                if reward > 0:
                    field_intensity = 0.5 + abs(reward) * 0.5  # Positive rewards create stronger memories
                else:
                    field_intensity = 0.3  # Negative rewards create aversive memories
                # Store reward-modified intensity
                self._last_reward = reward
            else:
                self._last_reward = 0.0
        else:
            self._last_reward = 0.0
        
        return UnifiedFieldExperience(
            timestamp=time.time(),
            raw_sensory_input=raw_input,
            field_coordinates=field_coords,
            field_intensity=field_intensity,
            experience_id=f"experience_{self.brain_cycles}"
        )
    
    def _apply_field_experience(self, experience: UnifiedFieldExperience):
        """
        Apply experience to unified field - UPDATED for new 11D structure.
        """
        # Normalize pattern for memory encoding
        # This ensures loud voices don't create stronger memories than whispers
        normalized_coords = self._normalize_pattern_for_memory(experience.field_coordinates)
        
        # Create normalized experience for memory operations
        memory_experience = UnifiedFieldExperience(
            timestamp=experience.timestamp,
            raw_sensory_input=experience.raw_sensory_input,
            field_coordinates=normalized_coords,
            field_intensity=experience.field_intensity,
            experience_id=experience.experience_id
        )
        
        # First, check for resonance with existing topology regions
        self._activate_resonant_memories(memory_experience)
        
        # Map all 37 field coordinates to 11D field indices
        coords = experience.field_coordinates
        
        # Map continuous coordinates to discrete field indices
        indices = []
        
        # Spatial dimensions (0-2)
        for i in range(3):
            coord = torch.clamp(coords[i], -2.0, 2.0)  # Handle amplified range
            idx = int((coord + 2.0) * 0.25 * (self.spatial_resolution - 1))
            indices.append(max(0, min(self.spatial_resolution - 1, idx)))
        
        # Map remaining dimensions based on actual field shape
        # Scale dimension (3)
        scale_size = self.unified_field.shape[3]
        scale_idx = int((torch.clamp(coords[3], -1.0, 1.0) + 1) * 0.5 * (scale_size - 1))
        indices.append(max(0, min(scale_size - 1, scale_idx)))
        
        # Time dimension (4)
        time_size = self.unified_field.shape[4]
        time_idx = int((torch.clamp(coords[4], -1.0, 1.0) + 1) * 0.5 * (time_size - 1))
        indices.append(max(0, min(time_size - 1, time_idx)))
        
        # Oscillatory dimensions (5-10) mapped to field dimension 5
        oscillatory_size = self.unified_field.shape[5]
        oscillatory_idx = 0
        for i in range(5, 11):
            oscillatory_idx += coords[i].item() * (2 ** (i - 5))
        oscillatory_idx = int((oscillatory_idx + 32) * (1.0 / 64.0) * oscillatory_size) % oscillatory_size
        indices.append(oscillatory_idx)
        
        # Flow dimensions (11-18) mapped to field dimension 6
        flow_size = self.unified_field.shape[6]
        flow_idx = 0
        for i in range(11, 19):
            flow_idx += coords[i].item() * (2 ** (i - 11))
        flow_idx = int((flow_idx + 128) * (1.0 / 256.0) * flow_size) % flow_size
        indices.append(flow_idx)
        
        # Topology dimensions (19-24) mapped to field dimension 7
        topology_size = self.unified_field.shape[7]
        topology_idx = int(torch.sum(coords[19:25]).item() + 3) % topology_size
        indices.append(topology_idx)
        
        # Energy dimensions (25-28) mapped to field dimension 8
        energy_size = self.unified_field.shape[8]
        energy_idx = int(torch.sum(torch.abs(coords[25:29])).item() * 2) % energy_size
        indices.append(energy_idx)
        
        # Coupling dimensions (29-33) mapped to field dimension 9
        coupling_size = self.unified_field.shape[9]
        coupling_idx = int(torch.sum(coords[29:34]).item() + 2.5) % coupling_size
        indices.append(coupling_idx)
        
        # Emergence dimensions (34-36) mapped to field dimension 10
        emergence_size = self.unified_field.shape[10]
        emergence_idx = int(torch.sum(coords[34:37]).item() + 1.5) % emergence_size
        indices.append(emergence_idx)
        
        # Apply experience as field imprint
        self._last_imprint_indices = indices  # Store for topology discovery
        
        if not self.quiet_mode:
            print(f"   🎯 Imprinting at spatial indices: {indices[:3]}, full indices: {indices}")
            
        self._apply_multidimensional_imprint_new(
            tuple(indices),
            experience.field_intensity
        )
        
        # Store experience
        self.field_experiences.append(experience)
        
        # Discover topology regions and constraints using normalized pattern
        self._discover_field_topology(memory_experience)
    
    def _apply_multidimensional_imprint_new(self, indices: Tuple[int, ...], intensity: float):
        """Apply field imprint for new 11D structure."""
        # Simply add intensity at the specified location
        # The 11D structure allows for much richer patterns
        try:
            old_val = self.unified_field[indices].item()
            self.unified_field[indices] += intensity
            new_val = self.unified_field[indices].item()
            
            if not self.quiet_mode:
                print(f"   🎯 Imprint result: {old_val:.4f} -> {new_val:.4f} at {indices[:3]}")
            
            # Apply small Gaussian blur around the point
            for dim in range(len(indices)):
                for offset in [-1, 1]:
                    neighbor_indices = list(indices)
                    neighbor_indices[dim] = indices[dim] + offset
                    
                    # Check bounds
                    if 0 <= neighbor_indices[dim] < self.unified_field.shape[dim]:
                        neighbor_tuple = tuple(neighbor_indices)
                        self.unified_field[neighbor_tuple] += intensity * 0.5
        except IndexError as e:
            if not self.quiet_mode:
                print(f"⚠️  IndexError in imprint: {e}, indices: {indices}, shape: {self.unified_field.shape}")
    
    def _apply_multidimensional_imprint(self, spatial_indices: Tuple[int, int, int, int, int], 
                                       dynamics_values: torch.Tensor, intensity: float):
        """Apply Gaussian imprint across all field dimensions - ENHANCED for better differentiation."""
        x_idx, y_idx, z_idx, scale_idx, time_idx = spatial_indices
        
        # Create 5D Gaussian in spatial+scale+time dimensions
        spatial_spread = 1.5  # Tighter spread for more distinct patterns
        
        # Use vectorized operations for efficiency
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-1, 2):
                    x_pos = x_idx + dx
                    y_pos = y_idx + dy
                    z_pos = z_idx + dz
                    
                    # Check spatial bounds
                    if (0 <= x_pos < self.spatial_resolution and
                        0 <= y_pos < self.spatial_resolution and
                        0 <= z_pos < self.spatial_resolution):
                        
                        # Calculate spatial distance
                        spatial_distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        spatial_weight = math.exp(-(spatial_distance**2) / (2 * spatial_spread**2))
                        
                        # Apply across scale and time with modulated intensity
                        # Include dynamics values to create unique patterns
                        for s_offset in range(-1, 2):
                            s_pos = scale_idx + s_offset
                            if 0 <= s_pos < 10:
                                for t_offset in range(-1, 2):
                                    t_pos = time_idx + t_offset
                                    if 0 <= t_pos < 15:
                                        # Combine spatial weight with scale/time distance
                                        scale_time_distance = math.sqrt(s_offset**2 + t_offset**2)
                                        total_weight = spatial_weight * math.exp(-(scale_time_distance**2) / 4.0)
                                        
                                        # Modulate by dynamics values for unique patterns
                                        if dynamics_values.numel() > 0:
                                            # Use first few dynamics values to modulate the imprint
                                            modulation = 1.0 + 0.2 * torch.mean(dynamics_values[:5]).item()
                                            total_weight *= modulation
                                        
                                        # Apply weighted intensity
                                        self.unified_field[x_pos, y_pos, z_pos, s_pos, t_pos] += intensity * total_weight
    
    def _evolve_unified_field(self):
        """
        Evolve the unified field - this replaces ALL discrete learning and adaptation.
        Uses homeostatic non-uniform decay for natural memory consolidation.
        """
        # Apply field decay (back to simple uniform decay for now)
        self.unified_field *= self.field_decay_rate
        
        # Add very small baseline to prevent complete zeros, but preserve actual activations
        baseline = 0.0001  # Much smaller to not interfere with real values
        zero_mask = self.unified_field == 0.0
        self.unified_field[zero_mask] = baseline
        
        # Apply field diffusion (simple 3D spatial diffusion)
        self._apply_spatial_diffusion()
        
        # Apply constraint-guided evolution
        self._apply_constraint_guided_evolution()
        
        # Update topology regions (only if they exist)
        if self.topology_regions:
            self._update_topology_regions()
        
        # Calculate gradient flows for action generation
        self._calculate_gradient_flows()
        
        # ENERGY NORMALIZATION: Prevent unbounded growth
        # Calculate current total energy
        current_energy = torch.sum(torch.abs(self.unified_field)).item()
        target_energy = 100000.0  # Target equilibrium energy
        
        # If energy exceeds threshold, normalize it back
        if current_energy > target_energy * 1.5:  # 50% above target
            normalization_factor = target_energy / current_energy
            self.unified_field *= normalization_factor
            
            if not self.quiet_mode and self.brain_cycles % 100 == 0:
                print(f"   ⚡ Energy normalized: {current_energy:.0f} → {target_energy:.0f}")
        
        self.field_evolution_cycles += 1
        
        # PERFORMANCE: Run maintenance every 25 cycles for M1 Mac development
        # This keeps performance stable without affecting learning
        if self.field_evolution_cycles % 25 == 0:
            self._trigger_maintenance()
    
    def _apply_spatial_diffusion(self):
        """Apply diffusion in spatial dimensions - OPTIMIZED with vectorized operations."""
        # PERFORMANCE FIX: Use vectorized averaging instead of nested loops
        # OPTIMIZED: Apply spatial diffusion with minimal performance impact
        # Only diffuse occasionally to avoid performance bottleneck  
        if self.field_evolution_cycles % 10 == 0:  # Every 10th cycle
            # Simple vectorized diffusion on first 3 spatial dimensions only
            # Get interior points to avoid boundary issues
            if self.spatial_resolution > 2:
                interior_slice = slice(1, self.spatial_resolution - 1)
                
                # Calculate diffused values for interior points using slicing
                # Average of 6-connected neighbors in 3D space
                diffused_interior = (
                    self.unified_field[interior_slice, interior_slice, interior_slice] +
                    self.unified_field[2:, interior_slice, interior_slice] +
                    self.unified_field[:-2, interior_slice, interior_slice] +
                    self.unified_field[interior_slice, 2:, interior_slice] +
                    self.unified_field[interior_slice, :-2, interior_slice] +
                    self.unified_field[interior_slice, interior_slice, 2:] +
                    self.unified_field[interior_slice, interior_slice, :-2]
                ) / 7.0
                
                # Apply diffusion to interior points only
                self.unified_field[interior_slice, interior_slice, interior_slice] = (
                    (1 - self.field_diffusion_rate) * self.unified_field[interior_slice, interior_slice, interior_slice] +
                    self.field_diffusion_rate * diffused_interior
                )
    
    def _apply_constraint_guided_evolution(self):
        """Apply constraint-guided field evolution using N-dimensional constraints."""
        # Constraint system is now enabled
        
        # Discover new constraints from current field state
        if self.field_evolution_cycles % 5 == 0:  # Discover constraints periodically
            # Convert gradient flows to format expected by constraint system
            formatted_gradients = {}
            if 'gradient_x' in self.gradient_flows:
                formatted_gradients['gradient_dim_0'] = self.gradient_flows['gradient_x']
            if 'gradient_y' in self.gradient_flows:
                formatted_gradients['gradient_dim_1'] = self.gradient_flows['gradient_y']
            if 'gradient_z' in self.gradient_flows:
                formatted_gradients['gradient_dim_2'] = self.gradient_flows['gradient_z']
                
            new_constraints = self.constraint_field.discover_constraints(
                self.unified_field, 
                formatted_gradients
            )
            
            if len(new_constraints) > 0 and not self.quiet_mode:
                print(f"🔗 Discovered {len(new_constraints)} new constraints")
        
        # Apply existing constraints to guide field evolution
        constraint_forces = self.constraint_field.enforce_constraints(self.unified_field)
        
        # Apply constraint forces with appropriate scaling
        if torch.any(constraint_forces != 0):
            # Scale down to prevent instability
            self.unified_field += constraint_forces * self.field_evolution_rate * 0.1
            
        # Apply gradient smoothness constraint periodically for stability
        if self.field_evolution_cycles % 10 == 0:
            gradient_penalty = 0.01
            
            # Simple Laplacian smoothing on spatial dimensions
            for dim in range(min(3, len(self.unified_field.shape))):
                if self.unified_field.shape[dim] > 2:
                    # Create slices for computing differences
                    slices_center = [slice(1, -1) if i == dim else slice(None) for i in range(len(self.unified_field.shape))]
                    slices_left = [slice(None, -2) if i == dim else slice(None) for i in range(len(self.unified_field.shape))]
                    slices_right = [slice(2, None) if i == dim else slice(None) for i in range(len(self.unified_field.shape))]
                    
                    # Laplacian approximation
                    laplacian = (self.unified_field[slices_left] + self.unified_field[slices_right] - 
                                2 * self.unified_field[slices_center])
                    
                    # Apply smoothing
                    self.unified_field[slices_center] += gradient_penalty * laplacian
    
    def _discover_field_topology(self, experience: UnifiedFieldExperience):
        """Discover stable topology regions - this replaces discrete pattern storage."""
        # Look for stable field regions (high activation, low gradient)
        field_position = experience.field_coordinates  # Use full 37D coordinates
        
        # Check if we already have a region near this position
        existing_region = None
        spatial_threshold = 0.5  # Distance threshold for considering regions "same"
        
        for region_key, region_info in self.topology_regions.items():
            # Check spatial distance (first 3 dimensions)
            spatial_distance = torch.norm(field_position[:3] - region_info['center'][:3]).item()
            if spatial_distance < spatial_threshold:
                existing_region = region_key
                break
        
        # Use existing region key or create new one
        if existing_region:
            region_key = existing_region
        else:
            region_key = f"region_{len(self.topology_regions)}"
            
        stability_threshold = self.topology_stability_threshold
        
        # Calculate local field stability
        # field_position is in [-2, 2] range after amplification
        x_idx = int((torch.clamp(field_position[0], -2.0, 2.0) + 2.0) * 0.25 * (self.spatial_resolution - 1))
        y_idx = int((torch.clamp(field_position[1], -2.0, 2.0) + 2.0) * 0.25 * (self.spatial_resolution - 1))
        z_idx = int((torch.clamp(field_position[2], -2.0, 2.0) + 2.0) * 0.25 * (self.spatial_resolution - 1))
        
        if not self.quiet_mode:
            print(f"   📍 Field position: [{field_position[0]:.2f}, {field_position[1]:.2f}, {field_position[2]:.2f}] -> indices: [{x_idx}, {y_idx}, {z_idx}]")
        
        if (0 <= x_idx < self.spatial_resolution and 
            0 <= y_idx < self.spatial_resolution and
            0 <= z_idx < self.spatial_resolution):
            
            # Sample local SPATIAL field values only
            # We care about spatial stability, not averaging across all dimensions
            if hasattr(self, '_last_imprint_indices'):
                # Sample the 3x3x3 spatial region at the specific dimension indices
                local_spatial_region = self.unified_field[
                    max(0, x_idx-1):min(self.spatial_resolution, x_idx+2),
                    max(0, y_idx-1):min(self.spatial_resolution, y_idx+2),
                    max(0, z_idx-1):min(self.spatial_resolution, z_idx+2),
                    self._last_imprint_indices[3],  # scale
                    self._last_imprint_indices[4],  # time
                    self._last_imprint_indices[5],  # oscillatory
                    self._last_imprint_indices[6],  # flow
                    self._last_imprint_indices[7],  # topology
                    self._last_imprint_indices[8],  # energy
                    self._last_imprint_indices[9],  # coupling
                    self._last_imprint_indices[10]  # emergence
                ]
                
                # Use max for detection (more robust to sparse activations)
                region_max = torch.max(local_spatial_region).item()
                region_mean = torch.mean(local_spatial_region).item()
                region_std = torch.std(local_spatial_region.float()).item()
                
                # NaN protection for statistics
                if torch.isnan(torch.tensor(region_std)) or local_spatial_region.numel() <= 1:
                    region_std = 0.1  # Default small variance
                
                if not self.quiet_mode:
                    print(f"   🔍 Topology check: region max={region_max:.4f}, mean={region_mean:.4f}, std={region_std:.4f}, threshold={stability_threshold}")
            else:
                # Fallback to checking just center point
                region_max = 0.0
                region_mean = 0.0
                region_std = 1.0
                if not self.quiet_mode:
                    print("   ⚠️  No imprint indices available for topology check")
            
            # High activation + low variance = stable topology
            # Use max instead of mean for robustness to sparse activations
            if region_max > stability_threshold and region_std < 0.5:  # More lenient std threshold
                if existing_region:
                    # Update existing region
                    region = self.topology_regions[region_key]
                    region['activation'] = 0.9 * region['activation'] + 0.1 * region_mean  # Smooth update
                    region['stability'] = 1.0 / (region_std + 1e-6)
                    region['last_activation'] = time.time()
                    region['activation_count'] += 1
                    region['importance'] = min(region['importance'] * 1.1, 10.0)  # Increase importance
                    # Update field indices to latest imprint location
                    region['field_indices'] = self._last_imprint_indices.copy()
                    
                    if not self.quiet_mode:
                        print(f"   🏔️ Topology region reinforced: {region_key} "
                              f"(activation={region['activation']:.3f}, count={region['activation_count']})")
                else:
                    # Create new region
                    self.topology_regions[region_key] = {
                        'center': field_position.clone(),  # Full 37D coordinates
                        'field_indices': self._last_imprint_indices.copy(),  # Store field indices
                        'activation': region_mean,
                        'stability': 1.0 / (region_std + 1e-6),
                        'discovery_time': time.time(),
                        'discovery_cycle': self.brain_cycles,
                        'last_activation': time.time(),
                        'activation_count': 1,
                        'importance': 1.0,  # Initial importance
                        'decay_rate': self.field_decay_rate,  # Can be adjusted per region
                        'consolidation_level': 0,  # Increases with maintenance cycles
                        'associated_regions': []  # Links to related memories
                    }
                    self.topology_discoveries += 1
                    
                    if not self.quiet_mode:
                        print(f"   🏔️ Topology region discovered: {region_key} "
                              f"(activation={region_mean:.3f}, stability={1.0/(region_std+1e-6):.1f})")
    
    def _activate_resonant_memories(self, experience: UnifiedFieldExperience):
        """Activate topology regions that resonate with current experience."""
        resonance_threshold = 0.7  # Similarity threshold for activation
        
        for region_key, region_info in self.topology_regions.items():
            # Compute similarity between experience and stored region
            similarity = torch.nn.functional.cosine_similarity(
                experience.field_coordinates.unsqueeze(0),
                region_info['center'].unsqueeze(0)
            ).item()
            
            if similarity > resonance_threshold:
                # Reactivate this memory region
                region_info['last_activation'] = time.time()
                region_info['activation_count'] += 1
                region_info['importance'] *= 1.1  # Boost importance
                
                # Strengthen the field at this region's spatial location
                if len(region_info['center']) >= 3:
                    x_idx = int((region_info['center'][0] + 1) * 0.5 * (self.spatial_resolution - 1))
                    y_idx = int((region_info['center'][1] + 1) * 0.5 * (self.spatial_resolution - 1))
                    z_idx = int((region_info['center'][2] + 1) * 0.5 * (self.spatial_resolution - 1))
                    
                    if (0 <= x_idx < self.spatial_resolution and 
                        0 <= y_idx < self.spatial_resolution and
                        0 <= z_idx < self.spatial_resolution):
                        # Reinforce this memory in the field
                        self.unified_field[x_idx, y_idx, z_idx] *= (1 + similarity * 0.1)
    
    def _update_topology_regions(self):
        """Update existing topology regions based on field evolution and importance."""
        regions_to_remove = []
        current_time = time.time()
        
        for region_key, region_info in self.topology_regions.items():
            center = region_info['center']
            
            # Apply forgetting curve based on time since last activation
            time_since_activation = current_time - region_info['last_activation']
            memory_half_life = 3600.0  # 1 hour half-life
            retention = math.exp(-time_since_activation / memory_half_life)
            region_info['importance'] *= retention
            
            # Convert center coordinates to indices using actual field shape
            x_idx = max(0, min(self.spatial_resolution - 1, int((center[0] + 1) * 0.5 * (self.spatial_resolution - 1))))
            y_idx = max(0, min(self.spatial_resolution - 1, int((center[1] + 1) * 0.5 * (self.spatial_resolution - 1))))
            z_idx = max(0, min(self.spatial_resolution - 1, int((center[2] + 1) * 0.5 * (self.spatial_resolution - 1))))
            
            # Use actual field dimensions
            scale_idx = max(0, min(self.unified_field.shape[3] - 1, int((center[3] + 1) * 0.5 * (self.unified_field.shape[3] - 1))))
            time_idx = max(0, min(self.unified_field.shape[4] - 1, int((center[4] + 1) * 0.5 * (self.unified_field.shape[4] - 1))))
            
            # For higher dimensions, we need the stored indices
            if 'field_indices' in region_info:
                # Use stored indices for dimensions 5-10
                indices = [x_idx, y_idx, z_idx, scale_idx, time_idx] + region_info['field_indices'][5:]
            else:
                # Fallback: just check the spatial center
                indices = [x_idx, y_idx, z_idx, scale_idx, time_idx, 0, 0, 0, 0, 0, 0]
            
            try:
                # Check the same 3x3x3 spatial region used during discovery
                if 'field_indices' in region_info:
                    if not self.quiet_mode:
                        print(f"   🔍 Checking region with indices: {region_info['field_indices'][:3]}, stored: {region_info['field_indices'][3:]}")
                    
                    local_spatial_region = self.unified_field[
                        max(0, x_idx-1):min(self.spatial_resolution, x_idx+2),
                        max(0, y_idx-1):min(self.spatial_resolution, y_idx+2),
                        max(0, z_idx-1):min(self.spatial_resolution, z_idx+2),
                        region_info['field_indices'][3],  # scale
                        region_info['field_indices'][4],  # time
                        region_info['field_indices'][5],  # oscillatory
                        region_info['field_indices'][6],  # flow
                        region_info['field_indices'][7],  # topology
                        region_info['field_indices'][8],  # energy
                        region_info['field_indices'][9],  # coupling
                        region_info['field_indices'][10]  # emergence
                    ]
                    # Use max activation in the region instead of mean
                    # This is more robust to sparse activations
                    current_activation = torch.max(local_spatial_region).item()
                    
                    # Also check the exact point and mean
                    exact_point = self.unified_field[tuple(region_info['field_indices'][:11])].item()
                    region_mean = torch.mean(local_spatial_region).item()
                    # Use the best of exact point or max for robustness
                    current_activation = max(current_activation, exact_point)
                    if not self.quiet_mode:
                        print(f"   🔍 Region check: exact={exact_point:.4f}, max={current_activation:.4f}, mean={region_mean:.4f}")
                        print(f"   🔍 Spatial region shape: {local_spatial_region.shape}, values > 0.02: {(local_spatial_region > 0.02).sum().item()}")
                else:
                    # Fallback to single point
                    current_activation = self.unified_field[tuple(indices[:11])].item()
                
                # Update region activation
                region_info['activation'] = float(current_activation)
                
                if not self.quiet_mode:
                    print(f"   🔍 Updating {region_key}: activation={current_activation:.4f}, threshold={self.topology_stability_threshold * 0.5:.4f}")
                
                # Remove regions that have become too weak
                # More lenient removal - only remove if truly dead
                removal_threshold = 0.001  # Much lower than discovery threshold
                if current_activation < removal_threshold:
                    regions_to_remove.append(region_key)
                    if not self.quiet_mode:
                        print(f"   ❌ Marking {region_key} for removal (activation={current_activation:.6f} < {removal_threshold})")
            except IndexError:
                regions_to_remove.append(region_key)
        
        # Remove weak regions
        for region_key in regions_to_remove:
            del self.topology_regions[region_key]
    
    def _calculate_gradient_flows(self):
        """Calculate gradient flows for action generation - OPTIMIZED with local computation."""
        # Use optimized gradient calculator with local region only
        # This provides massive speedup while preserving functionality
        
        # Use the actual robot position from last imprint, not field center
        if hasattr(self, '_last_imprint_indices') and self._last_imprint_indices:
            # Use the actual robot position (first 3 indices)
            region_center = tuple(self._last_imprint_indices[:3])
        else:
            # Fallback to field center
            region_center = None
            
        computed_gradients = self.gradient_calculator.calculate_gradients(
            self.unified_field,
            dimensions_to_compute=[0, 1, 2, 3, 4],  # Spatial + scale + time dimensions
            local_region_only=True,  # Only compute in 3x3x3 region
            region_center=region_center,  # Use actual robot position
            region_size=3  # 3x3x3 local region
        )
        
        # Map to legacy gradient names for compatibility
        gradient_mapping = {
            'gradient_dim_0': 'gradient_x',
            'gradient_dim_1': 'gradient_y',
            'gradient_dim_2': 'gradient_z',
            'gradient_dim_3': 'gradient_scale',
            'gradient_dim_4': 'gradient_time'
        }
        
        self.gradient_flows.clear()
        for new_name, old_name in gradient_mapping.items():
            if new_name in computed_gradients:
                self.gradient_flows[old_name] = computed_gradients[new_name]
        
        # Log cache performance occasionally
        if not self.quiet_mode and self.field_evolution_cycles % 100 == 0:
            cache_stats = self.gradient_calculator.get_cache_stats()
            if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
                print(f"⚡ Gradient cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    def _field_gradients_to_robot_action(self) -> FieldNativeAction:
        """
        Generate robot action from field gradients - FIXED with proper multi-dimensional aggregation.
        """
        # Calculate dominant gradients for motor action
        motor_gradients = torch.zeros(4, device=self.device)  # 4D motor space
        
        if self.gradient_flows:
            # Get center position and define local region
            center_idx = self.spatial_resolution // 2
            window_size = 3  # Local region size
            half_window = window_size // 2
            
            # Define local region bounds
            x_slice = slice(max(0, center_idx - half_window), min(self.spatial_resolution, center_idx + half_window + 1))
            y_slice = slice(max(0, center_idx - half_window), min(self.spatial_resolution, center_idx + half_window + 1))
            z_slice = slice(max(0, center_idx - half_window), min(self.spatial_resolution, center_idx + half_window + 1))
            
            # Extract and aggregate gradients from local region
            if 'gradient_x' in self.gradient_flows:
                # Get local region of X gradient
                grad_x_local = self.gradient_flows['gradient_x'][x_slice, y_slice, z_slice]
                # Aggregate across spatial dimensions, preserving others
                grad_x_aggregated = torch.mean(grad_x_local, dim=(0, 1, 2))
                # Weight by magnitude and take weighted average across remaining dimensions
                if grad_x_aggregated.numel() > 1:
                    weights = torch.abs(grad_x_aggregated) + 1e-8
                    weights = weights / torch.sum(weights)
                    motor_gradients[0] = torch.sum(grad_x_aggregated * weights).item()
                else:
                    motor_gradients[0] = grad_x_aggregated.item() if grad_x_aggregated.numel() == 1 else 0.0
            
            if 'gradient_y' in self.gradient_flows:
                # Get local region of Y gradient
                grad_y_local = self.gradient_flows['gradient_y'][x_slice, y_slice, z_slice]
                grad_y_aggregated = torch.mean(grad_y_local, dim=(0, 1, 2))
                if grad_y_aggregated.numel() > 1:
                    weights = torch.abs(grad_y_aggregated) + 1e-8
                    weights = weights / torch.sum(weights)
                    motor_gradients[1] = torch.sum(grad_y_aggregated * weights).item()
                else:
                    motor_gradients[1] = grad_y_aggregated.item() if grad_y_aggregated.numel() == 1 else 0.0
            
            if 'gradient_z' in self.gradient_flows:
                # Get local region of Z gradient
                grad_z_local = self.gradient_flows['gradient_z'][x_slice, y_slice, z_slice]
                grad_z_aggregated = torch.mean(grad_z_local, dim=(0, 1, 2))
                if grad_z_aggregated.numel() > 1:
                    weights = torch.abs(grad_z_aggregated) + 1e-8
                    weights = weights / torch.sum(weights)
                    motor_gradients[2] = torch.sum(grad_z_aggregated * weights).item()
                else:
                    motor_gradients[2] = grad_z_aggregated.item() if grad_z_aggregated.numel() == 1 else 0.0
            
            # Fourth motor dimension from scale and time gradients
            fourth_component = 0.0
            if 'gradient_scale' in self.gradient_flows:
                grad_scale_local = self.gradient_flows['gradient_scale'][x_slice, y_slice, z_slice]
                grad_scale_aggregated = torch.mean(grad_scale_local, dim=(0, 1, 2))
                if grad_scale_aggregated.numel() > 1:
                    weights = torch.abs(grad_scale_aggregated) + 1e-8
                    weights = weights / torch.sum(weights)
                    fourth_component += torch.sum(grad_scale_aggregated * weights).item()
                else:
                    fourth_component += grad_scale_aggregated.item() if grad_scale_aggregated.numel() == 1 else 0.0
            
            if 'gradient_time' in self.gradient_flows:
                grad_time_local = self.gradient_flows['gradient_time'][x_slice, y_slice, z_slice]
                grad_time_aggregated = torch.mean(grad_time_local, dim=(0, 1, 2))
                if grad_time_aggregated.numel() > 1:
                    weights = torch.abs(grad_time_aggregated) + 1e-8
                    weights = weights / torch.sum(weights)
                    fourth_component += torch.sum(grad_time_aggregated * weights).item()
                else:
                    fourth_component += grad_time_aggregated.item() if grad_time_aggregated.numel() == 1 else 0.0
            
            motor_gradients[3] = fourth_component
        
        # Apply gradient following strength
        motor_commands = motor_gradients * self.gradient_following_strength
        
        # NaN/Inf protection
        motor_commands = torch.nan_to_num(motor_commands, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate gradient strength before clamping
        gradient_strength = torch.norm(motor_gradients).item()
        if torch.isnan(torch.tensor(gradient_strength)) or torch.isinf(torch.tensor(gradient_strength)):
            gradient_strength = 0.0
        
        # PREDICTION IMPROVEMENT ADDICTION: Apply learning-based action modulation
        prediction_modifier = self._get_prediction_improvement_addiction_modifier()
        motor_commands = motor_commands * prediction_modifier
        
        # Clamp to valid motor range first
        motor_commands = torch.clamp(motor_commands, -1.0, 1.0)
        
        # Check for tanh normalization issue (values too small → zero after tanh)
        motor_magnitude_after_clamp = torch.norm(motor_commands).item()
        
        # FALLBACK: If gradients are extremely weak, apply exploration
        if gradient_strength < 1e-5 or motor_magnitude_after_clamp < 1e-6:
            if not self.quiet_mode and self.brain_cycles % 50 == 0:
                print(f"   ⚠️  Weak gradients (strength={gradient_strength:.8f}), applying exploration")
            
            # Generate small random exploration action
            import random
            motor_commands = torch.tensor([
                0.2 * (random.random() - 0.5),  # Increased exploration strength
                0.2 * (random.random() - 0.5),
                0.1 * (random.random() - 0.5),
                0.1 * (random.random() - 0.5)
            ], dtype=torch.float32, device=self.device)
            action_confidence = 0.1  # Low confidence for exploration
        else:
            # Action confidence now comes from prediction accuracy, not gradient strength
            action_confidence = self._current_prediction_confidence
            
            # Debug output for successful gradient following
            if not self.quiet_mode and self.brain_cycles % 100 == 0:
                print(f"   ✅ Gradient following: strength={gradient_strength:.6f}, "
                      f"prediction_confidence={action_confidence:.3f}, "
                      f"motor=[{motor_commands[0]:.3f}, {motor_commands[1]:.3f}, {motor_commands[2]:.3f}, {motor_commands[3]:.3f}]")
        
        # MOTOR SMOOTHING: Blend with previous commands for smoother movement
        # This helps the brain learn the "feeling" of smooth control
        smoothed_commands = (
            (1 - self.motor_smoothing_factor) * motor_commands + 
            self.motor_smoothing_factor * self.previous_motor_commands
        )
        
        # Update previous commands for next cycle
        self.previous_motor_commands = smoothed_commands.clone()
        
        # Use smoothed commands for the action
        motor_commands = smoothed_commands
        
        action = FieldNativeAction(
            timestamp=time.time(),
            field_gradients=motor_gradients,
            motor_commands=motor_commands,
            action_confidence=action_confidence,
            gradient_strength=gradient_strength
        )
        
        self.field_actions.append(action)
        self.gradient_actions += 1
        
        return action
    
    def _get_prediction_improvement_addiction_modifier(self) -> float:
        """
        PREDICTION IMPROVEMENT ADDICTION: Reward actions that lead to learning acceleration.
        
        The "addiction" is to the rate of prediction improvement, not poor predictions.
        This creates curiosity-driven behavior that seeks learning opportunities.
        """
        # Need sufficient history for trend calculation
        if len(self._prediction_confidence_history) < 10:
            return 1.2  # Default exploration for insufficient data
        
        # Calculate improvement velocity (learning rate)
        recent_window = self._prediction_confidence_history[-10:]
        
        # Simple linear regression slope
        x_values = list(range(len(recent_window)))
        y_values = recent_window
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)
        
        if n * sum_xx - sum_x * sum_x == 0:
            improvement_rate = 0.0
        else:
            improvement_rate = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        
        # Track improvement rate history
        self._improvement_rate_history.append(improvement_rate)
        if len(self._improvement_rate_history) > 20:
            self._improvement_rate_history = self._improvement_rate_history[-20:]
        
        # ADDICTION LOGIC: Reward learning but enable exploitation when confidence is high
        if improvement_rate > 0.02:  # Strong learning happening
            modifier = 1.6  # Amplify actions that lead to rapid learning
            if not self.quiet_mode and self.brain_cycles % 50 == 0:
                print(f"🔥 LEARNING ADDICTION: Strong improvement ({improvement_rate:.4f}) → Amplifying actions")
        elif improvement_rate > 0.01:  # Moderate learning
            modifier = 1.3  # Encourage continued learning
        elif improvement_rate < -0.01:  # Learning degrading
            modifier = 1.4  # Explore to find better learning opportunities
        elif abs(improvement_rate) < 0.005:  # Stagnant learning
            # EXPLOITATION LOGIC: If confidence is high and learning is stagnant, exploit
            if self._current_prediction_confidence > 0.7 and self.brain_cycles > 100:
                modifier = 0.8  # Reduce exploration to enable exploitation
                if not self.quiet_mode and self.brain_cycles % 100 == 0:
                    print(f"🎯 High confidence + stagnant learning → Exploiting (conf={self._current_prediction_confidence:.3f})")
            else:
                modifier = 1.5  # Strong exploration to break stagnation
        else:  # Mild improvement
            modifier = 1.1  # Mild encouragement
        
        return modifier
    
    def _normalize_pattern_for_memory(self, field_coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize field coordinates for pattern-based memory encoding.
        This ensures memories are based on patterns, not raw magnitudes.
        A whisper should be as memorable as a shout if the pattern is distinctive.
        """
        coords = field_coords.clone()
        
        # Group coordinates by dimension families for family-wise normalization
        families = [
            (0, 5),    # Spatial (preserve relative positions)
            (5, 11),   # Oscillatory (normalize frequencies)
            (11, 19),  # Flow (normalize gradients)
            (19, 25),  # Topology (preserve binary nature)
            (25, 29),  # Energy (normalize intensities)
            (29, 34),  # Coupling (normalize correlations)
            (34, 37)   # Emergence (normalize combinations)
        ]
        
        for start, end in families:
            family_coords = coords[start:end]
            
            # Different normalization strategies per family
            if start == 0:  # Spatial - preserve relative positions
                # Center around origin but keep relative distances
                family_coords = family_coords - torch.mean(family_coords)
            elif start == 19:  # Topology - already binary-like
                # Just ensure [-1, 1] range
                family_coords = torch.clamp(family_coords, -1.0, 1.0)
            else:
                # Other families - normalize by pattern
                magnitude = torch.norm(family_coords)
                if magnitude > 1e-6:
                    # Normalize to unit sphere to capture pattern
                    family_coords = family_coords / magnitude
                else:
                    # Very weak signal - use small random pattern
                    family_coords = torch.randn_like(family_coords) * 0.1
            
            coords[start:end] = family_coords
        
        return coords
    
    def _calculate_experience_novelty(self, field_coordinates: torch.Tensor) -> float:
        """Calculate novelty of experience in field space."""
        if len(self.field_experiences) == 0:
            return 1.0  # First experience is maximally novel
        
        # Compare to recent experiences
        recent_experiences = self.field_experiences[-10:]  # Last 10 experiences
        
        min_distance = float('inf')
        for exp in recent_experiences:
            distance = torch.norm(field_coordinates - exp.field_coordinates).item()
            min_distance = min(min_distance, distance)
        
        # Normalize distance to [0, 1] range
        max_possible_distance = math.sqrt(self.total_dimensions * 4)  # Worst case: all dims at opposite extremes
        novelty = min(1.0, min_distance / max_possible_distance)
        
        return novelty
    
    def _get_field_brain_state(self, cycle_time_ms: float) -> Dict[str, Any]:
        """Get brain state information for robot interface compatibility."""
        field_stats = {
            'brain_type': 'field_native',
            'cycle_time_ms': cycle_time_ms,
            'brain_cycles': self.brain_cycles,
            'field_evolution_cycles': self.field_evolution_cycles,
            
            # Field-specific metrics
            'field_dimensions': self.total_dimensions,
            'field_max_activation': torch.max(self.unified_field).item(),
            'field_mean_activation': torch.mean(self.unified_field).item(),
            'field_total_energy': torch.sum(self.unified_field).item(),
            
            # Topology and learning
            'topology_regions_count': len(self.topology_regions),
            'topology_discoveries': self.topology_discoveries,
            'field_experiences_count': len(self.field_experiences),
            
            # Action generation
            'gradient_actions_count': self.gradient_actions,
            'last_action_confidence': self.field_actions[-1].action_confidence if self.field_actions else 0.0,
            'last_gradient_strength': self.field_actions[-1].gradient_strength if self.field_actions else 0.0,
            
            # Field dynamics by family
            'oscillatory_activity': self._get_family_activity(FieldDynamicsFamily.OSCILLATORY),
            'flow_activity': self._get_family_activity(FieldDynamicsFamily.FLOW),
            'topology_activity': self._get_family_activity(FieldDynamicsFamily.TOPOLOGY),
            'energy_activity': self._get_family_activity(FieldDynamicsFamily.ENERGY),
            'coupling_activity': self._get_family_activity(FieldDynamicsFamily.COUPLING),
            'emergence_activity': self._get_family_activity(FieldDynamicsFamily.EMERGENCE),
            
            # PREDICTION IMPROVEMENT ADDICTION: Learning metrics
            'prediction_confidence': self._current_prediction_confidence,
            'prediction_efficiency': self.calculate_prediction_efficiency(),
            'intrinsic_reward': self.compute_intrinsic_reward(),
            'improvement_rate': self._improvement_rate_history[-1] if self._improvement_rate_history else 0.0,
            'learning_addiction_modifier': self._get_prediction_improvement_addiction_modifier(),
        }
        
        return field_stats
    
    def _get_family_activity(self, family: FieldDynamicsFamily) -> float:
        """Get average activity level for a dimension family."""
        family_dims = [dim for dim in self.field_dimensions if dim.family == family]
        if not family_dims:
            return 0.0
        
        # Sample field activity in family-related regions
        total_activity = 0.0
        count = 0
        
        for dim in family_dims:
            # Sample some field regions (simplified)
            center = self.spatial_resolution // 2
            try:
                activity = torch.mean(self.unified_field[center-1:center+2, center-1:center+2, center, :, :]).item()
                total_activity += activity
                count += 1
            except IndexError:
                pass
        
        return total_activity / max(1, count)
    
    def _perform_field_maintenance(self):
        """
        Run periodic maintenance operations on the field.
        This is called off the hot path to prevent energy accumulation.
        """
        maintenance_start = time.perf_counter()
        
        # Apply energy dissipation to prevent unbounded accumulation
        self.unified_field *= self.field_energy_dissipation_rate
        
        # Clean up near-zero values to maintain sparsity
        threshold = 1e-6
        self.unified_field[torch.abs(self.unified_field) < threshold] = 0.0
        
        # Clean up stale topology regions and consolidate important ones
        regions_removed = 0
        regions_consolidated = 0
        regions_to_remove = []
        
        for region_key, region_info in self.topology_regions.items():
            # Consolidate important regions (make them more persistent)
            if region_info['importance'] > 2.0:  # Important memory
                region_info['consolidation_level'] += 1
                region_info['decay_rate'] *= 0.99  # Decay even slower
                regions_consolidated += 1
                
                # Strengthen this region in the field
                if len(region_info['center']) >= 3:
                    x_idx = int((region_info['center'][0] + 1) * 0.5 * (self.spatial_resolution - 1))
                    y_idx = int((region_info['center'][1] + 1) * 0.5 * (self.spatial_resolution - 1))
                    z_idx = int((region_info['center'][2] + 1) * 0.5 * (self.spatial_resolution - 1))
                    
                    if (0 <= x_idx < self.spatial_resolution and 
                        0 <= y_idx < self.spatial_resolution and
                        0 <= z_idx < self.spatial_resolution):
                        # Apply slower decay to important regions
                        self.unified_field[x_idx, y_idx, z_idx] /= region_info['decay_rate']
            
            # Remove unimportant regions that have faded
            elif region_info['importance'] < 0.1:
                regions_to_remove.append(region_key)
        
        # Remove stale regions
        for region_key in regions_to_remove:
            del self.topology_regions[region_key]
            regions_removed += 1
        
        # Clear old gradient flows to prevent memory buildup
        if len(self.gradient_flows) > 10:
            # Keep only the most recent gradient calculations
            keys_to_keep = ['gradient_x', 'gradient_y', 'gradient_z']
            self.gradient_flows = {k: v for k, v in self.gradient_flows.items() if k in keys_to_keep}
        
        # Clear gradient cache to ensure fresh calculations after maintenance
        self.gradient_calculator.clear_cache()
        
        # Trim experience history if too long
        max_experiences = 1000
        if len(self.field_experiences) > max_experiences:
            self.field_experiences = self.field_experiences[-max_experiences:]
        
        if len(self.field_actions) > max_experiences:
            self.field_actions = self.field_actions[-max_experiences:]
        
        maintenance_time = (time.perf_counter() - maintenance_start) * 1000
        
        if not self.quiet_mode:
            field_energy = torch.sum(self.unified_field).item()
            print(f"🔧 Field maintenance completed: {maintenance_time:.2f}ms, "
                  f"energy={field_energy:.3f}, consolidated={regions_consolidated}, removed={regions_removed}")
    
    def save_field_state(self, filepath: str, compress: bool = True) -> bool:
        """
        Save the current field state to disk with efficient compression.
        
        The field itself IS the memory - this persists the unified field topology
        that contains all learned patterns, concepts, and experiences.
        """
        import gzip
        import pickle
        import os
        
        try:
            # Prepare field state data
            field_state = {
                'unified_field': self.unified_field,
                'field_dimensions': self.field_dimensions,
                'total_dimensions': self.total_dimensions,
                'spatial_resolution': self.spatial_resolution,
                'temporal_window': self.temporal_window,
                'topology_regions': self.topology_regions,
                'gradient_flows': self.gradient_flows,
                'field_parameters': {
                    'field_decay_rate': self.field_decay_rate,
                    'field_diffusion_rate': self.field_diffusion_rate,
                    'gradient_following_strength': self.gradient_following_strength,
                    'topology_stability_threshold': self.topology_stability_threshold,
                    'field_evolution_rate': self.field_evolution_rate,
                    'constraint_discovery_rate': self.constraint_discovery_rate,
                },
                'field_statistics': {
                    'brain_cycles': self.brain_cycles,
                    'field_evolution_cycles': self.field_evolution_cycles,
                    'topology_discoveries': self.topology_discoveries,
                    'gradient_actions': self.gradient_actions,
                },
                'timestamp': time.time(),
                'field_shape': list(self.unified_field.shape)
            }
            
            if compress:
                # Use gzip compression for efficient storage
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(field_state, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(field_state, f)
            
            field_size_mb = self.unified_field.element_size() * self.unified_field.nelement() / (1024 * 1024)
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            compression_ratio = (field_size_mb / file_size_mb) if file_size_mb > 0 else 1.0
            
            if not self.quiet_mode:
                print(f"💾 Field state saved: {filepath}")
                print(f"   Field memory: {field_size_mb:.2f} MB")
                print(f"   File size: {file_size_mb:.2f} MB")
                print(f"   Compression: {compression_ratio:.1f}x")
            
            return True
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Failed to save field state: {e}")
            return False
    
    def load_field_state(self, filepath: str) -> bool:
        """
        Load field state from disk, restoring the unified field topology.
        
        This restores the field memory containing all learned patterns and experiences.
        """
        import gzip
        import pickle
        import os
        
        if not os.path.exists(filepath):
            if not self.quiet_mode:
                print(f"⚠️ Field state file not found: {filepath}")
            return False
        
        try:
            # Determine if file is compressed
            is_compressed = filepath.endswith('.gz') or filepath.endswith('.gzip')
            
            if is_compressed:
                with gzip.open(filepath, 'rb') as f:
                    field_state = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    field_state = pickle.load(f)
            
            # Restore field state
            self.unified_field = field_state['unified_field']
            self.field_dimensions = field_state['field_dimensions']
            self.total_dimensions = field_state['total_dimensions']
            self.spatial_resolution = field_state['spatial_resolution']
            self.temporal_window = field_state['temporal_window']
            self.topology_regions = field_state['topology_regions']
            self.gradient_flows = field_state['gradient_flows']
            
            # Restore parameters
            params = field_state['field_parameters']
            self.field_decay_rate = params['field_decay_rate']
            self.field_diffusion_rate = params['field_diffusion_rate']
            self.gradient_following_strength = params['gradient_following_strength']
            self.topology_stability_threshold = params['topology_stability_threshold']
            self.field_evolution_rate = params['field_evolution_rate']
            self.constraint_discovery_rate = params['constraint_discovery_rate']
            
            # Restore statistics
            stats = field_state['field_statistics']
            self.brain_cycles = stats['brain_cycles']
            self.field_evolution_cycles = stats['field_evolution_cycles']
            self.topology_discoveries = stats['topology_discoveries']
            self.gradient_actions = stats['gradient_actions']
            
            # Calculate load info
            load_time = field_state['timestamp']
            age_hours = (time.time() - load_time) / 3600.0
            field_size_mb = self.unified_field.element_size() * self.unified_field.nelement() / (1024 * 1024)
            
            if not self.quiet_mode:
                print(f"📂 Field state loaded: {filepath}")
                print(f"   Field memory: {field_size_mb:.2f} MB")
                print(f"   Field age: {age_hours:.2f} hours")
                print(f"   Topology regions: {len(self.topology_regions)}")
                print(f"   Brain cycles: {self.brain_cycles:,}")
            
            return True
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Failed to load field state: {e}")
            return False
    
    def save_field_delta(self, filepath: str, previous_field: torch.Tensor) -> bool:
        """
        Save incremental field changes with delta compression.
        
        Extremely efficient for continuous field updates - only stores changes.
        """
        import gzip
        import pickle
        import os
        
        try:
            # Calculate field differences
            diff_tensor = self.unified_field - previous_field
            changed_mask = torch.abs(diff_tensor) > 1e-6
            changed_count = torch.sum(changed_mask).item()
            
            if changed_count == 0:
                if not self.quiet_mode:
                    print("💾 No field changes detected - skipping delta save")
                return True
            
            # Extract only changed coordinates and values
            change_coords = torch.nonzero(changed_mask, as_tuple=False)
            change_values = diff_tensor[changed_mask]
            
            delta_data = {
                'field_shape': list(self.unified_field.shape),
                'change_coordinates': change_coords,
                'change_values': change_values,
                'changed_count': changed_count,
                'total_elements': self.unified_field.nelement(),
                'timestamp': time.time(),
                'brain_cycle': self.brain_cycles
            }
            
            # Compress and save delta
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(delta_data, f)
            
            # Calculate compression metrics
            original_size = self.unified_field.element_size() * self.unified_field.nelement()
            file_size = os.path.getsize(filepath)
            compression_ratio = original_size / file_size if file_size > 0 else 1.0
            change_percentage = (changed_count / self.unified_field.nelement()) * 100
            
            if not self.quiet_mode:
                print(f"💾 Field delta saved: {filepath}")
                print(f"   Changed elements: {changed_count:,} ({change_percentage:.4f}%)")
                print(f"   Delta compression: {compression_ratio:.1f}x")
                print(f"   File size: {file_size / 1024:.2f} KB")
            
            return True
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Failed to save field delta: {e}")
            return False
    
    def apply_field_delta(self, filepath: str) -> bool:
        """
        Apply incremental field changes from delta file.
        
        Reconstructs field state by applying saved deltas to current field.
        """
        import gzip
        import pickle
        import os
        
        if not os.path.exists(filepath):
            if not self.quiet_mode:
                print(f"⚠️ Field delta file not found: {filepath}")
            return False
        
        try:
            # Load delta data
            with gzip.open(filepath, 'rb') as f:
                delta_data = pickle.load(f)
            
            # Verify field shape compatibility
            expected_shape = tuple(delta_data['field_shape'])
            if self.unified_field.shape != expected_shape:
                if not self.quiet_mode:
                    print(f"❌ Field shape mismatch: {self.unified_field.shape} vs {expected_shape}")
                return False
            
            # Apply delta changes
            change_coords = delta_data['change_coordinates']
            change_values = delta_data['change_values']
            
            for i, coord in enumerate(change_coords):
                coord_tuple = tuple(coord.tolist())
                try:
                    self.unified_field[coord_tuple] += change_values[i]
                except IndexError:
                    pass  # Skip invalid coordinates
            
            # Update statistics
            changed_count = delta_data['changed_count']
            delta_age_seconds = time.time() - delta_data['timestamp']
            
            if not self.quiet_mode:
                print(f"📂 Field delta applied: {filepath}")
                print(f"   Applied changes: {changed_count:,}")
                print(f"   Delta age: {delta_age_seconds:.1f}s")
            
            return True
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Failed to apply field delta: {e}")
            return False
    
    def consolidate_field(self, consolidation_strength: float = 0.1) -> int:
        """
        Perform field consolidation - strengthen stable regions, decay weak ones.
        
        This is the field-native equivalent of memory consolidation during sleep.
        """
        consolidated_regions = 0
        
        try:
            # Apply stability-based consolidation
            field_magnitude = torch.abs(self.unified_field)
            stability_threshold = torch.median(field_magnitude) * 1.5
            
            # Strengthen stable high-activation regions
            stable_mask = field_magnitude > stability_threshold
            if torch.sum(stable_mask) > 0:
                self.unified_field[stable_mask] *= (1.0 + consolidation_strength)
                consolidated_regions = torch.sum(stable_mask).item()
            
            # Decay very weak regions (forgetting)
            weak_threshold = torch.median(field_magnitude) * 0.1
            weak_mask = field_magnitude < weak_threshold
            if torch.sum(weak_mask) > 0:
                self.unified_field[weak_mask] *= (1.0 - consolidation_strength * 0.5)
            
            # Update topology regions based on consolidation
            self._update_topology_regions()
            
            if not self.quiet_mode:
                print(f"🌙 Field consolidation complete")
                print(f"   Strengthened regions: {consolidated_regions:,}")
                print(f"   Topology regions: {len(self.topology_regions)}")
            
            return consolidated_regions
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Field consolidation failed: {e}")
            return 0
    
    def _start_maintenance_thread(self):
        """Start the background maintenance thread."""
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_worker,
            daemon=True,
            name="BrainMaintenance"
        )
        self._maintenance_thread.start()
    
    def _maintenance_worker(self):
        """Background worker that performs field maintenance when signaled."""
        while not self._shutdown_maintenance.is_set():
            try:
                # Wait for maintenance signal (timeout prevents hanging on shutdown)
                signal = self._maintenance_queue.get(timeout=1.0)
                
                if signal == "SHUTDOWN":
                    break
                    
                # Perform maintenance
                with torch.no_grad():
                    self._perform_field_maintenance()
                    
            except queue.Empty:
                # No maintenance needed, continue waiting
                continue
            except Exception as e:
                if not self.quiet_mode:
                    print(f"❌ Maintenance thread error: {e}")
    
    def _trigger_maintenance(self):
        """Signal the maintenance thread to run."""
        try:
            # Non-blocking put - if queue is full, skip (maintenance already pending)
            self._maintenance_queue.put_nowait("MAINTAIN")
        except queue.Full:
            pass  # Maintenance already queued
    
    def shutdown(self):
        """Cleanly shutdown the brain and maintenance thread."""
        self._shutdown_maintenance.set()
        if self._maintenance_thread:
            try:
                self._maintenance_queue.put_nowait("SHUTDOWN")
            except:
                pass
            self._maintenance_thread.join(timeout=2.0)
    
    def compute_intrinsic_reward(self) -> float:
        """
        Compute intrinsic reward based on prediction improvement rate.
        
        This creates the "addiction" to learning that drives curiosity.
        """
        if len(self._improvement_rate_history) < 2:
            return 0.5  # Neutral reward when insufficient data
        
        # Recent improvement rate
        recent_rate = self._improvement_rate_history[-1]
        
        # Convert improvement rate to reward signal
        # Positive improvement = high reward
        # Negative improvement = low reward (but not zero to maintain exploration)
        if recent_rate > 0:
            reward = 0.5 + min(0.5, recent_rate * 10)  # Scale to [0.5, 1.0]
        else:
            reward = 0.5 + max(-0.3, recent_rate * 5)  # Scale to [0.2, 0.5]
        
        return reward
    
    def calculate_prediction_efficiency(self) -> float:
        """
        Calculate prediction efficiency based on confidence improvement over time.
        
        This measures how well the brain is learning to predict.
        """
        if len(self._prediction_confidence_history) < 5:
            return 0.0
        
        # Calculate improvement trend
        early_conf = sum(self._prediction_confidence_history[:3]) / 3
        late_conf = sum(self._prediction_confidence_history[-3:]) / 3
        
        if early_conf == 0:
            return late_conf  # Pure improvement from zero
        
        improvement = (late_conf - early_conf) / early_conf
        
        # Normalize to [0, 1] range with 20% improvement = 1.0
        efficiency = min(1.0, max(0.0, improvement / 0.2))
        
        return efficiency
    
    def get_field_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive field memory statistics."""
        
        field_magnitude = torch.abs(self.unified_field)
        nonzero_count = torch.sum(field_magnitude > 1e-6).item()
        total_elements = self.unified_field.nelement()
        sparsity = 1.0 - (nonzero_count / total_elements)
        
        memory_size_mb = self.unified_field.element_size() * total_elements / (1024 * 1024)
        
        return {
            'field_shape': list(self.unified_field.shape),
            'total_elements': total_elements,
            'nonzero_elements': nonzero_count,
            'sparsity': sparsity,
            'memory_size_mb': memory_size_mb,
            'field_energy': torch.sum(field_magnitude).item(),
            'max_activation': torch.max(field_magnitude).item(),
            'mean_activation': torch.mean(field_magnitude).item(),
            'topology_regions': len(self.topology_regions),
            'gradient_flows': len(self.gradient_flows),
            'brain_cycles': self.brain_cycles,
            'field_evolution_cycles': self.field_evolution_cycles,
            'topology_discoveries': self.topology_discoveries,
        }
    
    def run_recommended_maintenance(self) -> Dict[str, bool]:
        """
        Run recommended maintenance tasks based on brain state.
        Called by brain_loop during idle cycles.
        
        Returns dict of maintenance tasks performed.
        """
        performed = {
            'energy_dissipation': False,
            'topology_cleanup': False,
            'memory_trim': False
        }
        
        # Check if we need energy dissipation
        cycles_since_maintenance = self.brain_cycles - self.last_maintenance_cycle
        if cycles_since_maintenance >= self.maintenance_interval:
            # Run maintenance
            self._run_field_maintenance()
            self.last_maintenance_cycle = self.brain_cycles
            performed['energy_dissipation'] = True
            performed['topology_cleanup'] = True
            performed['memory_trim'] = True
        
        return performed


# Factory function for easy creation
def create_unified_field_brain(spatial_resolution: int = 20,
                              temporal_window: float = 10.0,
                              field_evolution_rate: float = 0.1,
                              constraint_discovery_rate: float = 0.15,
                              quiet_mode: bool = False) -> UnifiedFieldBrain:
    """Create a unified field brain with standard configuration."""
    return UnifiedFieldBrain(
        spatial_resolution=spatial_resolution,
        temporal_window=temporal_window,
        field_evolution_rate=field_evolution_rate,
        constraint_discovery_rate=constraint_discovery_rate,
        quiet_mode=quiet_mode
    )


if __name__ == "__main__":
    # Test the unified field brain
    print("🧪 Testing Unified Field-Native Brain...")
    
    # Create field brain
    brain = create_unified_field_brain(
        spatial_resolution=15,
        temporal_window=8.0,
        field_evolution_rate=0.15,
        constraint_discovery_rate=0.2,
        quiet_mode=False
    )
    
    print(f"\n🌊 Testing field-native robot cycle processing:")
    
    # Test with simulated robot sensor data
    for cycle in range(10):
        # Simulate 24D robot sensors (position, distance sensors, camera, etc.)
        sensory_input = [
            0.5 + 0.1 * cycle,  # x position
            0.3 + 0.05 * cycle, # y position  
            0.1,                # z position
            0.8,                # distance sensor 1
            0.6,                # distance sensor 2
            0.4,                # distance sensor 3
            0.9 * (1 + 0.1 * math.sin(cycle * 0.5)),  # camera red
            0.7 * (1 + 0.1 * math.cos(cycle * 0.3)),  # camera green
            0.5 * (1 + 0.1 * math.sin(cycle * 0.7)),  # camera blue
            0.3,                # audio level
            0.2,                # temperature
            0.8,                # battery level
            0.9,                # wheel encoder 1
            0.85,               # wheel encoder 2
            0.1 * cycle,        # compass
            0.05 * cycle,       # gyroscope x
            0.02 * cycle,       # gyroscope y
            0.01 * cycle,       # gyroscope z
            0.7,                # accelerometer x
            0.3,                # accelerometer y
            1.0,                # accelerometer z
            0.5,                # touch sensor
            0.8,                # light sensor
            0.4                 # proximity sensor
        ]
        
        # Process through field brain
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        print(f"   Cycle {cycle+1}: motor_output={[f'{x:.3f}' for x in motor_output]}, "
              f"field_energy={brain_state['field_total_energy']:.3f}, "
              f"topology_regions={brain_state['topology_regions_count']}")
    
    # Display final field brain analysis
    print(f"\n📊 Final unified field brain analysis:")
    final_state = brain._get_field_brain_state(0)
    
    print(f"   🌊 Field Dynamics:")
    print(f"      Total dimensions: {final_state['field_dimensions']}")
    print(f"      Field max activation: {final_state['field_max_activation']:.4f}")
    print(f"      Field mean activation: {final_state['field_mean_activation']:.4f}")
    print(f"      Total field energy: {final_state['field_total_energy']:.3f}")
    
    print(f"\n   🏔️ Topology & Learning:")
    print(f"      Topology regions discovered: {final_state['topology_regions_count']}")
    print(f"      Total experiences: {final_state['field_experiences_count']}")
    print(f"      Topology discoveries: {final_state['topology_discoveries']}")
    
    print(f"\n   ⚡ Field Family Activities:")
    print(f"      Oscillatory (rhythm/frequency): {final_state['oscillatory_activity']:.4f}")
    print(f"      Flow (gradients/motion): {final_state['flow_activity']:.4f}")
    print(f"      Topology (stability/structure): {final_state['topology_activity']:.4f}")
    print(f"      Energy (intensity/activation): {final_state['energy_activity']:.4f}")
    print(f"      Coupling (correlations/binding): {final_state['coupling_activity']:.4f}")
    print(f"      Emergence (novelty/creativity): {final_state['emergence_activity']:.4f}")
    
    print(f"\n✅ Unified Field-Native Brain test completed!")
    print(f"🎯 Key field-native behaviors demonstrated:")
    print(f"   ✓ Robot sensors → unified field experience mapping")
    print(f"   ✓ Multi-dimensional field evolution and dynamics")
    print(f"   ✓ Topology region discovery (replaces pattern storage)")
    print(f"   ✓ Field gradient → robot action generation")
    print(f"   ✓ Dimension families organize by dynamics, not modality")
    print(f"   ✓ ALL intelligence emerges from unified field dynamics!")