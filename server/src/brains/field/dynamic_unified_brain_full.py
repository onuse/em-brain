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

from .core_brain import (
    FieldDimension, FieldDynamicsFamily, UnifiedFieldExperience,
    FieldNativeAction
)
from .dynamics.constraint_field_nd import ConstraintFieldND
from .dynamics.temporal_field_dynamics import TemporalExperience
from .optimized_gradients import create_optimized_gradient_calculator
from ...core.dynamic_dimension_calculator import DynamicDimensionCalculator
from .spontaneous_dynamics import SpontaneousDynamics
from ...utils.cognitive_autopilot import CognitiveAutopilot
from .blended_reality import integrate_blended_reality


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
                 field_evolution_rate: float = 0.1,
                 constraint_discovery_rate: float = 0.15,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """
        Initialize brain with dynamic dimensions and full feature set.
        """
        self.field_dimensions = field_dimensions
        self.tensor_shape = tensor_shape
        self.dimension_mapping = dimension_mapping
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        self.field_evolution_rate = field_evolution_rate
        self.constraint_discovery_rate = constraint_discovery_rate
        self.quiet_mode = quiet_mode
        
        # Total conceptual dimensions
        self.total_dimensions = len(field_dimensions)
        
        # Set device with MPS handling
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available() and len(tensor_shape) <= 16:
                # Avoid MPS for 11D tensors due to severe performance issues
                if len(tensor_shape) > 10:
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        # Create unified field with dynamic tensor shape
        self.unified_field = torch.zeros(tensor_shape, dtype=torch.float32, device=self.device)
        self.unified_field.fill_(0.0001)  # Baseline activation (prevents zero without interfering)
        
        # Initialize constraint dynamics
        self.constraint_field = ConstraintFieldND(
            field_shape=tensor_shape,
            dimension_names=[dim.name for dim in field_dimensions],
            constraint_discovery_rate=constraint_discovery_rate * 3,
            constraint_enforcement_strength=0.3,
            max_constraints=50,
            device=self.device,
            quiet_mode=quiet_mode
        )
        
        # Initialize optimized gradient calculator
        self.gradient_calculator = create_optimized_gradient_calculator(
            activation_threshold=0.01,
            cache_enabled=True,
            sparse_computation=True
        )
        
        # Field evolution parameters (from original brain)
        self.field_decay_rate = 0.95  # Increased decay to allow more persistent activation
        self.field_diffusion_rate = 0.05
        self.gradient_following_strength = 2.0  # Increased for stronger motor output
        self.topology_stability_threshold = 0.02
        self.field_energy_dissipation_rate = 0.90
        
        # Field state tracking
        self.field_experiences: deque = deque(maxlen=1000)
        self.field_actions: deque = deque(maxlen=1000)
        self.topology_regions: Dict[str, Dict] = {}
        self.gradient_flows: Dict[str, torch.Tensor] = {}
        
        # Temporal dynamics
        self.temporal_experiences: deque = deque(maxlen=int(temporal_window * 10))
        self.temporal_imprints: List[Any] = []
        self.working_memory: deque = deque(maxlen=50)
        
        # Motor smoothing
        self.previous_motor_commands = None
        self.motor_smoothing_factor = 0.3
        
        # Robot interface dimensions
        self.expected_sensory_dim = None
        self.expected_motor_dim = None
        
        # Performance tracking
        self.brain_cycles = 0
        self.field_evolution_cycles = 0
        self.topology_discoveries = 0
        self.gradient_actions = 0
        
        # Position tracking
        self._last_imprint_indices = None
        
        # Maintenance tracking
        self.last_maintenance_cycle = 0
        self.maintenance_interval = 100
        
        # Prediction system
        self._prediction_confidence_history = deque(maxlen=100)
        self._improvement_rate_history = deque(maxlen=50)
        self._current_prediction_confidence = 0.5
        self._predicted_field = None
        
        # Maintenance thread
        self._maintenance_queue = queue.Queue(maxsize=1)
        self._maintenance_thread = None
        self._shutdown_maintenance = threading.Event()
        self._start_maintenance_thread()
        
        # Spontaneous dynamics system
        self.spontaneous_enabled = True
        self.spontaneous = SpontaneousDynamics(
            field_shape=self.unified_field.shape,
            resting_potential=0.01,
            spontaneous_rate=0.001,
            coherence_scale=3.0,
            device=self.device
        )
        self._last_imprint_strength = 0.0
        self._last_prediction_error = 0.0
        
        # Cognitive autopilot for adaptive processing
        self.cognitive_autopilot = CognitiveAutopilot(
            autopilot_confidence_threshold=0.90,
            focused_confidence_threshold=0.70,
            stability_window=10
        )
        
        # Enable blended reality (confidence-based blending)
        self.blended_reality_enabled = True
        
        if not quiet_mode:
            print(f"🧠 Dynamic Unified Field Brain initialized (Full Features)")
            print(f"   Conceptual dimensions: {len(field_dimensions)}D")
            print(f"   Tensor shape: {tensor_shape} ({len(tensor_shape)}D)")
            print(f"   Memory usage: {self._calculate_memory_usage():.1f}MB")
            print(f"   Device: {self.device}")
            print(f"   Spontaneous dynamics: ENABLED")
            print(f"   Blended reality: ENABLED")
            self._print_dimension_summary()
    
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
        
        # Update cognitive autopilot state
        prediction_error = 0.0
        if hasattr(self, '_last_prediction_error'):
            prediction_error = self._last_prediction_error
        
        autopilot_state = self.cognitive_autopilot.update_cognitive_state(
            prediction_confidence=self._current_prediction_confidence,
            prediction_error=prediction_error,
            brain_state={'field_energy': float(torch.mean(torch.abs(self.unified_field)))}
        )
        
        # 3. Imprint experience in unified field
        self._imprint_unified_experience(field_experience)
        
        # 4. Add to temporal experiences
        self.temporal_experiences.append(field_experience)
        self.field_experiences.append(field_experience)
        
        # 5. Evolve unified field with all dynamics
        self._evolve_unified_field()
        
        # 6. Discover constraints from field topology
        if self.brain_cycles % 5 == 0:  # Discover constraints periodically
            self._discover_field_constraints()
        
        # 7. Apply constraint forces
        constraint_forces = self.constraint_field.enforce_constraints(self.unified_field)
        if constraint_forces is not None:
            self.unified_field += constraint_forces * 0.1
        
        # 8. Update topology regions (memory formation)
        if self.brain_cycles % 10 == 0:
            self._update_topology_regions()
        
        # 9. Generate motor commands from field gradients
        action = self._unified_field_to_robot_action(field_experience)
        self.field_actions.append(action)
        
        # 10. Create prediction for next cycle
        self._predicted_field = self._predict_next_field_state()
        
        # 11. Trigger maintenance if needed
        if self.brain_cycles - self.last_maintenance_cycle > self.maintenance_interval:
            self._trigger_maintenance()
        
        # Update tracking
        self.brain_cycles += 1
        cycle_time = time.perf_counter() - cycle_start
        
        # Create comprehensive brain state
        brain_state = {
            'cycle': self.brain_cycles,
            'cycle_time_ms': cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.unified_field))),
            'max_activation': float(torch.max(torch.abs(self.unified_field))),
            'active_constraints': len(self.constraint_field.active_constraints),
            'topology_regions': len(self.topology_regions),
            'prediction_confidence': self._current_prediction_confidence,
            'working_memory_size': len(self.working_memory),
            'conceptual_dims': self.total_dimensions,
            'tensor_dims': len(self.tensor_shape),
            'spontaneous_enabled': self.spontaneous_enabled,
            'cognitive_mode': self.cognitive_autopilot.current_mode.value,
            'cognitive_recommendations': self.cognitive_autopilot._generate_system_recommendations({})
        }
        
        return action.motor_commands.tolist(), brain_state
    
    def _robot_sensors_to_field_experience(self, sensory_input: List[float]) -> UnifiedFieldExperience:
        """
        Map robot sensors to unified field coordinates using dynamic mapping.
        """
        # Create field coordinates tensor
        field_coords = torch.zeros(self.total_dimensions, device=self.device)
        
        # Extract reward if present (last sensor by convention)
        reward = sensory_input[-1] if len(sensory_input) > 0 else 0.0
        field_intensity = 0.5 + reward * 0.5  # Map [-1,1] to [0,1]
        
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
            raw_sensory_input=torch.tensor(sensory_input, dtype=torch.float32, device=self.device),
            field_coordinates=field_coords,
            field_intensity=field_intensity,
            experience_id=f"exp_{self.brain_cycles}"
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
        
        # Imprint with intensity based on reward
        imprint_strength = 0.5 * experience.field_intensity  # Increased from 0.1 to 0.5
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
        # 1. Natural decay (memory fading)
        self.unified_field *= self.field_decay_rate
        
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
            if hasattr(self, 'blended_reality'):
                # Use blended reality system for weighting
                spontaneous_weight = self.blended_reality.calculate_spontaneous_weight()
                
                # sensory_gating: 0 = pure spontaneous, 1 = suppress spontaneous
                # So we invert our spontaneous weight
                sensory_gating = 1.0 - spontaneous_weight
            else:
                # Original implementation
                # Use prediction confidence for sensory gating
                # High confidence = less sensory attention (more spontaneous)
                # Low confidence = more sensory attention (less spontaneous)
                # Never completely ignore sensors - max suppression is 80%
                confidence_gating = self._current_prediction_confidence * 0.8
                
                # Also consider recent sensory strength (but weighted less)
                recency_gating = min(1.0, self._last_imprint_strength * 2.0) * 0.2
                
                # Combined gating: mostly confidence-based, slightly recency-based
                sensory_gating = confidence_gating + recency_gating
            
            # Add spontaneous activity
            spontaneous_activity = self.spontaneous.generate_spontaneous_activity(
                self.unified_field,
                sensory_gating=sensory_gating
            )
            self.unified_field += spontaneous_activity
        
        # 4. Ensure minimum baseline (prevents complete decay)
        self.unified_field = torch.maximum(self.unified_field, 
                                         torch.tensor(0.0001, device=self.device))
        
        self.field_evolution_cycles += 1
    
    def _discover_field_constraints(self):
        """
        Discover constraints from current field topology.
        """
        # Calculate field gradients for constraint discovery
        gradients = {}
        
        # Get spatial tensor dimensions
        spatial_range = self.dimension_mapping['family_tensor_ranges'].get(
            FieldDynamicsFamily.SPATIAL, (0, 3)
        )
        
        # Calculate gradients in spatial dimensions
        for dim in range(spatial_range[0], min(spatial_range[1], len(self.tensor_shape))):
            # Central differences
            grad = torch.zeros_like(self.unified_field)
            grad[1:-1] = (self.unified_field[2:] - self.unified_field[:-2]) / 2.0
            gradients[f'grad_dim_{dim}'] = grad
        
        # Discover constraints
        new_constraints = self.constraint_field.discover_constraints(
            self.unified_field, gradients
        )
        
        if new_constraints and not self.quiet_mode:
            print(f"🔍 Discovered {len(new_constraints)} new constraints")
    
    def _update_topology_regions(self):
        """
        Update memory regions based on field topology.
        """
        # Find high-activation regions
        high_activation = self.unified_field > 0.02
        
        # Find stable regions (low variance)
        field_std = torch.std(self.unified_field)
        stable_regions = self.unified_field > (field_std * 0.5)
        
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
            if len(self.topology_regions) > 100:
                # Remove oldest regions
                sorted_regions = sorted(self.topology_regions.items(), 
                                      key=lambda x: x[1]['last_activation'])
                for region_id, _ in sorted_regions[:20]:
                    del self.topology_regions[region_id]
    
    def _unified_field_to_robot_action(self, experience: UnifiedFieldExperience) -> FieldNativeAction:
        """
        Generate robot actions from unified field gradients.
        """
        # Get current position in tensor space
        tensor_indices = self._conceptual_to_tensor_indices(experience.field_coordinates)
        
        # Calculate gradients in spatial dimensions with local optimization
        spatial_dims = list(range(min(3, len(self.tensor_shape))))
        
        # Use local region optimization for efficiency
        region_center = tuple(tensor_indices[i] for i in range(min(3, len(tensor_indices))))
        all_gradients = self.gradient_calculator.calculate_gradients(
            field=self.unified_field,
            dimensions_to_compute=spatial_dims,
            local_region_only=True,
            region_center=region_center,
            region_size=3
        )
        
        # Extract gradients at current position
        gradients = []
        gradient_strength = 0.0
        
        for dim in spatial_dims:
            grad_key = f'gradient_dim_{dim}'
            if grad_key in all_gradients and dim < len(tensor_indices):
                # Get gradient tensor for this dimension
                grad_tensor = all_gradients[grad_key]
                
                # Extract gradient value at current position
                # Build index tuple matching the gradient tensor shape
                idx_list = []
                for i in range(len(grad_tensor.shape)):
                    if i < len(tensor_indices):
                        idx_list.append(min(tensor_indices[i], grad_tensor.shape[i] - 1))
                    else:
                        idx_list.append(0)
                
                try:
                    grad_val = float(grad_tensor[tuple(idx_list)])
                    gradients.append(grad_val)
                    gradient_strength += abs(grad_val)
                except:
                    gradients.append(0.0)
            else:
                gradients.append(0.0)
        
        gradient_strength = gradient_strength / max(1, len(gradients))
        
        # Initialize motor commands
        motor_dim = self.expected_motor_dim or 4
        motor_commands = torch.zeros(motor_dim, device=self.device)
        
        # Map gradients to motor commands
        # First 2 motors: X, Y movement
        if len(gradients) >= 2:
            motor_commands[0] = gradients[0] * self.gradient_following_strength
            motor_commands[1] = gradients[1] * self.gradient_following_strength
        
        # Additional motors can be mapped from other field dynamics
        if motor_dim > 2:
            # Use field energy in different regions for additional motors
            energy_dims = self.dimension_mapping['family_tensor_ranges'].get(
                FieldDynamicsFamily.ENERGY, (0, 0)
            )
            if energy_dims[1] > energy_dims[0]:
                energy_activation = torch.mean(
                    self.unified_field[tensor_indices[energy_dims[0]:energy_dims[1]]]
                )
                motor_commands[2] = (energy_activation - 0.5) * 0.5
        
        # Apply motor smoothing
        if self.previous_motor_commands is not None:
            motor_commands = (self.motor_smoothing_factor * self.previous_motor_commands + 
                            (1 - self.motor_smoothing_factor) * motor_commands)
        self.previous_motor_commands = motor_commands.clone()
        
        # Create action
        action = FieldNativeAction(
            timestamp=time.time(),
            field_gradients=torch.tensor(gradients, device=self.device),
            motor_commands=motor_commands,
            action_confidence=self._current_prediction_confidence,
            gradient_strength=gradient_strength
        )
        
        self.gradient_actions += 1
        
        return action
    
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
        if activation_magnitude < 0.01:
            # Near baseline - low confidence regardless of error
            self._current_prediction_confidence = 0.2
        else:
            # Normal confidence calculation based on error
            # Scale error by activation magnitude for better normalization
            normalized_error = prediction_error / (activation_magnitude + 0.1)
            self._current_prediction_confidence = float(1.0 / (1.0 + normalized_error * 2.0))
        
        self._prediction_confidence_history.append(self._current_prediction_confidence)
        self._last_prediction_error = float(prediction_error)
        
        # Track improvement rate
        if len(self._prediction_confidence_history) > 10:
            recent_confidence = np.mean(list(self._prediction_confidence_history)[-10:])
            older_confidence = np.mean(list(self._prediction_confidence_history)[-20:-10])
            improvement = recent_confidence - older_confidence
            self._improvement_rate_history.append(improvement)
    
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
    
    def _trigger_maintenance(self):
        """
        Trigger background maintenance.
        """
        try:
            self._maintenance_queue.put_nowait(self.brain_cycles)
            self.last_maintenance_cycle = self.brain_cycles
        except queue.Full:
            pass  # Maintenance already pending
    
    def _start_maintenance_thread(self):
        """
        Start the background maintenance thread.
        """
        def maintenance_worker():
            while not self._shutdown_maintenance.is_set():
                try:
                    # Wait for maintenance signal
                    cycle = self._maintenance_queue.get(timeout=1.0)
                    
                    # Perform maintenance
                    with torch.no_grad():
                        # Energy dissipation
                        self.unified_field *= self.field_energy_dissipation_rate
                        
                        # Prune weak activations
                        mask = torch.abs(self.unified_field) > 0.001
                        self.unified_field *= mask.float()
                        
                        # Reset to baseline where pruned
                        self.unified_field[~mask] = 0.0001
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    if not self.quiet_mode:
                        print(f"⚠️  Maintenance error: {e}")
        
        self._maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_thread.start()
    
    def __del__(self):
        """Cleanup when brain is destroyed."""
        if hasattr(self, '_shutdown_maintenance'):
            self._shutdown_maintenance.set()
        if hasattr(self, '_maintenance_thread') and self._maintenance_thread:
            self._maintenance_thread.join(timeout=1.0)