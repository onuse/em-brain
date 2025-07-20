#!/usr/bin/env python3
"""
Enhanced Field Dynamics

Implementation-agnostic field enhancements that work with any field implementation
through the FieldImplementation interface.

Key Features:
1. Phase Transitions - Dynamic regime changes based on field energy
2. Attractors/Repulsors - Stable/unstable field configurations
3. Energy Redistribution - Active field energy management
4. Field Coherence - Maintaining global field consistency
"""

import torch
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

try:
    from .field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience, 
        FieldNativeAction
    )
    from .adaptive_field_impl import FieldImplementation
except ImportError:
    from field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience,
        FieldNativeAction
    )
    from adaptive_field_impl import FieldImplementation


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transitions."""
    energy_threshold: float = 0.7        # Energy level triggering transitions
    stability_threshold: float = 0.3     # Stability requirement for transitions  
    transition_strength: float = 0.4     # Strength of transition effects
    coherence_factor: float = 0.6        # Global coherence influence
    decay_rate: float = 0.02             # Energy decay during transitions


@dataclass
class AttractorConfig:
    """Configuration for attractor/repulsor dynamics."""
    attractor_strength: float = 0.3      # Base attractor influence
    repulsor_strength: float = 0.2       # Base repulsor influence
    spatial_spread: float = 3.0          # Spatial influence radius
    temporal_persistence: float = 5.0    # How long attractors persist
    auto_discovery: bool = True          # Automatically discover attractors


class EnhancedFieldDynamics:
    """
    Enhanced field dynamics that work with any field implementation.
    
    Provides advanced field behaviors while maintaining implementation independence.
    """
    
    def __init__(self, field_impl: FieldImplementation, 
                 phase_config: Optional[PhaseTransitionConfig] = None,
                 attractor_config: Optional[AttractorConfig] = None,
                 quiet_mode: bool = False,
                 logger: Optional[Any] = None):
        
        self.field_impl = field_impl
        self.quiet_mode = quiet_mode
        self.logger = logger
        
        # Configuration
        self.phase_config = phase_config or PhaseTransitionConfig()
        self.attractor_config = attractor_config or AttractorConfig()
        
        # Phase transition tracking
        self.current_phase = "stable"
        self.phase_energy_history = []
        self.transition_cooldown = 0.0
        self.last_transition_time = 0.0
        
        # Attractor/repulsor management
        self.active_attractors = []  # List of attractor configurations
        self.active_repulsors = []   # List of repulsor configurations
        self.attractor_discovery_threshold = 0.8
        
        # Energy management
        self.global_energy_level = 0.0
        self.energy_flow_directions = {}
        self.coherence_metrics = {}
        
        if not quiet_mode:
            print(f"🌀 Enhanced Field Dynamics initialized for {field_impl.get_implementation_type()}")
    
    def evolve_with_enhancements(self, dt: float = 0.1, 
                               current_input_stream: Optional[List[float]] = None) -> None:
        """
        Evolve field with enhanced dynamics including phase transitions and attractors.
        """
        # 1. Update energy tracking
        self._update_energy_metrics()
        
        # 2. Detect and handle phase transitions
        self._handle_phase_transitions(dt)
        
        # 3. Apply attractor/repulsor dynamics
        self._apply_attractor_dynamics(dt)
        
        # 4. Manage energy redistribution
        self._redistribute_field_energy(dt)
        
        # 5. Standard field evolution
        self.field_impl.evolve_field(dt, current_input_stream)
        
        # 6. Update coherence metrics
        self._update_coherence_metrics()
        
        # 7. Auto-discover new attractors if enabled
        if self.attractor_config.auto_discovery:
            self._discover_new_attractors()
    
    def _update_energy_metrics(self) -> None:
        """Update global energy and flow metrics."""
        stats = self.field_impl.get_field_statistics()
        self.global_energy_level = stats.get('total_activation', 0.0)
        
        # Track energy history for phase detection
        self.phase_energy_history.append(self.global_energy_level)
        if len(self.phase_energy_history) > 20:  # Keep last 20 measurements
            self.phase_energy_history.pop(0)
        
        # Update energy flow from gradients
        gradients = self.field_impl.compute_field_gradients()
        self.energy_flow_directions = {}
        
        for grad_name, grad_tensor in gradients.items():
            if grad_tensor.numel() > 0:
                flow_magnitude = torch.mean(torch.abs(grad_tensor)).item()
                self.energy_flow_directions[grad_name] = flow_magnitude
    
    def _handle_phase_transitions(self, dt: float) -> None:
        """Detect and execute phase transitions based on field energy dynamics."""
        if self.transition_cooldown > 0:
            self.transition_cooldown -= dt
            return
        
        # Analyze energy stability
        if len(self.phase_energy_history) < 5:
            return
        
        recent_energy = self.phase_energy_history[-5:]
        
        # Protected variance calculation to prevent NaN and MPS float64 issues
        energy_tensor = torch.tensor(recent_energy, dtype=torch.float32)
        if len(energy_tensor) <= 1:
            energy_variance = 0.0
        else:
            variance_result = torch.var(energy_tensor.float())
            energy_variance = 0.0 if torch.isnan(variance_result) or torch.isinf(variance_result) else variance_result.item()
        
        energy_trend = recent_energy[-1] - recent_energy[0]
        
        # Determine if transition is needed
        transition_needed = False
        new_phase = self.current_phase
        
        if self.global_energy_level > self.phase_config.energy_threshold:
            if energy_variance < self.phase_config.stability_threshold:
                if self.current_phase != "high_energy":
                    new_phase = "high_energy"
                    transition_needed = True
            else:
                if self.current_phase != "chaotic":
                    new_phase = "chaotic"
                    transition_needed = True
        
        elif self.global_energy_level < 0.1:
            if self.current_phase != "low_energy":
                new_phase = "low_energy" 
                transition_needed = True
        
        else:
            if energy_variance < self.phase_config.stability_threshold:
                if self.current_phase != "stable":
                    new_phase = "stable"
                    transition_needed = True
        
        # Execute transition if needed
        if transition_needed:
            self._execute_phase_transition(new_phase, dt)
    
    def _execute_phase_transition(self, new_phase: str, dt: float) -> None:
        """Execute a phase transition with appropriate field modifications."""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.last_transition_time = time.time()
        self.transition_cooldown = 2.0  # Prevent rapid transitions
        
        if not self.quiet_mode:
            print(f"🔄 Phase transition: {old_phase} → {new_phase} (energy: {self.global_energy_level:.3f})")
        
        # Log phase transition for learning progress tracking
        if self.logger and hasattr(self.logger, 'log_phase_transition'):
            self.logger.log_phase_transition(new_phase)
        
        # Apply phase-specific field modifications
        transition_coords = self._generate_transition_coordinates()
        
        if new_phase == "high_energy":
            # Create stabilizing attractors (reduced intensity)
            self._create_phase_attractor(transition_coords, "stabilizing", 0.1)
        
        elif new_phase == "chaotic":
            # Introduce controlled perturbations (reduced intensity)
            self._create_phase_attractor(transition_coords, "perturbation", 0.05)
        
        elif new_phase == "low_energy":
            # Boost energy through activation (reduced intensity)
            self._create_phase_attractor(transition_coords, "energizing", 0.15)
        
        elif new_phase == "stable":
            # Gentle coherence enhancement (reduced intensity)
            self._create_phase_attractor(transition_coords, "coherence", 0.08)
    
    def _generate_transition_coordinates(self) -> torch.Tensor:
        """Generate field coordinates for phase transition."""
        # Use current field gradients to determine transition location
        gradients = self.field_impl.compute_field_gradients()
        
        if gradients:
            # Find region with highest gradient activity
            max_gradient = 0.0
            best_coords = None
            
            for grad_name, grad_tensor in gradients.items():
                grad_magnitude = torch.mean(torch.abs(grad_tensor)).item()
                if grad_magnitude > max_gradient:
                    max_gradient = grad_magnitude
                    
                    # Extract spatial coordinates from gradient peak
                    if grad_tensor.numel() > 0:
                        peak_indices = torch.argmax(torch.abs(grad_tensor.flatten()))
                        # Convert to normalized coordinates
                        coords = torch.randn(36, device=self.field_impl.field_device, dtype=torch.float32) * 0.1
                        coords[:3] = torch.rand(3, device=self.field_impl.field_device, dtype=torch.float32) * 2 - 1
                        best_coords = coords
            
            if best_coords is not None:
                return best_coords
        
        # Fallback: random coordinates with slight spatial bias
        coords = torch.randn(36, device=self.field_impl.field_device, dtype=torch.float32) * 0.3
        coords[:3] = torch.rand(3, device=self.field_impl.field_device, dtype=torch.float32) * 0.6 - 0.3
        return coords
    
    def _create_phase_attractor(self, coordinates: torch.Tensor, attractor_type: str, intensity: float) -> None:
        """Create a phase-specific attractor."""
        # ENERGY DEBUG: Log ALL attractor creation - this could be the energy source!
        coords_norm = torch.norm(coordinates).item()
        print(f"🔍 ATTRACTOR CREATE: type={attractor_type}, intensity={intensity:.6f}, coords_norm={coords_norm:.6f}")
        
        # CRITICAL: Check if attractor intensity is excessive
        if intensity > 0.5:
            print(f"   ⚠️  HIGH INTENSITY ATTRACTOR! This could cause energy spikes.")
        experience = UnifiedFieldExperience(
            timestamp=time.time(),
            field_coordinates=coordinates,
            raw_input_stream=torch.zeros(16, device=self.field_impl.field_device),
            field_intensity=intensity,
            dynamics_family_activations={
                FieldDynamicsFamily.ENERGY: intensity,
                FieldDynamicsFamily.TOPOLOGY: intensity * 0.7,
                FieldDynamicsFamily.EMERGENCE: intensity * 0.5
            }
        )
        
        self.field_impl.imprint_experience(experience)
        
        # Add to attractor tracking
        attractor_config = {
            'coordinates': coordinates,
            'type': attractor_type,
            'intensity': intensity,
            'creation_time': time.time(),
            'persistence': self.attractor_config.temporal_persistence
        }
        self.active_attractors.append(attractor_config)
    
    def _apply_attractor_dynamics(self, dt: float) -> None:
        """Apply influence from active attractors and repulsors."""
        current_time = time.time()
        
        # Remove expired attractors
        self.active_attractors = [
            attr for attr in self.active_attractors 
            if (current_time - attr['creation_time']) < attr['persistence']
        ]
        
        # Apply attractor influences
        for attractor in self.active_attractors:
            # Calculate time-based decay
            age = current_time - attractor['creation_time']
            decay_factor = math.exp(-age / attractor['persistence'])
            
            # Apply attractor influence with bounds checking
            current_intensity = attractor['intensity'] * decay_factor * self.attractor_config.attractor_strength
            # Prevent intensity amplification beyond biological range
            current_intensity = min(current_intensity, 1.0)
            
            if current_intensity > 0.01:  # Only apply if significant
                experience = UnifiedFieldExperience(
                    timestamp=current_time,
                    field_coordinates=attractor['coordinates'],
                    raw_input_stream=torch.zeros(16, device=self.field_impl.field_device),
                    field_intensity=current_intensity,
                    dynamics_family_activations={
                        FieldDynamicsFamily.TOPOLOGY: current_intensity,
                        FieldDynamicsFamily.FLOW: current_intensity * 0.5
                    }
                )
                
                self.field_impl.imprint_experience(experience)
    
    def _redistribute_field_energy(self, dt: float) -> None:
        """Actively redistribute field energy to maintain optimal dynamics."""
        if not self.energy_flow_directions:
            return
        
        # Find energy imbalances
        flow_magnitudes = list(self.energy_flow_directions.values())
        if not flow_magnitudes:
            return
        
        max_flow = max(flow_magnitudes)
        min_flow = min(flow_magnitudes)
        flow_imbalance = max_flow - min_flow
        
        # DISABLED: Energy redistribution creates new energy instead of redistributing
        # This is a major source of energy accumulation - commenting out until proper implementation
        # 
        # TODO: Implement true energy redistribution that moves energy instead of creating new energy
        # if flow_imbalance > 0.3:
        #     # Should implement: move energy from high-flow to low-flow regions
        #     # Current implementation: creates new energy (ENERGY LEAK!)
        #     pass
    
    def _update_coherence_metrics(self) -> None:
        """Update field coherence metrics for global consistency (optimized)."""
        gradients = self.field_impl.compute_field_gradients()
        
        self.coherence_metrics = {}
        
        if gradients:
            # OPTIMIZED: Use simple gradient magnitude consistency instead of expensive correlations
            gradient_magnitudes = []
            
            for grad_name, grad_tensor in gradients.items():
                if grad_tensor.numel() > 0:
                    # Use mean absolute gradient as coherence measure (much faster)
                    magnitude = torch.mean(torch.abs(grad_tensor)).item()
                    gradient_magnitudes.append(magnitude)
            
            if gradient_magnitudes:
                # Coherence = 1 - variance/mean (consistent magnitudes = high coherence)
                mean_magnitude = sum(gradient_magnitudes) / len(gradient_magnitudes)
                if mean_magnitude > 0:
                    variance = sum((m - mean_magnitude)**2 for m in gradient_magnitudes) / len(gradient_magnitudes)
                    coherence = max(0.0, 1.0 - (variance / (mean_magnitude**2 + 1e-6)))
                else:
                    coherence = 0.0
                
                self.coherence_metrics['gradient_coherence'] = coherence
            else:
                self.coherence_metrics['gradient_coherence'] = 0.0
            
            self.coherence_metrics['energy_coherence'] = min(1.0, self.global_energy_level / (self.phase_config.energy_threshold + 0.1))
    
    def _discover_new_attractors(self) -> None:
        """Automatically discover stable field regions and create attractors."""
        stats = self.field_impl.get_field_statistics()
        
        # Only discover during stable phases with sufficient energy
        if (self.current_phase in ["stable", "high_energy"] and 
            self.global_energy_level > self.attractor_discovery_threshold):
            
            # Limit number of active attractors
            if len(self.active_attractors) < 3:
                
                # Find regions of high stability in gradients
                gradients = self.field_impl.compute_field_gradients()
                
                for grad_name, grad_tensor in gradients.items():
                    if grad_tensor.numel() > 0:
                        grad_variance = torch.var(grad_tensor.float()).item()
                        grad_mean = torch.mean(torch.abs(grad_tensor)).item()
                        
                        # High activity + low variance = potential attractor
                        if grad_mean > 0.4 and grad_variance < 0.1:
                            discovery_coords = torch.randn(36, device=self.field_impl.field_device, dtype=torch.float32) * 0.2
                            
                            self._create_phase_attractor(
                                discovery_coords, 
                                "discovered", 
                                grad_mean * 0.5
                            )
                            
                            if not self.quiet_mode:
                                print(f"🎯 Discovered attractor in {grad_name} (strength: {grad_mean:.3f})")
                            break  # Only one discovery per evolution cycle
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about enhanced dynamics."""
        return {
            'current_phase': self.current_phase,
            'global_energy_level': self.global_energy_level,
            'active_attractors': len(self.active_attractors),
            'active_repulsors': len(self.active_repulsors),
            'energy_flow_count': len(self.energy_flow_directions),
            'coherence_metrics': self.coherence_metrics,
            'last_transition_time': self.last_transition_time,
            'transition_cooldown': self.transition_cooldown,
            'energy_history_length': len(self.phase_energy_history)
        }
    
    def manual_phase_transition(self, target_phase: str) -> bool:
        """Manually trigger a phase transition."""
        if self.transition_cooldown > 0:
            return False
        
        valid_phases = ["stable", "high_energy", "chaotic", "low_energy"]
        if target_phase not in valid_phases:
            return False
        
        self._execute_phase_transition(target_phase, 0.1)
        return True
    
    def add_manual_attractor(self, coordinates: torch.Tensor, intensity: float, 
                           persistence: Optional[float] = None) -> None:
        """Manually add an attractor at specified coordinates."""
        if persistence is None:
            persistence = self.attractor_config.temporal_persistence
        
        self._create_phase_attractor(coordinates, "manual", intensity)
        
        # Update the last attractor's persistence
        if self.active_attractors:
            self.active_attractors[-1]['persistence'] = persistence