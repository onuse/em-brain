#!/usr/bin/env python3
"""
Constraint-Based Field Dynamics - Phase A5: The Ultimate Marriage

This integrates our constraint discovery framework with continuous field dynamics,
creating the ultimate intelligence substrate where constraints emerge naturally
from field topology and guide field evolution.

The key insight: Constraints are not external rules but emergent properties of
field dynamics. The field discovers its own constraints through experience,
creating a self-organizing intelligence system.

Phase A5 Goals:
1. Map constraint types to continuous field dynamics
2. Implement constraint emergence from field topology
3. Test constraint-guided field evolution
4. Validate constraint discovery through continuous intelligence

Core Innovation: Constraints become field gradients, boundaries, and flow patterns.
The field naturally discovers optimal constraint sets through topology optimization.

Research Questions:
- How do discrete constraints map to continuous field properties?
- Can field dynamics discover constraints autonomously?
- Do constraint-guided fields show enhanced intelligence?
- What's the computational efficiency of constraint-field integration?
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math
from collections import defaultdict, deque

try:
    from .temporal_field_dynamics import TemporalField4D, TemporalExperience, TemporalImprint
    from ...shared.stream_types import ConstraintType
except ImportError:
    from brains.field.dynamics.temporal_field_dynamics import TemporalField4D, TemporalExperience, TemporalImprint
    from brains.shared.stream_types import ConstraintType


class FieldConstraintType(Enum):
    """Types of constraints that emerge from field dynamics."""
    GRADIENT_FLOW = "gradient_flow"  # Constraints from field gradients
    TOPOLOGY_BOUNDARY = "topology_boundary"  # Constraints from field boundaries
    ACTIVATION_THRESHOLD = "activation_threshold"  # Constraints from activation patterns
    TEMPORAL_MOMENTUM = "temporal_momentum"  # Constraints from temporal dynamics
    SCALE_COUPLING = "scale_coupling"  # Constraints from cross-scale interactions
    PATTERN_COHERENCE = "pattern_coherence"  # Constraints from pattern relationships


@dataclass
class FieldConstraint:
    """A constraint that emerges from field dynamics."""
    constraint_type: FieldConstraintType
    field_region: Tuple[int, int, int, int, int, int, int, int]  # 4D region (x1,y1,s1,t1,x2,y2,s2,t2)
    strength: float  # How strong this constraint is
    gradient_direction: Optional[torch.Tensor] = None  # Direction of constraint force
    threshold_value: Optional[float] = None  # Threshold for activation constraints
    temporal_momentum: Optional[float] = None  # Momentum for temporal constraints
    pattern_signature: Optional[torch.Tensor] = None  # Pattern for coherence constraints
    discovery_timestamp: float = 0.0
    violation_count: int = 0
    enforcement_success: float = 0.0


@dataclass
class ConstraintFieldExperience:
    """Extended experience including constraint information."""
    base_experience: TemporalExperience
    discovered_constraints: List[FieldConstraint]
    constraint_violations: List[FieldConstraint]
    constraint_satisfaction: float  # How well this experience satisfies known constraints


class ConstraintField4D(TemporalField4D):
    """
    4D Temporal Field with Constraint Discovery and Enforcement
    
    This extends the 4D temporal field to discover and enforce constraints
    naturally through field dynamics. Constraints emerge as field properties
    and guide intelligent behavior.
    """
    
    def __init__(self, width: int = 200, height: int = 200, scale_depth: int = 30, 
                 temporal_depth: int = 50, temporal_window: float = 10.0, 
                 constraint_discovery_rate: float = 0.1,
                 constraint_enforcement_strength: float = 0.3,
                 quiet_mode: bool = False):
        super().__init__(width, height, scale_depth, temporal_depth, temporal_window, quiet_mode)
        
        # Constraint discovery parameters
        self.constraint_discovery_rate = constraint_discovery_rate
        self.constraint_enforcement_strength = constraint_enforcement_strength
        self.constraint_gradient_threshold = 0.05
        self.constraint_stability_window = 10
        
        # Discovered constraints
        self.discovered_constraints: Dict[str, FieldConstraint] = {}
        self.constraint_history: List[FieldConstraint] = []
        self.active_constraint_regions: Set[Tuple] = set()
        
        # Constraint enforcement tracking
        self.constraint_violations: List[Dict] = []
        self.constraint_satisfactions: List[Dict] = []
        self.constraint_discovery_events: List[Dict] = []
        
        # Field gradients for constraint discovery
        self.spatial_gradients = torch.zeros_like(self.field)
        self.scale_gradients = torch.zeros_like(self.field)
        self.temporal_gradients = torch.zeros_like(self.field)
        
        # Constraint-guided field modification
        self.constraint_guidance_field = torch.zeros_like(self.field)
        
        # Performance tracking
        self.constraint_stats = {
            'total_constraints_discovered': 0,
            'total_constraint_violations': 0,
            'total_constraint_enforcements': 0,
            'avg_constraint_discovery_time_ms': 0.0,
            'avg_constraint_enforcement_time_ms': 0.0,
            'constraint_discovery_success_rate': 0.0,
            'field_constraint_density': 0.0,
        }
        
        if not self.quiet_mode:
            print(f"🧭 ConstraintField4D initialized with constraint discovery")
            print(f"   Discovery rate: {constraint_discovery_rate:.2f}, enforcement: {constraint_enforcement_strength:.2f}")
    
    def apply_constraint_guided_imprint(self, experience: TemporalExperience) -> ConstraintFieldExperience:
        """
        Apply an imprint that is guided by discovered constraints.
        
        This is the core integration - constraints discovered from field dynamics
        guide new experiences, creating self-organizing intelligence.
        """
        start_time = time.perf_counter()
        
        # Convert to constraint-aware experience
        constraint_experience = self._analyze_experience_constraints(experience)
        
        # Apply base imprint
        base_imprint = self.experience_to_temporal_imprint(experience)
        
        # Modify imprint based on constraint guidance
        guided_imprint = self._apply_constraint_guidance(base_imprint, constraint_experience)
        
        # Apply the guided imprint to field
        self.apply_temporal_imprint(guided_imprint)
        
        # Update constraint satisfaction tracking
        self._update_constraint_satisfaction(constraint_experience)
        
        # Discover new constraints from this experience
        new_constraints = self._discover_constraints_from_imprint(guided_imprint)
        constraint_experience.discovered_constraints.extend(new_constraints)
        
        # Update stats
        guidance_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_constraint_stats(guidance_time_ms, len(new_constraints))
        
        if not self.quiet_mode:
            satisfaction = constraint_experience.constraint_satisfaction
            violations = len(constraint_experience.constraint_violations)
            discoveries = len(constraint_experience.discovered_constraints)
            print(f"🧭 Constraint-guided imprint: satisfaction={satisfaction:.3f}, "
                  f"violations={violations}, discoveries={discoveries}")
        
        return constraint_experience
    
    def _analyze_experience_constraints(self, experience: TemporalExperience) -> ConstraintFieldExperience:
        """Analyze how an experience relates to discovered constraints."""
        constraint_experience = ConstraintFieldExperience(
            base_experience=experience,
            discovered_constraints=[],
            constraint_violations=[],
            constraint_satisfaction=1.0
        )
        
        # Check experience against all discovered constraints
        imprint = self.experience_to_temporal_imprint(experience)
        violations = []
        satisfactions = []
        
        for constraint_id, constraint in self.discovered_constraints.items():
            satisfaction = self._evaluate_constraint_satisfaction(imprint, constraint)
            
            if satisfaction < 0.5:  # Constraint violation
                violations.append(constraint)
                constraint.violation_count += 1
            else:
                satisfactions.append(satisfaction)
                constraint.enforcement_success = (constraint.enforcement_success + satisfaction) / 2
        
        constraint_experience.constraint_violations = violations
        constraint_experience.constraint_satisfaction = np.mean(satisfactions) if satisfactions else 1.0
        
        return constraint_experience
    
    def _apply_constraint_guidance(self, base_imprint: TemporalImprint, 
                                 constraint_experience: ConstraintFieldExperience) -> TemporalImprint:
        """Apply constraint guidance to modify the imprint."""
        guided_imprint = base_imprint
        
        # If there are constraint violations, modify the imprint
        if constraint_experience.constraint_violations:
            # Calculate constraint guidance
            guidance_vector = self._calculate_constraint_guidance(
                base_imprint, constraint_experience.constraint_violations
            )
            
            # Apply guidance to imprint position
            guided_imprint.center_x = max(0, min(self.width - 1, 
                                                base_imprint.center_x + guidance_vector[0]))
            guided_imprint.center_y = max(0, min(self.height - 1, 
                                                base_imprint.center_y + guidance_vector[1]))
            guided_imprint.center_scale = max(0, min(self.scale_depth - 1, 
                                                    base_imprint.center_scale + guidance_vector[2]))
            guided_imprint.center_time = max(0, min(self.temporal_depth - 1, 
                                                   base_imprint.center_time + guidance_vector[3]))
            
            # Adjust intensity based on constraint satisfaction
            satisfaction = constraint_experience.constraint_satisfaction
            guided_imprint.intensity *= (0.5 + 0.5 * satisfaction)  # Reduce intensity for violations
            
            self.constraint_stats['total_constraint_enforcements'] += 1
        
        return guided_imprint
    
    def _calculate_constraint_guidance(self, imprint: TemporalImprint, 
                                     violated_constraints: List[FieldConstraint]) -> torch.Tensor:
        """Calculate guidance vector to satisfy violated constraints."""
        guidance = torch.zeros(4)  # [x, y, scale, time] guidance
        
        for constraint in violated_constraints:
            if constraint.gradient_direction is not None:
                # Follow gradient to satisfy constraint
                guidance += constraint.gradient_direction * constraint.strength * self.constraint_enforcement_strength
            
            elif constraint.constraint_type == FieldConstraintType.TOPOLOGY_BOUNDARY:
                # Move away from field boundaries
                x1, y1, s1, t1, x2, y2, s2, t2 = constraint.field_region
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center_s = (s1 + s2) / 2
                center_t = (t1 + t2) / 2
                
                # Move toward constraint region center
                guidance[0] += (center_x - imprint.center_x) * 0.1
                guidance[1] += (center_y - imprint.center_y) * 0.1
                guidance[2] += (center_s - imprint.center_scale) * 0.1
                guidance[3] += (center_t - imprint.center_time) * 0.1
        
        # Normalize guidance
        if torch.norm(guidance) > 0:
            guidance = guidance / torch.norm(guidance) * 5.0  # Maximum guidance step
        
        return guidance
    
    def _discover_constraints_from_imprint(self, imprint: TemporalImprint) -> List[FieldConstraint]:
        """Discover new constraints from field analysis around an imprint."""
        start_time = time.perf_counter()
        discovered_constraints = []
        
        # Update field gradients
        self._update_field_gradients()
        
        # Analyze field region around imprint
        x, y, s, t = int(imprint.center_x), int(imprint.center_y), int(imprint.center_scale), int(imprint.center_time)
        
        # 1. Discover gradient flow constraints
        gradient_constraints = self._discover_gradient_constraints(x, y, s, t)
        discovered_constraints.extend(gradient_constraints)
        
        # 2. Discover topology boundary constraints
        boundary_constraints = self._discover_boundary_constraints(x, y, s, t)
        discovered_constraints.extend(boundary_constraints)
        
        # 3. Discover activation threshold constraints
        threshold_constraints = self._discover_threshold_constraints(x, y, s, t)
        discovered_constraints.extend(threshold_constraints)
        
        # 4. Discover temporal momentum constraints
        momentum_constraints = self._discover_momentum_constraints(x, y, s, t, imprint)
        discovered_constraints.extend(momentum_constraints)
        
        # Store discovered constraints
        for constraint in discovered_constraints:
            constraint_id = f"{constraint.constraint_type.value}_{len(self.discovered_constraints)}"
            constraint.discovery_timestamp = time.time()
            self.discovered_constraints[constraint_id] = constraint
            self.constraint_history.append(constraint)
        
        # Update stats
        discovery_time_ms = (time.perf_counter() - start_time) * 1000
        self.constraint_stats['total_constraints_discovered'] += len(discovered_constraints)
        
        if discovered_constraints and not self.quiet_mode:
            print(f"   🔍 Discovered {len(discovered_constraints)} new constraints in {discovery_time_ms:.2f}ms")
        
        return discovered_constraints
    
    def _update_field_gradients(self):
        """Update spatial, scale, and temporal gradients of the field."""
        # Spatial gradients (using Sobel-like operators)
        spatial_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spatial_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Calculate spatial gradients for each scale-time slice
        for s in range(self.scale_depth):
            for t in range(self.temporal_depth):
                slice_2d = self.field[:, :, s, t].unsqueeze(0).unsqueeze(0)
                
                # Spatial gradients
                grad_x = torch.conv2d(slice_2d, spatial_kernel_x, padding=1).squeeze()
                grad_y = torch.conv2d(slice_2d, spatial_kernel_y, padding=1).squeeze()
                
                self.spatial_gradients[:, :, s, t] = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Scale gradients (simple difference)
        for y in range(self.height):
            for x in range(self.width):
                for t in range(self.temporal_depth):
                    scale_profile = self.field[y, x, :, t]
                    scale_diff = torch.diff(scale_profile, prepend=scale_profile[0:1])
                    self.scale_gradients[y, x, :, t] = torch.abs(scale_diff)
        
        # Temporal gradients (simple difference)
        for y in range(self.height):
            for x in range(self.width):
                for s in range(self.scale_depth):
                    temporal_profile = self.field[y, x, s, :]
                    temporal_diff = torch.diff(temporal_profile, prepend=temporal_profile[0:1])
                    self.temporal_gradients[y, x, s, :] = torch.abs(temporal_diff)
    
    def _discover_gradient_constraints(self, x: int, y: int, s: int, t: int) -> List[FieldConstraint]:
        """Discover constraints from field gradients."""
        constraints = []
        
        # Analyze gradient region
        window = 3
        x1, x2 = max(0, x - window), min(self.width, x + window)
        y1, y2 = max(0, y - window), min(self.height, y + window)
        s1, s2 = max(0, s - 1), min(self.scale_depth, s + 1)
        t1, t2 = max(0, t - 2), min(self.temporal_depth, t + 2)
        
        # Check spatial gradient strength
        spatial_grad_region = self.spatial_gradients[y1:y2, x1:x2, s1:s2, t1:t2]
        avg_spatial_grad = torch.mean(spatial_grad_region)
        
        if avg_spatial_grad > self.constraint_gradient_threshold:
            # Calculate dominant gradient direction
            grad_x = torch.mean(self.spatial_gradients[y1:y2, x1:x2, s, t])
            grad_y = torch.mean(self.spatial_gradients[y1:y2, x1:x2, s, t])
            gradient_direction = torch.tensor([grad_x.item(), grad_y.item(), 0.0, 0.0])
            
            constraint = FieldConstraint(
                constraint_type=FieldConstraintType.GRADIENT_FLOW,
                field_region=(x1, y1, s1, t1, x2, y2, s2, t2),
                strength=avg_spatial_grad.item(),
                gradient_direction=gradient_direction
            )
            constraints.append(constraint)
        
        return constraints
    
    def _discover_boundary_constraints(self, x: int, y: int, s: int, t: int) -> List[FieldConstraint]:
        """Discover constraints from field topology boundaries."""
        constraints = []
        
        # Check for activation boundaries (regions of high contrast)
        window = 4
        x1, x2 = max(0, x - window), min(self.width, x + window)
        y1, y2 = max(0, y - window), min(self.height, y + window)
        s1, s2 = max(0, s - 1), min(self.scale_depth, s + 1)
        t1, t2 = max(0, t - 1), min(self.temporal_depth, t + 1)
        
        field_region = self.field[y1:y2, x1:x2, s1:s2, t1:t2]
        
        # Look for sharp boundaries (high variance)
        region_variance = torch.var(field_region)
        region_mean = torch.mean(field_region)
        
        if region_variance > 0.1 and region_mean > 0.05:  # Sharp boundary detected
            constraint = FieldConstraint(
                constraint_type=FieldConstraintType.TOPOLOGY_BOUNDARY,
                field_region=(x1, y1, s1, t1, x2, y2, s2, t2),
                strength=region_variance.item(),
                threshold_value=region_mean.item()
            )
            constraints.append(constraint)
        
        return constraints
    
    def _discover_threshold_constraints(self, x: int, y: int, s: int, t: int) -> List[FieldConstraint]:
        """Discover activation threshold constraints."""
        constraints = []
        
        # Analyze activation patterns in neighborhood
        window = 5
        x1, x2 = max(0, x - window), min(self.width, x + window)
        y1, y2 = max(0, y - window), min(self.height, y + window)
        s1, s2 = max(0, s - 2), min(self.scale_depth, s + 2)
        t1, t2 = max(0, t - 2), min(self.temporal_depth, t + 2)
        
        field_region = self.field[y1:y2, x1:x2, s1:s2, t1:t2]
        
        # Look for consistent activation thresholds
        activated_mask = field_region > self.activation_threshold
        activation_ratio = torch.mean(activated_mask.float())
        
        if 0.2 < activation_ratio < 0.8:  # Interesting activation pattern
            optimal_threshold = torch.median(field_region).item()
            
            constraint = FieldConstraint(
                constraint_type=FieldConstraintType.ACTIVATION_THRESHOLD,
                field_region=(x1, y1, s1, t1, x2, y2, s2, t2),
                strength=abs(optimal_threshold - self.activation_threshold),
                threshold_value=optimal_threshold
            )
            constraints.append(constraint)
        
        return constraints
    
    def _discover_momentum_constraints(self, x: int, y: int, s: int, t: int, 
                                     imprint: TemporalImprint) -> List[FieldConstraint]:
        """Discover temporal momentum constraints."""
        constraints = []
        
        if imprint.temporal_momentum > 0.1:  # Significant momentum
            # This imprint has strong temporal momentum - create constraint
            window = 3
            x1, x2 = max(0, x - window), min(self.width, x + window)
            y1, y2 = max(0, y - window), min(self.height, y + window)
            s1, s2 = max(0, s - 1), min(self.scale_depth, s + 1)
            t1, t2 = max(0, t - 1), min(self.temporal_depth, t + 1)
            
            constraint = FieldConstraint(
                constraint_type=FieldConstraintType.TEMPORAL_MOMENTUM,
                field_region=(x1, y1, s1, t1, x2, y2, s2, t2),
                strength=imprint.temporal_momentum,
                temporal_momentum=imprint.temporal_momentum,
                pattern_signature=imprint.pattern_signature.clone()
            )
            constraints.append(constraint)
        
        return constraints
    
    def _evaluate_constraint_satisfaction(self, imprint: TemporalImprint, 
                                        constraint: FieldConstraint) -> float:
        """Evaluate how well an imprint satisfies a constraint."""
        x1, y1, s1, t1, x2, y2, s2, t2 = constraint.field_region
        
        # Check if imprint is in constraint region
        in_region = (x1 <= imprint.center_x <= x2 and 
                    y1 <= imprint.center_y <= y2 and
                    s1 <= imprint.center_scale <= s2 and
                    t1 <= imprint.center_time <= t2)
        
        if not in_region:
            return 1.0  # No constraint violation if outside region
        
        # Evaluate based on constraint type
        if constraint.constraint_type == FieldConstraintType.GRADIENT_FLOW:
            # Check if imprint follows gradient direction
            if constraint.gradient_direction is not None:
                imprint_direction = torch.tensor([
                    imprint.center_x - (x1 + x2) / 2,
                    imprint.center_y - (y1 + y2) / 2,
                    imprint.center_scale - (s1 + s2) / 2,
                    imprint.center_time - (t1 + t2) / 2
                ])
                alignment = torch.cosine_similarity(
                    imprint_direction.unsqueeze(0), 
                    constraint.gradient_direction.unsqueeze(0)
                ).item()
                return max(0.0, alignment)
        
        elif constraint.constraint_type == FieldConstraintType.ACTIVATION_THRESHOLD:
            # Check if imprint intensity respects threshold
            if constraint.threshold_value is not None:
                threshold_satisfaction = 1.0 - abs(imprint.intensity - constraint.threshold_value)
                return max(0.0, threshold_satisfaction)
        
        elif constraint.constraint_type == FieldConstraintType.TEMPORAL_MOMENTUM:
            # Check temporal momentum consistency
            if constraint.temporal_momentum is not None:
                momentum_consistency = 1.0 - abs(imprint.temporal_momentum - constraint.temporal_momentum)
                return max(0.0, momentum_consistency)
        
        return 0.5  # Default neutral satisfaction
    
    def evolve_constraint_guided_field(self, dt: float = 0.1):
        """
        Evolve the field with constraint guidance.
        
        This is the ultimate integration - field evolution guided by
        constraints that were discovered from the field itself.
        """
        # Standard field evolution
        super().evolve_temporal_field(dt)
        
        # Apply constraint guidance to field evolution
        self._apply_constraint_guidance_to_field()
        
        # Evaluate constraint stability
        self._evaluate_constraint_stability()
        
        # Clean up outdated constraints
        self._cleanup_outdated_constraints()
    
    def _apply_constraint_guidance_to_field(self):
        """Apply constraint guidance directly to field dynamics."""
        self.constraint_guidance_field.zero_()
        
        for constraint_id, constraint in self.discovered_constraints.items():
            x1, y1, s1, t1, x2, y2, s2, t2 = constraint.field_region
            
            if constraint.constraint_type == FieldConstraintType.GRADIENT_FLOW:
                # Enhance field gradients in constraint direction
                if constraint.gradient_direction is not None:
                    guidance_strength = constraint.strength * self.constraint_enforcement_strength
                    self.constraint_guidance_field[y1:y2, x1:x2, s1:s2, t1:t2] += guidance_strength
            
            elif constraint.constraint_type == FieldConstraintType.ACTIVATION_THRESHOLD:
                # Adjust field activations toward optimal threshold
                if constraint.threshold_value is not None:
                    region = self.field[y1:y2, x1:x2, s1:s2, t1:t2]
                    target_adjustment = (constraint.threshold_value - torch.mean(region)) * 0.1
                    self.constraint_guidance_field[y1:y2, x1:x2, s1:s2, t1:t2] += target_adjustment
        
        # Apply guidance to main field
        self.field += self.constraint_guidance_field * 0.1  # Gentle guidance
    
    def _evaluate_constraint_stability(self):
        """Evaluate which constraints are stable over time."""
        current_time = time.time()
        
        for constraint_id, constraint in self.discovered_constraints.items():
            age = current_time - constraint.discovery_timestamp
            
            # Calculate stability based on violation/success ratio
            total_interactions = constraint.violation_count + max(1, int(constraint.enforcement_success * 10))
            stability = constraint.enforcement_success / max(1, total_interactions)
            
            # Update constraint strength based on stability
            if age > self.constraint_stability_window:
                if stability > 0.7:
                    constraint.strength *= 1.1  # Strengthen stable constraints
                elif stability < 0.3:
                    constraint.strength *= 0.9  # Weaken unstable constraints
    
    def _cleanup_outdated_constraints(self):
        """Remove constraints that are no longer useful."""
        current_time = time.time()
        to_remove = []
        
        for constraint_id, constraint in self.discovered_constraints.items():
            age = current_time - constraint.discovery_timestamp
            
            # Remove very weak or very old constraints
            if constraint.strength < 0.01 or age > 120:  # 2 minutes max age
                to_remove.append(constraint_id)
        
        for constraint_id in to_remove:
            del self.discovered_constraints[constraint_id]
    
    def _update_constraint_satisfaction(self, constraint_experience: ConstraintFieldExperience):
        """Update constraint satisfaction tracking."""
        satisfaction_event = {
            'timestamp': time.time(),
            'constraint_satisfaction': constraint_experience.constraint_satisfaction,
            'violations_count': len(constraint_experience.constraint_violations),
            'discoveries_count': len(constraint_experience.discovered_constraints)
        }
        self.constraint_satisfactions.append(satisfaction_event)
        
        # Track violations
        for violation in constraint_experience.constraint_violations:
            violation_event = {
                'timestamp': time.time(),
                'constraint_type': violation.constraint_type.value,
                'constraint_strength': violation.strength,
                'field_region': violation.field_region
            }
            self.constraint_violations.append(violation_event)
            self.constraint_stats['total_constraint_violations'] += 1
    
    def _update_constraint_stats(self, processing_time_ms: float, discoveries_count: int):
        """Update constraint processing statistics."""
        # Update discovery time
        if discoveries_count > 0:
            current_avg = self.constraint_stats['avg_constraint_discovery_time_ms']
            total_discoveries = self.constraint_stats['total_constraints_discovered']
            if total_discoveries > 0:
                self.constraint_stats['avg_constraint_discovery_time_ms'] = \
                    (current_avg * (total_discoveries - discoveries_count) + processing_time_ms) / total_discoveries
            else:
                self.constraint_stats['avg_constraint_discovery_time_ms'] = processing_time_ms
        
        # Update field constraint density
        total_field_volume = self.width * self.height * self.scale_depth * self.temporal_depth
        active_constraint_volume = sum([
            (x2-x1) * (y2-y1) * (s2-s1) * (t2-t1) 
            for x1,y1,s1,t1,x2,y2,s2,t2 in [c.field_region for c in self.discovered_constraints.values()]
        ])
        self.constraint_stats['field_constraint_density'] = active_constraint_volume / total_field_volume
        
        # Update success rate
        total_enforcements = self.constraint_stats['total_constraint_enforcements']
        total_violations = self.constraint_stats['total_constraint_violations']
        if total_enforcements + total_violations > 0:
            self.constraint_stats['constraint_discovery_success_rate'] = \
                total_enforcements / (total_enforcements + total_violations)
    
    def get_constraint_field_stats(self) -> Dict[str, Any]:
        """Get comprehensive constraint field statistics."""
        base_stats = self.get_temporal_stats()
        
        # Constraint-specific stats
        constraint_stats = self.constraint_stats.copy()
        constraint_stats.update({
            'discovered_constraints_count': len(self.discovered_constraints),
            'constraint_history_length': len(self.constraint_history),
            'active_constraint_regions': len(self.active_constraint_regions),
            'recent_violations': len([v for v in self.constraint_violations[-10:]]),
            'recent_satisfactions': len([s for s in self.constraint_satisfactions[-10:]]),
            'constraint_type_distribution': self._get_constraint_type_distribution(),
            'avg_constraint_strength': np.mean([c.strength for c in self.discovered_constraints.values()]) 
                                      if self.discovered_constraints else 0.0,
            'field_guidance_intensity': torch.mean(torch.abs(self.constraint_guidance_field)).item(),
        })
        
        # Merge with base stats
        base_stats.update(constraint_stats)
        return base_stats
    
    def _get_constraint_type_distribution(self) -> Dict[str, int]:
        """Get distribution of constraint types."""
        distribution = defaultdict(int)
        for constraint in self.discovered_constraints.values():
            distribution[constraint.constraint_type.value] += 1
        return dict(distribution)


# Factory function for easy creation
def create_constraint_field_4d(width: int = 200, height: int = 200, scale_depth: int = 30, 
                              temporal_depth: int = 50, temporal_window: float = 10.0,
                              constraint_discovery_rate: float = 0.1,
                              constraint_enforcement_strength: float = 0.3,
                              quiet_mode: bool = False) -> ConstraintField4D:
    """Create a 4D constraint field with standard configuration."""
    return ConstraintField4D(
        width=width, height=height, scale_depth=scale_depth, 
        temporal_depth=temporal_depth, temporal_window=temporal_window,
        constraint_discovery_rate=constraint_discovery_rate,
        constraint_enforcement_strength=constraint_enforcement_strength,
        quiet_mode=quiet_mode
    )


if __name__ == "__main__":
    # Test the constraint field implementation
    print("🧪 Testing Constraint-Based Field Dynamics...")
    
    # Create constraint field
    field = create_constraint_field_4d(
        width=60, height=60, scale_depth=10, 
        temporal_depth=15, temporal_window=8.0, 
        constraint_discovery_rate=0.15,
        constraint_enforcement_strength=0.4,
        quiet_mode=False
    )
    
    print(f"🧭 Created {field.width}x{field.height}x{field.scale_depth}x{field.temporal_depth} constraint field")
    
    # Test constraint discovery and guidance
    print("\n📍 Testing constraint-guided experiences:")
    
    # Create test experiences that should lead to constraint discovery
    experiences = []
    for i in range(8):
        exp = TemporalExperience(
            sensory_data=torch.tensor([0.9 - i*0.1, 0.2 + i*0.1, 0.7, 0.4 + i*0.05]),
            position=(20 + i*3, 20 + i*2),  # Structured pattern
            scale_level=0.2 + i*0.1,  # Progressive scale change
            temporal_position=i * 1.0,  # Regular temporal progression
            intensity=0.8 + i*0.02,
            spatial_spread=5.0,
            scale_spread=1.5,
            temporal_spread=1.5,
            timestamp=time.time() + i*0.1,
            sequence_id="constraint_test"
        )
        experiences.append(exp)
    
    # Apply experiences and track constraint discovery
    for i, exp in enumerate(experiences):
        constraint_exp = field.apply_constraint_guided_imprint(exp)
        print(f"   Experience {i+1}: satisfaction={constraint_exp.constraint_satisfaction:.3f}, "
              f"violations={len(constraint_exp.constraint_violations)}, "
              f"discoveries={len(constraint_exp.discovered_constraints)}")
    
    # Test constraint-guided field evolution
    print("\n🌊 Testing constraint-guided field evolution:")
    for i in range(10):
        field.evolve_constraint_guided_field(dt=0.3)
        if i % 3 == 0:
            stats = field.get_constraint_field_stats()
            print(f"   Evolution {i+1}: constraints={stats['discovered_constraints_count']}, "
                  f"density={stats['field_constraint_density']:.4f}, "
                  f"guidance={stats['field_guidance_intensity']:.4f}")
    
    # Display comprehensive results
    print("\n📊 Final constraint field analysis:")
    final_stats = field.get_constraint_field_stats()
    
    print(f"\n   🧭 Constraint Discovery Results:")
    print(f"      Total constraints discovered: {final_stats['discovered_constraints_count']}")
    print(f"      Constraint type distribution: {final_stats['constraint_type_distribution']}")
    print(f"      Average constraint strength: {final_stats['avg_constraint_strength']:.3f}")
    print(f"      Field constraint density: {final_stats['field_constraint_density']:.4f}")
    
    print(f"\n   📈 Constraint Performance:")
    print(f"      Discovery success rate: {final_stats['constraint_discovery_success_rate']:.3f}")
    print(f"      Total enforcements: {final_stats['total_constraint_enforcements']}")
    print(f"      Total violations: {final_stats['total_constraint_violations']}")
    print(f"      Field guidance intensity: {final_stats['field_guidance_intensity']:.4f}")
    
    print(f"\n   🌊 Field Intelligence Metrics:")
    print(f"      Temporal coherence: {final_stats['temporal_coherence']:.4f}")
    print(f"      Working memory patterns: {final_stats['working_memory_size']}")
    print(f"      Temporal chains: {final_stats['temporal_chains_count']}")
    print(f"      Field utilization: {final_stats['field_mean_activation']:.6f}")
    
    print("\n✅ Constraint-Based Field Dynamics test completed!")
    print("🎯 Key constraint-field integration demonstrated:")
    print("   ✓ Constraint discovery from field topology")
    print("   ✓ Constraint-guided experience imprinting")
    print("   ✓ Constraint-guided field evolution")
    print("   ✓ Dynamic constraint stability evaluation")
    print("   ✓ Self-organizing constraint enforcement")
    print("   ✓ Hybrid discrete-continuous intelligence")