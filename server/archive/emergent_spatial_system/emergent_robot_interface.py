#!/usr/bin/env python3
"""
Emergent Robot Interface

Maps between robot sensors/motors and field dynamics WITHOUT using coordinates.
Sensory patterns create field impressions directly, and motor commands emerge
from field evolution patterns.
"""

import torch
import numpy as np
import time
from typing import List, Tuple, Optional, Any, Dict
from collections import deque

from .field_types import UnifiedFieldExperience, FieldDynamicsFamily


class EmergentRobotInterface:
    """
    Translates between robot and field WITHOUT coordinate mappings.
    
    Key principles:
    1. Sensory patterns create unique field impressions
    2. Similar sensory patterns create similar field states
    3. Motor commands emerge from field dynamics
    4. No fixed spatial coordinates
    """
    
    def __init__(self, 
                 sensory_dim: int,
                 motor_dim: int,
                 field_dimensions: List[Any],
                 device: torch.device,
                 quiet_mode: bool = False):
        """
        Initialize emergent robot interface.
        
        Args:
            sensory_dim: Number of robot sensors
            motor_dim: Number of robot motors
            field_dimensions: Field dimension definitions
            device: Computation device
            quiet_mode: Suppress debug output
        """
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.field_dimensions = field_dimensions
        self.device = device
        self.quiet_mode = quiet_mode
        
        # Pattern memory for sensory-field mapping
        self.sensory_memory = deque(maxlen=100)
        self.pattern_clusters = {}
        
        # Motor emergence parameters
        self.motor_emergence_weights = self._initialize_motor_weights()
        
        # Statistics
        self.unique_patterns = 0
        self.pattern_matches = 0
        
        if not quiet_mode:
            print(f"🔄 Emergent Robot Interface initialized")
            print(f"   Sensors: {sensory_dim}D → Field impressions")
            print(f"   Motors: {motor_dim}D ← Field dynamics")
            print(f"   No coordinate mappings!")
    
    def sensory_pattern_to_field_experience(self, 
                                          sensory_input: List[float]) -> UnifiedFieldExperience:
        """
        Convert sensory pattern to field experience WITHOUT coordinates.
        
        The key innovation: sensory patterns create field impressions based on
        their intrinsic structure, not their mapping to spatial positions.
        """
        # Convert to tensor
        sensory_tensor = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        # Extract reward (last sensor by convention)
        reward = sensory_input[-1] if len(sensory_input) > 0 else 0.0
        
        # Create field impression from sensory pattern
        field_impression = self._create_field_impression(sensory_tensor)
        
        # Determine field intensity from pattern characteristics
        pattern_energy = torch.std(sensory_tensor).item()  # Variation indicates interest
        pattern_coherence = self._calculate_pattern_coherence(sensory_tensor)
        field_intensity = 0.5 + pattern_energy * 0.3 + pattern_coherence * 0.2
        
        # Create experience without coordinate mapping
        experience = UnifiedFieldExperience(
            timestamp=time.time(),
            raw_input_stream=sensory_tensor,  # Use correct field name
            field_coordinates=field_impression,  # This is now a pattern, not coordinates!
            field_intensity=field_intensity,
            dynamics_family_activations=self._get_dynamics_activations(field_impression)
        )
        
        # Track pattern for learning
        self._update_pattern_memory(sensory_tensor, field_impression)
        
        return experience
    
    def _create_field_impression(self, sensory_pattern: torch.Tensor) -> torch.Tensor:
        """
        Create a field impression from sensory pattern.
        
        This replaces coordinate mapping with pattern-based field activation.
        """
        # Initialize field impression
        field_impression = torch.zeros(len(self.field_dimensions), device=self.device)
        
        # Different aspects of the pattern activate different field dimensions
        pattern_features = self._extract_pattern_features(sensory_pattern)
        
        # Map features to field dimensions based on dynamics families
        dim_idx = 0
        for field_dim in self.field_dimensions:
            if field_dim.family == FieldDynamicsFamily.SPATIAL:
                # Spatial dimensions respond to pattern structure
                if 'symmetry' in pattern_features:
                    field_impression[dim_idx] = pattern_features['symmetry']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.OSCILLATORY:
                # Oscillatory dimensions respond to pattern rhythms
                if 'rhythm' in pattern_features:
                    field_impression[dim_idx] = pattern_features['rhythm']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.FLOW:
                # Flow dimensions respond to pattern gradients
                if 'gradient' in pattern_features:
                    field_impression[dim_idx] = pattern_features['gradient']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.TOPOLOGY:
                # Topology dimensions respond to pattern stability
                if 'stability' in pattern_features:
                    field_impression[dim_idx] = pattern_features['stability']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.ENERGY:
                # Energy dimensions respond to pattern intensity
                if 'intensity' in pattern_features:
                    field_impression[dim_idx] = pattern_features['intensity']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.COUPLING:
                # Coupling dimensions respond to pattern correlations
                if 'correlation' in pattern_features:
                    field_impression[dim_idx] = pattern_features['correlation']
                dim_idx += 1
            
            elif field_dim.family == FieldDynamicsFamily.EMERGENCE:
                # Emergence dimensions respond to pattern novelty
                if 'novelty' in pattern_features:
                    field_impression[dim_idx] = pattern_features['novelty']
                dim_idx += 1
        
        return field_impression
    
    def _get_dynamics_activations(self, field_impression: torch.Tensor) -> Dict[FieldDynamicsFamily, float]:
        """Calculate activation levels for each dynamics family."""
        activations = {}
        
        # Calculate activations based on field impression values
        dim_idx = 0
        # Initialize family_sums with all possible FieldDynamicsFamily values
        family_sums = {
            FieldDynamicsFamily.SPATIAL: [],
            FieldDynamicsFamily.OSCILLATORY: [],
            FieldDynamicsFamily.FLOW: [],
            FieldDynamicsFamily.TOPOLOGY: [],
            FieldDynamicsFamily.ENERGY: [],
            FieldDynamicsFamily.COUPLING: [],
            FieldDynamicsFamily.EMERGENCE: []
        }
        
        for field_dim in self.field_dimensions:
            if dim_idx < len(field_impression):
                # Use the family directly - it should already be an enum instance
                if field_dim.family in family_sums:
                    family_sums[field_dim.family].append(field_impression[dim_idx].item())
                dim_idx += 1
        
        # Average activations per family
        for family, values in family_sums.items():
            if values:
                activations[family] = sum(values) / len(values)
            else:
                activations[family] = 0.0
        
        return activations
    
    def _extract_pattern_features(self, pattern: torch.Tensor) -> dict:
        """Extract meaningful features from sensory pattern."""
        features = {}
        
        # Pattern symmetry (spatial organization)
        if len(pattern) >= 2:
            first_half = pattern[:len(pattern)//2]
            second_half = pattern[len(pattern)//2:]
            if len(first_half) == len(second_half):
                symmetry = 1.0 - torch.mean(torch.abs(first_half - second_half)).item()
                features['symmetry'] = symmetry
        
        # Pattern rhythm (temporal structure)
        if len(pattern) >= 4:
            diffs = torch.diff(pattern)
            rhythm = torch.std(diffs).item()
            features['rhythm'] = np.tanh(rhythm)  # Normalize
        
        # Pattern gradient (directional flow)
        if len(pattern) >= 2:
            gradient = torch.mean(torch.diff(pattern)).item()
            features['gradient'] = np.tanh(gradient)
        
        # Pattern stability (consistency)
        stability = 1.0 / (1.0 + torch.var(pattern).item())
        features['stability'] = stability
        
        # Pattern intensity (energy)
        intensity = torch.mean(torch.abs(pattern)).item()
        features['intensity'] = intensity
        
        # Pattern correlation (internal structure)
        if len(pattern) >= 3:
            correlation = self._calculate_autocorrelation(pattern)
            features['correlation'] = correlation
        
        # Pattern novelty (compared to memory)
        novelty = self._calculate_novelty(pattern)
        features['novelty'] = novelty
        
        return features
    
    def _calculate_pattern_coherence(self, pattern: torch.Tensor) -> float:
        """Calculate how coherent/structured a pattern is."""
        if len(pattern) < 2:
            return 0.5
        
        # Coherence based on autocorrelation
        autocorr = self._calculate_autocorrelation(pattern)
        
        # High autocorrelation = high coherence
        return autocorr
    
    def _calculate_autocorrelation(self, pattern: torch.Tensor) -> float:
        """Calculate pattern autocorrelation."""
        if len(pattern) < 2:
            return 0.0
        
        # Normalize pattern
        pattern_norm = pattern - torch.mean(pattern)
        if torch.std(pattern_norm) < 1e-6:
            return 0.0
        
        pattern_norm = pattern_norm / torch.std(pattern_norm)
        
        # Calculate autocorrelation at lag 1
        autocorr = torch.sum(pattern_norm[:-1] * pattern_norm[1:]) / (len(pattern) - 1)
        
        return autocorr.item()
    
    def _calculate_novelty(self, pattern: torch.Tensor) -> float:
        """Calculate how novel a pattern is compared to memory."""
        if len(self.sensory_memory) == 0:
            return 1.0
        
        # Compare to recent patterns
        min_similarity = 1.0
        for past_pattern, _ in self.sensory_memory:
            similarity = torch.nn.functional.cosine_similarity(
                pattern.unsqueeze(0),
                past_pattern.unsqueeze(0)
            ).item()
            min_similarity = min(min_similarity, similarity)
        
        # Novelty is inverse of similarity to past patterns
        novelty = 1.0 - min_similarity
        return novelty
    
    def _update_pattern_memory(self, sensory_pattern: torch.Tensor, field_impression: torch.Tensor):
        """Update pattern memory for learning."""
        self.sensory_memory.append((sensory_pattern.clone(), field_impression.clone()))
        
        # Check if this is a new unique pattern
        is_unique = True
        for past_pattern, _ in list(self.sensory_memory)[:-1]:
            similarity = torch.nn.functional.cosine_similarity(
                sensory_pattern.unsqueeze(0),
                past_pattern.unsqueeze(0)
            ).item()
            if similarity > 0.95:
                is_unique = False
                self.pattern_matches += 1
                break
        
        if is_unique:
            self.unique_patterns += 1
    
    def _initialize_motor_weights(self) -> dict:
        """Initialize weights for motor emergence from field dynamics."""
        return {
            'oscillatory_to_motor': torch.randn(self.motor_dim, 3, device=self.device) * 0.5,
            'flow_to_motor': torch.randn(self.motor_dim, 3, device=self.device) * 0.5,
            'energy_to_motor': torch.randn(self.motor_dim, 2, device=self.device) * 0.3,
            'topology_to_motor': torch.randn(self.motor_dim, 2, device=self.device) * 0.2
        }
    
    def get_statistics(self) -> dict:
        """Get interface statistics."""
        return {
            'unique_patterns': self.unique_patterns,
            'pattern_matches': self.pattern_matches,
            'pattern_diversity': self.unique_patterns / max(1, self.unique_patterns + self.pattern_matches)
        }