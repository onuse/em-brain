"""
Topology Region System for 4D Field Architecture

Reimplementation of topology regions adapted for the simplified 4D tensor brain.
Enables abstraction formation, causal tracking, and persistent memory through
stable field configurations.

Key improvements from archive:
1. Simplified coordinate system (4D instead of 37D)
2. Efficient region detection using field patterns
3. Causal relationship tracking between regions
4. Integration with consolidation system
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import torch.nn.functional as F


@dataclass
class TopologyRegion:
    """Represents a stable topology region in the field."""
    region_id: str
    spatial_center: Tuple[int, int, int]  # Center in first 3 dims
    feature_center: int  # Center in 4th dimension
    
    # Pattern characteristics
    activation_pattern: torch.Tensor  # The actual pattern
    mean_activation: float
    stability: float  # Inverse of variance
    coherence: float  # Spatial organization
    
    # Temporal tracking
    discovery_time: float
    last_activation: float
    activation_count: int = 0
    total_strength: float = 0.0  # Cumulative activation strength
    
    # Importance and persistence
    importance: float = 1.0
    consolidation_level: int = 0
    decay_rate: float = 0.995
    
    # Relationships
    associated_regions: Set[str] = field(default_factory=set)
    causal_predecessors: Dict[str, float] = field(default_factory=dict)  # region_id -> strength
    causal_successors: Dict[str, float] = field(default_factory=dict)
    
    # Abstraction level
    abstraction_level: int = 0  # 0 = concrete, higher = more abstract
    component_regions: Set[str] = field(default_factory=set)  # For hierarchical composition
    
    # Sensory prediction capabilities
    is_sensory_predictive: bool = False
    sensor_indices: List[int] = field(default_factory=list)  # Which sensors this region predicts
    sensory_prediction_history: deque = field(default_factory=lambda: deque(maxlen=50))
    prediction_confidence: float = 0.0
    prediction_momentum: torch.Tensor = None  # Temporal momentum for predictions
    
    def update_activation(self, strength: float):
        """Update region activation statistics."""
        self.last_activation = time.time()
        self.activation_count += 1
        self.total_strength += strength
        self.importance *= 1.1  # Boost importance on activation
        
    def add_causal_link(self, predecessor_id: Optional[str], successor_id: Optional[str], strength: float = 0.1):
        """Add causal relationship to another region."""
        if predecessor_id:
            self.causal_predecessors[predecessor_id] = (
                self.causal_predecessors.get(predecessor_id, 0.0) + strength
            )
        if successor_id:
            self.causal_successors[successor_id] = (
                self.causal_successors.get(successor_id, 0.0) + strength
            )
    
    def predict_from_field(self, field: torch.Tensor, temporal_field: torch.Tensor) -> torch.Tensor:
        """Generate sensory predictions based on field state and temporal features."""
        if not self.is_sensory_predictive or len(self.sensor_indices) == 0:
            return torch.zeros(len(self.sensor_indices))
        
        # Extract local temporal features around this region
        x, y, z = self.spatial_center
        f = self.feature_center
        
        # Get temporal features in local neighborhood
        x_min, x_max = max(0, x-2), min(field.shape[0], x+3)
        y_min, y_max = max(0, y-2), min(field.shape[1], y+3)
        z_min, z_max = max(0, z-2), min(field.shape[2], z+3)
        
        local_temporal = temporal_field[x_min:x_max, y_min:y_max, z_min:z_max, :]
        
        # Compute prediction based on temporal pattern
        temporal_mean = torch.mean(local_temporal, dim=(0, 1, 2))
        
        # Use learned mapping from temporal features to sensor values
        predictions = torch.zeros(len(self.sensor_indices), device=field.device)
        
        for i, sensor_idx in enumerate(self.sensor_indices):
            # Map temporal features to this sensor
            feature_idx = sensor_idx % temporal_field.shape[-1]
            prediction_val = temporal_mean[feature_idx]
            
            # Apply momentum if available
            if self.prediction_momentum is not None and i < self.prediction_momentum.shape[0]:
                prediction_val = 0.7 * prediction_val + 0.3 * self.prediction_momentum[i]
            
            predictions[i] = prediction_val
        
        # Update momentum
        self.prediction_momentum = predictions.clone().detach()
        
        return predictions
    
    def update_prediction_success(self, actual_values: torch.Tensor, predicted_values: torch.Tensor):
        """Update prediction confidence based on accuracy."""
        if len(self.sensor_indices) == 0:
            return
        
        # Compute prediction error for our sensors
        errors = []
        for i, sensor_idx in enumerate(self.sensor_indices):
            if sensor_idx < len(actual_values) and i < len(predicted_values):
                error = abs(actual_values[sensor_idx] - predicted_values[i]).detach().item()
                errors.append(error)
        
        if errors:
            avg_error = np.mean(errors)
            # Update confidence (0 = bad, 1 = perfect)
            self.prediction_confidence = 0.9 * self.prediction_confidence + 0.1 * (1.0 - min(1.0, avg_error))
            
            # Store in history
            self.sensory_prediction_history.append({
                'time': time.time(),
                'error': avg_error,
                'confidence': self.prediction_confidence
            })


class TopologyRegionSystem:
    """
    Manages topology regions for abstraction and causal understanding.
    
    This system:
    1. Detects stable patterns in the 4D field
    2. Tracks relationships between patterns
    3. Builds causal models through temporal correlation
    4. Enables abstraction through hierarchical composition
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 device: torch.device,
                 stability_threshold: float = 0.1,
                 similarity_threshold: float = 0.7,
                 max_regions: int = 200):
        """
        Initialize topology region system.
        
        Args:
            field_shape: Shape of the 4D field tensor
            device: Computation device
            stability_threshold: Minimum activation for region detection
            similarity_threshold: Threshold for pattern matching
            max_regions: Maximum number of regions to maintain
        """
        self.field_shape = field_shape
        self.device = device
        self.stability_threshold = stability_threshold
        self.similarity_threshold = similarity_threshold
        self.max_regions = max_regions
        
        # Region storage
        self.regions: Dict[str, TopologyRegion] = {}
        self.region_count = 0
        
        # Temporal tracking for causal detection
        self.activation_history = deque(maxlen=100)  # (time, region_id, strength)
        self.causal_window = 2.0  # seconds to look for causal relationships
        
        # Abstraction building
        self.abstraction_threshold = 3  # Min component regions for abstraction
        self.composition_history = deque(maxlen=50)  # Track co-activations
        
        # Performance tracking
        self.discovery_count = 0
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300.0  # 5 minutes
        
    def detect_topology_regions(self, 
                               field: torch.Tensor,
                               current_patterns: Optional[List] = None) -> List[str]:
        """
        Detect stable topology regions in current field state.
        
        Args:
            field: Current 4D field tensor
            current_patterns: Optional pre-extracted patterns
            
        Returns:
            List of activated region IDs
        """
        activated_regions = []
        current_time = time.time()
        
        # Extract salient regions from field
        salient_regions = self._find_salient_regions(field)
        
        for region_data in salient_regions:
            spatial_center = region_data['spatial_center']
            feature_center = region_data['feature_center']
            pattern = region_data['pattern']
            
            # Check if this matches existing region
            matched_region = self._find_matching_region(pattern, spatial_center)
            
            if matched_region:
                # Update existing region
                region = self.regions[matched_region]
                region.update_activation(region_data['strength'])
                
                # Update pattern with momentum
                region.activation_pattern = (
                    0.9 * region.activation_pattern + 0.1 * pattern
                )
                
                activated_regions.append(matched_region)
                
            elif region_data['strength'] > self.stability_threshold:
                # Create new region
                region_id = f"region_{self.region_count}"
                self.region_count += 1
                
                new_region = TopologyRegion(
                    region_id=region_id,
                    spatial_center=spatial_center,
                    feature_center=feature_center,
                    activation_pattern=pattern.clone(),
                    mean_activation=region_data['strength'],
                    stability=region_data['stability'],
                    coherence=region_data['coherence'],
                    discovery_time=current_time,
                    last_activation=current_time
                )
                
                self.regions[region_id] = new_region
                self.discovery_count += 1
                activated_regions.append(region_id)
                
                # Check for abstraction opportunities
                self._check_abstraction_formation(region_id)
        
        # Track activation sequence for causal detection
        for region_id in activated_regions:
            self.activation_history.append((current_time, region_id, self.regions[region_id].mean_activation))
            
        # Update causal relationships
        self._update_causal_links(activated_regions, current_time)
        
        # Periodic cleanup
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_weak_regions()
            self.last_cleanup_time = current_time
            
        return activated_regions
    
    def _find_salient_regions(self, field: torch.Tensor) -> List[Dict]:
        """Extract salient regions from field using local maxima detection."""
        salient_regions = []
        
        # Compute field statistics
        field_abs = torch.abs(field)
        mean_activation = torch.mean(field_abs).detach().item()
        std_activation = torch.std(field_abs).detach().item()
        max_activation = torch.max(field_abs).detach().item()
        
        # Threshold for salience - use a more sensitive threshold
        # Start detecting regions when activation is above mean
        salience_threshold = max(0.01, mean_activation * 0.8)
        
        # Simplified approach: find top activations instead of local maxima
        field_mean = field_abs.mean(dim=3)
        
        # Get top k points above threshold
        above_threshold = field_mean > salience_threshold
        if torch.sum(above_threshold) == 0:
            # If nothing above threshold, take top 5 points
            flat_field = field_mean.flatten()
            top_k = min(5, flat_field.numel())
            top_values, top_indices = torch.topk(flat_field, top_k)
            
            # Convert flat indices back to 3D
            max_indices = []
            for idx in top_indices:
                z = idx % field_mean.shape[2]
                y = (idx // field_mean.shape[2]) % field_mean.shape[1]
                x = idx // (field_mean.shape[1] * field_mean.shape[2])
                max_indices.append([x.detach().item(), y.detach().item(), z.detach().item()])
            max_indices = torch.tensor(max_indices, device=field.device)
        else:
            # Use points above threshold
            max_indices = torch.nonzero(above_threshold)
        
        # Already have max_indices from above
        
        for idx in max_indices[:10]:  # Limit to top 10 regions
            x, y, z = idx.tolist()
            
            # Extract local region
            x_start = max(0, x - 2)
            x_end = min(self.field_shape[0], x + 3)
            y_start = max(0, y - 2)
            y_end = min(self.field_shape[1], y + 3)
            z_start = max(0, z - 2)
            z_end = min(self.field_shape[2], z + 3)
            
            local_region = field[x_start:x_end, y_start:y_end, z_start:z_end, :]
            
            # Find dominant feature dimension
            feature_activations = field[x, y, z, :]
            feature_center = torch.argmax(torch.abs(feature_activations)).detach().item()
            
            # Extract pattern around center
            pattern = self._extract_pattern(field, (x, y, z), feature_center)
            
            # Compute region statistics
            strength = torch.max(torch.abs(local_region)).detach().item()
            stability = 1.0 / (torch.std(local_region).detach().item() + 1e-6)
            coherence = self._compute_coherence(local_region)
            
            salient_regions.append({
                'spatial_center': (x, y, z),
                'feature_center': feature_center,
                'pattern': pattern,
                'strength': strength,
                'stability': stability,
                'coherence': coherence
            })
        
        return salient_regions
    
    def _extract_pattern(self, field: torch.Tensor, 
                        spatial_center: Tuple[int, int, int],
                        feature_center: int,
                        size: int = 5) -> torch.Tensor:
        """Extract pattern around spatial center."""
        x, y, z = spatial_center
        half_size = size // 2
        
        # Get bounds
        x_start = max(0, x - half_size)
        x_end = min(self.field_shape[0], x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(self.field_shape[1], y + half_size + 1)
        z_start = max(0, z - half_size)
        z_end = min(self.field_shape[2], z + half_size + 1)
        
        # Extract pattern
        pattern = field[x_start:x_end, y_start:y_end, z_start:z_end, feature_center]
        
        # Normalize
        pattern = pattern / (torch.norm(pattern) + 1e-8)
        
        return pattern
    
    def _compute_coherence(self, region: torch.Tensor) -> float:
        """Compute spatial coherence of a region."""
        # Coherence = inverse of gradient magnitude
        if region.numel() < 2:
            return 0.0
            
        # Compute gradients in spatial dimensions
        grad_x = torch.diff(region, dim=0) if region.shape[0] > 1 else torch.zeros_like(region[:1])
        grad_y = torch.diff(region, dim=1) if region.shape[1] > 1 else torch.zeros_like(region[:, :1])
        grad_z = torch.diff(region, dim=2) if region.shape[2] > 1 else torch.zeros_like(region[:, :, :1])
        
        # Average gradient magnitude
        grad_mag = (
            torch.mean(torch.abs(grad_x)) +
            torch.mean(torch.abs(grad_y)) +
            torch.mean(torch.abs(grad_z))
        ) / 3.0
        
        # Coherence is inverse of gradient
        coherence = 1.0 / (1.0 + grad_mag.detach().item())
        
        return coherence
    
    def _find_matching_region(self, pattern: torch.Tensor, 
                             spatial_center: Tuple[int, int, int]) -> Optional[str]:
        """Find existing region that matches the pattern."""
        best_match = None
        best_similarity = 0.0
        
        for region_id, region in self.regions.items():
            # Check spatial proximity
            spatial_dist = np.sqrt(
                (region.spatial_center[0] - spatial_center[0])**2 +
                (region.spatial_center[1] - spatial_center[1])**2 +
                (region.spatial_center[2] - spatial_center[2])**2
            )
            
            # Only consider spatially close regions
            if spatial_dist < 5.0:
                # Compare patterns
                if pattern.shape == region.activation_pattern.shape:
                    similarity = F.cosine_similarity(
                        pattern.flatten().unsqueeze(0),
                        region.activation_pattern.flatten().unsqueeze(0)
                    ).detach().item()
                    
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_match = region_id
                        best_similarity = similarity
        
        return best_match
    
    def _update_causal_links(self, current_activations: List[str], current_time: float):
        """Update causal relationships based on temporal sequences."""
        # Look for patterns in activation history
        for i, (time1, region1, strength1) in enumerate(self.activation_history):
            if current_time - time1 > self.causal_window:
                continue
                
            # Check if any current activation follows this historical one
            for region2 in current_activations:
                if region2 != region1:
                    time_diff = current_time - time1
                    
                    # Causal strength based on temporal proximity and activation strength
                    causal_strength = (1.0 / (1.0 + time_diff)) * min(strength1, 1.0) * 0.1
                    
                    # Update causal links
                    if region1 in self.regions and region2 in self.regions:
                        self.regions[region1].add_causal_link(None, region2, causal_strength)
                        self.regions[region2].add_causal_link(region1, None, causal_strength)
                        
                        # Also mark as associated
                        self.regions[region1].associated_regions.add(region2)
                        self.regions[region2].associated_regions.add(region1)
    
    def _check_abstraction_formation(self, new_region_id: str):
        """Check if we can form abstractions from co-activated regions."""
        current_time = time.time()
        
        # Track this region in composition history
        self.composition_history.append((current_time, new_region_id))
        
        # Look for frequently co-activated regions
        recent_window = 5.0  # seconds
        recent_regions = [
            rid for t, rid in self.composition_history 
            if current_time - t < recent_window
        ]
        
        # Count co-occurrences
        from collections import Counter
        region_counts = Counter(recent_regions)
        
        # Find regions that frequently appear together
        frequent_regions = [
            rid for rid, count in region_counts.items() 
            if count >= self.abstraction_threshold
        ]
        
        if len(frequent_regions) >= 3:
            # Create abstract region representing the composition
            self._create_abstract_region(frequent_regions)
    
    def _create_abstract_region(self, component_ids: List[str]):
        """Create a higher-level abstract region from components."""
        # Compute abstract pattern as combination of components
        component_patterns = []
        spatial_centers = []
        
        for rid in component_ids:
            if rid in self.regions:
                region = self.regions[rid]
                component_patterns.append(region.activation_pattern.flatten())
                spatial_centers.append(region.spatial_center)
        
        if not component_patterns:
            return
            
        # Abstract pattern is the mean of components
        abstract_pattern = torch.stack(component_patterns).mean(dim=0)
        abstract_pattern = abstract_pattern / (torch.norm(abstract_pattern) + 1e-8)
        
        # Abstract spatial center is centroid
        abstract_center = tuple(
            int(np.mean([c[i] for c in spatial_centers]))
            for i in range(3)
        )
        
        # Create abstract region
        region_id = f"abstract_{self.region_count}"
        self.region_count += 1
        
        abstract_region = TopologyRegion(
            region_id=region_id,
            spatial_center=abstract_center,
            feature_center=0,  # Abstract regions span features
            activation_pattern=abstract_pattern.reshape(-1),  # Flatten for storage
            mean_activation=0.5,
            stability=1.0,
            coherence=1.0,
            discovery_time=time.time(),
            last_activation=time.time(),
            abstraction_level=1,  # Higher abstraction level
            component_regions=set(component_ids)
        )
        
        self.regions[region_id] = abstract_region
        
        # Link components to abstraction
        for rid in component_ids:
            if rid in self.regions:
                self.regions[rid].associated_regions.add(region_id)
    
    def _cleanup_weak_regions(self):
        """Remove weak or old regions to maintain memory limits."""
        if len(self.regions) <= self.max_regions:
            return
            
        current_time = time.time()
        
        # Score regions by importance and recency
        region_scores = []
        for rid, region in self.regions.items():
            # Forgetting curve
            time_factor = np.exp(-(current_time - region.last_activation) / 3600.0)  # 1 hour half-life
            
            # Combined score
            score = region.importance * time_factor * region.consolidation_level
            region_scores.append((rid, score))
        
        # Sort by score
        region_scores.sort(key=lambda x: x[1])
        
        # Remove lowest scoring regions
        n_remove = len(self.regions) - self.max_regions
        for rid, _ in region_scores[:n_remove]:
            # Don't remove abstract regions or highly connected ones
            if (self.regions[rid].abstraction_level == 0 and 
                len(self.regions[rid].associated_regions) < 5):
                del self.regions[rid]
    
    def get_active_abstractions(self) -> List[TopologyRegion]:
        """Get currently active abstract regions."""
        return [
            region for region in self.regions.values()
            if region.abstraction_level > 0 and 
            time.time() - region.last_activation < 60.0  # Active in last minute
        ]
    
    def get_causal_chain(self, start_region: str, max_depth: int = 5) -> List[List[str]]:
        """Get causal chains starting from a region."""
        if start_region not in self.regions:
            return []
            
        chains = []
        
        def build_chain(region_id: str, current_chain: List[str], depth: int):
            if depth >= max_depth:
                chains.append(current_chain.copy())
                return
                
            region = self.regions.get(region_id)
            if not region:
                return
                
            # Follow strongest causal links
            successors = sorted(
                region.causal_successors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 successors
            
            if not successors:
                chains.append(current_chain.copy())
            else:
                for successor_id, strength in successors:
                    if successor_id not in current_chain:  # Avoid cycles
                        new_chain = current_chain + [successor_id]
                        build_chain(successor_id, new_chain, depth + 1)
        
        build_chain(start_region, [start_region], 0)
        return chains
    
    def consolidate_regions(self, brain):
        """Consolidate important regions during maintenance."""
        for region in self.regions.values():
            if region.importance > 1.0:
                # Increase consolidation level
                region.consolidation_level += 1
                region.decay_rate *= 0.99  # Slower decay for consolidated regions
                
                # Strengthen the pattern in the field
                if hasattr(brain, 'unified_field'):
                    x, y, z = region.spatial_center
                    if (0 <= x < brain.unified_field.shape[0] and
                        0 <= y < brain.unified_field.shape[1] and
                        0 <= z < brain.unified_field.shape[2]):
                        # Reinforce the pattern
                        brain.unified_field[x, y, z, region.feature_center] *= 1.1
    
    def update_sensory_predictions(self, sensory_dim: int, recent_sensory: Optional[deque] = None):
        """
        Update which regions should predict which sensors based on activity patterns.
        This enables emergent sensory specialization.
        """
        if not self.regions or sensory_dim == 0:
            return
        
        # Get recently active regions
        current_time = time.time()
        active_regions = [
            (rid, r) for rid, r in self.regions.items()
            if current_time - r.last_activation < 10.0  # Active in last 10 seconds
        ]
        
        if not active_regions:
            return
        
        # Assign sensors to regions based on:
        # 1. Spatial distribution (spread sensors across regions)
        # 2. Temporal stability (stable regions are better predictors)
        # 3. Current prediction performance
        
        # Sort regions by stability and importance
        sorted_regions = sorted(
            active_regions,
            key=lambda x: x[1].stability * x[1].importance,
            reverse=True
        )
        
        # Distribute sensors across top regions
        sensors_per_region = max(1, sensory_dim // max(1, len(sorted_regions)))
        
        for i, (rid, region) in enumerate(sorted_regions):
            # Determine which sensors this region should predict
            if not region.is_sensory_predictive:
                # New predictive region - assign initial sensors
                start_idx = (i * sensors_per_region) % sensory_dim
                end_idx = min(start_idx + sensors_per_region, sensory_dim)
                
                region.sensor_indices = list(range(start_idx, end_idx))
                region.is_sensory_predictive = True
                region.prediction_confidence = 0.3  # Start with low confidence
                
            else:
                # Existing predictive region - update based on performance
                if region.prediction_confidence < 0.2 and len(region.sensory_prediction_history) > 10:
                    # Poor predictor - try different sensors
                    shift = sensors_per_region // 2
                    region.sensor_indices = [
                        (idx + shift) % sensory_dim for idx in region.sensor_indices
                    ]
                    region.prediction_confidence = 0.3  # Reset confidence
                    
                elif region.prediction_confidence > 0.7:
                    # Good predictor - maybe take on more sensors
                    if len(region.sensor_indices) < sensors_per_region * 2:
                        # Find unassigned sensors
                        all_assigned = set()
                        for _, r in self.regions.items():
                            if r.is_sensory_predictive:
                                all_assigned.update(r.sensor_indices)
                        
                        unassigned = [i for i in range(sensory_dim) if i not in all_assigned]
                        if unassigned:
                            # Take one unassigned sensor
                            region.sensor_indices.append(unassigned[0])
    
    def get_predictive_regions(self) -> List[TopologyRegion]:
        """Get all regions that are capable of sensory prediction."""
        return [r for r in self.regions.values() if r.is_sensory_predictive]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get system statistics."""
        current_time = time.time()
        
        active_regions = [
            r for r in self.regions.values()
            if current_time - r.last_activation < 300.0  # Active in last 5 min
        ]
        
        abstract_regions = [
            r for r in self.regions.values()
            if r.abstraction_level > 0
        ]
        
        predictive_regions = [
            r for r in self.regions.values()
            if r.is_sensory_predictive
        ]
        
        avg_prediction_confidence = np.mean([
            r.prediction_confidence for r in predictive_regions
        ]) if predictive_regions else 0.0
        
        return {
            'total_regions': len(self.regions),
            'active_regions': len(active_regions),
            'abstract_regions': len(abstract_regions),
            'predictive_regions': len(predictive_regions),
            'avg_prediction_confidence': avg_prediction_confidence,
            'discovery_count': self.discovery_count,
            'avg_connections': np.mean([
                len(r.associated_regions) for r in self.regions.values()
            ]) if self.regions else 0,
            'causal_links': sum(
                len(r.causal_successors) for r in self.regions.values()
            )
        }