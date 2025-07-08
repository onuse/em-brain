"""
Universal Actuator Effect Discovery System - Emerges actuator categories from pure experience.
No assumptions about legs, wheels, grippers, or any specific embodiment.
Discovers "movement-type", "interaction-type", "communication-type" through prediction patterns.
"""

import math
import time
import statistics
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ActuatorEffectPattern:
    """Tracks the effect pattern of a specific actuator on sensory dimensions."""
    actuator_id: str
    affected_sensor_dimensions: Dict[int, float]  # sensor_dim -> correlation strength
    effect_magnitude: float  # Average magnitude of effect
    effect_consistency: float  # How consistently this actuator produces effects
    temporal_delay: float  # How quickly effects appear after actuation
    effect_duration: float  # How long effects persist
    total_activations: int = 0
    successful_predictions: int = 0
    
    def get_effect_reliability(self) -> float:
        """Calculate how reliable this actuator's effects are."""
        if self.total_activations == 0:
            return 0.0
        return self.successful_predictions / self.total_activations
    
    def get_primary_affected_dimensions(self, threshold: float = 0.3) -> List[int]:
        """Get sensor dimensions most affected by this actuator."""
        return [dim for dim, correlation in self.affected_sensor_dimensions.items() 
                if correlation > threshold]


@dataclass
class EmergentActuatorCategory:
    """An emergent category of actuators discovered through effect similarity."""
    category_id: str
    member_actuators: Set[str] = field(default_factory=set)
    characteristic_effect_pattern: Dict[int, float] = field(default_factory=dict)  # Typical effect on sensors
    category_strength: float = 0.0  # How distinct this category is
    category_coherence: float = 0.0  # How similar members are to each other
    discovery_confidence: float = 0.0  # Confidence in this categorization
    
    # Emergent properties (discovered, not programmed)
    appears_spatial: bool = False  # Effects seem to change position-like sensors
    appears_manipulative: bool = False  # Effects seem to change object-state sensors
    appears_environmental: bool = False  # Effects seem to change ambient sensors
    
    def add_actuator(self, actuator_id: str, effect_pattern: ActuatorEffectPattern):
        """Add an actuator to this emergent category."""
        self.member_actuators.add(actuator_id)
        
        # Update characteristic pattern (weighted average)
        weight = len(self.member_actuators)
        for dim, correlation in effect_pattern.affected_sensor_dimensions.items():
            if dim in self.characteristic_effect_pattern:
                # Weighted update
                current = self.characteristic_effect_pattern[dim]
                self.characteristic_effect_pattern[dim] = (current * (weight - 1) + correlation) / weight
            else:
                self.characteristic_effect_pattern[dim] = correlation / weight
        
        self._update_emergent_properties()
    
    def _update_emergent_properties(self):
        """Update emergent properties based on effect patterns."""
        if not self.characteristic_effect_pattern:
            return
        
        # Analyze effect patterns to infer emergent properties
        dimensions = list(self.characteristic_effect_pattern.keys())
        correlations = list(self.characteristic_effect_pattern.values())
        
        # Spatial: Early dimensions (often position-related) with high, consistent correlation
        early_dim_effects = [corr for dim, corr in self.characteristic_effect_pattern.items() if dim < 6]
        if early_dim_effects and statistics.mean(early_dim_effects) > 0.4:
            self.appears_spatial = True
        
        # Manipulative: Mid-range dimensions with moderate correlation
        mid_dim_effects = [corr for dim, corr in self.characteristic_effect_pattern.items() if 6 <= dim < 12]
        if mid_dim_effects and 0.2 < statistics.mean(mid_dim_effects) < 0.6:
            self.appears_manipulative = True
        
        # Environmental: Broad, low-magnitude effects across many dimensions
        if len(dimensions) > 8 and statistics.mean(correlations) < 0.3:
            self.appears_environmental = True


class UniversalActuatorDiscovery:
    """
    Discovers actuator effect patterns and emergent categories purely from experience.
    Works with any embodiment: legs, wheels, grids, alien actuators.
    """
    
    def __init__(self):
        # Actuator effect tracking
        self.actuator_patterns: Dict[str, ActuatorEffectPattern] = {}
        self.sensory_history: List[List[float]] = []  # Recent sensory states
        self.actuator_history: List[Dict[str, float]] = []  # Recent actuator commands
        self.history_length = 20  # How many steps to remember
        
        # Effect discovery
        self.correlation_threshold = 0.2  # Minimum correlation to consider an effect
        self.temporal_window = 5  # How many steps to look ahead for effects
        self.minimum_activations = 3  # Min activations before considering an actuator
        
        # Emergent categorization
        self.emergent_categories: Dict[str, EmergentActuatorCategory] = {}
        self.category_similarity_threshold = 0.6  # How similar patterns must be to group
        self.next_category_id = 1
        
        # Discovery statistics
        self.total_observations = 0
        self.significant_effects_discovered = 0
        self.categories_formed = 0
        
    def observe_actuator_effects(self, actuator_commands: Dict[str, float], 
                                sensory_reading: List[float]) -> Dict[str, Any]:
        """
        Observe the effects of actuator commands on sensory readings.
        This is where universal actuator discovery happens.
        """
        self.total_observations += 1
        
        # Store current state
        self.actuator_history.append(actuator_commands.copy())
        self.sensory_history.append(sensory_reading.copy())
        
        # Maintain sliding window
        if len(self.actuator_history) > self.history_length:
            self.actuator_history.pop(0)
            self.sensory_history.pop(0)
        
        # Need enough history to detect effects
        if len(self.actuator_history) < self.temporal_window:
            return {"status": "collecting_history", "observations": self.total_observations}
        
        # Analyze effects for each actuator
        discovered_effects = {}
        for actuator_id in actuator_commands.keys():
            effect_analysis = self._analyze_actuator_effects(actuator_id)
            if effect_analysis:
                discovered_effects[actuator_id] = effect_analysis
        
        # Update emergent categorization
        self._update_emergent_categories()
        
        return {
            "status": "analyzing_effects",
            "observations": self.total_observations,
            "actuators_analyzed": len(discovered_effects),
            "emergent_categories": len(self.emergent_categories),
            "significant_effects": self.significant_effects_discovered
        }
    
    def _analyze_actuator_effects(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """Analyze the effect pattern of a specific actuator."""
        if actuator_id not in self.actuator_patterns:
            self.actuator_patterns[actuator_id] = ActuatorEffectPattern(
                actuator_id=actuator_id,
                affected_sensor_dimensions={},
                effect_magnitude=0.0,
                effect_consistency=0.0,
                temporal_delay=1.0,
                effect_duration=1.0
            )
        
        pattern = self.actuator_patterns[actuator_id]
        
        # Find activations of this actuator
        activations = []
        for i, commands in enumerate(self.actuator_history):
            if abs(commands.get(actuator_id, 0.0)) > 0.1:  # Significant activation
                activations.append(i)
        
        if len(activations) < self.minimum_activations:
            return None
        
        # Analyze effects for each activation
        effect_detections = []
        for activation_time in activations:
            effect = self._detect_effect_from_activation(actuator_id, activation_time)
            if effect:
                effect_detections.append(effect)
        
        if not effect_detections:
            return None
        
        # Update pattern based on detections
        self._update_actuator_pattern(pattern, effect_detections)
        
        return {
            "effect_magnitude": pattern.effect_magnitude,
            "affected_dimensions": pattern.get_primary_affected_dimensions(),
            "reliability": pattern.get_effect_reliability(),
            "activations_analyzed": len(activations)
        }
    
    def _detect_effect_from_activation(self, actuator_id: str, activation_time: int) -> Optional[Dict[str, float]]:
        """Detect sensory changes following an actuator activation."""
        if activation_time >= len(self.sensory_history) - 1:
            return None
        
        # Get sensory state before and after activation
        before_sensors = self.sensory_history[activation_time]
        
        # Look for effects in subsequent time steps
        effects = {}
        activation_strength = abs(self.actuator_history[activation_time].get(actuator_id, 0.0))
        
        for delay in range(1, min(self.temporal_window, len(self.sensory_history) - activation_time)):
            after_sensors = self.sensory_history[activation_time + delay]
            
            # Calculate changes for each sensor dimension
            for dim in range(min(len(before_sensors), len(after_sensors))):
                change = abs(after_sensors[dim] - before_sensors[dim])
                
                # Normalize by activation strength to find correlation
                if activation_strength > 0:
                    correlation = change / activation_strength
                    
                    # Significant effect detected
                    if correlation > self.correlation_threshold:
                        if dim not in effects:
                            effects[dim] = correlation
                        else:
                            effects[dim] = max(effects[dim], correlation)  # Take strongest correlation
        
        return effects if effects else None
    
    def _update_actuator_pattern(self, pattern: ActuatorEffectPattern, effect_detections: List[Dict[str, float]]):
        """Update an actuator's effect pattern based on new detections."""
        pattern.total_activations += len(effect_detections)
        
        if not effect_detections:
            return
        
        pattern.successful_predictions += len(effect_detections)
        
        # Aggregate effects across all detections
        all_effects = defaultdict(list)
        for detection in effect_detections:
            for dim, correlation in detection.items():
                all_effects[dim].append(correlation)
        
        # Update affected dimensions with average correlations
        for dim, correlations in all_effects.items():
            avg_correlation = statistics.mean(correlations)
            if dim in pattern.affected_sensor_dimensions:
                # Weighted update
                current = pattern.affected_sensor_dimensions[dim]
                pattern.affected_sensor_dimensions[dim] = (current + avg_correlation) / 2
            else:
                pattern.affected_sensor_dimensions[dim] = avg_correlation
        
        # Update overall pattern metrics
        all_correlations = []
        for detection in effect_detections:
            all_correlations.extend(detection.values())
        
        if all_correlations:
            pattern.effect_magnitude = statistics.mean(all_correlations)
            pattern.effect_consistency = 1.0 - (statistics.stdev(all_correlations) if len(all_correlations) > 1 else 0.0)
            
            if avg_correlation > self.correlation_threshold:
                self.significant_effects_discovered += 1
    
    def _update_emergent_categories(self):
        """Update emergent actuator categories based on effect pattern similarity."""
        # Get actuators with sufficient data
        mature_actuators = {
            actuator_id: pattern for actuator_id, pattern in self.actuator_patterns.items()
            if pattern.total_activations >= self.minimum_activations and pattern.effect_magnitude > 0.1
        }
        
        if len(mature_actuators) < 2:
            return
        
        # Find uncategorized actuators
        categorized_actuators = set()
        for category in self.emergent_categories.values():
            categorized_actuators.update(category.member_actuators)
        
        uncategorized = {
            actuator_id: pattern for actuator_id, pattern in mature_actuators.items()
            if actuator_id not in categorized_actuators
        }
        
        # Try to categorize uncategorized actuators
        for actuator_id, pattern in uncategorized.items():
            best_category = self._find_best_category_match(pattern)
            
            if best_category:
                best_category.add_actuator(actuator_id, pattern)
            else:
                # Create new category
                self._create_new_category(actuator_id, pattern)
    
    def _find_best_category_match(self, pattern: ActuatorEffectPattern) -> Optional[EmergentActuatorCategory]:
        """Find the best matching category for an actuator pattern."""
        best_match = None
        best_similarity = 0.0
        
        for category in self.emergent_categories.values():
            similarity = self._calculate_pattern_similarity(pattern, category.characteristic_effect_pattern)
            
            if similarity > self.category_similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = category
        
        return best_match
    
    def _calculate_pattern_similarity(self, pattern: ActuatorEffectPattern, 
                                    category_pattern: Dict[int, float]) -> float:
        """Calculate similarity between an actuator pattern and a category pattern."""
        if not pattern.affected_sensor_dimensions or not category_pattern:
            return 0.0
        
        # Get common dimensions
        pattern_dims = set(pattern.affected_sensor_dimensions.keys())
        category_dims = set(category_pattern.keys())
        common_dims = pattern_dims.intersection(category_dims)
        
        if not common_dims:
            return 0.0
        
        # Calculate correlation similarity for common dimensions
        similarities = []
        for dim in common_dims:
            pattern_corr = pattern.affected_sensor_dimensions[dim]
            category_corr = category_pattern[dim]
            
            # Similarity based on correlation difference
            max_corr = max(pattern_corr, category_corr)
            if max_corr > 0:
                similarity = 1.0 - abs(pattern_corr - category_corr) / max_corr
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Weight by overlap (more common dimensions = higher similarity)
        overlap_factor = len(common_dims) / max(len(pattern_dims), len(category_dims))
        avg_similarity = statistics.mean(similarities)
        
        return avg_similarity * overlap_factor
    
    def _create_new_category(self, actuator_id: str, pattern: ActuatorEffectPattern):
        """Create a new emergent category for an actuator."""
        category_id = f"emergent_category_{self.next_category_id}"
        self.next_category_id += 1
        
        new_category = EmergentActuatorCategory(
            category_id=category_id,
            category_strength=pattern.effect_magnitude,
            discovery_confidence=pattern.get_effect_reliability()
        )
        
        new_category.add_actuator(actuator_id, pattern)
        self.emergent_categories[category_id] = new_category
        self.categories_formed += 1
    
    def get_actuator_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get discovered actuator categories and their properties."""
        categories = {}
        
        for category_id, category in self.emergent_categories.items():
            categories[category_id] = {
                "member_actuators": list(category.member_actuators),
                "primary_affected_dimensions": [
                    dim for dim, corr in category.characteristic_effect_pattern.items() if corr > 0.3
                ],
                "emergent_properties": {
                    "appears_spatial": category.appears_spatial,
                    "appears_manipulative": category.appears_manipulative,
                    "appears_environmental": category.appears_environmental
                },
                "category_strength": category.category_strength,
                "discovery_confidence": category.discovery_confidence,
                "member_count": len(category.member_actuators)
            }
        
        return categories
    
    def get_actuator_analysis(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis of a specific actuator."""
        if actuator_id not in self.actuator_patterns:
            return None
        
        pattern = self.actuator_patterns[actuator_id]
        
        # Find which category this actuator belongs to
        category_membership = None
        for category_id, category in self.emergent_categories.items():
            if actuator_id in category.member_actuators:
                category_membership = {
                    "category_id": category_id,
                    "appears_spatial": category.appears_spatial,
                    "appears_manipulative": category.appears_manipulative,
                    "appears_environmental": category.appears_environmental
                }
                break
        
        return {
            "actuator_id": actuator_id,
            "effect_magnitude": pattern.effect_magnitude,
            "effect_reliability": pattern.get_effect_reliability(),
            "primary_affected_dimensions": pattern.get_primary_affected_dimensions(),
            "total_activations": pattern.total_activations,
            "category_membership": category_membership,
            "effect_pattern": pattern.affected_sensor_dimensions
        }
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about actuator discovery."""
        total_actuators = len(self.actuator_patterns)
        categorized_actuators = sum(len(cat.member_actuators) for cat in self.emergent_categories.values())
        
        # Category type distribution
        spatial_categories = sum(1 for cat in self.emergent_categories.values() if cat.appears_spatial)
        manipulative_categories = sum(1 for cat in self.emergent_categories.values() if cat.appears_manipulative)
        environmental_categories = sum(1 for cat in self.emergent_categories.values() if cat.appears_environmental)
        
        return {
            "total_observations": self.total_observations,
            "total_actuators_discovered": total_actuators,
            "actuators_with_significant_effects": self.significant_effects_discovered,
            "emergent_categories_formed": len(self.emergent_categories),
            "actuators_categorized": categorized_actuators,
            "categorization_coverage": categorized_actuators / total_actuators if total_actuators > 0 else 0.0,
            "category_distribution": {
                "spatial_categories": spatial_categories,
                "manipulative_categories": manipulative_categories,
                "environmental_categories": environmental_categories,
                "unclassified_categories": len(self.emergent_categories) - spatial_categories - manipulative_categories - environmental_categories
            },
            "discovery_efficiency": self.significant_effects_discovered / self.total_observations if self.total_observations > 0 else 0.0
        }
    
    def reset_discovery(self):
        """Reset all discovery data (for testing or relearning)."""
        self.actuator_patterns.clear()
        self.emergent_categories.clear()
        self.sensory_history.clear()
        self.actuator_history.clear()
        self.total_observations = 0
        self.significant_effects_discovered = 0
        self.categories_formed = 0
        self.next_category_id = 1