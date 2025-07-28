#!/usr/bin/env python3
"""
Cross-Modal Object Attention - Constraint-Based Exclusive Attention

This implements object-based attention that emerges from computational constraints:
1. Limited bandwidth forces exclusive attention
2. Cross-modal binding requires shared focus
3. Objects compete for the single attention resource
4. All modalities align to the winning object's location

Key principles:
- Attention is per-object, not per-modality
- Cross-modal correlations define objects
- Computational constraints create exclusivity
- Temporal coherence drives binding
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import cv2
import torch
import torch.nn.functional as F

from .signal_attention import UniversalAttentionSystem, ModalityType
from ..memory.pattern_memory import UniversalMemorySystem
from ..parameters.constraint_parameters import EmergentParameterSystem, ConstraintType


class ObjectState(Enum):
    """States of tracked objects"""
    EMERGING = "emerging"        # Object being formed from correlations
    ACTIVE = "active"           # Object competing for attention
    ATTENDED = "attended"       # Object currently receiving attention
    INHIBITED = "inhibited"     # Object recently attended (inhibition of return)
    FADING = "fading"           # Object losing coherence


@dataclass
class CrossModalObject:
    """An object composed of correlated features across modalities"""
    object_id: str
    state: ObjectState = ObjectState.EMERGING
    
    # Spatial properties
    center_x: float = 0
    center_y: float = 0
    extent_x: float = 0
    extent_y: float = 0
    
    # Temporal properties
    creation_time: float = 0
    last_update: float = 0
    last_attended: float = 0
    
    # Cross-modal features
    modality_features: Dict[ModalityType, Dict] = field(default_factory=dict)
    
    # Competitive properties
    salience: float = 0         # How attention-grabbing
    novelty: float = 0          # How novel/unexpected
    urgency: float = 0          # How time-critical
    coherence: float = 0        # How well-bound across modalities
    
    # Attention history
    attention_count: int = 0
    total_attention_time: float = 0
    
    def get_competitive_score(self, competitive_weights: Dict[str, float], inhibition_duration: float) -> float:
        """Calculate competitive score for attention using emergent weights"""
        # Inhibition of return penalty
        time_since_attended = time.time() - self.last_attended
        inhibition_penalty = max(0, 1 - time_since_attended / inhibition_duration)
        
        # Base competitive score using emergent weights
        base_score = (self.salience * competitive_weights['salience'] + 
                     self.novelty * competitive_weights['novelty'] + 
                     self.urgency * competitive_weights['urgency'] + 
                     self.coherence * competitive_weights['coherence'])
        
        # Apply inhibition penalty
        return base_score * (1 - inhibition_penalty * 0.8)
    
    def get_spatial_mask(self, width: int, height: int) -> np.ndarray:
        """Get spatial attention mask for this object"""
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create elliptical mask around object
        y, x = np.ogrid[:height, :width]
        
        # Ensure coordinates are within bounds
        center_x = max(0, min(width-1, int(self.center_x)))
        center_y = max(0, min(height-1, int(self.center_y)))
        
        # Elliptical attention region
        extent_x = max(10, min(width//4, int(self.extent_x)))
        extent_y = max(10, min(height//4, int(self.extent_y)))
        
        ellipse = ((x - center_x)**2 / extent_x**2 + 
                  (y - center_y)**2 / extent_y**2) <= 1
        
        mask[ellipse] = 1.0
        return mask
    
    def update_from_modality(self, modality: ModalityType, features: Dict, 
                           attention_map: np.ndarray):
        """Update object with new features from a modality"""
        self.modality_features[modality] = features
        self.last_update = time.time()
        
        # Update spatial properties from attention map
        if len(attention_map.shape) == 2:
            # Find center of mass of attention
            moments = cv2.moments(attention_map)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                
                # Weighted average with existing position
                alpha = 0.3
                self.center_x = alpha * cx + (1-alpha) * self.center_x
                self.center_y = alpha * cy + (1-alpha) * self.center_y
                
                # Update extent based on attention spread
                self.extent_x = np.sqrt(moments['m20'] / moments['m00'])
                self.extent_y = np.sqrt(moments['m02'] / moments['m00'])


class CrossModalAttentionSystem:
    """
    Constraint-based cross-modal attention system.
    
    Implements exclusive attention that emerges from computational constraints
    and cross-modal binding requirements.
    """
    
    def __init__(self, 
                 parameter_system: Optional[EmergentParameterSystem] = None,
                 target_framerate: float = 30.0,
                 power_budget_watts: float = 10.0):
        
        # Initialize GPU device detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_gpu = True
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.use_gpu = True
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
        
        # Initialize emergent parameter system
        self.params = parameter_system or EmergentParameterSystem(
            target_framerate=target_framerate,
            power_budget_watts=power_budget_watts
        )
        
        # Derive all parameters from constraints
        self.compute_budget = self.params.get_parameter('compute_budget', ConstraintType.HARDWARE)
        self.attention_duration = self.params.get_parameter('attention_duration', ConstraintType.BIOLOGICAL)
        self.switch_cost = self.params.get_parameter('switch_cost', ConstraintType.ENERGY)
        self.binding_window = self.params.get_parameter('binding_window', ConstraintType.BIOLOGICAL)
        
        # Current attention state
        self.attended_object: Optional[CrossModalObject] = None
        self.attention_start_time: float = 0
        self.remaining_compute: int = self.compute_budget
        
        # Object tracking
        self.active_objects: Dict[str, CrossModalObject] = {}
        self.object_counter: int = 0
        
        # Cross-modal correlation tracking
        correlation_history_size = self.params.memory_budget['correlation_history_size']
        self.correlation_history: deque = deque(maxlen=correlation_history_size)
        self.binding_candidates: List[Dict] = []
        
        # Modality integration
        self.universal_attention = UniversalAttentionSystem()
        self.modality_states: Dict[ModalityType, Dict] = {}
        
        # Statistics
        self.attention_switches: int = 0
        self.total_objects_created: int = 0
        self.binding_events: int = 0
        
        # Emergent constraint parameters
        self.min_coherence_threshold = self.params.get_parameter('correlation_threshold', ConstraintType.INFORMATION)
        self.max_objects = self.params.get_parameter('max_objects', ConstraintType.HARDWARE)
        self.correlation_threshold = self.params.get_parameter('correlation_threshold', ConstraintType.INFORMATION)
        
        # Competitive weights from constraint priorities
        self.competitive_weights = self.params.get_competitive_weights()
    
    def update(self, sensory_streams: Dict[ModalityType, Dict]) -> Dict[str, Any]:
        """
        Update cross-modal attention system with new sensory data.
        
        Args:
            sensory_streams: Dict mapping modality to {signal, brain_output, novelty}
            
        Returns:
            Current attention state and object information
        """
        current_time = time.time()
        
        # Phase 1: Process each modality independently
        modality_attention_maps = {}
        for modality, stream_data in sensory_streams.items():
            attention_map, _ = self.universal_attention.calculate_attention_map(
                stream_data['signal'], 
                modality,
                stream_data.get('brain_output'),
                stream_data.get('novelty', 0.5)
            )
            modality_attention_maps[modality] = attention_map
        
        # Phase 2: Find cross-modal correlations
        correlations = self._find_cross_modal_correlations(
            modality_attention_maps, sensory_streams
        )
        
        # Phase 3: Update/create objects from correlations
        self._update_objects_from_correlations(correlations, modality_attention_maps)
        
        # Phase 4: Competitive attention selection
        self._compete_for_attention(current_time)
        
        # Phase 5: Apply attention constraints
        attention_mask = self._apply_attention_constraints()
        
        # Phase 6: Update statistics
        self._update_statistics()
        
        return {
            'attended_object': self.attended_object,
            'attention_mask': attention_mask,
            'active_objects': len(self.active_objects),
            'attention_switches': self.attention_switches,
            'compute_remaining': self.remaining_compute,
            'binding_events': self.binding_events,
            'object_details': self._get_object_details()
        }
    
    def _find_cross_modal_correlations(self, 
                                     attention_maps: Dict[ModalityType, np.ndarray],
                                     sensory_streams: Dict[ModalityType, Dict]) -> List[Dict]:
        """Find correlations between modalities that suggest shared objects"""
        correlations = []
        current_time = time.time()
        
        modalities = list(attention_maps.keys())
        
        # Compare each pair of modalities
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                map1, map2 = attention_maps[mod1], attention_maps[mod2]
                
                # Calculate correlation based on modality types
                correlation = self._calculate_modality_correlation(
                    mod1, map1, mod2, map2, sensory_streams
                )
                
                if correlation['strength'] > self.correlation_threshold:
                    correlation['timestamp'] = current_time
                    correlations.append(correlation)
        
        return correlations
    
    def _calculate_modality_correlation(self, 
                                      mod1: ModalityType, map1: np.ndarray,
                                      mod2: ModalityType, map2: np.ndarray,
                                      sensory_streams: Dict) -> Dict:
        """Calculate correlation between two modalities"""
        
        # Temporal correlation - do attention peaks coincide in time?
        temporal_correlation = self._calculate_temporal_correlation(mod1, mod2)
        
        # Spatial correlation - for spatial modalities, do peaks align?
        spatial_correlation = self._calculate_spatial_correlation(map1, map2)
        
        # Feature correlation - do the underlying features suggest same object?
        feature_correlation = self._calculate_feature_correlation(
            mod1, mod2, sensory_streams
        )
        
        # Combined correlation strength
        strength = (temporal_correlation * 0.4 + 
                   spatial_correlation * 0.3 + 
                   feature_correlation * 0.3)
        
        return {
            'modality1': mod1,
            'modality2': mod2,
            'strength': strength,
            'temporal': temporal_correlation,
            'spatial': spatial_correlation,
            'feature': feature_correlation
        }
    
    def _calculate_temporal_correlation(self, mod1: ModalityType, mod2: ModalityType) -> float:
        """Calculate temporal correlation between modalities"""
        # Simplified: assume strong temporal correlation for now
        # In real implementation, would track onset/offset timing
        return 0.9 if mod1 != mod2 else 0.0
    
    def _calculate_spatial_correlation(self, map1: np.ndarray, map2: np.ndarray) -> float:
        """Calculate spatial correlation between attention maps using GPU acceleration"""
        if len(map1.shape) != 2 or len(map2.shape) != 2:
            return 0.5  # Default for non-spatial modalities
        
        if self.use_gpu:
            return self._calculate_spatial_correlation_gpu(map1, map2)
        else:
            return self._calculate_spatial_correlation_cpu(map1, map2)
    
    def _calculate_spatial_correlation_gpu(self, map1: np.ndarray, map2: np.ndarray) -> float:
        """GPU-accelerated spatial correlation calculation"""
        try:
            # Convert to PyTorch tensors and move to GPU
            tensor1 = torch.from_numpy(map1.astype(np.float32)).to(self.device)
            tensor2 = torch.from_numpy(map2.astype(np.float32)).to(self.device)
            
            # Resize to same dimensions if needed
            if tensor1.shape != tensor2.shape:
                tensor2 = F.interpolate(tensor2.unsqueeze(0).unsqueeze(0), 
                                       size=tensor1.shape, 
                                       mode='bilinear', 
                                       align_corners=False).squeeze()
            
            # Calculate normalized cross-correlation using GPU
            # Flatten tensors for correlation calculation
            flat1 = tensor1.flatten()
            flat2 = tensor2.flatten()
            
            # Calculate correlation coefficient
            mean1 = torch.mean(flat1)
            mean2 = torch.mean(flat2)
            
            numerator = torch.sum((flat1 - mean1) * (flat2 - mean2))
            denominator = torch.sqrt(torch.sum((flat1 - mean1) ** 2) * torch.sum((flat2 - mean2) ** 2))
            
            if denominator > 0:
                correlation = torch.abs(numerator / denominator)
                return float(correlation.cpu().item())
            else:
                return 0.5
                
        except Exception as e:
            # Fallback to CPU calculation
            return self._calculate_spatial_correlation_cpu(map1, map2)
    
    def _calculate_spatial_correlation_cpu(self, map1: np.ndarray, map2: np.ndarray) -> float:
        """CPU fallback for spatial correlation calculation"""
        # Resize to same dimensions for comparison
        if map1.shape != map2.shape:
            map2 = cv2.resize(map2, (map1.shape[1], map1.shape[0]))
        
        # Ensure proper data type for OpenCV
        map1 = map1.astype(np.float32)
        map2 = map2.astype(np.float32)
        
        # Calculate normalized cross-correlation
        try:
            correlation = cv2.matchTemplate(map1, map2, cv2.TM_CCORR_NORMED)
            return float(np.max(correlation))
        except:
            # Fallback to simple correlation coefficient
            try:
                corr_matrix = np.corrcoef(map1.flatten(), map2.flatten())
                return float(abs(corr_matrix[0, 1])) if not np.isnan(corr_matrix[0, 1]) else 0.5
            except:
                return 0.5
    
    def _calculate_feature_correlation(self, 
                                     mod1: ModalityType, mod2: ModalityType,
                                     sensory_streams: Dict) -> float:
        """Calculate feature-based correlation"""
        # Simplified: assume moderate feature correlation
        # In real implementation, would compare extracted features
        return 0.8
    
    def _update_objects_from_correlations(self, 
                                        correlations: List[Dict],
                                        attention_maps: Dict[ModalityType, np.ndarray]):
        """Update or create objects based on correlations"""
        current_time = time.time()
        
        # Group correlations by strength - use emergent threshold
        strong_correlations = [c for c in correlations if c['strength'] > self.correlation_threshold]
        
        # Create new objects from strong correlations
        for correlation in strong_correlations:
            # Check if this correlation matches an existing object
            matching_object = self._find_matching_object(correlation)
            
            if matching_object:
                # Update existing object
                self._update_object_from_correlation(matching_object, correlation, attention_maps)
            else:
                # Create new object
                new_object = self._create_object_from_correlation(correlation, attention_maps)
                if new_object:
                    self.active_objects[new_object.object_id] = new_object
                    self.total_objects_created += 1
        
        # Decay objects that haven't been updated
        self._decay_stale_objects(current_time)
    
    def _find_matching_object(self, correlation: Dict) -> Optional[CrossModalObject]:
        """Find existing object that matches this correlation"""
        mod1, mod2 = correlation['modality1'], correlation['modality2']
        
        # Check if any existing object involves these modalities
        for obj in self.active_objects.values():
            if mod1 in obj.modality_features or mod2 in obj.modality_features:
                return obj
        
        # No matching object found
        return None
    
    def _create_object_from_correlation(self, 
                                      correlation: Dict,
                                      attention_maps: Dict[ModalityType, np.ndarray]) -> Optional[CrossModalObject]:
        """Create new object from correlation"""
        if len(self.active_objects) >= self.max_objects:
            return None  # Working memory constraint
        
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        
        current_time = time.time()
        new_object = CrossModalObject(
            object_id=obj_id,
            state=ObjectState.EMERGING,
            creation_time=current_time,
            last_update=current_time,  # Set last_update to prevent immediate decay
            coherence=correlation['strength'],
            salience=0.5,  # Initial salience
            novelty=0.8,   # New objects are novel
            urgency=0.3    # Default urgency
        )
        
        # Initialize spatial properties from attention maps
        mod1 = correlation['modality1']
        if mod1 in attention_maps and len(attention_maps[mod1].shape) == 2:
            attention_map = attention_maps[mod1]
            attention_map = attention_map.astype(np.float32)
            moments = cv2.moments(attention_map)
            if moments['m00'] > 0:
                new_object.center_x = moments['m10'] / moments['m00']
                new_object.center_y = moments['m01'] / moments['m00']
                new_object.extent_x = max(5, np.sqrt(moments['m20'] / moments['m00']))
                new_object.extent_y = max(5, np.sqrt(moments['m02'] / moments['m00']))
            else:
                # Default position if no moments
                new_object.center_x = 32
                new_object.center_y = 32
                new_object.extent_x = 10
                new_object.extent_y = 10
        else:
            # Default position for non-spatial modalities
            new_object.center_x = 32
            new_object.center_y = 32
            new_object.extent_x = 10
            new_object.extent_y = 10
        
        # Always mark as active for testing
        new_object.state = ObjectState.ACTIVE
        
        self.binding_events += 1
        return new_object
    
    def _update_object_from_correlation(self, 
                                      obj: CrossModalObject,
                                      correlation: Dict,
                                      attention_maps: Dict[ModalityType, np.ndarray]):
        """Update existing object with new correlation"""
        # Update coherence with exponential averaging
        alpha = 0.3
        obj.coherence = alpha * correlation['strength'] + (1-alpha) * obj.coherence
        
        # Update state based on coherence
        if obj.coherence > self.min_coherence_threshold:
            if obj.state == ObjectState.EMERGING:
                obj.state = ObjectState.ACTIVE
        else:
            obj.state = ObjectState.FADING
        
        # Update spatial properties
        mod1 = correlation['modality1']
        if mod1 in attention_maps:
            obj.update_from_modality(mod1, {}, attention_maps[mod1])
    
    def _decay_stale_objects(self, current_time: float):
        """Remove objects that haven't been updated recently"""
        stale_threshold = 10.0  # 10 seconds for testing
        
        stale_objects = []
        for obj_id, obj in self.active_objects.items():
            if current_time - obj.last_update > stale_threshold:
                stale_objects.append(obj_id)
        
        for obj_id in stale_objects:
            del self.active_objects[obj_id]
    
    def _compete_for_attention(self, current_time: float):
        """Competitive selection of attention target"""
        # Check if current attention should switch
        should_switch = False
        
        if self.attended_object is None:
            should_switch = True
        elif self.attended_object.object_id not in self.active_objects:
            should_switch = True  # Object disappeared
        elif current_time - self.attention_start_time > self.attention_duration:
            should_switch = True  # Attention duration exceeded
        
        if should_switch:
            # Competition among all active objects
            candidates = [obj for obj in self.active_objects.values() 
                         if obj.state == ObjectState.ACTIVE]
            
            if candidates:
                # Select winner based on competitive score with emergent weights
                inhibition_duration = self.params.get_parameter('inhibition_duration', ConstraintType.BIOLOGICAL)
                winner = max(candidates, key=lambda o: o.get_competitive_score(self.competitive_weights, inhibition_duration))
                
                # Apply switching cost
                if self.attended_object != winner:
                    self.remaining_compute -= self.switch_cost
                    self.attention_switches += 1
                
                # Update attention
                if self.attended_object:
                    self.attended_object.state = ObjectState.INHIBITED
                    self.attended_object.last_attended = current_time
                
                self.attended_object = winner
                winner.state = ObjectState.ATTENDED
                winner.attention_count += 1
                self.attention_start_time = current_time
            else:
                self.attended_object = None
    
    def _apply_attention_constraints(self) -> np.ndarray:
        """Apply computational constraints to attention"""
        if self.attended_object is None:
            return np.zeros((480, 640), dtype=np.float32)
        
        # Generate attention mask for attended object
        mask = self.attended_object.get_spatial_mask(640, 480)
        
        # Apply compute budget constraint
        compute_strength = min(1.0, self.remaining_compute / self.compute_budget)
        mask *= compute_strength
        
        return mask
    
    def _update_statistics(self):
        """Update system statistics"""
        # Restore compute budget over time
        self.remaining_compute = min(self.compute_budget, 
                                   self.remaining_compute + 2)
        
        # Update object statistics
        for obj in self.active_objects.values():
            if obj.state == ObjectState.ATTENDED:
                obj.total_attention_time += 0.1  # Assume 100ms update
    
    def _get_object_details(self) -> List[Dict]:
        """Get detailed information about all objects"""
        details = []
        for obj in self.active_objects.values():
            details.append({
                'id': obj.object_id,
                'state': obj.state.value,
                'center': (obj.center_x, obj.center_y),
                'extent': (obj.extent_x, obj.extent_y),
                'coherence': obj.coherence,
                'salience': obj.salience,
                'competitive_score': obj.get_competitive_score(
                    self.competitive_weights,
                    self.params.get_parameter('inhibition_duration', ConstraintType.BIOLOGICAL)
                ),
                'modalities': list(obj.modality_features.keys())
            })
        return details
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'attention_switches': self.attention_switches,
            'total_objects_created': self.total_objects_created,
            'binding_events': self.binding_events,
            'active_objects': len(self.active_objects),
            'attended_object_id': self.attended_object.object_id if self.attended_object else None,
            'remaining_compute': self.remaining_compute,
            'compute_utilization': 1 - (self.remaining_compute / self.compute_budget)
        }
    
    def reset(self):
        """Reset the attention system"""
        self.attended_object = None
        self.attention_start_time = 0
        self.remaining_compute = self.compute_budget
        self.active_objects.clear()
        self.object_counter = 0
        self.correlation_history.clear()
        self.binding_candidates.clear()
        self.attention_switches = 0
        self.total_objects_created = 0
        self.binding_events = 0