#!/usr/bin/env python3
"""
Emergent Spatial Dynamics

Implements truly emergent spatial understanding where "places" and navigation
arise from field dynamics, not from coordinate systems.

Key concepts:
- Places are stable field configurations (attractors)
- Movement emerges from field evolution toward desired states
- Spatial relationships learned through experience
- No fixed coordinates - only field patterns and their relationships
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import math

from .field_types import UnifiedFieldExperience, FieldNativeAction, FieldDynamicsFamily
from ...parameters.cognitive_config import get_cognitive_config


@dataclass
class EmergentPlace:
    """A 'place' defined by field configuration, not coordinates."""
    place_id: str
    field_signature: torch.Tensor  # Complete field state that defines this place
    stability_score: float         # How stable/recognizable this configuration is
    visit_count: int              # Strengthens with each visit
    last_visit_time: float        # For recency tracking
    connections: Dict[str, float]  # Learned transitions to other places
    sensory_signature: Optional[torch.Tensor] = None  # Associated sensory pattern


@dataclass
class FieldMotorCoupling:
    """Maps field evolution patterns to motor tendencies."""
    oscillatory_to_forward: float = 1.0    # How oscillations map to forward motion
    flow_to_rotation: float = 1.0          # How flow patterns map to turning
    energy_to_speed: float = 0.5           # How field energy maps to speed
    tension_to_urgency: float = 2.0        # How field tension maps to action urgency


class EmergentSpatialDynamics:
    """
    Implements spatial understanding through field dynamics rather than coordinates.
    
    Core principles:
    1. Sensory patterns create field impressions (not coordinate mappings)
    2. Similar experiences create similar field states (natural clustering)
    3. Movement emerges from field evolution (not gradient following)
    4. Places are learned as stable attractors (not stored positions)
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 device: torch.device,
                 quiet_mode: bool = False):
        """
        Initialize emergent spatial dynamics.
        
        Args:
            field_shape: Shape of the unified field tensor
            device: Computation device
            quiet_mode: Suppress debug output
        """
        self.field_shape = field_shape
        self.device = device
        self.quiet_mode = quiet_mode
        
        # Load cognitive configuration
        self.cognitive_config = get_cognitive_config()
        
        # Place recognition and memory
        self.known_places: Dict[str, EmergentPlace] = {}
        self.current_place: Optional[str] = None
        self.place_recognition_threshold = 0.8  # Cosine similarity threshold
        
        # Field-motor coupling parameters
        self.motor_coupling = FieldMotorCoupling()
        
        # Navigation state
        self.navigation_target: Optional[EmergentPlace] = None
        self.field_tension: Optional[torch.Tensor] = None
        self.navigation_history = deque(maxlen=100)
        
        # Learning parameters
        self.place_stability_threshold = 0.7
        self.transition_learning_rate = 0.1
        
        # Statistics
        self.places_discovered = 0
        self.successful_navigations = 0
        self.total_navigations = 0
        
        if not quiet_mode:
            print(f"🗺️  Emergent Spatial Dynamics initialized")
            print(f"   Field shape: {field_shape}")
            print(f"   Device: {device}")
            print(f"   Places emerge from field stability, not coordinates")
    
    def process_spatial_experience(self, 
                                 current_field: torch.Tensor,
                                 sensory_input: Any,  # Can be list or tensor
                                 reward: float = 0.0) -> Dict[str, Any]:
        """
        Process current experience for spatial understanding.
        
        Args:
            current_field: Current unified field state
            sensory_input: Current sensory input
            reward: Reward signal for place importance
            
        Returns:
            Spatial understanding state including place recognition
        """
        # 1. Save previous place for transition learning
        previous_place = self.current_place
        
        # 2. Check if current field state matches a known place
        recognized_place = self._recognize_place(current_field)
        
        # 3. Update or create place based on field stability
        if self._is_field_stable(current_field):
            if recognized_place:
                self._strengthen_place(recognized_place, sensory_input, reward)
                self.current_place = recognized_place
            else:
                # Discover new place
                new_place = self._create_place(current_field, sensory_input, reward)
                self.current_place = new_place.place_id
                if not self.quiet_mode:
                    print(f"🏔️  Discovered new place: {new_place.place_id}")
        
        # 4. Learn transitions between places
        if previous_place and self.current_place and previous_place != self.current_place:
            self._learn_transition(previous_place, self.current_place)
        
        return {
            'current_place': self.current_place,
            'previous_place': previous_place,
            'known_places': len(self.known_places),
            'field_stability': self._calculate_field_stability(current_field),
            'navigation_active': self.navigation_target is not None
        }
    
    def navigate_to_place(self, target_place_id: str) -> bool:
        """
        Initiate navigation to a known place.
        
        Args:
            target_place_id: ID of the target place
            
        Returns:
            True if navigation initiated, False if place unknown
        """
        if target_place_id not in self.known_places:
            return False
        
        self.navigation_target = self.known_places[target_place_id]
        self.total_navigations += 1
        
        if not self.quiet_mode:
            print(f"🧭 Navigating to place: {target_place_id}")
        
        return True
    
    def compute_motor_emergence(self, 
                              current_field: torch.Tensor,
                              field_evolution: torch.Tensor) -> FieldNativeAction:
        """
        Generate motor commands from field evolution patterns.
        
        This is the key innovation: movement emerges from how the field
        wants to evolve, not from following spatial gradients.
        
        Args:
            current_field: Current field state
            field_evolution: How the field evolved in this cycle
            
        Returns:
            Motor action emerging from field dynamics
        """
        # 1. Calculate field tension if navigating
        if self.navigation_target:
            self.field_tension = self._calculate_field_tension(
                current_field, 
                self.navigation_target.field_signature
            )
        
        # 2. Extract movement patterns from field evolution
        motor_tendencies = self._extract_motor_tendencies(field_evolution)
        
        # 3. Modulate by navigation tension if active
        if self.field_tension is not None and self.navigation_target:
            motor_tendencies = self._apply_navigation_modulation(
                motor_tendencies, 
                self.field_tension
            )
        
        # 4. Convert to motor commands
        motor_commands = self._tendencies_to_motor_commands(motor_tendencies)
        
        # 5. Check if we've reached the target
        if self.navigation_target and self.current_place == self.navigation_target.place_id:
            self.successful_navigations += 1
            self.navigation_target = None
            self.field_tension = None
            if not self.quiet_mode:
                print(f"✅ Reached destination!")
        
        return FieldNativeAction(
            timestamp=time.time(),
            output_stream=motor_commands,  # Use correct field name
            field_gradients=torch.zeros(3, device=self.device),  # Not using coordinate gradients!
            confidence=motor_tendencies['confidence'],
            dynamics_family_contributions=self._get_dynamics_contributions(motor_tendencies)
        )
    
    def _recognize_place(self, current_field: torch.Tensor) -> Optional[str]:
        """Recognize if current field matches a known place."""
        best_match = None
        best_similarity = 0.0
        
        for place_id, place in self.known_places.items():
            similarity = self._field_similarity(current_field, place.field_signature)
            if similarity > self.place_recognition_threshold and similarity > best_similarity:
                best_match = place_id
                best_similarity = similarity
        
        return best_match
    
    def _is_field_stable(self, field: torch.Tensor) -> bool:
        """Check if field is stable enough to be considered a place."""
        # Use field energy and variance as stability metrics
        field_energy = torch.mean(torch.abs(field)).item()
        field_variance = torch.var(field).item()
        
        # Stable if energy is above minimal threshold and variance is reasonable
        # Lower thresholds for initial place discovery
        stability = field_energy > 0.001 and field_variance < 1.0
        
        if not self.quiet_mode and not stability:
            print(f"Field not stable: energy={field_energy:.3f}, variance={field_variance:.3f}")
        
        return stability
    
    def _create_place(self, field: torch.Tensor, sensory: Any, reward: float) -> EmergentPlace:
        """Create a new place from current field configuration."""
        self.places_discovered += 1
        place_id = f"place_{self.places_discovered}"
        
        # Importance affects initial stability
        stability = self.place_stability_threshold + reward * 0.2
        
        # Convert sensory to tensor if it's a list
        if isinstance(sensory, list):
            sensory_tensor = torch.tensor(sensory, dtype=torch.float32, device=self.device)
        else:
            sensory_tensor = sensory
        
        new_place = EmergentPlace(
            place_id=place_id,
            field_signature=field.clone().detach(),
            stability_score=stability,
            visit_count=1,
            last_visit_time=time.time(),
            connections={},
            sensory_signature=sensory_tensor.clone().detach()
        )
        
        self.known_places[place_id] = new_place
        return new_place
    
    def _strengthen_place(self, place_id: str, sensory: Any, reward: float):
        """Strengthen an existing place through revisiting."""
        place = self.known_places[place_id]
        place.visit_count += 1
        place.last_visit_time = time.time()
        place.stability_score = min(1.0, place.stability_score + 0.05 + reward * 0.1)
        
        # Convert sensory to tensor if it's a list
        if isinstance(sensory, list):
            sensory_tensor = torch.tensor(sensory, dtype=torch.float32, device=self.device)
        else:
            sensory_tensor = sensory
        
        # Update sensory signature with moving average
        if place.sensory_signature is not None:
            place.sensory_signature = 0.9 * place.sensory_signature + 0.1 * sensory_tensor
    
    def _learn_transition(self, from_place: str, to_place: str):
        """Learn connection between two places."""
        if from_place in self.known_places:
            place = self.known_places[from_place]
            if to_place in place.connections:
                # Strengthen existing connection
                place.connections[to_place] = min(1.0, 
                    place.connections[to_place] + self.transition_learning_rate)
            else:
                # Create new connection
                place.connections[to_place] = self.transition_learning_rate
            
            self.navigation_history.append((from_place, to_place, time.time()))
    
    def _calculate_field_tension(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate tension between current field and target field.
        
        This tension drives field evolution toward the target state.
        """
        # Simple difference creates basic tension
        tension = target - current
        
        # Weight by target stability (more stable = stronger attraction)
        if self.navigation_target:
            tension *= self.navigation_target.stability_score
        
        return tension
    
    def _extract_motor_tendencies(self, field_evolution: torch.Tensor) -> Dict[str, float]:
        """
        Extract motor tendencies from field evolution patterns.
        
        This is where movement emerges from field dynamics!
        """
        # Reshape field evolution to access different dimension families
        # Assuming first few dimensions are spatial-related
        
        # 1. Oscillatory patterns suggest forward/backward rhythm
        oscillatory_dims = min(3, len(self.field_shape))
        if len(field_evolution.shape) > oscillatory_dims:
            oscillatory_change = field_evolution.reshape(-1)[oscillatory_dims:oscillatory_dims+3]
            oscillatory_energy = torch.mean(torch.abs(oscillatory_change)).item()
            
            # Phase determines direction
            oscillatory_phase = torch.mean(oscillatory_change).item()
            forward_tendency = oscillatory_phase * oscillatory_energy * self.motor_coupling.oscillatory_to_forward
        else:
            forward_tendency = 0.0
        
        # 2. Flow patterns suggest turning
        flow_start = min(6, len(field_evolution.reshape(-1)))
        if len(field_evolution.shape) > flow_start:
            flow_change = field_evolution.reshape(-1)[flow_start:flow_start+3]
            # Asymmetry in flow suggests turning
            flow_asymmetry = (flow_change[0] - flow_change[-1]).item() if len(flow_change) > 1 else 0.0
            turn_tendency = flow_asymmetry * self.motor_coupling.flow_to_rotation
        else:
            turn_tendency = 0.0
        
        # 3. Overall energy suggests speed/urgency
        field_energy = torch.mean(torch.abs(field_evolution)).item()
        speed_tendency = field_energy * self.motor_coupling.energy_to_speed
        
        # 4. Field variance suggests confidence
        field_variance = torch.var(field_evolution).item()
        confidence = 1.0 / (1.0 + field_variance)
        
        return {
            'forward': forward_tendency,
            'turn': turn_tendency,
            'speed': speed_tendency,
            'confidence': confidence,
            'urgency': field_energy * self.motor_coupling.tension_to_urgency
        }
    
    def _apply_navigation_modulation(self, 
                                   tendencies: Dict[str, float], 
                                   tension: torch.Tensor) -> Dict[str, float]:
        """Modulate motor tendencies based on navigation tension."""
        # Tension direction influences turning
        tension_vector = tension.reshape(-1)[:3]  # Use first 3 dimensions
        tension_direction = torch.mean(tension_vector).item()
        
        # Modulate turning based on tension direction
        tendencies['turn'] += tension_direction * 0.5
        
        # Tension magnitude affects urgency
        tension_magnitude = torch.norm(tension).item()
        tendencies['urgency'] *= (1.0 + tension_magnitude * 0.5)
        
        return tendencies
    
    def _tendencies_to_motor_commands(self, tendencies: Dict[str, float]) -> torch.Tensor:
        """Convert motor tendencies to actual motor commands."""
        # Assume 4 motor dimensions: [forward, turn, speed_mod, action]
        motor_commands = torch.zeros(4, device=self.device)
        
        # Forward/backward (with speed modulation)
        motor_commands[0] = np.clip(tendencies['forward'] * tendencies['speed'], -1.0, 1.0)
        
        # Turning
        motor_commands[1] = np.clip(tendencies['turn'], -1.0, 1.0)
        
        # Speed modifier
        motor_commands[2] = np.clip(tendencies['speed'], 0.0, 1.0)
        
        # Action urgency
        motor_commands[3] = np.clip(tendencies['urgency'] * tendencies['confidence'], 0.0, 1.0)
        
        return motor_commands
    
    def _get_dynamics_contributions(self, tendencies: Dict[str, float]) -> Dict[FieldDynamicsFamily, float]:
        """Convert motor tendencies to dynamics family contributions."""
        return {
            FieldDynamicsFamily.OSCILLATORY: tendencies.get('forward', 0.0),
            FieldDynamicsFamily.FLOW: tendencies.get('turn', 0.0),
            FieldDynamicsFamily.ENERGY: tendencies.get('speed', 0.0),
            FieldDynamicsFamily.SPATIAL: 0.0,
            FieldDynamicsFamily.TOPOLOGY: 0.0,
            FieldDynamicsFamily.COUPLING: 0.0,
            FieldDynamicsFamily.EMERGENCE: tendencies.get('urgency', 0.0)
        }
    
    def _field_similarity(self, field1: torch.Tensor, field2: torch.Tensor) -> float:
        """Calculate similarity between two field configurations."""
        # Flatten and normalize
        f1_flat = field1.flatten()
        f2_flat = field2.flatten()
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            f1_flat.unsqueeze(0), 
            f2_flat.unsqueeze(0)
        ).item()
        
        return similarity
    
    def _calculate_field_stability(self, field: torch.Tensor) -> float:
        """Calculate overall field stability metric."""
        energy = torch.mean(torch.abs(field)).item()
        variance = torch.var(field).item()
        
        # Stability is high when energy is moderate and variance is low
        stability = (1.0 / (1.0 + variance)) * min(1.0, energy * 2.0)
        return stability
    
    def get_navigation_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get the learned navigation graph of places and connections."""
        graph = {}
        for place_id, place in self.known_places.items():
            connections = [(target, strength) for target, strength in place.connections.items()]
            graph[place_id] = sorted(connections, key=lambda x: x[1], reverse=True)
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get emergent spatial statistics."""
        success_rate = (self.successful_navigations / self.total_navigations 
                       if self.total_navigations > 0 else 0.0)
        
        return {
            'places_discovered': self.places_discovered,
            'current_place': self.current_place,
            'navigation_success_rate': success_rate,
            'total_navigations': self.total_navigations,
            'known_places': list(self.known_places.keys()),
            'navigation_graph': self.get_navigation_graph()
        }