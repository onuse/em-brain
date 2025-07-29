"""
Reward Topology Shaping

Minimal system where rewards create persistent deformations in field topology.
Goals emerge from the modified landscape without explicit representation.

Key insight: Just as gravity warps spacetime, creating paths that objects
naturally follow, rewards warp the field topology, creating paths that
thoughts naturally follow.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np


class RewardTopologyShaper:
    """
    Shapes field topology based on reward experiences.
    
    This is NOT a goal system or planner. It simply allows rewards
    to create persistent "impressions" in the field landscape that
    future activity naturally flows toward.
    
    Emergence expected:
    - Goal-seeking behavior without explicit goals
    - Value learning without value functions  
    - Intentionality without intentions
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 device: torch.device = torch.device('cpu'),
                 persistence_factor: float = 0.95,
                 max_attractors: int = 20):
        """
        Initialize topology shaper.
        
        Args:
            field_shape: Shape of the unified field
            device: Computation device
            persistence_factor: How slowly attractors fade (0.95 = slow fade)
            max_attractors: Maximum concurrent attractors (old ones fade)
        """
        self.field_shape = field_shape
        self.device = device
        self.persistence_factor = persistence_factor
        self.max_attractors = max_attractors
        
        # Attractor storage - these warp the field topology
        self.attractors = deque(maxlen=max_attractors)
        
        # Reward history for learning
        self.reward_history = deque(maxlen=1000)
        
    def process_reward(self, 
                      current_field: torch.Tensor,
                      reward: float,
                      threshold: float = 0.1) -> Optional[torch.Tensor]:
        """
        Process reward and create field deformation if significant.
        
        This is the KEY METHOD - it creates lasting changes in the field
        topology proportional to reward magnitude.
        
        Args:
            current_field: Current field state when reward occurred
            reward: Reward value (-1 to 1 typically)
            threshold: Minimum reward magnitude to create attractor
            
        Returns:
            Field deformation tensor if reward significant, None otherwise
        """
        # Only significant rewards create lasting impressions
        if abs(reward) < threshold:
            return None
            
        # The current field pattern becomes an attractor/repulsor
        # Positive rewards create attractors (field flows toward)
        # Negative rewards create repulsors (field flows away)
        
        # Create attractor with current field pattern
        attractor = {
            'pattern': current_field.detach().clone(),
            'strength': abs(reward),
            'valence': np.sign(reward),  # +1 attract, -1 repel
            'age': 0,
            'persistence': self.persistence_factor
        }
        
        self.attractors.append(attractor)
        self.reward_history.append(reward)
        
        # Return immediate deformation (optional, for visualization)
        return self._compute_deformation(current_field, attractor)
        
    def apply_topology_influence(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply all active attractors to warp field evolution.
        
        This is called during field evolution and creates the "downhill"
        gradients that make the field naturally flow toward rewarded states.
        
        CRITICAL: This single function enables goal-seeking behavior
        without any explicit goal representation or planning.
        
        Args:
            field: Current field state
            
        Returns:
            Topology influence to add to field evolution
        """
        if not self.attractors:
            return torch.zeros_like(field)
            
        total_influence = torch.zeros_like(field)
        
        # Each attractor creates a "force" on the field
        for attractor in self.attractors:
            # Age the attractor
            attractor['age'] += 1
            
            # Compute influence strength with decay
            strength = attractor['strength'] * (attractor['persistence'] ** attractor['age'])
            
            if strength < 0.001:  # Too weak to matter
                continue
                
            # Compute "gradient" toward (or away from) this attractor
            influence = self._compute_attractor_influence(
                field, 
                attractor['pattern'],
                strength,
                attractor['valence']
            )
            
            total_influence += influence
            
        # Clean up dead attractors
        self.attractors = deque(
            [a for a in self.attractors if a['strength'] * (a['persistence'] ** a['age']) >= 0.001],
            maxlen=self.max_attractors
        )
        
        return total_influence
        
    def _compute_attractor_influence(self,
                                   current_field: torch.Tensor,
                                   attractor_pattern: torch.Tensor,
                                   strength: float,
                                   valence: float) -> torch.Tensor:
        """
        Compute how an attractor influences the current field.
        
        This creates a "gradient" in field space, not physical space.
        The field naturally evolves to increase similarity to attractors
        and decrease similarity to repulsors.
        """
        # Compute difference vector in field space
        difference = attractor_pattern - current_field
        
        # Scale by strength and valence
        # Positive valence: field pulled toward attractor
        # Negative valence: field pushed away from repulsor
        influence = difference * strength * valence * 0.01
        
        # Add some nonlinearity - stronger influence when closer
        # This creates "basins of attraction"
        similarity = F.cosine_similarity(
            current_field.flatten().unsqueeze(0),
            attractor_pattern.flatten().unsqueeze(0)
        ).item()
        
        # Attractors pull stronger when close, repulsors push stronger when close
        if valence > 0:
            # Attractor - stronger pull when more similar
            influence *= (1 + similarity) / 2
        else:
            # Repulsor - stronger push when more similar  
            influence *= (1 - similarity) / 2
            
        return influence
        
    def _compute_deformation(self,
                           field: torch.Tensor,
                           attractor: Dict) -> torch.Tensor:
        """
        Compute immediate field deformation from new attractor.
        
        This is optional - just for visualization or immediate response.
        The real magic happens in apply_topology_influence during evolution.
        """
        return self._compute_attractor_influence(
            field,
            attractor['pattern'],
            attractor['strength'] * 0.1,  # Immediate effect is smaller
            attractor['valence']
        )
        
    def get_topology_state(self) -> Dict:
        """Get current topology shaping state for monitoring."""
        return {
            'active_attractors': len(self.attractors),
            'total_rewards': len(self.reward_history),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'strongest_attractor': max([a['strength'] * (a['persistence'] ** a['age']) 
                                      for a in self.attractors], default=0)
        }