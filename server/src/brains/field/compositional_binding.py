"""
Compositional Binding - The Language of Thought

Enables binding of discrete concepts through phase synchronization.
Like an orchestra where different sections play in harmony.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class Binding:
    """Represents a binding between multiple concepts."""
    concept_positions: List[torch.Tensor]  # Positions of bound concepts
    phase: float                           # Common phase for synchronization
    strength: float                        # Binding strength
    binding_id: int                        # Unique ID for this binding


class CompositionalBinding:
    """
    Enable composition of concepts through phase synchronization.
    
    Core principle: Concepts that need to be bound together oscillate
    in phase, while separate thoughts oscillate out of phase.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device):
        """
        Initialize compositional binding system.
        
        Args:
            field_shape: Shape of the field tensor [D, H, W, C]
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Phase field for synchronization
        self.phase_field = torch.zeros(field_shape[:3], device=device)
        self.phase_velocity = torch.zeros(field_shape[:3], device=device)
        
        # Binding parameters
        self.base_frequency = 40.0  # Hz - gamma band for binding
        self.coupling_strength = 0.1  # How strongly phases couple
        
        # Track active bindings
        self.bindings: List[Binding] = []
        self.next_binding_id = 0
        
        # Binding map (which locations belong to which binding)
        self.binding_map = torch.zeros(field_shape[:3], dtype=torch.int32, device=device)
        
    def bind_concepts(self, field: torch.Tensor, 
                     concept_positions: List[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """
        Bind multiple concepts together through phase synchronization.
        
        Args:
            field: Current field state
            concept_positions: List of positions to bind
            
        Returns:
            Updated field and binding ID
        """
        if len(concept_positions) < 2:
            return field, -1  # Need at least 2 concepts to bind
        
        # Create new binding
        binding_id = self.next_binding_id
        self.next_binding_id += 1
        
        # Assign common phase to all concepts in binding
        common_phase = torch.rand(1, device=self.device).item() * 2 * torch.pi
        
        # Synchronize phases in regions around concept positions
        for pos in concept_positions:
            x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
            
            # Define influence region
            radius = 3
            x_min = max(0, x - radius)
            x_max = min(self.field_shape[0], x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(self.field_shape[1], y + radius + 1)
            z_min = max(0, z - radius)
            z_max = min(self.field_shape[2], z + radius + 1)
            
            # Set phase in region
            self.phase_field[x_min:x_max, y_min:y_max, z_min:z_max] = common_phase
            
            # Mark in binding map
            self.binding_map[x_min:x_max, y_min:y_max, z_min:z_max] = binding_id
        
        # Create binding record
        new_binding = Binding(
            concept_positions=concept_positions,
            phase=common_phase,
            strength=1.0,
            binding_id=binding_id
        )
        self.bindings.append(new_binding)
        
        # Apply synchronized oscillation to bound regions
        field = self._apply_synchronized_oscillation(field, new_binding)
        
        return field, binding_id
    
    def unbind(self, binding_id: int) -> None:
        """
        Unbind concepts by desynchronizing their phases.
        
        Args:
            binding_id: ID of binding to break
        """
        # Find binding
        binding = None
        for b in self.bindings:
            if b.binding_id == binding_id:
                binding = b
                break
        
        if binding is None:
            return
        
        # Desynchronize phases
        for i, pos in enumerate(binding.concept_positions):
            x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
            
            # Assign different phase to each concept
            new_phase = i * 2 * torch.pi / len(binding.concept_positions)
            
            radius = 3
            x_min = max(0, x - radius)
            x_max = min(self.field_shape[0], x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(self.field_shape[1], y + radius + 1)
            z_min = max(0, z - radius)
            z_max = min(self.field_shape[2], z + radius + 1)
            
            self.phase_field[x_min:x_max, y_min:y_max, z_min:z_max] = new_phase
            
            # Clear binding map
            mask = self.binding_map[x_min:x_max, y_min:y_max, z_min:z_max] == binding_id
            self.binding_map[x_min:x_max, y_min:y_max, z_min:z_max][mask] = 0
        
        # Remove binding
        self.bindings = [b for b in self.bindings if b.binding_id != binding_id]
    
    def evolve_phases(self, field: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Evolve phase dynamics and apply to field.
        
        Bound regions maintain synchrony, unbound regions drift apart.
        
        Args:
            field: Current field state
            dt: Time step
            
        Returns:
            Field with phase dynamics applied
        """
        # Update phase velocities based on coupling
        self._update_phase_coupling()
        
        # Evolve phases
        self.phase_field += self.phase_velocity * dt
        self.phase_field = torch.fmod(self.phase_field, 2 * torch.pi)
        
        # Apply oscillations to field
        for binding in self.bindings:
            field = self._apply_synchronized_oscillation(field, binding)
        
        # Apply desynchronized oscillation to unbound regions
        unbound_mask = self.binding_map == 0
        if unbound_mask.any():
            # Random phase drift for unbound regions
            drift = torch.randn_like(self.phase_field) * 0.1
            self.phase_field[unbound_mask] += drift[unbound_mask]
            
            # Weaker oscillation for unbound regions
            oscillation = torch.sin(self.phase_field) * 0.05
            field[..., :] *= (1 + oscillation.unsqueeze(-1))
        
        return field
    
    def _update_phase_coupling(self):
        """
        Update phase velocities based on coupling between bound regions.
        
        Kuramoto model: phases in same binding attract each other.
        """
        # Reset velocities
        self.phase_velocity.zero_()
        
        for binding in self.bindings:
            if len(binding.concept_positions) < 2:
                continue
            
            # Compute mean phase of binding
            total_phase = 0
            count = 0
            
            for pos in binding.concept_positions:
                x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
                total_phase += self.phase_field[x, y, z].item()
                count += 1
            
            mean_phase = total_phase / count if count > 0 else 0
            
            # Pull all phases toward mean (Kuramoto coupling)
            for pos in binding.concept_positions:
                x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
                
                radius = 3
                x_min = max(0, x - radius)
                x_max = min(self.field_shape[0], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(self.field_shape[1], y + radius + 1)
                z_min = max(0, z - radius)
                z_max = min(self.field_shape[2], z + radius + 1)
                
                # Phase difference
                phase_diff = mean_phase - self.phase_field[x_min:x_max, y_min:y_max, z_min:z_max]
                
                # Kuramoto coupling
                self.phase_velocity[x_min:x_max, y_min:y_max, z_min:z_max] += (
                    self.coupling_strength * torch.sin(phase_diff) * binding.strength
                )
        
        # Add base frequency
        self.phase_velocity += self.base_frequency
    
    def _apply_synchronized_oscillation(self, field: torch.Tensor, 
                                       binding: Binding) -> torch.Tensor:
        """
        Apply synchronized oscillation to bound concepts.
        
        Modulates field amplitude with phase-locked oscillation.
        """
        # Create mask for this binding
        binding_mask = (self.binding_map == binding.binding_id).unsqueeze(-1)
        
        if binding_mask.any():
            # Phase-locked oscillation
            oscillation = torch.sin(self.phase_field + binding.phase).unsqueeze(-1)
            
            # Stronger oscillation for bound regions (enhances them)
            modulation = 1 + 0.2 * oscillation * binding.strength
            
            # Apply only to bound regions
            field = torch.where(binding_mask, field * modulation, field)
        
        return field
    
    def get_binding_strength(self, pos1: torch.Tensor, pos2: torch.Tensor) -> float:
        """
        Measure binding strength between two positions.
        
        Based on phase coherence.
        
        Args:
            pos1, pos2: Positions to check
            
        Returns:
            Binding strength (0 = unbound, 1 = perfectly bound)
        """
        x1, y1, z1 = pos1[0].item(), pos1[1].item(), pos1[2].item()
        x2, y2, z2 = pos2[0].item(), pos2[1].item(), pos2[2].item()
        
        phase1 = self.phase_field[x1, y1, z1]
        phase2 = self.phase_field[x2, y2, z2]
        
        # Phase coherence (1 = same phase, 0 = opposite phase)
        coherence = (1 + torch.cos(phase1 - phase2)) / 2
        
        return coherence.item()
    
    def find_bound_concepts(self, position: torch.Tensor) -> List[torch.Tensor]:
        """
        Find all concepts bound to the concept at given position.
        
        Args:
            position: Position to check
            
        Returns:
            List of positions bound to this one
        """
        x, y, z = position[0].item(), position[1].item(), position[2].item()
        binding_id = self.binding_map[x, y, z].item()
        
        if binding_id == 0:
            return []  # Not bound to anything
        
        # Find binding
        for binding in self.bindings:
            if binding.binding_id == binding_id:
                return binding.concept_positions
        
        return []
    
    def merge_bindings(self, binding_ids: List[int]) -> int:
        """
        Merge multiple bindings into one.
        
        Args:
            binding_ids: IDs of bindings to merge
            
        Returns:
            New binding ID
        """
        if len(binding_ids) < 2:
            return binding_ids[0] if binding_ids else -1
        
        # Collect all positions from bindings
        all_positions = []
        for binding_id in binding_ids:
            for binding in self.bindings:
                if binding.binding_id == binding_id:
                    all_positions.extend(binding.concept_positions)
                    break
        
        # Remove old bindings
        for binding_id in binding_ids:
            self.unbind(binding_id)
        
        # Create new merged binding
        if all_positions:
            _, new_id = self.bind_concepts(
                torch.zeros(self.field_shape, device=self.device),  # Dummy field
                all_positions
            )
            return new_id
        
        return -1