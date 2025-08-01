#!/usr/bin/env python3
"""
Field Strategic Planner - Discovers beneficial field configurations

Instead of planning action sequences, this module discovers field patterns that,
when installed in memory channels, create behavioral attractors leading to good outcomes.

Key insight: Strategy is not what you DO, it's what you ARE - a field configuration
that naturally biases behavior through emergent dynamics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from ...utils.tensor_ops import create_zeros, create_randn


@dataclass
class StrategicPattern:
    """A field configuration that creates behavioral tendencies."""
    pattern: torch.Tensor  # Shape: [D, H, W, 16] for memory channels
    score: float  # How beneficial this pattern is
    behavioral_signature: torch.Tensor  # What behaviors emerge (summary)
    behavioral_trajectory: torch.Tensor  # Full trajectory [T, 4] for similarity
    persistence: float  # How long pattern remains influential
    context_embedding: torch.Tensor  # When this pattern works well
    creation_time: float = 0.0  # When pattern was discovered
    usage_count: int = 0  # How often pattern has been retrieved


class FieldStrategicPlanner:
    """
    Discovers field patterns that shape behavior through natural dynamics.
    
    This replaces explicit action planning with pattern discovery - finding
    field configurations that create beneficial behavioral attractors.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 sensory_dim: int,
                 motor_dim: int,
                 device: torch.device = None):
        """
        Initialize the strategic pattern discoverer.
        
        Args:
            field_shape: Shape of the unified field [D, H, W, C]
            sensory_dim: Dimension of sensory input
            motor_dim: Dimension of motor output
            device: Computation device
        """
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.device = device if device else torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory channels for strategic patterns (temporal features)
        self.pattern_channels = slice(32, 48)  # 16 channels
        self.n_pattern_channels = 16
        
        # Pattern discovery parameters
        self.simulation_horizon = 100  # How far to look ahead
        self.pattern_decay_rate = 0.97  # How patterns persist
        self.field_decay_rate = 0.995  # Overall field decay
        
        # Simplified field dynamics for pattern evaluation
        self.diffusion_kernel = self._create_diffusion_kernel()
        
        # Thread pool for async pattern discovery
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PatternDiscovery")
        self._discovery_lock = threading.Lock()
        self._current_discovery_future = None
        
        # Learned pattern library (will grow over time)
        self.pattern_library = []
        self.max_library_size = 30
        
    def _create_diffusion_kernel(self) -> torch.Tensor:
        """Create 3D diffusion kernel for field evolution."""
        kernel = torch.zeros(1, 1, 3, 3, 3, device=self.device)
        # Simple 3D Laplacian
        kernel[0, 0, 1, 1, 1] = -6.0  # Center
        kernel[0, 0, 0, 1, 1] = 1.0   # Neighbors
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        return kernel * 0.05  # Diffusion strength
        
    def discover_strategic_pattern(self,
                                 current_field: torch.Tensor,
                                 reward_signal: float,
                                 exploration_level: float = 0.5,
                                 n_candidates: int = 16) -> StrategicPattern:
        """
        Discover a field pattern that creates beneficial behavioral dynamics.
        
        This is the core innovation - instead of planning actions, we find
        field configurations that naturally lead to good outcomes.
        
        Args:
            current_field: Current brain state
            reward_signal: Recent reward/value signal (deprecated, kept for compatibility)
            exploration_level: How much to explore vs exploit known patterns
            n_candidates: Number of patterns to evaluate
            
        Returns:
            Best strategic pattern found
        """
        best_pattern = None
        best_score = -float('inf')
        
        # Extract current context for pattern selection
        context = self._extract_context_embedding(current_field)
        
        # Measure current tensions to guide pattern generation
        current_tensions = self._measure_field_tensions(current_field)
        
        for i in range(n_candidates):
            # Generate candidate pattern
            if i < n_candidates // 2 and self.pattern_library:
                # Sometimes use/modify known patterns
                candidate = self._generate_from_library(context, exploration_level)
            else:
                # Generate novel pattern targeted at dominant tension
                candidate = self._generate_tension_targeted_pattern(current_tensions)
            
            # Evaluate how this pattern shapes behavior
            score, behavioral_signature, trajectory = self._evaluate_pattern(
                current_field,
                candidate,
                reward_signal  # Will be replaced by tension-based evaluation
            )
            
            if score > best_score:
                best_score = score
                best_pattern = StrategicPattern(
                    pattern=candidate,
                    score=score,
                    behavioral_signature=behavioral_signature,
                    behavioral_trajectory=trajectory,
                    persistence=self._measure_pattern_persistence(candidate),
                    context_embedding=context,
                    creation_time=time.time()
                )
        
        # Add to library if good enough
        if best_pattern and best_pattern.score > 0:
            self._add_to_library(best_pattern)
            
        return best_pattern
    
    def _generate_novel_pattern(self) -> torch.Tensor:
        """
        Generate a novel strategic pattern.
        
        These patterns create different behavioral attractors:
        - Gradients: Directional movement
        - Radials: Centering/avoiding behaviors  
        - Waves: Oscillatory/patrol behaviors
        - Sparse: Focused attention
        """
        pattern = torch.zeros(
            self.field_shape[0],
            self.field_shape[1],
            self.field_shape[2],
            self.n_pattern_channels,
            device=self.device
        )
        
        pattern_type = torch.randint(0, 4, (1,)).item()
        
        if pattern_type == 0:
            # Gradient pattern - creates directional flow
            direction = torch.randn(3, device=self.device)
            direction = direction / (torch.norm(direction) + 1e-8)
            
            for i in range(self.field_shape[0]):
                for j in range(self.field_shape[1]):
                    for k in range(self.field_shape[2]):
                        pos = torch.tensor([i, j, k], device=self.device, dtype=torch.float32)
                        # Project position onto direction
                        projection = torch.dot(pos, direction)
                        pattern[i, j, k, :8] = projection / self.field_shape[0]
                        
        elif pattern_type == 1:
            # Radial pattern - creates centering/avoiding behavior
            center = torch.tensor([
                torch.randint(0, self.field_shape[0], (1,)).item(),
                torch.randint(0, self.field_shape[1], (1,)).item(),
                torch.randint(0, self.field_shape[2], (1,)).item()
            ], device=self.device, dtype=torch.float32)
            
            for i in range(self.field_shape[0]):
                for j in range(self.field_shape[1]):
                    for k in range(self.field_shape[2]):
                        pos = torch.tensor([i, j, k], device=self.device, dtype=torch.float32)
                        dist = torch.norm(pos - center)
                        # Gaussian activation
                        pattern[i, j, k, :8] = torch.exp(-dist**2 / (self.field_shape[0]**2 / 4))
                        
        elif pattern_type == 2:
            # Wave pattern - creates oscillatory behavior
            frequency = torch.rand(1, device=self.device) * 0.5 + 0.1
            phase = torch.rand(1, device=self.device) * 2 * np.pi
            axis = torch.randint(0, 3, (1,)).item()
            
            for i in range(self.field_shape[0]):
                for j in range(self.field_shape[1]):
                    for k in range(self.field_shape[2]):
                        pos = [i, j, k][axis]
                        pattern[i, j, k, 8:12] = torch.sin(pos * frequency + phase)
                        
        else:
            # Sparse activation - creates focused behavior
            n_peaks = torch.randint(3, 8, (1,)).item()
            for _ in range(n_peaks):
                # Random peak location
                pos = [
                    torch.randint(0, self.field_shape[0], (1,)).item(),
                    torch.randint(0, self.field_shape[1], (1,)).item(),
                    torch.randint(0, self.field_shape[2], (1,)).item()
                ]
                # Random feature subset
                feature_start = torch.randint(0, 12, (1,)).item()
                pattern[pos[0], pos[1], pos[2], feature_start:feature_start+4] = torch.randn(4, device=self.device)
        
        # Add small noise for variation
        pattern += torch.randn_like(pattern) * 0.1
        
        # Normalize to reasonable range
        pattern = torch.tanh(pattern)
        
        return pattern
    
    def _generate_tension_targeted_pattern(self, tensions: Dict[str, float]) -> torch.Tensor:
        """
        Generate a pattern specifically designed to address the dominant tension.
        
        Different tensions require different types of patterns:
        - Information tension → Exploration patterns (gradients, waves)
        - Learning tension → Novelty patterns (sparse, high-frequency)
        - Confidence tension → Stabilizing patterns (radial, coherent)
        - Prediction tension → Corrective patterns (targeted adjustments)
        - Novelty tension → Variation patterns (noise, perturbations)
        """
        pattern = torch.zeros(
            self.field_shape[0],
            self.field_shape[1],
            self.field_shape[2],
            self.n_pattern_channels,
            device=self.device
        )
        
        # Find dominant tension
        dominant_tension = max(tensions.items(), key=lambda x: x[1] if x[0] != 'total' else -1)[0]
        
        if dominant_tension == 'information':
            # Create exploration-inducing gradient pattern
            # Multiple directional flows to encourage movement
            for i in range(3):  # Multiple exploration directions
                direction = torch.randn(3, device=self.device)
                direction = direction / (torch.norm(direction) + 1e-8)
                strength = tensions['information']  # Stronger gradients for higher tension
                
                for x in range(self.field_shape[0]):
                    for y in range(self.field_shape[1]):
                        for z in range(self.field_shape[2]):
                            pos = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
                            projection = torch.dot(pos, direction)
                            pattern[x, y, z, i*4:(i+1)*4] = projection * strength / self.field_shape[0]
                            
        elif dominant_tension == 'learning':
            # Create novelty-inducing sparse activation pattern
            # High-frequency components to break out of plateaus
            n_novel_peaks = int(10 * tensions['learning'])  # More peaks for higher tension
            for _ in range(n_novel_peaks):
                # Random peak location
                peak_pos = [
                    torch.randint(0, self.field_shape[0], (1,)).item(),
                    torch.randint(0, self.field_shape[1], (1,)).item(),
                    torch.randint(0, self.field_shape[2], (1,)).item()
                ]
                # Random feature activation
                feature_idx = torch.randint(0, 12, (1,)).item()
                # Strong, localized activation
                pattern[peak_pos[0], peak_pos[1], peak_pos[2], feature_idx] = torch.randn(1, device=self.device).item() * 2.0
                
                # Add some spatial spread
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if abs(dx) + abs(dy) + abs(dz) == 1:  # Adjacent cells only
                                x, y, z = peak_pos[0] + dx, peak_pos[1] + dy, peak_pos[2] + dz
                                if 0 <= x < self.field_shape[0] and 0 <= y < self.field_shape[1] and 0 <= z < self.field_shape[2]:
                                    pattern[x, y, z, feature_idx] += torch.randn(1, device=self.device).item() * 0.5
                                    
        elif dominant_tension == 'confidence':
            # When confidence is low, create exploration-inducing gradients
            # NOT stabilizing patterns - we need movement to gather information!
            
            # Create multiple directional search patterns
            n_search_directions = 3
            for i in range(n_search_directions):
                # Random search direction
                direction = torch.randn(3, device=self.device)
                direction = direction / (torch.norm(direction) + 1e-8)
                
                # Create gradient along this direction with some noise
                for x in range(self.field_shape[0]):
                    for y in range(self.field_shape[1]):
                        for z in range(self.field_shape[2]):
                            pos = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
                            # Project position onto direction
                            projection = torch.dot(pos, direction)
                            # Add some spiral/wave component for search behavior
                            spiral = torch.sin(projection * 0.5) * 0.3
                            pattern[x, y, z, i*5:(i+1)*5] = (projection / self.field_shape[0] + spiral) * tensions['confidence']
                            
        elif dominant_tension == 'prediction':
            # Create corrective wave pattern
            # Oscillatory patterns that help recalibrate predictions
            frequency = 0.2 + tensions['prediction'] * 0.3  # Higher frequency for higher tension
            for axis in range(3):  # Waves along each axis
                phase = torch.rand(1, device=self.device) * 2 * np.pi
                
                for x in range(self.field_shape[0]):
                    for y in range(self.field_shape[1]):
                        for z in range(self.field_shape[2]):
                            pos_val = [x, y, z][axis] / self.field_shape[axis]
                            wave = torch.sin(pos_val * 2 * np.pi * frequency + phase)
                            pattern[x, y, z, 8 + axis*2:10 + axis*2] = wave * tensions['prediction']
                            
        elif dominant_tension == 'novelty':
            # Create variation-inducing perturbation pattern
            # Controlled chaos to break monotony
            base_pattern = torch.randn_like(pattern) * 0.3
            
            # Add multi-scale structured variation without interpolation
            for i in range(4):  # Different scales of variation
                scale = 2 ** i
                # Create variation at different scales using simple averaging
                for x in range(0, self.field_shape[0], scale):
                    for y in range(0, self.field_shape[1], scale):
                        for z in range(0, self.field_shape[2], scale):
                            # Random value for this block
                            block_value = torch.randn(1, device=self.device).item() * tensions['novelty'] * 0.5
                            # Apply to block
                            x_end = min(x + scale, self.field_shape[0])
                            y_end = min(y + scale, self.field_shape[1])
                            z_end = min(z + scale, self.field_shape[2])
                            pattern[x:x_end, y:y_end, z:z_end, i*3:(i+1)*3] += block_value
            
            pattern += base_pattern
        
        # Normalize pattern to reasonable range
        pattern = torch.tanh(pattern)
        
        # Add small general activation to ensure pattern has effect
        pattern += torch.randn_like(pattern) * 0.05
        
        return pattern
    
    def _evaluate_pattern(self,
                        base_field: torch.Tensor,
                        pattern: torch.Tensor,
                        reward_signal: float) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Evaluate how a pattern shapes field evolution and behavior.
        
        Good patterns create:
        1. Coherent behavioral trajectories
        2. Tension resolution (satisfying intrinsic drives)
        3. Stable influence over time
        
        Note: reward_signal parameter kept for compatibility but now we use
        tension-based evaluation for true autonomy.
        """
        # Use the new tension-based evaluation
        return self._evaluate_pattern_tension_based(base_field, pattern)
    
    def _evolve_field_with_pattern(self, 
                                  field: torch.Tensor,
                                  pattern: torch.Tensor) -> torch.Tensor:
        """
        Evolve field one step with strategic pattern influence.
        
        The pattern in memory channels creates gradients that shape
        the evolution of the entire field.
        """
        # 1. Apply standard field dynamics
        evolved = field * self.field_decay_rate
        
        # 2. Apply diffusion
        for c in range(field.shape[-1]):
            if c < 32 or c >= 48:  # Don't diffuse pattern channels
                channel = field[:, :, :, c].unsqueeze(0).unsqueeze(0)
                if channel.shape[2] >= 3:
                    diffused = F.conv3d(channel, self.diffusion_kernel, padding=1)
                    evolved[:, :, :, c] += diffused.squeeze() * 0.1
        
        # 3. Pattern persistence in memory channels
        evolved[:, :, :, self.pattern_channels] = (
            field[:, :, :, self.pattern_channels] * self.pattern_decay_rate +
            pattern * (1 - self.pattern_decay_rate) * 0.1  # Gentle refresh
        )
        
        # 4. Pattern influence on other channels
        # The pattern creates gradients that influence content channels
        pattern_energy = field[:, :, :, self.pattern_channels].mean(dim=-1, keepdim=True)
        gradient_influence = torch.tanh(pattern_energy) * 0.05
        evolved[:, :, :, :32] += gradient_influence.expand(-1, -1, -1, 32)
        
        # 5. Apply nonlinearity
        evolved = torch.tanh(evolved * 1.02)
        
        # 6. Small spontaneous activity
        evolved += torch.randn_like(evolved) * 0.01
        
        return evolved
    
    def _extract_behavioral_tendency(self, field: torch.Tensor) -> torch.Tensor:
        """
        Extract emergent behavioral tendency from field state.
        
        This is where behavior emerges from field configuration,
        not from explicit motor commands.
        """
        # Spatial gradients indicate movement tendencies
        # This is simplified - real implementation would use motor cortex
        
        # Get center of mass of high activation regions
        activation = field[:, :, :, :32].mean(dim=-1)  # Content channels
        
        # Compute spatial gradients
        if activation.shape[0] > 1:
            dx = activation[1:, :, :].mean() - activation[:-1, :, :].mean()
        else:
            dx = 0.0
            
        if activation.shape[1] > 1:
            dy = activation[:, 1:, :].mean() - activation[:, :-1, :].mean()
        else:
            dy = 0.0
            
        if activation.shape[2] > 1:
            dz = activation[:, :, 1:].mean() - activation[:, :, :-1].mean()
        else:
            dz = 0.0
        
        # Pattern influence on behavior
        pattern_activation = field[:, :, :, self.pattern_channels].mean()
        
        # Behavioral vector emerges from field gradients
        behavior = torch.tensor([
            dx * 2.0,  # Forward/backward tendency
            dy * 2.0,  # Left/right tendency  
            dz * 0.5,  # Up/down (less important)
            pattern_activation  # Overall activation level
        ], device=self.device)
        
        return torch.tanh(behavior)
    
    def _evaluate_field_state(self,
                            field: torch.Tensor,
                            behavior: torch.Tensor,
                            reward_signal: float) -> float:
        """Evaluate the value of a field state."""
        # Content channel energy (generally good)
        content_energy = field[:, :, :, :32].abs().mean().item()
        
        # Pattern stability
        pattern_coherence = field[:, :, :, self.pattern_channels].std().item()
        
        # Behavioral value (movement in rewarding direction)
        movement_value = behavior[0].item() * reward_signal
        
        # Exploration value (some activation is good)
        exploration_value = min(behavior[3].item(), 0.5) * 2.0
        
        return (
            content_energy * 0.3 +
            pattern_coherence * 0.2 +
            movement_value * 0.4 +
            exploration_value * 0.1
        )
    
    def _measure_behavioral_coherence(self, trajectory: List[torch.Tensor]) -> float:
        """Measure smoothness/coherence of behavioral trajectory."""
        if len(trajectory) < 2:
            return 0.0
            
        # Convert to tensor
        traj_tensor = torch.stack(trajectory)
        
        # Compute smoothness (low variation between steps)
        differences = torch.diff(traj_tensor, dim=0)
        smoothness = 1.0 / (1.0 + torch.mean(torch.abs(differences)).item())
        
        # Compute purposefulness (consistent direction)
        mean_direction = traj_tensor.mean(dim=0)
        consistency = torch.mean(
            torch.cosine_similarity(traj_tensor, mean_direction.unsqueeze(0), dim=1)
        ).item()
        
        return (smoothness + consistency) / 2.0
    
    def _measure_pattern_persistence(self, pattern: torch.Tensor) -> float:
        """Estimate how long a pattern will remain influential."""
        # Patterns with more structure persist longer
        structure_score = pattern.std().item()
        
        # Patterns with balanced activation persist longer  
        balance_score = 1.0 - torch.abs(pattern.mean()).item()
        
        # Convert to persistence estimate (in cycles)
        persistence = 10.0 + structure_score * 20.0 + balance_score * 10.0
        
        return min(persistence, 50.0)  # Cap at 50 cycles
    
    def _extract_context_embedding(self, field: torch.Tensor) -> torch.Tensor:
        """Extract compressed context from current field state."""
        # Simple context: statistics of content channels
        content = field[:, :, :, :32]
        
        context = torch.tensor([
            content.mean().item(),
            content.std().item(),
            content.abs().max().item(),
            field[:, :, :, 48:].mean().item() if field.shape[-1] > 48 else 0.0  # Error signal
        ], device=self.device)
        
        return context
    
    def _generate_from_library(self, 
                             context: torch.Tensor,
                             exploration: float) -> torch.Tensor:
        """
        Generate pattern through field resonance with stored patterns.
        
        Instead of simple lookup, patterns are selected through resonance -
        the current field state naturally activates similar stored patterns.
        """
        if not self.pattern_library:
            return self._generate_novel_pattern()
            
        # Compute resonance with all stored patterns
        resonances = []
        for stored_pattern in self.pattern_library:
            # Context similarity (field state resonance)
            context_resonance = F.cosine_similarity(
                context,
                stored_pattern.context_embedding,
                dim=0
            ).item()
            
            # Boost recently successful patterns
            recency_boost = np.exp(-0.01 * (time.time() - stored_pattern.creation_time) / 3600)
            
            # Combine with pattern strength and usage
            total_resonance = (
                context_resonance * 0.6 +  # Current context match
                stored_pattern.score * 0.3 * recency_boost +  # Success history
                min(stored_pattern.usage_count / 10, 1.0) * 0.1  # Familiarity
            )
            resonances.append(total_resonance)
        
        # Multiple patterns can resonate - blend top candidates
        resonances = np.array(resonances)
        
        if exploration < 0.3:  # Exploitation mode - use best match
            best_idx = np.argmax(resonances)
            base_pattern = self.pattern_library[best_idx].pattern.clone()
            self.pattern_library[best_idx].usage_count += 1
        else:  # Exploration mode - blend multiple patterns
            # Softmax to get blending weights
            resonance_weights = np.exp(resonances * 2) / np.sum(np.exp(resonances * 2))
            
            # Blend top 3 patterns
            top_indices = np.argsort(resonances)[-3:]
            base_pattern = torch.zeros_like(self.pattern_library[0].pattern)
            
            for idx in top_indices:
                weight = resonance_weights[idx]
                base_pattern += self.pattern_library[idx].pattern * weight
                self.pattern_library[idx].usage_count += 1
        
        # Add exploration variation
        if exploration > 0:
            # Exploration creates variations on resonant patterns
            noise = torch.randn_like(base_pattern) * exploration * 0.2
            base_pattern = torch.tanh(base_pattern + noise)
        
        return base_pattern
    
    def _compute_behavioral_similarity(self, traj1: torch.Tensor, traj2: torch.Tensor) -> float:
        """
        Compute similarity between two behavioral trajectories.
        
        This is the key to recognizing patterns that create similar behaviors
        even if their structure differs. We use multiple metrics:
        1. Trajectory shape similarity (DTW-like)
        2. Overall direction similarity
        3. Dynamics similarity (velocity patterns)
        """
        # Ensure trajectories have same length by interpolating
        len1, len2 = traj1.shape[0], traj2.shape[0]
        if len1 != len2:
            # Interpolate to common length
            target_len = max(len1, len2)
            if len1 < target_len:
                traj1 = F.interpolate(traj1.T.unsqueeze(0), size=target_len, mode='linear')[0].T
            else:
                traj2 = F.interpolate(traj2.T.unsqueeze(0), size=target_len, mode='linear')[0].T
        
        # 1. Shape similarity (how similar are the paths?)
        shape_sim = F.cosine_similarity(traj1.flatten(), traj2.flatten(), dim=0).item()
        
        # 2. Direction similarity (do they go the same way?)
        dir1 = traj1[-1] - traj1[0]  # Overall direction
        dir2 = traj2[-1] - traj2[0]
        direction_sim = F.cosine_similarity(dir1, dir2, dim=0).item()
        
        # 3. Dynamics similarity (similar acceleration patterns?)
        if len(traj1) > 2:
            vel1 = torch.diff(traj1, dim=0)
            vel2 = torch.diff(traj2, dim=0)
            dynamics_sim = F.cosine_similarity(vel1.flatten(), vel2.flatten(), dim=0).item()
        else:
            dynamics_sim = shape_sim
        
        # Weighted combination
        similarity = 0.4 * shape_sim + 0.3 * direction_sim + 0.3 * dynamics_sim
        
        return similarity
    
    def _add_to_library(self, pattern: StrategicPattern):
        """Add successful pattern to library with behavioral similarity check."""
        # Check if behaviorally similar pattern exists
        for stored in self.pattern_library:
            # Compare behavioral trajectories, not just signatures
            behavioral_similarity = self._compute_behavioral_similarity(
                pattern.behavioral_trajectory,
                stored.behavioral_trajectory
            )
            
            if behavioral_similarity > 0.85:  # Similar behavior
                # Update if better or blend if comparable
                if pattern.score > stored.score * 1.2:  # Significantly better
                    # Replace with better pattern
                    stored.pattern = pattern.pattern
                    stored.score = pattern.score
                    stored.behavioral_trajectory = pattern.behavioral_trajectory
                    stored.context_embedding = pattern.context_embedding
                elif pattern.score > stored.score * 0.8:  # Comparable
                    # Blend patterns for robustness
                    blend_weight = pattern.score / (pattern.score + stored.score)
                    stored.pattern = (
                        stored.pattern * (1 - blend_weight) + 
                        pattern.pattern * blend_weight
                    )
                    stored.score = max(stored.score, pattern.score)
                    stored.usage_count += 1
                return
        
        # Add new pattern
        self.pattern_library.append(pattern)
        
        # Maintain size limit, but consider usage
        if len(self.pattern_library) > self.max_library_size:
            # Score by recency-weighted performance
            def pattern_value(p):
                age_factor = np.exp(-0.1 * (time.time() - p.creation_time) / 3600)  # Decay over hours
                usage_factor = 1.0 + 0.1 * p.usage_count
                return p.score * age_factor * usage_factor
            
            self.pattern_library.sort(key=pattern_value, reverse=True)
            self.pattern_library = self.pattern_library[:self.max_library_size]
    
    def discover_async(self,
                      current_field: torch.Tensor,
                      reward_signal: float,
                      callback: Optional[Callable] = None) -> Future:
        """
        Discover patterns asynchronously in background.
        
        This allows the brain to think deeply about strategy while
        remaining responsive to immediate needs.
        """
        field_snapshot = current_field.clone().detach()
        
        def _background_discovery():
            try:
                pattern = self.discover_strategic_pattern(
                    field_snapshot,
                    reward_signal,
                    n_candidates=32  # More thorough search in background
                )
                
                if callback:
                    callback(pattern)
                    
                return pattern
            except Exception as e:
                print(f"Background pattern discovery error: {e}")
                return None
        
        with self._discovery_lock:
            # Cancel existing discovery
            if self._current_discovery_future and not self._current_discovery_future.done():
                self._current_discovery_future.cancel()
            
            # Start new discovery
            future = self.executor.submit(_background_discovery)
            self._current_discovery_future = future
            
        return future
    
    def find_similar_patterns(self, target_behavior: torch.Tensor, n_results: int = 3) -> List[StrategicPattern]:
        """
        Find patterns that create similar behaviors.
        
        This allows the brain to ask "what patterns create forward movement?"
        and get back patterns that all produce that behavior, even if they
        work through different field dynamics.
        """
        if not self.pattern_library:
            return []
        
        # Create a simple trajectory from the target behavior if needed
        if target_behavior.dim() == 1:
            # Extend single vector to trajectory
            target_trajectory = target_behavior.unsqueeze(0).repeat(10, 1)
        else:
            target_trajectory = target_behavior
        
        # Score all patterns by behavioral similarity
        similarities = []
        for pattern in self.pattern_library:
            sim = self._compute_behavioral_similarity(
                target_trajectory,
                pattern.behavioral_trajectory
            )
            similarities.append((sim, pattern))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [pattern for _, pattern in similarities[:n_results]]
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pattern library."""
        if not self.pattern_library:
            return {
                'library_size': 0,
                'avg_score': 0.0,
                'avg_usage': 0.0,
                'behavior_clusters': 0
            }
        
        # Basic stats
        scores = [p.score for p in self.pattern_library]
        usages = [p.usage_count for p in self.pattern_library]
        
        # Cluster patterns by behavioral similarity
        behavior_clusters = self._count_behavior_clusters()
        
        return {
            'library_size': len(self.pattern_library),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'avg_usage': np.mean(usages),
            'most_used': np.max(usages),
            'behavior_clusters': behavior_clusters,
            'newest_pattern_age': time.time() - max(p.creation_time for p in self.pattern_library)
        }
    
    def _count_behavior_clusters(self, threshold: float = 0.7) -> int:
        """Count distinct behavioral clusters in the library."""
        if len(self.pattern_library) < 2:
            return len(self.pattern_library)
        
        # Simple clustering by behavioral similarity
        clusters = []
        for pattern in self.pattern_library:
            found_cluster = False
            for cluster in clusters:
                # Check similarity with cluster representative
                sim = self._compute_behavioral_similarity(
                    pattern.behavioral_trajectory,
                    cluster[0].behavioral_trajectory
                )
                if sim > threshold:
                    cluster.append(pattern)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append([pattern])
        
        return len(clusters)
    
    def _measure_field_tensions(self, field: torch.Tensor) -> Dict[str, float]:
        """
        Measure intrinsic tensions in the field that drive behavior.
        
        These tensions represent unmet intrinsic drives:
        - Information tension: Low field energy creates exploration need
        - Learning tension: Stagnant improvement creates novelty need
        - Confidence tension: Low confidence creates resolution need
        - Prediction tension: High errors create adaptation need
        
        Returns:
            Dict mapping tension type to magnitude (0-1 scale)
        """
        # Information/Energy tension: Low energy = high tension
        content_energy = field[:, :, :, :32].abs().mean().item()
        information_tension = max(0.0, 1.0 - content_energy * 2.0)  # Scale for sensitivity
        
        # Learning velocity tension: Detect stagnation from evolution parameters
        # Channel 55 often contains learning rate or plasticity indicators
        learning_indicators = field[:, :, :, 55].mean().item() if field.shape[-1] > 55 else 0.5
        learning_tension = max(0.0, min(1.0, learning_indicators))
        
        # Confidence tension: Low confidence = high tension
        # Channel 58 often contains confidence indicators
        confidence_level = field[:, :, :, 58].mean().item() if field.shape[-1] > 58 else 0.5
        confidence_tension = max(0.0, 1.0 - confidence_level)
        
        # Prediction error tension: High errors = high tension
        # Channels 48-52 often contain error signals
        if field.shape[-1] > 52:
            prediction_errors = field[:, :, :, 48:52].abs().mean().item()
            prediction_tension = min(1.0, prediction_errors * 2.0)  # Scale for sensitivity
        else:
            prediction_tension = 0.5
        
        # Novelty tension: Low novelty when patterns are too stable
        pattern_variance = field[:, :, :, self.pattern_channels].std().item()
        novelty_tension = max(0.0, 1.0 - pattern_variance * 3.0)  # Low variance = high tension
        
        return {
            'information': information_tension,
            'learning': learning_tension,
            'confidence': confidence_tension,
            'prediction': prediction_tension,
            'novelty': novelty_tension,
            'total': (information_tension + learning_tension + confidence_tension + 
                     prediction_tension + novelty_tension) / 5.0
        }
    
    def _evaluate_pattern_tension_based(self,
                                      base_field: torch.Tensor,
                                      pattern: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Evaluate a pattern based on how well it resolves field tensions.
        
        A good pattern reduces multiple tensions simultaneously, creating
        a more balanced and satisfied field state.
        
        Args:
            base_field: Current field state with tensions
            pattern: Candidate pattern to evaluate
            
        Returns:
            Tuple of (tension_relief_score, behavioral_signature, trajectory)
        """
        # Measure initial tensions
        initial_tensions = self._measure_field_tensions(base_field)
        
        # Install pattern in test field
        test_field = base_field.clone()
        test_field[:, :, :, self.pattern_channels] = pattern
        
        # Track behavioral trajectory and tension evolution
        behavioral_trajectory = []
        tension_timeline = []
        cumulative_relief = 0.0
        
        # Simulate field evolution with pattern
        field = test_field
        for t in range(self.simulation_horizon):
            # Evolve field
            field = self._evolve_field_with_pattern(field, pattern)
            
            # Extract behavioral tendency
            behavior = self._extract_behavioral_tendency(field)
            behavioral_trajectory.append(behavior)
            
            # Measure current tensions
            current_tensions = self._measure_field_tensions(field)
            tension_timeline.append(current_tensions['total'])
            
            # Calculate tension relief at this timestep
            tension_relief = initial_tensions['total'] - current_tensions['total']
            
            # Accumulate with temporal discount
            cumulative_relief += tension_relief * (0.95 ** t)
        
        # Bonus for specific tension resolutions
        final_tensions = self._measure_field_tensions(field)
        
        # Information tension relief (promotes exploration when needed)
        info_relief = max(0, initial_tensions['information'] - final_tensions['information'])
        
        # Learning tension relief (breaks out of plateaus)
        learning_relief = max(0, initial_tensions['learning'] - final_tensions['learning'])
        
        # Confidence tension relief (reduces uncertainty)
        confidence_relief = max(0, initial_tensions['confidence'] - final_tensions['confidence'])
        
        # Prediction tension relief (improves accuracy)
        prediction_relief = max(0, initial_tensions['prediction'] - final_tensions['prediction'])
        
        # Total score combines cumulative relief with specific achievements
        total_score = (
            cumulative_relief * 10.0 +  # General tension reduction
            info_relief * 3.0 +         # Exploration bonus
            learning_relief * 3.0 +     # Learning velocity bonus
            confidence_relief * 2.0 +   # Uncertainty reduction bonus
            prediction_relief * 2.0     # Accuracy improvement bonus
        )
        
        # Add behavioral coherence bonus
        behavioral_trajectory_tensor = torch.stack(behavioral_trajectory)
        if len(behavioral_trajectory) > 1:
            coherence = self._measure_behavioral_coherence(behavioral_trajectory)
            total_score += coherence * 5.0
        
        # Penalize patterns that increase tensions
        if final_tensions['total'] > initial_tensions['total']:
            total_score *= 0.5  # Halve score for tension-increasing patterns
        
        # Extract behavioral signature
        behavioral_signature = behavioral_trajectory_tensor.mean(dim=0)
        
        return total_score, behavioral_signature, behavioral_trajectory_tensor
    
    def shutdown(self):
        """Clean shutdown of background threads."""
        self.executor.shutdown(wait=True)