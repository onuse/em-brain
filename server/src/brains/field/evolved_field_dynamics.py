"""
Evolved Field Dynamics

The next generation of field dynamics where evolution rules are part of the field itself.
This replaces UnifiedFieldDynamics as the core system, starting with minimal self-modification
and naturally evolving toward full autonomy.

Key principle: No fixed parameters. Everything emerges.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from collections import deque
import numpy as np

from ...utils.tensor_ops import (
    create_zeros, create_randn, safe_mean, safe_var, 
    safe_normalize, field_energy, field_information
)


class EvolvedFieldDynamics:
    """
    Field dynamics where evolution rules emerge from field topology.
    
    This is THE field dynamics system - not an option or variant.
    It starts nearly identical to fixed dynamics but naturally evolves
    its own rules through experience.
    
    The field's last features encode local dynamics, starting with
    sensible defaults that gradually specialize based on experience.
    
    Core paradigm: Energy = Information
    - High field energy = high information content = need for rest/consolidation
    - Low field energy = low information = ready for new experiences
    - This creates natural explore/exploit/rest cycles
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 pattern_memory_size: int = 100,
                 confidence_window: int = 50,
                 initial_spontaneous_rate: float = 0.001,
                 initial_resting_potential: float = 0.01,
                 temporal_features: int = 16,
                 dynamics_features: int = 16,
                 device: torch.device = torch.device('cpu')):
        """Initialize evolved field dynamics."""
        self.device = device
        self.field_shape = field_shape
        
        # Feature allocation
        self.total_features = field_shape[-1]
        self.temporal_features = temporal_features
        self.dynamics_features = dynamics_features
        self.content_features = self.total_features - dynamics_features
        
        # Temporal persistence configuration (for working memory)
        self.spatial_features = self.content_features - temporal_features
        
        # Pattern memory for novelty detection
        self.pattern_memory = deque(maxlen=pattern_memory_size * 2)  # Increased size
        self.pattern_energies = deque(maxlen=pattern_memory_size * 2)
        self.pattern_timestamps = deque(maxlen=pattern_memory_size * 2)  # Track when patterns were seen
        
        # Prediction tracking for confidence
        self.prediction_errors = deque(maxlen=confidence_window)
        self.smoothed_confidence = 0.5
        
        # Learning improvement tracking for exploration
        self.error_improvement_history = deque(maxlen=50)  # Track improvement rate
        self.learning_plateau_cycles = 0  # How long since improvement
        
        # Information and state tracking
        self.information_history = deque(maxlen=1000)
        self.smoothed_information = 0.5
        self.cycles_without_input = 0
        
        # Current state
        self._last_novelty = 0.0
        self._last_modulation = {}
        
        # Traveling wave parameters
        self.wave_vectors = self._initialize_wave_vectors()
        self.phase_offset = 0.0
        
        # Start with very subtle self-modification
        # This will naturally increase as the system experiences the world
        self.self_modification_strength = 0.01  # Starts very low
        self.evolution_count = 0
        
        # Use initial rates only for initialization
        self._initial_spontaneous = initial_spontaneous_rate
        self._initial_resting = initial_resting_potential
    
    def evolve_field(self, field: torch.Tensor, 
                     external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolve field using emergent dynamics.
        
        The field itself determines how it evolves, with the strength
        of self-modification naturally increasing through experience.
        """
        # Split field
        content = field[:, :, :, :self.content_features]
        dynamics = field[:, :, :, self.content_features:]
        
        # Extract evolution parameters from dynamics
        decay_rates = torch.sigmoid(dynamics[:, :, :, 0:4])
        diffusion_strengths = torch.tanh(dynamics[:, :, :, 4:8]) * 0.1
        coupling_weights = dynamics[:, :, :, 8:12]
        plasticity_params = torch.sigmoid(dynamics[:, :, :, 12:16])
        
        # Get current modulation
        modulation = self._last_modulation
        if modulation:
            # Apply global modulation
            global_decay = modulation.get('decay_rate', 0.999)
            exploration = modulation.get('exploration_drive', 0.5)
            internal_drive = modulation.get('internal_drive', 0.5)
        else:
            global_decay = 0.999
            exploration = 0.5
            internal_drive = 0.5
        
        # 1. Multi-scale decay with working memory
        new_content = self._apply_differential_decay(content, decay_rates, global_decay)
        
        # 2. Spontaneous activity (traveling waves, etc.)
        if internal_drive > 0:
            spontaneous = self._generate_spontaneous_activity(
                new_content.shape, plasticity_params, internal_drive
            )
            new_content += spontaneous
        
        # 3. Learned diffusion
        new_content = self._apply_learned_diffusion(new_content, diffusion_strengths)
        
        # 4. Nonlinear coupling
        new_content = self._apply_coupling(new_content, coupling_weights, plasticity_params)
        
        # 5. External input
        if external_input is not None:
            input_strength = modulation.get('imprint_strength', 0.5) if modulation else 0.5
            new_content = self._integrate_input(new_content, external_input, input_strength, dynamics)
        
        # 6. Self-modification of dynamics
        new_dynamics = self._evolve_dynamics(dynamics, new_content, exploration)
        
        # 7. Gradually increase self-modification strength
        self._update_self_modification_strength()
        
        # Combine and return
        self.evolution_count += 1
        return torch.cat([new_content, new_dynamics], dim=-1)
    
    def _apply_differential_decay(self, content: torch.Tensor, 
                                 decay_rates: torch.Tensor, 
                                 global_decay: float) -> torch.Tensor:
        """Apply decay with working memory through differential rates."""
        # Split into spatial and temporal features
        spatial_content = content[:, :, :, :self.spatial_features]
        temporal_content = content[:, :, :, self.spatial_features:]
        
        # Multi-scale decay for spatial features
        new_spatial = spatial_content.clone()
        for scale in range(min(4, decay_rates.shape[-1])):
            scale_features = slice(scale, None, 4)
            if scale < new_spatial.shape[-1]:
                scale_decay = decay_rates[:, :, :, scale].unsqueeze(-1) * global_decay * 0.95
                new_spatial[:, :, :, scale_features] *= scale_decay
        
        # Slower decay for temporal features (working memory)
        temporal_decay = torch.mean(decay_rates, dim=-1, keepdim=True) * global_decay * 0.995
        new_temporal = temporal_content * temporal_decay
        
        # Temporal momentum for predictive dynamics
        if self.temporal_features > 1:
            momentum_field = torch.zeros_like(new_temporal)
            momentum_field[:, :, :, 1:] = new_temporal[:, :, :, :-1] * 0.15
            new_temporal = new_temporal * 0.85 + momentum_field
        
        return torch.cat([new_spatial, new_temporal], dim=-1)
    
    def _generate_spontaneous_activity(self, shape: Tuple, 
                                      plasticity: torch.Tensor,
                                      internal_drive: float) -> torch.Tensor:
        """Generate spontaneous activity including traveling waves."""
        # Base fluctuations
        noise = torch.randn(shape, device=self.device) * self._initial_spontaneous
        
        # Traveling waves
        waves = self._generate_traveling_waves(shape)
        
        # Local recurrence
        mean_plasticity = torch.mean(plasticity[:, :, :, 0]).item()
        recurrence_strength = mean_plasticity * internal_drive * 0.1
        
        # Combine
        spontaneous = noise + waves * 2.0
        spontaneous *= internal_drive
        
        return spontaneous
    
    def _apply_learned_diffusion(self, content: torch.Tensor, 
                                diffusion_strengths: torch.Tensor) -> torch.Tensor:
        """Apply diffusion with learned, region-specific strengths."""
        new_content = content.clone()
        
        for dim in range(3):  # Spatial dimensions
            diff_strength = diffusion_strengths[:, :, :, dim].unsqueeze(-1)
            
            # Use roll for MPS compatibility
            pos = torch.roll(content, shifts=-1, dims=dim)
            neg = torch.roll(content, shifts=1, dims=dim)
            
            laplacian = pos + neg - 2 * content
            new_content += laplacian * diff_strength
        
        return new_content
    
    def _apply_coupling(self, content: torch.Tensor,
                       coupling_weights: torch.Tensor,
                       plasticity: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear coupling between regions."""
        activation = torch.tanh(content * plasticity[:, :, :, 0].unsqueeze(-1))
        coupling_influence = create_zeros(content.shape, device=self.device)
        
        # Nearest neighbor coupling
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    
                    shifted = torch.roll(torch.roll(torch.roll(
                        activation, dx, dims=0), dy, dims=1), dz, dims=2)
                    
                    weight_idx = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                    weight = coupling_weights[:, :, :, weight_idx % 4].unsqueeze(-1)
                    
                    coupling_influence += shifted * weight * 0.1
        
        # Apply with plasticity modulation
        plasticity_mod = plasticity[:, :, :, 1].unsqueeze(-1)
        return content + coupling_influence * plasticity_mod
    
    def _integrate_input(self, content: torch.Tensor, 
                        external_input: torch.Tensor,
                        input_strength: float,
                        dynamics: torch.Tensor) -> torch.Tensor:
        """Integrate external input and update dynamics."""
        # Add input to content
        if external_input.shape[-1] <= self.content_features:
            content[:, :, :, :external_input.shape[-1]] += external_input * input_strength
        else:
            content += external_input[:, :, :, :self.content_features] * input_strength
        
        return content
    
    def _evolve_dynamics(self, dynamics: torch.Tensor, 
                        content: torch.Tensor,
                        exploration: float) -> torch.Tensor:
        """Evolve the dynamics based on content state."""
        # Compute content statistics
        content_energy = torch.mean(torch.abs(content), dim=-1, keepdim=True)
        content_variance = torch.var(content, dim=-1, keepdim=True)
        content_sparsity = torch.mean((torch.abs(content) < 0.1).float(), dim=-1, keepdim=True)
        
        # Create dynamics update
        dynamics_update = create_zeros(dynamics.shape, device=self.device)
        
        # High energy → more persistence (lower decay)
        dynamics_update[:, :, :, 0] = -content_energy.squeeze(-1) * self.self_modification_strength
        
        # High variance → more diffusion
        dynamics_update[:, :, :, 4] = content_variance.squeeze(-1) * self.self_modification_strength
        
        # Sparsity → more coupling
        dynamics_update[:, :, :, 8] = content_sparsity.squeeze(-1) * self.self_modification_strength * 0.5
        
        # Exploration modulates plasticity
        dynamics_update[:, :, :, 12] = (content_energy.squeeze(-1) * exploration - 0.5) * self.self_modification_strength * 0.5
        
        # Apply update with momentum
        momentum = 0.95
        return dynamics * momentum + dynamics_update * (1 - momentum)
    
    def _update_self_modification_strength(self):
        """Gradually increase self-modification over time."""
        # Logarithmic growth that continues beyond 10%
        # Starts at 1%, reaches 10% at 10k cycles, 20% at 100k, etc.
        self.self_modification_strength = 0.01 + 0.09 * np.log10(1 + self.evolution_count / 1000)
    
    def initialize_field_dynamics(self, field: torch.Tensor):
        """Initialize dynamics features with sensible defaults."""
        dynamics = field[:, :, :, self.content_features:]
        
        # Decay rates - start near 1 (minimal decay)
        dynamics[:, :, :, 0:4] = torch.randn_like(dynamics[:, :, :, 0:4]) * 0.1 + 2.0
        
        # Diffusion - small values
        dynamics[:, :, :, 4:8] = torch.randn_like(dynamics[:, :, :, 4:8]) * 0.05
        
        # Coupling - small values
        dynamics[:, :, :, 8:12] = torch.randn_like(dynamics[:, :, :, 8:12]) * 0.05
        
        # Plasticity - moderate values
        dynamics[:, :, :, 12:16] = torch.randn_like(dynamics[:, :, :, 12:16]) * 0.2
    
    # Compatibility methods from UnifiedFieldDynamics
    def compute_field_state(self, field: torch.Tensor) -> Dict[str, float]:
        """Compute field state metrics."""
        content = field[:, :, :, :self.content_features]
        
        information = field_information(content)
        variance = float(safe_var(content))
        combined_information = information * (1.0 + variance)
        
        self.information_history.append(combined_information)
        self.smoothed_information = 0.95 * self.smoothed_information + 0.05 * combined_information
        
        return {
            'raw_information': information,
            'variance': variance,
            'combined_information': combined_information,
            'smoothed_information': self.smoothed_information,
            # Keep old names for compatibility
            'raw_energy': information,
            'combined_energy': combined_information,
            'smoothed_energy': self.smoothed_information
        }
    
    def compute_novelty(self, field: torch.Tensor) -> float:
        """Compute novelty of current field state."""
        # Always ensure we have timestamps deque initialized
        if not hasattr(self, 'pattern_timestamps'):
            self.pattern_timestamps = deque(maxlen=len(self.pattern_memory) * 2)
            
        if len(self.pattern_memory) == 0:
            # First pattern - high novelty
            # Create a proper initial pattern from field
            content = field[:, :, :, :self.content_features]
            field_small = content[:8, :8, :8, :8] if content.shape[0] >= 8 else content
            field_flat = field_small.flatten()
            if len(field_flat) > 512:
                indices = torch.linspace(0, len(field_flat)-1, 512, dtype=torch.long, device=field.device)
                field_flat = field_flat[indices]
            field_norm = field_flat / (torch.norm(field_flat) + 1e-8)
            self.pattern_memory.append(field_norm.detach().clone())
            self.pattern_timestamps.append(self.evolution_count)
            self._last_novelty = 1.0
            return 1.0
        
        content = field[:, :, :, :self.content_features]
        
        # Downsample for efficiency
        field_small = content[:8, :8, :8, :8] if content.shape[0] >= 8 else content
        field_flat = field_small.flatten()
        
        if len(field_flat) > 512:
            indices = torch.linspace(0, len(field_flat)-1, 512, dtype=torch.long, device=field.device)
            field_flat = field_flat[indices]
        
        field_norm = field_flat / (torch.norm(field_flat) + 1e-8)
        
        # Compare to recent memory with temporal weighting
        max_similarity = 0.0
        current_time = self.evolution_count
        
        # Only compare to recent patterns (temporal forgetting)
        recent_patterns = []
        # Handle case where timestamps might not exist for old patterns
        if len(self.pattern_timestamps) < len(self.pattern_memory):
            # Backfill timestamps for existing patterns
            while len(self.pattern_timestamps) < len(self.pattern_memory):
                self.pattern_timestamps.append(0)
        
        for i, (pattern, timestamp) in enumerate(zip(self.pattern_memory, self.pattern_timestamps)):
            age = current_time - timestamp
            if age < 1000:  # Only consider patterns from last 1000 cycles
                weight = np.exp(-age / 500)  # Exponential decay
                similarity = float(torch.dot(field_norm, pattern)) * weight
                max_similarity = max(max_similarity, similarity)
                recent_patterns.append(i)
        
        # Add temporal variation bonus
        temporal_bonus = 0.1 * np.sin(current_time * 0.01)  # Slow oscillation
        
        # Store new pattern with timestamp
        self.pattern_memory.append(field_norm.detach().clone())
        self.pattern_timestamps.append(current_time)
        
        # Calculate novelty with minimum floor
        base_novelty = 1.0 - max_similarity
        novelty = max(0.1, base_novelty + temporal_bonus)  # Never less than 0.1
        
        self._last_novelty = novelty
        
        return novelty
    
    def update_confidence(self, prediction_error: float):
        """Update confidence based on prediction accuracy and track improvement."""
        self.prediction_errors.append(prediction_error)
        
        if len(self.prediction_errors) > 5:
            recent_error = np.mean(list(self.prediction_errors)[-10:])
            
            # Track improvement in prediction
            if len(self.prediction_errors) > 20:
                old_error = np.mean(list(self.prediction_errors)[-20:-10])
                improvement = old_error - recent_error  # Positive = getting better
                self.error_improvement_history.append(improvement)
                
                # Check if learning has plateaued
                if len(self.error_improvement_history) > 10:
                    recent_improvement = np.mean(list(self.error_improvement_history)[-10:])
                    if abs(recent_improvement) < 0.001:  # Very little change
                        self.learning_plateau_cycles += 1
                    else:
                        self.learning_plateau_cycles = 0
        else:
            recent_error = 0.5
        
        raw_confidence = 1.0 / (1.0 + recent_error * 5.0)
        self.smoothed_confidence = 0.9 * self.smoothed_confidence + 0.1 * raw_confidence
    
    def compute_field_modulation(self, 
                                field_state: Dict[str, float],
                                has_sensory_input: bool = True) -> Dict[str, float]:
        """Compute modulation parameters."""
        information = field_state.get('smoothed_information', field_state.get('smoothed_energy', 0.5))
        novelty = self._last_novelty
        confidence = self.smoothed_confidence
        
        if not has_sensory_input:
            self.cycles_without_input += 1
        else:
            self.cycles_without_input = 0
        
        is_dreaming = self.cycles_without_input > 100
        # Normalize information with higher threshold to prevent quick saturation
        # Most field states should fall in 0.1-1.5 range, normalize to 0-1
        norm_information = np.clip(information / 3.0, 0.0, 1.0)
        
        internal_drive = (norm_information + confidence) / 2.0
        if is_dreaming:
            internal_drive = 0.95
        
        # Base exploration with minimum floor
        # Core paradigm: information content in the field
        # High information means the brain needs rest to process/consolidate
        # Low information means the brain is ready for new experiences
        # This creates a natural explore/exploit/rest cycle
        base_exploration = (1.0 - norm_information) * (0.5 + 0.5 * novelty)
        
        # Learning-driven exploration: addiction to prediction improvement
        # When learning plateaus, exploration pressure builds up
        improvement_pressure = 0.0
        if hasattr(self, 'learning_plateau_cycles'):
            # Exploration pressure grows with plateau duration
            # After 100 cycles of no improvement, adds 0.3 to exploration
            # After 200 cycles, adds 0.5 (saturates at 0.5)
            improvement_pressure = min(0.5, self.learning_plateau_cycles / 200.0)
        
        # Metabolic baseline - even resting brains explore a little
        metabolic_baseline = 0.1
        
        # Combine all exploration pressures
        exploration_drive = base_exploration + improvement_pressure + metabolic_baseline
        
        # Natural ceiling at 1.0, no artificial minimum needed
        exploration_drive = min(1.0, exploration_drive)
        
        # Add uncertainty-based exploration bonus
        uncertainty_bonus = (1.0 - confidence) * 0.2
        exploration_drive = min(1.0, exploration_drive + uncertainty_bonus)
        
        sensory_amplification = 1.0 + (1.0 - confidence) * 0.5
        imprint_strength = 0.1 + 0.7 * (1.0 - internal_drive)
        motor_noise = exploration_drive * 0.5  # Increased from 0.4
        decay_rate = 0.999 - 0.01 * norm_information
        attention_novelty_bias = (1.0 - internal_drive) * 0.8
        
        self._last_modulation = {
            'internal_drive': internal_drive,
            'spontaneous_weight': internal_drive,
            'exploration_drive': exploration_drive,
            'sensory_amplification': sensory_amplification,
            'imprint_strength': imprint_strength,
            'motor_noise': motor_noise,
            'decay_rate': decay_rate,
            'attention_novelty_bias': attention_novelty_bias,
            'is_dreaming': is_dreaming,
            'information': norm_information,
            'energy': norm_information,  # Keep for compatibility
            'confidence': confidence,
            'novelty': novelty
        }
        
        return self._last_modulation
    
    def get_state_description(self) -> str:
        """Get human-readable state description."""
        if not self._last_modulation:
            return "Initializing..."
        
        m = self._last_modulation
        
        if m['is_dreaming']:
            return "💤 DREAM: Pure internal dynamics"
        
        info_desc = "High information" if m['information'] > 0.6 else "Low information"
        confidence_desc = "confident" if m['confidence'] > 0.6 else "uncertain"
        
        # Determine mode based on exploration drive and learning state
        if hasattr(self, 'learning_plateau_cycles') and self.learning_plateau_cycles > 100:
            mode = "EXPLORING (learning plateaued)"
        elif m['exploration_drive'] > 0.7:
            mode = "EXPLORING"
        elif m['exploration_drive'] > 0.5:
            mode = "exploring"
        else:
            mode = "exploiting"
        
        balance = f"{int(m['internal_drive']*100)}% internal"
        
        # Add self-modification indicator
        self_mod = f"[SM: {self.self_modification_strength:.0%}]"
        
        return f"🧠 {info_desc}, {confidence_desc}, {mode} | {balance} {self_mod}"
    
    def get_working_memory_state(self, field: torch.Tensor) -> Dict[str, Any]:
        """Extract working memory state from temporal features."""
        content = field[:, :, :, :self.content_features]
        temporal_field = content[:, :, :, self.spatial_features:]
        
        memory_patterns = []
        for i in range(min(self.temporal_features, temporal_field.shape[-1])):
            pattern = temporal_field[:, :, :, i]
            if torch.sum(torch.abs(pattern)) > 0.1:
                memory_patterns.append(pattern)
        
        if memory_patterns:
            memory_tensor = torch.stack(memory_patterns)
            mean_activation = torch.mean(torch.abs(memory_tensor))
            temporal_coherence = self._compute_temporal_coherence(memory_tensor)
        else:
            mean_activation = torch.tensor(0.0)
            temporal_coherence = torch.tensor(0.0)
        
        return {
            'n_patterns': len(memory_patterns),
            'mean_activation': float(mean_activation.item()) if isinstance(mean_activation, torch.Tensor) else float(mean_activation),
            'temporal_coherence': float(temporal_coherence.item()) if isinstance(temporal_coherence, torch.Tensor) else float(temporal_coherence),
            # Don't include raw tensor patterns in telemetry - they're too large and not JSON serializable
            'pattern_summary': {
                'count': len(memory_patterns),
                'avg_size': sum(p.numel() for p in memory_patterns[:5]) / min(5, len(memory_patterns)) if memory_patterns else 0
            }
        }
    
    def _compute_temporal_coherence(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compute coherence between temporal patterns."""
        if patterns.shape[0] < 2:
            return torch.tensor(0.0)
        
        correlations = []
        for i in range(patterns.shape[0] - 1):
            pattern1 = patterns[i].flatten()
            pattern2 = patterns[i + 1].flatten()
            
            if torch.std(pattern1) > 0 and torch.std(pattern2) > 0:
                corr = F.cosine_similarity(
                    pattern1.unsqueeze(0),
                    pattern2.unsqueeze(0)
                )
                correlations.append(corr)
        
        return torch.mean(torch.stack(correlations)) if correlations else torch.tensor(0.0)
    
    def _initialize_wave_vectors(self) -> torch.Tensor:
        """Initialize random wave vectors for spontaneous activity."""
        n_waves = np.random.randint(3, 6)
        wave_dims = min(3, len(self.field_shape))
        wave_vectors = create_randn((n_waves, wave_dims), device=self.device)
        
        for i in range(n_waves):
            wave_vectors[i] = safe_normalize(wave_vectors[i])
        
        return wave_vectors
    
    def _generate_traveling_waves(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate coherent traveling wave patterns."""
        wave_shape = shape[:3] if len(shape) >= 3 else shape
        waves = create_zeros(wave_shape, device=self.device)
        
        coords = []
        for i in range(len(wave_shape)):
            coord = torch.arange(wave_shape[i], device=self.device, dtype=torch.float32)
            coord = coord.view([-1 if j == i else 1 for j in range(len(wave_shape))])
            coords.append(coord)
        
        for wave_vec in self.wave_vectors:
            phase = self.phase_offset
            for i in range(len(coords)):
                if i < len(wave_vec):
                    phase = phase + wave_vec[i] * coords[i] / 3.0
            
            wave_amplitude = self._initial_spontaneous * 5.0
            waves += torch.sin(phase) * wave_amplitude
        
        self.phase_offset += 0.1
        
        if len(shape) > len(wave_shape):
            for _ in range(len(shape) - len(wave_shape)):
                waves = waves.unsqueeze(-1)
            waves = waves.expand(shape).clone()
            waves *= 0.1
        
        return waves
    
    def reset_waves(self):
        """Reset traveling wave patterns."""
        self.wave_vectors = self._initialize_wave_vectors()
        self.phase_offset = 0.0
    
    def get_emergent_properties(self) -> Dict[str, float]:
        """Get properties of the evolved system."""
        return {
            'self_modification_strength': self.self_modification_strength,
            'evolution_cycles': self.evolution_count,
            'smoothed_information': self.smoothed_information,
            'smoothed_confidence': self.smoothed_confidence,
            'cycles_without_input': self.cycles_without_input
        }
    
    def generate_uncertainty_map(self, field: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial uncertainty map for predictive sampling.
        
        High uncertainty regions indicate where the field would benefit
        from focused sensory sampling (e.g., visual glimpses).
        
        Returns:
            3D uncertainty map matching spatial dimensions of field
        """
        # Extract content features (excluding dynamics features)
        content = field[:, :, :, :self.content_features]
        
        # Compute local variance as proxy for uncertainty
        # High variance = transitioning/uncertain regions
        local_variance = torch.var(content, dim=3, keepdim=False)
        
        # Factor in temporal dynamics
        # Regions with high temporal features are less certain
        if self.temporal_features > 0:
            temporal_activity = torch.mean(
                torch.abs(content[:, :, :, self.spatial_features:]),
                dim=3
            )
            uncertainty = local_variance + 0.5 * temporal_activity
        else:
            uncertainty = local_variance
        
        # Include global confidence inverse
        # Low confidence = high uncertainty everywhere
        confidence_factor = 1.0 - self.smoothed_confidence
        uncertainty = uncertainty * (1.0 + confidence_factor)
        
        # Add exploration bonus to unexplored regions
        # This encourages sampling of quiet areas occasionally
        exploration_noise = torch.randn_like(uncertainty) * 0.1
        uncertainty = uncertainty + torch.relu(exploration_noise)
        
        # Normalize to [0, 1] range
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)
        
        return uncertainty