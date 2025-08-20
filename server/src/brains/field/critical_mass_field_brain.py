#!/usr/bin/env python3
"""
Critical Mass Field Brain - Robot Integration
Sufficient complexity for emergence, integrated with robot interface
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EmergenceConfig:
    """Configuration tuned for emergence with robot"""
    # Field parameters - matched to robot needs
    field_size: Tuple[int, int, int, int] = (48, 48, 48, 96)  # ~21M params
    
    # Constraints that create pressure
    memory_slots: int = 32  # Enough for concepts but forces compression
    energy_budget: int = 500  # Enough for complex processing but forces selection
    attention_bandwidth: int = 20  # Can track multiple patterns but must prioritize
    resonance_slots: int = 64  # Room for concept formation
    
    # Interaction parameters
    swarm_size: int = 1000  # Enough for consensus but computationally feasible
    superposition_branches: int = 100  # Multiple futures but must choose
    binding_capacity: int = 8  # Can bind concepts but limited
    
    # Dynamics parameters
    diffusion_rate: float = 0.1
    momentum_decay: float = 0.9
    discomfort_threshold: float = 0.3
    resonance_threshold: float = 2.0  # Std devs above mean for resonance
    
    # Robot-specific
    sensor_dims: int = 5  # ultrasonic, vision_detected, audio_level, etc.
    motor_dims: int = 6  # pan, tilt, motor1-4


class CriticalMassFieldBrain:
    """
    Critical Mass Brain adapted for robot control.
    Implements the standard brain interface while testing emergence hypothesis.
    """
    
    def __init__(self, config: Optional[EmergenceConfig] = None):
        """Initialize the critical mass brain"""
        self.config = config or EmergenceConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu':
            logger.warning("GPU not available, running on CPU - will be slower")
        
        logger.info(f"Initializing Critical Mass Brain on {self.device}")
        logger.info(f"Field size: {self.config.field_size}")
        logger.info(f"Total parameters: {np.prod(self.config.field_size) / 1e6:.1f}M")
        
        # Initialize all interacting systems
        self._init_field_dynamics()
        self._init_resonance_system()
        self._init_binding_system()
        self._init_superposition_system()
        self._init_swarm_system()
        self._init_preference_system()
        self._init_memory_system()
        
        # Metrics to observe emergence
        self.metrics = {
            'cycles': 0,
            'concepts_formed': 0,
            'bindings_active': 0,
            'superposition_collapse_confidence': 0,
            'swarm_consensus_strength': 0,
            'preference_stability': 0,
            'memory_recall_accuracy': 0,
            'emergent_patterns': [],
            'discomfort_history': []
        }
        
        # History for learning
        self.history = {
            'states': [],
            'actions': [],
            'outcomes': [],
            'resonances': []
        }
        
        logger.info("Critical Mass Brain initialized successfully")
    
    def _init_field_dynamics(self):
        """Initialize the base field with dynamics"""
        size = self.config.field_size
        
        # Main field and momentum
        self.field = torch.randn(*size, device=self.device) * 0.1
        self.momentum = torch.zeros_like(self.field)
        
        # Diffusion kernel for spatial propagation
        self.diffusion_kernel = self._create_diffusion_kernel()
        
        # Discomfort field for intrinsic motivation
        self.discomfort_field = torch.zeros_like(self.field)
        
    def _init_resonance_system(self):
        """Initialize resonance detection and stabilization"""
        # Resonance storage in frequency domain
        self.resonances = {}  # frequency_hash -> (pattern, strength, age)
        self.resonance_buffer = torch.zeros(
            self.config.resonance_slots,
            *self.config.field_size,
            device=self.device,
            dtype=torch.complex64
        )
        
        # Resonance coupling matrix (which resonate together)
        self.resonance_coupling = torch.zeros(
            self.config.resonance_slots,
            self.config.resonance_slots,
            device=self.device
        )
    
    def _init_binding_system(self):
        """Initialize phase-based binding for composition"""
        # Phase field for binding
        self.phase_field = torch.zeros(
            self.config.field_size[:-1],
            device=self.device
        )
        
        # Binding registry (what's currently bound)
        self.active_bindings = []  # List of bound concept groups
        
        # Binding strength matrix
        self.binding_strength = torch.zeros(
            self.config.binding_capacity,
            self.config.binding_capacity,
            device=self.device
        )
    
    def _init_superposition_system(self):
        """Initialize quantum-like superposition for planning"""
        # Superposition of possible futures
        self.superposition = torch.zeros(
            self.config.superposition_branches,
            *self.config.field_size,
            device=self.device
        )
        
        # Coherence scores for each branch
        self.coherences = torch.zeros(
            self.config.superposition_branches,
            device=self.device
        )
        
        # Action associated with each branch
        self.branch_actions = torch.zeros(
            self.config.superposition_branches,
            self.config.motor_dims,
            device=self.device
        )
    
    def _init_swarm_system(self):
        """Initialize swarm consensus for interpretation"""
        # Each swarm agent has unique weights
        self.swarm_weights = torch.randn(
            self.config.swarm_size,
            self.config.field_size[-1],
            device=self.device
        ) * 0.1
        
        # Agent specializations
        self.swarm_specializations = torch.softmax(
            torch.randn(self.config.swarm_size, 8, device=self.device),
            dim=1
        )
    
    def _init_preference_system(self):
        """Initialize preference learning from outcomes"""
        # Preference field (what patterns are preferred)
        self.preference_field = torch.zeros_like(self.field)
        
        # Preference stability tracking
        self.preference_momentum = torch.zeros_like(self.field)
        
        # Goal field (persistent objectives)
        self.goal_field = torch.zeros_like(self.field)
    
    def _init_memory_system(self):
        """Initialize holographic memory for recall"""
        # Holographic storage
        self.memory_hologram = torch.zeros(
            *self.config.field_size,
            device=self.device,
            dtype=torch.complex64
        )
        
        # Memory index (cue -> memory mapping)
        self.memory_index = {}
    
    def process(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        Main processing method - compatible with robot interface.
        
        Args:
            sensor_data: Dictionary with sensor readings
                - 'ultrasonic': Distance in cm
                - 'vision_detected': 1.0 if object detected, 0.0 otherwise
                - 'audio_level': Audio intensity 0-1
                - 'battery': Battery level 0-1
                - 'temperature': Temperature in Celsius
        
        Returns:
            Dictionary with motor commands:
                - 'pan': -1 to 1
                - 'tilt': -1 to 1  
                - 'motor1' through 'motor4': -1 to 1
        """
        self.metrics['cycles'] += 1
        
        # Convert sensor dict to array
        sensory_input = np.array([
            sensor_data.get('ultrasonic', 50.0) / 100.0,  # Normalize to 0-1
            sensor_data.get('vision_detected', 0.0),
            sensor_data.get('audio_level', 0.0),
            sensor_data.get('battery', 1.0),
            (sensor_data.get('temperature', 25.0) - 25.0) / 10.0  # Normalize around 25C
        ])
        
        # Full processing cycle with all systems interacting
        action, emergence_indicators = self._process_cycle(sensory_input)
        
        # Convert action array to motor dict
        motor_commands = {
            'pan': float(np.tanh(action[0])),
            'tilt': float(np.tanh(action[1])),
            'motor1': float(np.tanh(action[2])),
            'motor2': float(np.tanh(action[3])),
            'motor3': float(np.tanh(action[4])),
            'motor4': float(np.tanh(action[5]))
        }
        
        # Log emergence indicators periodically
        if self.metrics['cycles'] % 100 == 0:
            logger.info(f"Cycle {self.metrics['cycles']} emergence indicators:")
            for key, value in emergence_indicators.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.3f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        return motor_commands
    
    def _process_cycle(self, sensory_input: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Full processing cycle with all systems interacting.
        This is where emergence happens.
        """
        # Phase 1: Input Processing
        interpreted_input = self._swarm_interpret(sensory_input)
        self._inject_into_field(interpreted_input)
        
        # Phase 2: Field Evolution
        self._evolve_field_dynamics()
        
        # Phase 3: Pattern Formation
        self._detect_and_stabilize_resonances()
        self._update_phase_binding()
        
        # Phase 4: Memory Operations
        self._store_to_holographic_memory()
        recalled = self._attempt_recall()
        
        # Phase 5: Preference Update
        discomfort = self._compute_discomfort()
        self._update_preferences(discomfort)
        self._update_goals()
        
        # Phase 6: Future Simulation
        self._create_superposition()
        self._simulate_futures()
        
        # Phase 7: Decision Making
        action = self._collapse_to_decision()
        
        # Phase 8: Learning
        self._update_resonance_coupling()
        self._strengthen_successful_patterns()
        
        # Record state for learning
        self._record_history()
        
        # Compute emergence metrics
        emergence_indicators = self._measure_emergence()
        
        return action, emergence_indicators
    
    def _swarm_interpret(self, sensory_input: np.ndarray) -> torch.Tensor:
        """Swarm consensus interpretation of input"""
        # Convert to tensor
        input_tensor = torch.tensor(sensory_input, device=self.device, dtype=torch.float32)
        
        # Expand input to match swarm weights dimension
        if len(input_tensor) < self.config.field_size[-1]:
            expanded_input = torch.zeros(self.config.field_size[-1], device=self.device)
            expanded_input[:len(input_tensor)] = input_tensor
            input_tensor = expanded_input
        
        # Each agent interprets differently
        interpretations = self.swarm_weights @ input_tensor
        
        # Apply specializations
        specialized = interpretations.unsqueeze(1) * self.swarm_specializations
        
        # Consensus through voting
        consensus = specialized.mean(dim=0)
        
        # Expand to field dimensions
        field_input = torch.zeros(self.config.field_size, device=self.device)
        field_input[..., :len(consensus)] = consensus.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Measure consensus strength
        variance = specialized.var(dim=0).mean()
        self.metrics['swarm_consensus_strength'] = 1.0 / (1.0 + variance.item())
        
        return field_input
    
    def _inject_into_field(self, input_field: torch.Tensor):
        """Inject interpreted input into field"""
        # Modulate by current attention
        attention_mask = self._compute_attention_mask()
        
        # Inject with attention weighting
        self.field += input_field * attention_mask * 0.1
    
    def _evolve_field_dynamics(self):
        """Evolve field with all dynamics"""
        # Compute all forces in parallel
        diffusion = self._apply_diffusion()
        gradients = self._compute_gradients()
        resonance_pressure = self._compute_resonance_pressure()
        preference_pull = self._compute_preference_pull()
        
        # Update momentum
        self.momentum = (self.config.momentum_decay * self.momentum + 
                        (1 - self.config.momentum_decay) * 
                        (diffusion + gradients + resonance_pressure + preference_pull))
        
        # Update field
        self.field += self.momentum * 0.01
        
        # Apply constraints
        self.field = self._apply_constraints(self.field)
        
        # Bounded activation
        self.field = torch.tanh(self.field)
    
    def _apply_diffusion(self) -> torch.Tensor:
        """3D diffusion for information spreading"""
        field_batched = self.field.permute(3, 0, 1, 2).unsqueeze(0)
        
        kernel = self.diffusion_kernel.repeat(
            self.config.field_size[-1], 1, 1, 1, 1
        )
        
        diffused = F.conv3d(
            field_batched,
            kernel,
            padding=1,
            groups=self.config.field_size[-1]
        )
        
        return diffused.squeeze(0).permute(1, 2, 3, 0) * self.config.diffusion_rate
    
    def _compute_gradients(self) -> torch.Tensor:
        """Information gradients create flow"""
        grad_x = torch.gradient(self.field, dim=0)[0]
        grad_y = torch.gradient(self.field, dim=1)[0]
        grad_z = torch.gradient(self.field, dim=2)[0]
        
        return torch.stack([grad_x, grad_y, grad_z]).norm(dim=0) * 0.01
    
    def _compute_resonance_pressure(self) -> torch.Tensor:
        """Resonances create attractors in field"""
        pressure = torch.zeros_like(self.field)
        
        for i, resonance in enumerate(self.resonance_buffer):
            if resonance.abs().mean() > 0.01:
                # Resonance creates local attractor
                pattern = torch.fft.ifftn(resonance, dim=(0,1,2)).real
                strength = self.resonance_coupling[i, i]  # Self-coupling is strength
                pressure += pattern * strength * 0.01
        
        return pressure
    
    def _compute_preference_pull(self) -> torch.Tensor:
        """Preferences guide field evolution"""
        # Pull toward preferred states
        preference_gradient = self.preference_field - self.field
        
        # Modulated by goal field
        goal_modulation = torch.sigmoid(self.goal_field.abs())
        
        return preference_gradient * goal_modulation * 0.01
    
    def _apply_constraints(self, field: torch.Tensor) -> torch.Tensor:
        """Apply resource constraints that force abstraction"""
        # Memory constraint via FFT compression
        spectrum = torch.fft.fftn(field, dim=(0, 1, 2))
        magnitudes = spectrum.abs()
        
        # Keep only top-k frequencies
        flat_mags = magnitudes.flatten()
        k = min(self.config.memory_slots * field.shape[-1], flat_mags.numel())
        threshold = torch.topk(flat_mags, k).values[-1]
        
        mask = (magnitudes >= threshold).to(torch.complex64)
        compressed_spectrum = spectrum * mask
        
        compressed = torch.fft.ifftn(compressed_spectrum, dim=(0, 1, 2)).real
        
        # Energy constraint
        energy = compressed.abs().sum(dim=-1)
        energy_limit = self.config.energy_budget / compressed[..., 0].numel()
        energy_mask = torch.sigmoid((energy_limit - energy) * 10).unsqueeze(-1)
        
        return compressed * energy_mask
    
    def _detect_and_stabilize_resonances(self):
        """Find and strengthen stable patterns"""
        # FFT to find resonances
        spectrum = torch.fft.fftn(self.field, dim=(0, 1, 2))
        power = spectrum.abs()
        
        # Detect peaks (resonances)
        mean_power = power.mean()
        std_power = power.std()
        threshold = mean_power + self.config.resonance_threshold * std_power
        
        # Find new resonances
        is_resonance = power > threshold
        new_resonances = torch.where(is_resonance)
        
        # Add to resonance buffer
        n_new = min(len(new_resonances[0]), 
                   self.config.resonance_slots - self.metrics['concepts_formed'])
        
        for i in range(n_new):
            idx = self.metrics['concepts_formed']
            if idx < self.config.resonance_slots:
                self.resonance_buffer[idx] = spectrum[
                    new_resonances[0][i],
                    new_resonances[1][i],
                    new_resonances[2][i]
                ]
                self.metrics['concepts_formed'] += 1
                
                # Initialize coupling
                self.resonance_coupling[idx, idx] = 1.0
    
    def _update_phase_binding(self):
        """Update phase relationships for binding"""
        # Find patterns that should bind (high correlation)
        active_resonances = self.resonance_buffer[:self.metrics['concepts_formed']]
        
        if len(active_resonances) > 1:
            # Compute pairwise correlations
            correlations = torch.zeros(
                len(active_resonances),
                len(active_resonances),
                device=self.device
            )
            
            for i, r1 in enumerate(active_resonances):
                for j, r2 in enumerate(active_resonances):
                    if i != j:
                        correlation = (r1.conj() * r2).real.mean()
                        correlations[i, j] = correlation
            
            # Bind highly correlated patterns
            binding_threshold = correlations.mean() + correlations.std()
            to_bind = torch.where(correlations > binding_threshold)
            
            # Update binding strength
            for i, j in zip(to_bind[0], to_bind[1]):
                if i < j:  # Avoid duplicates
                    self.resonance_coupling[i, j] += 0.1
                    self.resonance_coupling[j, i] += 0.1
            
            self.metrics['bindings_active'] = (self.resonance_coupling > 0.1).sum().item() // 2
    
    def _create_superposition(self):
        """Create superposition of possible futures"""
        # Generate random action branches
        self.branch_actions = torch.randn(
            self.config.superposition_branches,
            self.config.motor_dims,
            device=self.device
        ) * 0.5
        
        # Each branch is current field + action effect
        for i in range(self.config.superposition_branches):
            # Simulate action effect on field
            action_effect = self.branch_actions[i].reshape(1, 1, 1, self.config.motor_dims)
            action_field = torch.zeros_like(self.field)
            action_field[..., :self.config.motor_dims] = action_effect
            
            # Create future state
            self.superposition[i] = self.field + action_field * 0.1
    
    def _simulate_futures(self):
        """Simulate all futures in parallel"""
        # Evolve each superposition branch
        for i in range(self.config.superposition_branches):
            future = self.superposition[i]
            
            # Quick evolution (simplified)
            future = torch.tanh(future + torch.randn_like(future) * 0.01)
            
            # Measure coherence
            variance = future.var()
            entropy = -(future * torch.log(future.abs() + 1e-10)).sum()
            
            self.coherences[i] = 1.0 / (1.0 + variance + entropy * 0.001)
    
    def _collapse_to_decision(self) -> np.ndarray:
        """Collapse superposition to single action"""
        # Weight by coherence
        weights = torch.softmax(self.coherences * 10, dim=0)
        
        # Weighted average of actions (could also do winner-take-all)
        action = (self.branch_actions * weights.unsqueeze(1)).sum(dim=0)
        
        # Record confidence
        self.metrics['superposition_collapse_confidence'] = weights.max().item()
        
        return action.cpu().numpy()
    
    def _compute_discomfort(self) -> float:
        """Compute field discomfort"""
        variance = self.field.var()
        uniformity = 1.0 / (1.0 + variance)
        
        # Also consider goal distance
        if self.goal_field.abs().mean() > 0.01:
            goal_distance = (self.goal_field - self.field).abs().mean()
            discomfort = uniformity * 0.5 + goal_distance * 0.5
        else:
            discomfort = uniformity
        
        self.metrics['discomfort_history'].append(discomfort.item())
        return discomfort.item()
    
    def _update_preferences(self, discomfort: float):
        """Update preferences based on discomfort change"""
        if len(self.history['outcomes']) > 0:
            prev_discomfort = self.history['outcomes'][-1]
            delta = discomfort - prev_discomfort
            
            # Good outcome - strengthen current patterns
            if delta < 0:
                self.preference_momentum = 0.9 * self.preference_momentum + 0.1 * self.field
                self.preference_field += self.preference_momentum * 0.01
            # Bad outcome - weaken current patterns
            else:
                self.preference_momentum = 0.9 * self.preference_momentum - 0.1 * self.field
                self.preference_field += self.preference_momentum * 0.01
        
        # Track stability
        if len(self.history['outcomes']) > 10:
            recent_var = np.var(self.history['outcomes'][-10:])
            self.metrics['preference_stability'] = 1.0 / (1.0 + recent_var)
    
    def _update_goals(self):
        """Update persistent goals"""
        # Goals emerge from consistent preferences
        if self.metrics['preference_stability'] > 0.7:
            self.goal_field = 0.95 * self.goal_field + 0.05 * self.preference_field
    
    def _store_to_holographic_memory(self):
        """Store current state holographically"""
        # Create interference pattern
        reference = torch.randn_like(self.field, dtype=torch.complex64)
        interference = self.field.to(torch.complex64) * reference
        
        # Superpose onto hologram
        self.memory_hologram += interference * 0.01
        
        # Index by current resonances
        if self.metrics['concepts_formed'] > 0:
            # Convert to tuple of floats for hashable key
            key_values = []
            for i in range(min(3, self.metrics['concepts_formed'])):
                val = self.resonance_buffer[i].abs().mean().item()
                key_values.append(val)
            key = tuple(key_values)
            self.memory_index[key] = self.field.clone()
    
    def _attempt_recall(self) -> Optional[torch.Tensor]:
        """Try to recall relevant memory"""
        if len(self.memory_index) > 0:
            # Use current resonances as cue
            if self.metrics['concepts_formed'] > 0:
                cue = self.resonance_buffer[0]
                
                # Holographic recall
                correlation = self.memory_hologram * cue
                reconstruction = torch.fft.ifftn(
                    torch.fft.fftn(correlation, dim=(0,1,2)),
                    dim=(0,1,2)
                ).real
                
                # Measure recall accuracy (correlation with any stored memory)
                max_correlation = 0
                for stored in self.memory_index.values():
                    corr = (reconstruction * stored).mean().item()
                    max_correlation = max(max_correlation, abs(corr))
                
                self.metrics['memory_recall_accuracy'] = max_correlation
                
                return reconstruction
        
        return None
    
    def _update_resonance_coupling(self):
        """Learn which resonances work together"""
        # Strengthen coupling between co-active resonances
        active = self.resonance_buffer[:self.metrics['concepts_formed']]
        
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                if active[i].abs().mean() > 0.01 and active[j].abs().mean() > 0.01:
                    # Measure co-activation
                    coactivation = (active[i].abs() * active[j].abs()).mean()
                    
                    # Update coupling
                    self.resonance_coupling[i, j] += coactivation * 0.01
                    self.resonance_coupling[j, i] = self.resonance_coupling[i, j]
    
    def _strengthen_successful_patterns(self):
        """Reinforce patterns that led to good outcomes"""
        if len(self.history['outcomes']) > 1:
            if self.history['outcomes'][-1] < self.history['outcomes'][-2]:
                # Good outcome - strengthen active resonances
                for i in range(self.metrics['concepts_formed']):
                    self.resonance_coupling[i, i] *= 1.01  # Strengthen stability
    
    def _record_history(self):
        """Record state for learning"""
        self.history['states'].append(self.field.clone().detach())
        self.history['outcomes'].append(self._compute_discomfort())
        self.history['resonances'].append(
            self.resonance_buffer[:self.metrics['concepts_formed']].clone().detach()
        )
        
        # Limit history size
        max_history = 100
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]
    
    def _compute_attention_mask(self) -> torch.Tensor:
        """Compute attention based on energy and resonance"""
        # Energy-based attention
        energy = self.field.abs().sum(dim=-1)
        
        # Resonance-based attention
        resonance_attention = torch.zeros_like(energy)
        for resonance in self.resonance_buffer[:self.metrics['concepts_formed']]:
            if resonance.abs().mean() > 0.01:
                pattern = torch.fft.ifftn(resonance, dim=(0,1,2)).real
                resonance_attention += pattern.abs().mean(dim=-1)
        
        # Combine
        attention = torch.sigmoid(energy + resonance_attention)
        
        # Apply bandwidth constraint
        flat_attention = attention.flatten()
        k = min(self.config.attention_bandwidth, flat_attention.numel())
        threshold = torch.topk(flat_attention, k).values[-1]
        
        mask = (attention >= threshold).float().unsqueeze(-1)
        
        return mask
    
    def _create_diffusion_kernel(self) -> torch.Tensor:
        """Create 3D Laplacian kernel"""
        kernel = torch.zeros(1, 1, 3, 3, 3, device=self.device)
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        return kernel
    
    def _measure_emergence(self) -> Dict:
        """Measure indicators of emergent intelligence"""
        # Pattern emergence
        if self.metrics['cycles'] % 10 == 0:
            # Check for stable patterns
            if len(self.history['resonances']) > 2:
                pattern_stability = 0
                for i in range(len(self.history['resonances'][-1])):
                    if i < len(self.history['resonances'][-2]):
                        r1 = self.history['resonances'][-1][i]
                        r2 = self.history['resonances'][-2][i]
                        stability = (r1.conj() * r2).real.mean().item()
                        pattern_stability += abs(stability)
                
                self.metrics['emergent_patterns'].append(pattern_stability)
        
        return {
            'concepts_formed': self.metrics['concepts_formed'],
            'bindings_active': self.metrics['bindings_active'],
            'decision_confidence': self.metrics['superposition_collapse_confidence'],
            'consensus_strength': self.metrics['swarm_consensus_strength'],
            'preference_stability': self.metrics['preference_stability'],
            'memory_accuracy': self.metrics['memory_recall_accuracy'],
            'pattern_emergence': len(self.metrics['emergent_patterns']),
            'field_energy': self.field.abs().mean().item(),
            'resonance_energy': self.resonance_buffer.abs().mean().item(),
            'coupling_strength': self.resonance_coupling.abs().mean().item(),
        }
    
    def get_telemetry(self) -> Dict:
        """Get telemetry for monitoring"""
        recent_discomfort = np.mean(self.metrics['discomfort_history'][-10:]) if self.metrics['discomfort_history'] else 0
        
        return {
            'cycles': self.metrics['cycles'],
            'concepts_formed': self.metrics['concepts_formed'],
            'bindings_active': self.metrics['bindings_active'],
            'decision_confidence': self.metrics['superposition_collapse_confidence'],
            'preference_stability': self.metrics['preference_stability'],
            'memory_accuracy': self.metrics['memory_recall_accuracy'],
            'discomfort': recent_discomfort,
            'field_energy': self.field.abs().mean().item(),
            'device': self.device,
            'emergence_score': self._calculate_emergence_score()
        }
    
    def _calculate_emergence_score(self) -> float:
        """Calculate overall emergence score"""
        indicators = [
            self.metrics['concepts_formed'] > 0,
            self.metrics['bindings_active'] > 0,
            self.metrics['superposition_collapse_confidence'] > 0.5,
            self.metrics['preference_stability'] > 0.3,
            self.metrics['memory_recall_accuracy'] > 0.1,
            len(self.metrics['emergent_patterns']) > 0
        ]
        
        return sum(indicators) / len(indicators)
    
    def reset(self):
        """Reset the brain state"""
        logger.info("Resetting Critical Mass Brain")
        
        # Reinitialize all systems
        self._init_field_dynamics()
        self._init_resonance_system()
        self._init_binding_system()
        self._init_superposition_system()
        self._init_swarm_system()
        self._init_preference_system()
        self._init_memory_system()
        
        # Reset metrics
        self.metrics = {
            'cycles': 0,
            'concepts_formed': 0,
            'bindings_active': 0,
            'superposition_collapse_confidence': 0,
            'swarm_consensus_strength': 0,
            'preference_stability': 0,
            'memory_recall_accuracy': 0,
            'emergent_patterns': [],
            'discomfort_history': []
        }
        
        # Reset history
        self.history = {
            'states': [],
            'actions': [],
            'outcomes': [],
            'resonances': []
        }
        
        logger.info("Brain reset complete")