#!/usr/bin/env python3
"""
GPU Future Simulator - Mental Simulation for Action Evaluation

This module enables the brain to mentally simulate multiple possible futures
in parallel on the GPU, dramatically improving action selection by actually
testing what might happen rather than relying on simple linear projections.

Key features:
- Simulates 8-32 futures per action candidate
- Runs entirely on GPU in parallel with CPU operations
- Provides uncertainty estimates based on outcome variance
- Integrates seamlessly with existing action-prediction system
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from ...utils.tensor_ops import create_zeros, create_randn
from .action_prediction_system import PredictiveAction


@dataclass
class SimulatedAction:
    """Enhanced action candidate with GPU simulation results."""
    original: PredictiveAction
    
    # Simulation results
    outcome_trajectories: torch.Tensor  # [n_futures, horizon, sensory_dim]
    outcome_variance: float  # How much futures diverge
    outcome_stability: float  # How stable the prediction is
    surprise_potential: float  # Expected novelty
    convergence_time: int  # When futures converge (if they do)
    
    # Aggregated prediction (weighted by confidence)
    simulated_outcome: torch.Tensor  # [sensory_dim]
    simulation_confidence: float  # Based on convergence


class GPUFutureSimulator:
    """
    Simulates multiple possible futures in parallel on GPU.
    
    This is the key to intelligent action selection - rather than guessing
    what might happen, we actually simulate it at high speed.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 sensory_dim: int,
                 motor_dim: int,
                 n_futures: int = 32,
                 horizon: int = 20,
                 device: torch.device = None):
        """
        Initialize the future simulator.
        
        Args:
            field_shape: Shape of the unified field
            sensory_dim: Dimension of sensory input
            motor_dim: Dimension of motor output
            n_futures: Number of parallel futures to simulate
            horizon: How many cycles to simulate ahead
            device: GPU device to use
        """
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.n_futures = n_futures
        self.horizon = horizon
        self.device = device if device else torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simplified field dynamics for GPU (batched operations)
        self.decay_rate = 0.995
        self.diffusion_kernel = self._create_diffusion_kernel()
        
        # Learned forward model: field + action -> next field
        # This is a simplified version that can run efficiently on GPU
        self.forward_model = self._create_forward_model()
        
        # Buffer for simulated trajectories
        self.trajectory_buffer = create_zeros(
            (n_futures, horizon, *field_shape), 
            device=self.device
        )
        
        # Action effect templates (learned over time)
        self.action_effects = create_zeros(
            (motor_dim, *field_shape),
            device=self.device
        )
        torch.nn.init.xavier_uniform_(self.action_effects, gain=0.1)
        
        # Thread pool for async evaluation
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GPUPlanning")
        self._planning_lock = threading.Lock()
        self._current_planning_future = None
        
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
        
    def _create_forward_model(self) -> torch.nn.Module:
        """Create simplified forward dynamics model for GPU."""
        # For now, use deterministic dynamics
        # In future, this could be a learned neural network
        return None  # We'll use analytical dynamics for efficiency
        
    def evaluate_action_candidates(self,
                                 candidates: List[PredictiveAction],
                                 current_field: torch.Tensor,
                                 confidence: float,
                                 exploration_drive: float = 0.5) -> List[SimulatedAction]:
        """
        Evaluate action candidates by simulating their outcomes.
        
        Args:
            candidates: List of action candidates to evaluate
            current_field: Current field state
            confidence: Current prediction confidence
            exploration_drive: How much to vary futures (0=deterministic, 1=high variance)
            
        Returns:
            List of enhanced candidates with simulation results
        """
        simulated_actions = []
        
        for candidate in candidates:
            # Fork multiple futures with this action
            initial_states = self._fork_futures_with_action(
                current_field,
                candidate.motor_values,
                n_variations=min(8, self.n_futures // len(candidates)),
                variance=exploration_drive * 0.1
            )
            
            # Simulate futures forward
            trajectories = self._simulate_futures(initial_states, self.horizon)
            
            # Extract sensory predictions from trajectories
            sensory_trajectories = self._extract_sensory_predictions(trajectories)
            
            # Analyze the simulated futures
            analysis = self._analyze_futures(sensory_trajectories)
            
            # Create enhanced action with simulation results
            simulated_actions.append(SimulatedAction(
                original=candidate,
                outcome_trajectories=sensory_trajectories,
                outcome_variance=analysis['variance'],
                outcome_stability=analysis['stability'],
                surprise_potential=analysis['surprise'],
                convergence_time=analysis['convergence_time'],
                simulated_outcome=analysis['mean_outcome'],
                simulation_confidence=analysis['confidence']
            ))
            
        return simulated_actions
        
    def evaluate_async(self,
                      candidates: List[PredictiveAction],
                      current_field: torch.Tensor,
                      confidence: float,
                      exploration_drive: float = 0.5,
                      callback: Optional[Callable] = None) -> Future:
        """
        Evaluate action candidates asynchronously in background.
        
        This is the key to decoupled planning - evaluation happens in
        background while the brain continues with cached plans.
        
        Args:
            candidates: List of action candidates to evaluate
            current_field: Current field state (will be cloned)
            confidence: Current prediction confidence
            exploration_drive: How much to vary futures
            callback: Optional callback when evaluation completes
            
        Returns:
            Future that will contain list of SimulatedActions
        """
        # Clone field to avoid race conditions
        field_snapshot = current_field.clone().detach()
        
        def _background_evaluation():
            try:
                # Move computation to GPU thread
                # Use appropriate device context based on backend
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        # Perform full evaluation
                        results = self.evaluate_action_candidates(
                            candidates=candidates,
                            current_field=field_snapshot,
                            confidence=confidence,
                            exploration_drive=exploration_drive
                        )
                else:
                    # For MPS or CPU, no special device context needed
                    results = self.evaluate_action_candidates(
                        candidates=candidates,
                        current_field=field_snapshot,
                        confidence=confidence,
                        exploration_drive=exploration_drive
                    )
                    
                # Optional callback
                if callback:
                    callback(results)
                    
                return results
            except Exception as e:
                print(f"Background planning error: {e}")
                return []
        
        # Submit to executor
        with self._planning_lock:
            # Cancel any existing planning
            if self._current_planning_future and not self._current_planning_future.done():
                self._current_planning_future.cancel()
            
            # Start new planning
            future = self.executor.submit(_background_evaluation)
            self._current_planning_future = future
            
        return future
        
    def evaluate_candidates_with_timeout(self,
                                       candidates: List[PredictiveAction],
                                       current_field: torch.Tensor,
                                       confidence: float,
                                       exploration_drive: float = 0.5,
                                       timeout: float = 0.1) -> List[SimulatedAction]:
        """
        Evaluate candidates with a timeout, returning partial results if needed.
        
        This allows the brain to get whatever planning is available within
        the time budget, falling back to simple predictions if needed.
        
        Args:
            candidates: List of action candidates
            current_field: Current field state
            confidence: Current prediction confidence
            exploration_drive: Exploration parameter
            timeout: Maximum time to wait (seconds)
            
        Returns:
            List of simulated actions (may be empty if timeout)
        """
        future = self.evaluate_async(
            candidates=candidates,
            current_field=current_field,
            confidence=confidence,
            exploration_drive=exploration_drive
        )
        
        try:
            # Wait for results with timeout
            return future.result(timeout=timeout)
        except:
            # Timeout or error - return empty list
            return []
        
    def _fork_futures_with_action(self,
                                field: torch.Tensor,
                                action: torch.Tensor,
                                n_variations: int,
                                variance: float) -> torch.Tensor:
        """
        Create multiple future variations from current state + action.
        
        Args:
            field: Current field state [D, H, W, C]
            action: Motor action to apply
            n_variations: Number of variations to create
            variance: How much to vary the futures
            
        Returns:
            Batch of initial states [n_variations, D, H, W, C]
        """
        # Repeat field for all variations
        batch_fields = field.unsqueeze(0).repeat(n_variations, 1, 1, 1, 1)
        
        # Apply action influence to field
        action_influence = self._compute_action_influence(action)
        # Action influence affects the entire field, not just first 4 channels
        batch_fields += action_influence.unsqueeze(0) * 0.1  # Scale down influence
        
        # Add variation to create different futures
        if variance > 0:
            noise = create_randn(batch_fields.shape, device=self.device) * variance
            # Apply more noise to dynamic features, less to stable features
            noise_mask = torch.ones_like(batch_fields)
            noise_mask[:, :, :, :, :16] *= 0.5  # Less noise on spatial features
            batch_fields += noise * noise_mask
            
        return batch_fields
        
    def _compute_action_influence(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute how an action influences the field.
        
        Args:
            action: Motor action vector
            
        Returns:
            Field-shaped influence pattern
        """
        # Weight action effects by motor values
        influence = torch.zeros(self.field_shape, device=self.device)
        
        # Simple model: each motor dimension affects certain field regions
        for i, motor_val in enumerate(action):
            if i < self.action_effects.shape[0]:
                influence += self.action_effects[i] * motor_val
                
        return influence
        
    def _simulate_futures(self,
                        initial_states: torch.Tensor,
                        horizon: int) -> torch.Tensor:
        """
        Simulate batch of futures forward in time.
        
        Args:
            initial_states: Batch of initial field states [B, D, H, W, C]
            horizon: Number of steps to simulate
            
        Returns:
            Trajectories [B, T, D, H, W, C]
        """
        batch_size = initial_states.shape[0]
        trajectories = []
        
        current_states = initial_states
        
        for t in range(horizon):
            # Evolve all futures in parallel
            next_states = self._batch_evolve_fields(current_states)
            trajectories.append(next_states)
            current_states = next_states
            
        return torch.stack(trajectories, dim=1)
        
    def _batch_evolve_fields(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Evolve batch of fields one step (simplified dynamics for GPU).
        
        Args:
            fields: Batch of fields [B, D, H, W, C]
            
        Returns:
            Evolved fields [B, D, H, W, C]
        """
        batch_size = fields.shape[0]
        
        # 1. Apply decay
        evolved = fields * self.decay_rate
        
        # 2. Apply diffusion (batched convolution)
        # Process each feature channel
        for c in range(fields.shape[-1]):
            # Get channel data [B, 1, D, H, W]
            channel_data = fields[:, :, :, :, c].unsqueeze(1)
            
            # Apply 3D convolution for diffusion
            diffused = F.conv3d(
                channel_data,
                self.diffusion_kernel,
                padding=1
            )
            
            # Add diffusion to evolution
            evolved[:, :, :, :, c] += diffused.squeeze(1)
            
        # 3. Apply simple nonlinearity
        evolved = torch.tanh(evolved * 1.1)
        
        # 4. Add small spontaneous activity
        spontaneous = create_randn(evolved.shape, device=self.device) * 0.01
        evolved += spontaneous
        
        return evolved
        
    def _extract_sensory_predictions(self, 
                                    trajectories: torch.Tensor) -> torch.Tensor:
        """
        Extract predicted sensory values from field trajectories.
        
        Args:
            trajectories: Field evolution trajectories [B, T, D, H, W, C]
            
        Returns:
            Sensory predictions [B, T, sensory_dim]
        """
        batch_size, horizon = trajectories.shape[:2]
        
        # Simple extraction: average activation in specific regions
        # In future, this could use learned attention or topology regions
        sensory_predictions = []
        
        for t in range(horizon):
            fields = trajectories[:, t]
            
            # Extract features from different spatial regions
            # This is a simplified version - could be much more sophisticated
            predictions = []
            
            # Divide field into sensory_dim regions and extract features
            region_size = self.field_shape[0] // int(np.sqrt(self.sensory_dim))
            
            for i in range(self.sensory_dim):
                # Simple spatial grid mapping
                row = i // int(np.sqrt(self.sensory_dim))
                col = i % int(np.sqrt(self.sensory_dim))
                
                r_start = row * region_size
                r_end = min(r_start + region_size, self.field_shape[0])
                c_start = col * region_size
                c_end = min(c_start + region_size, self.field_shape[1])
                
                # Extract mean activation from region
                region_activation = fields[:, r_start:r_end, c_start:c_end, :, :16].mean(dim=(1,2,3,4))
                predictions.append(region_activation)
                
            sensory_pred = torch.stack(predictions, dim=1)
            sensory_predictions.append(sensory_pred)
            
        return torch.stack(sensory_predictions, dim=1)
        
    def _analyze_futures(self, 
                        sensory_trajectories: torch.Tensor) -> Dict[str, any]:
        """
        Analyze simulated futures to extract useful statistics.
        
        Args:
            sensory_trajectories: Predicted sensory values [B, T, sensory_dim]
            
        Returns:
            Dictionary of analysis results
        """
        batch_size, horizon, sensory_dim = sensory_trajectories.shape
        
        # Compute variance across futures at each timestep
        variances = torch.var(sensory_trajectories, dim=0)  # [T, sensory_dim]
        mean_variance = variances.mean().item()
        
        # Compute stability (how much predictions change over time)
        differences = torch.diff(sensory_trajectories, dim=1)
        stability = 1.0 - torch.mean(torch.abs(differences)).item()
        
        # Find convergence time (when variance drops below threshold)
        variance_trajectory = variances.mean(dim=1)  # [T]
        convergence_mask = variance_trajectory < 0.1
        if convergence_mask.any():
            convergence_time = convergence_mask.nonzero()[0].item()
        else:
            convergence_time = horizon
            
        # Compute surprise potential (expected novelty)
        final_states = sensory_trajectories[:, -1]  # [B, sensory_dim]
        surprise = torch.std(final_states, dim=0).mean().item()
        
        # Mean outcome (weighted by convergence)
        weights = torch.exp(-torch.arange(horizon, device=self.device).float() * 0.1)
        weights = weights.unsqueeze(0).unsqueeze(2)  # [1, T, 1]
        weighted_mean = (sensory_trajectories * weights).sum(dim=1) / weights.sum()
        mean_outcome = weighted_mean.mean(dim=0)  # [sensory_dim]
        
        # Confidence based on convergence and stability
        confidence = stability * (1.0 - min(1.0, convergence_time / horizon))
        
        return {
            'variance': mean_variance,
            'stability': stability,
            'surprise': surprise,
            'convergence_time': convergence_time,
            'mean_outcome': mean_outcome,
            'confidence': confidence
        }
        
    def update_action_effects(self, 
                            action: torch.Tensor,
                            field_before: torch.Tensor,
                            field_after: torch.Tensor,
                            learning_rate: float = 0.01):
        """
        Update learned action effects based on observed outcomes.
        
        Args:
            action: Motor action that was taken
            field_before: Field state before action
            field_after: Field state after action
            learning_rate: How fast to update the model
        """
        # Compute observed field change
        field_change = field_after - field_before
        
        # Update action effect templates
        for i, motor_val in enumerate(action):
            if i < self.action_effects.shape[0] and abs(motor_val) > 0.01:
                # Attribute part of the change to this motor dimension
                self.action_effects[i] += learning_rate * motor_val * field_change
                
        # Decay unused action effects
        self.action_effects *= 0.999
        
    def shutdown(self):
        """Clean shutdown of background threads."""
        self.executor.shutdown(wait=True)