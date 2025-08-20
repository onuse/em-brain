"""
Temporal Planning - Thinking Ahead

Enables multi-step planning through predictive wave propagation.
Like ripples in a pond showing possible futures.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Plan:
    """Represents a multi-step plan."""
    steps: List[torch.Tensor]  # Sequence of motor commands
    predicted_states: List[torch.Tensor]  # Predicted field states
    total_reward: float  # Expected total comfort
    confidence: float  # Confidence in this plan


class TemporalPlanning:
    """
    Enable planning through predictive wave propagation.
    
    Core principle: Each possible action creates a "future wave" that
    propagates through the field, allowing evaluation of consequences.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device, 
                 horizon: int = 10):
        """
        Initialize temporal planning system.
        
        Args:
            field_shape: Shape of the field tensor [D, H, W, C]
            device: Computation device
            horizon: How many steps to plan ahead
        """
        self.field_shape = field_shape
        self.device = device
        self.horizon = horizon
        
        # Future prediction buffers
        self.future_buffer = torch.zeros(horizon, *field_shape, device=device)
        self.action_buffer = torch.zeros(horizon, 6, device=device)  # 6 motor dims
        
        # Wave propagation parameters
        self.wave_speed = 0.2  # How fast future waves propagate
        self.wave_decay = 0.9  # How fast future waves decay
        
        # Planning parameters
        self.n_samples = 5  # Number of action sequences to sample
        self.temperature = 0.1  # For action sampling (lower = more deterministic)
        
        # Goal representation (if any)
        self.current_goal = None
        self.goal_position = None
        
    def plan_sequence(self, field: torch.Tensor, 
                     current_motors: List[float],
                     comfort_fn=None) -> Plan:
        """
        Plan a sequence of actions by simulating futures.
        
        Args:
            field: Current field state
            current_motors: Current motor values
            comfort_fn: Function to evaluate field comfort
            
        Returns:
            Best plan found
        """
        best_plan = None
        best_reward = float('-inf')
        
        # Sample multiple action sequences
        for _ in range(self.n_samples):
            # Generate candidate action sequence
            action_sequence = self._sample_action_sequence(current_motors)
            
            # Simulate this sequence
            predicted_states = self._simulate_sequence(field, action_sequence)
            
            # Evaluate sequence
            reward = self._evaluate_sequence(predicted_states, comfort_fn)
            
            if reward > best_reward:
                best_reward = reward
                best_plan = Plan(
                    steps=action_sequence,
                    predicted_states=predicted_states,
                    total_reward=reward,
                    confidence=self._compute_confidence(predicted_states)
                )
        
        return best_plan
    
    def _sample_action_sequence(self, current_motors: List[float]) -> List[torch.Tensor]:
        """
        Sample a sequence of motor commands.
        
        Uses current motors as starting point with gradual exploration.
        """
        sequence = []
        motors = torch.tensor(current_motors, device=self.device)
        
        for t in range(self.horizon):
            # Add noise that increases with time (more uncertain about distant future)
            noise_scale = self.temperature * (1 + t / self.horizon)
            noise = torch.randn_like(motors) * noise_scale
            
            # Smooth evolution from current motors
            next_motors = motors * 0.9 + noise
            next_motors = torch.tanh(next_motors)  # Keep in [-1, 1]
            
            sequence.append(next_motors)
            motors = next_motors
        
        return sequence
    
    def _simulate_sequence(self, field: torch.Tensor, 
                          action_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Simulate field evolution under action sequence.
        
        Uses wave propagation to predict future states.
        """
        predicted_states = []
        current_field = field.clone()
        
        for t, action in enumerate(action_sequence):
            # Create action wave
            wave = self._action_to_wave(action, t)
            
            # Propagate wave through field
            current_field = self._propagate_wave(current_field, wave)
            
            # Apply decay
            current_field *= self.wave_decay
            
            # Store predicted state
            predicted_states.append(current_field.clone())
            self.future_buffer[t] = current_field
        
        return predicted_states
    
    def _action_to_wave(self, action: torch.Tensor, time_step: int) -> torch.Tensor:
        """
        Convert motor action to wave pattern in field.
        
        Different actions create different wave signatures.
        """
        wave = torch.zeros(self.field_shape, device=self.device)
        
        # Each motor creates a specific wave pattern
        for i, motor_value in enumerate(action):
            if i >= 6:  # Limit to 6 motors
                break
            
            # Create wave source at specific location based on motor
            # Motors 0-1: movement (affects lower field)
            # Motors 2-3: rotation (affects middle field)
            # Motors 4-5: other (affects upper field)
            
            z_layer = i * (self.field_shape[0] // 6)
            z_layer = min(z_layer, self.field_shape[0] - 1)
            
            # Create Gaussian wave source
            center_y = self.field_shape[1] // 2
            center_x = self.field_shape[2] // 2
            
            for x in range(self.field_shape[1]):
                for y in range(self.field_shape[2]):
                    distance = ((x - center_x)**2 + (y - center_y)**2) ** 0.5
                    amplitude = motor_value * torch.exp(torch.tensor(-distance / 10, device=self.device))
                    
                    # Add wave with temporal modulation
                    phase = time_step * 0.5 + i * torch.pi / 3
                    wave[z_layer, x, y, :4] += amplitude * torch.sin(
                        torch.tensor(phase, device=self.device)
                    )
        
        return wave * 0.1  # Scale down
    
    def _propagate_wave(self, field: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        """
        Propagate wave through field using diffusion-like dynamics.
        
        Waves spread and interfere, creating complex future patterns.
        """
        # Add wave to field
        field = field + wave * self.wave_speed
        
        # Propagate using convolution (wave spreading)
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27  # 3x3x3 average
        
        # Reshape for conv3d
        field_conv = field.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
        
        # Apply propagation
        propagated = F.conv3d(field_conv, kernel.expand(field.shape[3], 1, 3, 3, 3),
                             padding=1, groups=field.shape[3])
        
        # Reshape back
        field_propagated = propagated.squeeze(0).permute(1, 2, 3, 0)
        
        # Mix original and propagated
        field = field * 0.8 + field_propagated * 0.2
        
        return field
    
    def _evaluate_sequence(self, predicted_states: List[torch.Tensor],
                          comfort_fn=None) -> float:
        """
        Evaluate quality of predicted sequence.
        
        Based on comfort, goal achievement, and stability.
        """
        total_reward = 0
        
        for t, state in enumerate(predicted_states):
            # Decay factor for distant future (less certain, less weight)
            time_discount = 0.95 ** t
            
            # Basic reward: field energy and variance (activity is good)
            energy = torch.abs(state).mean().item()
            variance = state.var().item()
            basic_reward = energy * 0.5 + variance * 0.5
            
            # Comfort reward if function provided
            if comfort_fn is not None:
                comfort = comfort_fn(state)
                basic_reward += comfort * 2.0
            
            # Goal reward if goal is set
            if self.current_goal is not None:
                goal_distance = self._distance_to_goal(state)
                goal_reward = 1.0 / (1.0 + goal_distance)
                basic_reward += goal_reward * 3.0
            
            total_reward += basic_reward * time_discount
        
        return total_reward
    
    def _compute_confidence(self, predicted_states: List[torch.Tensor]) -> float:
        """
        Compute confidence in predictions.
        
        Based on stability and coherence of predicted states.
        """
        if len(predicted_states) < 2:
            return 0.5
        
        # Measure stability (how much states change)
        total_change = 0
        for i in range(1, len(predicted_states)):
            change = (predicted_states[i] - predicted_states[i-1]).abs().mean().item()
            total_change += change
        
        avg_change = total_change / (len(predicted_states) - 1)
        
        # High stability = high confidence
        confidence = 1.0 / (1.0 + avg_change * 10)
        
        return min(max(confidence, 0.0), 1.0)
    
    def set_goal(self, goal_pattern: torch.Tensor, position: Optional[torch.Tensor] = None):
        """
        Set a goal for planning.
        
        Args:
            goal_pattern: Desired field pattern
            position: Optional specific position for goal
        """
        self.current_goal = goal_pattern
        self.goal_position = position
    
    def _distance_to_goal(self, state: torch.Tensor) -> float:
        """
        Measure distance from state to goal.
        
        Uses pattern similarity or position distance.
        """
        if self.current_goal is None:
            return 0.0
        
        if self.goal_position is not None:
            # Position-based goal
            x, y, z = self.goal_position
            goal_region = state[x-2:x+3, y-2:y+3, z-2:z+3, :]
            goal_activity = torch.abs(goal_region).mean().item()
            return 1.0 - goal_activity  # Want high activity at goal
        else:
            # Pattern-based goal
            similarity = F.cosine_similarity(
                state.flatten().unsqueeze(0),
                self.current_goal.flatten().unsqueeze(0)
            ).item()
            return 1.0 - similarity
    
    def get_immediate_action(self, plan: Plan) -> List[float]:
        """
        Extract the immediate action from a plan.
        
        Args:
            plan: Multi-step plan
            
        Returns:
            Motor commands for immediate execution
        """
        if plan is None or not plan.steps:
            return [0.0] * 6
        
        return plan.steps[0].cpu().tolist()
    
    def visualize_future(self, plan: Plan) -> torch.Tensor:
        """
        Create visualization of predicted future.
        
        Returns energy map of predicted states.
        """
        if plan is None or not plan.predicted_states:
            return torch.zeros(self.horizon, *self.field_shape[:3], device=self.device)
        
        future_energy = torch.zeros(self.horizon, *self.field_shape[:3], device=self.device)
        
        for t, state in enumerate(plan.predicted_states[:self.horizon]):
            future_energy[t] = torch.abs(state).mean(dim=3)
        
        return future_energy