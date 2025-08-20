"""
Intrinsic Tensions - The Source of Motivation

This system creates inherent field tensions that drive behavior without external rewards.
The brain "wants" certain states and feels "discomfort" when not in them.

Core principle: Motivation emerges from the tension between current and comfortable states.
"""

import torch
from typing import Tuple, Dict, Any


class IntrinsicTensions:
    """
    Creates inherent tensions in the field that generate motivation.
    
    No rewards, no goals - just inherent discomforts that drive action.
    """
    
    def __init__(self, field_shape: Tuple[int, int, int, int], device: torch.device):
        """
        Initialize tension system.
        
        Args:
            field_shape: Shape of the 4D field tensor
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Comfort parameters - what the field "wants"
        self.resting_potential = 0.1  # Field wants to be slightly active, not zero
        self.min_gradient = 0.01  # Below this, field feels "bored"
        self.max_flatness = 0.95  # Too uniform = uncomfortable
        self.comfort_variance = 0.05  # Ideal local variance
        
        # Oscillation parameters
        self.base_frequency = 0.1  # Natural rhythm
        self.frequency_variance = 0.02  # Different regions oscillate differently
        
        # Create frequency map - different regions have different natural rhythms
        self.frequency_map = self.base_frequency + torch.randn(field_shape, device=device) * self.frequency_variance
        
        # Phase accumulator for oscillations
        self.phase = torch.zeros(field_shape, device=device)
        
        # Asymmetric decay map - creates natural biases/personality
        self.decay_map = 0.995 + torch.randn(field_shape, device=device) * 0.005
        self.decay_map = torch.clamp(self.decay_map, 0.98, 1.0)
        
        # Track cycles for varying tensions
        self.cycle = 0
        
    def apply_tensions(self, field: torch.Tensor, prediction_error: float = 0.0) -> torch.Tensor:
        """
        Apply all intrinsic tensions to the field.
        
        This is where motivation originates - not from rewards but from
        the field's inherent desire for certain states.
        
        Args:
            field: Current field state
            prediction_error: Current prediction error (creates additional tension)
            
        Returns:
            Field with tensions applied
        """
        self.cycle += 1
        
        # 1. RESTING POTENTIAL - field wants to be around resting_potential, not zero
        # When field average is too low, it feels "starved"
        # When too high, it feels "oversaturated"
        field_mean = field.mean()
        starvation = (self.resting_potential - field_mean) * 0.01
        field = field + starvation  # Global pull toward resting state
        
        # 2. GRADIENT HUNGER - field gets uncomfortable when too uniform
        # Compute local variance (how much variation in small neighborhoods)
        local_variance = self._compute_local_variance(field)
        
        # Where variance is too low, inject noise (boredom → exploration)
        boredom_mask = local_variance < self.min_gradient
        boredom_noise = torch.randn_like(field) * 0.02
        field = torch.where(boredom_mask, field + boredom_noise, field)
        
        # 3. OSCILLATORY DRIVE - natural rhythms prevent stasis
        # Update phase
        self.phase += self.frequency_map
        
        # Create oscillation that varies with field strength
        # Stronger field regions oscillate more
        oscillation = 0.01 * torch.sin(self.phase) * (1 + torch.abs(field))
        field = field + oscillation
        
        # 4. ASYMMETRIC DECAY - creates personality/preferences
        # Some regions naturally maintain activity better
        field = field * self.decay_map
        
        # 5. PREDICTION ERROR TENSION - surprise creates discomfort
        if prediction_error > 0.01:
            # Error creates literal turbulence
            error_heat = torch.randn_like(field) * prediction_error * 0.1
            field = field + error_heat
            
            # Error also disrupts natural rhythms (uncomfortable)
            self.phase += torch.randn_like(self.phase) * prediction_error
        
        # 6. EDGE DETECTION - boundaries are interesting
        # Compute gradients
        gradients = self._compute_gradients(field)
        gradient_magnitude = torch.sqrt(gradients[0]**2 + gradients[1]**2 + gradients[2]**2)
        
        # Enhance regions with strong gradients (make edges more prominent)
        edge_enhancement = gradient_magnitude * 0.01
        field = field + edge_enhancement
        
        # 7. INFORMATION STARVATION - too little activity is uncomfortable
        activity_level = torch.abs(field).mean()
        if activity_level < 0.05:  # Getting too quiet
            # Inject energy proportional to how starved we are
            starvation_energy = (0.05 - activity_level) * 10
            field = field + torch.randn_like(field) * starvation_energy * 0.05
        
        return field
    
    def _compute_local_variance(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute local variance in small neighborhoods.
        High variance = interesting, Low variance = boring.
        """
        # Use convolution with small kernel to compute local statistics
        kernel_size = 3
        padding = kernel_size // 2
        
        # Unfold to get local patches
        # For simplicity, just compute variance along channel dimension
        var_per_channel = torch.var(field, dim=-1, keepdim=True)
        
        # Expand to match field shape
        local_variance = var_per_channel.expand_as(field)
        
        return local_variance
    
    def _compute_gradients(self, field: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute spatial gradients of the field.
        These represent "edges" or boundaries between different regions.
        """
        # Simple gradient computation using diff
        # Prepend first element to maintain shape
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        
        return dx, dy, dz
    
    def get_comfort_metrics(self, field: torch.Tensor) -> Dict[str, float]:
        """
        Measure how "comfortable" the current field state is.
        GPU-optimized with batched computation to reduce synchronization.
        
        This helps us understand what drives the system's behavior.
        """
        # Batch all metric computations on GPU
        field_mean_gpu = field.mean()
        field_var_gpu = field.var()
        activity_gpu = torch.abs(field).mean()
        local_var_gpu = self._compute_local_variance(field).mean()
        
        # Stack for single CPU transfer
        metrics_gpu = torch.stack([
            field_mean_gpu,
            field_var_gpu,
            activity_gpu,
            local_var_gpu
        ])
        
        # Single CPU transfer for all metrics
        metrics_cpu = metrics_gpu.cpu().numpy()
        
        field_mean = metrics_cpu[0]
        field_var = metrics_cpu[1]
        activity = metrics_cpu[2]
        local_var = metrics_cpu[3]
        
        # Compute comfort scores (using numpy for efficiency)
        resting_comfort = 1.0 - abs(field_mean - self.resting_potential) / self.resting_potential
        variance_comfort = min(local_var / self.comfort_variance, 1.0)
        activity_comfort = min(activity / 0.1, 1.0)  # Want at least 0.1 activity
        
        # Overall comfort is minimum of all factors (weakest link)
        overall_comfort = min(resting_comfort, variance_comfort, activity_comfort)
        
        return {
            'overall_comfort': overall_comfort,
            'resting_comfort': resting_comfort,
            'variance_comfort': variance_comfort,
            'activity_comfort': activity_comfort,
            'field_mean': field_mean,
            'field_variance': field_var,
            'activity_level': activity,
            'local_variance': local_var
        }
    
    def reset(self):
        """Reset oscillation phases and other temporal states."""
        self.phase = torch.zeros(self.field_shape, device=self.device)
        self.cycle = 0


class MotivationalDynamics:
    """
    Higher-level motivational dynamics that emerge from tensions.
    
    This class interprets the comfort metrics to understand what
    "drives" or "motivates" the system at any moment.
    """
    
    def __init__(self):
        self.comfort_history = []
        self.action_history = []
        
    def interpret_motivation(self, comfort_metrics: Dict[str, float]) -> str:
        """
        Interpret current motivational state from comfort metrics.
        
        Returns a human-readable description of what's driving the system.
        """
        # Use torch to find minimum instead of Python's min
        comfort_values = torch.tensor([
            comfort_metrics['resting_comfort'],
            comfort_metrics['variance_comfort'],
            comfort_metrics['activity_comfort']
        ])
        lowest_comfort = comfort_values.min().item()
        
        if comfort_metrics['activity_comfort'] == lowest_comfort:
            if comfort_metrics['activity_level'] < 0.05:
                return "STARVED - Desperately seeking stimulation"
            else:
                return "OVERSTIMULATED - Seeking calm"
        
        elif comfort_metrics['variance_comfort'] == lowest_comfort:
            if comfort_metrics['local_variance'] < 0.01:
                return "BORED - Seeking novelty and variation"
            else:
                return "CHAOTIC - Seeking patterns and stability"
        
        elif comfort_metrics['resting_comfort'] == lowest_comfort:
            if comfort_metrics['field_mean'] < 0.05:
                return "DORMANT - Waking up, seeking energy"
            else:
                return "HYPERACTIVE - Seeking equilibrium"
        
        if comfort_metrics['overall_comfort'] > 0.8:
            return "CONTENT - Maintaining comfortable state"
        elif comfort_metrics['overall_comfort'] > 0.5:
            return "EXPLORING - Mildly seeking improvement"
        else:
            return "DISTRESSED - Urgently seeking comfort"
    
    def should_explore(self, comfort_metrics: Dict[str, float]) -> bool:
        """
        Decide whether the system should explore (vs exploit).
        
        Low comfort → High exploration (try something new)
        High comfort → Low exploration (maintain current state)
        """
        # Track comfort over time
        self.comfort_history.append(comfort_metrics['overall_comfort'])
        
        # If comfort is low, definitely explore
        if comfort_metrics['overall_comfort'] < 0.3:
            return True
        
        # If comfort hasn't improved recently, explore
        if len(self.comfort_history) > 10:
            recent_comfort = self.comfort_history[-5:]
            older_comfort = self.comfort_history[-10:-5]
            # Use Python's built-in mean to avoid numpy dependency
            recent_mean = sum(recent_comfort) / len(recent_comfort)
            older_mean = sum(older_comfort) / len(older_comfort)
            if recent_mean <= older_mean:
                return True
        
        # Otherwise, exploration probability based on comfort
        # High comfort = low exploration probability
        explore_prob = 1.0 - comfort_metrics['overall_comfort']
        # Use torch for random generation (stays on GPU if possible)
        return torch.rand(1, device=self.device).item() < explore_prob * 0.3  # Max 30% exploration when comfortable