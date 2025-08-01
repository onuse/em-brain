#!/usr/bin/env python3
"""
Field-native strategic planning through persistent attractor patterns.
No symbolic goals - just field configurations that shape behavior.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

class FieldStrategicPlanner:
    """
    Discovers field patterns that, when held in memory channels,
    create behavioral attractors leading to good outcomes.
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, int, int, int],
                 device: torch.device):
        self.field_shape = field_shape
        self.device = device
        
        # Which channels hold strategic patterns
        # Using temporal features (32-47) as "working memory"
        self.strategy_channels = slice(32, 48)
        
        # Discovered beneficial patterns
        self.pattern_library = {}
        
    def discover_strategic_pattern(self, 
                                 current_field: torch.Tensor,
                                 recent_rewards: List[float],
                                 n_candidates: int = 16) -> torch.Tensor:
        """
        Discover a field pattern that shapes future evolution beneficially.
        
        This is the "deep think" - but instead of planning actions,
        we're discovering the shape of thoughts that lead to good outcomes.
        """
        best_pattern = None
        best_value = -float('inf')
        
        for _ in range(n_candidates):
            # Generate candidate pattern for memory channels
            candidate = self._generate_candidate_pattern()
            
            # Test how this pattern shapes field evolution
            value = self._evaluate_pattern_influence(
                current_field,
                candidate,
                recent_rewards
            )
            
            if value > best_value:
                best_value = value
                best_pattern = candidate
                
        return best_pattern
    
    def _generate_candidate_pattern(self) -> torch.Tensor:
        """
        Generate a candidate strategic pattern.
        
        These aren't random - they're variations on successful patterns,
        combinations of known patterns, or novel explorations.
        """
        pattern = torch.zeros(
            self.field_shape[0],
            self.field_shape[1], 
            self.field_shape[2],
            16,  # Just the strategy channels
            device=self.device
        )
        
        # Create different types of patterns
        pattern_type = torch.randint(0, 4, (1,)).item()
        
        if pattern_type == 0:
            # Gradient pattern - creates directional flow
            for i in range(self.field_shape[0]):
                pattern[i, :, :, :] = i / self.field_shape[0]
                
        elif pattern_type == 1:
            # Radial pattern - creates centering behavior
            center = torch.tensor([s//2 for s in self.field_shape[:3]], device=self.device)
            for i in range(self.field_shape[0]):
                for j in range(self.field_shape[1]):
                    for k in range(self.field_shape[2]):
                        dist = torch.norm(torch.tensor([i,j,k], device=self.device).float() - center)
                        pattern[i,j,k,:] = torch.exp(-dist/10)
                        
        elif pattern_type == 2:
            # Wave pattern - creates oscillatory behavior
            freq = torch.rand(1, device=self.device) * 0.5
            for i in range(self.field_shape[0]):
                pattern[i, :, :, :] = torch.sin(i * freq)
                
        else:
            # Sparse activation - creates focused behavior
            n_peaks = torch.randint(2, 8, (1,)).item()
            for _ in range(n_peaks):
                pos = [torch.randint(0, s, (1,)).item() for s in self.field_shape[:3]]
                pattern[pos[0], pos[1], pos[2], :] = torch.randn(16, device=self.device)
                
        # Add noise for variation
        pattern += torch.randn_like(pattern) * 0.1
        
        return pattern
    
    def _evaluate_pattern_influence(self,
                                  base_field: torch.Tensor,
                                  pattern: torch.Tensor,
                                  recent_rewards: List[float]) -> float:
        """
        Evaluate how a strategic pattern influences field evolution.
        
        Good patterns create:
        1. Stable attractors (persistent behavior)
        2. Reward correlation (lead to good outcomes)
        3. Robustness (work across variations)
        """
        # Create test field with pattern installed
        test_field = base_field.clone()
        test_field[:, :, :, self.strategy_channels] = pattern
        
        # Simulate evolution
        trajectory_value = 0.0
        field = test_field
        
        for t in range(50):  # Longer horizon for strategic evaluation
            # Evolve field (simplified - would use actual field dynamics)
            field = self._simple_field_evolution(field)
            
            # Extract behavioral tendency from field
            behavior_vector = self._extract_behavior_tendency(field)
            
            # Evaluate this state
            # (In reality, would predict reward from field configuration)
            state_value = self._estimate_state_value(field, behavior_vector, recent_rewards)
            
            # Discount future values
            trajectory_value += state_value * (0.95 ** t)
            
        # Also evaluate pattern stability
        stability = self._measure_pattern_stability(pattern, field)
        
        return trajectory_value + stability * 0.2
    
    def _simple_field_evolution(self, field: torch.Tensor) -> torch.Tensor:
        """Simplified field evolution for pattern evaluation."""
        # Decay
        evolved = field * 0.99
        
        # Diffusion (simplified)
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27
        for c in range(field.shape[-1]):
            channel = field[:, :, :, c].unsqueeze(0).unsqueeze(0)
            if channel.shape[2] >= 3:  # Only if large enough
                diffused = F.conv3d(channel, kernel, padding=1)
                evolved[:, :, :, c] += diffused.squeeze() * 0.1
        
        # Nonlinearity
        evolved = torch.tanh(evolved * 1.05)
        
        # Pattern persistence in memory channels
        evolved[:, :, :, self.strategy_channels] = field[:, :, :, self.strategy_channels] * 0.95
        
        return evolved
    
    def _extract_behavior_tendency(self, field: torch.Tensor) -> torch.Tensor:
        """Extract behavioral tendency from field state."""
        # In real implementation, this would use motor cortex mapping
        # Here, we just take mean activation in different regions
        
        # Divide field into quadrants and extract tendencies
        mid_x = field.shape[0] // 2
        mid_y = field.shape[1] // 2
        
        forward_tendency = field[:mid_x, :, :, :].mean()
        backward_tendency = field[mid_x:, :, :, :].mean()
        left_tendency = field[:, :mid_y, :, :].mean()
        right_tendency = field[:, mid_y:, :, :].mean()
        
        return torch.tensor([
            forward_tendency - backward_tendency,
            right_tendency - left_tendency
        ], device=self.device)
    
    def _estimate_state_value(self, 
                            field: torch.Tensor,
                            behavior: torch.Tensor,
                            recent_rewards: List[float]) -> float:
        """Estimate value of a field state."""
        # In reality, would use learned value function
        # Here, use simple heuristics
        
        # High energy in content features (0-31) is generally good
        content_energy = field[:, :, :, :32].abs().mean().item()
        
        # Low energy in error features might indicate stability
        if field.shape[-1] > 48:
            error_energy = field[:, :, :, 48:].abs().mean().item()
        else:
            error_energy = 0.0
            
        # Movement toward higher rewards
        if recent_rewards:
            reward_gradient = recent_rewards[-1] - sum(recent_rewards) / len(recent_rewards)
            movement_value = behavior[0].item() * reward_gradient
        else:
            movement_value = 0.0
            
        return content_energy - error_energy * 0.5 + movement_value * 2.0
    
    def _measure_pattern_stability(self, 
                                 pattern: torch.Tensor,
                                 evolved_field: torch.Tensor) -> float:
        """Measure how well pattern persists through evolution."""
        if evolved_field.shape[-1] > 47:
            evolved_pattern = evolved_field[:, :, :, self.strategy_channels]
            correlation = F.cosine_similarity(
                pattern.flatten(),
                evolved_pattern.flatten(),
                dim=0
            )
            return correlation.item()
        return 0.0

def demonstrate_field_strategic_planning():
    """Show how field-native strategic planning works."""
    
    print("Field-Native Strategic Planning")
    print("="*50)
    
    # Setup
    device = torch.device('cpu')  # For demo
    field_shape = (16, 16, 16, 64)  # Smaller for demo
    current_field = torch.randn(*field_shape, device=device) * 0.1
    
    planner = FieldStrategicPlanner(field_shape, device)
    
    print("\n1. Current field state:")
    print(f"   Shape: {current_field.shape}")
    print(f"   Energy: {current_field.abs().mean():.3f}")
    
    print("\n2. Discovering strategic pattern...")
    print("   (This is the 'deep think' - finding beneficial field configurations)")
    
    # Simulate recent rewards
    recent_rewards = [0.2, 0.3, 0.5, 0.4, 0.6]  # Improving trend
    
    # Discover pattern
    strategic_pattern = planner.discover_strategic_pattern(
        current_field,
        recent_rewards,
        n_candidates=8
    )
    
    print(f"   Found pattern with shape: {strategic_pattern.shape}")
    print(f"   Pattern energy: {strategic_pattern.abs().mean():.3f}")
    
    print("\n3. Installing pattern in memory channels...")
    current_field[:, :, :, 32:48] = strategic_pattern
    
    print("\n4. Observing natural behavior emergence...")
    field = current_field
    for cycle in range(5):
        # Natural field evolution
        field = planner._simple_field_evolution(field)
        
        # Extract emergent behavior
        behavior = planner._extract_behavior_tendency(field)
        
        print(f"   Cycle {cycle+1}: Forward: {behavior[0]:+.3f}, Turn: {behavior[1]:+.3f}")
    
    print("\n" + "="*50)
    print("Key Insights:")
    print("- No explicit goals or rules")
    print("- Pattern shapes field evolution")  
    print("- Behavior emerges from field dynamics")
    print("- Strategy IS the field configuration")

if __name__ == "__main__":
    demonstrate_field_strategic_planning()