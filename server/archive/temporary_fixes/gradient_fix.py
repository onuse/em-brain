#!/usr/bin/env python3
"""
Gradient extraction fix for UnifiedFieldBrain.

This module contains the corrected implementation of gradient extraction
and action generation to replace the broken implementation in core_brain.py.
"""

import torch
import math
from typing import Tuple, Dict, Any


def calculate_proper_gradient_flows(unified_field: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate gradients properly for the multi-dimensional field.
    
    Args:
        unified_field: The full unified field tensor
        
    Returns:
        Dictionary with gradient tensors for each spatial dimension
    """
    # Use torch.gradient for efficient computation
    # This returns a tuple of gradients for each dimension
    gradients = torch.gradient(unified_field)
    
    # Extract spatial gradients (first 3 dimensions)
    gradient_flows = {
        'gradient_x': gradients[0],  # Gradient along X axis
        'gradient_y': gradients[1],  # Gradient along Y axis
        'gradient_z': gradients[2],  # Gradient along Z axis
        'gradient_scale': gradients[3] if len(gradients) > 3 else None,
        'gradient_time': gradients[4] if len(gradients) > 4 else None
    }
    
    return gradient_flows


def extract_local_gradients(gradient_flows: Dict[str, torch.Tensor], 
                          center_position: Tuple[int, int, int],
                          window_size: int = 3) -> Dict[str, torch.Tensor]:
    """
    Extract gradients from a local region around the center position.
    
    Args:
        gradient_flows: Dictionary of gradient tensors
        center_position: (x, y, z) center position
        window_size: Size of local region to extract
        
    Returns:
        Dictionary with aggregated gradients for each dimension
    """
    cx, cy, cz = center_position
    half_window = window_size // 2
    
    # Define the local region bounds
    x_slice = slice(max(0, cx - half_window), cx + half_window + 1)
    y_slice = slice(max(0, cy - half_window), cy + half_window + 1)
    z_slice = slice(max(0, cz - half_window), cz + half_window + 1)
    
    local_gradients = {}
    
    for key, gradient_tensor in gradient_flows.items():
        if gradient_tensor is not None and key.startswith('gradient_'):
            # Extract local region
            local_region = gradient_tensor[x_slice, y_slice, z_slice]
            
            # Aggregate across spatial dimensions, preserving others
            # This gives us the average gradient in the local region
            aggregated = torch.mean(local_region, dim=(0, 1, 2))
            
            local_gradients[key] = aggregated
    
    return local_gradients


def generate_motor_commands_from_gradients(local_gradients: Dict[str, torch.Tensor],
                                         gradient_following_strength: float = 1.0,
                                         total_dimensions: int = 37) -> Tuple[torch.Tensor, float]:
    """
    Generate motor commands from properly extracted gradients.
    
    Args:
        local_gradients: Dictionary of aggregated local gradients
        gradient_following_strength: Scaling factor for gradient following
        total_dimensions: Total field dimensions for normalization
        
    Returns:
        Tuple of (motor_commands, gradient_strength)
    """
    # Extract spatial gradients
    grad_x = local_gradients.get('gradient_x', torch.zeros(1))
    grad_y = local_gradients.get('gradient_y', torch.zeros(1))
    grad_z = local_gradients.get('gradient_z', torch.zeros(1))
    
    # For each gradient, aggregate across all non-spatial dimensions
    # This gives us the dominant gradient direction
    if grad_x.numel() > 1:
        # Weight by magnitude and take weighted average
        weights = torch.abs(grad_x)
        weights = weights / (torch.sum(weights) + 1e-8)
        x_component = torch.sum(grad_x * weights)
    else:
        x_component = grad_x.item() if grad_x.numel() == 1 else 0.0
    
    if grad_y.numel() > 1:
        weights = torch.abs(grad_y)
        weights = weights / (torch.sum(weights) + 1e-8)
        y_component = torch.sum(grad_y * weights)
    else:
        y_component = grad_y.item() if grad_y.numel() == 1 else 0.0
    
    if grad_z.numel() > 1:
        weights = torch.abs(grad_z)
        weights = weights / (torch.sum(weights) + 1e-8)
        z_component = torch.sum(grad_z * weights)
    else:
        z_component = grad_z.item() if grad_z.numel() == 1 else 0.0
    
    # Include scale and time gradients for the 4th motor dimension
    scale_component = 0.0
    time_component = 0.0
    
    if 'gradient_scale' in local_gradients and local_gradients['gradient_scale'] is not None:
        grad_scale = local_gradients['gradient_scale']
        if grad_scale.numel() > 1:
            weights = torch.abs(grad_scale)
            weights = weights / (torch.sum(weights) + 1e-8)
            scale_component = torch.sum(grad_scale * weights).item()
        else:
            scale_component = grad_scale.item() if grad_scale.numel() == 1 else 0.0
    
    if 'gradient_time' in local_gradients and local_gradients['gradient_time'] is not None:
        grad_time = local_gradients['gradient_time']
        if grad_time.numel() > 1:
            weights = torch.abs(grad_time)
            weights = weights / (torch.sum(weights) + 1e-8)
            time_component = torch.sum(grad_time * weights).item()
        else:
            time_component = grad_time.item() if grad_time.numel() == 1 else 0.0
    
    # Combine into motor commands
    motor_gradients = torch.tensor([
        x_component,
        y_component,
        z_component,
        (scale_component + time_component) * 0.5  # Combine scale/time for 4th dimension
    ], dtype=torch.float32)
    
    # Calculate gradient strength before scaling
    gradient_strength = torch.norm(motor_gradients).item()
    
    # Apply gradient following strength
    motor_commands = motor_gradients * gradient_following_strength
    
    # Normalize if too strong
    if torch.norm(motor_commands) > 2.0:
        motor_commands = motor_commands / torch.norm(motor_commands) * 2.0
    
    return motor_commands, gradient_strength


def improved_field_gradients_to_robot_action(brain_self) -> 'FieldNativeAction':
    """
    Drop-in replacement for the broken _field_gradients_to_robot_action method.
    
    This should be used to replace the method in UnifiedFieldBrain.
    
    Args:
        brain_self: The UnifiedFieldBrain instance (self)
        
    Returns:
        FieldNativeAction with proper motor commands
    """
    from brains.field.core_brain import FieldNativeAction
    import time
    import random
    
    # Calculate gradients properly
    gradient_flows = calculate_proper_gradient_flows(brain_self.unified_field)
    brain_self.gradient_flows = gradient_flows  # Update brain's gradient flows
    
    # Get center position
    center_x = brain_self.spatial_resolution // 2
    center_y = brain_self.spatial_resolution // 2
    center_z = brain_self.spatial_resolution // 2
    
    # Extract local gradients
    local_gradients = extract_local_gradients(
        gradient_flows, 
        (center_x, center_y, center_z),
        window_size=5  # Larger window for better averaging
    )
    
    # Generate motor commands
    motor_commands, gradient_strength = generate_motor_commands_from_gradients(
        local_gradients,
        brain_self.gradient_following_strength,
        brain_self.total_dimensions
    )
    
    # Apply prediction improvement modifier
    prediction_modifier = brain_self._get_prediction_improvement_addiction_modifier()
    motor_commands = motor_commands * prediction_modifier
    
    # Clamp to valid range
    motor_commands = torch.clamp(motor_commands, -1.0, 1.0)
    
    # Check if we need exploration fallback
    motor_magnitude = torch.norm(motor_commands).item()
    
    if gradient_strength < 1e-5 or motor_magnitude < 1e-6:
        # Weak gradients - apply exploration
        if not brain_self.quiet_mode and brain_self.brain_cycles % 50 == 0:
            print(f"   ⚠️  Weak gradients (strength={gradient_strength:.8f}), applying exploration")
        
        # Generate exploration action
        motor_commands = torch.tensor([
            0.2 * (random.random() - 0.5),
            0.2 * (random.random() - 0.5),
            0.1 * (random.random() - 0.5),
            0.1 * (random.random() - 0.5)
        ], dtype=torch.float32)
        action_confidence = 0.1
    else:
        # Good gradients - calculate confidence
        action_confidence = min(1.0, gradient_strength * 10)  # Scale gradient strength
        
        if not brain_self.quiet_mode and brain_self.brain_cycles % 100 == 0:
            print(f"   ✅ Gradient following: strength={gradient_strength:.6f}, "
                  f"motor_range=[{torch.min(motor_commands).item():.4f}, "
                  f"{torch.max(motor_commands).item():.4f}]")
    
    # Create action
    action = FieldNativeAction(
        timestamp=time.time(),
        field_gradients=motor_commands.clone(),  # Store the actual gradients
        motor_commands=motor_commands,
        action_confidence=action_confidence,
        gradient_strength=gradient_strength
    )
    
    brain_self.field_actions.append(action)
    brain_self.gradient_actions += 1
    
    return action


# Example of how to patch the brain:
def patch_unified_field_brain():
    """
    Patch the UnifiedFieldBrain class to use the fixed gradient extraction.
    
    This should be called after importing UnifiedFieldBrain but before
    creating instances.
    """
    from brains.field.core_brain import UnifiedFieldBrain
    
    # Replace the broken method with our fixed version
    UnifiedFieldBrain._field_gradients_to_robot_action = improved_field_gradients_to_robot_action
    UnifiedFieldBrain._calculate_gradient_flows = lambda self: None  # We calculate in action generation
    
    print("✅ UnifiedFieldBrain gradient extraction patched successfully")


if __name__ == "__main__":
    # Test the gradient extraction fix
    print("Testing gradient extraction fix...")
    
    # Create test field
    test_field = torch.randn(10, 10, 10, 5, 5)  # Simplified test field
    
    # Calculate gradients
    gradients = calculate_proper_gradient_flows(test_field)
    
    print(f"Gradient shapes:")
    for key, grad in gradients.items():
        if grad is not None:
            print(f"  {key}: {grad.shape}")
    
    # Test local extraction
    local_grads = extract_local_gradients(gradients, (5, 5, 5))
    
    print(f"\nLocal gradient shapes:")
    for key, grad in local_grads.items():
        if grad is not None:
            print(f"  {key}: {grad.shape}")
    
    # Test motor command generation
    motor_cmds, strength = generate_motor_commands_from_gradients(local_grads)
    
    print(f"\nMotor commands: {motor_cmds}")
    print(f"Gradient strength: {strength:.6f}")