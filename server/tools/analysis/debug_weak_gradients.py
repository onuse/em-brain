#!/usr/bin/env python3
"""
Debug Weak Gradient Investigation

This tool investigates why the field brain generates extremely weak gradients
that result in zero actions after tanh normalization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
import math

# Import the field brain
sys.path.append('/Users/jkarlsson/Documents/Projects/robot-project/brain/server/src')
from brains.field.core_brain import UnifiedFieldBrain

def analyze_gradient_weakness():
    """Analyze why gradients are so weak they produce zero actions."""
    
    print("🔍 Debug: Weak Gradient Analysis")
    print("=" * 60)
    
    # Create brain with different parameters to isolate the issue
    brain = UnifiedFieldBrain(
        spatial_resolution=10,  # Smaller for easier debugging
        temporal_window=5.0,
        field_evolution_rate=0.1,
        constraint_discovery_rate=0.15,
        quiet_mode=True
    )
    
    # Initial field state
    print(f"\n1️⃣ Initial Field State:")
    print(f"   Unified field shape: {brain.unified_field.shape}")
    print(f"   Field max: {torch.max(brain.unified_field).item():.6f}")
    print(f"   Field mean: {torch.mean(brain.unified_field).item():.6f}")
    print(f"   Field sum: {torch.sum(brain.unified_field).item():.6f}")
    print(f"   Field non-zero count: {torch.count_nonzero(brain.unified_field).item()}")
    
    # Process a few cycles and examine what happens
    print(f"\n2️⃣ Processing Robot Cycles:")
    
    for cycle in range(5):
        # Create test sensory input
        sensory_input = [
            0.5 + 0.2 * cycle,  # x position - significant change
            0.3 + 0.15 * cycle, # y position - significant change
            0.1,                # z position
            0.8 + 0.1 * math.sin(cycle), # distance sensors with variation
            0.6 + 0.1 * math.cos(cycle),
            0.4,
            *[0.5 + 0.1 * math.sin(cycle + i) for i in range(18)]  # 18 more varied sensors
        ]
        
        # Process cycle and examine intermediate states
        print(f"\n   Cycle {cycle + 1}:")
        
        # 1. Check field experience mapping
        field_experience = brain._robot_sensors_to_field_experience(sensory_input)
        print(f"      Field coords range: [{torch.min(field_experience.field_coordinates).item():.3f}, {torch.max(field_experience.field_coordinates).item():.3f}]")
        print(f"      Field intensity: {field_experience.field_intensity:.6f}")
        
        # 2. Apply experience and check field changes
        old_field_max = torch.max(brain.unified_field).item()
        brain._apply_field_experience(field_experience)
        new_field_max = torch.max(brain.unified_field).item()
        print(f"      Field change: {old_field_max:.6f} → {new_field_max:.6f} (Δ={new_field_max-old_field_max:.6f})")
        
        # 3. Evolve field and check evolution effects
        old_field_sum = torch.sum(brain.unified_field).item()
        brain._evolve_unified_field()
        new_field_sum = torch.sum(brain.unified_field).item()
        print(f"      Field evolution: sum {old_field_sum:.3f} → {new_field_sum:.3f} (decay effect: {new_field_sum/old_field_sum:.6f})")
        
        # 4. Examine gradient calculation in detail
        brain._calculate_gradient_flows()
        
        # Check gradient magnitudes
        total_grad_magnitude = 0.0
        for grad_name, grad_tensor in brain.gradient_flows.items():
            grad_magnitude = torch.norm(grad_tensor).item()
            grad_max = torch.max(torch.abs(grad_tensor)).item()
            grad_mean = torch.mean(torch.abs(grad_tensor)).item()
            print(f"      {grad_name}: norm={grad_magnitude:.8f}, max={grad_max:.8f}, mean={grad_mean:.8f}")
            total_grad_magnitude += grad_magnitude
        
        print(f"      Total gradient magnitude: {total_grad_magnitude:.8f}")
        
        # 5. Check motor gradient extraction
        center_idx = brain.spatial_resolution // 2
        motor_gradients = torch.zeros(4)
        
        if 'gradient_x' in brain.gradient_flows:
            grad_x = brain.gradient_flows['gradient_x'][center_idx, center_idx, center_idx, :, :]
            motor_gradients[0] = torch.mean(grad_x).item()
            print(f"      Motor gradient X: slice_shape={grad_x.shape}, mean={motor_gradients[0]:.8f}")
        
        if 'gradient_y' in brain.gradient_flows:
            grad_y = brain.gradient_flows['gradient_y'][center_idx, center_idx, center_idx, :, :]
            motor_gradients[1] = torch.mean(grad_y).item()
            print(f"      Motor gradient Y: slice_shape={grad_y.shape}, mean={motor_gradients[1]:.8f}")
        
        field_momentum = torch.mean(brain.unified_field).item()
        motor_gradients[3] = field_momentum
        print(f"      Field momentum: {field_momentum:.8f}")
        
        # 6. Apply gradient following strength and examine final motor commands
        motor_commands_before = motor_gradients * brain.gradient_following_strength
        motor_commands_after = torch.clamp(motor_commands_before, -1.0, 1.0)
        
        print(f"      Motor gradients: {[f'{x:.8f}' for x in motor_gradients.tolist()]}")
        print(f"      * strength ({brain.gradient_following_strength}): {[f'{x:.8f}' for x in motor_commands_before.tolist()]}")
        print(f"      After clamp: {[f'{x:.8f}' for x in motor_commands_after.tolist()]}")
        
        # Process complete cycle
        final_motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        print(f"      Final motor output: {[f'{x:.8f}' for x in final_motor_output]}")
        print(f"      Action confidence: {brain_state.get('last_action_confidence', 0):.8f}")
        print(f"      Gradient strength: {brain_state.get('last_gradient_strength', 0):.8f}")

def analyze_field_parameters():
    """Analyze the field parameters that might cause weak gradients."""
    
    print(f"\n3️⃣ Field Parameter Analysis:")
    
    brain = UnifiedFieldBrain(spatial_resolution=10, quiet_mode=True)
    
    # Key parameters affecting gradient strength
    print(f"   Field decay rate: {brain.field_decay_rate} (applied each cycle)")
    print(f"   Field diffusion rate: {brain.field_diffusion_rate}")
    print(f"   Gradient following strength: {brain.gradient_following_strength}")
    print(f"   Topology stability threshold: {brain.topology_stability_threshold}")
    
    # Calculate cumulative decay effect
    cycles = 100
    cumulative_decay = brain.field_decay_rate ** cycles
    print(f"   After {cycles} cycles, field decays by factor: {cumulative_decay:.6f}")
    
    # Field imprint analysis
    print(f"\n   Field Imprint Parameters:")
    print(f"   Spatial spread in imprint: 2.0")
    print(f"   Gaussian weight formula: intensity * exp(-(distance^2) / (2 * 2.0^2))")
    
    # Show how small gradients get filtered out
    test_gradients = [0.001, 0.0001, 0.00001, 0.000001]
    print(f"\n   Gradient Scaling Effect:")
    for grad in test_gradients:
        scaled = grad * brain.gradient_following_strength
        clamped = max(-1.0, min(1.0, scaled))
        print(f"   {grad:.6f} * {brain.gradient_following_strength} = {scaled:.6f} → clamped: {clamped:.6f}")

def test_gradient_amplification():
    """Test what happens with amplified parameters."""
    
    print(f"\n4️⃣ Gradient Amplification Test:")
    
    # Test with amplified parameters
    brain = UnifiedFieldBrain(
        spatial_resolution=10,
        field_evolution_rate=0.3,      # Higher evolution
        constraint_discovery_rate=0.3,  # Higher constraints
        quiet_mode=True
    )
    
    # Increase key parameters
    brain.field_decay_rate = 0.999      # Much slower decay
    brain.gradient_following_strength = 1.0  # Maximum strength
    
    print(f"   Modified parameters:")
    print(f"   Field decay rate: {brain.field_decay_rate} (slower decay)")
    print(f"   Gradient following strength: {brain.gradient_following_strength} (maximum)")
    
    # Process with stronger inputs
    for cycle in range(3):
        # Create stronger sensory input
        sensory_input = [
            1.0 if i % 2 == 0 else -1.0 for i in range(24)  # Extreme alternating values
        ]
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        print(f"   Cycle {cycle + 1}: motor={[f'{x:.6f}' for x in motor_output]}, "
              f"field_energy={brain_state['field_total_energy']:.6f}")

def diagnose_root_cause():
    """Diagnose the root cause of weak gradients."""
    
    print(f"\n5️⃣ Root Cause Analysis:")
    
    brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)  # Very small for detailed analysis
    
    # Manually inject a strong field value to check gradient calculation
    print(f"   Injecting strong field values...")
    center = brain.spatial_resolution // 2
    
    # Create a strong gradient by setting adjacent cells to different values
    brain.unified_field[center-1, center, center, 0, 0] = -1.0  # Left side
    brain.unified_field[center+1, center, center, 0, 0] = 1.0   # Right side
    
    print(f"   Field values around center:")
    print(f"   Left [{center-1},{center},{center}]: {brain.unified_field[center-1, center, center, 0, 0].item():.3f}")
    print(f"   Center [{center},{center},{center}]: {brain.unified_field[center, center, center, 0, 0].item():.3f}")
    print(f"   Right [{center+1},{center},{center}]: {brain.unified_field[center+1, center, center, 0, 0].item():.3f}")
    
    # Calculate gradients manually
    brain._calculate_gradient_flows()
    
    # Check what gradient this produces
    if 'gradient_x' in brain.gradient_flows:
        grad_x_at_center = brain.gradient_flows['gradient_x'][center, center, center, 0, 0].item()
        print(f"   Calculated X gradient at center: {grad_x_at_center:.6f}")
        
        # Expected gradient: (1.0 - (-1.0)) / 2.0 = 1.0
        expected = (1.0 - (-1.0)) / 2.0
        print(f"   Expected X gradient: {expected:.6f}")
        print(f"   Match: {'YES' if abs(grad_x_at_center - expected) < 0.001 else 'NO'}")
    
    # Test motor extraction from this gradient
    motor_gradients = torch.zeros(4)
    if 'gradient_x' in brain.gradient_flows:
        grad_x = brain.gradient_flows['gradient_x'][center, center, center, :, :]
        motor_gradients[0] = torch.mean(grad_x).item()
        print(f"   Motor gradient from slice: {motor_gradients[0]:.6f}")
        print(f"   Slice shape: {grad_x.shape}, slice values: {grad_x.flatten().tolist()}")
    
    # Final motor command
    final_motor = motor_gradients[0] * brain.gradient_following_strength
    print(f"   Final motor command: {motor_gradients[0]:.6f} * {brain.gradient_following_strength} = {final_motor:.6f}")

if __name__ == "__main__":
    print("🧪 Debugging Weak Gradient Generation in Field Brain")
    print("=" * 70)
    
    try:
        analyze_gradient_weakness()
        analyze_field_parameters()
        test_gradient_amplification()
        diagnose_root_cause()
        
        print(f"\n✅ Weak Gradient Analysis Complete!")
        print(f"\n🔍 Key Findings Expected:")
        print(f"   • Field decay rate too aggressive (0.995^n quickly → 0)")
        print(f"   • Field imprints too weak or too localized")
        print(f"   • Gradient calculation averaging across too many dimensions")
        print(f"   • Gradient following strength too low (0.3)")
        print(f"   • Field evolution parameters insufficient to maintain activation")
        
    except Exception as e:
        print(f"❌ Error during gradient analysis: {e}")
        import traceback
        traceback.print_exc()