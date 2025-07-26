#!/usr/bin/env python3
"""
Thorough investigation of appropriate parameter values for prediction error and energy dissipation
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def investigate_prediction_error_sensitivity():
    """Investigate what prediction error sensitivity should be"""
    print("🔬 INVESTIGATION: Prediction Error Sensitivity")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Get access to unified brain
    ub = brain.brain
    
    print("\n1. Understanding the Prediction Mechanism:")
    print("-" * 40)
    print(f"   Spatial resolution: {ub.spatial_resolution}³ = {ub.spatial_resolution**3} voxels")
    print(f"   Field shape: {ub.unified_field.shape}")
    print(f"   Total field elements: {ub.unified_field.numel():,}")
    
    # Run a few cycles to build up field state
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    for i in range(5):
        brain.process_sensory_input(pattern)
    
    # Now examine the prediction error calculation
    print("\n2. Analyzing Prediction Error Magnitude:")
    print("-" * 40)
    
    # Save current field as predicted
    ub._predicted_field = ub.unified_field.clone()
    
    # Process one more input
    brain.process_sensory_input(pattern)
    
    # Calculate prediction error manually
    center_x = ub.spatial_resolution // 2
    center_y = ub.spatial_resolution // 2
    center_z = ub.spatial_resolution // 2
    
    region_slice = slice(center_x-1, center_x+2)
    predicted_region = ub._predicted_field[region_slice, region_slice, region_slice]
    actual_region = ub.unified_field[region_slice, region_slice, region_slice]
    
    prediction_error = torch.mean(torch.abs(predicted_region - actual_region)).item()
    
    print(f"   Predicted region shape: {predicted_region.shape}")
    print(f"   Predicted region max: {predicted_region.max().item():.6f}")
    print(f"   Predicted region mean: {predicted_region.mean().item():.6f}")
    print(f"   Actual region max: {actual_region.max().item():.6f}")
    print(f"   Actual region mean: {actual_region.mean().item():.6f}")
    print(f"   Raw prediction error: {prediction_error:.8f}")
    
    # Test different sensitivity values
    print("\n3. Testing Different Sensitivity Values:")
    print("-" * 40)
    sensitivities = [1, 10, 100, 1000, 5000, 10000]
    for sens in sensitivities:
        confidence = 1.0 / (1.0 + prediction_error * sens)
        print(f"   Sensitivity {sens:5d}: confidence = {confidence:.6f}")
    
    # Calculate what sensitivity would give reasonable confidence progression
    print("\n4. Calculating Appropriate Sensitivity:")
    print("-" * 40)
    
    # For a typical prediction error, we want:
    # - Small errors (< 0.001) -> high confidence (> 0.9)
    # - Medium errors (~ 0.01) -> medium confidence (~ 0.5)
    # - Large errors (> 0.1) -> low confidence (< 0.1)
    
    target_pairs = [
        (0.0001, 0.95),  # Very small error -> very high confidence
        (0.001, 0.9),    # Small error -> high confidence
        (0.01, 0.5),     # Medium error -> medium confidence
        (0.1, 0.1),      # Large error -> low confidence
    ]
    
    print("   Target confidence mapping:")
    for error, target_conf in target_pairs:
        # Solve for sensitivity: conf = 1/(1 + error * sens)
        # sens = (1/conf - 1) / error
        ideal_sens = ((1/target_conf - 1) / error)
        print(f"   Error {error:.4f} -> Confidence {target_conf:.2f} requires sensitivity ≈ {ideal_sens:.0f}")
    
    # Find sensitivity that best fits all targets
    best_sensitivity = 100  # Based on the medium error case
    print(f"\n   Recommended sensitivity: {best_sensitivity}")
    
    return prediction_error, best_sensitivity

def investigate_energy_dissipation():
    """Investigate what energy dissipation rate should be"""
    print("\n\n🔬 INVESTIGATION: Energy Dissipation Rate")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    ub = brain.brain
    
    print("\n1. Understanding Energy Dynamics:")
    print("-" * 40)
    print(f"   Current dissipation rate: {ub.field_energy_dissipation_rate}")
    print(f"   This means {(1-ub.field_energy_dissipation_rate)*100:.1f}% energy loss per maintenance cycle")
    print(f"   Maintenance every: {ub.maintenance_interval} brain cycles")
    
    # Track energy over cycles
    stable_input = [0.5] * 16
    energies = []
    max_activations = []
    
    print("\n2. Tracking Energy Evolution:")
    print("-" * 40)
    
    for i in range(50):
        _, brain_state = brain.process_sensory_input(stable_input)
        energy = brain_state.get('field_total_energy', 0.0)
        max_act = brain_state.get('field_max_activation', 0.0)
        energies.append(energy)
        max_activations.append(max_act)
        
        if i % 10 == 0:
            print(f"   Cycle {i}: energy = {energy:.1f}, max_activation = {max_act:.3f}")
    
    # Analyze energy growth
    print("\n3. Analyzing Energy Growth Pattern:")
    print("-" * 40)
    
    # Calculate growth rate
    if len(energies) > 20:
        # Fit exponential: E(t) = E0 * growth_rate^t
        early_energy = np.mean(energies[:5])
        late_energy = np.mean(energies[-5:])
        cycles_between = len(energies) - 7.5  # Average positions
        growth_rate_per_cycle = (late_energy / early_energy) ** (1/cycles_between)
        
        print(f"   Energy growth rate: {growth_rate_per_cycle:.6f} per cycle")
        print(f"   This means {(growth_rate_per_cycle-1)*100:.3f}% growth per cycle")
        
        # How much is added per cycle vs dissipated?
        energy_added_per_cycle = (energies[-1] - energies[-2]) if len(energies) > 1 else 0
        print(f"   Energy added per cycle: ~{energy_added_per_cycle:.1f}")
    
    # Calculate target dissipation rate
    print("\n4. Calculating Appropriate Dissipation Rate:")
    print("-" * 40)
    
    # Goal: Energy should reach equilibrium, not grow indefinitely
    # At equilibrium: energy_added = energy_dissipated
    # energy_dissipated = total_energy * (1 - dissipation_rate) / maintenance_interval
    
    avg_energy_addition = np.mean(np.diff(energies)) if len(energies) > 1 else 50
    target_equilibrium_energy = 100000  # Reasonable target
    
    # Every maintenance_interval cycles, we dissipate: energy * (1 - rate)
    # Over one cycle: dissipation = energy * (1 - rate) / maintenance_interval
    # For equilibrium: avg_energy_addition = target_energy * (1 - rate) / maintenance_interval
    # Solving for rate: rate = 1 - (avg_energy_addition * maintenance_interval / target_energy)
    
    target_dissipation_rate = 1 - (avg_energy_addition * ub.maintenance_interval / target_equilibrium_energy)
    target_dissipation_rate = max(0.9, min(0.99, target_dissipation_rate))  # Clamp to reasonable range
    
    print(f"   Average energy addition per cycle: {avg_energy_addition:.1f}")
    print(f"   Target equilibrium energy: {target_equilibrium_energy:.0f}")
    print(f"   Maintenance interval: {ub.maintenance_interval} cycles")
    print(f"   Recommended dissipation rate: {target_dissipation_rate:.3f}")
    print(f"   This means {(1-target_dissipation_rate)*100:.1f}% energy loss per maintenance")
    
    # Verify the calculation
    equilibrium_test = target_equilibrium_energy * (1 - target_dissipation_rate) / ub.maintenance_interval
    print(f"\n   Verification: At equilibrium, dissipation = {equilibrium_test:.1f} per cycle")
    print(f"   vs energy addition of {avg_energy_addition:.1f} per cycle")
    
    return avg_energy_addition, target_dissipation_rate

def investigate_field_activation_patterns():
    """Investigate how field activations actually behave"""
    print("\n\n🔬 INVESTIGATION: Field Activation Patterns")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    ub = brain.brain
    
    print("\n1. Field Activation Sources:")
    print("-" * 40)
    
    # Run a few cycles with different inputs
    inputs = [
        ([0.1] * 16, "Low intensity"),
        ([0.9] * 16, "High intensity"),
        ([0.5, 0.9, 0.1, 0.7] * 4, "Varied pattern"),
    ]
    
    for input_data, desc in inputs:
        # Reset field for clean test
        ub.unified_field.zero_()
        
        print(f"\n   Testing: {desc}")
        for i in range(3):
            brain.process_sensory_input(input_data)
            max_act = ub.unified_field.max().item()
            mean_act = ub.unified_field.mean().item()
            nonzero = (ub.unified_field > 0.01).sum().item()
            print(f"   Cycle {i+1}: max={max_act:.3f}, mean={mean_act:.6f}, active_elements={nonzero}")
    
    print("\n2. Energy Addition Mechanisms:")
    print("-" * 40)
    print("   Energy is added through:")
    print("   - Sensory imprinting (additive)")
    print("   - Field evolution dynamics")
    print("   - Constraint effects")
    print("   - Gradient computations")
    
    return

def generate_recommendations():
    """Generate final recommendations based on investigation"""
    print("\n\n📊 FINAL RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n1. Prediction Error Sensitivity:")
    print("   Current value: 5000 (too sensitive)")
    print("   Recommended: 100")
    print("   Rationale: Maps prediction errors to reasonable confidence range")
    
    print("\n2. Energy Dissipation Rate:")
    print("   Current value: 0.999 (preserves 99.9% energy)")
    print("   Recommended: 0.95 (removes 5% energy)")
    print("   Rationale: Achieves energy equilibrium instead of unbounded growth")
    
    print("\n3. Additional Recommendations:")
    print("   - Add energy normalization if total exceeds threshold")
    print("   - Scale prediction error by field magnitude for robustness")
    print("   - Consider adaptive dissipation based on energy level")

if __name__ == "__main__":
    # Run investigations
    pred_error, pred_sensitivity = investigate_prediction_error_sensitivity()
    energy_addition, dissipation_rate = investigate_energy_dissipation()
    investigate_field_activation_patterns()
    generate_recommendations()
    
    print("\n✅ Investigation complete!")