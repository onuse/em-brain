#!/usr/bin/env python3
"""
Debug Energy Sum Cancellation Issue

Test whether the field energy calculation using torch.sum() shows zero variance
despite field evolution because positive/negative values cancel out.
"""

import sys
import os
import numpy as np
import torch
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter

def debug_energy_sum_cancellation():
    """Test the energy calculation issue with sum vs other methods"""
    print("🔍 Debugging Field Energy Sum Cancellation Issue")
    print("=" * 80)
    print("Testing whether torch.sum() cancels out field changes that should show variance")
    
    # Create TCP adapter
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Storage for different energy calculation methods
    energies_sum = []       # torch.sum() - current method (problematic)
    energies_abs_sum = []   # torch.sum(torch.abs()) - better
    energies_norm = []      # torch.norm() - best
    energies_mean_abs = []  # torch.mean(torch.abs()) - normalized
    
    evolution_cycles = []
    brain_state_energies = [] # What the brain reports
    
    print("\n📊 Processing 15 cycles with varied inputs...")
    for i in range(15):
        # Create varied input pattern
        base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
        varied_pattern = [x + 0.1 * np.sin(i * 0.3 + j * 0.5) for j, x in enumerate(base_pattern)]
        
        # Process through TCP adapter
        action, brain_state = adapter.process_sensory_input(varied_pattern)
        
        # Get field tensor
        field_brain = adapter.field_brain
        unified_field = field_brain.field_impl.unified_field
        
        # Calculate energy using different methods
        energy_sum = torch.sum(unified_field).item()
        energy_abs_sum = torch.sum(torch.abs(unified_field)).item()
        energy_norm = torch.norm(unified_field).item()
        energy_mean_abs = torch.mean(torch.abs(unified_field)).item()
        
        # Store results
        energies_sum.append(energy_sum)
        energies_abs_sum.append(energy_abs_sum)
        energies_norm.append(energy_norm)
        energies_mean_abs.append(energy_mean_abs)
        
        # Track evolution and brain state
        cycles = brain_state.get('field_evolution_cycles', 0)
        brain_energy = brain_state.get('field_energy', 0.0)
        
        evolution_cycles.append(cycles)
        brain_state_energies.append(brain_energy)
        
        # Print every 3rd cycle for monitoring
        if i % 3 == 0:
            print(f"   Cycle {i:2d}: evolution={cycles}, sum={energy_sum:.6f}, abs_sum={energy_abs_sum:.6f}, norm={energy_norm:.6f}")
    
    # Calculate variances
    variance_sum = np.var(energies_sum)
    variance_abs_sum = np.var(energies_abs_sum)
    variance_norm = np.var(energies_norm)
    variance_mean_abs = np.var(energies_mean_abs)
    variance_brain_state = np.var(brain_state_energies)
    
    # Calculate ranges
    range_sum = max(energies_sum) - min(energies_sum)
    range_abs_sum = max(energies_abs_sum) - min(energies_abs_sum)
    range_norm = max(energies_norm) - min(energies_norm)
    range_evolution = max(evolution_cycles) - min(evolution_cycles)
    
    print("\n" + "=" * 80)
    print("🔍 ENERGY CALCULATION COMPARISON")
    print("=" * 80)
    
    print(f"Field Evolution:")
    print(f"   Evolution cycles progressed: {min(evolution_cycles)} → {max(evolution_cycles)} (Δ{range_evolution})")
    print(f"   Evolution variance: {np.var(evolution_cycles):.6f}")
    
    print(f"\nEnergy Calculation Methods:")
    print(f"   torch.sum() variance:           {variance_sum:.9f}")
    print(f"   torch.sum(abs()) variance:      {variance_abs_sum:.9f}")
    print(f"   torch.norm() variance:          {variance_norm:.9f}")
    print(f"   torch.mean(abs()) variance:     {variance_mean_abs:.9f}")
    print(f"   Brain state energy variance:    {variance_brain_state:.9f}")
    
    print(f"\nEnergy Ranges:")
    print(f"   torch.sum() range:              {range_sum:.6f}")
    print(f"   torch.sum(abs()) range:         {range_abs_sum:.6f}")
    print(f"   torch.norm() range:             {range_norm:.6f}")
    
    print(f"\nEnergy Values:")
    print(f"   torch.sum() values:             {min(energies_sum):.6f} to {max(energies_sum):.6f}")
    print(f"   torch.sum(abs()) values:        {min(energies_abs_sum):.6f} to {max(energies_abs_sum):.6f}")
    print(f"   torch.norm() values:            {min(energies_norm):.6f} to {max(energies_norm):.6f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("🔬 ANALYSIS")
    print("=" * 80)
    
    field_evolved = range_evolution > 0
    sum_shows_variance = variance_sum > 0.000001
    abs_sum_shows_variance = variance_abs_sum > 0.000001
    norm_shows_variance = variance_norm > 0.000001
    
    print(f"Field evolution occurred: {'✅ YES' if field_evolved else '❌ NO'}")
    print(f"torch.sum() shows variance: {'✅ YES' if sum_shows_variance else '❌ NO'}")
    print(f"torch.sum(abs()) shows variance: {'✅ YES' if abs_sum_shows_variance else '❌ NO'}")
    print(f"torch.norm() shows variance: {'✅ YES' if norm_shows_variance else '❌ NO'}")
    
    if field_evolved and not sum_shows_variance:
        print(f"\n🚨 PROBLEM IDENTIFIED:")
        print(f"   Field IS evolving (evolution cycles: {min(evolution_cycles)} → {max(evolution_cycles)})")
        print(f"   But torch.sum() shows {variance_sum:.9f} variance (nearly zero)")
        print(f"   This suggests positive/negative field values are canceling out!")
        
        # Check for cancellation evidence
        sum_values = energies_sum
        positive_changes = sum([1 for i in range(1, len(sum_values)) if sum_values[i] > sum_values[i-1]])
        negative_changes = sum([1 for i in range(1, len(sum_values)) if sum_values[i] < sum_values[i-1]])
        
        print(f"\n   Evidence of cancellation:")
        print(f"   Sum values oscillate: {positive_changes} increases, {negative_changes} decreases")
        print(f"   Sum mean: {np.mean(sum_values):.6f} (close to zero suggests cancellation)")
        
        print(f"\n💡 SOLUTION:")
        print(f"   Use torch.sum(torch.abs(field)) instead of torch.sum(field)")
        print(f"   Or use torch.norm(field) for energy calculation")
        print(f"   These methods show proper variance: {variance_abs_sum:.9f} and {variance_norm:.9f}")
        
        return True  # Problem confirmed
    
    elif field_evolved and sum_shows_variance:
        print(f"\n✅ NO PROBLEM:")
        print(f"   Field evolves and torch.sum() shows proper variance")
        return False
    
    elif not field_evolved:
        print(f"\n⚠️ DIFFERENT ISSUE:")
        print(f"   Field is not evolving - this is a different problem")
        return False
    
    return False

def test_simple_cancellation_demo():
    """Simple demonstration of the cancellation issue"""
    print("\n\n🎯 Simple Cancellation Demonstration")
    print("=" * 50)
    
    # Create two tensors that have different patterns but same sum
    tensor1 = torch.tensor([1.0, 2.0, -1.0, -2.0])  # sum = 0
    tensor2 = torch.tensor([1.5, 1.5, -1.5, -1.5])  # sum = 0, but different pattern
    tensor3 = torch.tensor([2.0, 1.0, -2.0, -1.0])  # sum = 0, but different again
    
    tensors = [tensor1, tensor2, tensor3]
    
    sum_values = []
    abs_sum_values = []
    norm_values = []
    
    for i, tensor in enumerate(tensors):
        sum_val = torch.sum(tensor).item()
        abs_sum_val = torch.sum(torch.abs(tensor)).item()
        norm_val = torch.norm(tensor).item()
        
        sum_values.append(sum_val)
        abs_sum_values.append(abs_sum_val)
        norm_values.append(norm_val)
        
        print(f"   Tensor {i+1}: {tensor.tolist()}")
        print(f"      sum={sum_val:.3f}, abs_sum={abs_sum_val:.3f}, norm={norm_val:.3f}")
    
    print(f"\n   Variance comparison:")
    print(f"      sum variance:     {np.var(sum_values):.6f}")
    print(f"      abs_sum variance: {np.var(abs_sum_values):.6f}")
    print(f"      norm variance:    {np.var(norm_values):.6f}")
    
    print(f"\n   💡 This shows why torch.sum() can hide real field changes!")

def main():
    """Run energy sum cancellation analysis"""
    
    # Test 1: Real field evolution with different energy calculations
    problem_found = debug_energy_sum_cancellation()
    
    # Test 2: Simple demonstration of the principle
    test_simple_cancellation_demo()
    
    print(f"\n" + "=" * 80)
    print(f"🎯 CONCLUSION")
    print(f"=" * 80)
    if problem_found:
        print(f"✅ ROOT CAUSE IDENTIFIED:")
        print(f"   Field IS evolving, but torch.sum() energy calculation")
        print(f"   shows zero variance due to positive/negative cancellation")
        print(f"   Solution: Use torch.sum(torch.abs()) or torch.norm()")
    else:
        print(f"⚠️ Issue not reproduced in this test")
        print(f"   Field energy calculation may work correctly")

if __name__ == "__main__":
    main()