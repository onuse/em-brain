#!/usr/bin/env python3
"""
Debug Field Energy Calculation
Investigate why field energy has zero variance in TCP server vs standalone
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_factory import BrainFactory

def debug_field_energy():
    """Debug field energy calculation methods"""
    print("🔍 Debugging field energy calculation methods...")
    
    # Clear memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
    
    field_energies_sum = []
    field_energies_abs_sum = []
    field_energies_norm = []
    
    for i in range(10):
        action, brain_state = brain.process_sensory_input(pattern)
        
        # Get the actual field tensor
        field_brain = brain.field_brain_adapter.field_brain
        unified_field = field_brain.field_impl.unified_field
        
        # Different energy calculations
        energy_sum = torch.sum(unified_field).item()  # Current method (can cancel out)
        energy_abs_sum = torch.sum(torch.abs(unified_field)).item()  # Better method
        energy_norm = torch.norm(unified_field).item()  # Even better method
        
        field_energies_sum.append(energy_sum)
        field_energies_abs_sum.append(energy_abs_sum)
        field_energies_norm.append(energy_norm)
        
        if i % 3 == 0:
            print(f"   Cycle {i}:")
            print(f"      Sum method: {energy_sum:.6f}")
            print(f"      Abs sum method: {energy_abs_sum:.6f}")
            print(f"      Norm method: {energy_norm:.6f}")
            print(f"      Brain state energy: {brain_state.get('field_energy', 0.0):.6f}")
    
    print(f"\n📊 Energy Method Comparison:")
    print(f"   Sum method variance: {np.var(field_energies_sum):.6f}")
    print(f"   Abs sum variance: {np.var(field_energies_abs_sum):.6f}")  
    print(f"   Norm variance: {np.var(field_energies_norm):.6f}")
    
    print(f"\n   Sum method range: {min(field_energies_sum):.6f} to {max(field_energies_sum):.6f}")
    print(f"   Abs sum range: {min(field_energies_abs_sum):.6f} to {max(field_energies_abs_sum):.6f}")
    print(f"   Norm range: {min(field_energies_norm):.6f} to {max(field_energies_norm):.6f}")
    
    brain.finalize_session()

if __name__ == "__main__":
    debug_field_energy()