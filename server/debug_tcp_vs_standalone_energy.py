#!/usr/bin/env python3
"""
Debug TCP vs Standalone Field Energy
Compare field energy calculation behavior between TCP server mode and standalone mode
"""

import sys
import os
import numpy as np
import torch
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.brain_factory import BrainFactory
from src.brains.field.tcp_adapter import FieldBrainTCPAdapter, FieldBrainConfig

def debug_standalone_mode():
    """Test field energy in standalone mode"""
    print("🔍 Testing STANDALONE mode...")
    
    # Clean memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    # Test pattern - the same pattern repeated
    base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
    
    field_energies = []
    field_sums = []
    field_abs_sums = []
    field_norms = []
    
    print("   Processing 10 cycles with same input...")
    for i in range(10):
        action, brain_state = brain.process_sensory_input(base_pattern)
        
        # Get field energy from brain state
        field_energy = brain_state.get('field_energy', 0.0)
        field_energies.append(field_energy)
        
        # Get actual field values for detailed analysis
        if hasattr(brain, 'field_brain_adapter') and brain.field_brain_adapter:
            field_brain = brain.field_brain_adapter.field_brain
            if hasattr(field_brain, 'field_impl') and field_brain.field_impl:
                unified_field = field_brain.field_impl.unified_field
                
                field_sum = torch.sum(unified_field).item()
                field_abs_sum = torch.sum(torch.abs(unified_field)).item()
                field_norm = torch.norm(unified_field).item()
                
                field_sums.append(field_sum)
                field_abs_sums.append(field_abs_sum)  
                field_norms.append(field_norm)
                
                if i < 3:  # Print first few
                    print(f"      Cycle {i}: energy={field_energy:.6f}, sum={field_sum:.6f}, abs_sum={field_abs_sum:.6f}, norm={field_norm:.6f}")
    
    # Calculate variance
    energy_variance = np.var(field_energies)
    sum_variance = np.var(field_sums) if field_sums else 0.0
    abs_sum_variance = np.var(field_abs_sums) if field_abs_sums else 0.0
    norm_variance = np.var(field_norms) if field_norms else 0.0
    
    print(f"   STANDALONE Results:")
    print(f"      Field energy variance: {energy_variance:.9f}")
    print(f"      Field sum variance: {sum_variance:.9f}")
    print(f"      Field abs_sum variance: {abs_sum_variance:.9f}")
    print(f"      Field norm variance: {norm_variance:.9f}")
    print(f"      Energy range: {min(field_energies):.6f} to {max(field_energies):.6f}")
    
    brain.finalize_session()
    return field_energies, energy_variance

def debug_tcp_adapter_mode():
    """Test field energy through TCP adapter (same as TCP server uses)"""
    print("\n🔍 Testing TCP ADAPTER mode...")
    
    # Clean memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    # Create TCP adapter directly (what the server uses internally)
    config = FieldBrainConfig(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True
    )
    tcp_adapter = FieldBrainTCPAdapter(config)
    
    # Test with the same pattern but simulate TCP dimension mismatch
    base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D
    
    # Simulate what happens when client sends 24D data but adapter expects 16D
    oversized_pattern = base_pattern + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 24D
    
    field_energies = []
    field_sums = []
    field_abs_sums = []
    field_norms = []
    
    print(f"   Processing 10 cycles with 24D→16D truncation...")
    for i in range(10):
        # Test both original pattern and oversized pattern
        test_pattern = oversized_pattern if i % 2 == 0 else base_pattern
        
        action, brain_state = tcp_adapter.process_sensory_input(test_pattern)
        
        # Get field energy from brain state
        field_energy = brain_state.get('field_energy', 0.0)
        field_energies.append(field_energy)
        
        # Get actual field values
        field_brain = tcp_adapter.field_brain
        if hasattr(field_brain, 'field_impl') and field_brain.field_impl:
            unified_field = field_brain.field_impl.unified_field
            
            field_sum = torch.sum(unified_field).item()
            field_abs_sum = torch.sum(torch.abs(unified_field)).item()
            field_norm = torch.norm(unified_field).item()
            
            field_sums.append(field_sum)
            field_abs_sums.append(field_abs_sum)
            field_norms.append(field_norm)
            
            if i < 3:  # Print first few  
                input_len = len(test_pattern)
                print(f"      Cycle {i}: input={input_len}D, energy={field_energy:.6f}, sum={field_sum:.6f}, abs_sum={field_abs_sum:.6f}, norm={field_norm:.6f}")
    
    # Calculate variance
    energy_variance = np.var(field_energies)
    sum_variance = np.var(field_sums)
    abs_sum_variance = np.var(field_abs_sums)
    norm_variance = np.var(field_norms)
    
    print(f"   TCP ADAPTER Results:")
    print(f"      Field energy variance: {energy_variance:.9f}")
    print(f"      Field sum variance: {sum_variance:.9f}")
    print(f"      Field abs_sum variance: {abs_sum_variance:.9f}")
    print(f"      Field norm variance: {norm_variance:.9f}")
    print(f"      Energy range: {min(field_energies):.6f} to {max(field_energies):.6f}")
    
    return field_energies, energy_variance

def debug_consistent_input_tcp_mode():
    """Test field energy with perfectly consistent input in TCP mode"""
    print("\n🔍 Testing TCP mode with PERFECTLY CONSISTENT input...")
    
    # Clean memory 
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    config = FieldBrainConfig(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True
    )
    tcp_adapter = FieldBrainTCPAdapter(config)
    
    # Use exactly the same pattern every time (no variation at all)
    fixed_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D exactly
    
    field_energies = []
    field_evolution_cycles = []
    
    print(f"   Processing 10 cycles with identical 16D input...")
    for i in range(10):
        action, brain_state = tcp_adapter.process_sensory_input(fixed_pattern)
        
        field_energy = brain_state.get('field_energy', 0.0)
        evolution_cycles = brain_state.get('field_evolution_cycles', 0)
        
        field_energies.append(field_energy)
        field_evolution_cycles.append(evolution_cycles)
        
        if i < 5:
            print(f"      Cycle {i}: energy={field_energy:.6f}, evolution_cycles={evolution_cycles}")
    
    energy_variance = np.var(field_energies)
    
    print(f"   CONSISTENT TCP Results:")
    print(f"      Field energy variance: {energy_variance:.9f}")
    print(f"      Energy range: {min(field_energies):.6f} to {max(field_energies):.6f}")
    print(f"      Evolution cycles: {min(field_evolution_cycles)} to {max(field_evolution_cycles)}")
    print(f"      Evolution progressed: {'Yes' if max(field_evolution_cycles) > min(field_evolution_cycles) else 'No'}")
    
    return field_energies, energy_variance

def main():
    """Run comprehensive comparison between TCP and standalone modes"""
    print("🔍 TCP vs Standalone Field Energy Comparison")
    print("=" * 80)
    
    # Test 1: Standalone mode
    standalone_energies, standalone_variance = debug_standalone_mode()
    
    # Test 2: TCP adapter mode (with dimension handling)
    tcp_energies, tcp_variance = debug_tcp_adapter_mode()
    
    # Test 3: TCP mode with perfectly consistent input
    consistent_energies, consistent_variance = debug_consistent_input_tcp_mode()
    
    print("\n" + "=" * 80)
    print("🔍 SUMMARY COMPARISON")
    print("=" * 80)
    print(f"STANDALONE mode variance:    {standalone_variance:.9f}")
    print(f"TCP ADAPTER mode variance:   {tcp_variance:.9f}")
    print(f"TCP CONSISTENT mode variance: {consistent_variance:.9f}")
    
    print(f"\nVariance comparison:")
    if standalone_variance > 0 and tcp_variance == 0:
        print("❌ PROBLEM: TCP adapter shows ZERO variance while standalone shows variance")
        print("   This confirms the core issue!")
    elif standalone_variance > 0 and tcp_variance > 0:
        ratio = tcp_variance / standalone_variance
        print(f"✅ Both show variance. TCP/Standalone ratio: {ratio:.3f}")
    elif standalone_variance == 0 and tcp_variance == 0:
        print("⚠️ Both show zero variance - may indicate field not evolving")
    
    print(f"\nField evolution comparison:")
    print(f"   Check field evolution cycles in the output above")
    print(f"   If cycles increase but variance is 0, the field is evolving")
    print(f"   but the energy calculation method cancels out changes")

if __name__ == "__main__":
    main()