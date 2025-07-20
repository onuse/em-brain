#!/usr/bin/env python3
"""
Simple NaN Bug Test

Focused test to reproduce the specific NaN bug in enhanced dynamics
variance calculation without complex experience setup.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'src'))

def test_variance_calculation_bug():
    """Test the exact variance calculation that causes NaN."""
    print("🔬 Testing Enhanced Dynamics Variance Calculation Bug")
    print("=" * 60)
    
    # Simulate the conditions that cause the bug
    test_cases = [
        ("All zeros", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Very small values", [1e-10, 1e-11, 1e-10, 1e-12, 1e-10]),
        ("Mixed zeros and small", [0.0, 1e-8, 0.0, 1e-9, 0.0]),
        ("Normal small values", [0.1, 0.05, 0.08, 0.02, 0.06]),
        ("Large values", [1000.0, 999.5, 1001.2, 998.8, 1000.7])
    ]
    
    print("\n🧪 Testing different energy history scenarios:")
    
    for name, recent_energy in test_cases:
        print(f"\n--- {name} ---")
        print(f"Energy history: {recent_energy}")
        
        try:
            # This is the exact calculation from enhanced_dynamics.py line 150
            energy_variance = torch.var(torch.tensor(recent_energy)).item()
            
            print(f"Variance result: {energy_variance}")
            
            # Check for problematic values
            if np.isnan(energy_variance):
                print("🚨 NaN detected!")
            elif np.isinf(energy_variance):
                print("⚠️ Infinity detected!")
            elif energy_variance == 0.0:
                print("ℹ️ Zero variance (expected for uniform values)")
            else:
                print("✅ Normal variance value")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
    
    return True

def test_enhanced_dynamics_simulation():
    """Simulate the enhanced dynamics phase detection logic."""
    print(f"\n🔧 Testing Enhanced Dynamics Phase Detection Logic")
    print("=" * 60)
    
    # Simulate the phase energy history building up over time
    phase_energy_history = []
    
    print(f"\nSimulating energy accumulation over cycles:")
    
    for cycle in range(10):
        # Simulate field energy (starts at 0, gradually increases)
        if cycle == 0:
            field_energy = 0.0
        elif cycle < 3:
            field_energy = 0.0  # Still zero for first few cycles
        else:
            field_energy = (cycle - 2) * 0.1  # Gradual increase
        
        phase_energy_history.append(field_energy)
        
        print(f"Cycle {cycle}: field_energy={field_energy:.6f}")
        
        # Test variance calculation once we have 5+ values
        if len(phase_energy_history) >= 5:
            recent_energy = phase_energy_history[-5:]
            
            print(f"  Recent energy (last 5): {recent_energy}")
            
            try:
                # The problematic calculation
                energy_variance = torch.var(torch.tensor(recent_energy)).item()
                
                print(f"  Energy variance: {energy_variance:.10f}")
                
                # Check the conditions that use this variance
                stability_threshold = 0.3
                energy_threshold = 1.0
                
                if np.isnan(energy_variance) or np.isinf(energy_variance):
                    print(f"  🚨 NaN/Inf variance would break phase detection!")
                    return False
                
                # Check phase detection logic
                if field_energy > energy_threshold:
                    if energy_variance < stability_threshold:
                        phase = "high_energy"
                    else:
                        phase = "chaotic"
                elif field_energy < 0.1:
                    phase = "low_energy"
                else:
                    if energy_variance < stability_threshold:
                        phase = "stable"
                    else:
                        phase = "transitional"
                
                print(f"  Detected phase: {phase}")
                
            except Exception as e:
                print(f"  💥 Variance calculation failed: {e}")
                return False
    
    print(f"\n✅ Enhanced dynamics simulation completed successfully")
    return True

def test_torch_variance_edge_cases():
    """Test PyTorch variance function with edge cases."""
    print(f"\n🧮 Testing PyTorch Variance Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        ("Empty tensor", []),
        ("Single value", [1.0]),
        ("Two identical", [1.0, 1.0]),
        ("All zeros", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Tiny values", [1e-15, 1e-15, 1e-15, 1e-15, 1e-15]),
        ("Mixed tiny", [1e-15, 1e-14, 1e-15, 1e-16, 1e-15]),
        ("NaN input", [1.0, float('nan'), 1.0]),
        ("Inf input", [1.0, float('inf'), 1.0])
    ]
    
    for name, values in edge_cases:
        print(f"\n--- {name} ---")
        print(f"Input: {values}")
        
        if len(values) == 0:
            print("⏭️ Skipping empty tensor")
            continue
            
        try:
            tensor = torch.tensor(values)
            variance = torch.var(tensor)
            variance_item = variance.item()
            
            print(f"Variance tensor: {variance}")
            print(f"Variance item: {variance_item}")
            
            if np.isnan(variance_item):
                print("🚨 Results in NaN!")
            elif np.isinf(variance_item):
                print("⚠️ Results in Infinity!")
            else:
                print("✅ Normal result")
                
        except Exception as e:
            print(f"💥 Exception: {e}")

def main():
    """Run all NaN bug tests."""
    print("🔬 Simple NaN Bug Reproduction Test")
    print("Testing the specific variance calculation in enhanced dynamics")
    
    # Test 1: Basic variance calculation scenarios
    test_variance_calculation_bug()
    
    # Test 2: Full enhanced dynamics simulation
    success = test_enhanced_dynamics_simulation()
    
    # Test 3: PyTorch edge cases
    test_torch_variance_edge_cases()
    
    print(f"\n🎯 CONCLUSION:")
    if success:
        print(f"   ✅ Enhanced dynamics simulation works correctly")
        print(f"   🤔 The NaN bug may be more subtle or environment-specific")
        print(f"   💡 Recommendation: Add defensive NaN checks anyway")
    else:
        print(f"   🚨 Successfully reproduced the NaN bug!")
        print(f"   🎯 The variance calculation is indeed the culprit")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Add NaN protection to variance calculation")
    print(f"   2. Test the fix with real enhanced dynamics")
    print(f"   3. Re-run the biological embodiment experiment")

if __name__ == "__main__":
    main()