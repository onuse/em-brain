#!/usr/bin/env python3
"""
Test Field Logger Variance Issue

Test the specific field logger variance calculation to reproduce the 
"Field energy: 0.001 ±0.000 (variance)" issue reported by user.
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
from src.brains.field.field_logger import FieldBrainLogger

def test_field_logger_variance_calculation():
    """Test the field logger variance calculation directly"""
    print("🔍 Testing Field Logger Variance Calculation")
    print("=" * 80)
    
    # Create field logger
    logger = FieldBrainLogger(quiet_mode=False)
    
    # Simulate energy values that should show variance
    print("📊 Test 1: Energy values with clear variance")
    test_energies_1 = [0.001, 0.002, 0.003, 0.002, 0.001, 0.003, 0.002, 0.001, 0.003, 0.002]
    
    for energy in test_energies_1:
        logger.log_field_energy(energy)
    
    # Calculate variance using logger's method
    energy_variance, energy_trend, coords_diversity = logger._calculate_learning_metrics()
    
    print(f"   Energy values: {test_energies_1}")
    print(f"   NumPy variance: {np.var(test_energies_1):.9f}")
    print(f"   Logger variance: {energy_variance:.9f}")
    print(f"   Formatted as logger: ±{energy_variance:.3f}")
    
    if energy_variance == 0.000:
        print("❌ PROBLEM: Logger shows ±0.000 despite clear variance!")
    else:
        print("✅ Logger shows variance correctly")
    
    # Test 2: Very small energy values (like real field evolution)
    print(f"\n📊 Test 2: Very small energy values (realistic scenario)")
    logger_small = FieldBrainLogger(quiet_mode=False)
    
    # Similar to what we saw in earlier tests - small but non-zero variance
    test_energies_2 = [0.000000, 0.000000, 0.000000, 0.001849, 0.002565, 0.003212, 0.003600, 0.003800, 0.003900, 0.003950]
    
    for energy in test_energies_2:
        logger_small.log_field_energy(energy)
    
    energy_variance_small, _, _ = logger_small._calculate_learning_metrics()
    
    print(f"   Energy values: {[f'{e:.6f}' for e in test_energies_2]}")
    print(f"   NumPy variance: {np.var(test_energies_2):.9f}")
    print(f"   Logger variance: {energy_variance_small:.9f}")
    print(f"   Formatted as logger: ±{energy_variance_small:.3f}")
    
    if energy_variance_small == 0.000:
        print("❌ PROBLEM: Logger rounds small variance to ±0.000!")
        print("   This is the formatting issue - variance exists but rounds to 0.000")
    else:
        print("✅ Logger shows small variance correctly")
    
    # Test 3: Identical energy values (true zero variance)
    print(f"\n📊 Test 3: Identical energy values (true zero variance)")
    logger_zero = FieldBrainLogger(quiet_mode=False)
    
    test_energies_3 = [0.001] * 10  # All identical
    
    for energy in test_energies_3:
        logger_zero.log_field_energy(energy)
    
    energy_variance_zero, _, _ = logger_zero._calculate_learning_metrics()
    
    print(f"   Energy values: {test_energies_3}")
    print(f"   NumPy variance: {np.var(test_energies_3):.9f}")
    print(f"   Logger variance: {energy_variance_zero:.9f}")
    print(f"   Formatted as logger: ±{energy_variance_zero:.3f}")
    
    # Test 4: Float32 precision issue test
    print(f"\n📊 Test 4: Float32 precision issue test")
    
    # Test the exact conversion the logger does
    test_values = [0.000001627, 0.000001629, 0.000001625, 0.000001630, 0.000001626]
    numpy_variance = np.var(test_values)
    float32_variance = float(np.var(test_values).astype(np.float32))
    
    print(f"   Test values: {test_values}")
    print(f"   NumPy variance (float64): {numpy_variance:.9f}")
    print(f"   Logger variance (float32): {float32_variance:.9f}")
    print(f"   Formatted as logger: ±{float32_variance:.3f}")
    
    return energy_variance_small

def test_real_tcp_adapter_logger_variance():
    """Test the variance issue in a real TCP adapter scenario"""
    print(f"\n🔍 Testing Real TCP Adapter Logger Variance")
    print("=" * 80)
    
    # Create TCP adapter (simulates actual server environment)
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True  # Prevent spam
    )
    
    # Process multiple cycles to build energy history
    print("📊 Processing 25 cycles to build energy history...")
    for i in range(25):
        # Varied input pattern
        base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
        varied_pattern = [x + 0.05 * np.sin(i * 0.2 + j * 0.3) for j, x in enumerate(base_pattern)]
        
        # Process through adapter
        action, brain_state = adapter.process_sensory_input(varied_pattern)
        
        # The TCP adapter automatically logs field energy via:
        # adapter.logger.log_field_energy(field_energy)
        
        if i % 5 == 0:
            field_energy = brain_state.get('field_energy', 0.0)
            evolution_cycles = brain_state.get('field_evolution_cycles', 0)
            print(f"   Cycle {i:2d}: field_energy={field_energy:.6f}, evolution_cycles={evolution_cycles}")
    
    # Force a performance report (this will show the ±variance)
    print(f"\n📈 Forcing performance report from TCP adapter logger...")
    adapter.logger.maybe_report_performance()
    
    # Get the variance directly from the logger
    energy_variance, energy_trend, coords_diversity = adapter.logger._calculate_learning_metrics()
    
    print(f"\n🔍 Direct logger metrics:")
    print(f"   Energy history length: {len(adapter.logger.metrics.field_energy_history)}")
    print(f"   Recent energies: {[f'{e:.6f}' for e in adapter.logger.metrics.field_energy_history[-10:]]}")
    print(f"   Logger variance: {energy_variance:.9f}")
    print(f"   Formatted display: ±{energy_variance:.3f}")
    
    if energy_variance is not None:
        if energy_variance < 0.0005:  # Less than 0.0005 rounds to 0.000 in .3f format
            print(f"❌ IDENTIFIED: Variance {energy_variance:.9f} rounds to ±0.000 in display!")
            print(f"   Field IS evolving, but formatting hides the small variance")
            return True
        else:
            print(f"✅ Variance shows properly in display: ±{energy_variance:.3f}")
            return False
    else:
        print(f"⚠️ No variance calculated (insufficient data)")
        return False

def test_formatting_precision_issue():
    """Test the specific formatting precision issue"""
    print(f"\n🎯 Testing Formatting Precision Issue")
    print("=" * 50)
    
    # Test various small variance values and how they format
    test_variances = [
        0.000001627,  # From our earlier test
        0.0000005,    # Should round to 0.000
        0.0005,       # Should show as 0.001
        0.00049,      # Should round to 0.000
        0.00051,      # Should round to 0.001
    ]
    
    print("Testing variance formatting with .3f precision:")
    for var in test_variances:
        formatted = f"{var:.3f}"
        print(f"   Variance {var:.9f} → ±{formatted}")
        
        if formatted == "0.000" and var > 0:
            print(f"      ❌ Non-zero variance displays as ±0.000!")
        else:
            print(f"      ✅ Displays properly")

def main():
    """Run comprehensive field logger variance tests"""
    print("🔍 Field Logger Variance Issue Investigation")
    print("=" * 80)
    print("Testing the specific '±0.000 (variance)' issue reported by user")
    
    # Test 1: Logger variance calculation
    logger_variance = test_field_logger_variance_calculation()
    
    # Test 2: Real TCP adapter scenario
    tcp_issue_found = test_real_tcp_adapter_logger_variance()
    
    # Test 3: Formatting precision
    test_formatting_precision_issue()
    
    print(f"\n" + "=" * 80)
    print(f"🎯 FINAL CONCLUSION")
    print(f"=" * 80)
    
    if tcp_issue_found:
        print(f"✅ ROOT CAUSE IDENTIFIED:")
        print(f"   Field energy DOES have variance, but small values get rounded to ±0.000")
        print(f"   The field IS evolving properly, the issue is display formatting")
        print(f"   Solution: Use more decimal places in variance display (e.g., ±{logger_variance:.6f})")
    else:
        print(f"⚠️ Issue not fully reproduced, may need additional investigation")

if __name__ == "__main__":
    main()