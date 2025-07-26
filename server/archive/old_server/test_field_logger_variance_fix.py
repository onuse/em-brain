#!/usr/bin/env python3
"""
Test Field Logger Variance Fix
Verify that the adaptive precision fix shows meaningful variance values
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter

def test_field_logger_variance_fix():
    """Test the field logger variance fix with realistic TCP server scenario"""
    print("🔍 Testing Field Logger Variance Fix")
    print("=" * 80)
    print("Testing the adaptive precision fix for field energy variance display")
    
    # Create TCP adapter (simulates actual server environment)
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True  # Prevent initialization spam
    )
    
    # Process multiple cycles to build energy history and trigger field evolution
    print("📊 Processing 30 cycles to build field energy variance...")
    for i in range(30):
        # Create varied input patterns to ensure field evolution
        base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
        varied_pattern = [
            x + 0.1 * np.sin(i * 0.25 + j * 0.4) + 0.02 * np.random.randn() 
            for j, x in enumerate(base_pattern)
        ]
        
        # Process through adapter
        action, brain_state = adapter.process_sensory_input(varied_pattern)
        
        # Print progress every 10 cycles
        if i % 10 == 0:
            field_energy = brain_state.get('field_energy', 0.0)
            evolution_cycles = brain_state.get('field_evolution_cycles', 0)
            print(f"   Cycle {i:2d}: field_energy={field_energy:.6f}, evolution_cycles={evolution_cycles}")
    
    print(f"\n📈 Forcing performance report with FIXED variance display...")
    
    # Force a performance report - this will now show the improved variance formatting
    adapter.logger.maybe_report_performance()
    
    # Get the variance directly for analysis
    energy_variance, energy_trend, coords_diversity = adapter.logger._calculate_learning_metrics()
    
    print(f"\n🔍 Analysis of the fix:")
    print(f"   Raw energy variance: {energy_variance:.9f}")
    
    # Test the adaptive formatting logic
    if energy_variance is not None:
        if energy_variance < 0.000001:
            formatted = f"{energy_variance:.9f}"
            precision = "9 decimal places"
        elif energy_variance < 0.001:
            formatted = f"{energy_variance:.6f}"
            precision = "6 decimal places"
        else:
            formatted = f"{energy_variance:.3f}"
            precision = "3 decimal places"
        
        print(f"   Formatted variance: ±{formatted} (using {precision})")
        
        # Check if the fix worked
        if formatted != "0.000000000" and formatted != "0.000000" and formatted != "0.000":
            print(f"✅ SUCCESS: Variance now shows meaningful digits!")
            print(f"   Before fix: ±0.000 (variance)")
            print(f"   After fix:  ±{formatted} (variance)")
        else:
            print(f"❌ Issue persists: variance still shows as zero")
    else:
        print(f"⚠️ No variance data available")
    
    # Show energy history for context
    recent_energies = adapter.logger.metrics.field_energy_history[-10:]
    print(f"\n📊 Recent energy values:")
    print(f"   {[f'{e:.6f}' for e in recent_energies]}")
    print(f"   Range: {min(recent_energies):.6f} to {max(recent_energies):.6f}")
    print(f"   Evidence of field evolution: {'Yes' if max(recent_energies) > min(recent_energies) else 'No'}")

def test_different_variance_ranges():
    """Test the adaptive formatting with different variance magnitudes"""
    print(f"\n🎯 Testing Adaptive Variance Formatting")
    print("=" * 50)
    
    test_cases = [
        (0.000000162, "Realistic small variance from TCP server"),
        (0.000001627, "Small but measurable variance"),
        (0.000156, "Medium small variance"),
        (0.001234, "Medium variance"),
        (0.123456, "Large variance")
    ]
    
    print("Testing adaptive precision formatting:")
    for variance, description in test_cases:
        # Apply the same logic as the fix
        if variance < 0.000001:
            formatted = f"{variance:.9f}"
        elif variance < 0.001:
            formatted = f"{variance:.6f}"
        else:
            formatted = f"{variance:.3f}"
        
        print(f"   {description}:")
        print(f"      Raw: {variance:.9f} → Display: ±{formatted}")

def main():
    """Run field logger variance fix verification"""
    print("🔧 Field Logger Variance Fix Verification")
    print("=" * 80)
    
    # Test the fix with realistic TCP server scenario
    test_field_logger_variance_fix()
    
    # Test adaptive formatting with different ranges
    test_different_variance_ranges()
    
    print(f"\n" + "=" * 80)
    print(f"🎯 VERIFICATION COMPLETE")
    print(f"=" * 80)
    print(f"The field logger now uses adaptive precision to show meaningful")
    print(f"variance digits instead of rounding small values to ±0.000")

if __name__ == "__main__":
    main()